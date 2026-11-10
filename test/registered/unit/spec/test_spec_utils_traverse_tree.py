"""Regression test for spec_utils.traverse_tree calling xgrammar with tensors.

xgrammar 0.2.0 tightened its FFI binding and rejects 0-d tensors where Python
ints are expected. The dfs in traverse_tree recurses with `retrieve_next_token[curr]`
and reads `draft_tokens[curr]`, both of which return 0-d tensors and must be
explicitly cast before being handed to the grammar matcher.
"""

import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.speculative.spec_utils import (
    GrammarTree,
    generate_token_bitmask,
    traverse_tree,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestTraverseTreePassesIntsToGrammar(unittest.TestCase):
    def _record_grammar(self):
        """A grammar mock that records every call argument and rejects torch tensors."""
        grammar = MagicMock()
        grammar.is_terminated.return_value = False
        accept_calls = []
        fill_calls = []

        def record_accept(token):
            if isinstance(token, torch.Tensor):
                raise TypeError(f"accept_token got torch.Tensor: {token!r}")
            accept_calls.append(token)

        def record_fill(bitmask, idx):
            if isinstance(idx, torch.Tensor):
                raise TypeError(f"fill_vocab_mask got torch.Tensor idx: {idx!r}")
            fill_calls.append(idx)

        grammar.accept_token.side_effect = record_accept
        grammar.fill_vocab_mask.side_effect = record_fill
        grammar.rollback.return_value = None
        return grammar, accept_calls, fill_calls

    def _chain(self, verify_ids_2d):
        """Row 0 of the links chain-verify algorithms actually feed traverse_tree."""
        links = GrammarTree.from_linear_chain(verify_ids_2d).resolve()
        return tuple(t[0] for t in links)

    def test_branching_tree_passes_ints(self):
        # Binary tree exercises both child recursion and sibling recursion:
        #   0 ─┬─ 1
        #      └─ 2 ─── 3
        retrieve_next_token = torch.tensor([1, -1, 3, -1], dtype=torch.int32)
        retrieve_next_sibling = torch.tensor([-1, 2, -1, -1], dtype=torch.int32)
        draft_tokens = torch.tensor([100, 11, 22, 33], dtype=torch.int64)
        # all bits set: every draft token passes the parent's bitmask check
        bitmask = torch.full((4, 4), -1, dtype=torch.int32)

        grammar, accept_calls, fill_calls = self._record_grammar()
        traverse_tree(
            retrieve_next_token,
            retrieve_next_sibling,
            draft_tokens,
            grammar,
            bitmask,
        )

        self.assertEqual(set(accept_calls), {11, 22, 33})
        self.assertEqual(set(fill_calls), {0, 1, 2, 3})
        for token in accept_calls:
            self.assertIsInstance(token, int)
        for idx in fill_calls:
            self.assertIsInstance(idx, int)

    def test_linear_chain_visits_all_positions_in_order(self):
        # Chain-verify algorithms (DFLASH/DSPARK) have no branching, so their tree
        # degenerates to 0 -- 1 -- 2 -- 3 with column 0 the already-committed token.
        rnt, rns, draft_tokens = self._chain(torch.tensor([[100, 11, 22, 33]]))
        self.assertEqual(rnt.tolist(), [1, 2, 3, -1])
        self.assertEqual(rns.tolist(), [-1, -1, -1, -1])
        bitmask = torch.full((4, 4), -1, dtype=torch.int32)  # all allowed

        grammar, accept_calls, fill_calls = self._record_grammar()
        traverse_tree(rnt, rns, draft_tokens, grammar, bitmask)

        # Root (col 0) is never accepted; every draft token is, in chain order.
        self.assertEqual(accept_calls, [11, 22, 33])
        self.assertEqual(fill_calls, [0, 1, 2, 3])
        for token in accept_calls:
            self.assertIsInstance(token, int)
        for idx in fill_calls:
            self.assertIsInstance(idx, int)

    def test_linear_chain_stops_at_grammar_reject(self):
        # A draft token the grammar disallows must stop the descent: no accept/fill
        # for that node or anything after it, so the mask rows past it stay unfilled
        # and only the already-filled prefix can be committed.
        rnt, rns, draft_tokens = self._chain(torch.tensor([[100, 5, 7, 9]]))
        bitmask = torch.full((4, 4), -1, dtype=torch.int32)  # all allowed
        # Disallow token id 7 (draft_tokens[2]) in node 1's mask (its parent).
        bitmask[1, 7 // 32] &= ~(1 << (7 % 32))

        grammar, accept_calls, fill_calls = self._record_grammar()
        traverse_tree(rnt, rns, draft_tokens, grammar, bitmask)

        # Node 1 accepted+filled; node 2 rejected -> node 2 and node 3 skipped.
        self.assertEqual(accept_calls, [5])
        self.assertEqual(fill_calls, [0, 1])


def _reference_traverse_tree(
    retrieve_next_token, retrieve_next_sibling, draft_tokens, grammar, bitmask, vocab_size
):
    """The original torch-indexing traverse_tree, kept as the semantic reference."""

    def dfs(curr, parent_pos):
        if curr == 0:
            is_accepted = True
        else:
            parent_bitmask = bitmask[parent_pos]
            current_token = draft_tokens[curr]
            if vocab_size and current_token >= vocab_size:
                is_accepted = False
            else:
                is_accepted = (
                    parent_bitmask[current_token // 32] & (1 << (current_token % 32))
                ) != 0
        if is_accepted:
            if curr != 0:
                grammar.accept_token(int(draft_tokens[curr]))
            if not grammar.is_terminated():
                grammar.fill_vocab_mask(bitmask, curr)
                if retrieve_next_token[curr] != -1:
                    dfs(int(retrieve_next_token[curr]), curr)
            if curr != 0:
                grammar.rollback(1)
        if retrieve_next_sibling[curr] != -1:
            dfs(int(retrieve_next_sibling[curr]), parent_pos)

    dfs(0, -1)


class _StatefulGrammar:
    """Deterministic grammar: the mask row and termination depend on the whole
    accepted-token history, so any divergence in visit order, accept/rollback
    pairing or reject decisions shows up in the filled bitmask or the call log."""

    def __init__(self, num_words, seed):
        self.state = []
        self.num_words = num_words
        self.seed = seed
        self.log = []

    def accept_token(self, token):
        assert type(token) is int
        self.state.append(token)
        self.log.append(("accept", token))

    def rollback(self, k):
        del self.state[-k:]
        self.log.append(("rollback", k))

    def is_terminated(self):
        return bool(self.state) and (sum(self.state) + self.seed) % 11 == 0

    def fill_vocab_mask(self, bitmask, idx):
        assert type(idx) is int
        g = torch.Generator().manual_seed(hash((self.seed, *self.state)) & 0xFFFFFFFF)
        # int32 words with every bit (sign bit included) populated at random
        row = torch.randint(-(2**31), 2**31, (self.num_words,), generator=g)
        bitmask[idx] = row.to(torch.int32)
        self.log.append(("fill", idx))


def _random_tree(n, gen):
    """Random tree over nodes 0..n-1 in first-child / next-sibling form."""
    rnt = [-1] * n
    rns = [-1] * n
    last_child = {}
    for node in range(1, n):
        parent = int(torch.randint(0, node, (1,), generator=gen))
        if parent in last_child:
            rns[last_child[parent]] = node
        else:
            rnt[parent] = node
        last_child[parent] = node
    return rnt, rns


class TestTraverseTreeMatchesReference(unittest.TestCase):
    VOCAB = 200  # 7 int32 words; the last word is partially past vocab_size

    def _check(self, rnt, rns, draft, seed):
        n = len(draft)
        words = (self.VOCAB + 31) // 32
        results = []
        for impl in ("ref", "new"):
            grammar = _StatefulGrammar(words, seed)
            bitmask = torch.full((n, words), -1, dtype=torch.int32)
            t_rnt = torch.tensor(rnt, dtype=torch.int64)
            t_rns = torch.tensor(rns, dtype=torch.int64)
            t_draft = torch.tensor(draft, dtype=torch.int64)
            if impl == "ref":
                _reference_traverse_tree(
                    t_rnt, t_rns, t_draft, grammar, bitmask, self.VOCAB
                )
            else:
                traverse_tree(
                    t_rnt, t_rns, t_draft, grammar, bitmask, vocab_size=self.VOCAB
                )
            self.assertEqual(grammar.state, [])
            results.append((grammar.log, bitmask))
        self.assertEqual(results[0][0], results[1][0])
        self.assertTrue(torch.equal(results[0][1], results[1][1]))

    def test_random_trees_and_chains(self):
        gen = torch.Generator().manual_seed(0)
        for case in range(300):
            n = int(torch.randint(1, 12, (1,), generator=gen))
            if case % 2:
                rnt, rns = _random_tree(n, gen)
            else:
                rnt = list(range(1, n)) + [-1]
                rns = [-1] * n
            # include ids >= vocab_size and ids whose bit is the int32 sign bit
            draft = torch.randint(0, self.VOCAB + 8, (n,), generator=gen).tolist()
            for i in range(1, n, 3):
                draft[i] = 32 * int(torch.randint(0, 6, (1,), generator=gen)) + 31
            self._check(rnt, rns, draft, seed=case)

    def test_generate_token_bitmask_matches_reference(self):
        gen = torch.Generator().manual_seed(1)
        bs, chain = 9, 6
        words = (self.VOCAB + 31) // 32
        verify = torch.randint(0, self.VOCAB + 4, (bs, chain), generator=gen)
        rnt, rns, draft = GrammarTree.from_linear_chain(verify).resolve()

        class _Req:
            def __init__(self, grammar):
                self.grammar = grammar

        def make_reqs():
            reqs = []
            for i in range(bs):
                if i % 4 == 3:
                    reqs.append(_Req(None))  # non-grammar rows stay all-ones
                    continue
                g = _StatefulGrammar(words, seed=100 + i)
                g.allocate_vocab_mask = lambda vocab_size, batch_size, device: (
                    torch.full((batch_size, words), -1, dtype=torch.int32)
                )
                reqs.append(_Req(g))
            return reqs

        ref_reqs = make_reqs()
        ref_mask = torch.full((bs * chain, words), -1, dtype=torch.int32)
        for i, req in enumerate(ref_reqs):
            if req.grammar is not None:
                _reference_traverse_tree(
                    rnt[i],
                    rns[i],
                    draft[i],
                    req.grammar,
                    ref_mask[i * chain : (i + 1) * chain],
                    self.VOCAB,
                )

        new_reqs = make_reqs()
        new_mask, _ = generate_token_bitmask(new_reqs, rnt, rns, draft, self.VOCAB)
        self.assertTrue(torch.equal(ref_mask, new_mask))
        for a, b in zip(ref_reqs, new_reqs):
            if a.grammar is not None:
                self.assertEqual(a.grammar.log, b.grammar.log)


if __name__ == "__main__":
    unittest.main()
