"""The batched chain-verify grammar bitmask matches the reference DFS exactly.

Uses real xgrammar matchers over a tiny vocabulary, wrapped the way the server
wraps them (ReasonerGrammarObject around XGrammarGrammar), and checks over many
random steps that the filled rows, the matcher states and the wrapper
bookkeeping equal what generate_token_bitmask produces. Consecutive steps reuse
the ring's host buffers, so stale-row reset is exercised too.
"""

import random
import unittest

import torch

from sglang.srt.constrained.reasoner_grammar_backend import ReasonerGrammarObject
from sglang.srt.constrained.xgrammar_backend import (
    MAX_ROLLBACK_TOKENS,
    XGrammarGrammar,
    XGrammarGrammarBackend,
)
from sglang.srt.speculative import grammar_chain_mask
from sglang.srt.speculative.spec_utils import (
    GrammarTree,
    generate_token_bitmask,
    traverse_tree,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

xgr = __import__("xgrammar")

# 70 tokens -> 3 bitmask words, the last one partial.
VOCAB = ["a", "b", "c", "d", "x", "<eos>", "</think>"] + [f"w{i}" for i in range(63)]
EOS, THINK_END = 5, 6
VOCAB_SIZE = len(VOCAB)
GRAMMAR = 'root ::= ("ab" | "c" | "d" | ("w" [0-9]+))* "x"'


def _compiled():
    info = xgr.TokenizerInfo(VOCAB, stop_token_ids=[EOS])
    return xgr.GrammarCompiler(info, cache_enabled=False).compile_grammar(GRAMMAR)


class _Req:
    def __init__(self, grammar):
        self.grammar = grammar


def _make_grammar(ctx, kind):
    inner = XGrammarGrammar(
        xgr.GrammarMatcher(ctx, max_rollback_tokens=MAX_ROLLBACK_TOKENS),
        VOCAB_SIZE,
        ctx,
        None,
        key_string="test",
    )
    if kind == "plain":
        return inner
    wrapper = ReasonerGrammarObject(
        grammar=inner,
        think_end_ids=[THINK_END],
        allocate_vocab_mask_fn=XGrammarGrammarBackend.allocate_vocab_mask,
        move_vocab_mask_fn=XGrammarGrammarBackend.move_vocab_mask,
        apply_vocab_mask_fn=XGrammarGrammarBackend.apply_vocab_mask,
    )
    wrapper.maybe_init_reasoning(kind == "thinking")
    return wrapper


def _legal_tokens(grammar):
    mask = XGrammarGrammarBackend.allocate_vocab_mask(VOCAB_SIZE, 1, "cpu")
    grammar.fill_vocab_mask(mask, 0)
    bits = mask[0].tolist()
    return [t for t in range(VOCAB_SIZE) if (bits[t >> 5] >> (t & 31)) & 1]


def _state(grammar):
    """Everything observable about a request's grammar after a traversal."""
    inner = grammar.grammar if isinstance(grammar, ReasonerGrammarObject) else grammar
    out = {"current": grammar.current_token}
    if isinstance(grammar, ReasonerGrammarObject):
        out.update(
            think=grammar.tokens_in_think,
            after=grammar.tokens_after_end,
            matched=grammar._matched_think_end_tokens,
            hist=list(grammar._thinking_match_history),
        )
    if inner is not None:
        out.update(
            inner_current=inner.current_token,
            accepted=list(inner.accepted_tokens),
            terminated=inner.is_terminated(),
        )
        if not inner.is_terminated():
            out["next_mask"] = _legal_tokens(inner)
    return out


def _draft_chain(grammar, chain, rng):
    """A chain that mostly follows the grammar, with occasional illegal or
    out-of-vocab tokens and think-end markers."""
    ids = [rng.randrange(VOCAB_SIZE)]
    probe = grammar
    for _ in range(chain - 1):
        r = rng.random()
        if r < 0.08:
            ids.append(VOCAB_SIZE + rng.randrange(8))
        elif r < 0.16:
            ids.append(THINK_END)
        elif r < 0.25:
            ids.append(rng.randrange(VOCAB_SIZE))
        else:
            legal = (
                _legal_tokens(probe.grammar)
                if isinstance(probe, ReasonerGrammarObject)
                and probe._is_generation()
                and not probe.grammar.is_terminated()
                else list(range(VOCAB_SIZE))
            )
            ids.append(rng.choice(legal) if legal else EOS)
    return ids


class TestChainMaskMatchesReference(unittest.TestCase):
    def setUp(self):
        grammar_chain_mask._ring.clear()

    def _run(self, seed, steps=12, max_bs=24, chain=6):
        rng = random.Random(seed)
        ctx = _compiled()
        kinds = ["plain", "thinking", "generation", None]
        bs = max_bs
        req_kinds = [rng.choice(kinds) for _ in range(bs)]
        ref = [_Req(_make_grammar(ctx, k) if k else None) for k in req_kinds]
        new = [_Req(_make_grammar(ctx, k) if k else None) for k in req_kinds]
        next_token = list(range(1, chain)) + [-1]
        next_sibling = [-1] * chain

        def fallback(row_tokens, grammar, rows):
            traverse_tree(
                next_token,
                next_sibling,
                row_tokens,
                grammar,
                rows,
                vocab_size=VOCAB_SIZE,
            )

        for step in range(steps):
            # Varying batch: drop a random prefix/suffix so rows move around.
            lo = rng.randint(0, 3)
            hi = rng.randint(bs - 3, bs)
            ref_b, new_b = ref[lo:hi], new[lo:hi]
            drafts = [
                (
                    _draft_chain(r.grammar, chain, rng)
                    if r.grammar
                    else [rng.randrange(VOCAB_SIZE) for _ in range(chain)]
                )
                for r in ref_b
            ]
            verify = torch.tensor(drafts, dtype=torch.int64)
            ref_mask, _ = generate_token_bitmask(
                ref_b, *GrammarTree.from_linear_chain(verify).resolve(), VOCAB_SIZE
            )
            chain_mask, _ = grammar_chain_mask.generate_chain_token_bitmask(
                new_b, verify, VOCAB_SIZE, fallback
            )
            if ref_mask is None:
                self.assertIsNone(chain_mask)
                continue
            self.assertTrue(
                torch.equal(ref_mask, chain_mask.host), f"seed={seed} step={step}"
            )
            if chain_mask.num_constrained == 0:
                self.assertTrue(bool((ref_mask == -1).all()))
            for a, b in zip(ref_b, new_b):
                if a.grammar is not None:
                    self.assertEqual(_state(a.grammar), _state(b.grammar))
            # Commit the same legal tokens on both sides before the next step.
            for a, b in zip(ref, new):
                if a.grammar is None:
                    continue
                for _ in range(rng.randint(0, 2)):
                    g = a.grammar
                    if g.is_terminated():
                        break
                    if isinstance(g, ReasonerGrammarObject) and g._is_thinking():
                        tok = (
                            THINK_END
                            if rng.random() < 0.3
                            else rng.randrange(VOCAB_SIZE)
                        )
                    else:
                        legal = _legal_tokens(g)
                        if not legal:
                            break
                        tok = rng.choice(legal)
                    a.grammar.accept_token(tok)
                    b.grammar.accept_token(tok)

    def test_random_steps(self):
        for seed in range(30):
            self._run(seed)

    def test_all_reasoning_batch_is_unconstrained(self):
        ctx = _compiled()
        reqs = [_Req(_make_grammar(ctx, "thinking")) for _ in range(5)]
        verify = torch.tensor([[1, 2, 3, 4, 1, 2]] * 5, dtype=torch.int64)
        chain_mask, _ = grammar_chain_mask.generate_chain_token_bitmask(
            reqs, verify, VOCAB_SIZE, lambda *a: self.fail("no fallback expected")
        )
        self.assertEqual(chain_mask.num_constrained, 0)
        self.assertTrue(bool((chain_mask.host == -1).all()))
        self.assertTrue(all(r.grammar.current_token == 2 for r in reqs))


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestRingDeviceCopy(unittest.TestCase):
    def test_slot_reuse_waits_for_its_copy(self):
        """A slot's host rows are not rewritten before its H2D has read them."""
        ring = grammar_chain_mask.HostMaskRing()
        rng = torch.Generator().manual_seed(0)
        grammar = _make_grammar(_compiled(), "plain")
        seen = []
        for step in range(12):
            rows = [6, 12, 30, 12][step % 4]
            # Keep the stream busy so each copy is still pending when the ring
            # comes back to its slot.
            torch.cuda._sleep(2_000_000)
            slot = ring.acquire(grammar, VOCAB_SIZE, rows)
            data = torch.randint(
                -(2**31), 2**31, slot.mask[:rows].shape, generator=rng
            ).to(torch.int32)
            slot.mask[:rows] = data
            seen.append((data, ring.copy_to_device(slot, rows, "cuda")))
        torch.cuda.synchronize()
        for data, dev in seen:
            self.assertTrue(torch.equal(data, dev.cpu()))
        ring.clear()


if __name__ == "__main__":
    unittest.main()
