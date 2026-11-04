"""Grammar bitmask for chain-verify speculative decoding (DSPARK / DFLASH).

``generate_token_bitmask`` walks every request's verify tree in Python: per node a
fill, a bit test, an accept and a rollback, each through two wrapper layers. At
decode batch sizes that host loop runs for tens of milliseconds every step and
leaves the scheduler no slack to prepare the next batch.

For a linear chain the walk is level-synchronous instead: at depth ``k`` every
live xgrammar matcher fills its row in one multi-threaded native call, the next
draft token is checked against all those rows with one numpy gather, and the
survivors accept it in one native call. The rows, the matcher states and the
wrapper bookkeeping come out exactly as the per-request DFS leaves them.

Requests the fast path does not model (non-xgrammar grammars, strict-thinking
token filters, a chain that may cross the think-end marker) fall back to the
reference ``traverse_tree`` on their own rows.

The host bitmask comes from a small ring of pinned buffers instead of a fresh
``torch.full`` per step; only rows written by an earlier use and not rewritten
now are reset to the unconstrained value. A batch whose rows are all
unconstrained (e.g. everyone is still reasoning) skips the copy and the apply
kernel altogether.
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import numpy as np
import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)

# Enough slots that the H2D issued from a slot has always drained by the time
# the ring comes back to it; reuse still waits on the slot's event to be safe.
_NUM_SLOTS = 3


class _MaskSlot:
    def __init__(self, mask: torch.Tensor):
        self.mask = mask
        self.view = mask.numpy()
        # Rows that may hold a constraint (anything other than all-ones).
        self.dirty = np.zeros(mask.shape[0], dtype=bool)
        # Set once the H2D out of ``mask`` has been issued; the host rows must
        # not be rewritten before it fires.
        self.copied = None

    def wait(self) -> None:
        """Waits on the slot if needed."""
        if self.copied is not None:
            self.copied.synchronize()
            self.copied = None


class HostMaskRing:
    """Reusable pinned host bitmasks, one per in-flight verify step."""

    def __init__(self):
        self._slots: List[Optional[_MaskSlot]] = [None] * _NUM_SLOTS
        self._next = 0

    def acquire(
        self, grammar: BaseGrammarObject, vocab_size: int, rows: int
    ) -> _MaskSlot:
        """Allocates or resets the next slot."""
        idx = self._next
        self._next = (idx + 1) % _NUM_SLOTS
        slot = self._slots[idx]
        if slot is not None:
            slot.wait()
        if slot is None or slot.mask.shape[0] < rows:
            # The backend allocator decides shape, dtype and pinning; its fresh
            # buffer is all-ones, i.e. no row is dirty yet.
            mask = grammar.allocate_vocab_mask(
                vocab_size=vocab_size, batch_size=max(rows, 1), device="cpu"
            )
            slot = self._slots[idx] = _MaskSlot(mask)
        return slot

    @staticmethod
    def copy_to_device(slot: _MaskSlot, rows: int, device) -> torch.Tensor:
        """Transfers the slot's host bitmask to the given device."""
        host = slot.mask[:rows]
        # non_blocking is safe: the slot is pinned, and stream order keeps the
        # copy ahead of the sampler's apply_vocab_mask.
        out = host.to(device, non_blocking=True)
        if out.is_cuda:
            slot.copied = torch.cuda.Event()
            slot.copied.record()
        elif out.data_ptr() == host.data_ptr():
            # A host "device" would alias the reused slot.
            out = out.clone()
        return out

    def clear(self) -> None:
        """Cleans up any pending transfers."""
        for slot in self._slots:
            if slot is not None:
                slot.wait()
        self._slots = [None] * _NUM_SLOTS


_ring = HostMaskRing()
_batch_matcher = None


def _get_batch_matcher():
    global _batch_matcher
    if _batch_matcher is None:
        from xgrammar import BatchGrammarMatcher

        _batch_matcher = BatchGrammarMatcher(
            max_threads=envs.SGLANG_SPEC_GRAMMAR_FILL_THREADS.get()
        )
    return _batch_matcher


class ChainMask:
    """The filled host rows of one verify step, plus how to stage them."""

    def __init__(self, slot: _MaskSlot, rows: int, grammar, num_constrained: int):
        self._slot = slot
        self.host = slot.mask[:rows]
        self.grammar = grammar
        self.num_constrained = num_constrained

    def to_device(self, device) -> torch.Tensor:
        """Transfers the host bitmask to the given device."""
        return _ring.copy_to_device(self._slot, self.host.shape[0], device)


def _split_reasoner(grammar):
    """Return (wrapper, inner, state) for the fast path, or None to fall back.

    state is "thinking", "free" (generation with no inner grammar) or "matcher".
    """
    reasoner_type, xgrammar_type = _grammar_types()
    if type(grammar) is xgrammar_type:
        return None, grammar, "matcher"
    if type(grammar) is not reasoner_type or grammar.enable_token_filter:
        return None
    if grammar._is_thinking():
        return grammar, None, "thinking"
    if grammar._is_generation():
        inner = grammar.grammar
        if inner is None:
            return grammar, None, "free"
        if type(inner) is xgrammar_type:
            return grammar, inner, "matcher"
    return None


@functools.cache
def _grammar_types():
    from sglang.srt.constrained.reasoner_grammar_backend import (
        ReasonerGrammarObject,
    )
    from sglang.srt.constrained.xgrammar_backend import XGrammarGrammar

    return ReasonerGrammarObject, XGrammarGrammar


def generate_chain_token_bitmask(
    reqs: List[Req],
    draft_tokens_cpu: torch.Tensor,
    vocab_size: int,
    fallback: Callable,
) -> Tuple[Optional[ChainMask], Optional[BaseGrammarObject]]:
    """Linear-chain equivalent of ``generate_token_bitmask``.

    ``draft_tokens_cpu`` is (bs, chain_len), column 0 the committed token.
    ``fallback(row_tokens, grammar, mask_rows)`` is the reference traversal of one
    request's chain into its rows. Returns (None, None) when no request carries a
    grammar, matching the reference.
    """
    bs, chain = draft_tokens_cpu.shape
    assert len(reqs) == bs
    first = next((r.grammar for r in reqs if r.grammar is not None), None)
    if first is None:
        return None, None

    slot = _ring.acquire(first, vocab_size, bs * chain)
    view = slot.view
    words = view.shape[1]
    rows = bs * chain
    tokens = draft_tokens_cpu.numpy()
    token_rows = draft_tokens_cpu.tolist()
    written = np.zeros(view.shape[0], dtype=bool)

    live_matchers, live_reqs, live_base = [], [], []
    for i, req in enumerate(reqs):
        grammar = req.grammar
        if grammar is None:
            continue
        split = _split_reasoner(grammar)
        row_tokens = token_rows[i]
        if split is not None and split[2] in ("thinking", "free"):
            wrapper = split[0]
            if split[2] == "thinking" and not set(wrapper.think_end_ids).isdisjoint(
                row_tokens[1:]
            ):
                split = None  # the chain may leave the reasoning section
            else:
                # Nothing is filled: every row stays all-ones, so the reference
                # accepts each in-vocab draft token up to the first one past the
                # vocab and rolls them all back. Only current_token survives.
                n = 0
                while n + 1 < chain and row_tokens[n + 1] < vocab_size:
                    n += 1
                if n:
                    wrapper.current_token = row_tokens[n]
                continue
        if split is None:
            base = i * chain
            view[base : base + chain] = -1
            written[base : base + chain] = True
            fallback(row_tokens, grammar, slot.mask[base : base + chain])
            continue
        wrapper, inner, _ = split
        if inner.is_terminated():
            continue
        live_matchers.append(inner.matcher)
        live_reqs.append((wrapper, inner))
        live_base.append(i * chain)

    batch = _get_batch_matcher() if live_matchers else None
    depth = [0] * len(live_matchers)
    active = list(range(len(live_matchers)))
    for k in range(chain):
        if not active:
            break
        fill_rows = [live_base[j] + k for j in active]
        batch.batch_fill_next_token_bitmask(
            [live_matchers[j] for j in active], slot.mask, fill_rows
        )
        written[fill_rows] = True
        if k + 1 == chain:
            break
        rows_np = np.asarray(fill_rows)
        nxt = tokens[rows_np // chain, k + 1]
        in_vocab = nxt < vocab_size
        word = np.where(in_vocab, nxt >> 5, 0)
        if word.size and int(word.max()) >= words:
            in_vocab &= word < words
            word = np.where(in_vocab, word, 0)
        bits = (view[rows_np, word].view(np.uint32) >> (nxt & 31).astype(np.uint32)) & 1
        ok = in_vocab & (bits != 0)
        survivors = [j for j, keep in zip(active, ok.tolist()) if keep]
        if not survivors:
            break
        accept_tokens = [int(x) for x in nxt[ok]]
        accepted = batch.batch_accept_token(
            [live_matchers[j] for j in survivors], accept_tokens
        )
        if not all(accepted):
            bad = accepted.index(False)
            raise ValueError(
                f"Tokens not accepted: {accept_tokens[bad]}\n"
                f"Key string: {live_reqs[survivors[bad]][1].key_string}"
            )
        for j, tok in zip(survivors, accept_tokens):
            depth[j] += 1
            wrapper, inner = live_reqs[j]
            inner.current_token = tok
            if wrapper is not None:
                wrapper.current_token = tok
        active = [j for j in survivors if not live_matchers[j].is_terminated()]

    undo = [(live_matchers[j], d) for j, d in enumerate(depth) if d]
    if undo:
        batch.batch_rollback([m for m, _ in undo], [d for _, d in undo])

    # Rows an earlier step constrained but this step did not rewrite go back to
    # all-ones; rows past this batch are left for whoever covers them next.
    stale = slot.dirty & ~written
    stale[rows:] = False
    if stale.any():
        view[stale] = -1
    slot.dirty[:rows] = written[:rows]
    num_constrained = int(written[:rows].sum())
    return ChainMask(slot, rows, first, num_constrained), first
