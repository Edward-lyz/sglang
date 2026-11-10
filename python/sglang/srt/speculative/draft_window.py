"""Draft-side sliding-window (SWA) helpers shared by DFLASH and DSPARK.

Both algorithms give the draft worker a *private compact* ``req_to_token`` table
over the target's global KV index space: each row holds the most recent
``window_size`` committed slots followed by this step's verify-block slots. The
draft's ``seq_lens`` is clamped to that compact length, so draft attention
structurally cannot see keys outside the window -- no kernel window parameter is
involved, and block-internal bidirectionality (``AttentionType.ENCODER_ONLY``)
is untouched.

The table is a *view* of target-owned slots: it never allocates and never frees.
Every step rewrites ``[0, draft_prefix_len + block_size)`` before the backend
reads that same range, so stale rows from a finished request are never observed.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.kernels.ops.speculative.cache_locs import (
    rebuild_compact_draft_req_to_token_func,
)


def compact_draft_seq_lens(
    *,
    seq_lens: torch.Tensor,
    window_size: int,
    page_size: int,
    device: str | torch.device,
) -> torch.Tensor:
    """Per-request length of the compact draft view (device-side, exact)."""
    visible_lens = torch.clamp(
        seq_lens.to(dtype=torch.int32, device=device), max=int(window_size)
    )
    if page_size <= 1:
        return visible_lens

    # Paged FA backends derive the page table from local token positions, so the
    # compact suffix must start on a page boundary. Keep up to page_size - 1 extra
    # tokens on the left to preserve valid local page structure.
    seq_lens_i64 = seq_lens.to(torch.int64)
    visible_lens_i64 = visible_lens.to(torch.int64)
    visible_start = seq_lens_i64 - visible_lens_i64
    aligned_start = visible_start - torch.remainder(visible_start, page_size)
    return (seq_lens_i64 - aligned_start).to(torch.int32)


def compact_draft_seq_lens_host(
    *,
    host_seq_lens: torch.Tensor,
    window_size: int,
    page_size: int,
    out: torch.Tensor,
) -> None:
    """Sync-free host upper bound for compact_draft_seq_lens.

    Deliberately NOT the exact page-align arithmetic: that mapping is a
    non-monotonic sawtooth in [window, window+page), so evaluating it on an
    over-estimated host len (the reserved overlap bound) could UNDER-shoot
    the true device value. min(len, window+page) is its monotonic envelope
    (always >= the exact compact len); consumers only need an upper bound.
    """
    bound = int(window_size) + (page_size if page_size > 1 else 0)
    lens = host_seq_lens.to(dtype=torch.int64, device="cpu")
    out.copy_(torch.clamp(lens, max=bound).to(torch.int32))


def fill_compact_seq_lens_cpu_bound(
    *,
    batch_seq_lens_cpu: Optional[torch.Tensor],
    reserved_seq_lens_cpu: Optional[torch.Tensor],
    draft_prefix_lens: torch.Tensor,
    window_size: int,
    page_size: int,
    out: torch.Tensor,
) -> None:
    """Fill the seq_lens_cpu planning bound, sync-free when a host-side length
    source is available; backends consume it as a safe upper bound (same
    contract as the non-compact path in forward_batch_generation)."""
    host_lens = (
        batch_seq_lens_cpu if batch_seq_lens_cpu is not None else reserved_seq_lens_cpu
    )
    if host_lens is None:
        # Last resort: the legacy blocking D2H copy.
        out.copy_(draft_prefix_lens)
        return
    compact_draft_seq_lens_host(
        host_seq_lens=host_lens,
        window_size=window_size,
        page_size=page_size,
        out=out,
    )


def rebuild_compact_draft_req_to_token(
    *,
    draft_req_to_token: torch.Tensor,
    target_req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    draft_prefix_lens: torch.Tensor,
    verify_cache_loc_2d: torch.Tensor,
    bs: int,
    block_size: int,
    slot_translator=None,
    max_span: Optional[int] = None,
) -> None:
    """Write the draft-local compact req->token rows: the committed suffix
    window at [0, draft_prefix_len) plus the verify block slots after it.

    With ``slot_translator`` the row holds DRAFT-SWA ids instead of target-global
    ones, matching where the inject path writes the KV. Read and write side must
    agree: a row of global ids over a window-sized draft pool would index out of
    bounds, and a row of swa ids without a translated write would read stale KV.

    CUDA/HIP triton path only -- the caller owns the non-triton fallback.
    """
    suffix_start = prefix_lens.to(torch.int64) - draft_prefix_lens.to(torch.int64)
    rebuild_compact_draft_req_to_token_func(
        draft_req_to_token=draft_req_to_token,
        target_req_to_token=target_req_to_token,
        req_pool_indices=req_pool_indices,
        suffix_start=suffix_start,
        draft_prefix_lens=draft_prefix_lens,
        verify_out_cache_loc_2d=verify_cache_loc_2d,
        batch_size=bs,
        block_size=block_size,
    )
    if slot_translator is None:
        return

    # Translate the rows in place. Only [0, draft_prefix_len + block_size) was
    # written, and only that range is read (seq_lens is clamped to it), so a
    # bounded per-row slice covers exactly the live span. Rows beyond it keep
    # whatever they held; nothing reads them.
    #
    # ``draft_req_to_token`` is int32 (ReqToTokenPool default) but the slot map
    # holds int64 ids -- cast back into the row's dtype so an in-place assign
    # doesn't raise "dtypes must match". Slot ids stay well within int32 for
    # any realistic pool.
    # ``max_span``: host-known upper bound on draft_prefix_len + block_size.
    # When omitted (tests / non-hot callers) the exact device-side max is used,
    # which costs one D2H sync -- the decode hot path always passes the bound.
    if max_span is None:
        max_span = int(draft_prefix_lens.max()) + int(block_size)
    rows = req_pool_indices.to(torch.int64)
    live = draft_req_to_token[rows, :max_span]
    draft_req_to_token[rows, :max_span] = slot_translator(live).to(
        dtype=draft_req_to_token.dtype
    )
