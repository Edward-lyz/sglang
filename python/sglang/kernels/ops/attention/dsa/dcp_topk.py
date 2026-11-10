"""Global DSA top-k over a DCP-striped index-K (SGLANG_K31_INDEXER_DCP).

Every DCP rank scores only the tokens it owns and keeps its local top-k
candidates (an element of the global top-k is always in its owner's local
top-k). The ranks then agree on the global k-th score by a radix select over
the candidates' order-preserving uint32 keys: each round histograms the next
8 key bits of the candidates still tied with the running prefix and
all-reduces the 256-bin histogram across the group. After the last round the
candidates above the prefix bucket are taken, and the bucket itself is split
by rank order (all-reduced per-rank bucket counts) so exactly
min(topk, row length) slots are selected group-wide. Each rank returns its
selected local KV rows packed to the front (-1 tail), encoded as widened
slots (row * dcp + rank) like the replicated-index path, and their count.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_NUM_BINS = 256


@triton.jit
def _order_key(v):
    # fp32 -> uint32 whose unsigned order matches the float order.
    bits = v.to(tl.uint32, bitcast=True)
    flip = tl.where((bits >> 31) != 0, 0xFFFFFFFF, 0x80000000).to(tl.uint32)
    return (bits ^ flip).to(tl.int64)


@triton.jit
def _load_keys(logits_ptr, logits_stride, raw_ptr, row, K: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    raw = tl.load(raw_ptr + row * K + offs, mask=offs < K, other=-1)
    valid = raw >= 0
    v = tl.load(logits_ptr + row * logits_stride + raw, mask=valid, other=0.0)
    return offs, valid, _order_key(v)


@triton.jit
def _pick_bin(hist_ptr, row, need):
    # Bucket b holding the need-th largest key: above(b) < need <= above(b) + h(b).
    bins = tl.arange(0, 256)
    h = tl.load(hist_ptr + row * 256 + bins).to(tl.int32)
    above = tl.sum(h, 0) - tl.cumsum(h, 0)
    hit = (above < need) & (above + h >= need)
    b = tl.max(tl.where(hit, bins, 0), 0)
    above_b = tl.sum(tl.where(hit, above, 0), 0)
    return b.to(tl.int64), above_b


@triton.jit
def _radix_round_kernel(
    logits_ptr,
    logits_stride,
    raw_ptr,
    len_ptr,
    state_ptr,
    hist_prev_ptr,
    hist_out_ptr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    ROUND: tl.constexpr,
    SHIFT: tl.constexpr,
):
    row = tl.program_id(0)
    if ROUND == 0:
        need = tl.minimum(tl.load(len_ptr + row), K).to(tl.int32)
        prefix = tl.zeros((), dtype=tl.int64)
    else:
        prefix = tl.load(state_ptr + row * 2)
        need = tl.load(state_ptr + row * 2 + 1).to(tl.int32)
        b, above = _pick_bin(hist_prev_ptr, row, need)
        prefix = prefix * 256 + b
        need = need - above
    tl.store(state_ptr + row * 2, prefix)
    tl.store(state_ptr + row * 2 + 1, need.to(tl.int64))

    _, valid, key = _load_keys(logits_ptr, logits_stride, raw_ptr, row, K, BLOCK)
    if ROUND == 0:
        match = valid
    else:
        match = valid & ((key >> (SHIFT + 8)) == prefix)
    bins = (key >> SHIFT) & 255
    tl.atomic_add(hist_out_ptr + row * 256 + bins, 1.0, mask=match)


@triton.jit
def _bucket_count_kernel(
    logits_ptr,
    logits_stride,
    raw_ptr,
    state_ptr,
    hist_last_ptr,
    counts_ptr,
    num_rows,
    RANK: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    SHIFT: tl.constexpr,
):
    # Final pick, then this rank's count of candidates in the prefix bucket.
    row = tl.program_id(0)
    prefix = tl.load(state_ptr + row * 2)
    need = tl.load(state_ptr + row * 2 + 1).to(tl.int32)
    b, above = _pick_bin(hist_last_ptr, row, need)
    prefix = prefix * 256 + b
    need = need - above
    tl.store(state_ptr + row * 2, prefix)
    tl.store(state_ptr + row * 2 + 1, need.to(tl.int64))

    _, valid, key = _load_keys(logits_ptr, logits_stride, raw_ptr, row, K, BLOCK)
    in_bucket = valid & ((key >> SHIFT) == prefix)
    tl.store(
        counts_ptr + RANK * num_rows + row,
        tl.sum(in_bucket.to(tl.float32), 0),
    )


@triton.jit
def _select_compact_kernel(
    logits_ptr,
    logits_stride,
    raw_ptr,
    local_rows_ptr,
    state_ptr,
    counts_ptr,
    out_ptr,
    out_counts_ptr,
    num_rows,
    RANK: tl.constexpr,
    DCP_SIZE: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    SHIFT: tl.constexpr,
):
    row = tl.program_id(0)
    prefix = tl.load(state_ptr + row * 2)
    need = tl.load(state_ptr + row * 2 + 1).to(tl.int32)
    # Ties in the prefix bucket are given to the lower DCP ranks first.
    before = tl.zeros((), dtype=tl.int32)
    for r in tl.static_range(RANK):
        before += tl.load(counts_ptr + r * num_rows + row).to(tl.int32)
    mine = tl.load(counts_ptr + RANK * num_rows + row).to(tl.int32)
    quota = tl.minimum(tl.maximum(need - before, 0), mine)

    offs, valid, key = _load_keys(logits_ptr, logits_stride, raw_ptr, row, K, BLOCK)
    high = key >> SHIFT
    in_bucket = valid & (high == prefix)
    bucket_rank = tl.cumsum(in_bucket.to(tl.int32), 0) - in_bucket.to(tl.int32)
    selected = valid & ((high > prefix) | (in_bucket & (bucket_rank < quota)))
    sel = selected.to(tl.int32)
    pos = tl.cumsum(sel, 0) - sel
    count = tl.sum(sel, 0)
    rows = tl.load(local_rows_ptr + row * K + offs, mask=offs < K, other=-1)
    # Widened slot (row * dcp + rank) of each selected local row: the same
    # encoding as the replicated-index top-k, so the attention side is shared.
    tl.store(out_ptr + row * K + pos, rows * DCP_SIZE + RANK, mask=selected)
    tl.store(out_ptr + row * K + offs, -1, mask=(offs >= count) & (offs < K))
    tl.store(out_counts_ptr + row, count)


def dcp_global_topk(
    logits: torch.Tensor,
    raw_indices: torch.Tensor,
    local_rows: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    group,
    dcp_rank: int,
    dcp_size: int,
    rounds: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the group-wide top-k from each rank's local top-k candidates.

    logits: [rows, W] fp32 local scores (unit column stride);
    raw_indices / local_rows: [rows, topk] int32 local candidate positions
    (-1 padded) and their local KV rows, as produced by the local top-k;
    lengths: [rows] global (unsharded) row lengths. ``group`` is the DCP
    GroupCoordinator; every rank must call this with the same shapes.
    Returns (selected widened slots -- local row * dcp_size + dcp_rank --
    packed to the front with a -1 tail, per-row selected count), both int32.
    """
    num_rows = raw_indices.shape[0]
    device = raw_indices.device
    assert 1 <= rounds <= 4, rounds
    assert raw_indices.shape == local_rows.shape == (num_rows, topk)
    assert logits.dtype == torch.float32 and logits.stride(1) == 1
    out = torch.empty((num_rows, topk), dtype=torch.int32, device=device)
    out_counts = torch.empty((num_rows,), dtype=torch.int32, device=device)
    if num_rows == 0:
        return out, out_counts

    raw_indices = raw_indices.contiguous()
    local_rows = local_rows.contiguous()
    lengths = lengths.to(torch.int32).contiguous()
    block = triton.next_power_of_2(topk)
    state = torch.empty((num_rows, 2), dtype=torch.int64, device=device)
    hists = torch.zeros((rounds, num_rows, _NUM_BINS), dtype=torch.float32, device=device)
    bucket_counts = torch.zeros((dcp_size, num_rows), dtype=torch.float32, device=device)
    grid = (num_rows,)

    hist_prev = hists[0]
    for r in range(rounds):
        _radix_round_kernel[grid](
            logits,
            logits.stride(0),
            raw_indices,
            lengths,
            state,
            hist_prev,
            hists[r],
            K=topk,
            BLOCK=block,
            ROUND=r,
            SHIFT=24 - 8 * r,
            num_warps=8,
        )
        hist_prev = group.all_reduce(hists[r])

    final_shift = 32 - 8 * rounds
    _bucket_count_kernel[grid](
        logits,
        logits.stride(0),
        raw_indices,
        state,
        hist_prev,
        bucket_counts,
        num_rows,
        RANK=dcp_rank,
        K=topk,
        BLOCK=block,
        SHIFT=final_shift,
        num_warps=8,
    )
    bucket_counts = group.all_reduce(bucket_counts)
    _select_compact_kernel[grid](
        logits,
        logits.stride(0),
        raw_indices,
        local_rows,
        state,
        bucket_counts,
        out,
        out_counts,
        num_rows,
        RANK=dcp_rank,
        DCP_SIZE=dcp_size,
        K=topk,
        BLOCK=block,
        SHIFT=final_shift,
        num_warps=8,
    )
    return out, out_counts
