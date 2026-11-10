"""[fusion-opt] DSpark sampling accept straight from the logits.

The stock sampling accept materializes fp32 softmax(logits / T) for every target row
(bs * (gamma + 1)) and every draft row (bs * gamma) -- two full-vocab writes plus reads --
and then runs one program per request that walks the whole vocab twice for the
residual / bonus sample. Only a handful of values are needed: p and q at the gamma
candidate tokens, and one residual row per request.

Here each row gets only its log-sum-exp (one read-only pass, split over vocab tiles);
the chain test computes p = exp(l / T - lse) and q the same way at the candidates; the
final residual (or bonus) sample is spread over (bs, vocab-tile) programs and the CDF
search only scans the tile it lands in. Rows of requests with top_p < 1 (gathered by
SGLANG_OPT_DSPARK_TOPP_ROW_SKIP) still get explicit renormalized probs and are read from
that small buffer.

Same accept rule, same uniforms and the same vocab-order CDF as
speculative_sampling_classic_kernel; p / q agree with the materialized probs to fp32
rounding, so decisions can differ only when a coin lands within rounding of p / q.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

BLOCK_V = 4096
_TILE_STATS_WARPS = 4


@triton.jit
def _argmax_pair(av, ai, bv, bi):
    # same rule as row_argmax._argmax_pair (torch.argmax: NaN first, lowest index on ties)
    an, bn = av != av, bv != bv
    take_a = (an & ~bn) | ((an == bn) & ((av > bv) | (((av == bv) | an) & (ai < bi))))
    return tl.where(take_a, av, bv), tl.where(take_a, ai, bi)


@triton.jit
def _tile_stats_kernel(logits_ptr, temp_ptr, tile_max_ptr, tile_sum_ptr, am_v_ptr, am_i_ptr, vocab,
                       rows_per_request, row_stride, n_tiles, BLOCK: tl.constexpr, WITH_ARGMAX: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    temp = tl.load(temp_ptr + row // rows_per_request).to(tl.float32)
    offs = tile * BLOCK + tl.arange(0, BLOCK)
    vmask = offs < vocab
    raw = tl.load(logits_ptr + row.to(tl.int64) * row_stride + offs, mask=vmask, other=-float("inf")).to(tl.float32)
    if WITH_ARGMAX:
        # the greedy rows' target argmax rides on this read (raw logits, before 1 / T)
        bv, bi = tl.reduce((raw, tl.where(vmask, offs, vocab)), 0, _argmax_pair)
        tl.store(am_v_ptr + row * n_tiles + tile, bv)
        tl.store(am_i_ptr + row * n_tiles + tile, bi)
    x = raw / temp
    m = tl.max(x, axis=0)
    m_safe = tl.where(m == -float("inf"), 0.0, m)
    e = tl.where(vmask, tl.exp(x - m_safe), 0.0)
    tl.store(tile_max_ptr + row * n_tiles + tile, m)
    tl.store(tile_sum_ptr + row * n_tiles + tile, tl.sum(e, axis=0))


@triton.jit
def _lse_combine_kernel(tile_max_ptr, tile_sum_ptr, lse_ptr, am_v_ptr, am_i_ptr, am_out_ptr, n_tiles,
                        BLOCK_T: tl.constexpr, WITH_ARGMAX: tl.constexpr):
    row = tl.program_id(0)
    t = tl.arange(0, BLOCK_T)
    tmask = t < n_tiles
    tmax = tl.load(tile_max_ptr + row * n_tiles + t, mask=tmask, other=-float("inf"))
    tsum = tl.load(tile_sum_ptr + row * n_tiles + t, mask=tmask, other=0.0)
    gmax = tl.max(tmax, axis=0)
    gsum = tl.sum(tl.where(tmask, tsum * tl.exp(tmax - gmax), 0.0), axis=0)
    # all -inf row: gmax = -inf, gsum = NaN -> lse NaN, like the softmax path's NaN row
    tl.store(lse_ptr + row, gmax + tl.log(gsum))
    if WITH_ARGMAX:
        av = tl.load(am_v_ptr + row * n_tiles + t, mask=tmask, other=-float("inf"))
        ai = tl.load(am_i_ptr + row * n_tiles + t, mask=tmask, other=0x7FFFFFFF)
        _, idx = tl.reduce((av, ai), 0, _argmax_pair)
        tl.store(am_out_ptr + row, idx.to(tl.int64))


def row_lse(logits: torch.Tensor, temperatures: torch.Tensor, rows_per_request: int, with_argmax: bool = False):
    """log-sum-exp of logits / T per row (T per request, rows_per_request rows each); with_argmax also
    returns the raw-logits row argmax (int64, torch.argmax semantics) from the same read."""
    num_rows, vocab = logits.shape
    if logits.stride(-1) != 1:
        logits = logits.contiguous()
    bs = num_rows // rows_per_request
    temps = temperatures.reshape(bs).to(torch.float32).contiguous()
    n_tiles = triton.cdiv(vocab, BLOCK_V)
    tile_max = torch.empty((num_rows, n_tiles), dtype=torch.float32, device=logits.device)
    tile_sum = torch.empty_like(tile_max)
    lse = torch.empty((num_rows,), dtype=torch.float32, device=logits.device)
    if with_argmax:
        am_v = torch.empty_like(tile_max)
        am_i = torch.empty((num_rows, n_tiles), dtype=torch.int32, device=logits.device)
        am = torch.empty((num_rows,), dtype=torch.int64, device=logits.device)
    else:
        am_v = am_i = am = lse  # never touched
    _tile_stats_kernel[(num_rows, n_tiles)](logits, temps, tile_max, tile_sum, am_v, am_i, vocab, rows_per_request,
                                            logits.stride(0), n_tiles, BLOCK=BLOCK_V, WITH_ARGMAX=with_argmax,
                                            num_warps=_TILE_STATS_WARPS)
    _lse_combine_kernel[(num_rows,)](tile_max, tile_sum, lse, am_v, am_i, am, n_tiles,
                                     BLOCK_T=triton.next_power_of_2(n_tiles), WITH_ARGMAX=with_argmax)
    return (lse, am) if with_argmax else lse


@triton.jit
def _p_at(TL, tl_stride, TT, LSE_T, TP, tp_stride, slot, b, row, tok, S: tl.constexpr):
    # target prob of `tok` on verify row `row` of request b
    if slot >= 0:
        p = tl.load(TP + (slot * S + row).to(tl.int64) * tp_stride + tok)
    else:
        temp = tl.load(TT + b).to(tl.float32)
        x = tl.load(TL + (b * S + row).to(tl.int64) * tl_stride + tok).to(tl.float32)
        p = tl.exp(x / temp - tl.load(LSE_T + b * S + row))
    return p


@triton.jit
def _fused_chain_accept_kernel(
    Predicts, AcceptIndex, AcceptTokenNum, RowState,
    Candidates, RetriveIndex, UniformSamples,
    TL, tl_stride, TT, LSE_T, TP, tp_stride, Slot,
    DL, dl_stride, DT, LSE_D,
    stride_cand_b, stride_idx_b, stride_uni_b,
    S: tl.constexpr, G: tl.constexpr,
):
    b = tl.program_id(0)
    slot = tl.load(Slot + b)
    dtemp = tl.load(DT + b).to(tl.float32)
    root = tl.load(RetriveIndex + b * stride_idx_b)
    tl.store(AcceptIndex + b * stride_idx_b, root)
    last_idx = root
    cur_row = 0
    num_accept = 0
    step = 1
    cont = 1
    while (step < S) and (cont == 1):
        tok = tl.load(Candidates + b * stride_cand_b + step)
        p = _p_at(TL, tl_stride, TT, LSE_T, TP, tp_stride, slot, b, cur_row, tok, S)
        xd = tl.load(DL + (b * G + cur_row).to(tl.int64) * dl_stride + tok).to(tl.float32)
        q = tl.exp(xd / dtemp - tl.load(LSE_D + b * G + cur_row))
        coin = tl.load(UniformSamples + b * stride_uni_b + (step - 1))
        if coin * q < p:
            num_accept += 1
            cur_row = step
            tl.store(Predicts + last_idx, tok)
            nxt = tl.load(RetriveIndex + b * stride_idx_b + step)
            tl.store(AcceptIndex + b * stride_idx_b + num_accept, nxt)
            last_idx = nxt
            step += 1
        else:
            cont = 0
    tl.store(AcceptTokenNum + b, num_accept)
    tl.store(RowState + b * 3 + 0, cur_row)
    tl.store(RowState + b * 3 + 1, last_idx)
    tl.store(RowState + b * 3 + 2, cont)


@triton.jit
def _residual_vals(TL, tl_stride, TT, LSE_T, TP, tp_stride, slot, DL, dl_stride, DT, LSE_D,
                   b, row, all_acc, v_start, VOCAB, S: tl.constexpr, G: tl.constexpr, BLOCK: tl.constexpr):
    offs = v_start + tl.arange(0, BLOCK)
    mask = offs < VOCAB
    if slot >= 0:
        p = tl.load(TP + (slot * S + row).to(tl.int64) * tp_stride + offs, mask=mask, other=0.0)
    else:
        temp = tl.load(TT + b).to(tl.float32)
        x = tl.load(TL + (b * S + row).to(tl.int64) * tl_stride + offs, mask=mask, other=-float("inf")).to(tl.float32)
        p = tl.where(mask, tl.exp(x / temp - tl.load(LSE_T + b * S + row)), 0.0)
    if all_acc:
        val = p
    else:
        dtemp = tl.load(DT + b).to(tl.float32)
        xd = tl.load(DL + (b * G + row).to(tl.int64) * dl_stride + offs, mask=mask, other=-float("inf")).to(tl.float32)
        q = tl.where(mask, tl.exp(xd / dtemp - tl.load(LSE_D + b * G + row)), 0.0)
        q = tl.where(q == q, q, 0.0)
        d = p - q
        val = tl.where(d > 0.0, d, 0.0)
    return val


@triton.jit
def _fused_residual_sum_kernel(RowState, ChunkSums, TL, tl_stride, TT, LSE_T, TP, tp_stride, Slot,
                               DL, dl_stride, DT, LSE_D, n_chunks, VOCAB,
                               S: tl.constexpr, G: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    c = tl.program_id(1)
    row = tl.load(RowState + b * 3 + 0)
    all_acc = tl.load(RowState + b * 3 + 2)
    slot = tl.load(Slot + b)
    val = _residual_vals(TL, tl_stride, TT, LSE_T, TP, tp_stride, slot, DL, dl_stride, DT, LSE_D,
                         b, row, all_acc, c * BLOCK, VOCAB, S, G, BLOCK)
    tl.store(ChunkSums + b * n_chunks + c, tl.sum(val))


@triton.jit
def _fused_residual_pick_kernel(Predicts, RowState, ChunkSums, UniformFinal, TL, tl_stride, TT, LSE_T, TP,
                                tp_stride, Slot, DL, dl_stride, DT, LSE_D, n_chunks, VOCAB,
                                S: tl.constexpr, G: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    row = tl.load(RowState + b * 3 + 0)
    last_idx = tl.load(RowState + b * 3 + 1)
    all_acc = tl.load(RowState + b * 3 + 2)
    slot = tl.load(Slot + b)
    norm_sum = 0.0
    for c in range(0, n_chunks):
        norm_sum += tl.load(ChunkSums + b * n_chunks + c)
    target_u = tl.load(UniformFinal + b) * norm_sum
    cum = 0.0
    hit = -1
    for c in range(0, n_chunks):
        s = tl.load(ChunkSums + b * n_chunks + c)
        if hit < 0:
            if cum + s > target_u:
                hit = c
        cum += s
    final_token = VOCAB - 1
    if hit >= 0:
        # the in-tile cumsum can round either side of the tile sum: rerun the classic
        # block test from hit - 1 (same prefix of tile sums as the classic pass 2)
        start = tl.maximum(hit - 1, 0)
        cum = 0.0
        for c in range(0, n_chunks):
            if c < start:
                cum += tl.load(ChunkSums + b * n_chunks + c)
        found = 0
        for c in range(0, n_chunks):
            if (c >= start) and (found == 0):
                val = _residual_vals(TL, tl_stride, TT, LSE_T, TP, tp_stride, slot, DL, dl_stride, DT, LSE_D,
                                     b, row, all_acc, c * BLOCK, VOCAB, S, G, BLOCK)
                total = cum + tl.cumsum(val, axis=0)
                m = total > target_u
                if tl.max(m, axis=0):
                    final_token = c * BLOCK + tl.argmax(m.to(tl.int32), axis=0)
                    found = 1
                cum += tl.sum(val)
    tl.store(Predicts + last_idx, final_token)


def fused_chain_sampling_from_logits(
    *,
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_final: torch.Tensor,
    target_logits: torch.Tensor,  # [bs * S, V]
    target_temps: torch.Tensor,  # [bs] / [bs, 1]
    draft_logits: torch.Tensor,  # [bs * G, V]
    draft_temps: torch.Tensor,  # [bs] / [bs, 1]
    topp_probs: Optional[torch.Tensor] = None,  # [n_tp * S, V] renormed target probs
    topp_slot: Optional[torch.Tensor] = None,  # int32 [bs], row slot in topp_probs or -1
    target_lse: Optional[torch.Tensor] = None,  # [bs * S] row_lse(target_logits, target_temps, S) if known
) -> None:
    """ fused_chain_sampling_from_logits """
    bs, S = candidates.shape
    G = S - 1
    vocab = target_logits.shape[-1]
    dev = candidates.device
    if target_logits.stride(-1) != 1:
        target_logits = target_logits.contiguous()
    if draft_logits.stride(-1) != 1:
        draft_logits = draft_logits.contiguous()
    tt = target_temps.reshape(bs).to(torch.float32).contiguous()
    dt = draft_temps.reshape(bs).to(torch.float32).contiguous()
    lse_t = row_lse(target_logits, tt, S) if target_lse is None else target_lse
    lse_d = row_lse(draft_logits, dt, G)
    if topp_probs is None:
        topp_probs = lse_t  # never read: every slot is -1
        topp_slot = torch.full((bs,), -1, dtype=torch.int32, device=dev)
        tp_stride = 0
    else:
        tp_stride = topp_probs.stride(0)
    n_chunks = triton.cdiv(vocab, BLOCK_V)
    row_state = torch.empty((bs, 3), dtype=torch.int32, device=dev)
    chunk_sums = torch.empty((bs, n_chunks), dtype=torch.float32, device=dev)
    common_t = (target_logits, target_logits.stride(0), tt, lse_t, topp_probs, tp_stride, topp_slot)
    common_d = (draft_logits, draft_logits.stride(0), dt, lse_d)
    _fused_chain_accept_kernel[(bs,)](
        predicts, accept_index, accept_token_num, row_state, candidates, retrive_index, uniform_samples,
        *common_t, *common_d, candidates.stride(0), retrive_index.stride(0), uniform_samples.stride(0),
        S=S, G=G,
    )
    _fused_residual_sum_kernel[(bs, n_chunks)](
        row_state, chunk_sums, *common_t, *common_d, n_chunks, vocab, S=S, G=G, BLOCK=BLOCK_V,
    )
    _fused_residual_pick_kernel[(bs,)](
        predicts, row_state, chunk_sums, uniform_samples_final, *common_t, *common_d, n_chunks, vocab,
        S=S, G=G, BLOCK=BLOCK_V,
    )
