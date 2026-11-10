"""Fused host-side scheduling scalars for the decode steps (DSpark).

Each of these used to be a handful of 1-3 us eager torch kernels, launched right before a
CUDA-graph replay, where nothing else can overlap them:

  * stage_sampling_params_fused  -- clamp(temperatures) + (top_k <= 1) greedy mask + rng counter bump
  * local_num_token_non_padded_into -- clamp(global - rank_offset, 0, chunk) into the graph buffer
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _stage_sampling_params_kernel(
    temp_in_ptr,
    topk_in_ptr,
    temp_out_ptr,
    greedy_out_ptr,
    counter_ptr,
    bs,
    min_temp,
    BUMP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < bs
    t = tl.load(temp_in_ptr + offs, mask=m, other=1.0).to(tl.float32)
    # torch.clamp(min=) semantics: a NaN passes through unchanged
    t = tl.where(t < min_temp, min_temp, t)
    tl.store(temp_out_ptr + offs, t, mask=m)
    k = tl.load(topk_in_ptr + offs, mask=m, other=1)
    tl.store(greedy_out_ptr + offs, (k <= 1).to(tl.uint8), mask=m)
    if BUMP:
        if pid == 0:
            tl.store(counter_ptr, tl.load(counter_ptr) + 1)


def can_stage_sampling_params(
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    temp_out: torch.Tensor,
    greedy_out: torch.Tensor,
    counter: torch.Tensor,
    bs: int,
) -> bool:
    """Fusion up to 128 consecutive temperature top_k tests into one kernel launch."""
    return (
        bs > 0
        and all(t.is_cuda for t in (temperatures, top_ks, temp_out, greedy_out, counter))
        and temperatures.dim() == 1
        and top_ks.dim() == 1
        and temperatures.is_contiguous()
        and top_ks.is_contiguous()
        and temperatures.numel() >= bs
        and top_ks.numel() >= bs
        and temp_out.dtype == torch.float32
        and temp_out.numel() >= bs
        and greedy_out.dtype == torch.bool
        and greedy_out.numel() >= bs
        and counter.dtype == torch.int64
        and counter.numel() == 1
        and temperatures.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and top_ks.dtype in (torch.int32, torch.int64)
    )


def stage_sampling_params_fused(
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    temp_out: torch.Tensor,
    greedy_out: torch.Tensor,
    counter: torch.Tensor,
    bs: int,
    min_temp: float = 1e-5,
) -> None:
    """temp_out[:bs] = clamp(temperatures[:bs], min=min_temp); greedy_out[:bs] = top_ks[:bs] <= 1;
    counter += 1 -- one launch."""
    assert can_stage_sampling_params(temperatures, top_ks, temp_out, greedy_out, counter, bs)
    block = 1024
    _stage_sampling_params_kernel[(triton.cdiv(bs, block),)](
        temperatures,
        top_ks,
        temp_out,
        greedy_out.view(torch.uint8),
        counter,
        bs,
        min_temp,
        BUMP=True,
        BLOCK=block,
        num_warps=4,
    )


@triton.jit
def _local_num_token_non_padded_kernel(global_ptr, out_ptr, rank_offset, chunk):
    g = tl.load(global_ptr).to(tl.int64)
    v = tl.minimum(tl.maximum(g - rank_offset, 0), chunk)
    tl.store(out_ptr, v.to(out_ptr.dtype.element_ty))


def local_num_token_non_padded_into(
    out: torch.Tensor,
    global_num_token_non_padded: torch.Tensor,
    rank_offset: int,
    tokens_per_rank: int,
) -> None:
    """out[0] = clamp(global[0] - rank_offset, 0, tokens_per_rank) -- replaces sub + clamp + copy_."""
    assert out.is_cuda and global_num_token_non_padded.is_cuda
    assert out.numel() == 1 and global_num_token_non_padded.numel() == 1
    _local_num_token_non_padded_kernel[(1,)](
        global_num_token_non_padded, out, int(rank_offset), int(tokens_per_rank)
    )
