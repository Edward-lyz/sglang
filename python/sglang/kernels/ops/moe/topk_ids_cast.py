"""One-launch ``topk_idx.to(int32)`` + padded-row mask for the MegaMoE gate path.

``_try_mega_gate`` returns the gate kernel's int64 ids; the unfused path casts
them to int32 and then runs ``mask_topk_ids`` (rows >= num_token_non_padded
become -1). Both ops sit on the critical gate chain in front of the MoE, so
they are merged here. Integer-only, hence bit-identical to the two-step path.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _cast_mask_topk_ids_kernel(
    src,
    dst,
    num_non_padded,
    n_elems,
    top_k,
    HAS_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_range = offs < n_elems
    v = tl.load(src + offs, mask=in_range, other=0).to(tl.int32)
    if HAS_MASK:
        n = tl.load(num_non_padded).to(tl.int32)
        v = tl.where(offs // top_k >= n, -1, v)
    tl.store(dst + offs, v, mask=in_range)


def can_cast_mask_topk_ids(
    topk_idx: torch.Tensor, num_token_non_padded: Optional[torch.Tensor]
) -> bool:
    """Check we can cast the topk indices."""
    return (
        topk_idx.is_cuda
        and topk_idx.dtype in (torch.int32, torch.int64)
        and topk_idx.dim() == 2
        and topk_idx.is_contiguous()
        and 0 < topk_idx.numel() < 2**31
        and (
            num_token_non_padded is None
            or (num_token_non_padded.is_cuda and num_token_non_padded.numel() == 1)
        )
    )


def cast_mask_topk_ids(
    topk_idx: torch.Tensor, num_token_non_padded: Optional[torch.Tensor]
) -> torch.Tensor:
    """``topk_idx.to(int32)`` with rows >= ``num_token_non_padded`` set to -1."""
    assert can_cast_mask_topk_ids(topk_idx, num_token_non_padded)
    out = torch.empty(topk_idx.shape, dtype=torch.int32, device=topk_idx.device)
    n = topk_idx.numel()
    block = 1024
    _cast_mask_topk_ids_kernel[(triton.cdiv(n, block),)](
        topk_idx,
        out,
        num_token_non_padded if num_token_non_padded is not None else topk_idx,
        n,
        topk_idx.shape[1],
        HAS_MASK=num_token_non_padded is not None,
        BLOCK=block,
        num_warps=4,
    )
    return out
