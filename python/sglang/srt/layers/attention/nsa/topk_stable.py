"""NSA TopK helpers: env-based impl selection and stable index transforms."""

from __future__ import annotations

from typing import Optional

import torch

from sglang.kernels.jit.topk_stable import TOPK_STABLE_MAX_SEQ_LEN
from sglang.srt.environ import envs


def get_nsa_topk_impl() -> str:
    """Return the active TopK implementation name ('cuda' or 'cuda_stable')."""
    value = (envs.SGLANG_NSA_TOPK_IMPL.get() or "cuda").strip().lower()
    if value not in {"cuda", "cuda_stable"}:
        raise ValueError(
            f"Invalid SGLANG_NSA_TOPK_IMPL={value!r}, expected one of {{'cuda', 'cuda_stable'}}"
        )
    return value


def expand_req_ids(cu_seqlens_q: torch.Tensor, b_rows: int) -> torch.Tensor:
    """Map each query row to its request index via cumulative sequence lengths."""
    row_ids = torch.arange(b_rows, device=cu_seqlens_q.device, dtype=cu_seqlens_q.dtype)
    return torch.searchsorted(cu_seqlens_q[1:], row_ids, right=True)


def transform_index_page_table_decode_stable(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
) -> torch.Tensor:
    """Transform TopK token indices to page table indices for paged KV cache."""
    req_ids = expand_req_ids(cu_seqlens_q.to(torch.int32), topk_indices.size(0)).to(
        torch.int64
    )
    src_rows = page_table.index_select(0, req_ids)
    max_col = src_rows.size(1) - 1
    valid_mask = (topk_indices >= 0) & (topk_indices <= max_col)
    safe_indices = topk_indices.clamp(min=0, max=max_col).to(torch.int64)
    result = torch.gather(src_rows, dim=1, index=safe_indices)
    return result.masked_fill(~valid_mask, -1).to(torch.int32)


def transform_index_ragged_stable(
    topk_indices: torch.Tensor,
    topk_indices_offset: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply ragged offset to TopK indices, preserving -1 padding."""
    if topk_indices_offset is None:
        return topk_indices.to(torch.int32)

    if (
        envs.SGLANG_OPT_USE_FUSED_RAGGED_INDEX_TRANSFORM.get()
        and topk_indices.is_cuda
        and topk_indices.dtype == torch.int32
    ):
        from sglang.kernels.jit.ragged_index_transform import (
            transform_index_ragged_stable as fused_transform,
        )

        return fused_transform(topk_indices, topk_indices_offset)

    mask = topk_indices != -1
    return torch.where(mask, topk_indices + topk_indices_offset.unsqueeze(1), topk_indices).to(
        torch.int32
    )
