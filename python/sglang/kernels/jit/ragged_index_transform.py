"""Fused ragged index transform (GLM 5.2 / DeepSeek-V3.2 NSA stable-TopK path).

Registers the ``ragged_index_transform`` CUDA kernel as an in-tree JIT module
and exposes :func:`transform_index_ragged_stable`, a drop-in for the eager
``nsa/topk_stable.py::transform_index_ragged_stable``: it collapses the 3
``elementwise_kernel`` launches (`!= -1`, `+ offset` broadcast, `torch.where`)
plus cast into one fused CUDA launch (one read + one write pass).

    out[i, j] = -1 if topk_indices[i, j] == -1
                else int32(topk_indices[i, j] + topk_indices_offset[i])
"""
from __future__ import annotations

from typing import Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

# Tuned config (see project/op/glm52_ragged_index_transform history.md): int4
# (16B) vectorized ld/st, BLK=128 + ILP=4 (4 int4 loads in flight -> high MLP on
# this L1TEX/latency-bound op), __ldg+__stcs cache policy, one block per row.
# C == 2048 (TOPK) is divisible by 4 so the vectorized path is taken; VEC=1 is
# the fallback for non-aligned C.
_DEFAULT_VEC = 4
_DEFAULT_BLK = 128
_DEFAULT_RPB = 1
_DEFAULT_POL = 2
_DEFAULT_PERSIST = False
_DEFAULT_ILP = 4


@cache_once
def _jit_module(
    offset_i64: bool,
    vec_size: int,
    block_threads: int,
    rows_per_block: int,
    cache_policy: int,
    persist: bool,
    ilp: int,
):
    args = make_cpp_args(
        offset_i64, vec_size, block_threads, rows_per_block, cache_policy, persist, ilp
    )
    return load_jit(
        "ragged_index_transform",
        *args,
        cuda_files=["elementwise/ragged_index_transform.cuh"],
        cuda_wrappers=[
            ("run", f"RaggedIndexTransformKernel<{args}>::run"),
        ],
    )


def transform_index_ragged_stable(
    topk_indices: torch.Tensor,
    topk_indices_offset: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    vec_size: int = _DEFAULT_VEC,
    block_threads: int = _DEFAULT_BLK,
    rows_per_block: int = _DEFAULT_RPB,
    cache_policy: int = _DEFAULT_POL,
    persist: bool = _DEFAULT_PERSIST,
    ilp: int = _DEFAULT_ILP,
) -> torch.Tensor:
    """Drop-in for topk_stable.transform_index_ragged_stable (one fused launch).

    Args:
        topk_indices: ``[M, C]`` int32, contiguous, -1 padded.
        topk_indices_offset: ``[M]`` int32 or int64.
        out: optional preallocated ``[M, C]`` int32 output.

    Returns:
        ``[M, C]`` int32.
    """
    assert topk_indices.dtype == torch.int32, (
        f"topk_indices must be int32, got {topk_indices.dtype}"
    )
    assert topk_indices.dim() == 2 and topk_indices.is_contiguous(), (
        "topk_indices must be a contiguous 2D tensor"
    )
    assert topk_indices_offset.dim() == 1, "topk_indices_offset must be 1D [M]"
    assert topk_indices_offset.size(0) == topk_indices.size(0), (
        "topk_indices_offset length must equal topk_indices rows"
    )
    assert topk_indices_offset.dtype in (torch.int32, torch.int64), (
        f"offset must be int32/int64, got {topk_indices_offset.dtype}"
    )
    assert topk_indices_offset.is_contiguous(), "offset must be contiguous"

    if out is None:
        out = torch.empty_like(topk_indices)
    else:
        assert out.shape == topk_indices.shape and out.dtype == torch.int32
        assert out.is_contiguous()

    m, c = topk_indices.shape
    if m == 0:
        return out

    vec = vec_size if (c % vec_size == 0) else 1
    offset_i64 = topk_indices_offset.dtype == torch.int64
    module = _jit_module(
        offset_i64, vec, block_threads, rows_per_block, cache_policy, persist, ilp
    )
    module.run(topk_indices, out, topk_indices_offset)
    return out
