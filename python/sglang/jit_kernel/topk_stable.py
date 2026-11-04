"""Python wrapper for the stable TopK JIT kernel."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


TOPK = 2048
TOPK_STABLE_MAX_SEQ_LEN = 1048576


@cache_once
def _jit_topk_stable_module() -> Module:
    """Load and cache the stable TopK JIT module."""
    return load_jit(
        "topk_stable",
        cuda_files=["elementwise/topk_stable.cuh"],
        cuda_wrappers=[("fast_topk_stable", "TopKStableKernel::run")],
        extra_cuda_cflags=["--use_fast_math"],
    )


def fast_topk_stable(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    row_starts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run deterministic TopK selection with stable tie-breaking.

    Args:
        score: [B, SEQ_LEN] float32 logits.
        lengths: [B] int32 valid lengths per row.
        topk: Must be 2048.
        row_starts: [B] int32 optional row offsets.

    Returns:
        [B, 2048] int32 sorted indices.
    """
    assert topk == TOPK, "fast_topk_stable only supports topk=2048"
    assert score.dim() == 2

    indices = score.new_empty((score.size(0), topk), dtype=torch.int32)
    module = _jit_topk_stable_module()
    module.fast_topk_stable(score, indices, lengths, row_starts)
    return indices
