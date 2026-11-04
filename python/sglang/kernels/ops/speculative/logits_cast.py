"""Row-wise bf16 -> fp32 logits copy for the tall speculative-verify logits.

``dst.copy_(src)`` / ``src.float()`` on [rows, vocab] bf16 logits runs PyTorch's
generic direct-copy kernel at roughly a third of HBM bandwidth (546 us for
1536 x 129280 on B200); this kernel is a plain vectorized widening copy. The
conversion is exact, so the result is bit-identical.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_BLOCK = 4096


@triton.jit
def _bf16_to_fp32_rows_kernel(
    src,
    dst,
    n_cols,
    src_stride,
    dst_stride,
    BLOCK: tl.constexpr,
):
    # int64 row base: row * stride overflows int32 for bs * vocab >= 2^31.
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_cols
    v = tl.load(src + row * src_stride + cols, mask=mask, other=0.0)
    tl.store(dst + row * dst_stride + cols, v.to(tl.float32), mask=mask)


def can_copy_bf16_to_fp32(src: torch.Tensor, dst: torch.Tensor | None = None) -> bool:
    """Whether ``copy_bf16_to_fp32`` applies (``dst`` None: to a fresh row-major fp32)."""
    ok = (
        src.is_cuda
        and src.dtype == torch.bfloat16
        and src.dim() == 2
        and src.stride(1) == 1
        and src.shape[0] > 0
    )
    if dst is not None:
        ok = (
            ok
            and dst.is_cuda
            and dst.dtype == torch.float32
            and dst.shape == src.shape
            and dst.stride(1) == 1
        )
    return ok


def copy_bf16_to_fp32(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """``dst.copy_(src)`` for 2-D bf16 -> fp32 CUDA tensors with unit column stride."""
    assert can_copy_bf16_to_fp32(src, dst)
    rows, cols = src.shape
    _bf16_to_fp32_rows_kernel[(rows, triton.cdiv(cols, _BLOCK))](
        src, dst, cols, src.stride(0), dst.stride(0), BLOCK=_BLOCK, num_warps=8
    )
    return dst
