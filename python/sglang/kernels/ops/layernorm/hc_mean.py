"""Mean over the hyper-connection axis of a [rows, hc, hidden] bf16 residual stream.

``x.mean(dim=1)`` on the 3-D mHC tensor runs PyTorch's generic reduce kernel (~35 us for
1536 x 4 x 5120 on B200, ~3 per decode step for the DSpark aux hidden states); this is a
single read-once pass. Accumulation is fp32 and the 1/hc scale is exact for hc a power of
two, so the result matches ``mean`` up to the fp32 add order (see the unit test in the
round log).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _hc_mean_kernel(
    x_ptr,
    y_ptr,
    H,
    x_stride_m,
    y_stride_m,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # int64: row * x_stride_m overflows int32 past 104857 rows at hc=4, hidden=5120.
    row = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs < H
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for k in tl.static_range(HC):
        acc += tl.load(x_ptr + row * x_stride_m + k * H + offs, mask=mask, other=0.0).to(
            tl.float32
        )
    tl.store(y_ptr + row * y_stride_m + offs, (acc * (1.0 / HC)).to(y_ptr.dtype.element_ty), mask=mask)


def can_hc_mean(x: torch.Tensor) -> bool:
    """Check we can perform the kernel."""
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and x.dim() == 3
        and x.is_contiguous()
        and x.shape[0] > 0
        and x.shape[1] in (2, 4, 8)
    )


def hc_mean(x: torch.Tensor) -> torch.Tensor:
    """``x.mean(dim=1)`` for contiguous bf16 ``[rows, hc, hidden]`` with hc in {2, 4, 8}."""
    assert can_hc_mean(x)
    rows, hc, h = x.shape
    y = torch.empty((rows, h), dtype=x.dtype, device=x.device)
    block_h = 1024
    _hc_mean_kernel[(rows, triton.cdiv(h, block_h))](
        x, y, h, x.stride(0), y.stride(0), HC=hc, BLOCK_H=block_h, num_warps=4
    )
    return y
