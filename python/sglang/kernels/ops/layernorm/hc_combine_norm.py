"""Collapse the hyper-connection streams and normalize the sublayer input."""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _hc_combine_norm(
    X,
    P,
    W,
    Y,
    SX: tl.constexpr,
    SP: tl.constexpr,
    EPS: tl.constexpr,
    PARTS: tl.constexpr,
):
    row, part = tl.program_id(0), tl.program_id(1)
    h = tl.arange(0, 8192)
    value = tl.full((8192,), 0, tl.float32)
    for c in tl.static_range(4):
        pre = tl.load(P + row * SP + c).to(tl.float32)
        x = tl.load(X + row * SX + c * 5120 + h, h < 5120, 0).to(tl.float32)
        value += x * pre
    # The unfused combine stores BF16 before RMSNorm reads it.
    value = value.to(tl.bfloat16).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(value * value, 0) / 5120 + EPS)
    mask = (h >= part * (5120 // PARTS)) & (h < (part + 1) * (5120 // PARTS))
    weight = tl.load(W + h, mask, 0).to(tl.float32)
    tl.store(Y + row * 5120 + h, value * inv_rms * weight, mask)


@triton.jit
def _hc_combine_norm_prefill(
    X, P, W, Y, SX: tl.constexpr, SP: tl.constexpr, EPS: tl.constexpr
):
    # Large batches have enough rows to use one CTA per row without repeating
    # the combine and RMS reduction for each output partition.
    row = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, 8192)
    value = tl.full((8192,), 0, tl.float32)
    for c in tl.static_range(4):
        pre = tl.load(P + row * SP + c).to(tl.float32)
        x = tl.load(X + row * SX + c * 5120 + h, h < 5120, 0).to(tl.float32)
        value += x * pre
    value = value.to(tl.bfloat16).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(value * value, 0) / 5120 + EPS)
    weight = tl.load(W + h, h < 5120, 0).to(tl.float32)
    tl.store(Y + row * 5120 + h, value * inv_rms * weight, h < 5120)


def hc_combine_norm(
    x: torch.Tensor, pre: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """Fuse four-stream combine and RMSNorm for BF16 batches of width 5120.

    Every row count takes the same per-row reduction in the same order, so the
    result is batch-invariant across the grid-split variants below."""
    m = x.shape[0]
    assert x.shape == (m, 20480)
    assert pre.shape == (m, 4) and pre.stride(1) == 1
    assert weight.shape == (5120,) and weight.is_contiguous()
    assert x.dtype == weight.dtype == torch.bfloat16 and x.stride(1) == 1
    y = torch.empty((m, 5120), dtype=x.dtype, device=x.device)
    if m >= 4096:
        _hc_combine_norm_prefill[(m,)](
            x, pre, weight, y, x.stride(0), pre.stride(0), eps, num_warps=4
        )
        return y
    # Four CTAs per row trade redundant statistics for more concurrent loads
    # when only a few speculative tokens are being processed; wider batches have
    # enough rows to split the 5120 columns fewer ways.
    parts = 4 if m <= 8 else (2 if m <= 48 else 1)
    _hc_combine_norm[(m, parts)](
        x, pre, weight, y, x.stride(0), pre.stride(0), eps, parts, num_warps=8
    )
    return y



@triton.jit
def _hc_combine_norm_mxfp8(
    X,
    P,
    W,
    Y,
    Q,
    SF,
    m_pad,
    SX: tl.constexpr,
    SP: tl.constexpr,
    EPS: tl.constexpr,
    PARTS: tl.constexpr,
    QEPS: tl.constexpr,
    COPY0: tl.constexpr = False,
):
    # _hc_combine_norm plus the per-32 ue8m0 cast of its bf16 output in DeepGEMM's
    # layout (sglang_per_token_group_quant_fp8, column-major TMA-aligned scales).
    # COPY0: the combine is copy 0 as is (pre = one-hot [1, 0, 0, 0]); only that
    # copy is read and P is ignored.
    row, part = tl.program_id(0), tl.program_id(1)
    h = tl.arange(0, 8192)
    value = tl.full((8192,), 0, tl.float32)
    if COPY0:
        value = tl.load(X + row * SX + h, h < 5120, 0).to(tl.float32)
    else:
        for c in tl.static_range(4):
            pre = tl.load(P + row * SP + c).to(tl.float32)
            x = tl.load(X + row * SX + c * 5120 + h, h < 5120, 0).to(tl.float32)
            value += x * pre
    value = value.to(tl.bfloat16).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(value * value, 0) / 5120 + EPS)
    mask = (h >= part * (5120 // PARTS)) & (h < (part + 1) * (5120 // PARTS))
    weight = tl.load(W + h, mask, 0).to(tl.float32)
    yb = (value * inv_rms * weight).to(tl.bfloat16)
    tl.store(Y + row * 5120 + h, yb, mask)
    y = tl.reshape(yb.to(tl.float32), (256, 32))
    amax = tl.maximum(tl.max(tl.abs(y), axis=1), QEPS)
    rb = (amax * (1.0 / 448.0)).to(tl.int32, bitcast=True)
    e8 = ((rb >> 23) & 0xFF) + ((rb & 0x7FFFFF) != 0).to(tl.int32)
    e8 = tl.minimum(tl.maximum(e8, 1), 254)
    inv_scale = ((254 - e8) << 23).to(tl.float32, bitcast=True)
    q = tl.minimum(tl.maximum(y * inv_scale[:, None], -448.0), 448.0)
    q = tl.reshape(q, (8192,)).to(tl.float8e4nv)
    tl.store(Q + row * 5120 + h, q, mask)
    g = tl.arange(0, 256)
    gmask = (g >= part * (160 // PARTS)) & (g < (part + 1) * (160 // PARTS))
    tl.store(SF + ((g // 4) * m_pad + row) * 4 + (g % 4), e8.to(tl.uint8), gmask)


def hc_combine_norm_mxfp8(
    x: torch.Tensor,
    pre: Optional[torch.Tensor],
    weight: torch.Tensor,
    eps: float,
):
    """hc_combine_norm that also returns its output cast to MXFP8 for a DeepGEMM
    linear: ``(y_bf16 [m, 5120], (y_fp8 [m, 5120], scales int32 [m, 40] view))``.
    Decode-sized batches only (the split-row grid); None for m >= 4096."""
    m = x.shape[0]
    assert x.shape == (m, 20480)
    # pre None: copy 0 only (the one-hot combine), see COPY0 in the kernel.
    assert pre is None or (pre.shape == (m, 4) and pre.stride(1) == 1)
    assert weight.shape == (5120,) and weight.is_contiguous()
    assert x.dtype == weight.dtype == torch.bfloat16 and x.stride(1) == 1
    if m >= 4096:
        return None
    m_pad = (m + 3) // 4 * 4
    y = torch.empty((m, 5120), dtype=x.dtype, device=x.device)
    q = torch.empty((m, 5120), dtype=torch.float8_e4m3fn, device=x.device)
    sf = torch.empty((5120 // 128, m_pad), dtype=torch.int32, device=x.device)
    if m > 0:
        parts = 4 if m <= 8 else (2 if m <= 48 else 1)
        _hc_combine_norm_mxfp8[(m, parts)](
            x, x if pre is None else pre, weight, y, q, sf.view(torch.uint8), m_pad,
            x.stride(0), 0 if pre is None else pre.stride(0), eps, parts, 1e-10,
            COPY0=pre is None, num_warps=8,
        )
    return y, (q, sf.mT[:m])
