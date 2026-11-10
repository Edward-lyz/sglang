"""Fused shared-expert SiLU+clamp + MXFP8 swizzled-quant (graph-safe fusion).

Collapses DSV4.1-Flash MoE shared-expert Reuse-layer op79 `silu_mul_clamp_kernel`
and op80 flashinfer `MXFP8QuantizeSwizzledKernel` into ONE Triton launch:

    gate_up[M,4608] bf16 -> (q_fp8[M,2304] e4m3 row-major,
                             sf uint8 [M*72] in flashinfer 128x4 swizzle)

Byte-identical to `silu_and_mul_clamp(gate_up, 10)` then
`flashinfer.mxfp8_quantize(x, is_sf_swizzled_layout=True, alignment=32, backend="cuda")`,
so the UNCHANGED flashinfer blockscaled GEMM (op81) consumes it. No fp8 matmul is
introduced (static grid, no host sync, no in-kernel alloc) -> CUDA-graph safe.
Quant math + 128x4 swizzle verified byte-exact vs flashinfer on B200 (sm_100).
"""
import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

INV448 = float(np.float32(1.0 / 448.0))


@triton.jit
def _silu(g, u, SILU_MODE: tl.constexpr):
    if SILU_MODE == 0:
        e = libdevice.fast_expf(-g); s = libdevice.fast_dividef(g, 1.0 + e)
    else:
        e = tl.exp(-g); s = g / (1.0 + e)
    return s * u


@triton.jit
def _fused_kernel(
    gate_up_ptr, q_ptr, sf_ptr, inv448, limit,
    M, N: tl.constexpr, N2: tl.constexpr, G: tl.constexpr, Kt: tl.constexpr,
    GROUP: tl.constexpr, GPAD: tl.constexpr, BLOCK_M: tl.constexpr, SILU_MODE: tl.constexpr,
):
    pid = tl.program_id(0)
    r0 = pid * BLOCK_M
    gs = tl.arange(0, GPAD)
    ks = tl.arange(0, GROUP)
    col = gs[:, None] * GROUP + ks[None, :]
    gmask = gs < G
    kt = gs // 4
    ki = gs % 4
    for i in tl.static_range(BLOCK_M):
        row = r0 + i
        if row < M:
            base = row * N
            m2 = gmask[:, None]
            gate = tl.load(gate_up_ptr + base + col, mask=m2, other=0.0).to(tl.float32)
            up = tl.load(gate_up_ptr + base + N2 + col, mask=m2, other=0.0).to(tl.float32)
            g = tl.minimum(gate, limit)
            u = tl.minimum(tl.maximum(up, -limit), limit)
            val = _silu(g, u, SILU_MODE)
            y = val.to(tl.bfloat16).to(tl.float32)
            amax = tl.maximum(tl.max(tl.abs(y), axis=1), 1e-10)
            raw = amax * inv448
            bits = raw.to(tl.int32, bitcast=True)
            E = (bits >> 23) & 0xFF
            mant = bits & 0x7FFFFF
            e8 = tl.minimum(E + (mant != 0).to(tl.int32), 255)
            yscale = ((254 - e8) << 23).to(tl.float32, bitcast=True)
            q = tl.minimum(tl.maximum(y * yscale[:, None], -448.0), 448.0).to(tl.float8e4nv)
            tl.store(q_ptr + row * N2 + col, q, mask=m2)
            mt = row // 128
            mi = row % 128
            outer = mi // 32
            inner = mi % 32
            off = (((mt * Kt + kt) * 32 + inner) * 16 + outer * 4 + ki)
            tl.store(sf_ptr + off, e8.to(tl.uint8), mask=gmask)


def fused_silu_and_mul_mxfp8_swizzled_quant(gate_up, swiglu_limit=10.0, group=32,
                                            silu_mode=0, block_m=1, num_warps=2):
    """gate_up[M,2*H] bf16 -> (q_fp8[M,H] e4m3, sf uint8[M*H/group] 128x4 swizzle).

    Drop-in for silu_and_mul_clamp(gate_up, swiglu_limit) + flashinfer.mxfp8_quantize
    (is_sf_swizzled_layout=True, alignment=group). Requires H%group==0, (H/group)%4==0,
    and M a multiple of 128 (decode shapes satisfy this)."""
    assert gate_up.dtype == torch.bfloat16 and gate_up.is_contiguous()
    M, N = gate_up.shape
    N2 = N // 2
    assert N2 % group == 0 and (N2 // group) % 4 == 0
    G = N2 // group
    GPAD = triton.next_power_of_2(G)
    q = torch.empty((M, N2), device=gate_up.device, dtype=torch.float8_e4m3fn)
    sf = torch.empty((M * G,), device=gate_up.device, dtype=torch.uint8)
    grid = (triton.cdiv(M, block_m),)
    _fused_kernel[grid](
        gate_up, q, sf, INV448, float(swiglu_limit),
        M, N, N2, G, G // 4, group, GPAD, block_m, silu_mode,
        num_warps=num_warps,
    )
    return q, sf
