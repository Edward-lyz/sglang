"""Fused MLA o_proj wo_b-input MXFP8 swizzled quantization (DSV4.1 decode).

Collapses the two launches the decode wo_b input path runs after the bf16 wo_a
GEMM (`_apply_wo_a_bf16_matmul` -> `torch.einsum("tgd,grd->tgr")`, which returns a
NON-contiguous [T,G,R] `o`, stride (R, T*R, 1)):

  - the `o.flatten(1)` strided->contiguous copy   (profile op65 `out_elementwise`)
  - flashinfer cute-dsl `mxfp8_quantize(...)`      (profile op66 `MXFP8QuantizeSwizzledKernel`)

into ONE Triton launch that reads the strided wo_a output directly and emits the
FP8 activation (row-major [T,G*R]) + UE8M0 scale in flashinfer's 128x4 swizzle, wrapped
as `Mxfp8SwizzledInput` so the UNCHANGED flashinfer blockscaled wo_b GEMM (op67) consumes
it. No contiguous intermediate, no FP8 matmul, static grid, no host sync -> CUDA-graph
safe by construction. Byte-identical to the cute-dsl two-kernel path on realistic decode
activations (verified on B200 sm_100); flag-gated, default OFF, unsupported shapes fall
back to the original path.
"""
import numpy as np
import torch
import triton
import triton.language as tl

from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput

INV448 = float(np.float32(1.0 / 448.0))
_GROUP = 32


@triton.jit
def sglfuse_oproj_mxfp8(
    o_ptr, q_ptr, sf_ptr, inv448, s_t, s_g,
    T, K: tl.constexpr, R: tl.constexpr, NG: tl.constexpr, Kt: tl.constexpr,
    SUBG: tl.constexpr, GROUP: tl.constexpr, GPAD: tl.constexpr, BLOCK_M: tl.constexpr,
):
    """Fused MLA o_proj wo_b-input MXFP8 swizzled quantization."""
    pid = tl.program_id(0)
    r0 = pid * BLOCK_M
    gid = tl.arange(0, GPAD)
    ks = tl.arange(0, GROUP)
    gmask = gid < NG
    gg = gid // SUBG
    sub = gid % SUBG
    in_col = gg[:, None] * s_g + sub[:, None] * GROUP + ks[None, :]
    out_col = gid[:, None] * GROUP + ks[None, :]
    kt = gid // 4
    ki = gid % 4
    for i in tl.static_range(BLOCK_M):
        row = r0 + i
        if row < T:
            m2 = gmask[:, None]
            y = tl.load(o_ptr + row * s_t + in_col, mask=m2, other=0.0).to(tl.float32)
            amax = tl.max(tl.abs(y), axis=1)
            normalized = amax * inv448
            exp = (normalized.to(tl.int32, bitcast=True) >> 23) & 0xFF
            sf = tl.where(normalized <= 0, 0, tl.minimum(exp + 1, 254))  # cute-dsl UE8M0
            inv = tl.where(sf == 0, 0, (254 - sf) << 23).to(tl.float32, bitcast=True)
            q = tl.minimum(tl.maximum(y * inv[:, None], -448.0), 448.0).to(tl.float8e4nv)
            tl.store(q_ptr + row * K + out_col, q, mask=m2)
            mt = row // 128
            mi = row % 128
            off = (((mt * Kt + kt) * 32 + (mi % 32)) * 16 + (mi // 32) * 4 + ki)
            tl.store(sf_ptr + off, sf.to(tl.uint8), mask=gmask)


def is_supported(o: torch.Tensor, group: int = _GROUP) -> bool:
    """True when `o` is the strided bf16 [T,G,R] wo_a output this kernel can fuse."""
    if not (isinstance(o, torch.Tensor) and o.dim() == 3 and o.dtype == torch.bfloat16):
        return False
    if not o.is_cuda or o.stride(2) != 1:
        return False
    T, G, R = o.shape
    K = G * R
    return (R % group == 0) and ((K // group) % 4 == 0) and (T % 128 == 0)


def fused_o_proj_mxfp8_swizzled_input(o: torch.Tensor, group: int = _GROUP,
                                      block_m: int = 1, num_warps: int = 2):
    """o[T,G,R] bf16 -> Mxfp8SwizzledInput(q_fp8[T,G*R], ue8m0 128x4-swizzled scale),
    fusing the flatten copy + flashinfer mxfp8 swizzled quantize. Returns None when
    the shape is unsupported so the caller can fall back to the original path."""
    if not is_supported(o, group):
        return None
    T, G, R = o.shape
    K = G * R
    NG = K // group
    GPAD = triton.next_power_of_2(NG)
    q = torch.empty((T, K), device=o.device, dtype=torch.float8_e4m3fn)
    sf = torch.empty((T * NG,), device=o.device, dtype=torch.uint8)
    sglfuse_oproj_mxfp8[(triton.cdiv(T, block_m),)](
        o, q, sf, INV448, o.stride(0), o.stride(1),
        T, K, R, NG, NG // 4, R // group, group, GPAD, block_m,
        num_warps=num_warps,
    )
    return Mxfp8SwizzledInput(q, sf)
