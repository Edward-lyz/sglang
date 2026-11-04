"""[fusion-opt] silu(gate) * up (+ swiglu clamp) -> per-32 ue8m0 FP8 quant, written
straight into DeepGEMM's MN-major TMA-aligned packed-ue8m0 scale layout.

Replaces silu_and_mul_clamp (bf16 out) + sglang_per_token_group_quant_fp8(group 32,
column-major, TMA-aligned, ue8m0) in front of a DeepGEMM MXFP8 GEMM. The existing JIT
fused kernel (silu_and_mul_contig_post_quant) only supports 128-wide groups.
Scale layout: int32 [ceil4(K/32)/4, ceil4(M)], element (quad, m) packs the e8m0 of the
4 consecutive 32-groups 4*quad..4*quad+3 of row m (byte j = group 4*quad+j).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_mul_mxfp8_kernel(
    x_ptr,
    stride_x,
    q_ptr,
    stride_q,
    s_ptr,  # int32 [aligned_k // 4, aligned_mn]
    rows,
    aligned_mn,
    inter,
    limit,
    APPLY_LIMIT: tl.constexpr,
    ROWS: tl.constexpr,
    QUADS: tl.constexpr,
):
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    quad = tl.program_id(1) * QUADS + tl.arange(0, QUADS)
    cols = tl.program_id(1) * (QUADS * 128) + tl.arange(0, QUADS * 128)
    r64 = r.to(tl.int64)
    live = (r[:, None] < rows) & (cols[None, :] < inter)
    base = x_ptr + r64[:, None] * stride_x + cols[None, :]
    gate = tl.load(base, mask=live, other=0.0)
    up = tl.load(base + inter, mask=live, other=0.0)
    if APPLY_LIMIT:
        # clamp in bf16 like the reference kernels
        lim = limit.to(tl.bfloat16)
        gate = tl.minimum(gate, lim)
        up = tl.minimum(tl.maximum(up, -lim), lim)
    gf = gate.to(tl.float32)
    val = gf / (1.0 + tl.exp(-gf)) * up.to(tl.float32)
    # round through bf16 like silu_and_mul_clamp's bf16 output, so the quantized
    # values are the ones the unfused silu -> bf16 -> group-quant chain produces
    val = val.to(tl.bfloat16).to(tl.float32)
    v3 = tl.reshape(val, (ROWS, QUADS * 4, 32))
    amax = tl.maximum(tl.max(tl.abs(v3), axis=2), 1e-10)
    bits = (amax / 448.0).to(tl.int32, bitcast=True)
    e8 = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)  # ceil to 2^k
    scale = (e8 << 23).to(tl.float32, bitcast=True)
    q = tl.reshape(v3 / scale[:, :, None], (ROWS, QUADS * 128)).to(tl.float8e4nv)
    tl.store(q_ptr + r64[:, None] * stride_q + cols[None, :], q, mask=live)
    shift = (tl.arange(0, 4) * 8)[None, None, :]
    packed = tl.sum(tl.reshape(e8, (ROWS, QUADS, 4)) << shift, axis=2)
    num_quads = (inter // 32 + 3) // 4
    s_off = quad[None, :].to(tl.int64) * aligned_mn + r64[:, None]
    tl.store(s_ptr + s_off, packed, mask=(r[:, None] < aligned_mn) & (quad[None, :] < num_quads))


def silu_mul_mxfp8_deepgemm(
    gate_up: torch.Tensor,
    swiglu_limit,
    rows_per_prog: int = 16,
    quads: int = 2,
    num_warps: int = 4,
):
    """gate_up [M, 2I] bf16 -> (q [M, I] fp8_e4m3, scale int32 [M, ceil4(I/32)/4] view
    with the strides create_per_token_group_quant_fp8_output_scale(column_major,
    tma_aligned, ue8m0) returns)."""
    assert gate_up.dtype == torch.bfloat16 and gate_up.stride(-1) == 1
    m, n = gate_up.shape
    inter = n // 2
    assert inter % 128 == 0
    k_groups = inter // 32
    aligned_mn = (m + 3) // 4 * 4
    aligned_k = (k_groups + 3) // 4 * 4
    base = torch.empty((aligned_k // 4, aligned_mn), dtype=torch.int32, device=gate_up.device)
    q = torch.empty((m, inter), dtype=torch.float8_e4m3fn, device=gate_up.device)
    if m:
        grid = (triton.cdiv(aligned_mn, rows_per_prog), triton.cdiv(aligned_k // 4, quads))
        _silu_mul_mxfp8_kernel[grid](
            gate_up,
            gate_up.stride(0),
            q,
            q.stride(0),
            base,
            m,
            aligned_mn,
            inter,
            float(swiglu_limit) if swiglu_limit is not None else 0.0,
            APPLY_LIMIT=swiglu_limit is not None,
            ROWS=rows_per_prog,
            QUADS=quads,
            num_warps=num_warps,
        )
    return q, base.transpose(0, 1)[:m, :]
