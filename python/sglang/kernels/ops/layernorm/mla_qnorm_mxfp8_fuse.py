"""Fused MLA q-path RMSNorm + wq_b-input MXFP8 swizzled quantization (DSV4.1 decode).

Collapses the two vendor launches the decode q sub-path runs between the q down-proj
and the wq_b up-proj (profile ops 56/58):

  - op56 ``self.q_norm(q_a)``            flashinfer RMSNormKernel (q_lora_rank=1280)
  - op58 wq_b input quantize            flashinfer MXFP8QuantizeSwizzledKernel (ue8m0, 128x4)

into ONE Triton launch that reads the bf16 q down-proj, applies the RMSNorm in fp32,
and emits (a) the normed bf16 activation the indexer still consumes and (b) the
FP8 activation + UE8M0 128x4-swizzled scale wrapped as ``Mxfp8SwizzledInput`` so the
UNCHANGED flashinfer blockscaled wq_b GEMM (op59) consumes it with no in-wq_b quant.

The 2-kernel vendor path rounds the RMSNorm result to bf16 (the q_norm output dtype)
before the quant reads it, so this kernel does the same (norm fp32 -> round bf16 ->
quantize from bf16), which is what makes the ue8m0 scale byte-reproducible. Byte-identical
to ``flashinfer.rmsnorm`` + ``flashinfer.mxfp8_quantize(is_sf_swizzled_layout=True)`` on
B200 sm_100 (scale 0-byte mismatch, e4m3 100% exact); flag-gated (SGLANG_FUSE_MLA_QNORM_MXFP8),
default OFF, unsupported shapes return None so the caller falls back to the vendor path.
"""
import torch
import triton
import triton.language as tl

from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput

_GROUP = 32


@triton.jit
def sglfuse_mla_qnorm_mxfp8(
    x_ptr, w_ptr, qn_ptr, qfp8_ptr, sf_ptr, eps, M, s_row,
    G: tl.constexpr, K: tl.constexpr, BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr, NG: tl.constexpr, G4: tl.constexpr,
):
    """Fused MLA q-path RMSNorm + wq_b-input MXFP8 swizzled quantization."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    k = tl.arange(0, BLOCK_K)
    kmask = k < K
    x = tl.load(x_ptr + rows[:, None] * s_row + k[None, :],
                mask=row_mask[:, None] & kmask[None, :], other=0.0).to(tl.float32)
    w = tl.load(w_ptr + k, mask=kmask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=1) / K
    rinv = tl.rsqrt(var + eps)
    normed = x * rinv[:, None] * w[None, :]
    nb = normed.to(tl.bfloat16)
    tl.store(qn_ptr + rows[:, None] * K + k[None, :], nb,
             mask=row_mask[:, None] & kmask[None, :])
    y = nb.to(tl.float32)                                  # quantize from the bf16 value
    y3 = tl.reshape(y, (BLOCK_M, NG, 32))
    amax = tl.max(tl.abs(y3), axis=2)
    raw = amax * (1.0 / 448.0)
    rb = raw.to(tl.int32, bitcast=True)
    e8 = ((rb >> 23) & 0xFF) + ((rb & 0x7FFFFF) != 0).to(tl.int32)
    e8 = tl.minimum(tl.maximum(e8, 0), 255)
    inv_scale = ((254 - e8) << 23).to(tl.float32, bitcast=True)
    q = y3 * inv_scale[:, :, None]
    q = tl.minimum(tl.maximum(q, -448.0), 448.0)
    q = tl.reshape(q, (BLOCK_M, BLOCK_K)).to(tl.float8e4nv)
    tl.store(qfp8_ptr + rows[:, None] * K + k[None, :], q,
             mask=row_mask[:, None] & kmask[None, :])
    g = tl.arange(0, NG)
    gmask = g < G
    m_tile = rows // 128
    m_in = rows % 128
    row_group = m_in % 32
    row_sub = m_in // 32
    off = (((m_tile[:, None] * G4 + (g // 4)[None, :]) * 32 + row_group[:, None]) * 16
           + row_sub[:, None] * 4 + (g % 4)[None, :])
    tl.store(sf_ptr + off, e8.to(tl.uint8), mask=row_mask[:, None] & gmask[None, :])


def is_supported(q_raw: torch.Tensor, group: int = _GROUP) -> bool:
    """True when q_raw is the 2-D bf16 [M, q_lora_rank] q down-proj this kernel can fuse."""
    if not (isinstance(q_raw, torch.Tensor) and q_raw.dim() == 2):
        return False
    if q_raw.dtype != torch.bfloat16 or not q_raw.is_cuda or q_raw.stride(-1) != 1:
        return False
    K = q_raw.shape[1]
    return K % group == 0 and (K // group) % 4 == 0


def _scale_numel(M: int, K: int, group: int = _GROUP) -> int:
    padded_m = ((M + 127) // 128) * 128
    g = K // group
    padded_cols = ((g + 3) // 4) * 4
    return padded_m * padded_cols


def fused_qnorm_mxfp8_swizzled_input(q_raw, weight, eps, block_m: int = 1, num_warps: int = 4):
    """RMSNorm(q_raw, weight, eps) fused with the wq_b-input MXFP8 128x4-swizzled quant.

    Returns (q_normed_bf16 [M,K], Mxfp8SwizzledInput(q_fp8 [M,K], sf [padded])). The
    normed bf16 keeps the indexer / other consumers working; the Mxfp8SwizzledInput
    feeds the unchanged flashinfer blockscaled wq_b GEMM. Returns None on an
    unsupported shape so the caller falls back to the vendor two-kernel path.
    """
    if not is_supported(q_raw):
        return None
    M, K = q_raw.shape
    G = K // _GROUP
    BLOCK_K = 1 << (K - 1).bit_length()
    q_normed = torch.empty((M, K), dtype=torch.bfloat16, device=q_raw.device)
    q_fp8 = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=q_raw.device)
    sf = torch.empty(_scale_numel(M, K), dtype=torch.uint8, device=q_raw.device)
    if M > 0:
        sglfuse_mla_qnorm_mxfp8[(triton.cdiv(M, block_m),)](
            q_raw, weight, q_normed, q_fp8, sf, float(eps), M, q_raw.stride(0),
            G=G, K=K, BLOCK_M=block_m, BLOCK_K=BLOCK_K, NG=BLOCK_K // _GROUP,
            G4=G // 4, num_warps=num_warps,
        )
    return q_normed, Mxfp8SwizzledInput(q_fp8, sf)
