"""Fused MLA q-path RMSNorm + wq_b-input MXFP8 quantization in DeepGEMM's layout.

Same fusion as ``mla_qnorm_mxfp8_fuse`` (op56 q_norm + op58 wq_b input quant in one
launch), but for the DeepGEMM MXFP8 dense route (SGLANG_OPT_BLOCK_FP8_MXFP8_DEEPGEMM=1),
which takes per-32 UE8M0 scales packed 4 per int32, MN-major with the token dim
aligned to 4 (``Mxfp8DeepGemmInput``), instead of flashinfer's 128x4 swizzled bytes.

The two-kernel path rounds the RMSNorm result to bf16 (q_norm's output dtype) and
``sglang_per_token_group_quant_fp8(group 32, ue8m0, column-major, TMA-aligned)``
quantizes that bf16 tensor, so this kernel does the same (norm in fp32 -> bf16 ->
amax clamped to eps -> power-of-two scale rounded up -> e4m3). Flag-gated
(SGLANG_FUSE_QNORM_MXFP8_DEEPGEMM), default off; unsupported shapes return None.
"""

import torch
import triton
import triton.language as tl

_GROUP = 32
_EPS = 1e-10


@triton.jit
def _qnorm_mxfp8_deepgemm_kernel(
    x_ptr, w_ptr, qn_ptr, qfp8_ptr, sf_ptr, eps, M, s_row, m_pad, quant_eps,
    K: tl.constexpr, G: tl.constexpr, BLOCK_K: tl.constexpr, NG: tl.constexpr,
):
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK_K)
    kmask = k < K
    x = tl.load(x_ptr + row * s_row + k, mask=kmask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + k, mask=kmask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / K
    rinv = tl.rsqrt(var + eps)
    nb = (x * rinv * w).to(tl.bfloat16)
    tl.store(qn_ptr + row * K + k, nb, mask=kmask)
    y = tl.reshape(nb.to(tl.float32), (NG, 32))
    amax = tl.maximum(tl.max(tl.abs(y), axis=1), quant_eps)
    rb = (amax * (1.0 / 448.0)).to(tl.int32, bitcast=True)
    e8 = ((rb >> 23) & 0xFF) + ((rb & 0x7FFFFF) != 0).to(tl.int32)
    e8 = tl.minimum(tl.maximum(e8, 1), 254)
    inv_scale = ((254 - e8) << 23).to(tl.float32, bitcast=True)
    q = tl.minimum(tl.maximum(y * inv_scale[:, None], -448.0), 448.0)
    q = tl.reshape(q, (BLOCK_K,)).to(tl.float8e4nv)
    tl.store(qfp8_ptr + row * K + k, q, mask=kmask)
    g = tl.arange(0, NG)
    # int32 [K/128, m_pad] storage; byte g%4 of word (g//4, row) holds group g
    off = ((g // 4) * m_pad + row) * 4 + (g % 4)
    tl.store(sf_ptr + off, e8.to(tl.uint8), mask=g < G)


def is_supported(q_raw: torch.Tensor) -> bool:
    """Input q_raw tensor checker."""
    if not (isinstance(q_raw, torch.Tensor) and q_raw.dim() == 2):
        return False
    if q_raw.dtype != torch.bfloat16 or not q_raw.is_cuda or q_raw.stride(-1) != 1:
        return False
    return q_raw.shape[1] % 128 == 0


def fused_qnorm_mxfp8_deepgemm(q_raw: torch.Tensor, weight: torch.Tensor, eps: float):
    """Returns (q_normed_bf16 [M, K], (q_fp8 [M, K], scales int32 [M, K/128] view)),
    or None for an unsupported input."""
    if not is_supported(q_raw):
        return None
    M, K = q_raw.shape
    G = K // _GROUP
    m_pad = (M + 3) // 4 * 4
    BLOCK_K = 1 << (K - 1).bit_length()
    q_normed = torch.empty((M, K), dtype=torch.bfloat16, device=q_raw.device)
    q_fp8 = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=q_raw.device)
    sf_store = torch.empty((K // 128, m_pad), dtype=torch.int32, device=q_raw.device)
    if M > 0:
        _qnorm_mxfp8_deepgemm_kernel[(M,)](
            q_raw, weight, q_normed, q_fp8, sf_store.view(torch.uint8), float(eps), M,
            q_raw.stride(0), m_pad, _EPS,
            K=K, G=G, BLOCK_K=BLOCK_K, NG=BLOCK_K // _GROUP, num_warps=4,
        )
    return q_normed, (q_fp8, sf_store.mT[:M])
