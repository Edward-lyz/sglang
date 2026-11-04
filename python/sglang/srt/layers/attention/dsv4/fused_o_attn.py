"""Helpers for FlashMLA's fused ``norm + RoPE + sparse attention + inverse RoPE + fp8 cast`` op
(``flash_mla.fused_norm_rope_attn_rope_cast``, DeepSeek-V4.1, SM100).

The op consumes a q whose rows were permuted by ``permute_q_b_proj`` and writes the attention output
as per-32 ue8m0 fp8 that feeds ``deep_gemm.fp8_einsum(..., recipe=(1, 1, 32))`` against a ``wo_a``
prepared by ``prepare_wo_a``. Everything here is self-contained (no sglang imports) so it can be unit
tested without the engine. Requires an upstream DeepGEMM that supports (1, 1, 32) SF layouts.
"""

from typing import Optional, Tuple

import torch

ROPE_DIM = 64
SF_GRAN = 32


def fused_o_attn_available() -> bool:
    """ fused_o_attn_available """
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        from flash_mla import fused_norm_rope_attn_rope_cast  # noqa: F401
    except ImportError:
        return False
    return True


def q_row_permutation(h_q: int, d_q: int, device) -> torch.Tensor:
    """out_row[i] = destination row of wq_b row i under ``permute_q_b_proj``."""
    rows = torch.arange(h_q * d_q, device=device)
    d = rows % d_q
    return d % 16 + (rows // d_q) * 16 + (d // 16) * (h_q * 16)


def unpermute_q(q_perm: torch.Tensor, h_q: int, d_q: int) -> torch.Tensor:
    """[T, h_q*d_q] (permuted layout) -> [T, h_q, d_q]; for consumers that need the plain layout."""
    t = q_perm.shape[0]
    return q_perm.view(t, d_q // 16, h_q, 16).transpose(1, 2).reshape(t, h_q, d_q)


def make_cos_sin_cache(freqs_cis: torch.Tensor) -> torch.Tensor:
    """complex64 [N, 32] (engine's freqs_cis) -> float32 [N, 64] = [cos | sin]."""
    return torch.cat([freqs_cis.real, freqs_cis.imag], dim=-1).float().contiguous()


def quantize_rows_ue8m0(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """bf16/fp32 [N, K] -> (fp8 e4m3 [N, K], float32 pow2 scales [N, K/32]), 1x32 groups.

    Lossless for weights that came from e4m3 x 2^e block-scaled storage: the scale is the smallest
    power of two keeping |x| <= 448, so every value stays on the e4m3 grid.
    """
    n, k = w.shape
    g = w.float().view(n, k // SF_GRAN, SF_GRAN)
    amax = g.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    q = (g / scale).to(torch.float8_e4m3fn).view(n, k)
    return q, scale.squeeze(-1)


def prepare_wo_a(
    wo_a: torch.Tensor, n_groups: int, heads_per_group: int, head_dim: int = 512
) -> Tuple[torch.Tensor, torch.Tensor]:
    """wo_a [G*R, K] (bf16, K = heads_per_group*head_dim) -> (fp8 [G, R, K], sf) in the layout
    ``fp8_einsum("bhr,hdr->bhd", ..., recipe=(1, 1, 32))`` expects after ``permute_wv_proj``."""
    import deep_gemm
    from flash_mla import fused_norm_rope_attn_rope_cast as fused

    n, k = wo_a.shape
    r = n // n_groups
    w_fp8, sf = quantize_rows_ue8m0(wo_a)
    sf = deep_gemm.transform_sf_into_required_layout(
        sf.view(n_groups, r, k // SF_GRAN).contiguous(),
        r,
        k,
        num_groups=n_groups,
        recipe=(1, 1, SF_GRAN),
        is_sfa=False,
    )
    return fused.permute_wv_proj((w_fp8.view(n_groups, r, k), sf), heads_per_group, head_dim)


def fused_prefill(
    q_perm: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    sm_scale: float,
    n_wv_group: int,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """q_perm [T, h_q, 512] bf16 (permuted, NOT roped); kv [S, 1, 512] bf16; indices [T, 1, topk] int32.
    Returns (out_fp8, out_sf) ready for ``wo_a_einsum``."""
    from flash_mla import fused_norm_rope_attn_rope_cast as fused

    out_fp8, out_sf, _, _ = fused.prefill(
        False, 1e-6, positions.to(torch.int32), False, ROPE_DIM, cos_sin_cache,
        n_wv_group, SF_GRAN, True, True, True,
        q_perm, kv, indices, sm_scale, 512, attn_sink, topk_length,
    )  # fmt: skip
    return out_fp8, out_sf


def fused_decode(
    q_perm: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    sm_scale: float,
    n_wv_group: int,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode counterpart of fused_prefill, reading the paged V4.1 caches directly.

    q_perm [T, h_q, 512] bf16 (permuted, NOT roped); k_cache [blocks, page, 1, 528] (fp8 V4.1);
    indices [T, topk] int32 page-relative slots (-1 invalid); extra_k_cache is 528 B (fp8) or 288 B (fp4).
    Returns (out_fp8, out_sf) ready for ``wo_a_einsum``."""
    from flash_mla import fused_norm_rope_attn_rope_cast as fused

    out_fp8, out_sf, _ = fused.decode(
        False, 1e-6, positions.to(torch.int32), False, ROPE_DIM, cos_sin_cache,
        n_wv_group, SF_GRAN, True, True, True,
        q_perm, k_cache, indices, sm_scale, 512, attn_sink, topk_length,
        extra_k_cache, extra_indices, extra_topk_length,
    )  # fmt: skip
    return out_fp8, out_sf


def wo_a_einsum(
    out_fp8: torch.Tensor, out_sf: torch.Tensor, wo_a_fp8: torch.Tensor, wo_a_sf: torch.Tensor
) -> torch.Tensor:
    """fp8 einsum ``"bhr,hdr->bhd"`` of the fused-attn output ``(out_fp8, out_sf)`` with wo_a's
    fp8 weight ``(wo_a_fp8, wo_a_sf)`` using ``recipe=(1, 1, 32)``; returns bf16 ``[T, G, D]``
    ready for ``wo_b``."""
    import deep_gemm

    t, g, _ = out_fp8.shape
    out = torch.empty(t, g, wo_a_fp8.shape[1], dtype=torch.bfloat16, device=out_fp8.device)
    deep_gemm.fp8_einsum(
        "bhr,hdr->bhd", (out_fp8, out_sf), (wo_a_fp8, wo_a_sf), out, recipe=(1, 1, SF_GRAN)
    )
    return out
