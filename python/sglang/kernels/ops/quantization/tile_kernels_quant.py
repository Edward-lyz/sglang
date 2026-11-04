"""[fusion-opt] TileKernels (deepseek-ai/TileKernels) activation casts for the DeepGEMM
MXFP8 linear route.

Both helpers return the exact operand `sglang_per_token_group_quant_fp8(x, 32,
column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=True)` returns:
e4m3 data [M, K] and int32 [M, ceil4(K/32)/4] packed-ue8m0 scales, MN-major with the
token dim aligned to 4 (stride (1, align4(M))), i.e. what deep_gemm reads with
recipe (1, 32).

* per_token_cast_mxfp8: tile_kernels.quant.per_token_cast(x, 'e4m3', 32, ...)
* rmsnorm_cast_mxfp8: tile_kernels.quant.norm_forward_and_per_token_cast, i.e.
  RMSNorm (rounded to bf16 first, as the unfused path) + the same cast in one pass,
  also returning the bf16 normed rows for other consumers.

tile_kernels needs tilelang >= 0.1.15 and compiles each shape class on first use
(TileLang JIT); it is imported lazily and a failed import disables both helpers.
"""

import logging
import sys
from typing import Optional, Tuple

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# sglang's group quant clamps amax at eps=1e-10; tile_kernels defaults to 1e-4, which
# would coarsen near-zero groups. Matching it keeps the two casts bit-identical.
_AMAX_CLAMP = 1e-10

_tk = None
_tk_failed = False


def _tile_kernels():
    """Lazily import tile_kernels once; return the module, or None if unavailable."""
    global _tk, _tk_failed
    if _tk is not None or _tk_failed:
        return _tk
    path = envs.SGLANG_TILE_KERNELS_PATH.get()
    if path and path not in sys.path:
        sys.path.insert(0, path)
    try:
        import tile_kernels
        import tile_kernels.quant  # noqa: F401

        tile_kernels.set_amax_clamp_for_quant("e4m3", False, _AMAX_CLAMP)
        _tk = tile_kernels
        import tilelang

        logger.info(
            "[fusion-opt] tile_kernels loaded from %s (tilelang %s)",
            getattr(tile_kernels, "__file__", "?"),
            getattr(tilelang, "__version__", "?"),
        )
    except Exception as e:  # ImportError, or tilelang API mismatch at import time
        _tk_failed = True
        logger.warning(
            "[fusion-opt] tile_kernels unavailable (%s: %s); "
            "SGLANG_OPT_TK_* falls back to the stock kernels",
            type(e).__name__,
            e,
        )
    return _tk


def tk_per_token_cast_enabled() -> bool:
    """Whether SGLANG_OPT_TK_PER_TOKEN_CAST is on and tile_kernels is loadable."""
    return envs.SGLANG_OPT_TK_PER_TOKEN_CAST.get() and _tile_kernels() is not None


def tk_qnorm_cast_enabled() -> bool:
    """Whether SGLANG_OPT_TK_QNORM_CAST is on and tile_kernels is loadable."""
    return envs.SGLANG_OPT_TK_QNORM_CAST.get() and _tile_kernels() is not None


def per_token_cast_mxfp8(
    x: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """x: bf16 [M, K], K % 64 == 0, unit stride on K. None when unsupported."""
    tk = _tile_kernels()
    if (
        tk is None
        or x.dim() != 2
        or x.dtype != torch.bfloat16
        or x.shape[1] % 64 != 0
        or x.stride(1) != 1
    ):
        return None
    return tk.quant.per_token_cast(
        x,
        "e4m3",
        32,
        use_tma_aligned_col_major_sf=True,
        round_sf=True,
        use_packed_ue8m0=True,
    )


def rmsnorm_cast_mxfp8(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> Optional[Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
    """RMSNorm(x) * weight -> (normed bf16 [M, K], (e4m3 [M, K], packed sf)).

    x: bf16 [M, K] with K % 32 == 0, K <= 8192 and a row stride divisible by 16
    elements (a column slice of a fused projection output is fine). None when
    unsupported."""
    tk = _tile_kernels()
    if (
        tk is None
        or x.dim() != 2
        or x.dtype != torch.bfloat16
        or weight.dtype != x.dtype
        or not weight.is_contiguous()
        or x.shape[1] % 32 != 0
        or x.shape[1] > 8192
        or x.stride(1) != 1
        or x.stride(0) % 16 != 0
    ):
        return None
    (q, sf), normed, _rstd, _ = tk.quant.norm_forward_and_per_token_cast(
        x,
        weight,
        eps,
        "e4m3",
        32,
        use_tma_aligned_col_major_sf=True,
        round_sf=True,
        use_packed_ue8m0=True,
        with_out_bf16=True,
    )
    return normed, (q, sf)
