"""DeepGEMM `mega_mhc`: the whole mHC sublayer boundary in one launch.

DeepSeek-V4.1 runs the "predecessor-pre" mHC scheme (``hc_pre_from_prev_sublayer``):
at every sublayer boundary the model computes

    R1 = hc_post(x, R0, post_mix, comb_mix)              # new residual
    S1 = rmsnorm(sum_k pre_mix[k] * R1[:, k, :])         # next sublayer input
    (pre1, post1, comb1) = hc_mix_stats(R1, fn, ...)     # mixes for the NEXT boundary

where ``pre_mix`` is the pre-mix produced by the *previous* boundary. That is
exactly DeepGEMM's "shifted" mega_mhc kernel: it takes the shifted pre-mix as an
input and emits the freshly computed one as ``new_prev_mix``. Four launches
(post + combine + RMSNorm + pre-norm GEMM/sinkhorn) collapse into one.

Gated by ``SGLANG_OPT_DEEPGEMM_MEGA_MHC``; :func:`mega_mhc_boundary` returns
``None`` for any geometry the kernel does not cover so callers fall through to
the existing kernels untouched.

CUDA graphs
-----------
DeepGEMM's host API asserts the stream is not capturing, but only on its
lazy-init path: ``mega_mhc.hpp:33`` sits inside ``get_split_barriers()``, whose
cache is keyed by stream. Measured behaviour (un-patched library, SM100):

* warmup on the stream that is then captured -> captures and replays correctly,
  for any shape (the cache key is the stream, not the shape);
* no warmup, or warmup on a different stream -> the assert throws, *before* any
  work is issued on the stream, so the capture survives and the caller may emit
  the unfused path into the same graph.

sglang's graph runner already runs two warmup forwards on the capture stream
(``full_cuda_graph_backend.capture_one``) with that stream current
(``parallel_state.graph_capture``), so the fused kernel does get captured. We
therefore always attempt the launch and treat the capture assert as a per-call
fallback rather than refusing to run under capture.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# The kernel hardwires 4 routes (deep_gemm/layout/mega_mhc.cuh: kNumRoutes) and
# splits the hidden dim across 4 warps of 256-element packs.
_SUPPORTED_HC_MULT = 4
_HIDDEN_ALIGNMENT = 1024

_RUNTIME_DISABLED = False
_IMPORT_WARNED = False
# Keyed by boundary label so each of the two mHC boundaries per layer
# ("attn->FFN" and "FFN->attn") reports once instead of masking the other.
_FIRED_LOGGED: set = set()
_CAPTURED_LOGGED: set = set()
_CAPTURE_FALLBACK_WARNED = False

# DeepGEMM raises this (host-side, before any stream work) when it would have to
# lazily initialise its per-stream workspace while the stream is capturing.
_CAPTURE_ASSERT_MARK = "CaptureStatus"


def _get_mega_mhc():
    global _RUNTIME_DISABLED, _IMPORT_WARNED
    try:
        import deep_gemm

        return deep_gemm.mega_mhc
    except Exception as err:  # noqa: BLE001
        if not _IMPORT_WARNED:
            logger.warning("deep_gemm.mega_mhc is unavailable, falling back: %s", err)
            _IMPORT_WARNED = True
        _RUNTIME_DISABLED = True
        return None


def is_mega_mhc_supported(
    x: torch.Tensor,
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_mult: int,
    norm_weight: Optional[torch.Tensor],
) -> bool:
    """True if the kernel can run this call."""
    if _RUNTIME_DISABLED or norm_weight is None:
        return False
    if hc_mult != _SUPPORTED_HC_MULT or x.dim() != 2 or residual.dim() != 3:
        return False
    num_tokens, hidden = x.shape
    if num_tokens == 0 or hidden % _HIDDEN_ALIGNMENT != 0:
        return False
    if residual.shape != (num_tokens, hc_mult, hidden):
        return False
    if x.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16:
        return False
    if norm_weight.dtype != torch.bfloat16 or norm_weight.shape != (hidden,):
        return False
    mix_hc = (2 + hc_mult) * hc_mult
    return fn.dtype == torch.float32 and tuple(fn.shape) == (mix_hc, hc_mult * hidden)


def mega_mhc_boundary(
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_pre_mix: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    hc_mult: int,
    hc_norm_eps: float,
    hc_eps: float,
    hc_post_mult: float,
    sinkhorn_iters: int,
    norm_weight: Optional[torch.Tensor],
    norm_eps: float,
    label: str = "mHC",
) -> Optional[Tuple[torch.Tensor, ...]]:
    """One mHC boundary of the predecessor-pre scheme.

    Returns ``(new_residual, layer_input, new_pre, new_post, new_comb)`` with
    ``new_pre/new_post`` shaped ``(T, hc_mult)`` and ``new_comb`` ``(T, hc_mult,
    hc_mult)`` -- the same contracts as ``hc_post`` / ``_hc_mix_stats`` -- or
    ``None`` when the kernel cannot serve this call.
    """
    global _RUNTIME_DISABLED, _FIRED_LOGGED, _CAPTURED_LOGGED
    global _CAPTURE_FALLBACK_WARNED

    if not is_mega_mhc_supported(x, residual, fn, hc_mult, norm_weight):
        return None
    mega_mhc = _get_mega_mhc()
    if mega_mhc is None:
        return None

    capturing = torch.cuda.is_current_stream_capturing()
    num_tokens, hidden = x.shape
    device = x.device
    # The kernel reads the per-token coefficients through TMA descriptors built
    # from (T, hc_mult, 1) / (T, hc_mult, hc_mult) fp32 views.
    prev_pre_in = prev_pre_mix.reshape(num_tokens, hc_mult, 1).float().contiguous()
    post_in = post_mix.reshape(num_tokens, hc_mult, 1).float().contiguous()
    comb_in = comb_mix.reshape(num_tokens, hc_mult, hc_mult).float().contiguous()

    new_residual = torch.empty_like(residual)
    new_pre = torch.empty((num_tokens, hc_mult, 1), dtype=torch.float32, device=device)
    new_post = torch.empty((num_tokens, hc_mult, 1), dtype=torch.float32, device=device)
    new_comb = torch.empty(
        (num_tokens, hc_mult, hc_mult), dtype=torch.float32, device=device
    )
    layer_input = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device=device)

    try:
        mega_mhc(
            x.contiguous(),
            residual.contiguous(),
            prev_pre_in,
            post_in,
            comb_in,
            fn.contiguous(),
            hc_scale.float().contiguous(),
            hc_base.float().contiguous(),
            hc_mult,
            hc_norm_eps,
            hc_eps,
            hc_post_mult,
            hc_eps,
            sinkhorn_iters,
            norm_weight,
            norm_eps,
            1.0,
            new_residual,
            new_pre,
            new_post,
            new_comb,
            layer_input,
        )
    except Exception as err:  # noqa: BLE001
        if _CAPTURE_ASSERT_MARK in str(err):
            # Lazy init refused under capture. Nothing was issued on the stream,
            # so the graph stays valid: fall back for this call only and keep
            # the fused path enabled for streams that were warmed up.
            if not _CAPTURE_FALLBACK_WARNED:
                logger.warning(
                    "deep_gemm.mega_mhc declined a capturing stream that was not "
                    "warmed up; this graph records the unfused mHC path"
                )
                _CAPTURE_FALLBACK_WARNED = True
            return None
        logger.warning("deep_gemm.mega_mhc failed, disabling the fused path: %s", err)
        _RUNTIME_DISABLED = True
        return None

    if label not in _FIRED_LOGGED:
        logger.info(
            "deep_gemm.mega_mhc is serving the %s mHC boundary "
            "(tokens=%d, hidden=%d, hc_mult=%d)",
            label,
            num_tokens,
            hidden,
            hc_mult,
        )
        _FIRED_LOGGED.add(label)
    if capturing and label not in _CAPTURED_LOGGED:
        logger.info(
            "deep_gemm.mega_mhc [%s] was captured into a CUDA graph (tokens=%d)",
            label,
            num_tokens,
        )
        _CAPTURED_LOGGED.add(label)

    return (
        new_residual,
        layer_input,
        new_pre.squeeze(-1),
        new_post.squeeze(-1),
        new_comb,
    )
