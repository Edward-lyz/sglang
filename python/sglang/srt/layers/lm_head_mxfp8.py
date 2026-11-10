"""[fusion-opt] SGLANG_OPT_LM_HEAD_MXFP8: the bf16 lm_head as an MXFP8 DeepGEMM GEMM.

The weight is quantized once (per-32 ue8m0 scales, DeepGEMM's packed MN-major layout)
on the first eager call, i.e. during the CUDA-graph warmup before capture; calls that
come first inside a capture keep the bf16 matmul. Lossy: decode logits change.
"""

import logging
from typing import Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_STATE = "_lm_head_mxfp8_state"
_ROWS_PER_CHUNK = 8192


def _quantize_weight(weight: torch.Tensor):
    import deep_gemm.utils.layout

    n, k = weight.shape
    q_chunks, s_chunks = [], []
    for i in range(0, n, _ROWS_PER_CHUNK):
        blk = weight[i : i + _ROWS_PER_CHUNK].float().view(-1, k // 32, 32)
        amax = blk.abs().amax(-1).clamp(min=1e-10)
        scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
        q_chunks.append((blk / scale.unsqueeze(-1)).to(torch.float8_e4m3fn).view(-1, k))
        s_chunks.append(scale)
    scale = torch.cat(s_chunks)
    packed = deep_gemm.utils.layout.get_mn_major_tma_aligned_packed_ue8m0_tensor(scale)
    return torch.cat(q_chunks), packed


def _prepare(lm_head) -> Optional[tuple]:
    state = getattr(lm_head, _STATE, None)
    if state is not None:
        return state or None
    weight = getattr(lm_head, "weight", None)
    if torch.cuda.is_current_stream_capturing():
        return None
    from sglang.srt.layers import deep_gemm_wrapper

    ok = (
        weight is not None
        and weight.is_cuda
        and weight.dtype == torch.bfloat16
        and weight.dim() == 2
        and weight.shape[0] % 64 == 0
        and weight.shape[1] % 128 == 0
        and deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
        and deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
    )
    state = _quantize_weight(weight) if ok else ()
    setattr(lm_head, _STATE, state)
    from sglang.srt.utils.community_opt import log_community_opt_once

    log_community_opt_once(
        "fusion-opt lm_head_mxfp8",
        "lm_head %s as MXFP8 DeepGEMM (weight %s)",
        "runs" if ok else "stays bf16",
        tuple(weight.shape) if weight is not None else None,
    )
    return state or None


def maybe_mxfp8_lm_head(
    hidden_states: torch.Tensor, lm_head
) -> Optional[torch.Tensor]:
    """bf16 logits from the MXFP8 lm_head, or None to use the original matmul."""
    if not envs.SGLANG_OPT_LM_HEAD_MXFP8.get():
        return None
    if (
        hidden_states.dtype != torch.bfloat16
        or hidden_states.dim() < 2
        or hidden_states.numel() == 0
    ):
        return None
    state = _prepare(lm_head)
    if state is None or hidden_states.shape[-1] != state[0].shape[1]:
        return None
    from sglang.srt.layers.quantization.fp8_utils import (
        _deepgemm_w8a8_mxfp8_linear_with_fallback,
    )

    # The helper flattens leading dims and restores them on the output.
    return _deepgemm_w8a8_mxfp8_linear_with_fallback(hidden_states, state[0], state[1])
