"""Fused q RMSNorm + per-token group FP8 quant + k_nope RMSNorm (GLM 5.2 MLA).

Registers the ``qkv_norm_quant`` CUDA kernel as an in-tree JIT module and exposes
:func:`fused_qkv_norm_quant`, which collapses the three back-to-back launches in
``DeepseekV2AttentionMLA.forward_absorb_prepare`` (q_a_layernorm, kv_a_layernorm,
q_b_proj activation FP8 group-quant) into a single kernel launch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# FP8 e4m3 range for NVIDIA Hopper/Blackwell (this operator targets B200/sm_100).
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0
FP8_MIN = -448.0


@cache_once
def _jit_qkv_norm_quant_module(
    quant_group_size: int,
    scale_ue8m0: bool,
    write_q_norm: bool,
    q_hidden: int,
    k_hidden: int,
) -> "Module":
    """Build and cache the JIT module specialised for the given template args."""
    # Template args order must match QKVNormQuantKernel<
    #   kGroupSize, kScaleUE8M0, kUsePDL, kWriteQNorm, kQHidden, kKHidden>.
    # Specialised per (group_size, ue8m0, pdl, write_q_norm, q_hidden, k_hidden).
    args = make_cpp_args(
        quant_group_size,
        scale_ue8m0,
        is_arch_support_pdl(),
        write_q_norm,
        q_hidden,
        k_hidden,
    )
    return load_jit(
        "qkv_norm_quant",
        *args,
        cuda_files=["elementwise/qkv_norm_quant.cuh"],
        cuda_wrappers=[("run", f"QKVNormQuantKernel<{args}>::run")],
    )


def fused_qkv_norm_quant(
    q: torch.Tensor,
    q_weight: torch.Tensor,
    q_eps: float,
    k: torch.Tensor,
    k_weight: torch.Tensor,
    k_eps: float,
    *,
    group_size: int = 128,
    scale_ue8m0: bool = True,
    write_q_norm: bool = True,
    q_norm_output: Optional[torch.Tensor] = None,
    q_output: Optional[torch.Tensor] = None,
    q_output_scale: Optional[torch.Tensor] = None,
    k_norm_output: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused q RMSNorm + per-token group FP8 quant, and k_nope RMSNorm.

    Collapses the GLM 5.2 / DeepSeek MLA ``forward_absorb_prepare`` hot path
    (three back-to-back launches touching the same token row) into one launch::

        q      = q_a_layernorm(q)          # RMSNormKernel
        k_nope = kv_a_layernorm(k_nope)    # RMSNormKernel
        q_fp8, q_scale = <q_b_proj activation quant>  # per_token_group_quant_8bit

    Returns ``(q_norm, q_fp8, q_scale, k_norm)`` where:
      * ``q_norm``  : bf16 RMSNorm(q)                      (feeds the indexer / q_lora)
      * ``q_fp8``   : float8_e4m3fn per-token group quant  (feeds q_b_proj block-fp8 GEMM)
      * ``q_scale`` : UE8M0 (int32-packed, column-major TMA-aligned) or fp32 scales
      * ``k_norm``  : bf16 RMSNorm(k_nope)                 (unquantized)

    ``q_scale`` is allocated by the same
    ``create_per_token_group_quant_fp8_output_scale`` helper the deep_gemm
    activation quant uses, so ``(q_fp8, q_scale)`` is a drop-in for the q_b_proj
    block-fp8 path (``Fp8LinearMethod.apply`` consumes a ``(input, input_scale)``
    tuple directly). Set ``write_q_norm=False`` to skip the bf16 q_norm store when
    q_lora is not consumed (non-NSA), saving bandwidth at large N.
    """
    assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16, "q/k must be bf16"
    assert q.dim() == 2 and k.dim() == 2, "q/k must be 2D [num_tokens, hidden]"
    assert q.shape[0] == k.shape[0], "q and k must share num_tokens"
    assert q.shape[-1] % group_size == 0, "q hidden must be divisible by group_size"
    assert q.stride(-1) == 1 and k.stride(-1) == 1, "q/k last dim must be contiguous"
    if scale_ue8m0:
        assert group_size == 128, "UE8M0 scale requires group_size == 128"

    num_tokens = q.shape[0]

    if q_norm_output is None:
        q_norm_output = torch.empty_like(q)
    if q_output is None:
        q_output = torch.empty(q.shape, device=q.device, dtype=FP8_DTYPE)
    if q_output_scale is None:
        # Imported lazily: keeps this low-level jit_kernel module free of a
        # top-level dependency on the srt.layers.quantization package.
        from sglang.kernels.ops.quantization.fp8_kernel import (
            create_per_token_group_quant_fp8_output_scale,
        )

        q_output_scale = create_per_token_group_quant_fp8_output_scale(
            x_shape=q_output.shape,
            device=q.device,
            group_size=group_size,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=scale_ue8m0,
        )
    if k_norm_output is None:
        k_norm_output = torch.empty_like(k)

    if num_tokens > 0:
        module = _jit_qkv_norm_quant_module(
            group_size, scale_ue8m0, write_q_norm, q.shape[-1], k.shape[-1]
        )
        module.run(
            q,
            q_weight,
            q_norm_output,
            q_output,
            q_output_scale,
            k,
            k_weight,
            k_norm_output,
            float(q_eps),
            float(k_eps),
            float(FP8_MIN),
            float(FP8_MAX),
        )

    return q_norm_output, q_output, q_output_scale, k_norm_output
