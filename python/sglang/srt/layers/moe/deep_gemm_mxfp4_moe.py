# SPDX-License-Identifier: Apache-2.0
"""DeepGemm MXFP4 MoE kernel."""
from typing import Optional

import torch

from sglang.srt.layers.quantization.mxfp4_tensor import (
    mxfp8_quantize,
    mxfp8_quantize_masked,
)
from sglang.srt.utils import is_cuda_alike

_is_cuda_alike = is_cuda_alike()


def _prepare_mxfp8_scale(
    scale: torch.Tensor,
    k_gemm: int,
    masked_m: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Prepare activation scale for MXFP8-MXFP4 grouped GEMM kernel.

    The kernel expects scale shape [G, M, K/32] in float8_e8m0fnu with
    swizzled layout. Input scale may be:
      - int32 packed UE8M0 (4 e8m0 bytes per int32), or
      - float32 / float8_e8m0fnu with coarser block size

    This function delegates all heavy lifting (unpack, repeat, swizzle) to
    a single fused Triton kernel to avoid Python-level elementwise ops
    (.contiguous(), .view(), .reshape()) that break CUDA graph capture
    and add unnecessary kernel launch overhead.
    """
    is_int32_packed = scale.dtype == torch.int32

    # Compute uint8-equivalent last dim for repeat factor calculation
    actual_sf = scale.shape[-1] * 4 if is_int32_packed else scale.shape[-1]
    needed_sf = k_gemm // 32
    assert needed_sf % actual_sf == 0, (
        f"Scale last dim {actual_sf} does not evenly divide "
        f"needed K/32={needed_sf} (k_gemm={k_gemm})"
    )
    repeat_factor = needed_sf // actual_sf
    if masked_m is not None:
        return mxfp8_quantize_masked(
            scale, repeat_factor, masked_m,
            is_int32_packed=is_int32_packed,
        )
    return mxfp8_quantize(scale, repeat_factor, is_int32_packed=is_int32_packed)


def deep_gemm_mxfp4_moe_deepep_ll(
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
    w13_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_scale: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    workspace: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    DeepGemm MXFP4 MoE layer for the DeepEP low-latency path.
    Follows the pattern of DeepGemmRunnerCore._run_masked_gemm.

    Parameters:
    - hidden_states: [num_experts, m, k] FP8 input activations
    - hidden_states_scale: scale for the input activations
    - w13_weight: [num_experts, 2*n, k] MXFP4 gate+up weights (float4_e2m1fn_x2)
    - w13_scale: [num_experts, 2*n, k//32] weight scales (float8_e8m0fnu)
    - w2_weight: [num_experts, k, n] MXFP4 down weights (float4_e2m1fn_x2)
    - w2_scale: [num_experts, k, n//32] weight scales (float8_e8m0fnu)
    - masked_m: [num_experts] actual token count per expert
    - expected_m: expected max tokens per expert

    Returns:
    - torch.Tensor: [num_experts, m, k] bf16 output
    """
    from sglang.srt.layers import deep_gemm_wrapper
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_8bit,
    )

    device = hidden_states.device
    num_groups, m, k = hidden_states.shape
    n = w13_weight.size(1)  # 2 * intermediate_size

    # Prepare activation scale for GroupGemm-0
    # hidden_states_scale = _prepare_mxfp8_scale(
    #     hidden_states_scale, k, masked_m=masked_m
    # )

    # GroupGemm-0: gate+up projection
    gateup_output = torch.empty(
        (num_groups, m, n), device=device, dtype=torch.bfloat16
    )
    deep_gemm_wrapper.grouped_gemm_nt_f8f4bf16_masked(
        (hidden_states, hidden_states_scale),
        (w13_weight, w13_scale),
        gateup_output,
        masked_m,
        expected_m,
        disable_ue8m0_cast=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        workspace=workspace,
    )

    # Act: SiLU + Mul + Quantize to FP8 with block scales
    scale_block_size = 128
    down_input, down_input_scale = sglang_per_token_group_quant_8bit(
        x=gateup_output,
        dst_dtype=torch.float8_e4m3fn,
        group_size=scale_block_size,
        masked_m=masked_m,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        fuse_silu_and_mul=True,
        enable_v2=True,
    )
    del gateup_output

    # GroupGemm-1: down projection
    n2 = w2_weight.shape[1]
    k_gemm1 = down_input.shape[-1]  # contraction dim = intermediate_size

    # Prepare activation scale for GroupGemm-1
    down_input_scale = _prepare_mxfp8_scale(
        down_input_scale, k_gemm1, masked_m=masked_m
    )

    down_output = torch.empty(
        (num_groups, m, n2), device=device, dtype=torch.bfloat16
    )
    deep_gemm_wrapper.grouped_gemm_nt_f8f4bf16_masked(
        (down_input, down_input_scale),
        (w2_weight, w2_scale),
        down_output,
        masked_m,
        expected_m,
        disable_ue8m0_cast=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        workspace=workspace,
    )

    return down_output
