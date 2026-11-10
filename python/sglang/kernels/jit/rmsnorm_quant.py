"""Python wrapper for the rmsnorm quant kernel."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_rmsnorm_per_token_quant_module(
    quant_group_size: int,
    scale_ue8m0: bool,
) -> Module:
    args = make_cpp_args(quant_group_size, scale_ue8m0, is_arch_support_pdl())
    return load_jit(
        "rmsnorm_per_token_quant",
        *args,
        cuda_files=["rmsnorm_quant/fused_rmsnorm_per_token_quant.cuh"],
        cuda_wrappers=[("run", f"RMSNormPerTokenQuantKernel<{args}>::run")],
    )


@cache_once
def _jit_fused_add_rmsnorm_per_token_quant_module(
    quant_group_size: int,
    scale_ue8m0: bool,
) -> Module:
    args = make_cpp_args(quant_group_size, scale_ue8m0, is_arch_support_pdl())
    return load_jit(
        "fused_add_rmsnorm_per_token_quant",
        *args,
        cuda_files=["rmsnorm_quant/fused_rmsnorm_per_token_quant.cuh"],
        cuda_wrappers=[
            ("run", f"FusedAddRMSNormPerTokenQuantKernel<{args}>::run")
        ],
    )


def rmsnorm_per_token_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    norm_output: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    quant_group_size: int = 128,
    scale_ue8m0: bool = False,
    eps: float = 1e-5,
    min_8bit: float = -448.0,
    max_8bit: float = 448.0,
) -> None:
    """Per token quantization for RMSNorm."""
    module = _jit_rmsnorm_per_token_quant_module(quant_group_size, scale_ue8m0)
    module.run(
        input,
        weight,
        norm_output,
        output,
        output_scale,
        float(eps),
        float(min_8bit),
        float(max_8bit),
    )


def fused_add_rmsnorm_per_token_quant(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    norm_output: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    quant_group_size: int = 128,
    scale_ue8m0: bool = False,
    eps: float = 1e-5,
    min_8bit: float = -448.0,
    max_8bit: float = 448.0,
) -> None:
    """Per token quantization for fused add RMSNorm."""
    module = _jit_fused_add_rmsnorm_per_token_quant_module(
        quant_group_size, scale_ue8m0
    )
    module.run(
        input,
        residual,
        weight,
        norm_output,
        output,
        output_scale,
        float(eps),
        float(min_8bit),
        float(max_8bit),
    )
