"""CUDA JIT K3 MLA output gate: out = x * sigmoid(gate) in one kernel."""

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

_THREADS: int = 256


@cache_once
def _jit_mla_output_gate_module() -> Module:
    args = make_cpp_args(_THREADS, is_arch_support_pdl())
    return load_jit(
        "kimi_k3_mla_output_gate_" + str(_THREADS),
        *args,
        cuda_files=["kimi_k3/mla_output_gate.cuh"],
        cuda_wrappers=[
            ("run", f"MlaOutputGateKernel<{args}>::run"),
            ("run_quant", f"MlaOutputGateKernel<{args}>::run_quant"),
        ],
        extra_cuda_cflags=["-O3"],
    )


def covered(x: torch.Tensor, gate: torch.Tensor) -> bool:
    return (
        x.dtype == torch.bfloat16
        and gate.dtype == torch.bfloat16
        and x.shape == gate.shape
        and x.is_contiguous()
        and gate.is_contiguous()
        and x.numel() % 8 == 0
        and x.numel() > 0
    )


def kimi_k3_mla_output_gate(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """out = bf16(x * bf16(sigmoid(gate))); double rounding matches the
    unfused torch.sigmoid + mul pair bit-for-bit. Caller checks covered()."""
    out = torch.empty_like(x)
    _jit_mla_output_gate_module().run(x.view(-1), gate.view(-1), out.view(-1))
    return out


def quant_covered(x: torch.Tensor) -> bool:
    """kimi_k3_mla_output_gate_quant also needs [T, H] with H % 512 == 0."""
    return x.dim() == 2 and x.shape[1] % 512 == 0


def kimi_k3_mla_output_gate_quant(
    x: torch.Tensor, gate: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """kimi_k3_mla_output_gate plus the DeepGEMM block-FP8 input quant of its
    output from the same kernel: returns ``(out, q, scale)`` with q / scale
    bit-identical to ``sglang_per_token_group_quant_fp8(out, 128,
    column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=True)``.
    Caller checks covered() and quant_covered()."""
    from sglang.kernels.ops.quantization.fp8_kernel import (
        create_per_token_group_quant_fp8_output_scale,
    )

    out = torch.empty_like(x)
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = create_per_token_group_quant_fp8_output_scale(
        x_shape=x.shape,
        device=x.device,
        group_size=128,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )
    _jit_mla_output_gate_module().run_quant(
        x.view(-1), gate.view(-1), out.view(-1), q, scale
    )
    return out, q, scale
