"""Per-token-group quantization to FP4 E2M1 via a JIT-compiled CUDA kernel."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_per_token_group_quant_fp4_module() -> Module:
    return load_jit(
        "per_token_group_quant_fp4",
        cuda_files=["gemm/per_token_group_quant_fp4.cuh"],
        cuda_wrappers=[
            ("per_token_group_quant_fp4", "per_token_group_quant_fp4<bf16_t>")
        ],
        extra_cuda_cflags=["--use_fast_math"],
    )


@register_custom_op(
    op_name="per_token_group_quant_fp4",
    mutates_args=["output_packed", "output_sf"],
)
def per_token_group_quant_fp4(
    input: torch.Tensor,
    output_packed: torch.Tensor,
    output_sf: torch.Tensor,
) -> None:
    """
    Per-token-group quantization to FP4 E2M1 format (CUDA JIT kernel).

    Matches ``deep_gemm.utils.per_token_cast_to_fp4`` with ``use_ue8m0=True``,
    ``use_packed_ue8m0=True`` and ``gran_k=32``.

    Args:
        input: [M, 128] bfloat16 tensor.
        output_packed: [M, 64] int8 tensor (two FP4 nibbles packed per byte).
        output_sf: [M, 1] int32 tensor (4 UE8M0 bytes packed into one int32).
    """
    module = _jit_per_token_group_quant_fp4_module()
    module.per_token_group_quant_fp4(input, output_packed, output_sf)


def per_token_cast_to_fp4_cuda(
    x: torch.Tensor,
    gran_k: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a ``[M, 128]`` bf16 tensor to packed FP4 with UE8M0 scales.

    Args:
        x: [M, 128] bfloat16, contiguous.
        gran_k: quantization group size along the last dim; only 32 is supported.

    Returns:
        packed: [M, 64] int8, packed FP4 E2M1.
        sf: [M, 1] int32, 4 UE8M0 bytes packed into one int32.
    """
    assert x.dtype == torch.bfloat16, f"expected bfloat16, got {x.dtype}"
    assert x.is_contiguous(), "input must be contiguous"
    M, N = x.shape
    assert N == 128 and gran_k == 32, f"unsupported shape/gran_k: {N=} {gran_k=}"

    packed = torch.empty((M, N // 2), dtype=torch.int8, device=x.device)
    sf = torch.empty((M, 1), dtype=torch.int32, device=x.device)
    if M > 0:
        per_token_group_quant_fp4(x, packed, sf)
    return packed, sf
