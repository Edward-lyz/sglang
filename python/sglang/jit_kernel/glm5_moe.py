"""Apply the MegaMoE tail to the output of the model."""
from __future__ import annotations

from typing import Optional

import torch

from sglang.jit_kernel.utils import cache_once, load_jit


@cache_once
def _jit_mega_moe_tail_module():
    return load_jit(
        "glm5_mega_moe_tail",
        cuda_files=["glm5/mega_moe_tail.cuh"],
        cuda_wrappers=[("run", "MegaMoETailKernel::run")],
        extra_cuda_cflags=["-use_fast_math"],
    )


def mega_moe_tail(
    y: torch.Tensor,
    shared_output: Optional[torch.Tensor],
    routed_scaling_factor: float,
) -> torch.Tensor:
    """Apply the MegaMoE tail to the output of the model."""
    assert y.dtype == torch.bfloat16, "y must be bfloat16"
    assert y.is_cuda and y.is_contiguous()
    assert y.dim() == 2
    if shared_output is not None:
        assert shared_output.dtype == torch.bfloat16
        assert shared_output.is_cuda and shared_output.is_contiguous()
        assert shared_output.shape == y.shape

    if y.numel() == 0 or (
        shared_output is None and float(routed_scaling_factor) == 1.0
    ):
        return y

    module = _jit_mega_moe_tail_module()
    module.run(y, shared_output, float(routed_scaling_factor))
    return y
