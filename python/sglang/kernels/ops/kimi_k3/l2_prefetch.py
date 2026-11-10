"""CUDA JIT L2 warm-up for K3 decode weights issued under communication waits."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_l2_prefetch_module() -> Module:
    return load_jit(
        "kimi_k3_l2_prefetch",
        cuda_files=["kimi_k3/l2_prefetch.cuh"],
        cuda_wrappers=[("run", "L2PrefetchKernel::run")],
        extra_cuda_cflags=["-O3"],
    )


def covered(tensor: torch.Tensor) -> bool:
    """Whether l2_prefetch can take the tensor: contiguous CUDA memory with
    a 16B-aligned start and size (cp.async.bulk granularity)."""
    return (
        tensor.is_cuda
        and tensor.is_contiguous()
        and tensor.data_ptr() % 16 == 0
        and (tensor.numel() * tensor.element_size()) % 16 == 0
    )


def l2_prefetch(tensor: torch.Tensor) -> None:
    """Issue cp.async.bulk.prefetch.L2 over the whole tensor and return; the
    data lands in L2 asynchronously. Caller checks covered()."""
    _jit_l2_prefetch_module().run(tensor.view(-1).view(torch.uint8))
