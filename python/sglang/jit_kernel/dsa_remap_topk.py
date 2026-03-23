"""JIT CUDA kernel: DSA remap topk indices to physical device addresses."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
    run_jit_kernel,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_dsa_remap_topk_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "dsa_remap_topk",
        *args,
        cuda_files=["nsa/dsa_remap_topk.cuh"],
        cuda_wrappers=[
            ("run", f"DSARemapTopk<{args}>::run"),
        ],
    )


def dsa_remap_topk(
    topk: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    gpu_slot: torch.Tensor,
    host_slot: torch.Tensor,
    req_to_token: torch.Tensor,
    remapped: torch.Tensor,
    page_size: int,
    max_context_len: int,
    host_slot_stride_0: int,
) -> None:
    """Remap topk: host-backed misses must already be loaded before this call."""
    run_jit_kernel(
        _jit_dsa_remap_topk_module,
        topk,
        req_pool_indices,
        seq_lens,
        gpu_slot,
        host_slot,
        req_to_token,
        remapped,
        page_size,
        max_context_len,
        gpu_slot.stride(0),
        req_to_token.stride(0),
        host_slot_stride_0,
    )
