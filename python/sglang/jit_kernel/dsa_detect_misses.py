"""JIT CUDA kernel: DSA miss detection (read-only scan, CUDA graph safe)."""

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
def _jit_dsa_detect_misses_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "dsa_detect_misses",
        *args,
        cuda_files=["nsa/dsa_detect_misses.cuh"],
        cuda_wrappers=[
            ("run", f"DSADetectMisses<{args}>::run"),
        ],
    )


def dsa_detect_misses(
    topk: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    gpu_slot: torch.Tensor,
    host_slot: torch.Tensor,
    miss_req_indices: torch.Tensor,
    miss_logical_pages: torch.Tensor,
    miss_host_slots: torch.Tensor,
    miss_count: torch.Tensor,
    page_size: int,
    max_context_len: int,
    max_misses: int,
    gpu_slot_stride_0: int,
    host_slot_stride_0: int,
) -> None:
    """Detect cache misses. miss_count must be zeroed before call."""
    run_jit_kernel(
        _jit_dsa_detect_misses_module,
        topk,
        req_pool_indices,
        seq_lens,
        gpu_slot,
        host_slot,
        miss_req_indices,
        miss_logical_pages,
        miss_host_slots,
        miss_count,
        page_size,
        max_context_len,
        max_misses,
        gpu_slot_stride_0,
        host_slot_stride_0,
    )
