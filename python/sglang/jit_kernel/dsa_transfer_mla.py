from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, run_jit_kernel

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_dsa_transfer_mla_module() -> Module:
    return load_jit(
        "dsa_transfer_mla",
        cuda_files=["nsa/dsa_transfer_mla.cuh"],
        cuda_wrappers=[
            ("run", "DSATransferMLAPages::run"),
        ],
    )


def dsa_transfer_mla_pages(
    *,
    src_ptrs: torch.Tensor,
    dst_ptrs: torch.Tensor,
    src_page_starts: torch.Tensor,
    dst_page_starts: torch.Tensor,
    active_pages: torch.Tensor,
    page_size: int,
    token_stride_bytes: int,
) -> None:
    run_jit_kernel(
        _jit_dsa_transfer_mla_module,
        src_ptrs,
        dst_ptrs,
        src_page_starts,
        dst_page_starts,
        active_pages,
        page_size,
        token_stride_bytes,
    )
