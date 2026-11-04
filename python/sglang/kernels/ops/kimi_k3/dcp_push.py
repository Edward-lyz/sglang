"""K3 DCP MLA decode peer-memory exchange (csrc/kimi_k3/comm/dcp_push.cuh).

Replaces ``dcp_fused_pack`` + ``all_to_all_single`` of the a2a DCP reduce with
one kernel that writes each rank's chunks straight into the peers' symmetric
receive buffers, in the same ``[N src][payload | fp32 lse]`` layout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_module() -> Module:
    return load_jit(
        "kimi_k3_dcp_push",
        cuda_files=["kimi_k3/comm/dcp_push.cuh"],
        cuda_wrappers=[("run", "DcpPushKernel::run")],
        extra_cuda_cflags=["-O3"],
    )


def dcp_push(
    out: torch.Tensor,
    lse: torch.Tensor,
    ptr_table: torch.Tensor,
    counter: torch.Tensor,
    chunk_bytes: int,
    lse_offset_bytes: int,
    rank: int,
    num_slices: int,
    block_size: int = 512,
) -> None:
    """Store this rank's per-peer chunks of out (+ fp32 lse) straight into the
    peers' symmetric receive buffers (ptr_table), in the pack + all_to_all
    layout, and wait until every peer's chunks for this rank have arrived."""
    _jit_module().run(
        out,
        lse,
        ptr_table,
        counter,
        chunk_bytes,
        lse_offset_bytes,
        rank,
        num_slices,
        block_size,
    )
