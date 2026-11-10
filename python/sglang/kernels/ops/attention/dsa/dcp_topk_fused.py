"""One-kernel global DSA top-k over a DCP-striped index-K
(SGLANG_K31_INDEXER_DCP, csrc/dsa/dcp_topk.cuh).

Same contract as :func:`dcp_topk.dcp_global_topk`, but exact and in one
kernel: a radix select whose per-round histograms are exchanged per CTA -- all
of the CTA's rows in one batch -- as self-flagged LL words in the peers'
symmetric receive buffers; once a row's group-wide boundary bucket holds
<= 1024 keys the ranks exchange those keys and rank them locally (fewer
rounds); > 1024 exactly equal keys fall back to a count split by rank.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

TOPK = 2048
_BINS = 256
_ROUNDS = 4  # radix slots (kRadixRounds)
_ROWS_PER_CTA = 4  # kMaxRowsPerCta in csrc/dsa/dcp_topk.cuh
_GATHER_WORDS = 1 + 1024  # kGatherWords: count + up to kGatherTotal keys
_STATE: dict = {}


@cache_once
def _jit_module() -> Module:
    return load_jit(
        "dsa_dcp_topk",
        cuda_files=["dsa/dcp_topk.cuh"],
        cuda_wrappers=[("run", "DcpTopkKernel::run")],
        extra_cuda_cflags=["-O3"],
    )


def _buffers(group, device: torch.device, max_ctas: int) -> dict:
    """Symmetric LL receive buffer (8 B words = 32 data bits + epoch flag):
    radix slots [2 parity][4][src][max_ctas][rows per CTA][bins / 2], then the
    gather slot [2][src][max_ctas][rows per CTA][1 + 1024]; rendezvoused once
    per group, plus the local per-CTA epochs. Allocation is collective, so the
    first call must be eager (outside graph capture)."""
    key = (group.unique_name, device.index, max_ctas)
    state = _STATE.get(key)
    if state is not None:
        return state
    assert not torch.cuda.is_current_stream_capturing(), (
        "dcp_topk buffers must be allocated in the eager warmup"
    )
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm

    n = group.world_size
    words = 2 * n * max_ctas * _ROWS_PER_CTA * (_ROUNDS * _BINS // 2 + _GATHER_WORDS)
    recv = symm.empty(2 * words, dtype=torch.int32, device=device)  # 8 B per word
    recv.zero_()
    name = group.cpu_group.group_name
    h_recv = symm.rendezvous(recv, name)
    table = torch.tensor([int(p) for p in h_recv.buffer_ptrs], dtype=torch.int64)
    epochs = torch.zeros(max_ctas, dtype=torch.int32, device=device)
    torch.cuda.synchronize(device)
    dist.barrier(group=group.cpu_group)
    state = dict(recv=recv, table=table, epochs=epochs)
    _STATE[key] = state
    return state


def max_fused_rows(device: torch.device) -> int:
    """Largest row count one call takes (rows per CTA x CTAs)."""
    return _ROWS_PER_CTA * torch.cuda.get_device_properties(device).multi_processor_count


def dcp_global_topk_fused(
    logits: torch.Tensor,
    raw_indices: torch.Tensor,
    local_rows: torch.Tensor,
    lengths: torch.Tensor,
    group,
    dcp_rank: int,
    num_ctas: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the group-wide top-2048 from each rank's local top-2048.

    Returns (widened slots ``local_row * dcp + rank`` packed to the front with
    a -1 tail, per-row selected count). Every rank must call with the same row
    count and ``num_ctas``; rows <= :func:`max_fused_rows`.
    """
    rows = raw_indices.shape[0]
    assert raw_indices.shape == local_rows.shape == (rows, TOPK)
    device = raw_indices.device
    out = torch.empty((rows, TOPK), dtype=torch.int32, device=device)
    out_counts = torch.empty((rows,), dtype=torch.int32, device=device)
    if rows == 0:
        return out, out_counts
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    num_ctas = sms if num_ctas <= 0 else min(num_ctas, sms)
    state = _buffers(group, device, sms)
    _jit_module().run(
        logits,
        raw_indices.contiguous(),
        local_rows.contiguous(),
        lengths.to(torch.int32).contiguous(),
        out,
        out_counts,
        state["table"],
        state["epochs"],
        dcp_rank,
        num_ctas,
    )
    return out, out_counts
