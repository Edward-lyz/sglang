"""Prefill-side GPU gather staging for the speculative draft KV transfer.

Under PD DCP the decode draft pool is head-sharded: each decode attn-TP rank
owns only *some* heads of every token. The prefill draft pool is token-major
(``[token][head][head_dim]``), so one destination's slice is strided on the
source side -- a ``num_heads * head_dim`` block every ``total_heads * head_dim``.
Expressed as a flat RDMA descriptor list that is one descriptor per token per
pool entry: a 24k-token chunk becomes ~24k descriptors of 128B, per entry, per
destination rank. Building those lists is pure Python + pybind work that holds
the GIL, which starves the scheduler thread of the PP stage owning the draft
pool (the last stage) and stalls the whole pipeline ring behind it.

This module gathers the head slice into contiguous GPU staging memory first, so
the descriptors collapse to one per contiguous destination run (~one per page).
Nothing changes on the wire: the same bytes land at the same destination
addresses, so the peer needs no matching change.
"""

from __future__ import annotations

import contextlib
import logging
import queue
from typing import Callable, List, Optional, Sequence, Tuple

import msgspec
import numpy as np
import numpy.typing as npt
import torch
import triton
import triton.language as tl

from sglang.srt.disaggregation.common.staging_buffer import StagingBuffer
from sglang.srt.disaggregation.common.utils import contiguous_run_starts_and_counts

logger = logging.getLogger(__name__)

# Staging slots are grown to a multiple of this so a slowly growing chunk size
# does not re-allocate (and re-register) on every transfer.
_BUFFER_GRANULARITY_BYTES = 8 * 1024 * 1024

# Bit-preserving integer view per pool element size: the gather only moves bytes,
# so we never want dtype conversion (fp8 in particular has no arithmetic here).
_INT_DTYPE_BY_SIZE = {1: torch.int8, 2: torch.int16, 4: torch.int32}

_BLOCK_SIZE = 1024


@triton.jit
def _gather_token_slices_kernel(
    entry_ptrs,
    token_indices,
    staging,
    stride_pool_token,
    head_offset,
    per_entry_elems,
    num_blocks,
    ELEMS_PER_TOKEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Gather ``token_indices``' head slice of every entry into ``staging``.

    Entry ``e``'s slice of the ``i``-th requested token lands at
    ``e * per_entry_elems + i * ELEMS_PER_TOKEN``, i.e. staging holds
    ``[entry][gathered_token][head][head_dim]`` densely packed. All offsets are
    in elements of the pool dtype (``staging`` is viewed as the same width).

    Grid-strided over the ``num_blocks`` blocks each entry needs, so the launch
    may use fewer programs than blocks: one CTA occupies one SM, so the grid size
    is what bounds how much of the GPU this steals from the concurrent forward
    (see ``gather_token_slices``' ``max_ctas``).
    """
    entry_id = tl.program_id(0)
    entry_ptr = tl.load(entry_ptrs + entry_id).to(staging.dtype)

    for block_id in range(tl.program_id(1), num_blocks, tl.num_programs(1)):
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < per_entry_elems

        t_idx = offsets // ELEMS_PER_TOKEN
        e_idx = offsets % ELEMS_PER_TOKEN
        pool_token = tl.load(token_indices + t_idx, mask=mask, other=0)

        src_offsets = (
            pool_token.to(tl.int64) * stride_pool_token.to(tl.int64)
            + head_offset.to(tl.int64)
            + e_idx
        )
        vals = tl.load(entry_ptr + src_offsets, mask=mask)

        dst_offsets = entry_id.to(tl.int64) * per_entry_elems.to(tl.int64) + offsets
        tl.store(staging + dst_offsets, vals, mask=mask)


class DraftGatherLayout(msgspec.Struct, frozen=True):
    """Element-unit geometry of one destination's draft head slice.

    Built only after the tensor-derived geometry has been checked against the
    byte-unit descriptor math of the caller, so a layout instance is a proof
    that gathering and then sending contiguously moves exactly the same bytes as
    the per-token descriptor list would have.
    """

    head_offset_elems: int
    elems_per_token: int
    stride_pool_token_elems: int
    dtype_size: int

    @property
    def bytes_per_token(self) -> int:
        """
        bytes_per_token
        """
        return self.elems_per_token * self.dtype_size


def build_draft_gather_layout(
    *,
    entry_tensors: Optional[List[torch.Tensor]],
    entry_ptrs: Sequence[int],
    head_start: int,
    num_heads: int,
    src_token_item_len: int,
    src_head_slice_offset: int,
    heads_bytes_per_token: int,
) -> Optional[DraftGatherLayout]:
    """Validate that ``entry_tensors`` describe the caller's byte geometry.

    Returns ``None`` (with a one-shot log) when anything disagrees, so the
    caller can fall back to the per-token descriptor path.
    """
    if not entry_tensors or len(entry_tensors) != len(entry_ptrs):
        logger.info_once(
            "PD DCP draft gather disabled: draft pool entry tensors unavailable "
            f"(tensors={0 if not entry_tensors else len(entry_tensors)} "
            f"ptrs={len(entry_ptrs)})"
        )
        return None

    first = entry_tensors[0]
    if first.dim() < 3:
        logger.info_once(
            f"PD DCP draft gather disabled: entry rank {first.dim()} is not "
            "[..., head, head_dim]"
        )
        return None

    head_dim = first.shape[-1]
    total_heads = first.shape[-2]
    dtype_size = first.element_size()
    reason = _layout_mismatch_reason(
        entry_tensors=entry_tensors,
        entry_ptrs=entry_ptrs,
        head_start=head_start,
        num_heads=num_heads,
        head_dim=head_dim,
        total_heads=total_heads,
        dtype_size=dtype_size,
        src_token_item_len=src_token_item_len,
        src_head_slice_offset=src_head_slice_offset,
        heads_bytes_per_token=heads_bytes_per_token,
    )
    if reason is not None:
        logger.info_once(f"PD DCP draft gather disabled: {reason}")
        return None

    return DraftGatherLayout(
        head_offset_elems=head_start * head_dim,
        elems_per_token=num_heads * head_dim,
        stride_pool_token_elems=total_heads * head_dim,
        dtype_size=dtype_size,
    )


def _layout_mismatch_reason(
    *,
    entry_tensors: List[torch.Tensor],
    entry_ptrs: Sequence[int],
    head_start: int,
    num_heads: int,
    head_dim: int,
    total_heads: int,
    dtype_size: int,
    src_token_item_len: int,
    src_head_slice_offset: int,
    heads_bytes_per_token: int,
) -> Optional[str]:
    """Return why the tensors can't back the byte geometry, or ``None`` if they can."""
    if total_heads * head_dim * dtype_size != src_token_item_len:
        return (
            f"token stride mismatch: tensor {total_heads}*{head_dim}*{dtype_size} "
            f"!= kv_item_len {src_token_item_len}"
        )
    if head_start * head_dim * dtype_size != src_head_slice_offset:
        return (
            f"head offset mismatch: tensor {head_start}*{head_dim}*{dtype_size} "
            f"!= {src_head_slice_offset}"
        )
    if num_heads * head_dim * dtype_size != heads_bytes_per_token:
        return (
            f"slice size mismatch: tensor {num_heads}*{head_dim}*{dtype_size} "
            f"!= {heads_bytes_per_token}"
        )
    if head_start + num_heads > total_heads:
        return (
            f"head range [{head_start}, {head_start + num_heads}) exceeds {total_heads}"
        )
    for idx, (entry, ptr) in enumerate(zip(entry_tensors, entry_ptrs)):
        if entry.data_ptr() != ptr:
            return f"entry {idx} pointer 0x{entry.data_ptr():x} != registered 0x{ptr:x}"
        if entry.shape[-1] != head_dim or entry.shape[-2] != total_heads:
            return f"entry {idx} shape {tuple(entry.shape)} differs from entry 0"
        if entry.element_size() != dtype_size or not entry.is_contiguous():
            return f"entry {idx} dtype/contiguity differs from entry 0"
    return None


class DraftKVGatherPool:
    """Fixed set of registered GPU staging slots, borrowed by transfer workers.

    Slots are sized lazily: the required bytes depend on the chunk length and on
    the peer's head count, neither of which is known at manager construction.
    A slot that is too small is re-allocated (and re-registered) once, rounded up
    to ``_BUFFER_GRANULARITY_BYTES``.
    """

    def __init__(
        self,
        *,
        gpu_id: int,
        num_slots: int,
        register: Callable[[int, int], None],
        deregister: Callable[[int], None],
        custom_mem_pool=None,
    ):
        self._gpu_id = gpu_id
        self._device = f"cuda:{gpu_id}"
        self._register = register
        self._deregister = deregister
        self._custom_mem_pool = custom_mem_pool
        self._slots: List[Optional[StagingBuffer]] = [None] * num_slots
        self._free: queue.Queue = queue.Queue()
        for slot in range(num_slots):
            self._free.put(slot)

    @contextlib.contextmanager
    def borrow(self, required_bytes: int):
        """Yield a registered staging buffer of at least ``required_bytes``."""
        slot = self._free.get()
        try:
            yield self._ensure_slot(slot, required_bytes)
        finally:
            self._free.put(slot)

    def _ensure_slot(self, slot: int, required_bytes: int) -> StagingBuffer:
        buffer = self._slots[slot]
        if buffer is not None and buffer.fits(required_bytes):
            return buffer
        if buffer is not None:
            self._slots[slot] = None
            self._deregister(buffer.get_ptr())
            del buffer
        size_bytes = (
            (required_bytes + _BUFFER_GRANULARITY_BYTES - 1)
            // _BUFFER_GRANULARITY_BYTES
            * _BUFFER_GRANULARITY_BYTES
        )
        buffer = StagingBuffer(
            size_bytes, self._device, self._gpu_id, self._custom_mem_pool
        )
        self._register(buffer.get_ptr(), buffer.get_size())
        logger.info(
            f"Draft KV gather slot {slot} sized to "
            f"{size_bytes / (1024 * 1024):.1f} MB (needed {required_bytes} B)"
        )
        self._slots[slot] = buffer
        return buffer


def staging_bytes_needed(
    *, layout: DraftGatherLayout, num_entries: int, num_tokens: int
) -> int:
    """
    staging_bytes_needed
    """
    return num_entries * num_tokens * layout.bytes_per_token


def build_entry_ptr_tensor(
    *, entry_tensors: Sequence[torch.Tensor], gpu_id: int
) -> torch.Tensor:
    """Device-side pointer table for the gather kernel.

    Built once per pool: a pool tensor's address is fixed for the process, and
    every transfer re-checks it against the registered pointer anyway (see
    ``build_draft_gather_layout``). Rebuilding it per transfer cost ~0.1 ms of
    GIL-holding list-to-tensor work -- the very thing this module exists to cut.
    """
    return torch.tensor(
        [entry.data_ptr() for entry in entry_tensors],
        dtype=torch.int64,
        device=f"cuda:{gpu_id}",
    )


def gather_token_slices(
    *,
    entry_ptrs_gpu: torch.Tensor,
    layout: DraftGatherLayout,
    src_token_indices: npt.NDArray[np.int64],
    staging_buffer: StagingBuffer,
    gpu_id: int,
    max_ctas: int,
) -> None:
    """Gather every entry's head slice for ``src_token_indices`` into staging.

    Everything -- index copy and kernel -- is enqueued on one private stream and
    then waited on, so the kernel needs no cross-stream ordering and the default
    stream (which the scheduler thread enqueues on) is never touched.

    Reading the pool requires the KV writes to be *already complete*, which the
    caller guarantees on the host side before queuing the transfer; the per-token
    RDMA fallback reads the same pool memory from this thread and has exactly the
    same requirement, so the gather adds none. ``cudaStreamSynchronize`` releases
    the GIL, so the wait costs the transfer thread only, not the scheduler.

    ``max_ctas`` caps the total number of CTAs launched (0 = one per block, i.e.
    uncapped). The kernel runs concurrently with the model forward on its own
    stream, so the grid is the only throttle available: N CTAs occupy at most N
    SMs, at the cost of a proportionally longer kernel.
    """
    num_entries = entry_ptrs_gpu.numel()
    num_tokens = src_token_indices.size
    per_entry_elems = num_tokens * layout.elems_per_token
    total_bytes = staging_bytes_needed(
        layout=layout, num_entries=num_entries, num_tokens=num_tokens
    )
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    int_dtype = _INT_DTYPE_BY_SIZE[layout.dtype_size]
    staging_typed = staging_buffer.buffer[:total_bytes].view(int_dtype)

    num_blocks = triton.cdiv(per_entry_elems, _BLOCK_SIZE)
    ctas_per_entry = num_blocks
    if max_ctas > 0:
        ctas_per_entry = min(num_blocks, max(1, max_ctas // num_entries))

    gather_stream = _gather_stream_of(staging_buffer, device)
    with torch.cuda.stream(gather_stream):
        token_idx_tensor = torch.from_numpy(src_token_indices).to(device, torch.int64)
        _gather_token_slices_kernel[(num_entries, ctas_per_entry)](
            entry_ptrs_gpu,
            token_idx_tensor,
            staging_typed,
            layout.stride_pool_token_elems,
            layout.head_offset_elems,
            per_entry_elems,
            num_blocks,
            layout.elems_per_token,
            _BLOCK_SIZE,
        )
    gather_stream.synchronize()


def _gather_stream_of(staging_buffer: StagingBuffer, device: str) -> torch.cuda.Stream:
    """One private stream per staging slot; slots are never shared concurrently."""
    stream = staging_buffer.draft_gather_stream
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        staging_buffer.draft_gather_stream = stream
    return stream


def build_gathered_descriptors(
    *,
    staging_ptr: int,
    dst_entry_ptrs: Sequence[int],
    dst_token_indices: npt.NDArray[np.int64],
    layout: DraftGatherLayout,
    dst_token_item_len: int,
    dst_head_slice_offset: int,
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Descriptors sending the gathered slices from staging to the peer's pool.

    Staging token ``i`` of entry ``e`` sits at a fixed stride, so a run of
    consecutive destination tokens is contiguous on both sides and collapses into
    a single descriptor -- typically one per destination page instead of one per
    token.
    """
    num_tokens = dst_token_indices.size
    src_positions = np.arange(num_tokens, dtype=np.int64)
    pos_starts, dst_starts, counts = contiguous_run_starts_and_counts(
        src_positions, dst_token_indices
    )
    bytes_per_token = layout.bytes_per_token
    run_src = pos_starts * bytes_per_token
    run_dst = dst_starts * dst_token_item_len + dst_head_slice_offset
    run_len = counts * bytes_per_token

    entry_stride = num_tokens * bytes_per_token
    src_addrs = np.concatenate(
        [
            staging_ptr + entry * entry_stride + run_src
            for entry in range(len(dst_entry_ptrs))
        ]
    )
    dst_addrs = np.concatenate([ptr + run_dst for ptr in dst_entry_ptrs])
    lengths = np.tile(run_len, len(dst_entry_ptrs))
    return src_addrs, dst_addrs, lengths
