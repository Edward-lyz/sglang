"""Prefill-side GPU gather staging for the PD DCP token relayout
(SGLANG_K31_PD_DCP_GATHER).

Under PD DCP a decode rank owns every ``dcp_size``-th token, so on the source
side its tokens are strided: the MLA KV relayout and the striped DSA index-K
relayout move one token per RDMA descriptor, per layer -- millions of
descriptors (and the GIL-holding NumPy/pybind work to build them) for a long
prompt. On the destination the same tokens are consecutive local rows, so
gathering them into contiguous GPU staging first collapses the descriptors to
about one per destination page per layer. The same bytes land at the same
destination addresses; the peer needs no change.

The gather is a byte-row copy: for every entry (layer buffer) ``e`` and
requested row ``i`` it copies ``row_bytes`` from ``entry_ptr[e] +
src_offsets[i]`` to ``staging + (e * num_rows + i) * row_bytes``.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import numpy.typing as npt
import torch
import triton
import triton.language as tl

from sglang.srt.disaggregation.common.draft_kv_gather import _gather_stream_of
from sglang.srt.disaggregation.common.staging_buffer import StagingBuffer

_BLOCK_SIZE = 1024


@triton.jit
def _gather_rows_kernel(
    entry_ptrs,
    src_offsets,
    staging,
    per_entry_elems,
    num_blocks,
    ROW_ELEMS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    entry_id = tl.program_id(0)
    entry_ptr = tl.load(entry_ptrs + entry_id).to(staging.dtype)
    for block_id in range(tl.program_id(1), num_blocks, tl.num_programs(1)):
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < per_entry_elems
        row = offsets // ROW_ELEMS
        col = offsets % ROW_ELEMS
        src = tl.load(src_offsets + row, mask=mask, other=0) + col
        vals = tl.load(entry_ptr + src, mask=mask)
        dst = entry_id.to(tl.int64) * per_entry_elems.to(tl.int64) + offsets
        tl.store(staging + dst, vals, mask=mask)


def _elem_dtype(row_bytes: int, *offset_arrays: npt.NDArray[np.int64]):
    # Widest integer view that keeps every row and offset element-aligned.
    for size, dtype in ((4, torch.int32), (2, torch.int16)):
        if row_bytes % size == 0 and all(not (a % size).any() for a in offset_arrays):
            return size, dtype
    return 1, torch.int8


def gather_rows(
    *,
    entry_ptrs_gpu: torch.Tensor,
    src_offsets: npt.NDArray[np.int64],
    row_bytes: int,
    staging_buffer: StagingBuffer,
    gpu_id: int,
    max_ctas: int,
    staging_offset: int = 0,
) -> None:
    """Gather ``row_bytes`` at ``src_offsets`` (bytes from each entry's base)
    of every entry into staging (``[entry][row]`` dense). Runs on the staging
    slot's private stream and waits for it (see draft_kv_gather)."""
    num_entries = entry_ptrs_gpu.numel()
    num_rows = src_offsets.size
    elem, dtype = _elem_dtype(row_bytes, src_offsets, np.array([staging_offset]))
    row_elems = row_bytes // elem
    per_entry_elems = num_rows * row_elems
    total_bytes = num_entries * num_rows * row_bytes
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    staging = staging_buffer.buffer[staging_offset : staging_offset + total_bytes].view(
        dtype
    )
    num_blocks = triton.cdiv(per_entry_elems, _BLOCK_SIZE)
    ctas = num_blocks
    if max_ctas > 0:
        ctas = min(num_blocks, max(1, max_ctas // num_entries))
    stream = _gather_stream_of(staging_buffer, device)
    with torch.cuda.stream(stream):
        offsets = torch.from_numpy(src_offsets // elem).to(device, torch.int64)
        _gather_rows_kernel[(num_entries, ctas)](
            entry_ptrs_gpu,
            offsets,
            staging,
            per_entry_elems,
            num_blocks,
            row_elems,
            _BLOCK_SIZE,
        )
    stream.synchronize()


def entry_ptr_tensor(tensors: Sequence[torch.Tensor], gpu_id: int) -> torch.Tensor:
    return torch.tensor(
        [t.data_ptr() for t in tensors], dtype=torch.int64, device=f"cuda:{gpu_id}"
    )


def row_runs(
    dst_rows: npt.NDArray[np.int64], rows_per_page: int = 0
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """(start positions, lengths) of the maximal runs of consecutive
    destination rows; with ``rows_per_page`` a run also breaks at page
    boundaries (for paged layouts whose rows are not contiguous across
    pages)."""
    n = dst_rows.size
    if n == 0:
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty.copy()
    brk = np.ones(n, dtype=bool)
    brk[1:] = dst_rows[1:] != dst_rows[:-1] + 1
    if rows_per_page:
        brk |= dst_rows % rows_per_page == 0
    starts = np.flatnonzero(brk).astype(np.int64)
    lengths = np.diff(np.append(starts, n)).astype(np.int64)
    return starts, lengths


def gathered_descriptors(
    *,
    staging_ptr: int,
    staging_offset: int,
    num_rows: int,
    row_bytes: int,
    dst_entry_ptrs: Sequence[int],
    dst_run_offsets: npt.NDArray[np.int64],
    run_starts: npt.NDArray[np.int64],
    run_lengths: npt.NDArray[np.int64],
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """One descriptor per (entry, run): staging rows ``run_starts[k] ..
    +run_lengths[k]`` of entry ``e`` to ``dst_entry_ptrs[e] +
    dst_run_offsets[k]``."""
    entry_stride = num_rows * row_bytes
    src = np.concatenate(
        [
            staging_ptr + staging_offset + e * entry_stride + run_starts * row_bytes
            for e in range(len(dst_entry_ptrs))
        ]
    )
    dst = np.concatenate([ptr + dst_run_offsets for ptr in dst_entry_ptrs])
    lengths = np.tile(run_lengths * row_bytes, len(dst_entry_ptrs))
    return src, dst, lengths
