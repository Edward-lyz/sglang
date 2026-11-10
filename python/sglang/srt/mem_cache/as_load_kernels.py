"""Triton kernels and utilities for AS load segment packing and unpacking."""

from collections import OrderedDict
import threading
from dataclasses import dataclass
from typing import List

import torch
import triton
import triton.language as tl

_METADATA_COLUMNS = 7
_DATA_PTR = 0
_LOCAL_ROWS_PTR = 1
_FULL_ROWS_PTR = 2
_ROW_STRIDE_BYTES = 3
_ROW_BYTES = 4
_LOCAL_ROW_COUNT = 5
_PACKED_OFFSET = 6
_DESCRIPTOR_COLUMNS = 3
_DESCRIPTOR_SEGMENT = 0
_DESCRIPTOR_ROW_OFFSET = 1
_DESCRIPTOR_BYTE_OFFSET = 2
_BLOCK_ROWS = 128
_BLOCK_BYTES = 512
_DESCRIPTOR_CACHE_MAX_ENTRIES = 128
_DESCRIPTOR_CACHE = OrderedDict()
_DESCRIPTOR_CACHE_LOCK = threading.Lock()
_WARMED_KERNELS = set()
_WARMUP_LOCK = threading.Lock()


@dataclass(frozen=True)
class AsLoadSegment:
    """Descriptor for a single AS load buffer segment."""

    buffer: torch.Tensor
    local_rows: torch.Tensor
    full_rows: torch.Tensor
    row_stride_bytes: int
    row_bytes: int
    local_row_count: int
    packed_offset: int

    @property
    def local_nbytes(self) -> int:
        """Return total byte size of the local rows for this segment."""
        return self.local_row_count * self.row_bytes


def make_as_load_segment(
    buffer: torch.Tensor,
    local_rows: torch.Tensor,
    full_rows: torch.Tensor,
    packed_offset: int,
) -> AsLoadSegment:
    """Validate inputs and construct an AsLoadSegment."""
    if buffer.ndim == 0:
        raise ValueError("AS load buffer must have a row dimension")
    if local_rows.dtype != torch.int64 or full_rows.dtype != torch.int64:
        raise ValueError("AS load row indices must use torch.int64")
    if not local_rows.is_contiguous() or not full_rows.is_contiguous():
        raise ValueError("AS load row indices must be contiguous")

    inner_elements = 1
    for dim in range(buffer.ndim - 1, 0, -1):
        if buffer.stride(dim) != inner_elements:
            raise ValueError("AS load buffer rows must be contiguous")
        inner_elements *= buffer.shape[dim]

    row_bytes = inner_elements * buffer.element_size()
    row_stride_bytes = buffer.stride(0) * buffer.element_size()
    if row_stride_bytes < row_bytes:
        raise ValueError("AS load buffer row stride is smaller than its row size")

    return AsLoadSegment(
        buffer=buffer,
        local_rows=local_rows,
        full_rows=full_rows,
        row_stride_bytes=row_stride_bytes,
        row_bytes=row_bytes,
        local_row_count=local_rows.numel(),
        packed_offset=packed_offset,
    )


def build_as_load_metadata(
    segments: List[AsLoadSegment], device: torch.device
) -> torch.Tensor:
    """Build a 2-D int64 metadata tensor from a list of AsLoadSegments."""
    rows = [
        [
            segment.buffer.data_ptr(),
            segment.local_rows.data_ptr(),
            segment.full_rows.data_ptr(),
            segment.row_stride_bytes,
            segment.row_bytes,
            segment.local_row_count,
            segment.packed_offset,
        ]
        for segment in segments
    ]
    return torch.tensor(rows, dtype=torch.int64, device=device)


def build_as_load_descriptors(
    segments: List[AsLoadSegment], device: torch.device
) -> torch.Tensor:
    """Return the shared immutable descriptor layout for these segment shapes."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    layout = tuple(
        (int(segment.local_row_count), int(segment.row_bytes)) for segment in segments
    )
    cache_key = (device.type, device.index, layout)

    with _DESCRIPTOR_CACHE_LOCK:
        descriptors = _DESCRIPTOR_CACHE.get(cache_key)
        if descriptors is None:
            rows = [
                (segment_id, row_offset, byte_offset)
                for segment_id, (local_row_count, row_bytes) in enumerate(layout)
                for row_offset in range(0, local_row_count, _BLOCK_ROWS)
                for byte_offset in range(0, row_bytes, _BLOCK_BYTES)
            ]
            descriptors = torch.tensor(rows, dtype=torch.int64, device=device).reshape(
                -1, _DESCRIPTOR_COLUMNS
            )
            _DESCRIPTOR_CACHE[cache_key] = descriptors
            if len(_DESCRIPTOR_CACHE) > _DESCRIPTOR_CACHE_MAX_ENTRIES:
                _DESCRIPTOR_CACHE.popitem(last=False)
        else:
            _DESCRIPTOR_CACHE.move_to_end(cache_key)

    if descriptors.is_cuda:
        descriptors.record_stream(torch.cuda.current_stream(descriptors.device))
    return descriptors


def _clear_as_load_descriptor_cache() -> None:
    with _DESCRIPTOR_CACHE_LOCK:
        _DESCRIPTOR_CACHE.clear()


def select_as_load_vector_bytes(segments: List[AsLoadSegment]) -> int:
    """Select the widest type that preserves alignment for every segment."""
    for width in (8, 4, 2):
        if all(
            segment.buffer.data_ptr() % width == 0
            and segment.row_stride_bytes % width == 0
            and segment.row_bytes % width == 0
            and segment.packed_offset % width == 0
            for segment in segments
        ):
            return width
    return 1


def make_inplace_allgather_pair(
    segment: AsLoadSegment,
    first_page: int,
    n_blocks: int,
    page_size: int,
    world_size: int,
    rank: int,
):
    """Return NCCL's in-place (recv, send) views for a contiguous segment."""
    full_row_count = segment.local_row_count * world_size
    if (
        segment.row_stride_bytes != segment.row_bytes
        or segment.full_rows.numel() != full_row_count
    ):
        return None
    if full_row_count == n_blocks:
        row_start = first_page
    elif full_row_count == n_blocks * page_size:
        row_start = first_page * page_size
    else:
        return None
    if row_start + full_row_count > segment.buffer.shape[0]:
        return None

    recv = (
        segment.buffer.narrow(0, row_start, full_row_count)
        .view(torch.uint8)
        .reshape(-1)
    )
    send_offset = rank * segment.local_nbytes
    send = recv.narrow(0, send_offset, segment.local_nbytes)
    expected_recv_ptr = segment.buffer.data_ptr() + row_start * segment.row_stride_bytes
    if (
        not recv.is_contiguous()
        or recv.data_ptr() != expected_recv_ptr
        or send.data_ptr() != recv.data_ptr() + send_offset
    ):
        return None
    return recv, send


def rank_major_packed_offset(
    rank: int,
    local_row: int,
    byte_offset: int,
    packed_rank_bytes: int,
    segment_offset: int,
    row_bytes: int,
) -> int:
    """Compute the packed buffer offset for a given rank and row position."""
    return (
        rank * packed_rank_bytes + segment_offset + local_row * row_bytes + byte_offset
    )


@triton.jit
def _pack_as_load_segments_kernel(
    metadata,
    descriptors,
    packed,
    VECTOR_BYTES: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    # Triton JIT rejects ordinary Python globals in kernels. Keep these layout
    # offsets literal and in sync with build_as_load_metadata/descriptors.
    descriptor_base = tl.program_id(0) * 3
    segment_id = tl.load(descriptors + descriptor_base)
    descriptor_row_offset = tl.load(descriptors + descriptor_base + 1)
    descriptor_byte_offset = tl.load(descriptors + descriptor_base + 2)
    metadata_base = segment_id * 7

    data_addr = tl.load(metadata + metadata_base)
    rows_addr = tl.load(metadata + metadata_base + 1)
    row_stride = tl.load(metadata + metadata_base + 3)
    row_bytes = tl.load(metadata + metadata_base + 4)
    row_count = tl.load(metadata + metadata_base + 5)
    packed_offset = tl.load(metadata + metadata_base + 6)

    row_positions = descriptor_row_offset + tl.arange(0, BLOCK_ROWS)
    byte_offsets = descriptor_byte_offset + (
        tl.arange(0, BLOCK_BYTES // VECTOR_BYTES) * VECTOR_BYTES
    )
    row_mask = row_positions < row_count
    byte_mask = byte_offsets < row_bytes
    rows = tl.cast(rows_addr, tl.pointer_type(tl.int64))
    row = tl.load(rows + row_positions, mask=row_mask, other=0)
    source = tl.cast(data_addr, tl.pointer_type(tl.uint8))
    mask = row_mask[:, None] & byte_mask[None, :]
    source = source + row[:, None] * row_stride + byte_offsets[None, :]
    destination = (
        packed
        + packed_offset
        + row_positions[:, None] * row_bytes
        + byte_offsets[None, :]
    )
    if VECTOR_BYTES == 8:
        values = tl.load(tl.cast(source, tl.pointer_type(tl.uint64)), mask=mask)
        tl.store(tl.cast(destination, tl.pointer_type(tl.uint64)), values, mask=mask)
    elif VECTOR_BYTES == 4:
        values = tl.load(tl.cast(source, tl.pointer_type(tl.uint32)), mask=mask)
        tl.store(tl.cast(destination, tl.pointer_type(tl.uint32)), values, mask=mask)
    elif VECTOR_BYTES == 2:
        values = tl.load(tl.cast(source, tl.pointer_type(tl.uint16)), mask=mask)
        tl.store(tl.cast(destination, tl.pointer_type(tl.uint16)), values, mask=mask)
    else:
        values = tl.load(source, mask=mask)
        tl.store(destination, values, mask=mask)


@triton.jit
def _unpack_as_load_segments_kernel(
    metadata,
    descriptors,
    gathered,
    packed_rank_bytes,
    world_size: tl.constexpr,
    VECTOR_BYTES: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    # See _pack_as_load_segments_kernel for the literal-layout requirement.
    descriptor_base = tl.program_id(0) * 3
    rank = tl.program_id(1)
    segment_id = tl.load(descriptors + descriptor_base)
    descriptor_row_offset = tl.load(descriptors + descriptor_base + 1)
    descriptor_byte_offset = tl.load(descriptors + descriptor_base + 2)
    metadata_base = segment_id * 7

    data_addr = tl.load(metadata + metadata_base)
    rows_addr = tl.load(metadata + metadata_base + 2)
    row_stride = tl.load(metadata + metadata_base + 3)
    row_bytes = tl.load(metadata + metadata_base + 4)
    local_row_count = tl.load(metadata + metadata_base + 5)
    packed_offset = tl.load(metadata + metadata_base + 6)

    local_row_positions = descriptor_row_offset + tl.arange(0, BLOCK_ROWS)
    byte_offsets = descriptor_byte_offset + (
        tl.arange(0, BLOCK_BYTES // VECTOR_BYTES) * VECTOR_BYTES
    )
    row_mask = local_row_positions < local_row_count
    byte_mask = byte_offsets < row_bytes
    full_row_positions = rank * local_row_count + local_row_positions
    # Rank-major offsets can exceed int32 even when one rank's slice does not.
    rank_offset = tl.cast(rank, tl.int64) * tl.cast(packed_rank_bytes, tl.int64)
    source_offset = (
        rank_offset
        + packed_offset
        + local_row_positions[:, None] * row_bytes
        + byte_offsets[None, :]
    )

    rows = tl.cast(rows_addr, tl.pointer_type(tl.int64))
    row = tl.load(rows + full_row_positions, mask=row_mask, other=0)
    mask = row_mask[:, None] & byte_mask[None, :]
    source = gathered + source_offset
    destination = tl.cast(data_addr, tl.pointer_type(tl.uint8))
    destination = destination + row[:, None] * row_stride + byte_offsets[None, :]
    if VECTOR_BYTES == 8:
        values = tl.load(tl.cast(source, tl.pointer_type(tl.uint64)), mask=mask)
        tl.store(tl.cast(destination, tl.pointer_type(tl.uint64)), values, mask=mask)
    elif VECTOR_BYTES == 4:
        values = tl.load(tl.cast(source, tl.pointer_type(tl.uint32)), mask=mask)
        tl.store(tl.cast(destination, tl.pointer_type(tl.uint32)), values, mask=mask)
    elif VECTOR_BYTES == 2:
        values = tl.load(tl.cast(source, tl.pointer_type(tl.uint16)), mask=mask)
        tl.store(tl.cast(destination, tl.pointer_type(tl.uint16)), values, mask=mask)
    else:
        values = tl.load(source, mask=mask)
        tl.store(destination, values, mask=mask)


def pack_as_load_segments(
    metadata: torch.Tensor,
    descriptors: torch.Tensor,
    packed: torch.Tensor,
    vector_bytes: int,
) -> None:
    """Launch the pack kernel to gather local rows into packed buffer."""
    grid = (descriptors.shape[0],)
    _pack_as_load_segments_kernel[grid](
        metadata,
        descriptors,
        packed,
        VECTOR_BYTES=vector_bytes,
        BLOCK_ROWS=_BLOCK_ROWS,
        BLOCK_BYTES=_BLOCK_BYTES,
        num_warps=8,
        num_stages=2,
    )


def unpack_as_load_segments(
    metadata: torch.Tensor,
    descriptors: torch.Tensor,
    gathered: torch.Tensor,
    packed_rank_bytes: int,
    world_size: int,
    vector_bytes: int,
) -> None:
    """Launch the unpack kernel to scatter gathered rows back to local buffers."""
    grid = (descriptors.shape[0], world_size)
    _unpack_as_load_segments_kernel[grid](
        metadata,
        descriptors,
        gathered,
        packed_rank_bytes,
        world_size=world_size,
        VECTOR_BYTES=vector_bytes,
        BLOCK_ROWS=_BLOCK_ROWS,
        BLOCK_BYTES=_BLOCK_BYTES,
        num_warps=8,
        num_stages=2,
    )


def warmup_as_load_kernels(
    metadata: torch.Tensor,
    descriptors: torch.Tensor,
    local: torch.Tensor,
    gathered: torch.Tensor,
    packed_rank_bytes: int,
    world_size: int,
    vector_bytes: int,
    *,
    allow_module_load: bool = True,
) -> None:
    """Compile and load the exact pack/unpack variants before concurrent use."""
    with _WARMUP_LOCK:
        pack_grid = (descriptors.shape[0],)
        pack_kernel = _pack_as_load_segments_kernel.warmup(
            metadata,
            descriptors,
            local,
            VECTOR_BYTES=vector_bytes,
            BLOCK_ROWS=_BLOCK_ROWS,
            BLOCK_BYTES=_BLOCK_BYTES,
            num_warps=8,
            num_stages=2,
            grid=pack_grid,
        )
        unpack_grid = (descriptors.shape[0], world_size)
        unpack_kernel = _unpack_as_load_segments_kernel.warmup(
            metadata,
            descriptors,
            gathered,
            packed_rank_bytes,
            world_size=world_size,
            VECTOR_BYTES=vector_bytes,
            BLOCK_ROWS=_BLOCK_ROWS,
            BLOCK_BYTES=_BLOCK_BYTES,
            num_warps=8,
            num_stages=2,
            grid=unpack_grid,
        )

        # 中文：.warmup() 只编译，CUDA module/function 仍可能尚未加载。
        # 同步编译结果也覆盖 Triton 的异步编译模式；这里不执行 kernel，
        # 所以启动阶段的全零 metadata 不会被当成真实 KV 指针解引用。
        kernels = tuple(
            kernel.result() if hasattr(kernel, "result") else kernel
            for kernel in (pack_kernel, unpack_kernel)
        )
        # 中文：metadata 的奇数行切片只有 8 字节对齐，可能对应另一份二进制。
        # 用实际编译产物的 hash 区分专门化，不能仅按 world/vector_bytes 判定已就绪。
        cache_key = (
            metadata.device.type,
            metadata.device.index,
            *(kernel.hash for kernel in kernels),
        )
        if cache_key in _WARMED_KERNELS:
            return
        if not allow_module_load:
            # 中文：forward 已在途时禁止补做模块加载；交给现有 AS 失败归约
            # 和请求回退处理，避免 cuModuleLoadData 再次把整个 TP 组卡死。
            raise RuntimeError(
                "AS pack/unpack variant was not preloaded before threaded forward "
                f"(world_size={world_size}, vector_bytes={vector_bytes})"
            )
        for kernel in kernels:
            # 已核对配套 Triton：_init_handles 只加载模块和取得函数句柄，不发射 kernel。
            kernel._init_handles()
        # 只有两份模块都加载成功，才允许后续请求跳过初始化。
        _WARMED_KERNELS.add(cache_key)
