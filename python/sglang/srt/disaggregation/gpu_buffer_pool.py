"""gpu_buffer_pool.py -- Variable-size GPU buffer pool for RDMA receive buffers.

Pre-allocates a contiguous GPU/CPU buffer and manages it as fixed-size pages.
Each allocation returns a contiguous region (pointer), suitable for RDMA writes
where the destination must be a single contiguous address range.

Anti-fragmentation design:
  - Free blocks are tracked as an ordered list of (start, length) runs.
  - On free(), adjacent blocks are immediately coalesced (merged).
  - Allocation uses best-fit: picks the smallest block that satisfies the
    request, minimising leftover fragments.

Thread-safe: multiple threads can allocate/free concurrently.
When the pool is full or too fragmented, allocate() blocks until memory
is freed (with optional timeout).
"""

from __future__ import annotations

import math
import threading
import time
from typing import Optional

import torch

Pointer = int
PageIndex = int
PageCount = int
FreeBlock = tuple[PageIndex, PageCount]


class GpuBufferPool:
    """Page-based contiguous memory pool with coalescing and best-fit.

    Parameters
    ----------
    total_bytes : int
        Total pool size in bytes. Rounded down to a multiple of page_size.
    page_size : int
        Size of each page in bytes (default 2 MB).
    device : str
        Torch device for the backing buffer (e.g. "cuda:0" or "cpu").
    pin_memory : bool
        If True and device is "cpu", allocate page-locked (pinned) memory
        for faster DMA / RDMA transfers.
    """

    def __init__(
        self,
        total_bytes: int,
        page_size: int = 2 * 1024 * 1024,
        device: str = "cuda:0",
        pin_memory: bool = False,
        zero: bool = True,
    ):
        """Allocate the backing buffer, rounded down to a whole number of pages.

        ``zero=False`` skips the memset: for a large pool whose every region is
        fully overwritten before it is handed out, zeroing only costs startup
        time and first-touch memory.
        """
        if page_size <= 0:
            raise ValueError(f"page_size must be > 0, got {page_size}")
        if total_bytes < page_size:
            raise ValueError(
                f"total_bytes ({total_bytes}) must be >= page_size ({page_size})")

        self._page_size = page_size
        self._num_pages = total_bytes // page_size
        self._actual_bytes = self._num_pages * page_size
        self._device = device

        # Backing buffer
        allocate = torch.zeros if zero else torch.empty
        if pin_memory and str(device) == "cpu":
            self._buf = allocate(self._actual_bytes, dtype=torch.uint8,
                                 device="cpu", pin_memory=True)
        else:
            self._buf = allocate(self._actual_bytes, dtype=torch.uint8,
                                 device=device)
        self._base_ptr: Pointer = self._buf.data_ptr()

        # Free list: list of [start_page, num_pages], sorted by start_page.
        # Invariant: blocks never overlap or touch — adjacent blocks are always
        # coalesced on free().
        self._free_blocks: list[FreeBlock] = [(0, self._num_pages)]

        # Maps start_page -> num_pages for each live allocation.
        self._alloc_map: dict[PageIndex, PageCount] = {}

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    # -- properties ----------------------------------------------------------

    @property
    def data_ptr(self) -> Pointer:
        """Base pointer of the backing buffer."""
        return self._base_ptr

    @property
    def total_bytes(self) -> int:
        """Usable pool size (may be slightly less than requested due to rounding)."""
        return self._actual_bytes

    @property
    def page_size(self) -> int:
        """Page size in bytes."""
        return self._page_size

    @property
    def num_pages(self) -> int:
        """Total number of pages in the pool."""
        return self._num_pages

    def available_bytes(self) -> int:
        """Total free bytes (may be fragmented across multiple blocks)."""
        with self._lock:
            return self._free_pages_locked() * self._page_size

    def used_bytes(self) -> int:
        """Total used bytes in the pool."""
        return self.total_bytes - self.available_bytes()

    def num_allocations(self) -> int:
        """Number of live allocations."""
        with self._lock:
            return len(self._alloc_map)

    def largest_free_block_bytes(self) -> int:
        """Size of the largest contiguous free region (in bytes).

        This is the maximum single allocation that can succeed immediately.
        Useful for monitoring fragmentation: if available_bytes is large but
        largest_free_block_bytes is small, the pool is fragmented.
        """
        with self._lock:
            if not self._free_blocks:
                return 0
            return max(n for _, n in self._free_blocks) * self._page_size

    def fragmentation_ratio(self) -> float:
        """Fragmentation metric in [0, 1].

        0 = no fragmentation (all free memory in one block).
        1 = maximum fragmentation (many tiny blocks).
        Defined as 1 - largest_free / total_free. Returns 0 when pool is full.
        """
        with self._lock:
            total_free = self._free_pages_locked()
            if total_free == 0:
                return 0.0
            largest = max(n for _, n in self._free_blocks)
            return 1.0 - largest / total_free

    # -- allocate / free -----------------------------------------------------

    def try_allocate(self, nbytes: int) -> Optional[Pointer]:
        """Try to allocate a contiguous region of at least *nbytes* bytes.

        Unlike allocate(), this never blocks. Returns the pointer (int) on
        success, or ``None`` immediately if the pool is too fragmented or full.

        Raises ``ValueError`` if *nbytes* <= 0 or exceeds total pool size.
        """
        if nbytes <= 0:
            raise ValueError(f"nbytes must be > 0, got {nbytes}")

        need_pages = math.ceil(nbytes / self._page_size)
        if need_pages > self._num_pages:
            raise ValueError(
                f"requested {nbytes} bytes ({need_pages} pages) exceeds "
                f"pool capacity ({self._num_pages} pages, "
                f"{self._actual_bytes} bytes)")

        with self._cond:
            block_idx = self._best_fit(need_pages)
            if block_idx is None:
                return None
            start, size = self._free_blocks[block_idx]
            self._alloc_map[start] = need_pages
            if size == need_pages:  
                self._free_blocks.pop(block_idx)
            else:   # 分配的pages大于需要的pages，需要将剩余的pages插入到free_blocks中
                self._free_blocks[block_idx] = (
                    start + need_pages, size - need_pages)
            return self._base_ptr + start * self._page_size

    def allocate(self, nbytes: int, timeout: Optional[float] = None) -> Pointer:
        """Allocate a contiguous region of at least *nbytes* bytes.

        Uses best-fit: picks the smallest free block that satisfies the
        request, reducing leftover fragments.

        Returns the pointer (int) to the start of the allocated region.
        Blocks if the pool is too fragmented or full; raises ``TimeoutError``
        if *timeout* seconds elapse before a suitable region is found.
        Raises ``ValueError`` immediately if *nbytes* exceeds total pool size.
        """
        if nbytes <= 0:
            raise ValueError(f"nbytes must be > 0, got {nbytes}")

        need_pages = math.ceil(nbytes / self._page_size)
        if need_pages > self._num_pages:
            raise ValueError(
                f"requested {nbytes} bytes ({need_pages} pages) exceeds "
                f"pool capacity ({self._num_pages} pages, "
                f"{self._actual_bytes} bytes)")

        deadline = (time.monotonic() + timeout) if timeout is not None else None

        with self._cond:
            while True:
                block_idx = self._best_fit(need_pages)
                if block_idx is not None:
                    start, size = self._free_blocks[block_idx]
                    # Carve from the front of this block
                    self._alloc_map[start] = need_pages
                    if size == need_pages:
                        self._free_blocks.pop(block_idx)
                    else:
                        self._free_blocks[block_idx] = (
                            start + need_pages, size - need_pages)
                    return self._base_ptr + start * self._page_size

                if deadline is None:
                    self._cond.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        free_pages = self._free_pages_locked()
                        raise TimeoutError(
                            f"allocation of {nbytes} bytes ({need_pages} pages) "
                            f"timed out; free={free_pages}/{self._num_pages} pages")
                    self._cond.wait(timeout=remaining)

    def free(self, ptr: Pointer) -> None:
        """Free a previously allocated region by its pointer.

        Adjacent free blocks are immediately coalesced to prevent
        fragmentation buildup.
        """
        page_idx = (ptr - self._base_ptr) // self._page_size

        with self._cond:
            if page_idx not in self._alloc_map:
                raise ValueError(
                    f"pointer 0x{ptr:x} (page {page_idx}) "
                    f"is not a tracked allocation")
            num_pages = self._alloc_map.pop(page_idx)
            self._insert_and_coalesce(page_idx, num_pages)
            self._cond.notify_all()

    # -- internal ------------------------------------------------------------

    def _free_pages_locked(self) -> PageCount:
        """Total free pages. Caller must hold the lock."""
        return sum(n for _, n in self._free_blocks)

    def _best_fit(self, need: PageCount) -> Optional[int]:
        """Find the smallest free block with size >= *need*. Caller holds lock."""
        best_idx = None
        best_size = self._num_pages + 1  # sentinel larger than any block
        for i, (_, size) in enumerate(self._free_blocks):
            if need <= size < best_size:
                best_idx = i
                best_size = size
                if size == need:
                    break  # perfect fit, no need to keep searching
        return best_idx

    def _insert_and_coalesce(self, start: PageIndex, length: PageCount) -> None:
        """Insert [start, length] into the free list and merge neighbours.

        Maintains the sorted order and the no-touching invariant.
        Caller must hold the lock.
        """
        blocks = self._free_blocks

        # Binary search for insertion point (sorted by start_page)
        lo, hi = 0, len(blocks)
        while lo < hi:
            mid = (lo + hi) // 2
            if blocks[mid][0] < start:
                lo = mid + 1
            else:
                hi = mid
        pos = lo

        # Check adjacency with neighbours
        merge_prev = (pos > 0
                      and blocks[pos - 1][0] + blocks[pos - 1][1] == start)
        merge_next = (pos < len(blocks)
                      and start + length == blocks[pos][0])

        if merge_prev and merge_next:
            # Bridge: prev + new + next -> single block
            blocks[pos - 1] = (blocks[pos - 1][0],
                               blocks[pos - 1][1] + length + blocks[pos][1])
            blocks.pop(pos)
        elif merge_prev:
            blocks[pos - 1] = (blocks[pos - 1][0],
                               blocks[pos - 1][1] + length)
        elif merge_next:
            blocks[pos] = (start, length + blocks[pos][1])
        else:
            blocks.insert(pos, (start, length))

    def __repr__(self) -> str:
        """Occupancy summary; useful when a pull is waiting on a full pool."""
        with self._lock:
            free = self._free_pages_locked()
            n_blocks = len(self._free_blocks)
            largest = max((n for _, n in self._free_blocks), default=0)
        return (
            f"GpuBufferPool(total={self._actual_bytes}, "
            f"pages={self._num_pages}, free_pages={free}, "
            f"free_blocks={n_blocks}, largest_free={largest}, "
            f"page_size={self._page_size}, device={self._device!r})"
        )
