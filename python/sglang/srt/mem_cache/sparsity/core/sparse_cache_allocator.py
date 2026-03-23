"""Persistent sparse cache allocator for DSA offload.

The sparse cache reuses the dense KV buffer, but it owns a fixed carve-out of
real physical pages reserved from the main allocator at startup.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


class SparseCacheAllocator:
    """Manages a fixed set of KV pages reserved for the DSA sparse cache.

    At init time, create_from_allocator() carves out a fraction of the main
    KV pool's free pages.  These pages are never returned to the main pool —
    they form the eviction-managed sparse cache that DSARuntime uses to hold
    hot prompt pages on device.  The main pool sees them as "reserved" and
    excludes them from its available count, which is why get_sparse_reserved_token_count()
    reports them separately to avoid false OOM alarms.
    """

    def __init__(
        self,
        reserved_page_starts: torch.Tensor,
        page_size: int,
        device: torch.device,
    ):
        self.page_size = page_size
        self.device = device
        reserved_page_starts = reserved_page_starts.to(device=device, dtype=torch.int64)
        assert reserved_page_starts.ndim == 1
        assert reserved_page_starts.numel() > 0, (
            "DSA sparse cache carve-out must reserve at least one page"
        )
        assert bool((reserved_page_starts >= page_size).all().item()), (
            "Reserved sparse pages must exclude the padded slot 0"
        )
        self.reserved_page_starts = reserved_page_starts.contiguous()
        self._capacity = int(reserved_page_starts.numel())

    @classmethod
    def create_from_allocator(
        cls,
        allocator,
        cache_ratio: float,
        page_size: int,
        device: torch.device,
    ) -> "SparseCacheAllocator":
        num_free_pages = len(allocator.free_pages) + len(allocator.release_pages)
        num_reserved_pages = max(int(num_free_pages * cache_ratio), 1)
        reserved_page_starts = allocator.reserve_free_pages(num_reserved_pages)
        if reserved_page_starts is None:
            raise RuntimeError(
                f"Failed to reserve sparse cache pages from the main allocator: {num_reserved_pages=}"
            )
        return cls(
            reserved_page_starts=reserved_page_starts,
            page_size=page_size,
            device=device,
        )

    @property
    def capacity(self) -> int:
        return self._capacity
