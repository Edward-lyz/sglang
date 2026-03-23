"""Simple page table for DSA offload.

Replaces the 4-state DeepSeekDSAState with a direct mapping:
  gpu_slot[req, page] >= 0  → page is in GPU sparse cache at this slot
  gpu_slot[req, page] == -1 → page is not in GPU cache (on host only)
"""

from __future__ import annotations

from typing import Optional

import torch


class PageTable:
    """Track which logical pages are cached on GPU vs host-only."""

    def __init__(
        self,
        max_pool_size: int,
        max_pages_per_req: int,
        device: torch.device,
    ):
        self.device = device
        self.max_pages_per_req = max_pages_per_req

        # gpu_slot[req, page] = cache slot (token-level page-start) or -1
        self.gpu_slot = torch.full(
            (max_pool_size, max_pages_per_req), -1,
            dtype=torch.int64, device=device,
        )
        # host_slot[req, page] = host pool slot (token-level page-start) or -1
        self.host_slot = torch.full(
            (max_pool_size, max_pages_per_req), -1,
            dtype=torch.int64, device=device,
        )

    def register(self, req_pool_idx: int) -> None:
        """Initialize page table for a new request."""
        self.gpu_slot[req_pool_idx].fill_(-1)
        self.host_slot[req_pool_idx].fill_(-1)

    def set_host_slots(
        self,
        req_pool_idx: int,
        logical_pages: torch.Tensor,
        host_page_starts: torch.Tensor,
    ) -> None:
        """Record host slot mappings after D2H canonicalization."""
        logical_pages = logical_pages.to(device=self.device, dtype=torch.int64)
        self.host_slot[req_pool_idx, logical_pages] = host_page_starts.to(
            device=self.device, dtype=torch.int64
        )

    def clear(self, req_pool_idx: int) -> None:
        """Release all state for a finished request."""
        self.gpu_slot[req_pool_idx].fill_(-1)
        self.host_slot[req_pool_idx].fill_(-1)

    def get_cached_gpu_slots(self, req_pool_idx: int) -> Optional[torch.Tensor]:
        """Return all gpu_slot values that are >= 0 for one request."""
        mask = self.gpu_slot[req_pool_idx] >= 0
        if not mask.any():
            return None
        return self.gpu_slot[req_pool_idx, mask]

    def get_all_host_slots(self, req_pool_idx: int) -> Optional[torch.Tensor]:
        """Return all host_slot values that are >= 0 for one request."""
        mask = self.host_slot[req_pool_idx] >= 0
        if not mask.any():
            return None
        return self.host_slot[req_pool_idx, mask]
