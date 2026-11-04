"""
Copyright 2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache


class BaseTokenToKVPoolAllocator(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self._kvcache = kvcache
        self.need_sort = need_sort

        self.free_pages = None
        self.release_pages = None
        self.is_not_in_free_group = True
        self.free_group = []

    @property
    def size_full(self):
        return self.size

    def debug_print(self) -> str:
        return ""

    def available_size(self):
        return (len(self.free_pages) + len(self.release_pages)) * self.page_size

    def get_kvcache(self):
        return self._kvcache

    def restore_state(self, state):
        self.free_pages, self.release_pages = state

    def backup_state(self):
        return (self.free_pages, self.release_pages)

    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            self.free(torch.cat(self.free_group))

    def merge_and_sort_free(self):
        if len(self.release_pages) > 0:
            self.free_pages = torch.cat((self.free_pages, self.release_pages))
            self.free_pages, _ = torch.sort(self.free_pages)
            self.release_pages = torch.empty(
                (0,), dtype=self.release_pages.dtype, device=self.device
            )

    def get_cpu_copy(self, indices, mamba_indices=None):
        # FIXME: reuse the get_cpu_copy after paged allocator is implemented
        raise NotImplementedError()

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        # FIXME: reuse the load_cpu_copy after paged allocator is implemented
        raise NotImplementedError()

    def alloc_extend(self, *args, **kwargs):
        raise NotImplementedError("alloc_extend is only for paged allocator")

    def alloc_decode(self, *args, **kwargs):
        raise NotImplementedError("alloc_decode is only for paged allocator")

    def resize(self, config) -> None:
        self.size = config.max_total_num_tokens
        if self.page_size > 1:
            self.num_pages = config.max_total_num_tokens // self.page_size
        self.clear()

    @abc.abstractmethod
    def clear(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def alloc(self, need_size: int):
        raise NotImplementedError()

    @abc.abstractmethod
    def free(self, free_index: torch.Tensor):
        raise NotImplementedError()

    @staticmethod
    def _copy_for_free_group(free_index: torch.Tensor) -> torch.Tensor:
        """Take ownership before a caller can mutate a deferred tensor view."""
        return free_index.clone()

    def free_segment(self, free_index: torch.Tensor, *, start_pos: int):
        """Free ``kv_row[start_pos : start_pos + n]`` of one request.

        In page units the segment is ``[start_pos // ps, ceil(end / ps))``:
        ``start_pos`` sits on a page boundary, the end may fall mid-page, and
        the whole last page is released. Subclasses use that to skip the
        data-dependent dedup in ``free()``. Default: plain ``free()``.
        """
        assert (
            start_pos % self.page_size == 0
        ), f"segment start {start_pos} is not page-aligned"
        self.free(free_index)

    def free_segments(self, segments):
        """Free several ``(free_index, start_pos)`` segments of one request's
        kv row.

        Each segment covers the pages ``[start_pos // ps, ceil(end / ps))``.
        Starts sit on page boundaries, ends may fall mid-page, and the page
        ranges of consecutive segments do not overlap -- so in page units the
        segments are aligned and disjoint, and every page is released once.
        """
        ps = self.page_size
        prev_end = None
        for free_index, start_pos in segments:
            n = free_index.numel()
            if n == 0:
                continue
            assert (
                prev_end is None or start_pos // ps > (prev_end - 1) // ps
            ), f"segment at {start_pos} shares a page with the one ending at {prev_end}"
            prev_end = start_pos + n
            self.free_segment(free_index, start_pos=start_pos)
