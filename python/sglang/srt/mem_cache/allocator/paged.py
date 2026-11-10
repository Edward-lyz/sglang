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

"""
Page-aligned memory pool.
"""


from array import array
from typing import TYPE_CHECKING

import torch

from sglang.kernels.ops.memory.allocator import (
    alloc_decode_kernel,
    alloc_extend_kernel,
)
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.utils import (
    get_bool_env_var,
    get_num_new_pages,
    is_hip,
    next_power_of_2,
)

_is_hip = is_hip()

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache


def alloc_extend_naive(
    prefix_lens,
    seq_lens,
    last_loc,
    free_pages,
    out_indices,
    page_size,
    device,
):
    extend_lens = seq_lens - prefix_lens
    end_pos = torch.cumsum(extend_lens, 0)
    start_pos = end_pos - extend_lens
    num_new_pages = (seq_lens + page_size - 1) // page_size - (
        prefix_lens + page_size - 1
    ) // page_size
    num_full_new_pages = (seq_lens) // page_size - (
        prefix_lens + page_size - 1
    ) // page_size
    need_page = num_new_pages - num_full_new_pages
    end_new_pages = torch.cumsum(num_new_pages, 0)
    start_new_pages = end_new_pages - num_new_pages
    pos_in_page = torch.arange(page_size, device=device, dtype=torch.int32)
    for i in range(len(prefix_lens)):
        num1 = (
            min(
                seq_lens[i],
                (prefix_lens[i] + page_size - 1) // page_size * page_size,
            )
            - prefix_lens[i]
        )
        if num1:
            out_indices[start_pos[i] : start_pos[i] + num1] = (
                last_loc[i] + 1 + pos_in_page[:num1].view(-1)
            )

        if prefix_lens[i] + num1 == seq_lens[i]:
            continue

        num2 = (
            seq_lens[i] // page_size - (prefix_lens[i] + page_size - 1) // page_size
        ) * page_size
        if num2:
            pages = (
                free_pages[start_new_pages[i] : end_new_pages[i] - need_page[i]]
                * page_size
            )
            out_indices[start_pos[i] + num1 : start_pos[i] + num1 + num2] = (
                pages.view(-1, 1) + pos_in_page.view(1, -1)
            ).view(-1)

        if prefix_lens[i] + num1 + num2 == seq_lens[i]:
            continue

        num3 = seq_lens[i] - seq_lens[i] // page_size * page_size
        if num3:
            out_indices[end_pos[i] - num3 : end_pos[i]] = (
                free_pages[end_new_pages[i] - 1] * page_size + pos_in_page[:num3]
            ).view(-1)


class PagedTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """
    An allocator managing the indices to kv cache data.

    This class has the same interface as `TokenToKVPoolAllocator` but the output
    of one request is always page-aligned.

    TODO: fuse last_loc into the kernel.
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
    ):
        super().__init__(size, page_size, dtype, device, kvcache, need_sort)
        self.num_pages = size // page_size
        self.debug_mode = get_bool_env_var("SGLANG_DEBUG_MEMORY_POOL")

        # Pre-warm the torch.unique HIP kernel used in free(). When a request
        # finishes with a prompt that already exists in the radix tree (e.g.
        # bench_serving sending the same warmup+measured prompt), the radix
        # cache's _insert_helper frees the duplicate KV indices via
        # token_to_kv_pool_allocator.free(value[start:prefix_len]). That call
        # path runs `torch.unique(free_index // self.page_size)` on a
        # ~prompt_len-sized int64 tensor. The first such call on AMD ROCm
        # JIT-compiles rocPRIM sort/unique kernels and costs ~200ms, which
        # shows up as a mysterious "second-request slow" (Run 1) for
        # repeated-prompt benchmarks. Running it once at init time moves
        # that JIT cost to startup. This is a ROCm-only JIT cost, so the
        # warm-up is gated on _is_hip and skipped on other platforms.
        if _is_hip and torch.cuda.is_available():
            try:
                _warmup = torch.arange(1024, dtype=torch.int64, device=device)
                _ = torch.unique(_warmup // page_size)
                torch.cuda.synchronize()
            except Exception:
                pass
        self.clear()

    def alloc(self, need_size: int):
        # page-aligned allocation, returning contiguous indices of pages
        if self.debug_mode:
            assert (
                need_size % self.page_size == 0
            ), "The allocation size should be page-aligned"

        num_pages = need_size // self.page_size
        if self.need_sort and num_pages > len(self.free_pages):
            self.merge_and_sort_free()
        if num_pages > len(self.free_pages):
            return None

        out_pages = self.free_pages[:num_pages]
        self.free_pages = self.free_pages[num_pages:]
        self._contiguous_tail_pages = min(
            self._contiguous_tail_pages, len(self.free_pages)
        )

        out_indices = (
            out_pages[:, None] * self.page_size
            + torch.arange(self.page_size, device=self.device)
        ).reshape(-1)

        return out_indices

    def merge_and_sort_free(self):
        """合并待释放页并维护连续空闲段提示，避免GPU元数据同步回Host。"""
        if len(self.release_pages) == 0:
            return

        # free_pages is on the accelerator; reading it back to find the new
        # suffix would sync the host. CPU frees track page IDs to update the
        # hint afterward; GPU releases fall back to conservative updates.
        release_count = len(self.release_pages)
        known_release_page_ids = self._known_release_page_ids
        known_release_complete = (
            self._known_release_page_ids_complete
            and len(known_release_page_ids) == release_count
        )
        old_tail_pages = self._contiguous_tail_pages
        old_tail_end = self._contiguous_tail_end_page
        if old_tail_pages > 0 and self.free_pages.device.type == "cpu":
            # CPU/mock allocators can cheaply validate restored state;
            # the GPU path skips this to avoid sync.
            if (
                old_tail_end < old_tail_pages
                or old_tail_end > self.num_pages
                or old_tail_pages > len(self.free_pages)
            ):
                old_tail_pages = 0
                old_tail_end = 0
            else:
                expected_tail = torch.arange(
                    old_tail_end - old_tail_pages + 1,
                    old_tail_end + 1,
                    dtype=self.free_pages.dtype,
                )
                if not torch.equal(self.free_pages[-old_tail_pages:], expected_tail):
                    old_tail_pages = 0
                    old_tail_end = 0

        super().merge_and_sort_free()

        # Drop references to the consumed release batch to bound memory across merges.
        self._known_release_page_ids = []
        self._known_release_page_ids_complete = True

        if not known_release_complete:
            # A suffix at the highest page can't be displaced by an unknown release;
            # keep it but don't extend without CPU IDs.
            if old_tail_pages > 0 and old_tail_end == self.num_pages:
                self._contiguous_tail_pages = old_tail_pages
                self._contiguous_tail_end_page = old_tail_end
            else:
                self._contiguous_tail_pages = 0
                self._contiguous_tail_end_page = 0
            return

        known_ids = set(known_release_page_ids)
        if len(known_ids) != release_count or any(
            page_id < 1 or page_id > self.num_pages for page_id in known_ids
        ):
            self._contiguous_tail_pages = 0
            self._contiguous_tail_end_page = 0
            return

        if old_tail_pages > 0 and old_tail_end > 0:
            # The known tail was the end of free_pages. A freed page extends
            # it, or starts a newer high run if the adjacent page is still allocated.
            candidate_end = max(old_tail_end, max(known_ids, default=old_tail_end))
            old_tail_start = old_tail_end - old_tail_pages + 1
            if candidate_end == old_tail_end:
                tail_pages = old_tail_pages
                cursor = old_tail_start - 1
                while cursor in known_ids:
                    tail_pages += 1
                    cursor -= 1
            else:
                cursor = candidate_end
                while cursor > old_tail_end and cursor in known_ids:
                    cursor -= 1
                if cursor == old_tail_end:
                    tail_pages = candidate_end - old_tail_end + old_tail_pages
                    cursor = old_tail_start - 1
                    while cursor in known_ids:
                        tail_pages += 1
                        cursor -= 1
                else:
                    # Gap between old suffix and new release.
                    # The run ending at candidate_end is safe:
                    # all pages above the old suffix are in the CPU release set.
                    tail_pages = candidate_end - cursor
        elif known_ids and max(known_ids) == self.num_pages:
            # No old hint, but a CPU release containing the highest page gives
            # an unambiguous suffix. Lower endpoints could be hidden by unknown GPU frees.
            candidate_end = self.num_pages
            cursor = candidate_end
            while cursor in known_ids:
                cursor -= 1
            tail_pages = candidate_end - cursor
        else:
            self._contiguous_tail_pages = 0
            self._contiguous_tail_end_page = 0
            return

        if tail_pages <= 0 or tail_pages > len(self.free_pages):
            self._contiguous_tail_pages = 0
            self._contiguous_tail_end_page = 0
        else:
            self._contiguous_tail_pages = tail_pages
            self._contiguous_tail_end_page = candidate_end

    def alloc_contiguous(self, need_size: int):
        """Allocate a contiguous block of pages of the given size."""
        result = self.alloc_contiguous_with_page_ids(need_size)
        return None if result is None else result[0]

    def alloc_contiguous_with_page_ids(self, need_size: int):
        """Allocate from the known-contiguous free-page tail when available.

        Opt-in: normal allocations keep using ``alloc`` for unchanged ordering/fragmentation.
        Never scan the GPU free list here (host sync on fallback). Return page IDs from CPU
        metadata so AS skips a CUDA tensor-to-list conversion before the SDK read.
        """
        if (
            need_size % self.page_size != 0
            or not self.need_sort
            or str(self.device).startswith("npu")
        ):
            return None

        num_pages = need_size // self.page_size
        if num_pages == 0:
            return (
                torch.empty((0,), dtype=torch.int64, device=self.device),
                array("q"),
            )

        if num_pages > self._contiguous_tail_pages and len(self.release_pages) > 0:
            self.merge_and_sort_free()

        if num_pages > self._contiguous_tail_pages:
            return None

        page_end = self._contiguous_tail_end_page
        page_start = page_end - num_pages + 1
        out_pages = self.free_pages[-num_pages:]
        self.free_pages = self.free_pages[:-num_pages]
        self._contiguous_tail_pages -= num_pages
        self._contiguous_tail_end_page = page_start - 1
        out_indices = (
            out_pages[:, None] * self.page_size
            + torch.arange(self.page_size, device=self.device)
        ).reshape(-1)
        return out_indices, array("q", range(page_start, page_end + 1))

    def alloc_contiguous_by_scan_with_page_ids(self, need_size: int):
        """Find a free physical run after the constant-time tail hint misses.

        Only AS admission opts into this synchronized CPU scan. Normal paged
        allocations retain their existing order and behavior.
        """
        if (
            need_size < 0
            or need_size % self.page_size != 0
            or not self.need_sort
            or str(self.device).startswith("npu")
        ):
            return None
        num_pages = need_size // self.page_size
        if num_pages == 0:
            return (
                torch.empty((0,), dtype=torch.int64, device=self.device),
                array("q"),
            )
        if len(self.release_pages) > 0:
            self.merge_and_sort_free()
        if num_pages > len(self.free_pages):
            return None

        page_ids = self.free_pages.tolist()
        run_start = 0
        for end, page_id in enumerate(page_ids):
            if end > 0 and page_id != page_ids[end - 1] + 1:
                run_start = end
            if end - run_start + 1 == num_pages:
                stop = end + 1
                out_pages = self.free_pages[run_start:stop]
                out_indices = (
                    out_pages[:, None] * self.page_size
                    + torch.arange(self.page_size, device=self.device)
                ).reshape(-1)
                next_free_pages = torch.cat(
                    (self.free_pages[:run_start], self.free_pages[stop:])
                )
                remaining = page_ids[:run_start] + page_ids[stop:]
                self.free_pages = next_free_pages
                if remaining:
                    tail = 1
                    while (
                        tail < len(remaining)
                        and remaining[-tail - 1] == remaining[-tail] - 1
                    ):
                        tail += 1
                    self._contiguous_tail_pages = tail
                    self._contiguous_tail_end_page = remaining[-1]
                else:
                    self._contiguous_tail_pages = 0
                    self._contiguous_tail_end_page = 0
                return out_indices, array("q", page_ids[run_start:stop])
        return None

    def free_page_ids(self, page_ids: torch.Tensor):
        """Free physical page IDs without expanding token-level KV indices."""
        if page_ids.numel() == 0:
            return

        cpu_page_ids = None
        if page_ids.device.type == "cpu":
            cpu_page_ids = [
                int(page_id)
                for page_id in page_ids.to(dtype=torch.int64).reshape(-1).tolist()
            ]
            unique_page_ids = set(cpu_page_ids)
            if len(unique_page_ids) != len(cpu_page_ids):
                raise ValueError("Cannot free duplicate physical page IDs")
            if any(
                page_id < 1 or page_id > self.num_pages for page_id in unique_page_ids
            ):
                raise ValueError(
                    f"Physical page IDs must be within [1, {self.num_pages}]"
                )
            if unique_page_ids.intersection(self._known_release_page_ids):
                raise ValueError("Physical page IDs are already pending release")
            if self.free_pages.device.type == "cpu":
                existing_page_ids = set(self.free_pages.tolist())
                existing_page_ids.update(self.release_pages.tolist())
                if unique_page_ids.intersection(existing_page_ids):
                    raise ValueError("Physical page IDs are already free")

        if self.is_not_in_free_group:
            if self.need_sort:
                if cpu_page_ids is not None:
                    self._known_release_page_ids.extend(cpu_page_ids)
                else:
                    self._known_release_page_ids_complete = False
                page_ids = page_ids.to(device=self.device, dtype=torch.int64)
                self.release_pages = torch.cat((page_ids, self.release_pages))
            else:
                page_ids = page_ids.to(device=self.device, dtype=torch.int64)
                self.free_pages = torch.cat((page_ids, self.free_pages))
        else:
            # Preserve free_group's token-index contract for grouped frees.
            page_ids = page_ids.to(device=self.device, dtype=torch.int64)
            offsets = torch.arange(self.page_size, device=self.device)
            self.free_group.append(
                (page_ids[:, None] * self.page_size + offsets).reshape(-1)
            )

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        num_new_pages: int = None,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 1) % self.page_size == prefix_lens % self.page_size
            )

        bs = len(prefix_lens)
        if self.need_sort and extend_num_tokens // self.page_size + bs + 1 > len(
            self.free_pages
        ):
            self.merge_and_sort_free()

        out_indices = torch.empty(
            (extend_num_tokens,), dtype=torch.int64, device=self.device
        )

        alloc_extend_kernel[(bs,)](
            prefix_lens,
            seq_lens,
            last_loc,
            self.free_pages,
            out_indices,
            next_power_of_2(bs),
            self.page_size,
        )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        if num_new_pages is None:
            num_new_pages = get_num_new_pages(
                seq_lens=seq_lens_cpu,
                page_size=self.page_size,
                prefix_lens=prefix_lens_cpu,
            )
        if num_new_pages > len(self.free_pages):
            return None

        self.free_pages = self.free_pages[num_new_pages:]
        self._contiguous_tail_pages = min(
            self._contiguous_tail_pages, len(self.free_pages)
        )
        return out_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 2) % self.page_size == seq_lens % self.page_size
            )

        bs = len(seq_lens)
        if self.need_sort and bs > len(self.free_pages):
            self.merge_and_sort_free()

        out_indices = torch.empty((bs,), dtype=torch.int64, device=self.device)
        alloc_decode_kernel[(bs,)](
            seq_lens,
            last_loc,
            self.free_pages,
            out_indices,
            next_power_of_2(bs),
            self.page_size,
        )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=self.page_size,
            decode=True,
        )
        if num_new_pages > len(self.free_pages):
            return None

        self.free_pages = self.free_pages[num_new_pages:]
        self._contiguous_tail_pages = min(
            self._contiguous_tail_pages, len(self.free_pages)
        )
        return out_indices

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            free_page_indices = free_index[:: self.page_size] // self.page_size
            if self.need_sort:
                self._known_release_page_ids_complete = False
                self.release_pages = torch.cat((free_page_indices, self.release_pages))
            else:
                self.free_pages = torch.cat((free_page_indices, self.free_pages))
        else:
            self.free_group.append(free_index)

        if self.debug_mode:
            assert len(torch.unique(self.free_pages)) == len(self.free_pages)

    def free_group_end(self):
        """C4: release grouped ranges without losing per-request page
        boundaries. A grouped range can end in a partial page; concatenating
        token ranges before striding by page_size shifts the page boundary of
        every subsequent request and leaks pages. Extract page ids from each
        range first, then combine them for one allocator update."""
        self.is_not_in_free_group = True
        page_id_chunks = [
            free_index[:: self.page_size] // self.page_size
            for free_index in self.free_group
            if free_index.numel() > 0
        ]
        self.free_group = []
        if not page_id_chunks:
            return
        free_page_ids = torch.cat(page_id_chunks)
        if self.need_sort:
            self._known_release_page_ids_complete = False
            self.release_pages = torch.cat((free_page_ids, self.release_pages))
        else:
            self.free_pages = torch.cat((free_page_ids, self.free_pages))
        if self.debug_mode:
            assert len(torch.unique(self.free_pages)) == len(self.free_pages)

    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_pages = torch.arange(
            1, self.num_pages + 1, dtype=torch.int64, device=self.device
        )
        self.is_not_in_free_group = True
        self.free_group = []
        self.release_pages = torch.empty((0,), dtype=torch.int64, device=self.device)
        # 中文：只维护已知连续尾段和CPU释放页；未知GPU释放不推测页号，
        # 扫描是AS显式选择的回退。保留新版按请求边界的free_group_end实现。
        self._known_release_page_ids = []
        self._known_release_page_ids_complete = True
        self._contiguous_tail_pages = self.num_pages
        self._contiguous_tail_end_page = self.num_pages

    def get_cpu_copy(self, indices, mamba_indices=None):
        return self._kvcache.get_cpu_copy(indices, mamba_indices=mamba_indices)

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        return self._kvcache.load_cpu_copy(
            kv_cache_cpu, indices, mamba_indices=mamba_indices
        )
