from __future__ import annotations

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

"""
Page-aligned memory pool.
"""

import abc
from array import array
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.srt.utils import get_bool_env_var, get_num_new_pages, next_power_of_2

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

    def get_cpu_copy(self, *args, **kwargs):
        # FIXME: reuse the get_cpu_copy after paged allocator is implemented
        raise NotImplementedError()

    def load_cpu_copy(self, *args, **kwargs):
        # FIXME: reuse the load_cpu_copy after paged allocator is implemented
        raise NotImplementedError()

    def alloc_extend(self, *args, **kwargs):
        raise NotImplementedError("alloc_extend is only for paged allocator")

    def alloc_decode(self, *args, **kwargs):
        raise NotImplementedError("alloc_decode is only for paged allocator")

    @abc.abstractmethod
    def clear(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def alloc(self, need_size: int):
        raise NotImplementedError()

    @abc.abstractmethod
    def free(self, free_index: torch.Tensor):
        raise NotImplementedError()


class TokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """An allocator managing the indices to kv cache data."""

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
    ):
        super().__init__(size, 1, dtype, device, kvcache, need_sort)
        self.clear()

    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_pages = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )
        self.is_not_in_free_group = True
        self.free_group = []
        self.release_pages = torch.empty((0,), dtype=torch.int64, device=self.device)

    def available_size(self):
        # To avoid minor "len(free_pages) * 1" overhead
        return len(self.free_pages) + len(self.release_pages)

    def alloc(self, need_size: int):
        if self.need_sort and need_size > len(self.free_pages):
            self.merge_and_sort_free()

        if need_size > len(self.free_pages):
            return None

        select_index = self.free_pages[:need_size]
        self.free_pages = self.free_pages[need_size:]
        return select_index

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            if self.need_sort:
                self.release_pages = torch.cat((self.release_pages, free_index))
            else:
                self.free_pages = torch.cat((self.free_pages, free_index))
        else:
            self.free_group.append(free_index)

    def get_cpu_copy(self, indices):
        return self._kvcache.get_cpu_copy(indices)

    def load_cpu_copy(self, kv_cache_cpu, indices):
        return self._kvcache.load_cpu_copy(kv_cache_cpu, indices)


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


@triton.jit
def alloc_extend_kernel(
    pre_lens_ptr,
    seq_lens_ptr,
    last_loc_ptr,
    free_page_ptr,
    out_indices,
    bs_upper: tl.constexpr,
    page_size: tl.constexpr,
    max_num_extend_tokens: tl.constexpr,
    TILE_SIZE: tl.constexpr = 256,
):
    pid = tl.program_id(0)

    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
    pre_lens = tl.load(pre_lens_ptr + load_offset, mask=load_offset <= pid)
    extend_lens = seq_lens - pre_lens

    seq_len = tl.load(seq_lens_ptr + pid)
    pre_len = tl.load(pre_lens_ptr + pid)
    extend_len = seq_len - pre_len

    sum_extend_lens = tl.sum(extend_lens)
    output_start_loc = sum_extend_lens - extend_len

    num_pages_after = (seq_lens + page_size - 1) // page_size
    num_pages_before = (pre_lens + page_size - 1) // page_size
    num_new_pages = num_pages_after - num_pages_before

    num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
        pre_len + page_size - 1
    ) // page_size
    sum_num_new_pages = tl.sum(num_new_pages)
    new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

    # Part 1: fill the old partial page
    last_loc = tl.load(last_loc_ptr + pid)
    num_part1 = (
        min(seq_len, (pre_len + page_size - 1) // page_size * page_size) - pre_len
    )
    offset_one_page = tl.arange(0, page_size)
    tl.store(
        out_indices + output_start_loc + offset_one_page,
        last_loc + 1 + offset_one_page,
        mask=offset_one_page < num_part1,
    )
    if pre_len + num_part1 == seq_len:
        return

    # Part 2: fill the new full pages
    num_part2 = (
        seq_len // page_size * page_size
        - (pre_len + page_size - 1) // page_size * page_size
    )

    if max_num_extend_tokens > 0:
        offset_many_page = tl.arange(0, max_num_extend_tokens)
        page_start = tl.load(
            free_page_ptr + new_page_start_loc + offset_many_page // page_size,
            mask=offset_many_page < num_part2,
        )
        tl.store(
            out_indices + output_start_loc + num_part1 + offset_many_page,
            page_start * page_size + offset_many_page % page_size,
            mask=offset_many_page < num_part2,
        )
    else:
        cur_offset = 0
        while cur_offset < num_part2:
            offsets = cur_offset + tl.arange(0, TILE_SIZE)
            mask = offsets < num_part2

            page_indices = new_page_start_loc + (offsets // page_size)
            page_start = tl.load(free_page_ptr + page_indices, mask=mask)

            tl.store(
                out_indices + output_start_loc + num_part1 + offsets,
                page_start * page_size + (offsets % page_size),
                mask=mask,
            )
            cur_offset += TILE_SIZE

    if pre_len + num_part1 + num_part2 == seq_len:
        return

    # Part 3: fill the new partial page
    num_part3 = seq_len - seq_len // page_size * page_size
    start_loc = tl.load(
        free_page_ptr + new_page_start_loc + num_page_start_loc_self - 1
    )
    tl.store(
        out_indices + output_start_loc + num_part1 + num_part2 + offset_one_page,
        start_loc * page_size + offset_one_page,
        mask=offset_one_page < num_part3,
    )


@triton.jit
def alloc_decode_kernel(
    seq_lens_ptr,
    last_loc_ptr,
    free_page_ptr,
    out_indices,
    bs_upper: tl.constexpr,
    page_size: tl.constexpr,
):
    pid = tl.program_id(0)

    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
    pre_lens = tl.where(load_offset <= pid, seq_lens - 1, seq_lens)

    seq_len = tl.load(seq_lens_ptr + pid)
    pre_len = seq_len - 1

    num_pages_after = (seq_lens + page_size - 1) // page_size
    num_pages_before = (pre_lens + page_size - 1) // page_size
    num_new_pages = num_pages_after - num_pages_before

    num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
        pre_len + page_size - 1
    ) // page_size
    sum_num_new_pages = tl.sum(num_new_pages)
    new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

    if num_page_start_loc_self == 0:
        last_loc = tl.load(last_loc_ptr + pid)
        tl.store(out_indices + pid, last_loc + 1)
    else:
        page = tl.load(free_page_ptr + new_page_start_loc)
        tl.store(out_indices + pid, page * page_size)


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
        self.seen_max_num_extend_tokens_next_power_of_2 = 1
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

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 1) % self.page_size == prefix_lens % self.page_size
            )

        self.seen_max_num_extend_tokens_next_power_of_2 = max(
            self.seen_max_num_extend_tokens_next_power_of_2,
            next_power_of_2(extend_num_tokens),
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
            -1,
        )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

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

    def backup_state(self):
        return (
            self.free_pages,
            self.release_pages,
            self._contiguous_tail_pages,
            self._contiguous_tail_end_page,
            tuple(self._known_release_page_ids),
            self._known_release_page_ids_complete,
        )

    def restore_state(self, state):
        self.free_pages, self.release_pages = state[:2]
        self._contiguous_tail_pages, self._contiguous_tail_end_page = state[2:4]
        if len(state) >= 6:
            self._known_release_page_ids = list(state[4])
            self._known_release_page_ids_complete = state[5]
        else:
            # Backward compat with pre-CPU-metadata states: a non-empty release tensor is unknown.
            self._known_release_page_ids = []
            self._known_release_page_ids_complete = len(self.release_pages) == 0

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

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            free_page_indices = torch.unique(free_index // self.page_size)
            if self.need_sort:
                self.release_pages = torch.cat((free_page_indices, self.release_pages))
            else:
                self.free_pages = torch.cat((free_page_indices, self.free_pages))
        else:
            self.free_group.append(free_index)

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
        self._known_release_page_ids = []
        self._known_release_page_ids_complete = True
        self._contiguous_tail_pages = self.num_pages
        self._contiguous_tail_end_page = self.num_pages

    def get_cpu_copy(self, indices):
        return self._kvcache.get_cpu_copy(indices)

    def load_cpu_copy(self, kv_cache_cpu, indices):
        return self._kvcache.load_cpu_copy(kv_cache_cpu, indices)
