"""CPU tests for grouped frees in PagedTokenToKVPoolAllocator."""

import random
import unittest

import torch

from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE_SIZE = 64


def _ceil_align(x, page_size=PAGE_SIZE):
    return (x + page_size - 1) // page_size * page_size


class TestPagedAllocatorFreeGroup(unittest.TestCase):
    def _make_allocator(self, need_sort, num_pages=256, device="cpu"):
        return PagedTokenToKVPoolAllocator(
            size=num_pages * PAGE_SIZE,
            page_size=PAGE_SIZE,
            dtype=torch.int64,
            device=device,
            kvcache=None,
            need_sort=need_sort,
        )

    def _assert_no_duplicate_pages(self, allocator):
        pages = torch.cat((allocator.free_pages, allocator.release_pages)).cpu()
        self.assertEqual(len(torch.unique(pages)), len(pages))

    def _check_unaligned_segments(self, device):
        for need_sort in (True, False):
            allocator = self._make_allocator(need_sort, device=device)
            full = allocator.available_size()

            # Two requests with partial last pages: 100 tokens (2 pages) and
            # 70 tokens (2 pages). Strided slicing over the concatenation
            # would drop the second request's last page.
            a = allocator.alloc(_ceil_align(100))[:100]
            b = allocator.alloc(_ceil_align(70))[:70]

            allocator.free_group_begin()
            allocator.free(a)
            allocator.free(b)
            allocator.free_group_end()

            self.assertEqual(allocator.available_size(), full)
            self._assert_no_duplicate_pages(allocator)

    def test_unaligned_segments_cpu(self):
        self._check_unaligned_segments("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_unaligned_segments_cuda(self):
        self._check_unaligned_segments("cuda")

    def test_committed_and_over_alloc_tail(self):
        # Mirrors release_kv_cache under spec v2: free [0, committed) then the
        # over-allocated tail [ceil_align(committed), allocated).
        allocator = self._make_allocator(need_sort=True)
        full = allocator.available_size()

        committed, allocated = 125, 131
        tokens = allocator.alloc(_ceil_align(allocated))[:allocated]

        allocator.free_group_begin()
        allocator.free(tokens[:committed])
        allocator.free(tokens[_ceil_align(committed) :])
        allocator.free_group_end()

        self.assertEqual(allocator.available_size(), full)
        self._assert_no_duplicate_pages(allocator)

    def test_mixed_with_free_page_ids(self):
        allocator = self._make_allocator(need_sort=True)
        full = allocator.available_size()

        a = allocator.alloc(_ceil_align(100))[:100]
        b = allocator.alloc(2 * PAGE_SIZE)

        allocator.free_group_begin()
        allocator.free(a)
        allocator.free_page_ids(torch.unique(b // PAGE_SIZE))
        allocator.free_group_end()

        self.assertEqual(allocator.available_size(), full)
        self._assert_no_duplicate_pages(allocator)

    def test_random_grouped_release(self):
        rng = random.Random(0)
        allocator = self._make_allocator(need_sort=True, num_pages=2048)
        full = allocator.available_size()

        for _ in range(200):
            reqs = []
            for _ in range(rng.randint(1, 8)):
                allocated = rng.randint(1, 1000)
                committed = rng.randint(1, allocated)
                tokens = allocator.alloc(_ceil_align(allocated))
                self.assertIsNotNone(tokens)
                reqs.append((tokens[:allocated], committed))

            allocator.free_group_begin()
            for tokens, committed in reqs:
                allocator.free(tokens[:committed])
                allocator.free(tokens[_ceil_align(committed) :])
            allocator.free_group_end()

            self.assertEqual(allocator.available_size(), full)

        allocator.merge_and_sort_free()
        self._assert_no_duplicate_pages(allocator)

    def test_empty_group(self):
        allocator = self._make_allocator(need_sort=True)
        full = allocator.available_size()

        allocator.free_group_begin()
        allocator.free_group_end()

        self.assertTrue(allocator.is_not_in_free_group)
        self.assertEqual(allocator.free_group, [])
        self.assertEqual(allocator.available_size(), full)


if __name__ == "__main__":
    unittest.main()
