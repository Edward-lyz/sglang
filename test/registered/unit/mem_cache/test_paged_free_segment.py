"""free_segment / free_segments vs the torch.unique reference: page-aligned
starts over every tail alignment, the page-disjoint contract, and free-group
deferral. See PagedTokenToKVPoolAllocator.free_segment for why unique is avoided.

    python -m pytest test/registered/unit/mem_cache/test_paged_free_segment.py -v
"""

import unittest

import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

PAGE_SIZE = 4
NUM_PAGES = 64


def _make_allocator(need_sort=False, page_size=PAGE_SIZE):
    return PagedTokenToKVPoolAllocator(
        size=NUM_PAGES * page_size,
        page_size=page_size,
        dtype=torch.float16,
        device="cpu",
        kvcache=None,
        need_sort=need_sort,
    )


def _make_kv_row(alloc, num_tokens, page_size=PAGE_SIZE):
    # Page-aligned allocation, then trim to num_tokens: mirrors a request's
    # req_to_token row (token position t lives at page*page_size + t%page_size).
    num_pages = -(num_tokens // -page_size)
    indices = alloc.alloc(num_pages * page_size)
    return indices[:num_tokens]


class TestFreeSegment(unittest.TestCase):
    def test_matches_unique_over_tail_alignments(self):
        for num_tokens in (1, PAGE_SIZE, PAGE_SIZE + 1, 3 * PAGE_SIZE - 1):
            for start in range(0, num_tokens, PAGE_SIZE):
                for end in range(start + 1, num_tokens + 1):
                    alloc = _make_allocator()
                    row = _make_kv_row(alloc, num_tokens)
                    expected = torch.unique(row[start:end] // PAGE_SIZE)
                    before = len(alloc.free_pages)
                    alloc.free_segment(row[start:end], start_pos=start)
                    freed = alloc.free_pages[: len(alloc.free_pages) - before]
                    self.assertTrue(
                        torch.equal(torch.sort(freed)[0], expected),
                        f"{num_tokens=} {start=} {end=}",
                    )

    def test_empty_segment_is_noop(self):
        alloc = _make_allocator()
        row = _make_kv_row(alloc, PAGE_SIZE)
        before = len(alloc.free_pages)
        alloc.free_segment(row[:0], start_pos=0)
        self.assertEqual(len(alloc.free_pages), before)

    def test_unaligned_start_is_rejected(self):
        alloc = _make_allocator()
        row = _make_kv_row(alloc, 2 * PAGE_SIZE)
        for start in (1, PAGE_SIZE - 1, PAGE_SIZE + 1):
            with self.assertRaises(AssertionError):
                alloc.free_segment(row[start:], start_pos=start)

    def test_need_sort_routes_to_release_pages(self):
        alloc = _make_allocator(need_sort=True)
        row = _make_kv_row(alloc, 2 * PAGE_SIZE)
        alloc.free_segment(row, start_pos=0)
        self.assertEqual(len(alloc.release_pages), 2)
        self.assertEqual(len(alloc.free_pages), NUM_PAGES - 2)

    def test_group_defers_until_group_end(self):
        alloc = _make_allocator()
        row = _make_kv_row(alloc, 2 * PAGE_SIZE)
        before = len(alloc.free_pages)
        alloc.free_group_begin()
        alloc.free_segment(row, start_pos=0)
        self.assertEqual(len(alloc.free_pages), before)
        alloc.free_group_end()
        self.assertEqual(len(alloc.free_pages), before + 2)

    def test_group_owns_its_indices(self):
        # The deferred queue must not alias the caller's req_to_token row.
        alloc = _make_allocator()
        row = _make_kv_row(alloc, 2 * PAGE_SIZE)
        expected = torch.unique(row // PAGE_SIZE)
        alloc.free_group_begin()
        alloc.free_segment(row, start_pos=0)
        row.fill_(0)  # a later request rewrites the row before the group closes
        before = len(alloc.free_pages)
        alloc.free_group_end()
        freed = alloc.free_pages[: len(alloc.free_pages) - before]
        self.assertTrue(torch.equal(torch.sort(freed)[0], expected))

    def test_segments_reject_shared_page(self):
        alloc = _make_allocator()
        row = _make_kv_row(alloc, 3 * PAGE_SIZE)
        # second segment starts inside the page the first one already released
        with self.assertRaises(AssertionError):
            alloc.free_segments(
                [(row[0 : PAGE_SIZE + 1], 0), (row[PAGE_SIZE + 1 :], PAGE_SIZE + 1)]
            )

    def test_segments_accept_disjoint_pages(self):
        alloc = _make_allocator()
        row = _make_kv_row(alloc, 3 * PAGE_SIZE)
        expected = torch.unique(row // PAGE_SIZE)
        before = len(alloc.free_pages)
        alloc.free_segments(
            [(row[:PAGE_SIZE], 0), (row[PAGE_SIZE:], PAGE_SIZE)],
        )
        freed = alloc.free_pages[: len(alloc.free_pages) - before]
        self.assertTrue(torch.equal(torch.sort(freed)[0], expected))

    def test_dcp_widened_page_release_is_whole(self):
        # DCP: allocator page is page_size * dcp_size. A committed prefix that
        # ends mid-page releases the whole page, so the overallocated tail must
        # start at the next allocator page or it double-frees.
        ps = 8  # page_size 2 * dcp_size 4
        alloc = _make_allocator(page_size=ps)
        row = _make_kv_row(alloc, 3 * ps, page_size=ps)
        committed = ps + 2  # mid-page end, aligned on the narrow 2-token grid
        before = len(alloc.free_pages)
        alloc.free_group_begin()
        alloc.free_segment(row[:committed], start_pos=0)
        # next allocator page, not ceil_align(committed, 2)
        alloc.free_segment(row[2 * ps :], start_pos=2 * ps)
        alloc.free_group_end()
        freed = alloc.free_pages[: len(alloc.free_pages) - before]
        self.assertEqual(len(freed), len(torch.unique(freed)))
        self.assertTrue(torch.equal(torch.sort(freed)[0], torch.unique(row // ps)))


if __name__ == "__main__":
    unittest.main()
