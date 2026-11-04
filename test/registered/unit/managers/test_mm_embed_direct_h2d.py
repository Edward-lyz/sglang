"""Per-image embedding rows must land exactly where the host concat put them.

The direct-H2D path writes each image's pinned rows straight into the device
broadcast buffer instead of concatenating on the host first. It therefore owns
the row arithmetic that `torch.concat(...)[start:end]` used to do for free: an
off-by-one in the per-item window silently feeds another image's rows to the
model, which no shape check would catch. These cases pin the window math
against the concat reference.
"""

import unittest

import torch

from sglang.srt.managers.mm_schedule import copy_item_rows
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

HIDDEN = 4


class _Item:
    def __init__(self, rows, base):
        self.precomputed_embeddings = (
            torch.arange(rows * HIDDEN, dtype=torch.float32).reshape(rows, HIDDEN)
            + base
        )


def _reference(items, start_index, end_index):
    source = torch.concat(
        [item.precomputed_embeddings for item in items]
    ).reshape(-1, HIDDEN)
    return source[start_index:end_index]


class TestCopyItemRows(CustomTestCase):
    def _check(self, row_counts, start_index, end_index):
        items = [_Item(rows, 100 * i) for i, rows in enumerate(row_counts)]
        expected = _reference(items, start_index, end_index)
        destination = torch.zeros((end_index - start_index, HIDDEN))
        written = copy_item_rows(
            destination, items, start_index, end_index, HIDDEN
        )
        self.assertEqual(written, end_index - start_index)
        self.assertTrue(
            torch.equal(destination, expected),
            f"{row_counts} [{start_index},{end_index})",
        )

    def test_matches_concat_on_window_boundaries(self):
        # Whole range, a window inside one item, and windows that start or end
        # mid-item -- the three ways the per-item offset can be wrong.
        self._check([3, 5, 2], 0, 10)
        self._check([3, 5, 2], 3, 8)
        self._check([3, 5, 2], 1, 4)
        self._check([3, 5, 2], 4, 10)
        self._check([3, 5, 2], 2, 9)

    def test_single_row_window_inside_a_later_item(self):
        self._check([4, 4, 4], 9, 10)

    def test_empty_window_writes_nothing(self):
        items = [_Item(3, 0)]
        destination = torch.zeros((0, HIDDEN))
        self.assertEqual(copy_item_rows(destination, items, 2, 2, HIDDEN), 0)

    def test_items_beyond_the_window_are_not_read(self):
        items = [_Item(3, 0), _Item(3, 100), _Item(3, 200)]
        # Reading the third item would need its rows; drop them to prove the
        # loop stops once the window is filled.
        items[2].precomputed_embeddings = None
        destination = torch.zeros((4, HIDDEN))
        self.assertEqual(copy_item_rows(destination, items, 1, 5, HIDDEN), 4)
        self.assertTrue(torch.equal(destination, _reference(items[:2], 1, 5)))


if __name__ == "__main__":
    unittest.main()
