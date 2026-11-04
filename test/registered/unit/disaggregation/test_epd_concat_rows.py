"""A view over embedding parts must carry the same rows torch.cat produced.

The parts of one request land consecutively in a pinned slot, so the view path
returns them as one tensor without copying. If it ever returned a view of the
wrong span -- or a view over parts that are *not* an exact contiguous run --
the model would read another request's rows, with the right shape and no
error. These cases pin the fallback conditions against the torch.cat result.
"""

import unittest

import torch

from sglang.srt.disaggregation.encoder.receiver import concat_rows
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _buffer(rows, cols=3):
    return torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)


class TestConcatRows(CustomTestCase):
    def test_adjacent_parts_become_a_view(self):
        buffer = _buffer(10)
        parts = [buffer[0:4], buffer[4:7], buffer[7:10]]
        result = concat_rows(parts)
        self.assertTrue(torch.equal(result, buffer))
        # Same storage: no copy was made.
        self.assertEqual(
            result.untyped_storage().data_ptr(), buffer.untyped_storage().data_ptr()
        )

    def test_view_starts_at_the_first_part_not_the_buffer(self):
        buffer = _buffer(10)
        parts = [buffer[2:5], buffer[5:9]]
        result = concat_rows(parts)
        self.assertTrue(torch.equal(result, buffer[2:9]))

    def test_gap_between_parts_falls_back_to_a_copy(self):
        buffer = _buffer(10)
        parts = [buffer[0:3], buffer[4:7]]
        result = concat_rows(parts)
        self.assertTrue(torch.equal(result, torch.cat(parts, dim=0)))
        self.assertNotEqual(
            result.untyped_storage().data_ptr(), buffer.untyped_storage().data_ptr()
        )

    def test_out_of_order_or_separate_buffers_fall_back(self):
        buffer = _buffer(10)
        reversed_parts = [buffer[5:8], buffer[0:5]]
        self.assertTrue(
            torch.equal(
                concat_rows(reversed_parts), torch.cat(reversed_parts, dim=0)
            )
        )
        separate = [_buffer(3), _buffer(2)]
        self.assertTrue(torch.equal(concat_rows(separate), torch.cat(separate, dim=0)))

    def test_single_part_is_returned_unchanged(self):
        buffer = _buffer(4)
        self.assertIs(concat_rows([buffer]), buffer)


if __name__ == "__main__":
    unittest.main()
