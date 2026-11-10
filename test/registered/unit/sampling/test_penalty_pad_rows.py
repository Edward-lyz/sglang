"""pad_rows_host vs torch.nn.utils.rnn.pad_sequence: the padded stop-token-id
matrix must be identical, including the ragged, empty and all-empty cases.
See BatchedMinNewTokensPenalizer._prepare for why the per-row device tensors
were replaced by one pinned host buffer.

    python -m pytest test/registered/unit/sampling/test_penalty_pad_rows.py -v
"""

import unittest

import torch

from sglang.srt.sampling.penaltylib.orchestrator import pad_rows_host
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

VOCAB_SIZE = 32000


def _reference(rows, padding_value):
    return torch.nn.utils.rnn.pad_sequence(
        sequences=[torch.tensor(row, dtype=torch.int64) for row in rows],
        batch_first=True,
        padding_value=padding_value,
    )


class TestPadRowsHost(unittest.TestCase):
    def test_matches_pad_sequence(self):
        cases = [
            [[1]],
            [[1, 2, 3]],
            [[1], [2, 3], [4, 5, 6]],
            [[1, 2, 3], [4], [5, 6]],
            [[7, 8], [9, 10]],
            [[1], []],
            [[], [2, 3]],
        ]
        for rows in cases:
            got = pad_rows_host(
                rows,
                padding_value=VOCAB_SIZE,
                dtype=torch.int64,
                pin_memory=False,
            )
            want = _reference(rows, VOCAB_SIZE)
            self.assertEqual(got.shape, want.shape, f"{rows=}")
            self.assertTrue(torch.equal(got, want), f"{rows=} {got=} {want=}")

    def test_all_empty_rows(self):
        # pad_sequence on all-empty rows yields (bs, 0); so must we.
        rows = [[], [], []]
        got = pad_rows_host(
            rows, padding_value=VOCAB_SIZE, dtype=torch.int64, pin_memory=False
        )
        self.assertEqual(tuple(got.shape), (3, 0))

    def test_no_rows(self):
        got = pad_rows_host(
            [], padding_value=VOCAB_SIZE, dtype=torch.int64, pin_memory=False
        )
        self.assertEqual(tuple(got.shape), (0, 0))

    def test_scatter_add_equivalence(self):
        # The consumer builds stop_token_penalties via scatter_add_ on the
        # padded matrix; identical padding means identical penalties.
        rows = [[1, 5], [7], []]
        for build in (pad_rows_host, None):
            padded = (
                pad_rows_host(
                    rows,
                    padding_value=VOCAB_SIZE,
                    dtype=torch.int64,
                    pin_memory=False,
                )
                if build
                else _reference(rows, VOCAB_SIZE)
            )
            penalties = torch.zeros(
                (len(rows), VOCAB_SIZE + 1), dtype=torch.float32
            ).scatter_add_(
                dim=1,
                index=padded,
                src=torch.full_like(padded, -1.0, dtype=torch.float32),
            )[
                :, :VOCAB_SIZE
            ]
            if build:
                mine = penalties
            else:
                self.assertTrue(torch.equal(mine, penalties))


if __name__ == "__main__":
    unittest.main()
