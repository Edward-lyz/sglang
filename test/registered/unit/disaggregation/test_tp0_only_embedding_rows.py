"""Row accounting for the attn-tp rank 0 only receive path.

The broadcast shape has to be derived identically on every rank, or the group
deadlocks; these pin ``_chunk_row_range`` against the slice rank 0 actually
takes.
"""

import pytest
import torch

from sglang.srt.managers.mm_schedule import _chunk_row_range, get_embedding_chunk


@pytest.mark.parametrize(
    "prefix_len,seq_len,offsets",
    [
        (0, 10, [(0, 9)]),
        (4, 6, [(2, 8)]),
        (0, 4, [(2, 8)]),
        (6, 4, [(0, 3), (6, 9)]),
        (0, 12, [(0, 3), (6, 9)]),
        (20, 4, [(0, 3)]),
    ],
)
def test_row_range_matches_the_slice_get_embedding_chunk_takes(
    prefix_len, seq_len, offsets
):
    """A metadata-only rank must size the broadcast exactly like rank 0 slices."""
    embedding = torch.arange(40, dtype=torch.float32).reshape(20, 2)

    _, start, end = get_embedding_chunk(
        embedding=embedding,
        extend_prefix_len=prefix_len,
        extend_seq_len=seq_len,
        items_offset=offsets,
    )

    assert (start, end) == _chunk_row_range(prefix_len, seq_len, offsets)
