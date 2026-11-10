"""AS block-hash reuse must only keep blocks whose tokens provably match.

A kept hash describes the tokens of its block; keeping one block too many
silently gives AttentionStore a key for bytes it never wrote, so the request
reads back another prompt's KV. These cases pin the boundary conditions that
decide how many blocks survive multimodal padding.
"""

import unittest
from array import array
from types import SimpleNamespace

from sglang.srt.mem_cache.storage.asradix.block_hash_resume import (
    first_rewritten_token,
    reusable_hash_prefix,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

B = 4
HASHES = [10, 20, 30, 40]


def _ids(values):
    return array("q", values)


class TestReusableHashPrefix(CustomTestCase):
    def _prefix(self, padded, unpadded, first_rewritten, old=HASHES):
        return reusable_hash_prefix(
            padded=padded,
            unpadded=unpadded,
            old_hash_ids=old,
            first_rewritten=first_rewritten,
            block_size=B,
        )

    def test_keeps_only_whole_blocks_before_the_rewrite(self):
        unpadded = _ids(range(16))
        padded = _ids(range(16))
        padded[9] = 777
        # Rewrite at token 9 -> blocks 0 and 1 (tokens 0..7) are intact; block 2
        # holds token 9, so it must be rehashed even though 8 is unchanged.
        self.assertEqual(self._prefix(padded, unpadded, 9), [10, 20])

    def test_rewrite_inside_first_block_keeps_nothing(self):
        unpadded = _ids(range(16))
        padded = _ids(range(16))
        padded[2] = 777
        self.assertIsNone(self._prefix(padded, unpadded, 2))

    def test_length_change_keeps_nothing(self):
        self.assertIsNone(self._prefix(_ids(range(12)), _ids(range(16)), 9))

    def test_diverging_head_keeps_nothing(self):
        unpadded = _ids(range(16))
        padded = _ids(range(16))
        padded[1] = 999  # before first_rewritten: the caller's claim is wrong
        padded[9] = 777
        self.assertIsNone(self._prefix(padded, unpadded, 9))

    def test_prefix_is_capped_by_the_known_hashes(self):
        unpadded = _ids(range(40))
        padded = _ids(range(40))
        padded[39] = 777
        self.assertEqual(self._prefix(padded, unpadded, 39), HASHES)

    def test_no_old_hashes_or_no_rewrite_keeps_nothing(self):
        ids = _ids(range(16))
        self.assertIsNone(self._prefix(ids, ids, 9, old=[]))
        self.assertIsNone(self._prefix(ids, ids, 0))


class TestFirstRewrittenToken(CustomTestCase):
    def test_takes_the_earliest_offset_over_all_items(self):
        mm = SimpleNamespace(
            mm_items=[
                SimpleNamespace(offsets=[(30, 35), (12, 14)]),
                None,
                SimpleNamespace(offsets=[(20, 22)]),
            ]
        )
        self.assertEqual(first_rewritten_token(mm), 12)

    def test_no_items_or_no_offsets_means_no_rewrite(self):
        self.assertEqual(first_rewritten_token(None), 0)
        self.assertEqual(
            first_rewritten_token(SimpleNamespace(mm_items=[SimpleNamespace(offsets=[])])),
            0,
        )


if __name__ == "__main__":
    unittest.main()
