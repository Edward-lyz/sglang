"""The segment tokenize cache must return exactly what a plain encode returns.

Its output is the prefix-cache key and the AttentionStore block-hash input, so
a dropped, duplicated or stale segment does not raise -- it silently changes
the key and costs cache hits. These cases pin the three ways that can happen:
text lost at a split boundary, a repeat served from the cache, and a repeat
served after eviction.
"""

import unittest

from sglang.srt.entrypoints.openai.tokenize_cache import SegmentTokenizeCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SPECIALS = ["<|a|>", "<|bb|>"]


class _CharTokenizer:
    """One id per character, so concatenating segment encodes is exact."""

    all_special_tokens = SPECIALS

    def __init__(self):
        self.calls = 0

    def encode(self, text):
        self.calls += 1
        return [ord(ch) for ch in text]


def _cache(max_tokens=10_000):
    return SegmentTokenizeCache(_CharTokenizer(), max_tokens=max_tokens)


class TestSegmentTokenizeCache(CustomTestCase):
    def test_matches_plain_encode_across_boundaries(self):
        cache = _cache()
        for text in (
            "",
            "plain",
            "<|a|>",
            "<|a|>head<|bb|>tail",
            "head<|a|><|a|>mid<|bb|>",
            # A special token also appearing inside a longer run of text.
            "x<|bb|>y<|bb|><|a|>z",
        ):
            self.assertEqual(cache.encode(text), [ord(ch) for ch in text], text)

    def test_repeat_is_served_from_cache_and_is_identical(self):
        cache = _cache()
        text = "<|a|>system<|bb|>turn one"
        first = cache.encode(text)
        calls_after_first = cache._tokenizer.calls
        second = cache.encode(text + "<|bb|>turn two")
        self.assertEqual(first, [ord(ch) for ch in text])
        self.assertEqual(second, [ord(ch) for ch in text + "<|bb|>turn two"])
        # Only the new suffix segment was encoded the second time.
        self.assertEqual(cache._tokenizer.calls, calls_after_first + 1)

    def test_eviction_keeps_results_correct(self):
        # Budget fits one short segment, so every call evicts the previous one.
        cache = _cache(max_tokens=6)
        text = "<|a|>aaaa<|bb|>bbbb"
        expected = [ord(ch) for ch in text]
        for _ in range(3):
            self.assertEqual(cache.encode(text), expected)

    def test_no_specials_disables_the_cache(self):
        class _Bare(_CharTokenizer):
            all_special_tokens = []

        cache = SegmentTokenizeCache(_Bare(), max_tokens=100)
        self.assertFalse(cache.enabled)


if __name__ == "__main__":
    unittest.main()
