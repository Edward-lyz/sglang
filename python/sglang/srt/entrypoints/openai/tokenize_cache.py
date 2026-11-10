"""Encode a rendered chat prompt by reusing the token ids of repeated segments.

A multi-turn prompt re-sends its whole history, so the rendered text of turn N
is the text of turn N-1 plus a suffix. Tokenizing all of it costs ~110ms at
~66k tokens (measured, DeepSeek-V4.1 fast tokenizer) and runs on the tokenizer
worker's event loop, where it sits in front of the prefill. Splitting the text
immediately before each special token gives boundaries the tokenizer never
merges across, so each piece can be encoded once and reused by every later turn.

Gated by SGLANG_OPT_TOKENIZE_SEGMENT_CACHE. Cache hits must return exactly the
ids a single encode() would: the ids are the prefix-cache key, and a one-token
difference silently costs a cache hit instead of raising.
"""

from __future__ import annotations

import logging
import re
from collections import OrderedDict
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class SegmentTokenizeCache:
    """Per-process cache of `special-token-delimited segment -> token ids`."""

    def __init__(self, tokenizer, *, max_tokens: int, encode_kwargs: Optional[Dict] = None):
        self._tokenizer = tokenizer
        self._encode_kwargs = encode_kwargs or {}
        self._max_tokens = max_tokens
        self._entries: "OrderedDict[str, List[int]]" = OrderedDict()
        self._tokens = 0
        self._splitter = _build_splitter(tokenizer)
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        """False when the tokenizer exposes no special tokens to split on."""
        return self._splitter is not None

    def encode(self, text: str) -> List[int]:
        """Token ids of ``text``, encoding only the segments not seen before."""
        ids: List[int] = []
        for segment in self._splitter.split(text):
            if not segment:
                continue
            cached = self._entries.get(segment)
            if cached is None:
                self.misses += 1
                cached = self._tokenizer.encode(segment, **self._encode_kwargs)
                self._admit(segment, cached)
            else:
                self.hits += 1
                self._entries.move_to_end(segment)
            ids.extend(cached)
        return ids

    def _admit(self, segment: str, ids: List[int]) -> None:
        if len(ids) > self._max_tokens:
            return
        self._entries[segment] = ids
        self._tokens += len(ids)
        while self._tokens > self._max_tokens and self._entries:
            _, evicted = self._entries.popitem(last=False)
            self._tokens -= len(evicted)


def _build_splitter(tokenizer) -> Optional[re.Pattern]:
    """Match every position right before a special token, longest token first."""
    specials = set()
    for attr in ("all_special_tokens", "additional_special_tokens"):
        specials.update(token for token in getattr(tokenizer, attr, None) or () if token)
    added = getattr(tokenizer, "get_added_vocab", None)
    if callable(added):
        # Chat templates mark their role / tool markers as added special tokens;
        # keep only those also listed as special so ordinary added words, which
        # the tokenizer may merge with their neighbours, stay inside a segment.
        specials.update(token for token in added() if token in specials)
    if not specials:
        return None
    pattern = "|".join(re.escape(token) for token in sorted(specials, key=len, reverse=True))
    return re.compile(f"(?={pattern})")
