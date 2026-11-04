"""Reuse the unchanged head of a request's AttentionStore block hashes.

TokenizerManager hashes the unpadded prompt; the scheduler rehashes after
multimodal padding rewrites the placeholder positions. The hashes are a chain,
so every block before the first rewritten token keeps its value, and only the
tail has to be hashed again. Gated by SGLANG_OPT_AS_BLOCK_HASH_RESUME.
"""

from __future__ import annotations

from typing import Optional, Sequence


def reusable_hash_prefix(
    *,
    padded,
    unpadded,
    old_hash_ids: Optional[Sequence[int]],
    first_rewritten: int,
    block_size: int,
) -> Optional[Sequence[int]]:
    """The leading block hashes of ``padded`` that ``old_hash_ids`` already holds.

    None means nothing is reusable (hash the whole prompt). The returned prefix
    is only as long as the bytes that provably match: padding keeps the length,
    so a length change means the sequences diverge everywhere.
    """
    if not old_hash_ids or block_size <= 0 or first_rewritten <= 0:
        return None
    if padded is None or unpadded is None or len(padded) != len(unpadded):
        return None
    blocks = min(first_rewritten // block_size, len(old_hash_ids))
    if blocks <= 0:
        return None
    tokens = blocks * block_size
    try:
        head, old_head = memoryview(padded)[:tokens], memoryview(unpadded)[:tokens]
    except TypeError:
        return None
    if head.itemsize != old_head.itemsize or head != old_head:
        return None
    return old_hash_ids[:blocks]


def first_rewritten_token(multimodal_inputs) -> int:
    """Offset of the first token multimodal padding rewrites; 0 if none."""
    if multimodal_inputs is None:
        return 0
    starts = [
        start
        for item in multimodal_inputs.mm_items
        if item is not None
        for start, _end in item.offsets or ()
    ]
    return min(starts) if starts else 0
