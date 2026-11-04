"""Per-request prefix-cache trace for multimodal analysis, gated by SGLANG_LOG_MM_CACHE_TRACE.

One line per request at prefill finish: prompt and cached lengths, where the
first media span starts, and a short id of every media item's hash. Comparing
the ids of one image across turns of a conversation shows whether its pad
tokens, and so the prefix-cache key past it, stay stable. Never logs content.
"""

from __future__ import annotations

import logging

from sglang.srt.environ import envs


def maybe_log_mm_cache_trace(logger: logging.Logger, req) -> None:
    """Emit the trace line of a request whose prefill just finished."""
    if not envs.SGLANG_LOG_MM_CACHE_TRACE.get():
        return
    mm = req.multimodal_inputs
    items = [item for item in (mm.mm_items if mm is not None else []) if item is not None]
    spans = [(item.offsets[0][0], item) for item in items if item.offsets]
    # hash32@start-end per item, in prompt order.
    ids = ",".join(
        f"{(item.hash or 0) & 0xFFFFFFFF:08x}@{start}-{item.offsets[-1][1]}"
        for start, item in sorted(spans, key=lambda pair: pair[0])
    )
    logger.info(
        "MM cache trace rid=%s prompt=%d cached=%d n_mm=%d first_mm_offset=%d mm_ids=[%s]",
        req.rid,
        len(req.origin_input_ids),
        req.cached_tokens,
        len(items),
        min(start for start, _ in spans) if spans else -1,
        ids,
    )
