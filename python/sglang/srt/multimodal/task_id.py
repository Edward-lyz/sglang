"""Media identity derived from the served model name and the media URL.

A media item is identified by ``md5(model_id + media_url)``, with
``f"{sample_fps:.2f}"`` appended for video. The item hash and its pad value come
from that identity, so the pad value is a pure function of the request and every
rank derives the same one without holding the encoded tensor.

The identity follows the URL, which is what the serving path sends to the
encoder: the same URL keeps the same identity across requests and across
processes, and two items that differ only in position keep their own value.
"""

from __future__ import annotations

import hashlib
from typing import Optional

VIDEO_FPS_DECIMALS = 2


def compute_media_task_id(
    model_id: str, media_url: str, sample_fps: Optional[float] = None
) -> str:
    """Return the media task id: md5(model_id + url [+ fps])."""
    key = model_id + media_url
    if sample_fps is not None:
        key += f"{sample_fps:.{VIDEO_FPS_DECIMALS}f}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def media_task_hash(model_id: str, media_url: str) -> int:
    """Task id for an image or video item, as the integer MultimodalDataItem keeps.

    The whole digest becomes one integer, the way the EPD receiver reads its
    ``int(task_result["task_id"], 16)``. ``MultimodalDataItem`` masks the value
    to 64 bits, so both paths keep the same hash and derive the same pad value.
    """
    return int(compute_media_task_id(model_id, media_url), 16)
