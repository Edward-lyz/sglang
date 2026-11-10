"""Low-overhead, request-scoped tracing for AttentionStore operations.

Responsibilities are split across three pieces:

* :func:`build_unique_trace_id` — deterministic request id, generated upstream
  (at ``Req`` init, into ``TraceReq.unique_trace_id``), not in the recorder.
* :class:`ASCacheTraceRecorder` — generates AS events (including MD5 collection);
  it does **no** file IO and does not compute the unique id.
* :class:`ASTraceContext` — the per-request in-memory event buffer. It only
  collects events (thread-safe); JSON/JSONL writing, rank info, and online vs
  dump decisions are all owned by ``TraceManager``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import torch
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    return str(value)


def hash_ids(tokens: Any) -> list:
    """Extract hash ids without depending on a particular SDK Tokens version."""
    if tokens is None:
        return []
    values = getattr(tokens, "hash_ids", tokens)
    if values is None:
        return []
    if hasattr(values, "tolist"):
        values = values.tolist()
    try:
        return [_json_value(v) for v in values]
    except TypeError:
        return [_json_value(values)]


def build_unique_trace_id(req: Any) -> str:
    """Build a deterministic id from the final tokenized request.

    Called at ``Req`` init time so downstream components (ASCache, TraceManager)
    share the same id instead of the recorder minting its own.
    """
    cfg = getattr(req, "trace_config", None) or {}
    model = cfg.get("model_identity", "")
    extra_key = getattr(cfg, "extra_key", None)
    input_ids = getattr(req, "origin_input_ids", ())
    payload = json.dumps(
        [model, _json_value(extra_key), _json_value(input_ids)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_md5(tensor: Any) -> str:
    """Compute an MD5 digest of a tensor slice's physical contents."""
    data = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.md5(data).hexdigest()


def token_page_md5(buf: Any, page_id: int, page_size: int) -> str:
    """Compute MD5 for one page stored as ``page_size`` token rows."""
    row_start = page_id * page_size
    return _tensor_md5(buf[row_start : row_start + page_size])


def page_row_md5(buf: Any, page_id: int) -> str:
    """Compute MD5 for one page stored in a single page-indexed row."""
    return _tensor_md5(buf[page_id])


def _swa_window_md5(kv_cache: Any, buf: Any, page_id: int, write: bool) -> str:
    """MD5 of one SWA page row, normalized to the swa_win-token window region.
    """
    row = buf[page_id].detach().cpu().numpy().tobytes()
    if write:
        return hashlib.md5(row).hexdigest()
    swa_pool = getattr(kv_cache, "swa_kv_pool", None)
    bytes_per_token = getattr(swa_pool, "kv_cache_total_dim", None)
    win = getattr(getattr(kv_cache, "chunk_tail_swa_pool", None), "swa_win", None)
    if not bytes_per_token or not win:
        return hashlib.md5(row).hexdigest()
    start = win * bytes_per_token
    end = min(2 * win * bytes_per_token, len(row))
    return hashlib.md5(row[start:end]).hexdigest()


def collect_page_md5(
    cache: Any,
    data_ids: Iterable[int],
    write: bool = False,
    layers: Optional[list[int]] = None,
) -> dict:
    """Collect page digests from the active AsRadixCache buffer layout."""
    if cache is None or not hasattr(cache, "_get_layer_kv_tensors"):
        return {}
    try:
        page_ids = [int(page_id) for page_id in data_ids]
        available_layers = list(getattr(cache, "layer_ids", []))
        layers = [0] if layers is None else layers
        layers = [
            layer_id for layer_id in layers if layer_id in available_layers
        ]
        page_size = cache.page_size
        result = {"layers": {}}
        for layer_id in layers:
            key, value, value_is_page_indexed = cache._get_layer_kv_tensors(layer_id)
            item = {
                "key": [
                    token_page_md5(key, page_id, page_size) for page_id in page_ids
                ]
            }
            if value is not None:
                digest = page_row_md5 if value_is_page_indexed else token_page_md5
                item["value"] = [
                    digest(value, page_id)
                    if value_is_page_indexed
                    else digest(value, page_id, page_size)
                    for page_id in page_ids
                ]
            result["layers"][str(layer_id)] = item
        return result
    except Exception as error:
        logger.debug("AS trace: md5 collection failed: %s", error)
        return {}


class ASTraceContext:
    """Per-request AS trace session: thread-safe in-memory event buffer.

    Created at ``Req`` init with the upstream-computed ``unique_trace_id``
    (from ``TraceReq``). It only buffers events and carries no rank info, no
    paths, and does no file IO — ``TraceManager`` owns JSONL/manifest writing
    (dump mode) and the online payload (online mode), and injects rank info at
    finalize time. All mutating methods are thread-safe (``AS_READ``/``AS_MATCH``
    run on the scheduler thread while ``AS_WRITE`` runs on the AS write executor
    thread).
    """

    def __init__(
        self,
        req: Any,
        unique_trace_id: Optional[str] = None,
    ) -> None:
        self.req = req
        self.unique_trace_id = unique_trace_id
        self.rid = getattr(req, "rid", None)
        self._lock = threading.Lock()
        self._events: list = []
        self._closed = False

    def record(self, event: dict) -> None:
        """Append an event unless this trace context has been closed."""
        with self._lock:
            if self._closed:
                return
            self._events.append(event)

    def snapshot(self) -> list:
        """Return a point-in-time copy of the recorded events."""
        with self._lock:
            return list(self._events)

    def close(self) -> None:
        """Stop accepting new events for this trace context."""
        with self._lock:
            self._closed = True


class ASCacheTraceRecorder:
    """Generate AS trace events and push them to :class:`ASTraceContext`.

    No file IO, no rank info and no id generation: it builds the event dict
    (including MD5) and hands it to the context buffer. TraceManager resolves
    rank info and output mode at finalize time.
    """

    def __init__(
        self,
        req: Any,
        ctx: ASTraceContext,
        kv_cache: Any = None,
    ) -> None:
        self.req = req
        self.ctx = ctx
        self.kv_cache = kv_cache

    def record(self, event: str, **fields: Any) -> None:
        """Record an AS event with request metadata and optional fields."""
        item = {
            "event": event,
            "timestamp": time.time(),
            "unique_trace_id": self.ctx.unique_trace_id,
            "rid": getattr(self.req, "rid", None),
            **_json_value(fields),
        }
        self.ctx.record(item)

    def hash_ids(self, tokens: Any) -> list:
        """Extract JSON-compatible hash ids from a token container."""
        return hash_ids(tokens)

    def collect_page_md5(
        self, data_ids: Iterable[int], write: bool = False
    ) -> dict:
        """Collect page digests for the configured KV cache."""
        config = getattr(self.req, "trace_config", None) or {}
        as_config = config.get("as_cache", {})
        layers = as_config.get("layers") if isinstance(as_config, dict) else None
        if not isinstance(layers, list):
            layers = None
        return collect_page_md5(self.kv_cache, data_ids, write, layers)


def get_recorder(req: Any, kv_cache: Any = None) -> Optional[ASCacheTraceRecorder]:
    """Return the per-request AS recorder, creating/attaching it lazily.

    ``unique_trace_id`` is read from ``req.to_trace_req`` so normal trace and
    AS trace share the same id minted at ``Req`` init.
    """
    cfg = getattr(req, "trace_config", None) or {}
    if not cfg.get("enable_trace") or cfg.get("trace_type") not in (
        "as_cache",
        "all",
    ):
        return None
    ctx = getattr(req, "as_trace_context", None)
    if ctx is None:
        trace_req = getattr(req, "to_trace_req", None)
        unique_id = getattr(trace_req, "unique_trace_id", None) or build_unique_trace_id(
            req
        )
        ctx = ASTraceContext(req, unique_id)
        req.as_trace_context = ctx
    recorder = getattr(req, "_as_trace_recorder", None)
    if recorder is None:
        recorder = ASCacheTraceRecorder(req, ctx, kv_cache)
        req._as_trace_recorder = recorder
    return recorder
