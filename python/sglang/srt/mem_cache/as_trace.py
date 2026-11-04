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
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TransferDigestCapture:
    """Collect snapshots emitted by one concrete AttentionStore operation."""

    def __init__(self) -> None:
        self._snapshots: list[dict] = []

    def __call__(self, snapshot: dict) -> None:
        self._snapshots.append(snapshot)

    def value(self) -> dict:
        """Return captured transfer snapshots in JSON-ready form."""
        return {"transfers": self._snapshots} if self._snapshots else {}


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
    extra_key = cfg.get("extra_key")
    input_ids = getattr(req, "origin_input_ids", ())
    payload = json.dumps(
        [model, _json_value(extra_key), _json_value(input_ids)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
        self._pending_writes = 0
        self._sealed = False
        self._finalizer = None
        self._finalized = False

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

    def retain_write(self) -> None:
        """Keep the context open until one AS write is retired."""
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot retain AS write after trace close")
            self._pending_writes += 1

    def release_write(self) -> None:
        """Retire one AS write and finalize if the context is ready."""
        finalizer = None
        with self._lock:
            if self._pending_writes <= 0:
                return
            self._pending_writes -= 1
            if self._sealed and self._pending_writes == 0 and not self._finalized:
                self._finalized = True
                self._closed = True
                finalizer = self._finalizer
        if finalizer is not None:
            finalizer(self.snapshot(), 0)

    def set_finalizer(self, finalizer) -> None:
        """Register the callback that persists the completed AS trace.

        ``finalizer(events, pending_writes)`` runs once; ``pending_writes`` is
        non-zero only when :meth:`force_finalize` cut the wait short.
        """
        run_now = False
        with self._lock:
            if self._finalizer is not None:
                return
            self._finalizer = finalizer
            if self._sealed and self._pending_writes == 0 and not self._finalized:
                self._finalized = True
                self._closed = True
                run_now = True
        if run_now:
            finalizer(self.snapshot(), 0)

    def seal_registered(self) -> None:
        """Seal the context after cache_finished_req submitted AS writes.

        A no-op until a finalizer is registered: release paths that requeue the
        request (optimistic prefill retry, retraction) must not end the trace.
        """
        finalizer = None
        with self._lock:
            if self._sealed or self._finalizer is None:
                return
            self._sealed = True
            if self._pending_writes == 0 and not self._finalized:
                self._finalized = True
                self._closed = True
                finalizer = self._finalizer
        if finalizer is not None:
            finalizer(self.snapshot(), 0)

    @property
    def finalized(self) -> bool:
        """Whether the finalizer already ran."""
        with self._lock:
            return self._finalized

    def force_finalize(self) -> bool:
        """Finalize now, even with AS writes still pending (online timeout).

        The finalizer receives the number of writes that had not retired, so
        the payload can report an incomplete AS trace instead of hanging.
        """
        with self._lock:
            if self._finalized or self._finalizer is None:
                return False
            self._finalized = True
            self._closed = True
            finalizer = self._finalizer
            pending_writes = self._pending_writes
        finalizer(self.snapshot(), pending_writes)
        return True


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

    def new_digest_capture(self) -> TransferDigestCapture:
        """Create an operation-local sink for transport-layer digest snapshots."""
        return TransferDigestCapture()


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
