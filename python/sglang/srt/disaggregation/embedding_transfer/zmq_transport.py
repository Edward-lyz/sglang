"""
ZMQ-based embedding transport for EPD disaggregation.
"""

import json
import logging
import pickle
import time
from http import HTTPStatus
from typing import TYPE_CHECKING, Dict, List, Optional, Set

import torch
import zmq

from sglang.srt.disaggregation.embedding_transfer.base import (
    EmbeddingTransport,
    PullTarget,
    RequestKey,
    TransportResult,
)

if TYPE_CHECKING:
    from sglang.srt.disaggregation.embedding_data import EmbeddingData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ZMQ transport
# ---------------------------------------------------------------------------


class ZmqEmbeddingTransport(EmbeddingTransport):
    """Pull embedding data via ZMQ DEALER/ROUTER sockets.

    Uses a class-level shared socket pool so that all instances reuse one
    DEALER socket per encoder endpoint, avoiding fd exhaustion under high
    concurrency.

    Class-level state (shared across all instances):
        _ctx          – global ZMQ context (lazy-init)
        _poller       – single poller for all DEALER sockets
        _sockets      – endpoint -> DEALER socket
        _replies      – (request_id, task_id) -> TransportResult (reply buffer)
        _last_recv    – endpoint -> timestamp of last successful recv
        _pending_cnt  – endpoint -> number of in-flight requests

    Instance-level state (per WaitingImageRequest):
        _pending      – (request_id, task_id) -> endpoint (this instance's requests)
    """

    # ---- Class-level shared state ----
    _ctx: Optional[zmq.Context] = None
    _poller: Optional[zmq.Poller] = None
    _sockets: Dict[str, zmq.Socket] = {}
    _replies: Dict[RequestKey, "TransportResult"] = {}
    _last_recv: Dict[str, float] = {}
    _pending_cnt: Dict[str, int] = {}
    _settled: Set[RequestKey] = set()  # keys already settled by cleanup(), skip in _drain()
    STALE_TIMEOUT = 120  # seconds

    def __init__(self):
        self._pending: Dict[RequestKey, str] = {}  # (request_id, task_id) -> endpoint

    # ---- Class-level socket management ----

    @classmethod
    def _ensure_ctx(cls):
        if cls._ctx is None:
            cls._ctx = zmq.Context.instance()
            cls._poller = zmq.Poller()

    @classmethod
    def _get_dealer(cls, endpoint: str) -> zmq.Socket:
        """Get or create a DEALER socket for *endpoint*."""
        cls._ensure_ctx()
        sock = cls._sockets.get(endpoint)
        if sock is None:
            sock = cls._ctx.socket(zmq.DEALER)
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.RECONNECT_IVL, 500)
            sock.setsockopt(zmq.RECONNECT_IVL_MAX, 5000)
            sock.connect(endpoint)
            cls._sockets[endpoint] = sock
            cls._poller.register(sock, zmq.POLLIN)
            cls._last_recv[endpoint] = time.time()
            cls._pending_cnt[endpoint] = 0
            logger.info("Created DEALER socket for %s", endpoint)
        return sock

    @classmethod
    def _close_ep(cls, endpoint: str):
        """Close and remove a DEALER socket for *endpoint*."""
        sock = cls._sockets.pop(endpoint, None)
        if sock is None:
            return
        try:
            cls._poller.unregister(sock)
        except Exception:
            pass
        sock.close(linger=0)
        cls._last_recv.pop(endpoint, None)
        cls._pending_cnt.pop(endpoint, None)
        logger.info("Closed stale DEALER socket for %s", endpoint)

    @classmethod
    def _recycle_stale(cls):
        """Close sockets idle for > STALE_TIMEOUT with no pending requests."""
        now = time.time()
        for ep in list(cls._sockets):
            idle = now - cls._last_recv.get(ep, 0)
            if idle > cls.STALE_TIMEOUT and cls._pending_cnt.get(ep, 0) == 0:
                cls._close_ep(ep)

    @classmethod
    def close_all(cls):
        """Close every DEALER socket. Call on process shutdown."""
        for ep in list(cls._sockets):
            cls._close_ep(ep)
        cls._replies.clear()
        cls._settled.clear()

    # ---- Class-level drain: collect replies from all sockets ----

    @classmethod
    def _drain(cls):
        """Non-blocking: read all available replies into *_replies*."""
        if not cls._sockets:
            return
        try:
            events = dict(cls._poller.poll(timeout=0))
        except Exception:
            return

        for endpoint, sock in list(cls._sockets.items()):
            if not (events.get(sock, 0) & zmq.POLLIN):
                continue
            while True:
                try:
                    # Frame layout: [empty, task_id_bytes, meta, (opt)buffer]
                    parts = sock.recv_multipart(flags=zmq.NOBLOCK, copy=False)
                except zmq.Again:
                    break
                except Exception as e:
                    logger.warning("DEALER recv error on %s: %s", endpoint, e)
                    break
                # Validate frame count: [empty, task_id, req_id, meta, (opt)buffer]
                if len(parts) < 4:
                    logger.warning(
                        "Malformed DEALER reply on %s: expected >=4 frames, got %d",
                        endpoint, len(parts),
                    )
                    continue
                task_id = bytes(parts[1]).decode()
                req_id = bytes(parts[2]).decode() or task_id  # fallback for old server
                key: RequestKey = (req_id, task_id)
                # Skip late replies for already-settled requests
                if key in cls._settled:
                    cls._settled.discard(key)
                    logger.debug("Ignoring late reply for settled key=%s", key)
                    continue
                result = cls._parse_response(task_id, parts[3:])
                cls._replies[key] = result
                cls._last_recv[endpoint] = time.time()
                cls._pending_cnt[endpoint] = max(
                    0, cls._pending_cnt.get(endpoint, 1) - 1
                )

        cls._recycle_stale()

    # ---- EmbeddingTransport interface ----

    def send(self, pull_target: PullTarget) -> bool:
        """Send a pull request over the shared DEALER socket for the endpoint.

        Always returns True: the send is queued by ZMQ and never blocks on a
        buffer pool, so there is nothing for the caller to retry.
        """
        endpoint = pull_target.endpoint
        sock = self._get_dealer(endpoint)
        payload = json.dumps({
            "req_id": pull_target.request_id,
            "task_id": pull_target.task_id,
            "transfer_type": pull_target.transfer_type,
            "image_url": pull_target.image_url,
            "modality": pull_target.modality,
        }).encode()
        # DEALER → ROUTER: [empty_frame, payload]
        sock.send_multipart([b"", payload])
        key: RequestKey = (pull_target.request_id, pull_target.task_id)
        self._pending[key] = endpoint
        cls = self.__class__
        cls._pending_cnt[endpoint] = cls._pending_cnt.get(endpoint, 0) + 1
        return True

    def poll(self) -> List[TransportResult]:
        """Drain the shared poller, then return this instance's settled replies.

        Non-blocking; returns an empty list when nothing has landed yet.
        """
        self._drain()
        results: List[TransportResult] = []
        for key in list(self._pending):
            reply = self.__class__._replies.pop(key, None)
            if reply is not None:
                results.append(reply)
                del self._pending[key]
        return results

    def cleanup(self) -> None:
        """Release instance pending tracking; shared sockets stay open."""
        cls = self.__class__
        for key, ep in self._pending.items():
            cls._pending_cnt[ep] = max(0, cls._pending_cnt.get(ep, 1) - 1)
            cls._settled.add(key)  # Mark as settled to prevent double-decrement in _drain()
        self._pending.clear()

    # ---- Response parsing (reused from original) ----

    @staticmethod
    def _parse_response(task_id: str, parts) -> TransportResult:
        """Parse ZMQ multipart response into TransportResult."""
        try:
            recv_obj: EmbeddingData = pickle.loads(parts[0])
        except Exception as e:
            return TransportResult(
                task_id=task_id,
                error_msg=f"Failed to decode zmq response for {task_id}: {e}",
                error_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

        if getattr(recv_obj, "error_msg", None) is not None:
            return TransportResult(
                task_id=task_id,
                error_msg=recv_obj.error_msg,
                error_code=recv_obj.error_code,
            )

        if len(parts) < 2:
            return TransportResult(
                task_id=task_id,
                error_msg=f"Invalid embedding payload for {task_id}",
                error_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

        try:
            buf = parts[1].buffer if hasattr(parts[1], "buffer") else parts[1]
            embedding = torch.frombuffer(buf, dtype=recv_obj.dtype).reshape(
                recv_obj.shape
            )
        except Exception as e:
            return TransportResult(
                task_id=task_id,
                error_msg=f"Failed to parse embedding buffer for {task_id}: {e}",
                error_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

        # EmbeddingData is the pickled wire type; its field is image_grid_dim.
        grid_thw = recv_obj.image_grid_dim
        if not isinstance(grid_thw, torch.Tensor):
            grid_thw = torch.tensor(grid_thw)

        return TransportResult(
            task_id=task_id,
            embedding=embedding,
            grid_thw=grid_thw,
        )
