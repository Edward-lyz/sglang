"""Pull embeddings from the factory encoder pool.

The factory protocol is pull-shaped: Prefill takes the endpoint out of each task
result's ``encoder_addr`` and asks the encoder for the embedding. That is the
opposite direction from New EPD, where the scheduler registers a receive URL and
the encoder pushes to it (``/scheduler_receive_url``, ``/send``). The factory
encoder pool serves neither of those endpoints, so this module implements the
pull side rather than reusing the push control flow.

Two transports, per the cross-team protocol:

- ``mooncake``: one ``POST {endpoint}/epd/transfer`` per task, giving the encoder
  our Mooncake session id and landing-buffer address. The HTTP response *is* the
  completion, so there is no separate completion message and no
  ``/scheduler_receive_meta_data`` round trip -- the geometry is already in the
  descriptor.
- ``zmq``: a DEALER socket per encoder endpoint. We send the request; the reply
  carries a pickled metadata frame plus the embedding bytes.

``rdma`` also appears in the protocol but is a different flow (it advertises the
encoder's own ``item_addr`` / ``cache_size`` to read *from*), and the factory
receiver rejects it as not implemented; so does this module.

Sizes come from the descriptor either way, which is what lets a rank size its
landing buffer before talking to the encoder at all.

Landing buffers are CPU pinned memory taken from one pre-registered pool, as on
the factory side: it leaves HBM to the KV cache, and the copy prefill makes when
it consumes the rows becomes a DMA. A full pool is backpressure, not an error --
the caller retries on a later tick.
"""

from __future__ import annotations

import json
import logging
import pickle
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Dict, List, Optional

import torch
import zmq

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
}


def parse_torch_dtype(value) -> torch.dtype:
    """Parse a protocol dtype string; the wire uses both spellings."""
    if isinstance(value, torch.dtype):
        return value
    return _TORCH_DTYPE_MAP.get(str(value), torch.bfloat16)


class EmbeddingLandingPool:
    """One pre-registered CPU pinned pool for every mooncake EPD pull.

    Registering per request would put an MR create/destroy on the critical path,
    so the whole pool is registered with the embedding engine once and requests
    take page-aligned slices out of it.

    ``acquire`` returning None means "full right now" -- the caller keeps the
    request pending and retries, which is the pool doubling as backpressure.
    A request that could never fit raises instead, since retrying cannot help.
    """

    _instance: Optional["EmbeddingLandingPool"] = None
    _init_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "EmbeddingLandingPool":
        """The process-wide pool, created and registered on first use."""
        if cls._instance is not None:
            return cls._instance
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = cls(
                    int(envs.SGLANG_EPD_PINNED_POOL_GB.get() * (1024**3)),
                    envs.SGLANG_EPD_PINNED_PAGE_SIZE_MB.get() * 1024 * 1024,
                )
            return cls._instance

    @classmethod
    def get_instance_if_exists(cls) -> Optional["EmbeddingLandingPool"]:
        """The pool only for ranks that pulled; never allocate it lazily here."""
        return cls._instance

    def __init__(self, pool_bytes: int, page_size: int):
        """Allocate the pinned pool and register it with the embedding engine."""
        from sglang.srt.disaggregation.gpu_buffer_pool import GpuBufferPool

        self._pool = GpuBufferPool(
            pool_bytes, page_size, device="cpu", pin_memory=True
        )
        engine = self._engine()
        if engine is None:
            raise RuntimeError(
                "mooncake EPD needs the embedding transfer engine; set "
                "SGLANG_EMBEDDING_IB_DEVICE so it is initialized at startup"
            )
        engine.register(self._pool.data_ptr, self._pool.total_bytes)
        logger.info(
            "EmbeddingLandingPool: registered %.1f MB pinned host memory "
            "(%d pages x %.1f MB)",
            self._pool.total_bytes / 1e6,
            self._pool.num_pages,
            page_size / 1e6,
        )

    @staticmethod
    def _engine():
        """The Mooncake embedding engine, or None when it was never started."""
        from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
            get_mooncake_embedding_engine,
        )

        return get_mooncake_embedding_engine()

    @property
    def session_id(self) -> str:
        """Mooncake session the encoder writes to when it targets this pool."""
        return self._engine().session_id

    def acquire(self, byte_size: int) -> Optional[int]:
        """Page-aligned slice for ``byte_size``, or None while the pool is full."""
        return self._pool.try_allocate(byte_size)

    def release(self, address: int) -> None:
        """Hand one region back; only safe once no transfer can still write it."""
        self._pool.free(address)

    def used_bytes(self) -> int:
        """Bytes currently reserved by in-flight or not-yet-consumed pulls."""
        return self._pool.used_bytes()

    @property
    def size_bytes(self) -> int:
        """Total bytes of the registered pool."""
        return self._pool.total_bytes

    def view(self, address: int, byte_size: int, dtype: torch.dtype):
        """Zero-copy typed view of one region; no clone, so the pool bounds
        how much host memory in-flight embeddings can hold."""
        offset = address - self._pool.data_ptr
        return self._pool._buf[offset : offset + byte_size].view(dtype)


@dataclass
class PullTarget:
    """One item to fetch, built from a task result's transport descriptor."""

    task_id: str
    request_id: str
    part_idx: int
    num_parts: int
    transfer_type: str
    endpoint: str
    shape: List[int]
    dtype: str
    byte_size: int
    modality: str = "image"
    image_url: Optional[str] = None
    sent: bool = False


@dataclass
class TransportResult:
    """Outcome of one pull. ``embedding`` is None on the RDMA path, where the
    rows landed directly in the caller's buffer."""

    task_id: str
    part_idx: int
    embedding: Optional[torch.Tensor] = None
    grid_thw: Optional[torch.Tensor] = None
    error_msg: Optional[str] = None
    error_code: Optional[int] = None

    @property
    def success(self) -> bool:
        """A pull succeeded when it reported no error."""
        return self.error_msg is None


def descriptor_to_pull_target(descriptor, request_id: str) -> PullTarget:
    """Build a pull target from one normalized transport descriptor."""
    rows = int(descriptor["embedding_len"])
    dim = int(descriptor["embedding_dim"])
    dtype = str(descriptor["embedding_dtype"])
    byte_size = int(descriptor["embedding_size"]) or (
        rows * dim * torch.empty(0, dtype=parse_torch_dtype(dtype)).element_size()
    )
    return PullTarget(
        task_id=str(descriptor["task_id"] or descriptor["req_id"]),
        request_id=request_id,
        part_idx=int(descriptor["part_idx"]),
        num_parts=int(descriptor["num_parts"]),
        transfer_type=descriptor["transfer_type"],
        endpoint=f"{descriptor['ip']}:{descriptor['http_port']}",
        shape=[rows, dim],
        dtype=dtype,
        byte_size=byte_size,
        modality=descriptor["modality"].name.lower(),
        image_url=descriptor.get("image_url"),
    )


class FactoryZmqTransport:
    """Pull embeddings over DEALER sockets, one per encoder endpoint.

    Sockets are shared process-wide: a request-scoped socket would burn an fd
    per in-flight image, and reconnecting per request would lose the encoder's
    queued reply. ``_settled`` drops replies for requests that already finished,
    which matters precisely because the socket outlives them.
    """

    _ctx: Optional[zmq.Context] = None
    _poller: Optional[zmq.Poller] = None
    _sockets: Dict[str, "zmq.Socket"] = {}
    _replies: Dict[tuple, TransportResult] = {}
    _settled: set = set()
    # Bounds the settled-key set; see cleanup().
    _MAX_SETTLED = 4096

    def __init__(self):
        """Track only this request's keys; the sockets are shared."""
        # (request_id, task_id) -> part_idx, for this request only.
        self._pending: Dict[tuple, int] = {}

    @classmethod
    def _dealer(cls, endpoint: str):
        """The shared DEALER socket for one encoder endpoint, created on demand."""
        if cls._ctx is None:
            cls._ctx = zmq.Context.instance()
            cls._poller = zmq.Poller()
        sock = cls._sockets.get(endpoint)
        if sock is None:
            sock = cls._ctx.socket(zmq.DEALER)
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.RECONNECT_IVL, 500)
            sock.setsockopt(zmq.RECONNECT_IVL_MAX, 5000)
            sock.connect(f"tcp://{endpoint}")
            cls._sockets[endpoint] = sock
            cls._poller.register(sock, zmq.POLLIN)
            logger.info("Created encoder DEALER socket for tcp://%s", endpoint)
        return sock

    def send(self, target: PullTarget) -> bool:
        """Ask one encoder for this item; the reply carries the rows."""
        payload = json.dumps(
            {
                "req_id": target.request_id,
                "task_id": target.task_id,
                "transfer_type": target.transfer_type,
                "image_url": target.image_url,
                "modality": target.modality,
            }
        ).encode()
        # DEALER -> ROUTER prepends the empty delimiter frame.
        self._dealer(target.endpoint).send_multipart([b"", payload])
        self._pending[(target.request_id, target.task_id)] = target.part_idx
        return True

    @classmethod
    def _drain(cls) -> None:
        """Read every available reply into the shared buffer. Non-blocking."""
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
                    parts = sock.recv_multipart(flags=zmq.NOBLOCK, copy=False)
                except zmq.Again:
                    break
                except Exception as error:
                    logger.warning("DEALER recv failed on %s: %s", endpoint, error)
                    break
                # [empty, task_id, req_id, meta, embedding]
                if len(parts) < 4:
                    logger.warning(
                        "Malformed encoder reply on %s: %d frames", endpoint, len(parts)
                    )
                    continue
                task_id = bytes(parts[1]).decode()
                req_id = bytes(parts[2]).decode() or task_id
                key = (req_id, task_id)
                if key in cls._settled:
                    # The request is gone; dropping this keeps the shared socket
                    # from leaking its reply into an unrelated later request.
                    cls._settled.discard(key)
                    continue
                cls._replies[key] = cls._parse_reply(task_id, parts[3:])

    def poll(self) -> List[TransportResult]:
        """Return this request's landed replies; leave the rest pending."""
        self._drain()
        results = []
        for key in list(self._pending):
            reply = self._replies.pop(key, None)
            if reply is None:
                continue
            reply.part_idx = self._pending.pop(key)
            results.append(reply)
        return results

    def cleanup(self) -> List[Future]:
        """Forget this request's in-flight keys; the sockets stay open."""
        for key in self._pending:
            self._settled.add(key)
        self._pending.clear()
        # A settled key is only discarded when a late reply actually arrives,
        # which usually never happens, so the set has to be bounded here or it
        # grows for the lifetime of the process.
        if len(self._settled) > self._MAX_SETTLED:
            self._settled.clear()
        # ZMQ replies land in this process's own buffers, so nothing outside can
        # still be writing once the request is gone.
        return []

    @staticmethod
    def _parse_reply(task_id: str, parts) -> TransportResult:
        """Turn [meta, embedding] frames into a result.

        The metadata frame is a pickled object carrying shape/dtype/grid and,
        on failure, error_msg/error_code.
        """
        try:
            meta = pickle.loads(bytes(parts[0]))
        except Exception as error:
            return TransportResult(
                task_id=task_id,
                part_idx=0,
                error_msg=f"undecodable encoder reply for {task_id}: {error}",
                error_code=int(HTTPStatus.INTERNAL_SERVER_ERROR),
            )

        error_msg = getattr(meta, "error_msg", None) or (
            meta.get("error_msg") if isinstance(meta, dict) else None
        )
        if error_msg:
            error_code = getattr(meta, "error_code", None) or (
                meta.get("error_code") if isinstance(meta, dict) else None
            )
            return TransportResult(
                task_id=task_id,
                part_idx=0,
                error_msg=error_msg,
                error_code=error_code,
            )

        def field(name):
            """Read one metadata field, whichever shape the encoder pickled."""
            if isinstance(meta, dict):
                return meta.get(name)
            return getattr(meta, name, None)

        if len(parts) < 2:
            return TransportResult(
                task_id=task_id,
                part_idx=0,
                error_msg=f"encoder reply for {task_id} carried no embedding frame",
                error_code=int(HTTPStatus.INTERNAL_SERVER_ERROR),
            )
        try:
            buffer = parts[1].buffer if hasattr(parts[1], "buffer") else parts[1]
            # Clone: the ZMQ frame is recycled on the next recv.
            embedding = (
                torch.frombuffer(buffer, dtype=parse_torch_dtype(field("dtype")))
                .reshape(field("shape"))
                .clone()
            )
        except Exception as error:
            return TransportResult(
                task_id=task_id,
                part_idx=0,
                error_msg=f"unreadable embedding for {task_id}: {error}",
                error_code=int(HTTPStatus.INTERNAL_SERVER_ERROR),
            )

        grid = field("image_grid_thw")
        if grid is None:
            grid = field("image_grid_dim")
        if grid is not None and not isinstance(grid, torch.Tensor):
            grid = torch.as_tensor(grid)
        return TransportResult(
            task_id=task_id, part_idx=0, embedding=embedding, grid_thw=grid
        )


def build_epd_transfer_payload(
    target: PullTarget, session_id: str, buffer_address: int
) -> dict:
    """Request body for ``POST {endpoint}/epd/transfer``.

    Everything the encoder needs is in this one call -- session, destination
    address, byte count and geometry -- so unlike the New EPD push path there is
    no preceding metadata round trip.
    """
    return {
        "req_id": target.request_id,
        "task_id": target.task_id,
        # The encoder only echoes these back -- one task is one whole embedding
        # here, since pulls are deduplicated per task_id -- so report the
        # single-shard values the factory client sends rather than this item's
        # position, which would be meaningless to the encoder.
        "part_idx": 0,
        "num_parts": 1,
        "session_id": session_id,
        "buffer_address": buffer_address,
        "byte_size": target.byte_size,
        "dtype": target.dtype,
        "shape": list(target.shape),
        "image_url": target.image_url,
        "modality": target.modality,
    }


def epd_transfer_error(ack: dict) -> Optional[str]:
    """Error text when an ``/epd/transfer`` ack reports failure, else None."""
    if not isinstance(ack, dict):
        return f"encoder returned a non-object ack: {ack!r}"
    status = ack.get("status_code")
    if status is None or int(status) == int(HTTPStatus.OK):
        return None
    return f"encoder rejected the transfer ({status}): {ack.get('message')}"


class FactoryMooncakeTransport:
    """Pull embeddings by asking the encoder to RDMA into our buffer.

    The HTTP response to ``/epd/transfer`` is the completion signal, so a
    finished future means the rows have landed. Requests run on a shared thread
    pool because the scheduler loop must not block on them.
    """

    _pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix="epd-transfer")

    def __init__(self, session_id: str, timeout: float = 30.0):
        """Bind the pulls of one request to the session the encoder writes to."""
        self.session_id = session_id
        self.timeout = timeout
        self._pending: Dict[str, tuple] = {}

    def send(self, target: PullTarget, buffer_address: int) -> bool:
        """Ask one encoder to RDMA this item into ``buffer_address``."""
        payload = build_epd_transfer_payload(target, self.session_id, buffer_address)
        url = f"http://{target.endpoint}/epd/transfer"
        future = self._pool.submit(self._post, url, payload, self.timeout)
        self._pending[target.task_id] = (future, target, buffer_address)
        return True

    @staticmethod
    def _post(url: str, payload: dict, timeout: float) -> dict:
        """One blocking ``/epd/transfer`` call, run on the shared thread pool."""
        import requests

        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def poll(self) -> List[TransportResult]:
        """Return results for transfers whose HTTP call has returned."""
        if not self._pending:
            return []
        results, done = [], []
        for task_id, (future, target, _) in self._pending.items():
            if not future.done():
                continue
            done.append(task_id)
            results.append(self._resolve(task_id, future, target))
        for task_id in done:
            del self._pending[task_id]
        return results

    def cleanup(self) -> List[Future]:
        """Drop tracking and report transfers that could not be stopped.

        ``Future.cancel()`` does nothing to a call that already started, and the
        encoder writes into our buffer by RDMA regardless -- so a buffer whose
        transfer is still in flight must not be handed back yet. The returned
        futures are the ones the caller has to wait on before releasing it.
        """
        inflight = []
        for future, _, _ in self._pending.values():
            if not future.cancel():
                inflight.append(future)
        self._pending.clear()
        return inflight

    @staticmethod
    def _resolve(task_id: str, future: Future, target: PullTarget) -> TransportResult:
        """Turn a finished ``/epd/transfer`` call into a result for one item."""
        try:
            ack = future.result()
        except Exception as error:
            return TransportResult(
                task_id=task_id,
                part_idx=target.part_idx,
                error_msg=f"/epd/transfer failed for {task_id}: {error}",
                error_code=int(HTTPStatus.BAD_GATEWAY),
            )
        error_msg = epd_transfer_error(ack)
        if error_msg is not None:
            return TransportResult(
                task_id=task_id,
                part_idx=target.part_idx,
                error_msg=error_msg,
                error_code=int(HTTPStatus.BAD_GATEWAY),
            )
        # The rows are already in the caller's buffer; nothing to carry back
        # but the geometry needed to view them.
        return TransportResult(task_id=task_id, part_idx=target.part_idx)
