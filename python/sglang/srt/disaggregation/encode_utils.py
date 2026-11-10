"""Streaming batch-execution helpers for the factory encode workflow.

Trimmed to what EncodeWorkflow uses: ``normalize_transfer_type`` and the
``StreamExecutor`` sliding-window driver (plus its base/protocol). The K3
original also carried batch executors, a token-limit scheduler and RDMA/NIC
utilities that the New EPD encoder does not use; those are intentionally left
out of this port.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import dataclass
from typing import Dict, Generic, List, Optional, Protocol, Tuple, TypeVar

# Mapping from encoder_transfer_backend values to wire transfer_type values.
_TRANSFER_TYPE_MAP = {
    "zmq_to_scheduler": "zmq",
    "zmq_to_tokenizer": "zmq",
}


def normalize_transfer_type(backend: str) -> str:
    """Convert encoder_transfer_backend value to the wire transfer_type.

    ``zmq_to_scheduler`` and ``zmq_to_tokenizer`` are both mapped to ``"zmq"``;
    other values (e.g. ``"mooncake"``) are returned as-is.
    """
    return _TRANSFER_TYPE_MAP.get(backend, backend)


from sglang.srt.disaggregation.encode_collector import (  # noqa: E402
    BatchExecutorCollector,
)


class WithRidProto(Protocol):
    """Protocol for objects that carry a request ID."""

    rid: str

    def num_tokens(self) -> int:
        """Return num tokens in this req."""


class WithBathInfoProto(Protocol):
    """Protocol for objects that expose batch info (token count and batch size)."""

    def num_tokens(self) -> int:
        """Return the number of tokens in this result."""
        ...

    def batch_size(self) -> int:
        """Return the batch size of this result."""
        ...


ReqType = TypeVar("ReqType", bound=WithRidProto)
ResType = TypeVar("ResType", bound=WithBathInfoProto)


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class QueueEntry:
    """Queue entry."""

    req: ReqType
    future: Future
    enqueue_time: float


class StreamingImplProto(Protocol[ReqType, ResType]):
    """Streaming impl: submit one request at a time; schedule the next as soon
    as any completes. Suited to a thread/process pool where the executor keeps
    a sliding window instead of waiting for a whole batch.
    """

    name: str
    collector: BatchExecutorCollector

    def submit(self, req: ReqType) -> Future[ResType]:
        """Submit a single request and return a Future for its result."""
        ...


# PLACEHOLDER_BODY


class _BaseExecutor(Generic[ReqType, ResType]):
    """Common base for the executors: a bounded request queue + pending count."""

    def __init__(
        self, name: str, collector: BatchExecutorCollector, backlog: int = 100000
    ):
        """Initialize the base executor.

        Args:
            name: Name identifier for this executor (used in logging).
            collector: Prometheus metrics collector instance.
            backlog: Maximum queue capacity for pending requests.
        """
        self._name = name
        self._collector = collector
        self.stopped = False
        self.requests: queue.Queue[Optional[QueueEntry]] = queue.Queue(backlog)
        self._pending_num_tokens = 0
        self._pending_lock = threading.Lock()

    def __call__(self, req: ReqType) -> Future[ResType]:
        """Submit a single request to the executor queue.

        Args:
            req: The request object to enqueue.

        Returns:
            A Future that will contain the result once processed.

        Raises:
            RuntimeError: If the executor has been stopped.
        """
        if self.stopped:
            raise RuntimeError(f"batch executor {self._name} has been stopped!")
        future: Future[ResType] = Future()
        self.requests.put(
            QueueEntry(req=req, future=future, enqueue_time=time.monotonic()),
            block=False,
        )
        with self._pending_lock:
            self._pending_num_tokens += req.num_tokens()
        return future

    def stop(self):
        """Send a stop signal to the serve loop, allowing it to drain and exit."""
        self.requests.put(None)

    def serve(self):
        """Run the main serve loop. Must be implemented by subclasses."""
        raise NotImplementedError


class StreamExecutor(_BaseExecutor[ReqType, ResType]):
    """Streaming executor: submit requests one by one; as any completes, pull
    the next. The impl provides ``submit(req) -> Future`` and owns concurrency.
    """

    def __init__(
        self, impl: StreamingImplProto[ReqType, ResType], backlog: int = 100000
    ):
        """Initialize the stream executor.

        Args:
            impl: The streaming implementation that provides a submit() method.
            backlog: Maximum queue capacity for pending requests.
        """
        super().__init__(impl.name, impl.collector, backlog=backlog)
        self._impl = impl
        # impl_future -> (user_future, submit_time)
        self.in_flight: Dict[Future[ResType], Tuple[Future[ResType], float]] = {}

    def __get_reqs(self) -> List[QueueEntry]:
        """Drain every currently available request from the queue."""
        res: List[QueueEntry] = []

        while True:
            block = not (bool(res) or bool(self.in_flight) or self.stopped)
            try:
                entry = self.requests.get(block=block)
            except queue.Empty:
                break

            if entry is None:
                self.stopped = True
                logger.warning(
                    f"stream executor {self._name} recv a stop signal, "
                    "serve loop(thread) will exit soon!"
                )
                continue

            if entry.future.cancelled():
                logger.info(f"request {entry.req.rid} cancelled, skip it!")
                with self._pending_lock:
                    self._pending_num_tokens -= entry.req.num_tokens()
                continue

            res.append(entry)
        return res

    def serve(self):
        """Run the serve loop until stopped and all in-flight tasks complete."""
        while True:
            entries = self.__get_reqs()

            self._collector.set_queue_wait_num_reqs(self.requests.qsize())
            for entry in entries:
                self._collector.observe_queue_wait_seconds(
                    time.monotonic() - entry.enqueue_time
                )
                logger.debug(f"{self._name} submit {entry.req.rid}")
                impl_fut = self._impl.submit(entry.req)
                self.in_flight[impl_fut] = (entry.future, time.monotonic())

            if not self.in_flight:
                if self.stopped:
                    logger.info(
                        f"stream executor {self._name} stopped, exiting serve loop."
                    )
                    return
                continue

            done, _ = wait(self.in_flight.keys(), return_when=FIRST_COMPLETED)

            for impl_fut in done:
                user_fut, start_time = self.in_flight.pop(impl_fut)
                self._collector.observe_run_batch_seconds(time.monotonic() - start_time)
                try:
                    result = impl_fut.result()
                    if not user_fut.done():
                        user_fut.set_result(result)
                except Exception as e:
                    logger.error(f"{self._name} task failed: {e}", exc_info=True)
                    if not user_fut.done():
                        user_fut.set_exception(e)
