"""
Encoder-level metrics collector.

Centralizes all monitoring metrics for the disaggregated encoder (MMEncoder),
including mm_cache hit rate tracking and Prometheus metric definitions.
"""

# NOTE: `from __future__ import annotations` keeps the `Counter` / `Gauge`
# type hints below as strings, so they don't need `prometheus_client` imported
# at module load time.
from __future__ import annotations

import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from prometheus_client import Counter, Gauge, Histogram

# IMPORTANT: prometheus_client must NOT be imported at module top level.
# Its multiprocess/single-process `ValueClass` is frozen on first import based
# on whether `PROMETHEUS_MULTIPROC_DIR` is set. This module is pulled in eagerly
# via the scheduler import chain in the main process, *before*
# `set_prometheus_multiproc_dir()` runs, which would otherwise freeze the whole
# process into single-process mode and drop every main-process (tokenizer) metric
# from the /metrics scrape. So each collector imports prometheus_client lazily
# inside its __init__, matching the convention in observability/metrics_collector.py.


class CacheStats:
    """mm_cache hit rate counters at both request and byte granularity."""

    def __init__(self):
        """Start every hit/total counter at zero."""
        self.hit_count: int = 0
        self.total_count: int = 0
        self.hit_bytes: int = 0
        self.total_bytes: int = 0

    def record_hit(self, nbytes: int):
        """Record a cache hit."""
        self.hit_count += 1
        self.total_count += 1
        self.hit_bytes += nbytes
        self.total_bytes += nbytes

    def record_miss(self, nbytes: int):
        """Record a cache miss."""
        self.total_count += 1
        self.total_bytes += nbytes

    @property
    def req_hit_rate(self) -> float:
        """Request-level cache hit rate."""
        return self.hit_count / self.total_count if self.total_count > 0 else 0.0

    @property
    def byte_hit_rate(self) -> float:
        """Byte-level cache hit rate."""
        return self.hit_bytes / self.total_bytes if self.total_bytes > 0 else 0.0


class EncoderMetricsCollector:
    """Prometheus metrics collector for the disaggregated encoder.

    Emits the metric families of the 厂内 Grafana "Encoder监控" dashboard:
    request counters, embed-cache hit counters, the four pipeline stages
    (mm_loader / mm_process / mm_embed / mooncake_transfer) and the
    /epd/transfer endpoint latency.
    """

    def __init__(self, labels: Optional[Dict[str, str]] = None):
        """Build the encoder Prometheus metrics under an optional label set."""
        # Import lazily so ValueClass is chosen after PROMETHEUS_MULTIPROC_DIR is set.
        from prometheus_client import Counter, Gauge

        labels = labels or {}
        self._label_keys = list(labels.keys())
        self._label_vals = labels

        self.cache_stats: Dict[str, CacheStats] = defaultdict(CacheStats)

        # --- cache counters ---
        self.cache_hit_requests_total = Counter(
            name="sglang:encoder_cache_hit_requests_total",
            documentation="Total cache-hit encode requests.",
            labelnames=[*self._label_keys, "source"],
        )
        self.cache_total_requests_total = Counter(
            name="sglang:encoder_cache_total_requests_total",
            documentation="Total encode requests that queried the cache.",
            labelnames=[*self._label_keys, "source"],
        )
        self.cache_hit_bytes_total = Counter(
            name="sglang:encoder_cache_hit_bytes_total",
            documentation="Total embedding bytes served from cache.",
            labelnames=[*self._label_keys, "source"],
        )
        self.cache_total_bytes_total = Counter(
            name="sglang:encoder_cache_total_bytes_total",
            documentation="Total embedding bytes across all cache queries.",
            labelnames=[*self._label_keys, "source"],
        )

        # --- cache gauges ---
        self.cache_req_hit_rate = Gauge(
            name="sglang:encoder_cache_req_hit_rate",
            documentation="Request-level cache hit rate.",
            labelnames=[*self._label_keys, "source"],
            multiprocess_mode="mostrecent",
        )
        self.cache_byte_hit_rate = Gauge(
            name="sglang:encoder_cache_byte_hit_rate",
            documentation="Byte-level cache hit rate.",
            labelnames=[*self._label_keys, "source"],
            multiprocess_mode="mostrecent",
        )
        # Cache size gauges come from the community observability
        # EncoderMetricsCollector (sglang:encoder_cache_entries); registering
        # the same name twice raises in prometheus_client.

        # --- request counters ---
        self.num_requests_total = Counter(
            name="sglang:encoder_num_requests_total",
            documentation="Number of encode requests processed.",
            labelnames=self._label_keys,
        )
        self.num_aborted_requests_total = Counter(
            name="sglang:encoder_num_aborted_requests_total",
            documentation="Number of encode requests aborted.",
            labelnames=self._label_keys,
        )
        self.num_failed_requests_total = Counter(
            name="sglang:encoder_num_failed_requests_total",
            documentation="Number of encode requests failed.",
            labelnames=self._label_keys,
        )

        self._inc_counter(self.num_requests_total, 0)
        self._inc_counter(self.num_aborted_requests_total, 0)
        self._inc_counter(self.num_failed_requests_total, 0)
        # Export both K3 cache sources from start, so a dashboard shows 0
        # hits instead of "no data" until the first hit.
        for counter in (
            self.cache_hit_requests_total,
            self.cache_total_requests_total,
            self.cache_hit_bytes_total,
            self.cache_total_bytes_total,
        ):
            for source in ("workflow", "transfer"):
                self._inc_counter(counter, 0, source=source)

        # --- pipeline stages and /epd/transfer endpoint ---
        self.loader_stage = BatchExecutorCollector("mm_loader", labels=labels)
        self.process_stage = BatchExecutorCollector("mm_process", labels=labels)
        self.embed_stage = BatchExecutorCollector("mm_embed", labels=labels)
        self.transfer_stage = BatchExecutorCollector(
            "mooncake_transfer", labels=labels
        )
        self.transfer_endpoint = TransferEndpointCollector(labels=labels)

    # ---- helpers to handle optional labels ----

    def _inc_counter(self, counter: Counter, value: float = 1, **extra):
        """Increment a counter, handling optional labels."""
        all_labels = {**self._label_vals, **extra}
        if all_labels:
            counter.labels(**all_labels).inc(value)
        else:
            counter.inc(value)

    def _set_gauge(self, gauge: Gauge, value: float, **extra):
        """Set a gauge value, handling optional labels."""
        all_labels = {**self._label_vals, **extra}
        if all_labels:
            gauge.labels(**all_labels).set(value)
        else:
            gauge.set(value)

    # ---- public API ----

    def record_cache_hit(self, nbytes: int, source: str = "workflow"):
        """Record a cache hit event."""
        stats = self.cache_stats[source]
        stats.record_hit(nbytes)
        self._inc_counter(self.cache_hit_requests_total, source=source)
        self._inc_counter(self.cache_total_requests_total, source=source)
        self._inc_counter(self.cache_hit_bytes_total, nbytes, source=source)
        self._inc_counter(self.cache_total_bytes_total, nbytes, source=source)
        self._set_gauge(self.cache_req_hit_rate, stats.req_hit_rate, source=source)
        self._set_gauge(self.cache_byte_hit_rate, stats.byte_hit_rate, source=source)

    def record_cache_miss(self, nbytes: int, source: str = "workflow"):
        """Record a cache miss event."""
        stats = self.cache_stats[source]
        stats.record_miss(nbytes)
        self._inc_counter(self.cache_total_requests_total, source=source)
        self._inc_counter(self.cache_total_bytes_total, nbytes, source=source)
        self._set_gauge(self.cache_req_hit_rate, stats.req_hit_rate, source=source)
        self._set_gauge(self.cache_byte_hit_rate, stats.byte_hit_rate, source=source)

    def observe_one_finished_request(self):
        """Record one finished encode request."""
        self._inc_counter(self.num_requests_total)

    def observe_one_failed_request(self):
        """Record one failed encode request."""
        self._inc_counter(self.num_requests_total)
        self._inc_counter(self.num_failed_requests_total)

    def observe_one_aborted_request(self):
        """Record one aborted encode request."""
        self._inc_counter(self.num_requests_total)
        self._inc_counter(self.num_aborted_requests_total)


class TransferEndpointCollector:
    """Prometheus metrics for /epd/transfer endpoint end-to-end monitoring."""

    _LATENCY_BUCKETS = [
        0.001,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1,
        2,
        5,
        10,
        30,
        60,
    ]

    def __init__(self, labels: Optional[Dict[str, str]] = None):
        """Initialize transfer endpoint collector.

        Args:
            labels: Optional label dict attached to each metric.
        """
        # Import lazily so ValueClass is chosen after PROMETHEUS_MULTIPROC_DIR is set.
        from prometheus_client import Counter, Histogram

        labels = labels or {}
        self._label_keys = list(labels.keys())
        self._label_vals = labels

        self.total_seconds = Histogram(
            name="sglang:epd_transfer_total_seconds",
            documentation="End-to-end latency of /epd/transfer.",
            buckets=self._LATENCY_BUCKETS,
            labelnames=self._label_keys,
        )
        self.requests_total = Counter(
            name="sglang:epd_transfer_requests_total",
            documentation="Total /epd/transfer requests by status.",
            labelnames=[*self._label_keys, "status"],
        )

    def on_request_end(self, status: str, total_seconds: float):
        """Record end of /epd/transfer request."""
        if self._label_vals:
            self.requests_total.labels(**self._label_vals, status=status).inc()
            self.total_seconds.labels(**self._label_vals).observe(total_seconds)
        else:
            self.requests_total.labels(status=status).inc()
            self.total_seconds.observe(total_seconds)


class VitFetchCollector:
    """Prometheus metrics for vit-server task fetching."""

    _LATENCY_BUCKETS = [
        0.001,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1,
        2,
        5,
        10,
        30,
    ]
    _TASK_COUNT_BUCKETS = [0, 1, 2, 4, 8, 16, 32, 64]

    def __init__(self, labels: Optional[Dict[str, str]] = None):
        """Build the vit-fetch Prometheus metrics under an optional label set."""
        # Import lazily so ValueClass is chosen after PROMETHEUS_MULTIPROC_DIR is set.
        from prometheus_client import Gauge, Histogram

        labels = labels or {}
        self._label_keys = list(labels.keys())
        self._label_vals = labels

        self.fetch_tasks_count = Histogram(
            name="sglang:vit_fetch_tasks_count",
            documentation="Number of tasks fetched from vit-server per request.",
            buckets=self._TASK_COUNT_BUCKETS,
            labelnames=self._label_keys,
        )
        self.fetch_duration_seconds = Histogram(
            name="sglang:vit_fetch_duration_seconds",
            documentation="Latency of fetching tasks from vit-server in seconds.",
            buckets=self._LATENCY_BUCKETS,
            labelnames=self._label_keys,
        )
        self.free_tokens = Gauge(
            name="sglang:encoder_free_tokens",
            documentation="Available token budget for encoding (backpressure indicator).",
            labelnames=self._label_keys,
            multiprocess_mode="mostrecent",
        )

    def observe_task_count(self, task_count: int):
        """Record the number of tasks fetched in one poll."""
        if self._label_vals:
            self.fetch_tasks_count.labels(**self._label_vals).observe(task_count)
        else:
            self.fetch_tasks_count.observe(task_count)

    def observe_duration(self, duration: float):
        """Record the full poll cycle duration."""
        if self._label_vals:
            self.fetch_duration_seconds.labels(**self._label_vals).observe(duration)
        else:
            self.fetch_duration_seconds.observe(duration)

    def set_free_tokens(self, value: int):
        """Record current free token budget."""
        if self._label_vals:
            self.free_tokens.labels(**self._label_vals).set(value)
        else:
            self.free_tokens.set(value)


class BatchExecutorCollector:
    """Prometheus metrics collector for a batch executor stage."""

    def __init__(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Initialize Prometheus metrics collectors for a batch executor.

        Args:
            name: Metric name prefix used for all Prometheus metrics.
            labels: Optional label dict attached to each metric.
        """
        # Import lazily so ValueClass is chosen after PROMETHEUS_MULTIPROC_DIR is set.
        from prometheus_client import Gauge, Histogram

        labels = labels or {}
        self.serve_once_duration = Histogram(
            name=f"sglang:{name}_run_batch_seconds",
            documentation=f"Histogram of {name} run_batch execution time in seconds.",
            buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60],
            labelnames=labels.keys(),
        )
        self.serve_once_batch_size = Histogram(
            name=f"sglang:{name}_batch_size",
            documentation=f"Histogram of {name} batch size.",
            buckets=[1, 2, 4, 8, 16, 32, 64, 128],
            labelnames=labels.keys(),
        )
        self.serve_once_num_tokens = Histogram(
            name=f"sglang:{name}_batch_num_tokens",
            documentation=f"Histogram of {name} num_tokens per serve_once.",
            buckets=[64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
            labelnames=labels.keys(),
        )
        self.queue_wait_seconds = Histogram(
            name=f"sglang:{name}_queue_wait_seconds",
            documentation=f"Histogram of {name} per-request queue waiting time in seconds.",
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30],
            labelnames=labels.keys(),
        )
        self.queue_wait_num_reqs = Gauge(
            name=f"sglang:{name}_queue_wait_num_reqs",
            documentation=f"Number of requests waiting in {name} queue.",
            labelnames=labels.keys(),
            multiprocess_mode="livesum",
        )
        self.sched_bubble_sec = Histogram(
            name=f"sglang:{name}_sched_bubble_seconds",
            documentation=f"Histogram of {name} per-request sched bubble time in seconds.",
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30],
            labelnames=labels.keys(),
        )
        self.labels = labels
        # Backlog for queue_enter/queue_leave; callers run on worker threads.
        self._queue_lock = threading.Lock()
        self._queue_len = 0

    def observe_sched_bubble_seconds(self, cost: float):
        """Record scheduling bubble time."""
        if self.labels:
            self.sched_bubble_sec.labels(**self.labels).observe(cost)
        else:
            self.sched_bubble_sec.observe(cost)

    def observe_run_batch_seconds(self, cost: float):
        """Record batch execution time."""
        if self.labels:
            self.serve_once_duration.labels(**self.labels).observe(cost)
        else:
            self.serve_once_duration.observe(cost)

    def observe_batch_size(self, size: int):
        """Record batch size."""
        if self.labels:
            self.serve_once_batch_size.labels(**self.labels).observe(size)
        else:
            self.serve_once_batch_size.observe(size)

    def observe_num_tokens(self, num_tokens: int):
        """Record number of tokens."""
        if self.labels:
            self.serve_once_num_tokens.labels(**self.labels).observe(num_tokens)
        else:
            self.serve_once_num_tokens.observe(num_tokens)

    def observe_queue_wait_seconds(self, cost: float):
        """Record queue wait time."""
        if self.labels:
            self.queue_wait_seconds.labels(**self.labels).observe(cost)
        else:
            self.queue_wait_seconds.observe(cost)

    def set_queue_wait_num_reqs(self, num: int):
        """Set queue wait number of requests."""
        if self.labels:
            self.queue_wait_num_reqs.labels(**self.labels).set(num)
        else:
            self.queue_wait_num_reqs.set(num)

    def observe_batch(self, cost: float, batch_size: int, num_tokens: int):
        """Record one executed batch: run time, batch size and token (or byte) count."""
        self.observe_run_batch_seconds(cost)
        if batch_size and num_tokens:
            self.observe_batch_size(batch_size)
            self.observe_num_tokens(num_tokens)

    def queue_enter(self, num_reqs: int = 1):
        """Count requests that start waiting in this stage queue (thread-safe)."""
        with self._queue_lock:
            self._queue_len += num_reqs
            self.set_queue_wait_num_reqs(self._queue_len)

    def queue_leave(self, num_reqs: int = 1, wait_seconds: Optional[float] = None):
        """Count requests that leave the queue; observe their wait when known."""
        with self._queue_lock:
            self._queue_len -= num_reqs
            self.set_queue_wait_num_reqs(self._queue_len)
        if wait_seconds is not None:
            for _ in range(num_reqs):
                self.observe_queue_wait_seconds(wait_seconds)


class StageReadyMark:
    """Track one encode batch from "inputs ready" to "stage started".

    Worker threads call ``mark`` when a preprocess job of the batch finishes;
    the first call puts the batch in the stage queue and the last call sets
    the ready time. The event loop calls ``start`` right before the stage runs
    and ``discard`` when the batch fails first. A blocking forward also blocks
    the event loop, so only a thread-side ``mark`` can count waiting batches.
    """

    def __init__(self, stage: BatchExecutorCollector, num_reqs: int):
        """Bind the mark to one stage collector and one batch size."""
        self._stage = stage
        self._num_reqs = num_reqs
        self._lock = threading.Lock()
        self._ready_at: Optional[float] = None
        self._queued = False
        self._closed = False

    def mark(self):
        """Record that one input job of the batch finished."""
        with self._lock:
            if self._closed:
                return
            self._ready_at = time.perf_counter()
            if self._queued:
                return
            self._queued = True
        self._stage.queue_enter(self._num_reqs)

    def start(self):
        """Leave the queue and observe the wait since the batch became ready."""
        now = time.perf_counter()
        with self._lock:
            if self._closed:
                return
            self._closed = True
            # No worker-thread job (inline preprocess): ready means now.
            wait = now - self._ready_at if self._ready_at is not None else 0.0
            queued = self._queued
        if queued:
            self._stage.queue_leave(self._num_reqs, wait_seconds=wait)
            return
        for _ in range(self._num_reqs):
            self._stage.observe_queue_wait_seconds(wait)

    def discard(self):
        """Drop the batch from the queue without a wait sample (failure path)."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            queued = self._queued
        if queued:
            self._stage.queue_leave(self._num_reqs)


class QueueMeteredExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor that reports its queue to a stage collector.

    ``stage`` (set when metrics are on) gets the backlog and the submit ->
    start wait of every job. With ``ready_mark_var`` each finished job also
    marks the StageReadyMark held by that ContextVar at submit time.
    ``submit`` runs in the caller's context, also for ``loop.run_in_executor``.
    """

    def __init__(
        self,
        *args,
        ready_mark_var: Optional[ContextVar[Optional[StageReadyMark]]] = None,
        **kwargs,
    ):
        """Create the pool; ``stage`` stays None until metrics are attached."""
        super().__init__(*args, **kwargs)
        self.stage: Optional[BatchExecutorCollector] = None
        self._ready_mark_var = ready_mark_var

    def submit(self, fn, /, *args, **kwargs):
        """Submit ``fn`` and account its queue wait when metrics are on."""
        stage = self.stage
        mark = self._ready_mark_var.get() if self._ready_mark_var else None
        if stage is None and mark is None:
            return super().submit(fn, *args, **kwargs)

        enqueued = time.perf_counter()
        if stage is not None:
            stage.queue_enter()

        def run():
            if stage is not None:
                stage.queue_leave(wait_seconds=time.perf_counter() - enqueued)
            try:
                return fn(*args, **kwargs)
            finally:
                if mark is not None:
                    mark.mark()

        future = super().submit(run)
        if stage is not None:
            # Only a job that never started can be cancelled; ``run`` skipped.
            future.add_done_callback(lambda f: f.cancelled() and stage.queue_leave())
        return future
