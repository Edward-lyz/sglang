"""
Encoder-level metrics collector.

Centralizes all monitoring metrics for the disaggregated encoder (MMEncoder),
including mm_cache hit rate tracking and Prometheus metric definitions.
"""

# NOTE: `from __future__ import annotations` keeps the `Counter` / `Gauge`
# type hints below as strings, so they don't need `prometheus_client` imported
# at module load time.
from __future__ import annotations

from collections import defaultdict
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
    """Prometheus metrics collector for the disaggregated encoder."""

    def __init__(self, labels: Optional[Dict[str, str]] = None):
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
        self.cache_entries = Gauge(
            name="sglang:encoder_cache_entries",
            documentation="Current number of entries in mm_cache.",
            labelnames=self._label_keys,
            multiprocess_mode="mostrecent",
        )
        self.cache_used_bytes = Gauge(
            name="sglang:encoder_cache_used_bytes",
            documentation="Current bytes used by mm_cache.",
            labelnames=self._label_keys,
            multiprocess_mode="mostrecent",
        )

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

    def set_cache_state(self, num_entries: int, used_bytes: int):
        """Update cache state metrics."""
        self._set_gauge(self.cache_entries, num_entries)
        self._set_gauge(self.cache_used_bytes, used_bytes)

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

    _LATENCY_BUCKETS = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60]

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

    _LATENCY_BUCKETS = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30]
    _TASK_COUNT_BUCKETS = [0, 1, 2, 4, 8, 16, 32, 64]

    def __init__(self, labels: Optional[Dict[str, str]] = None):
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
        )
        self.sched_bubble_sec = Histogram(
            name=f"sglang:{name}_sched_bubble_seconds",
            documentation=f"Histogram of {name} per-request sched bubble time in seconds.",
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30],
            labelnames=labels.keys(),
        )
        self.labels = labels

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
