"""Prometheus metric collectors for multimodal encode receiver."""

from typing import Dict


class MMReceiverMetricsCollector:
    """Collect metrics for EncodeReceiver."""

    def __init__(self, labels: Dict[str, str]):
        from prometheus_client import Counter, Gauge, Histogram

        self.labels = labels

        self.waiting_requests = Gauge(
            name="sglang:mm_receiver_waiting_requests",
            documentation="Current number of waiting multimodal requests.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.failed_requests_total = Counter(
            name="sglang:mm_receiver_failed_requests_total",
            documentation="Total number of failed multimodal waiting requests.",
            labelnames=labels.keys(),
        )
        self.timeout_requests_total = Counter(
            name="sglang:mm_receiver_timeout_requests_total",
            documentation="Total number of timed out multimodal waiting requests.",
            labelnames=labels.keys(),
        )
        self.histogram_embedding_transfer_time = Histogram(
            name="sglang:mm_receiver_embedding_transfer_time",
            documentation="Histogram of embedding transfer time in EPD deployment.",
            labelnames=labels.keys(),
            buckets=[
                0.01,
                0.05,
                0.1,
                0.25,
                0.5,
                0.75,
                1,
                2,
                3,
                5,
                10,
                20,
                40,
                60,
                80,
                120,
            ],
        )
        self.gpu_buffer_pool_used_bytes = Gauge(
            name="sglang:gpu_buffer_pool_used_bytes",
            documentation="Number of used bytes in the GPU buffer pool.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.gpu_buffer_pool_total_bytes = Gauge(
            name="sglang:gpu_buffer_pool_total_bytes",
            documentation="Total number of bytes in the GPU buffer pool.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

    def set_waiting_requests(self, value: int) -> None:
        """Set the value of waiting requests."""
        self.waiting_requests.labels(**self.labels).set(max(0, int(value)))

    def increment_failed_requests(self) -> None:
        """Increment the count of failed requests."""
        self.failed_requests_total.labels(**self.labels).inc(1)

    def increment_timeout_requests(self) -> None:
        """Increment the count of timeout requests."""
        self.timeout_requests_total.labels(**self.labels).inc(1)

    def observe_embedding_transfer_time(self, value: float) -> None:
        """Observe embedding transfer time in EPD deployment."""
        self.histogram_embedding_transfer_time.labels(**self.labels).observe(value)

    def set_gpu_buffer_pool_used_bytes(self, value: int) -> None:
        """Set the number of used bytes in the GPU buffer pool."""
        self.gpu_buffer_pool_used_bytes.labels(**self.labels).set(max(0, int(value)))

    def set_gpu_buffer_pool_total_bytes(self, value: int) -> None:
        """Set the total number of bytes in the GPU buffer pool."""
        self.gpu_buffer_pool_total_bytes.labels(**self.labels).set(max(0, int(value)))
