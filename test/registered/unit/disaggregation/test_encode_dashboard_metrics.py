"""New EPD encoder emits the 厂内 Grafana "Encoder监控" metric families.

The dashboard queries the families below with a ``model_name`` filter. The
collector must register them next to the community observability
EncoderMetricsCollector, which already owns ``sglang:encoder_cache_entries``.
"""

import threading
from contextvars import ContextVar

from prometheus_client import REGISTRY, Gauge, generate_latest

from sglang.srt.disaggregation.encode_collector import (
    BatchExecutorCollector,
    EncoderMetricsCollector,
    QueueMeteredExecutor,
    StageReadyMark,
)

DASHBOARD_FAMILIES = [
    "sglang:encoder_num_requests_total",
    "sglang:encoder_num_aborted_requests_total",
    "sglang:encoder_num_failed_requests_total",
    "sglang:encoder_cache_hit_requests_total",
    "sglang:encoder_cache_total_requests_total",
    "sglang:encoder_cache_hit_bytes_total",
    "sglang:encoder_cache_total_bytes_total",
    "sglang:mm_loader_run_batch_seconds_sum",
    "sglang:mm_process_run_batch_seconds_sum",
    "sglang:mm_embed_run_batch_seconds_sum",
    "sglang:mm_embed_batch_size_sum",
    "sglang:mm_embed_batch_num_tokens_sum",
    "sglang:mooncake_transfer_run_batch_seconds_sum",
    "sglang:mooncake_transfer_batch_num_tokens_sum",
    "sglang:epd_transfer_total_seconds_sum",
    "sglang:mm_loader_queue_wait_num_reqs",
    "sglang:mm_loader_queue_wait_seconds_sum",
    "sglang:mm_process_queue_wait_seconds_sum",
    "sglang:mm_embed_queue_wait_num_reqs",
    "sglang:mm_embed_queue_wait_seconds_sum",
    "sglang:mooncake_transfer_queue_wait_num_reqs",
    "sglang:mooncake_transfer_queue_wait_seconds_sum",
]


def _sample(name: str, labels: dict) -> float:
    value = REGISTRY.get_sample_value(name, labels)
    assert value is not None, name
    return value


def test_dashboard_families_are_exported_with_model_name():
    # Stands in for the community collector's gauge of the same name.
    Gauge("sglang:encoder_cache_entries", "community gauge", ["model_name"])
    metrics = EncoderMetricsCollector({"model_name": "deepseek_v41_flash"})

    metrics.observe_one_finished_request()
    metrics.observe_one_failed_request()
    metrics.observe_one_aborted_request()
    metrics.record_cache_hit(1024, source="transfer")
    metrics.record_cache_miss(2048)
    metrics.loader_stage.observe_batch(0.01, 1, 502)
    metrics.process_stage.observe_batch(0.02, 1, 502)
    metrics.embed_stage.observe_batch(0.05, 2, 1004)
    metrics.transfer_stage.observe_batch(0.003, 1, 502 * 4096 * 2)
    metrics.transfer_endpoint.on_request_end("200", 0.004)
    for stage in (
        metrics.loader_stage,
        metrics.process_stage,
        metrics.embed_stage,
        metrics.transfer_stage,
    ):
        stage.queue_enter()
        stage.queue_leave(wait_seconds=0.001)

    text = generate_latest(REGISTRY).decode()
    for family in DASHBOARD_FAMILIES:
        assert f'{family}{{model_name="deepseek_v41_flash"' in text, family
    assert (
        'sglang:encoder_num_requests_total{model_name="deepseek_v41_flash"} 3.0'
        in text
    )


def test_metered_executor_reports_backlog_and_marks_ready():
    labels = {"model_name": "queue_probe"}
    stage = BatchExecutorCollector("probe_pool", labels=labels)
    embed = BatchExecutorCollector("probe_embed", labels=labels)
    ready_var: ContextVar = ContextVar("ready", default=None)
    pool = QueueMeteredExecutor(max_workers=1, ready_mark_var=ready_var)
    pool.stage = stage

    started, gate = threading.Event(), threading.Event()

    def block():
        started.set()
        return gate.wait(5)

    blocker = pool.submit(block)
    assert started.wait(5)
    mark = StageReadyMark(embed, num_reqs=2)
    ready_var.set(mark)
    queued = pool.submit(lambda: "done")
    # The single worker is busy, so the second job waits in the queue.
    assert _sample("sglang:probe_pool_queue_wait_num_reqs", labels) == 1

    gate.set()
    assert blocker.result() is True and queued.result() == "done"
    pool.shutdown(wait=True)
    assert _sample("sglang:probe_pool_queue_wait_num_reqs", labels) == 0
    assert _sample("sglang:probe_pool_queue_wait_seconds_count", labels) == 2

    # The finished job put the 2-request batch in the embed queue.
    assert _sample("sglang:probe_embed_queue_wait_num_reqs", labels) == 2
    mark.start()
    mark.discard()  # no-op after start
    assert _sample("sglang:probe_embed_queue_wait_num_reqs", labels) == 0
    assert _sample("sglang:probe_embed_queue_wait_seconds_count", labels) == 2


def test_ready_mark_discard_releases_backlog():
    labels = {"model_name": "discard_probe"}
    embed = BatchExecutorCollector("discard_embed", labels=labels)
    mark = StageReadyMark(embed, num_reqs=3)
    mark.mark()
    mark.mark()  # one batch enters the queue once
    assert _sample("sglang:discard_embed_queue_wait_num_reqs", labels) == 3
    mark.discard()
    mark.start()  # no-op after discard
    assert _sample("sglang:discard_embed_queue_wait_num_reqs", labels) == 0
    assert REGISTRY.get_sample_value(
        "sglang:discard_embed_queue_wait_seconds_count", labels
    ) in (None, 0.0)
