"""Factory encoder throughput paths: task batching, per-request cache, forward thread.

Concurrent factory tasks are coalesced into fused ``batch_encode`` calls, a
fused batch still serves each request from the prefix mm_cache, and the
vision forward can leave the event loop. All checks run without a model.
"""

import asyncio
import threading
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

import sglang.srt.disaggregation.encoder.server as encoder_server
from sglang.srt.disaggregation.encoder.server import (
    BadRequestError,
    FactoryEncodeBatcher,
    MMEncoder,
)
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.mem_cache.multimodal_cache import MultiModalStaticCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _staged(request):
    return SimpleNamespace(
        req_id=request["req_id"],
        embedding=torch.ones(2, 4),
        grid_dim=[[1, 2, 2]],
        original_image_sizes=None,
    )


class _Batches:
    """Stand-in for ``MMEncoder.batch_encode`` recording each fused call."""

    def __init__(self, fail_on=()):
        self.calls = []
        self.fail_on = set(fail_on)

    async def __call__(self, requests, modality, out_staged=None):
        self.calls.append([r["req_id"] for r in requests])
        await asyncio.sleep(0.01)
        keys = {r["req_id"] for r in requests} | {r.get("task_id") for r in requests}
        if self.fail_on & keys:
            return [(0, 0, 0, "bad image", HTTPStatus.BAD_REQUEST)] * len(requests)
        out_staged.extend(_staged(r) for r in requests)
        return [(32, 2, 4, None, None)] * len(requests)


def _task_of(encode_req_id):
    """``req_id#task_id`` of an encode id (drops the per-copy sequence number)."""
    return encode_req_id.rsplit("#", 1)[0]


def _request(i):
    return {"req_id": f"r{i}", "task_id": f"t{i}"}


def test_concurrent_tasks_share_fused_batches():
    async def run():
        batches = _Batches()
        batcher = FactoryEncodeBatcher(
            batches, max_batch=4, max_wait_s=0.005, max_inflight=1
        )
        results = await asyncio.gather(
            *(batcher.submit(_request(i), Modality.IMAGE) for i in range(6))
        )
        assert [r.req_id for r in results] == [f"r{i}" for i in range(6)]
        # Full batch at once; the rest waits for the single in-flight slot.
        assert batches.calls == [["r0", "r1", "r2", "r3"], ["r4", "r5"]]

    asyncio.run(run())


def test_a_lone_task_waits_at_most_the_batch_window():
    async def run():
        batches = _Batches()
        batcher = FactoryEncodeBatcher(
            batches, max_batch=8, max_wait_s=0.0, max_inflight=2
        )
        result = await asyncio.wait_for(
            batcher.submit(_request(0), Modality.IMAGE), timeout=1
        )
        assert result.req_id == "r0" and batches.calls == [["r0"]]

    asyncio.run(run())


def test_a_failing_task_is_isolated_from_its_batch():
    async def run():
        batches = _Batches(fail_on={"r1"})
        batcher = FactoryEncodeBatcher(
            batches, max_batch=3, max_wait_s=0.01, max_inflight=1
        )
        results = await asyncio.gather(
            *(batcher.submit(_request(i), Modality.IMAGE) for i in range(3)),
            return_exceptions=True,
        )
        assert results[0].req_id == "r0" and results[2].req_id == "r2"
        assert isinstance(results[1], BadRequestError)
        assert "bad image" in str(results[1])
        assert batches.calls[0] == ["r0", "r1", "r2"]
        assert sorted(batches.calls[1:]) == [["r0"], ["r1"], ["r2"]]

    asyncio.run(run())


def test_retry_starts_from_a_released_request_state():
    """A failed batch stages an error per request; the retry must not reuse it.

    MMEncoder refuses to stage an embedding over a staged error with another
    shape ("Embedding metadata mismatch"), so the batcher releases each
    request before retrying it alone.
    """

    async def run():
        errored = set()

        class _StatefulBatches(_Batches):
            async def __call__(self, requests, modality, out_staged=None):
                ids = {r["req_id"] for r in requests}
                if ids & errored:
                    return [(0, 0, 0, "Embedding metadata mismatch", 500)] * len(
                        requests
                    )
                results = await super().__call__(requests, modality, out_staged)
                if results[0][3] is not None:
                    errored.update(ids)
                return results

        async def release(request):
            errored.discard(request["req_id"])

        batcher = FactoryEncodeBatcher(
            _StatefulBatches(fail_on={"r1"}),
            max_batch=2,
            max_wait_s=0.01,
            max_inflight=1,
            release=release,
        )
        results = await asyncio.gather(
            *(batcher.submit(_request(i), Modality.IMAGE) for i in range(2)),
            return_exceptions=True,
        )
        assert results[0].req_id == "r0"
        assert isinstance(results[1], BadRequestError)

    asyncio.run(run())


def test_a_caller_that_left_mid_batch_gets_its_state_released():
    async def run():
        released = []

        async def release(request):
            released.append(request["req_id"])

        batches = _Batches()
        batcher = FactoryEncodeBatcher(
            batches, max_batch=2, max_wait_s=0.01, max_inflight=1, release=release
        )
        keep = asyncio.ensure_future(batcher.submit(_request(0), Modality.IMAGE))
        leave = asyncio.ensure_future(batcher.submit(_request(1), Modality.IMAGE))
        await asyncio.sleep(0.005)  # the batch is running
        leave.cancel()
        assert (await keep).req_id == "r0"
        await asyncio.sleep(0.02)
        assert released == ["r1"]

    asyncio.run(run())


def test_workflow_tasks_encode_together_under_their_own_request_ids():
    """Tasks of one request share req_id, so each encodes as ``req_id#task_id``."""

    async def run():
        encoder = MMEncoder.__new__(MMEncoder)
        batches = _Batches()
        released = []
        encoder.batch_encode = batches
        encoder.task_embeddings = {}

        async def no_hold(task_id):
            return None

        async def store(task_id, embedding, *args, **kwargs):
            encoder.task_embeddings[task_id] = SimpleNamespace(embedding=embedding)

        async def release(req_id):
            released.append(req_id)

        encoder._reuse_held_task_embedding = no_hold
        encoder.store_task_embedding = store
        encoder.release_request = release
        with (
            patch.object(
                encoder_server.envs.SGLANG_ENCODER_FACTORY_BATCH_SIZE,
                "get",
                return_value=4,
            ),
            patch.object(
                encoder_server.envs.SGLANG_ENCODER_FACTORY_BATCH_WAIT_MS,
                "get",
                return_value=5.0,
            ),
        ):
            results = await asyncio.gather(
                *(
                    encoder.encode_and_store_for_workflow(
                        ["u"], "req", f"t{i}", Modality.IMAGE
                    )
                    for i in range(3)
                )
            )
        assert [[_task_of(r) for r in call] for call in batches.calls] == [
            ["req#t0", "req#t1", "req#t2"]
        ]
        assert sorted(map(_task_of, released)) == ["req#t0", "req#t1", "req#t2"]
        assert all(r[0] == 2 for r in results)

    asyncio.run(run())


def _workflow_encoder(batches):
    """MMEncoder with the hold side stubbed and ``encode`` routed to ``batches``."""
    encoder = MMEncoder.__new__(MMEncoder)
    encoder.batch_encode = batches
    encoder.task_embeddings = {}
    encoder.released = []

    async def no_hold(task_id):
        return None

    async def store(task_id, embedding, *args, **kwargs):
        encoder.task_embeddings[task_id] = SimpleNamespace(embedding=embedding)

    async def release(req_id):
        encoder.released.append(req_id)

    async def encode(mm_items, modality, req_id, num_parts, part_idx, **kwargs):
        results = await batches(
            [{"req_id": req_id, "task_id": kwargs.get("task_id")}],
            modality,
            out_staged=kwargs["out_staged"],
        )
        return results[0]

    encoder._reuse_held_task_embedding = no_hold
    encoder.store_task_embedding = store
    encoder.release_request = release
    encoder.encode = encode
    return encoder


def _batch_size(size):
    return patch.object(
        encoder_server.envs.SGLANG_ENCODER_FACTORY_BATCH_SIZE, "get", return_value=size
    )


def test_video_tasks_are_not_batched_with_images():
    async def run():
        batches = _Batches()
        encoder = _workflow_encoder(batches)
        with _batch_size(4):
            await asyncio.gather(
                encoder.encode_and_store_for_workflow(
                    ["f0", "f1"],
                    "req",
                    "video",
                    Modality.IMAGE,
                    store_modality="video",
                    chunk_timestamps=["00:00", "00:01"],
                ),
                *(
                    encoder.encode_and_store_for_workflow(
                        ["u"], "req", f"t{i}", Modality.IMAGE
                    )
                    for i in range(2)
                ),
            )
        calls = sorted([_task_of(r) for r in call] for call in batches.calls)
        assert calls == [["req#t0", "req#t1"], ["req#video"]]

    asyncio.run(run())


@pytest.mark.parametrize("batch_size", [1, 4])
def test_a_bad_image_reports_400_with_or_without_batching(batch_size):
    """TaskResult takes status_code from the error: keep the encoder's 400."""

    async def run():
        encoder = _workflow_encoder(_Batches(fail_on={"bad"}))
        with _batch_size(batch_size):
            with pytest.raises(BadRequestError, match="bad image"):
                await encoder.encode_and_store_for_workflow(
                    ["u"], "req", "bad", Modality.IMAGE
                )
        assert [_task_of(r) for r in encoder.released] == ["req#bad"]

    asyncio.run(run())


def _cache_encoder():
    encoder = MMEncoder.__new__(MMEncoder)
    encoder.mm_cache = MultiModalStaticCache(1024 * 1024)
    encoder.mm_cache_lock = asyncio.Lock()
    encoder.device = torch.device("cpu")
    return encoder


def _fused_ctx(hashes, get_feature_fn, rows=2):
    items = [SimpleNamespace(hash=h, set_pad_value=lambda: None) for h in hashes]
    ctx = SimpleNamespace(
        req_id="req",
        modality=Modality.IMAGE,
        num_items=len(items),
        mm_feature=None,
        preprocess_result=SimpleNamespace(
            token_counts=[rows] * len(items), mm_inputs={}
        ),
        get_feature_fn=get_feature_fn,
        is_health_check=False,
        items_per_req=[1] * len(items),
        aux_data={},
    )
    return items, ctx


def _feature_by_hash(rows=2):
    """Rows of value ``hash`` per item, so results show which image they came from."""
    return Mock(
        side_effect=lambda items: torch.cat(
            [torch.full((rows, 4), float(it.hash)) for it in items]
        )
    )


def test_fused_batch_serves_each_request_from_the_cache():
    async def run():
        get_feature_fn = _feature_by_hash()
        encoder = _cache_encoder()
        encoder.metrics_collector = Mock()
        prefix_cache = patch.object(
            encoder_server,
            "get_mm",
            return_value=SimpleNamespace(enable_prefix_mm_cache=True),
        )
        with prefix_cache:
            items, ctx = _fused_ctx([1, 2], get_feature_fn)
            encoder._build_model_mm_items = Mock(return_value=items)
            first = await encoder._compute_direct_embedding(ctx, keep_on_gpu=False)

            items, ctx = _fused_ctx([2, 3, 1], get_feature_fn)
            encoder._build_model_mm_items = Mock(return_value=items)
            second = await encoder._compute_direct_embedding(ctx, keep_on_gpu=False)

        assert first[:, 0].tolist() == [1, 1, 2, 2]
        assert second[:, 0].tolist() == [2, 2, 3, 3, 1, 1]
        # The second forward ran only on the uncached image.
        assert [it.hash for it in get_feature_fn.call_args_list[1].args[0]] == [3]
        assert len(encoder.mm_cache) == 3
        assert encoder.metrics_collector.record_cache_hit.call_count == 2
        assert encoder.metrics_collector.record_cache_miss.call_count == 3

    asyncio.run(run())


def test_fully_cached_fused_batch_skips_the_forward():
    async def run():
        get_feature_fn = _feature_by_hash()
        encoder = _cache_encoder()
        with patch.object(
            encoder_server,
            "get_mm",
            return_value=SimpleNamespace(enable_prefix_mm_cache=True),
        ):
            for _ in range(2):
                items, ctx = _fused_ctx([5, 6], get_feature_fn)
                encoder._build_model_mm_items = Mock(return_value=items)
                out = await encoder._compute_direct_embedding(ctx, keep_on_gpu=False)
        assert get_feature_fn.call_count == 1
        assert out[:, 0].tolist() == [5, 5, 6, 6]

    asyncio.run(run())


def test_aggregate_fused_batch_bypasses_the_cache():
    """A legacy single aggregate item cannot be split per request."""

    async def run():
        get_feature_fn = Mock(return_value=torch.zeros(4, 4))
        encoder = _cache_encoder()
        _, ctx = _fused_ctx([1, 2], get_feature_fn)
        aggregate = SimpleNamespace(hash=9, set_pad_value=lambda: None)
        encoder._build_model_mm_items = Mock(return_value=[aggregate])
        with patch.object(
            encoder_server,
            "get_mm",
            return_value=SimpleNamespace(enable_prefix_mm_cache=True),
        ):
            await encoder._compute_direct_embedding(ctx, keep_on_gpu=False)
        assert len(encoder.mm_cache) == 0

    asyncio.run(run())


@pytest.mark.parametrize(
    "task_ids,cache,expected",
    [
        (["t0", "t1"], True, True),
        (["t0", None], True, False),
        (["t0"], False, False),
    ],
)
def test_factory_batches_with_the_cache_stay_on_the_host(task_ids, cache, expected):
    requests = [{"req_id": "r", "task_id": t} for t in task_ids]
    with patch.object(
        encoder_server,
        "get_mm",
        return_value=SimpleNamespace(enable_prefix_mm_cache=cache),
    ):
        assert MMEncoder._embeddings_go_to_host(requests) is expected


def test_forward_runs_off_the_event_loop_when_enabled():
    async def run(enabled):
        encoder = MMEncoder.__new__(MMEncoder)
        encoder.device = torch.device("cpu")
        threads = []

        def get_feature_fn(items):
            threads.append(threading.current_thread())
            assert torch.is_inference_mode_enabled()
            return torch.zeros(3, 2)

        with (
            patch.object(
                encoder_server.envs.SGLANG_ENCODER_FORWARD_IN_THREAD,
                "get",
                return_value=enabled,
            ),
            patch.object(
                encoder_server,
                "get_parallel",
                return_value=SimpleNamespace(tp_size=1),
            ),
        ):
            out = await encoder._feature_forward(get_feature_fn, [], to_cpu=True)
        assert out.shape == (3, 2)
        return threads[0] is threading.main_thread()

    assert asyncio.run(run(False)) is True
    assert asyncio.run(run(True)) is False
