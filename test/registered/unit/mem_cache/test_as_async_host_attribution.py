"""Deferred AS restore provenance survives re-match without crediting failures."""
import concurrent.futures
import contextlib
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.srt.managers.schedule_batch import Req, split_cached_prefix_by_tier
from sglang.srt.mem_cache.storage.asradix.checkpoint_coordinator import (
    SPACacheCoordinator,
    _SPLoadBackTask,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "prefix,span,planned,loaded,expected",
    [
        (768, None, 512, 512, 512),
        (768, None, 512, 0, 0),
        (768, (256, 768), 0, 0, 512),
        (512, (256, 768), 0, 0, 256),
        (128, (256, 768), 0, 0, 0),
        (0, (0, 768), 768, 768, 0),
        (1024, (256, 768), 256, 256, 768),
        (768, (256, 768), 512, 512, 512),
        (1024, (256, 768), 512, 512, 768),
    ],
)
def test_materialized_origin_intersects_the_admitted_prefix(
    prefix, span, planned, loaded, expected
):
    req = SimpleNamespace(
        prefix_indices=list(range(prefix)),
        _as_materialized_host_span=span,
        host_hit_length=planned,
        host_loaded_length=loaded,
    )
    host = Req.materialized_host_hit_len(req)
    assert host == expected
    tiers = split_cached_prefix_by_tier(
        prefix_len=prefix, host_hit_len=host, storage_hit_len=0,
        storage_hit_start=None, host_hit_is_storage=False,
    )
    assert tiers == (prefix - expected, expected, 0)


@pytest.mark.parametrize("result,remote_failure", [(1, False), (0, False), (RuntimeError("IO failed"), False), (1, True)])
def test_commit_credits_only_complete_all_rank_reads(result, remote_failure):
    coordinator = SPACacheCoordinator()
    cache = SimpleNamespace(
        page_size=256,
        tree_core=SimpleNamespace(is_eagle=False),
        insert=mock.Mock(),
        _log_as_result=mock.Mock(),
        _log_as_failure=mock.Mock(),
    )
    votes = []

    def reduce(counts, op):
        votes.append(counts.tolist())
        # Readiness and successful read counts are separate MIN operations.
        # A failed read on a peer must fail the count vote, not defer readiness.
        if len(votes) == 2 and remote_failure:
            counts.zero_()

    cache._all_reduce_attn_groups = reduce
    coordinator._free_loaded_pages = mock.Mock()
    req = SimpleNamespace(
        rid="deferred", prefix_indices=torch.arange(256),
        full_untruncated_fill_ids=list(range(768)), extra_key=None,
        _as_materialized_host_span=None, as_read_pending_lock=False,
        host_hit_length=0, host_loaded_length=0,
    )
    future = concurrent.futures.Future()
    future.set_result(result)
    release = mock.Mock()
    task = _SPLoadBackTask(
        rid=req.rid, req=req, load_info={}, loaded=torch.arange(256, 512),
        requested=1, local_hit_len=256, started=0, lock=release, future=future,
    )
    req._as_pending_load_back = task
    coordinator._pending_load_backs = [task]
    match = SimpleNamespace(device_indices=torch.arange(512), last_device_node=object())
    with mock.patch(
        "sglang.srt.mem_cache.storage.asradix.checkpoint_coordinator.UnifiedRadixCache.match_prefix",
        return_value=match,
    ):
        coordinator.commit_load_backs(cache)
    assert len(votes) == 2
    assert req._as_pending_load_back is None
    assert coordinator._pending_load_backs == []
    release.assert_called_once()
    if result == 1 and not remote_failure:
        cache.insert.assert_called_once()
        assert req._as_materialized_host_span == (256, 512)
        # Simulate next admission's device re-match, which has no outstanding IO.
        req.prefix_indices = torch.arange(512)
        assert Req.materialized_host_hit_len(req) == 256
    else:
        cache.insert.assert_not_called()
        assert req._as_materialized_host_span is None
        assert Req.materialized_host_hit_len(req) == 0
        coordinator._free_loaded_pages.assert_called_once()


@pytest.mark.parametrize("received,expected", [(3, 4), (2, 0)])
def test_background_shard_read_has_no_collectives_or_publish(received, expected):
    coordinator = SPACacheCoordinator()
    coordinator._prepare_read = mock.Mock(return_value=([100, 101, 102, 103], (), {}))
    cache = SimpleNamespace(
        _as_read_diffuser=SimpleNamespace(local_shard=lambda count: [0, 1]),
        ascli=SimpleNamespace(read=mock.Mock(return_value=received)),
    )
    task = _SPLoadBackTask(
        rid="sharded", req=object(), load_info={"hashes": [10, 20, 30, 40]},
        loaded=torch.arange(1024), requested=4, local_hit_len=0, started=0,
        lock=mock.Mock(),
    )
    with mock.patch.object(torch.distributed, "all_reduce", side_effect=AssertionError("background collective")):
        assert coordinator._run_load_back_read(cache, task) == expected
    cache.ascli.read.assert_called_once_with(
        [10, 20, 40], [100, 101, 103], timeout=5.0, remote_addrs=None,
    )
    assert task.diffuse_page_slots == (100, 101, 102, 103)
    assert task.diffuse_extra_blocks == (3,)


def test_background_reader_selects_device_and_waits_for_allocation():
    coordinator = SPACacheCoordinator()
    order = []
    task = SimpleNamespace(
        loaded=SimpleNamespace(is_cuda=True, device="cuda:3"),
        allocation_ready=object(),
    )
    context = mock.MagicMock()
    context.__enter__.side_effect = lambda: order.append("device")
    stream = SimpleNamespace(
        wait_event=lambda event: order.append(("wait", event))
    )
    coordinator._run_local_load_back_read = mock.Mock(
        side_effect=lambda *args: order.append("read") or 4
    )
    with mock.patch.object(torch.cuda, "device", return_value=context) as select:
        with mock.patch.object(torch.cuda, "Stream", return_value=stream) as create:
            with mock.patch.object(torch.cuda, "stream", side_effect=lambda s: contextlib.nullcontext()):
                assert coordinator._run_load_back_read(object(), task) == 4
                assert coordinator._run_load_back_read(object(), task) == 4
            create.assert_called_once_with(device="cuda:3")
    assert select.call_args_list == [mock.call("cuda:3"), mock.call("cuda:3")]
    assert order == ["device", ("wait", task.allocation_ready), "read"] * 2


@pytest.mark.parametrize("peer_success", [True, False])
@pytest.mark.parametrize("grouped", [False, True])
def test_sharded_publish_follows_all_rank_agreement_and_diffuse(peer_success, grouped):
    from sglang.srt.environ import envs

    coordinator = SPACacheCoordinator()
    order = []
    req = SimpleNamespace(
        rid="pipeline", prefix_indices=torch.arange(256),
        full_untruncated_fill_ids=list(range(1536)), extra_key=None,
        _as_materialized_host_span=None, as_read_pending_lock=False,
        host_hit_length=0, host_loaded_length=0,
    )
    future = concurrent.futures.Future()
    future.set_result(4)
    task = _SPLoadBackTask(
        rid=req.rid, req=req, load_info={"hashes": [10, 20, 30, 40]},
        loaded=torch.arange(256, 1280), requested=4, local_hit_len=256,
        started=0, lock=mock.Mock(), future=future,
        diffuse_page_slots=(1, 2, 3, 4), diffuse_extra_blocks=(3,),
    )
    req._as_pending_load_back = task
    coordinator._pending_load_backs = [task]
    coordinator._free_loaded_pages = mock.Mock()
    cache = SimpleNamespace(
        page_size=256, tree_core=SimpleNamespace(is_eagle=False),
        insert=mock.Mock(side_effect=lambda *args: order.append("publish")),
        _as_read_diffuser=SimpleNamespace(
            diffuse=mock.Mock(side_effect=lambda *args, **kwargs: order.append("diffuse")),
            diffuse_many=mock.Mock(side_effect=lambda *args, **kwargs: order.append("diffuse")),
        ),
        ascli=SimpleNamespace(
            emit_content_digest=mock.Mock(side_effect=lambda *args: order.append("digest"))
        ),
        _log_as_result=mock.Mock(), _log_as_failure=mock.Mock(),
    )
    votes = []

    def reduce(counts, op):
        votes.append(counts.tolist())
        order.append("ready" if len(votes) == 1 else "counts")
        if len(votes) == 2 and not peer_success:
            counts.zero_()
    cache._all_reduce_attn_groups = reduce
    match = SimpleNamespace(device_indices=torch.arange(1280), last_device_node=object())
    with mock.patch(
        "sglang.srt.mem_cache.storage.asradix.checkpoint_coordinator.UnifiedRadixCache.match_prefix",
        return_value=match,
    ):
        with envs.SGLANG_AS_READ_DIFFUSE_GROUP_COMMIT.override(grouped):
            coordinator.commit_load_backs(cache)
    assert order == (
        ["ready", "counts", "diffuse", "digest", "publish"]
        if peer_success else ["ready", "counts"]
    )
    if peer_success:
        if grouped:
            cache._as_read_diffuser.diffuse_many.assert_called_once_with(
                [((1, 2, 3, 4), 4, (3,))], peers_ready=True,
            )
            cache._as_read_diffuser.diffuse.assert_not_called()
        else:
            cache._as_read_diffuser.diffuse.assert_called_once_with(
                (1, 2, 3, 4), 4, extra_local_blocks=(3,), peers_ready=True,
            )
            cache._as_read_diffuser.diffuse_many.assert_not_called()
        req.prefix_indices = torch.arange(1280)
        assert Req.materialized_host_hit_len(req) == 1024
    else:
        assert req._as_materialized_host_span is None
        coordinator._free_loaded_pages.assert_called_once()


def test_peer_not_ready_retains_task_without_credit_or_free():
    coordinator = SPACacheCoordinator()
    future = mock.Mock()
    future.done.return_value = True
    req = SimpleNamespace(_as_materialized_host_span=None)
    release = mock.Mock()
    task = _SPLoadBackTask(
        rid="not-ready", req=req, load_info={}, loaded=torch.arange(256),
        requested=1, local_hit_len=0, started=0, lock=release, future=future,
    )
    req._as_pending_load_back = task
    coordinator._pending_load_backs = [task]
    coordinator._free_loaded_pages = mock.Mock()
    cache = SimpleNamespace(
        _all_reduce_attn_groups=mock.Mock(side_effect=lambda counts, op: counts.zero_()),
        insert=mock.Mock(),
    )
    coordinator.commit_load_backs(cache)
    cache._all_reduce_attn_groups.assert_called_once()
    future.result.assert_not_called()
    cache.insert.assert_not_called()
    coordinator._free_loaded_pages.assert_not_called()
    release.assert_not_called()
    assert coordinator._pending_load_backs == [task]
    assert req._as_pending_load_back is task
    assert req._as_materialized_host_span is None


@pytest.mark.parametrize("limit,count,full", [(32, 31, False), (32, 32, True), (0, 100, False)])
def test_pending_limit_bounds_allocated_read_batch(limit, count, full):
    from sglang.srt.environ import envs
    from sglang.srt.mem_cache.storage.asradix.cache import AttentionStoreUnifiedRadixCache
    cache = SimpleNamespace(
        attention_store_coordinator=SimpleNamespace(_pending_load_backs=[None] * count)
    )
    with envs.SGLANG_AS_LOAD_BACK_MAX_PENDING.override(limit):
        assert AttentionStoreUnifiedRadixCache.attention_store_load_back_batch_is_full(cache) is full
