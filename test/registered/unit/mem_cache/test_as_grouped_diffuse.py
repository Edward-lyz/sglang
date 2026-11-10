"""CPU protocol checks for grouping different request shard boundaries."""
import concurrent.futures
import contextlib
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.storage.asradix import read_diffuse as rd
from sglang.srt.mem_cache.storage.asradix.checkpoint_coordinator import (
    SPACacheCoordinator,
    _SPLoadBackTask,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


@pytest.mark.parametrize("rank", range(4))
def test_grouped_copy_restarts_ownership_at_request_boundaries(rank):
    batches = [
        ([2, 8, 1], 3, (2,)),
        ([11, 7, 13, 5, 19], 5, (4,)),
        ([], 0, ()),
    ]
    diffuser = object.__new__(rd.AsReadDiffuser)
    diffuser.cp_rank, diffuser.cp_size = rank, 4
    diffuser._device, diffuser._group = torch.device("cpu"), object()
    diffuser._batched = True
    diffuser._buffer_tables = {peer: peer for peer in range(4)}
    actual = torch.full((24,), -1, dtype=torch.int64)
    expected = actual.clone()
    source = [actual.clone() for _ in range(4)]
    for slots, count, extra in batches:
        for k, slot in enumerate(slots):
            expected[slot] = slot * 13
            owner = rd.shard_owner(k, 4, count)
            source[owner][slot] = expected[slot]
            if owner == rank or k in extra:
                actual[slot] = expected[slot]
    copied = {}

    def scatter(peer, indices):
        copied[peer] = indices.tolist()
        actual[indices.long()] = source[peer][indices.long()]

    stream = SimpleNamespace(synchronize=mock.Mock())
    with mock.patch.object(torch.cuda, "current_stream", return_value=stream):
        with mock.patch.object(torch.cuda, "stream", side_effect=lambda s: contextlib.nullcontext()):
            with mock.patch.object(torch.distributed, "barrier") as barrier:
                with mock.patch.object(rd.kd, "scatter_buffers", side_effect=scatter):
                    with mock.patch(
                        "sglang.srt.mem_cache.storage.asradix.transport.cache_digest_enabled",
                        return_value=False,
                    ):
                        diffuser.diffuse_many(batches, peers_ready=True)
    assert torch.equal(actual, expected)
    for peer, slots in copied.items():
        assert peer != rank
        assert slots == [
            slot for request, count, extra in batches
            for k, slot in enumerate(request)
            if rd.shard_owner(k, 4, count) == peer and k not in extra
        ]
    barrier.assert_called_once_with(group=diffuser._group)
    stream.synchronize.assert_called_once()


def test_grouped_read_rejects_bad_shape_before_any_collective():
    diffuser = object.__new__(rd.AsReadDiffuser)
    with mock.patch.object(torch.distributed, "barrier") as barrier:
        with pytest.raises(ValueError):
            diffuser.diffuse_many([([1], 2, ())])
    barrier.assert_not_called()


@pytest.mark.parametrize("remote_failure", [False, True])
def test_grouped_commit_only_diffuses_all_rank_successes_before_publish(remote_failure):
    coordinator = SPACacheCoordinator()
    order, tasks = [], []
    for i, result in enumerate([4, 4, RuntimeError("read failed")]):
        req = SimpleNamespace(rid=str(i), _as_pending_load_back=None)
        future = concurrent.futures.Future()
        future.set_result(result)
        task = _SPLoadBackTask(
            rid=req.rid, req=req, load_info={"hashes": [i] * 4},
            loaded=torch.arange(4), requested=4, local_hit_len=0,
            started=0, lock=mock.Mock(), future=future,
            diffuse_page_slots=tuple(range(i * 4, i * 4 + 4)),
            diffuse_extra_blocks=(3,),
        )
        req._as_pending_load_back = task
        tasks.append(task)
    coordinator._pending_load_backs = list(tasks)
    coordinator._free_loaded_pages = mock.Mock()
    coordinator._insert_loaded_prefix = mock.Mock(
        side_effect=lambda cache, task, count: order.append(("publish", task.rid))
    )

    def reduce(counts, op):
        if counts.numel() == 1:
            order.append("ready")
        else:
            order.append("agree")
            if remote_failure:
                counts[1] = 0

    grouped = mock.Mock(side_effect=lambda *args, **kw: order.append("diffuse"))
    cache = SimpleNamespace(
        page_size=256, _all_reduce_attn_groups=reduce,
        _as_read_diffuser=SimpleNamespace(diffuse_many=grouped, diffuse=mock.Mock()),
        ascli=SimpleNamespace(
            emit_content_digest=mock.Mock(side_effect=lambda *args: order.append("digest"))
        ),
        _log_as_result=mock.Mock(), _log_as_failure=mock.Mock(),
    )
    with envs.SGLANG_AS_READ_DIFFUSE_GROUP_COMMIT.override(True):
        coordinator.commit_load_backs(cache)
    successful = tasks[:1] if remote_failure else tasks[:2]
    grouped.assert_called_once_with(
        [
            (task.diffuse_page_slots, task.requested, task.diffuse_extra_blocks)
            for task in successful
        ],
        peers_ready=True,
    )
    assert order[:3] == ["ready", "agree", "diffuse"]
    assert [item for item in order if isinstance(item, tuple)] == [
        ("publish", task.rid) for task in successful
    ]
    cache._as_read_diffuser.diffuse.assert_not_called()
    assert coordinator._free_loaded_pages.call_count == 3 - len(successful)
    assert all(task.req._as_pending_load_back is None for task in tasks)
    assert all(task.lock.call_count == 1 for task in tasks)


class _RefillRequest:
    def __init__(self, name):
        self.name = name
        self._as_pending_load_back = object()
        self.as_read_pending_lock = False


def _refill_fixture():
    admitted, successful, failed, untouched = [
        _RefillRequest(name) for name in ("admitted", "ready", "failed", "untouched")
    ]
    pending = [successful, failed]
    order = []

    def commit():
        order.append("commit")
        for req in pending:
            req._as_pending_load_back = None
        successful.as_read_pending_lock = True

    scheduler = SimpleNamespace(
        enable_asradix_cache=True, chunked_req=None,
        req_to_token_pool=SimpleNamespace(mamba_allocator=None),
        waiting_queue=[untouched, failed, successful, admitted],
        tree_cache=SimpleNamespace(
            pending_ascache_load_back_requests=mock.Mock(return_value=tuple(pending)),
            commit_ascache_load_backs=mock.Mock(side_effect=commit),
        ),
    )
    adder = SimpleNamespace(
        can_run_list=[admitted], rem_chunk_tokens=66560, rem_input_tokens=1048576,
        new_chunked_req=None, preempt_list=[],
    )
    running = SimpleNamespace(batch_is_full=False)
    return scheduler, adder, running, successful


def test_ready_refill_excludes_failed_reads_and_preserves_queue_order():
    scheduler, adder, running, successful = _refill_fixture()
    with envs.SGLANG_AS_LOAD_BACK_COALESCE_READY.override(True):
        with envs.SGLANG_ENABLE_AS_LOAD_BACK_ASYNC.override(True):
            assert Scheduler._ascache_ready_refill_candidates(
                scheduler, adder, running
            ) == [successful]
    scheduler.tree_cache.commit_ascache_load_backs.assert_called_once()


@pytest.mark.parametrize(
    "gate",
    ["disabled", "sync", "nocache", "full", "empty", "chunk", "input",
     "continuation", "new_chunk", "preempt", "mamba", "no_pending"],
)
def test_ready_refill_skips_when_normal_admission_must_proceed(gate):
    scheduler, adder, running, _ = _refill_fixture()
    if gate == "nocache":
        scheduler.enable_asradix_cache = False
    elif gate == "full":
        running.batch_is_full = True
    elif gate == "empty":
        adder.can_run_list = []
    elif gate == "chunk":
        adder.rem_chunk_tokens = 0
    elif gate == "input":
        adder.rem_input_tokens = 0
    elif gate == "continuation":
        scheduler.chunked_req = object()
    elif gate == "new_chunk":
        adder.new_chunked_req = object()
    elif gate == "preempt":
        adder.preempt_list = [object()]
    elif gate == "mamba":
        scheduler.req_to_token_pool.mamba_allocator = object()
    elif gate == "no_pending":
        scheduler.tree_cache.pending_ascache_load_back_requests.return_value = ()
    with envs.SGLANG_AS_LOAD_BACK_COALESCE_READY.override(gate != "disabled"), \
         envs.SGLANG_AS_LOAD_BACK_REFILL_PASSES.override(1):
        with envs.SGLANG_ENABLE_AS_LOAD_BACK_ASYNC.override(gate != "sync"):
            assert Scheduler._ascache_ready_refill_candidates(
                scheduler, adder, running
            ) == ()
    scheduler.tree_cache.commit_ascache_load_backs.assert_not_called()
