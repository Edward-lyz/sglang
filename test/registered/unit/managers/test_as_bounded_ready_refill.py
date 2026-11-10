"""Keep bounded AS refill behind the original admission and topology gates."""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler


class Request:
    def __init__(self, *, pending=None, read_lock=False):
        self._as_pending_load_back = pending
        self.as_read_pending_lock = read_lock


def fixture():
    admitted, pending, fresh = Request(), Request(pending=object()), Request()
    scheduler = object.__new__(Scheduler)
    scheduler.enable_asradix_cache = True
    scheduler.chunked_req = None
    scheduler.req_to_token_pool = SimpleNamespace(mamba_allocator=None)
    scheduler.waiting_queue = [admitted, pending, fresh]
    scheduler.tree_cache = SimpleNamespace(
        pending_ascache_load_back_requests=Mock(return_value=(pending,)),
        commit_ascache_load_backs=Mock(),
    )

    def commit():
        pending._as_pending_load_back = None
        pending.as_read_pending_lock = True

    scheduler.tree_cache.commit_ascache_load_backs.side_effect = commit
    adder = SimpleNamespace(
        can_run_list=[admitted],
        rem_chunk_tokens=79360,
        rem_input_tokens=1046016,
        new_chunked_req=None,
        preempt_list=[],
    )
    running = SimpleNamespace(batch_is_full=False)
    return scheduler, adder, running, admitted, pending, fresh


def mode(passes):
    stack = ExitStack()
    for name, value in (
        ("SGLANG_AS_LOAD_BACK_REFILL_PASSES", passes),
        ("SGLANG_AS_LOAD_BACK_COALESCE_READY", True),
        ("SGLANG_ENABLE_AS_LOAD_BACK_ASYNC", True),
    ):
        stack.enter_context(
            patch(
                f"sglang.srt.managers.scheduler.envs.{name}.get",
                return_value=value,
            )
        )
    return stack


def test_original_mode_refills_only_completed_group_once():
    s, a, r, admitted, pending, fresh = fixture()
    with mode(1):
        assert s._ascache_ready_refill_candidates(a, r) == [pending]
        assert s._ascache_ready_refill_candidates(a, r, 1) == ()
    s.tree_cache.commit_ascache_load_backs.assert_called_once()


def test_bounded_mode_rescans_without_duplicate_admission():
    s, a, r, admitted, pending, fresh = fixture()
    with mode(4):
        assert s._ascache_ready_refill_candidates(a, r) == [pending, fresh]
        assert admitted in a.can_run_list
        assert len(a.can_run_list) == 1
        assert s._ascache_ready_refill_candidates(a, r, 4) == ()
    s.tree_cache.commit_ascache_load_backs.assert_called_once()


def test_bounded_mode_can_commit_initial_empty_batch():
    s, a, r, admitted, pending, fresh = fixture()
    a.can_run_list.clear()
    with mode(1):
        assert s._ascache_ready_refill_candidates(a, r) == ()
    with mode(4):
        assert s._ascache_ready_refill_candidates(a, r) == [admitted, pending, fresh]
    s.tree_cache.commit_ascache_load_backs.assert_called_once()


@pytest.mark.parametrize("passes", [0, -1, 9, 100])
def test_pass_count_clamped(passes):
    s, a, r, *_ = fixture()
    limit = max(1, min(8, passes))
    with mode(passes):
        assert s._ascache_ready_refill_candidates(a, r, limit) == ()
    s.tree_cache.commit_ascache_load_backs.assert_not_called()


@pytest.mark.parametrize(
    "owner,field,value",
    [
        ("scheduler", "enable_asradix_cache", False),
        ("scheduler", "chunked_req", object()),
        ("adder", "rem_chunk_tokens", None),
        ("adder", "rem_chunk_tokens", 0),
        ("adder", "rem_input_tokens", 0),
        ("adder", "new_chunked_req", object()),
        ("adder", "preempt_list", [object()]),
        ("running", "batch_is_full", True),
        ("pool", "mamba_allocator", object()),
    ],
)
def test_preserve_admission_guards(owner, field, value):
    s, a, r, *_ = fixture()
    target = {
        "scheduler": s, "adder": a, "running": r, "pool": s.req_to_token_pool
    }[owner]
    setattr(target, field, value)
    with mode(4):
        assert s._ascache_ready_refill_candidates(a, r) == ()
    s.tree_cache.commit_ascache_load_backs.assert_not_called()


def test_no_pending_group_never_waits_for_arrivals():
    s, a, r, *_ = fixture()
    s.tree_cache.pending_ascache_load_back_requests.return_value = ()
    with mode(4):
        assert s._ascache_ready_refill_candidates(a, r) == ()
    s.tree_cache.commit_ascache_load_backs.assert_not_called()


def test_commit_error_propagated_without_admitting():
    s, a, r, *_ = fixture()
    s.tree_cache.commit_ascache_load_backs.side_effect = RuntimeError("read failure")
    before = list(a.can_run_list)
    with mode(4), pytest.raises(RuntimeError, match="read failure"):
        s._ascache_ready_refill_candidates(a, r)
    assert a.can_run_list == before


def test_uncommitted_request_still_skipped():
    s, a, r, admitted, pending, fresh = fixture()
    s.tree_cache.commit_ascache_load_backs.side_effect = None
    with mode(4):
        assert s._ascache_ready_refill_candidates(a, r) == [fresh]
    assert pending._as_pending_load_back is not None
