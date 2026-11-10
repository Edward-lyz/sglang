"""Regression tests for `sglang:num_aborted_requests_total` over-counting.

Bug: `create_abort_task` is attached as a FastAPI background task, so it runs
after *every* streaming response — including ones that completed normally. The
`rid not in rid_to_state` short-circuit in `abort_request` that was supposed to
drop those was gated on `tokenizer_worker_num == 1`, so with several tokenizer
workers every finished request sent a no-op AbortReq to the scheduler AND bumped
the abort counter. Observed on a `--tokenizer-worker-num 8` decode node:
aborted == requests == 697 with zero real failures.
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeTokenizerManager:
    """Minimal stand-in exposing only what `abort_request` touches.

    Constructing a real TokenizerManager needs a model and live IPC sockets; the
    methods under test only read `server_args.tokenizer_worker_num` and
    `rid_to_state`, and delegate to `_dispatch_to_scheduler`.
    """

    def __init__(self, *, tokenizer_worker_num: int, live_rids=()):
        self.server_args = SimpleNamespace(tokenizer_worker_num=tokenizer_worker_num)
        self.rid_to_state = {rid: object() for rid in live_rids}
        self.enable_metrics = True
        self.metrics_collector = MagicMock()
        self.dispatched = []

    def _dispatch_to_scheduler(self, obj):
        self.dispatched.append(obj)


def _make_tm(*, tokenizer_worker_num: int, live_rids=()) -> _FakeTokenizerManager:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

    tm = _FakeTokenizerManager(
        tokenizer_worker_num=tokenizer_worker_num, live_rids=live_rids
    )
    tm.abort_request = TokenizerManager.abort_request.__get__(tm)
    tm.create_abort_task = TokenizerManager.create_abort_task.__get__(tm)
    return tm


class TestAbortRequestOnlyIfPending(CustomTestCase):
    def test_finished_request_not_aborted_with_multiple_workers(self):
        """only_if_pending must drop an already-finished rid even when
        tokenizer_worker_num > 1 — the pre-fix gate ignored it and dispatched."""
        tm = _make_tm(tokenizer_worker_num=8, live_rids=())
        tm.abort_request("gone-rid", only_if_pending=True)
        self.assertEqual(tm.dispatched, [])

    def test_pending_request_still_aborted_with_multiple_workers(self):
        """A rid still in flight must reach the scheduler."""
        tm = _make_tm(tokenizer_worker_num=8, live_rids=("live-rid",))
        tm.abort_request("live-rid", only_if_pending=True)
        self.assertEqual(len(tm.dispatched), 1)
        self.assertEqual(tm.dispatched[0].rid, "live-rid")

    def test_cross_process_abort_not_dropped(self):
        """Without only_if_pending (e.g. POST /abort_request) a rid absent from
        THIS worker's table must still be forwarded: it may belong to a sibling
        tokenizer worker."""
        tm = _make_tm(tokenizer_worker_num=8, live_rids=())
        tm.abort_request("other-worker-rid")
        self.assertEqual(len(tm.dispatched), 1)

    def test_single_worker_drops_finished_request_either_way(self):
        """With one worker the local table is authoritative, so a finished rid is
        dropped regardless of only_if_pending."""
        for only_if_pending in (False, True):
            tm = _make_tm(tokenizer_worker_num=1, live_rids=())
            tm.abort_request("gone-rid", only_if_pending=only_if_pending)
            self.assertEqual(tm.dispatched, [], f"{only_if_pending=}")

    def test_abort_request_does_not_count_metrics(self):
        """Requesting an abort is not the same as a request being aborted; the
        counter is bumped only where the abort takes effect."""
        tm = _make_tm(tokenizer_worker_num=8, live_rids=("live-rid",))
        tm.abort_request("live-rid")
        tm.metrics_collector.observe_one_aborted_request.assert_not_called()

    def test_abort_all_bypasses_the_pending_check(self):
        """abort_all has no rid to look up and must always be forwarded."""
        tm = _make_tm(tokenizer_worker_num=8, live_rids=())
        tm.abort_request(abort_all=True, only_if_pending=True)
        self.assertEqual(len(tm.dispatched), 1)
        self.assertTrue(tm.dispatched[0].abort_all)

    def test_create_abort_task_passes_only_if_pending(self):
        """The disconnect-fallback task fires for normally-finished streaming
        responses too, so it must opt into the pending check."""
        tm = _make_tm(tokenizer_worker_num=8, live_rids=())
        obj = SimpleNamespace(is_single=True, rid="finished-rid")
        background = tm.create_abort_task(obj)
        for task in background.tasks:
            asyncio.run(task.func(*task.args, **task.kwargs))
        self.assertEqual(tm.dispatched, [])

    def test_create_abort_task_still_aborts_a_live_request(self):
        """Its actual purpose — a client that disconnected mid-stream leaves the
        rid in flight, and that abort must go through."""
        tm = _make_tm(tokenizer_worker_num=8, live_rids=("running-rid",))
        obj = SimpleNamespace(is_single=True, rid="running-rid")
        background = tm.create_abort_task(obj)
        for task in background.tasks:
            asyncio.run(task.func(*task.args, **task.kwargs))
        self.assertEqual(len(tm.dispatched), 1)


if __name__ == "__main__":
    unittest.main()
