"""Batched startup sync must admit and abort exactly what the per-request one did.

`waiting_list` is indexed positionally by the status all_reduce that follows,
so every rank has to append the same requests in the same order. Collapsing
the per-request `all_gather` into one per arrival batch moves that decision
into `_settle_started_requests`; if it dropped a request, mixed up the order,
or missed a peer rank's failure, the ranks would disagree about the list and
one prompt's embeddings would be read for another request. These cases pin the
order, the per-index failure attribution, and the release of a failed request.
"""

import unittest
from http import HTTPStatus
from types import SimpleNamespace

from sglang.srt.disaggregation.encoder.receiver import MMReceiverBase
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _Waiting:
    def __init__(self, rid):
        self.rid = rid
        self.released = 0
        self.closed = 0

    def release_resources(self):
        self.released += 1

    def close_recv_socket(self):
        self.closed += 1


class _Receiver:
    """Stands in for the receiver with a scripted all_gather of error lists."""

    def __init__(self, tp_size, gathered=None):
        self.tp_size = tp_size
        self.waiting_list = []
        self.waiting_by_rid = {}
        self.gathers = 0
        self._gathered = gathered
        self.tp_group = SimpleNamespace(all_gather_object=self._all_gather)

    def _all_gather(self, local):
        self.gathers += 1
        return self._gathered if self._gathered is not None else [local]

    def create_req(self, recv_req):
        return recv_req

    settle = MMReceiverBase._settle_started_requests
    _discard_started_request = MMReceiverBase._discard_started_request


class TestSettleStartedRequests(CustomTestCase):
    def test_one_gather_for_the_whole_batch_in_arrival_order(self):
        waiting = [_Waiting(f"r{i}") for i in range(3)]
        receiver = _Receiver(
            tp_size=2, gathered=[[None, None, None], [None, None, None]]
        )
        started = [(f"req{i}", waiting[i], None) for i in range(3)]
        aborts = []
        receiver.settle(started, aborts)
        self.assertEqual([w.rid for w in receiver.waiting_list], ["r0", "r1", "r2"])
        self.assertEqual(aborts, [])
        # Three requests, one collective.
        self.assertEqual(receiver.gathers, 1)

    def test_single_rank_needs_no_collective(self):
        waiting = [_Waiting("r0")]
        receiver = _Receiver(tp_size=1)
        receiver.settle([("req0", waiting[0], None)], [])
        self.assertEqual([w.rid for w in receiver.waiting_list], ["r0"])
        self.assertEqual(receiver.gathers, 0)

    def test_peer_rank_failure_aborts_only_that_request(self):
        waiting = [_Waiting(f"r{i}") for i in range(3)]
        # Rank 0 started all three; rank 1 failed the middle one.
        receiver = _Receiver(
            tp_size=2, gathered=[[None, None, None], [None, "boom", None]]
        )
        receiver.waiting_by_rid = {w.rid: w for w in waiting}
        started = [(f"req{i}", waiting[i], None) for i in range(3)]
        aborts = []
        receiver.settle(started, aborts)
        self.assertEqual([w.rid for w in receiver.waiting_list], ["r0", "r2"])
        self.assertEqual(len(aborts), 1)
        request, message, status = aborts[0]
        self.assertEqual(request, "req1")
        self.assertIn("rank 1: boom", message)
        self.assertEqual(status, HTTPStatus.INTERNAL_SERVER_ERROR)
        # The failed request must give back its buffer and socket.
        self.assertEqual((waiting[1].released, waiting[1].closed), (1, 1))
        self.assertNotIn("r1", receiver.waiting_by_rid)
        self.assertEqual((waiting[0].released, waiting[0].closed), (0, 0))

    def test_local_failure_with_no_waiting_object_still_aborts(self):
        receiver = _Receiver(tp_size=1, gathered=[["boom"]])
        aborts = []
        receiver.settle([("req0", None, "boom")], aborts)
        self.assertEqual(receiver.waiting_list, [])
        self.assertEqual(len(aborts), 1)

    def test_empty_batch_skips_the_gather(self):
        receiver = _Receiver(tp_size=2)
        receiver.settle([], [])
        self.assertEqual(receiver.gathers, 0)


if __name__ == "__main__":
    unittest.main()
