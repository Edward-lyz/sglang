"""A just-started embedding pull must be re-checked before the loop ends.

The scheduler reaches the next loop top only after the current forward
returns, so a pull that completes a few hundred microseconds after it was
started would otherwise wait a whole forward (0.15-0.24 s measured) before the
request can enter a batch. These cases pin the three ways that guard can rot:
the poll never re-checks, the budget is read in the wrong unit (and so blocks
far too long), and the switch no longer turns the poll off.
"""

import time
import unittest

from sglang.srt.disaggregation.encoder.receiver import (
    MMReceiverBase,
    WaitingMMRequestStatus,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _Waiting:
    """Becomes ready on the ``ready_after``-th poll; never ready when None."""

    def __init__(self, ready_after):
        self.ready_after = ready_after
        self.polls = 0
        self.status = WaitingMMRequestStatus.PENDING

    def _try_recv_mm_data(self):
        self.polls += 1
        if self.ready_after is not None and self.polls >= self.ready_after:
            self.status = WaitingMMRequestStatus.SUCCESS


class _Receiver:
    def __init__(self, waiting_list):
        self.waiting_list = waiting_list
        self.drains = 0

    def _drain_scheduler_embeddings(self):
        self.drains += 1

    poll = MMReceiverBase._poll_pending_embeddings


class TestPollPendingEmbeddings(CustomTestCase):
    def _run(self, waiting_list, budget_ms):
        receiver = _Receiver(waiting_list)
        with envs.SGLANG_EPD_PULL_WAIT_MS.override(budget_ms):
            start = time.monotonic()
            receiver.poll()
            return receiver, time.monotonic() - start

    def test_pull_completing_after_several_polls_is_picked_up(self):
        waiting = _Waiting(ready_after=3)
        receiver, _ = self._run([waiting], 50.0)
        self.assertEqual(waiting.status, WaitingMMRequestStatus.SUCCESS)
        self.assertGreaterEqual(waiting.polls, 3)
        # The shared-socket landing path only runs inside the drain.
        self.assertGreaterEqual(receiver.drains, 3)

    def test_budget_is_milliseconds_and_bounds_a_stuck_pull(self):
        waiting = _Waiting(ready_after=None)
        _, elapsed = self._run([waiting], 20.0)
        self.assertEqual(waiting.status, WaitingMMRequestStatus.PENDING)
        self.assertLess(elapsed, 1.0)

    def test_zero_budget_does_not_poll(self):
        waiting = _Waiting(ready_after=1)
        receiver, _ = self._run([waiting], 0.0)
        self.assertEqual(waiting.polls, 0)
        self.assertEqual(receiver.drains, 0)
        self.assertEqual(waiting.status, WaitingMMRequestStatus.PENDING)

    def test_settled_requests_are_not_polled_again(self):
        done = _Waiting(ready_after=None)
        done.status = WaitingMMRequestStatus.FAIL
        receiver, _ = self._run([done], 20.0)
        self.assertEqual(done.polls, 0)
        self.assertEqual(receiver.drains, 0)


if __name__ == "__main__":
    unittest.main()
