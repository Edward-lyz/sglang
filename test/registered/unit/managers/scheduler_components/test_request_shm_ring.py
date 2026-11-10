import os
import pickle
import threading
import time
import unittest

import torch.multiprocessing as mp
import zmq

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

_CAPACITY = 4 << 20  # small enough to force wraps, writer back-pressure and spills


def _payload(i: int) -> bytes:
    size = 2_000_000 if i % 97 == 0 else (i * 7919) % 200_000 + 1
    return pickle.dumps({"i": i, "blob": bytes([i % 251]) * size})


def _check(raw) -> int:
    record = pickle.loads(raw)
    assert record["blob"] == bytes([record["i"] % 251]) * len(record["blob"])
    return record["i"]


def _follower(name, reader_index, upto_q, result_q):
    from sglang.srt.managers.scheduler_components.request_ring import _load_module

    ring = _load_module().RequestShmRing(name, reader_index)
    seen = []
    while (upto := upto_q.get()) >= 0:
        views = ring.read(upto)
        seen.extend(_check(view) for view in views)
        del views
        ring.release()
    result_q.put(seen)


class TestRequestShmRing(CustomTestCase):
    def test_every_reader_sees_the_same_ordered_stream(self):
        """All ranks must decode the same requests in the same order.

        Covers ring wrap-around, a writer blocked on the slowest reader, and
        frames above capacity/4 spilled to their own segment; a lost, reordered
        or torn frame would desynchronize the ranks' waiting queues.
        """
        from sglang.srt.managers.scheduler_components.request_ring import _load_module

        module = _load_module()
        endpoint = f"ipc:///tmp/sglang_ring_test_{os.getpid()}"
        context = zmq.Context()
        push = context.socket(zmq.PUSH)
        push.setsockopt(zmq.SNDHWM, 0)
        push.bind(endpoint)
        name = f"/sglang_req_test_{os.getpid()}"
        leader = module.RequestShmRing(name, _CAPACITY, 3, endpoint, 0)
        spawn = mp.get_context("spawn")
        upto_qs = [spawn.Queue() for _ in range(2)]
        result_q = spawn.Queue()
        followers = [
            spawn.Process(target=_follower, args=(name, k + 1, upto_qs[k], result_q))
            for k in range(2)
        ]
        for proc in followers:
            proc.start()
        total = 2000
        sender = threading.Thread(target=lambda: [push.send(_payload(i)) for i in range(total)])
        sender.start()
        seen = []
        deadline = time.time() + 120
        while len(seen) < total and time.time() < deadline:
            upto = leader.scan_limit(64)
            for q in upto_qs:
                q.put(upto)
            views = leader.read(upto)
            seen.extend(_check(view) for view in views)
            del views
            leader.release()
            time.sleep(0.001)
        for q in upto_qs:
            q.put(-1)
        results = [result_q.get(timeout=60) for _ in followers]
        sender.join()
        for proc in followers:
            proc.join(timeout=60)
        stats = leader.stats()
        leader.unlink_name()
        leader.shutdown()
        push.close(linger=0)
        context.term()

        self.assertEqual(seen, list(range(total)))
        self.assertEqual(results, [seen, seen])
        self.assertGreater(stats["spills"], 0)
        self.assertGreater(stats["full_wait_ms"], 0)
        leftovers = [f for f in os.listdir("/dev/shm") if f.startswith(name[1:])]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
