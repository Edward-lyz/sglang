"""Robustness tests for the disaggregation ZMQ control plane.

Regression cover for the 2026-09-08 GLM-53 decode freeze. A bootstrap PUSH socket
carried no SNDTIMEO, so once a prefill rank stopped draining its rank port the
send inside MooncakeKVReceiver.init() blocked forever on the decode scheduler's
main loop. With DP attention every rank steps in lockstep, so all 8 froze behind
it, with no exception and no watchdog trip.

These tests pin down the two guarantees of the fix:
  1. Every disaggregation PUSH socket is bounded, so send() cannot block forever.
  2. A timed-out send fails exactly one request; it never raises out of the
     scheduler loop and never wedges a transfer_worker.

CPU only: the heavyweight __init__ of the manager/receiver is bypassed on
purpose, so no GPU, no mooncake engine and no real prefill peer is needed.
"""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import zmq

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import (
    DISAGG_SEND_HWM,
    DISAGG_SEND_TIMEOUT_MS,
    configure_bounded_push_socket,
)
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVManager,
    MooncakeKVReceiver,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-cpu-only")

ABORT_HEADER = MooncakeKVManager.ABORT_HEADER

# Real dtype matters: init() calls kv_indices.tobytes(), so handing it bytes would
# fail on the attribute access and make the failure tests pass for the wrong reason.
KV_INDICES = np.array([0, 1, 2, 3], dtype=np.int32)

class _AgainSocket:
    """Stand-in for a PUSH socket whose peer never drains: every send times out."""

    def __init__(self):
        self.sent = []

    def send_multipart(self, parts, flags=0):
        raise zmq.error.Again("Resource temporarily unavailable")


class _OkSocket:
    """Stand-in for a healthy PUSH socket; records what was pushed."""

    def __init__(self):
        self.sent = []

    def send_multipart(self, parts, flags=0):
        self.sent.append(parts)

    def frames_of_type(self, is_abort: bool):
        return [p for p in self.sent if (p[0] == ABORT_HEADER) is is_abort]


class _FakeKVManager:
    """Minimal kv_mgr: records the failure/status calls the fix relies on."""

    def __init__(self):
        self.failures = []
        self.statuses = []
        self.abort_expectations = []
        self.local_ip = "10.0.0.1"
        self.rank_port = 7100
        self.attn_tp_size = 1
        self.registration_cooldown_lock = threading.Lock()
        self.registration_cooldown_until = {}
        self.registration_refresh_lock = threading.Lock()
        self.registration_refresh_last_sent = {}
        self.kv_args = SimpleNamespace(
            kv_data_ptrs=[0x1000],
            aux_data_ptrs=[0x2000],
            state_data_ptrs=[],
            state_item_lens=[],
            state_dim_per_tensor=[],
            engine_rank=3,
            kv_item_lens=[512],
        )

    def record_failure(self, room, message):
        self.failures.append((room, message))

    def update_status(self, room, status):
        self.statuses.append((room, status))

    def expect_abort_acks(self, room, count):
        self.abort_expectations.append((room, count))


def _make_receiver(sockets):
    """Build a MooncakeKVReceiver without running its heavyweight __init__."""
    receiver = object.__new__(MooncakeKVReceiver)
    receiver.kv_mgr = _FakeKVManager()
    receiver.session_id = "sess-1"
    receiver.bootstrap_room = 4242
    receiver.bootstrap_addr = "10.0.0.9:8200"
    receiver.required_dst_info_num = len(sockets)
    receiver.init_time = None
    receiver.abort_notified = False
    receiver.conclude_state = None
    receiver.bootstrap_infos = [
        {"rank_ip": "10.0.0.16", "rank_port": 7000 + i, "is_dummy": False}
        for i in range(len(sockets))
    ]
    by_port = {7000 + i: sock for i, sock in enumerate(sockets)}
    locks = {7000 + i: threading.Lock() for i in range(len(sockets))}
    receiver._connect_to_bootstrap_server = lambda info: (
        by_port[info["rank_port"]],
        locks[info["rank_port"]],
    )
    return receiver


def _make_manager(socket):
    manager = object.__new__(MooncakeKVManager)
    manager.attn_tp_rank = 0
    manager.pp_size = 1
    manager.pp_rank = 0
    manager.kv_args = SimpleNamespace(aux_data_ptrs=[0x2000], aux_item_lens=[8])
    manager._connect = lambda endpoint, is_ipv6=False: socket
    return manager


def test_configure_bounded_push_socket_sets_bounds():
    ctx = zmq.Context()
    try:
        sock = configure_bounded_push_socket(ctx.socket(zmq.PUSH))
        try:
            assert sock.getsockopt(zmq.SNDTIMEO) == DISAGG_SEND_TIMEOUT_MS
            assert sock.getsockopt(zmq.SNDHWM) == DISAGG_SEND_HWM
            # LINGER=0 so an undeliverable queued message cannot hold up teardown.
            assert sock.getsockopt(zmq.LINGER) == 0
        finally:
            sock.close()
    finally:
        ctx.term()


def test_bounded_push_send_gives_up_instead_of_blocking_forever():
    """The physical basis of the fix, against a real socket and a dead peer.

    Before the fix this loop never returned. A PUSH socket buffers optimistically
    even when nothing is listening, so the failure is not immediate: sends queue
    until SNDHWM is reached and only then block -- which is exactly why the
    incident looked like a sudden freeze after a period of normal operation.
    """
    ctx = zmq.Context()
    try:
        sock = configure_bounded_push_socket(ctx.socket(zmq.PUSH))
        # Shrink both bounds so saturation is reached in a test-friendly time.
        sock.setsockopt(zmq.SNDTIMEO, 200)
        sock.setsockopt(zmq.SNDHWM, 4)
        try:
            sock.connect("tcp://127.0.0.1:1")  # nothing listening, never drains
            queued = 0
            started = time.monotonic()
            with pytest.raises(zmq.error.Again):
                for _ in range(10_000):
                    sock.send_multipart([b"payload"])
                    queued += 1
            elapsed = time.monotonic() - started
        finally:
            sock.close()
    finally:
        ctx.term()

    assert queued < 10_000, "send should have started refusing once SNDHWM was hit"
    assert elapsed < 10, f"send should be bounded by SNDTIMEO, took {elapsed:.1f}s"


def test_init_never_raises_and_marks_the_request_failed():
    """init() runs in pop_preallocated(), which has no exception handling."""
    receiver = _make_receiver([_AgainSocket(), _AgainSocket()])

    receiver.init(kv_indices=KV_INDICES, aux_index=0)

    assert (receiver.bootstrap_room, KVPoll.Failed) in receiver.kv_mgr.statuses
    assert receiver.kv_mgr.failures, "the failure must be recorded for the poller"
    # The log/failure message has to name the peer, otherwise the next incident is
    # undiagnosable again.
    assert "10.0.0.16:7000" in receiver.kv_mgr.failures[0][1]
    # init_time stays unset: leaving it set would start the waiting timeout clock
    # for a request that never got off the ground.
    assert receiver.init_time is None


def test_init_skips_abort_when_no_send_landed():
    """No rank knows our page indices, so the pages are safe to release at once."""
    receiver = _make_receiver([_AgainSocket(), _AgainSocket()])

    receiver.init(kv_indices=KV_INDICES, aux_index=0)

    assert receiver.abort_notified is False
    assert receiver.kv_mgr.abort_expectations == []
    assert receiver.is_kv_safe_to_release() is True


def test_init_aborts_and_stops_at_the_first_failure_when_a_send_landed():
    """Rank 0 got our indices, so it must be told to stop before we free pages."""
    healthy, stalled, untouched = _OkSocket(), _AgainSocket(), _OkSocket()
    receiver = _make_receiver([healthy, stalled, untouched])

    receiver.init(kv_indices=KV_INDICES, aux_index=0)

    assert (receiver.bootstrap_room, KVPoll.Failed) in receiver.kv_mgr.statuses
    assert receiver.abort_notified is True
    assert healthy.frames_of_type(is_abort=True), "rank 0 must receive the ABORT"

    # init() breaks at the first failure: paying one SNDTIMEO per remaining rank
    # would stall the scheduler loop for seconds on an already-doomed request.
    assert untouched.frames_of_type(is_abort=False) == []
    # But the abort still fans out to every paired rank, and all of them are
    # registered as expected acks, so the quarantine holds the pages.
    assert untouched.frames_of_type(is_abort=True)
    assert receiver.kv_mgr.abort_expectations == [(receiver.bootstrap_room, 3)]


def test_init_succeeds_normally_when_every_peer_is_healthy():
    sockets = [_OkSocket(), _OkSocket()]
    receiver = _make_receiver(sockets)

    receiver.init(kv_indices=KV_INDICES, aux_index=0)

    assert receiver.kv_mgr.failures == []
    assert receiver.kv_mgr.statuses == []
    assert receiver.abort_notified is False
    assert receiver.init_time is not None
    for sock in sockets:
        assert len(sock.frames_of_type(is_abort=False)) == 1


def test_register_kv_args_raises_a_diagnosable_error():
    """Here raising is right: DecodePreallocQueue.add() aborts just this request."""
    receiver = _make_receiver([_AgainSocket()])

    with pytest.raises(RuntimeError) as excinfo:
        receiver._register_kv_args()

    message = str(excinfo.value)
    assert "10.0.0.16:7000" in message
    assert str(DISAGG_SEND_TIMEOUT_MS) in message
    # The zmq.Again is preserved as the cause for anyone reading the traceback.
    assert isinstance(excinfo.value.__cause__, zmq.error.Again)


def test_register_kv_args_cools_down_so_later_requests_fail_fast():
    """Bounding the send is not enough: it must not be paid once per request.

    Both _register_kv_args callers run on the decode scheduler loop and are driven
    per request, so without a cooldown a dead prefill costs one SNDTIMEO per
    request -- the engine keeps passing health checks while its throughput dies.
    """
    receiver = _make_receiver([_AgainSocket()])

    with pytest.raises(RuntimeError, match="Timed out"):
        receiver._register_kv_args()

    def _must_not_connect(info):
        pytest.fail("cooldown must short-circuit before touching the socket")

    receiver._connect_to_bootstrap_server = _must_not_connect
    with pytest.raises(RuntimeError, match="cooldown"):
        receiver._register_kv_args()


def test_successful_registration_leaves_no_cooldown():
    receiver = _make_receiver([_OkSocket()])
    receiver.kv_mgr.registration_cooldown_until[receiver._endpoint_key()] = 0.0

    receiver._register_kv_args()

    assert receiver.kv_mgr.registration_cooldown_until == {}


def test_refresh_registration_records_even_a_failed_attempt():
    """pop_transferred() calls this for every Failed request in the queue."""
    receiver = _make_receiver([_AgainSocket()])

    with pytest.raises(RuntimeError):
        receiver.refresh_registration()

    assert (
        receiver.kv_mgr.registration_refresh_last_sent
    ), "a failed refresh must still be recorded, or the dedup window never engages"
    # The next Failed request in the same batch is deduped instead of paying again.
    assert receiver.refresh_registration() is False


def test_sync_status_to_decode_endpoint_swallows_a_dead_peer():
    """Runs on a transfer_worker thread and loops over dst ranks: must not raise."""
    manager = _make_manager(_AgainSocket())

    result = manager.sync_status_to_decode_endpoint(
        "10.0.0.42", 7001, room=4242, status=KVPoll.Success, prefill_rank=0
    )

    assert result is False


def test_send_aux_data_to_endpoint_reports_failure_without_raising():
    manager = _make_manager(_AgainSocket())

    assert (
        manager.send_aux_data_to_endpoint(
            remote="10.0.0.42", dst_port=7001, room=4242, buffer_index=0,
            aux_index=0, data=b"aux",
        )
        is False
    )


def test_send_aux_tcp_turns_a_failed_push_into_a_return_code(monkeypatch):
    """A return code keeps the per-rank loop in transfer_worker intact."""
    monkeypatch.setattr(
        "sglang.srt.disaggregation.mooncake.conn.AuxDataCodec",
        SimpleNamespace(serialize_data_from_buffer=lambda addr, length: b"aux"),
    )
    manager = _make_manager(_AgainSocket())
    req = SimpleNamespace(endpoint="10.0.0.42", dst_port=7001, room=4242, dst_aux_index=0)

    assert manager.send_aux_tcp(req, prefill_aux_index=0, dst_aux_ptrs=[0x3000]) == -1




