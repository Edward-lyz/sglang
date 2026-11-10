"""Tokenizer->scheduler intake through the native shared-memory request ring.

Replaces, for one attention group on one node, the leader's Python zmq drain
plus the gloo broadcast of every request payload: a native thread on the
leader drains the socket into a shared-memory ring that every rank of the group
reads directly. Per poll the group only agrees on the last ring sequence to
consume (one int64 pair broadcast), plus the rare leader-local objects (RPC
requests, timeout aborts) that keep using pickle.
"""

from __future__ import annotations

import logging
import os
import pickle
import secrets
import socket
import time
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

import torch
import torch.distributed as dist

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_MODULE = None


def _load_module():
    global _MODULE
    if _MODULE is None:
        from torch.utils.cpp_extension import load

        _MODULE = load(
            name="request_shm_ring",
            sources=[str(Path(__file__).with_name("request_shm_ring.cpp"))],
            extra_cflags=["-O3", "-std=c++17"],
            extra_ldflags=["-lzmq", "-lrt"],
            with_cuda=False,
            verbose=False,
        )
    return _MODULE


class _BroadcastHop:
    """One broadcast hop: (global rank of this process, cpu group, global src)."""

    __slots__ = ("rank", "group", "src")

    def __init__(self, rank: int, group, src: int):
        self.rank, self.group, self.src = rank, group, src


class RequestRingIntake:
    """Per-rank endpoint of the request ring of one attention group."""

    def __init__(
        self,
        *,
        is_leader: bool,
        reader_index: int,
        num_readers: int,
        hops: Sequence[_BroadcastHop],
        endpoint: Optional[str],
        capacity_bytes: int,
        decode: Callable[[Any], Any],
        stats_interval_s: float,
    ):
        self.is_leader = is_leader
        self._hops = list(hops)
        self._decode = decode
        self._check_single_node()
        module = _load_module()
        name = None
        if is_leader:
            name = f"/sglang_req_{os.getpid()}_{secrets.token_hex(4)}"
            self._ring = module.RequestShmRing(
                name, capacity_bytes, num_readers, endpoint, 0
            )
        name = self._broadcast_obj(name)
        if not is_leader:
            self._ring = module.RequestShmRing(name, reader_index)
        self._barrier()
        if is_leader:
            # Every reader holds a mapping now; spill segments use their own names.
            self._ring.unlink_name()
            logger.info(
                "Request shm ring ready: name=%s capacity=%dMB readers=%d",
                name,
                capacity_bytes >> 20,
                num_readers,
            )
        self._stats_interval_s = stats_interval_s
        self._last_log = time.perf_counter()
        self._decode_s = 0.0
        self._polls = 0
        self._reqs = 0

    def recv(self, *, max_count: int, leader_extra: Optional[List]) -> List:
        """Every rank: this poll's tokenizer requests, then the leader's extras."""
        header = torch.zeros(2, dtype=torch.int64)
        extra_bytes = b""
        if self.is_leader:
            header[0] = self._ring.scan_limit(max(0, max_count))
            if leader_extra:
                extra_bytes = pickle.dumps(leader_extra, protocol=pickle.HIGHEST_PROTOCOL)
                header[1] = len(extra_bytes)
        for hop in self._hops:
            dist.broadcast(header, src=hop.src, group=hop.group)
        extra: List = []
        if header[1] > 0:
            payload = (
                torch.frombuffer(bytearray(extra_bytes), dtype=torch.uint8)
                if self.is_leader
                else torch.empty(int(header[1]), dtype=torch.uint8)
            )
            for hop in self._hops:
                dist.broadcast(payload, src=hop.src, group=hop.group)
            extra = leader_extra if self.is_leader else pickle.loads(payload.numpy().tobytes())
        started = time.perf_counter()
        views = self._ring.read(int(header[0]))
        reqs = [self._decode(view) for view in views]
        del views
        self._ring.release()
        self._account(time.perf_counter() - started, len(reqs))
        return reqs + extra

    def shutdown(self) -> None:
        """Stop the native intake thread and unmap the ring."""
        self._ring.shutdown()

    def _account(self, decode_s: float, num_reqs: int) -> None:
        if self._stats_interval_s <= 0 or not self.is_leader:
            return
        self._decode_s += decode_s
        self._polls += 1
        self._reqs += num_reqs
        now = time.perf_counter()
        if now - self._last_log < self._stats_interval_s:
            return
        stats = self._ring.stats()
        logger.info(
            "Request shm ring (%.0fs): polls=%d reqs=%d decode_ms=%.1f ring_msgs=%d "
            "ring_MB=%.1f spills=%d full_wait_ms=%.1f",
            now - self._last_log,
            self._polls,
            self._reqs,
            self._decode_s * 1000.0,
            stats["msgs"],
            stats["bytes"] / 2**20,
            stats["spills"],
            stats["full_wait_ms"],
        )
        self._last_log = now
        self._decode_s = 0.0
        self._polls = 0
        self._reqs = 0

    def _broadcast_obj(self, obj):
        data = [obj]
        for hop in self._hops:
            from sglang.srt.utils import broadcast_pyobj

            data = broadcast_pyobj(data, hop.rank, hop.group, src=hop.src)
        return data[0]

    def _barrier(self) -> None:
        for hop in self._hops:
            dist.barrier(group=hop.group)

    def _check_single_node(self) -> None:
        host = socket.gethostname()
        for hop in self._hops:
            hosts = [None] * dist.get_world_size(group=hop.group)
            dist.all_gather_object(hosts, host, group=hop.group)
            if len(set(hosts)) != 1:
                raise RuntimeError(
                    "SGLANG_ENABLE_REQUEST_SHM_RING needs every rank of the attention "
                    f"group on one node, got {sorted(set(hosts))}"
                )


def maybe_create_request_ring(
    *,
    ps,
    tp_group,
    tp_cpu_group,
    recv_from_tokenizer,
    scheduler_input_endpoint: str,
    decode: Callable[[Any], Any],
) -> Optional[RequestRingIntake]:
    """Collective over the tp group; returns None when the ring is disabled."""
    if not envs.SGLANG_ENABLE_REQUEST_SHM_RING.get():
        return None
    from sglang.srt.runtime_context import get_device, get_parallel

    parallel = get_parallel()
    unsupported = [
        reason
        for reason, bad in (
            ("pp_size > 1", ps.pp_size > 1),
            ("dp attention", parallel.enable_dp_attention),
            ("rust server", envs.SGLANG_RUST_SERVER.get()),
            ("scripted runtime", envs.SGLANG_TEST_SCRIPTED_RUNTIME.get()),
            # The idle sleeper polls the tokenizer socket the ring now owns.
            ("sleep_on_idle", get_device().sleep_on_idle),
            ("colocated batch gen input blocker", envs.SGLANG_ENABLE_COLOCATED_BATCH_GEN.get()),
        )
        if bad
    ]
    if unsupported:
        raise ValueError(
            f"SGLANG_ENABLE_REQUEST_SHM_RING does not support {', '.join(unsupported)}"
        )
    is_leader = ps.tp_rank == 0
    if is_leader:
        # The native thread owns the scheduler input socket from now on; a
        # second PULL peer would get every other request round-robin.
        recv_from_tokenizer.close(linger=0)
    hops = (
        [_BroadcastHop(tp_group.rank, tp_cpu_group, tp_group.ranks[0])]
        if ps.tp_size > 1
        else []
    )
    return RequestRingIntake(
        is_leader=is_leader,
        reader_index=ps.tp_rank,
        num_readers=ps.tp_size,
        hops=hops,
        endpoint=scheduler_input_endpoint if is_leader else None,
        capacity_bytes=envs.SGLANG_REQUEST_SHM_RING_MB.get() << 20,
        decode=decode,
        stats_interval_s=envs.SGLANG_LOG_PREFILL_SCHED_STATS_INTERVAL.get(),
    )
