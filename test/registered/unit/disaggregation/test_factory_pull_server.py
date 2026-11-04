"""Factory (厂内) pull ROUTER wire, exercised in-process (no GPU).

Drives PullZmqServer with a stub encoder holding one embedding and the real
prefill-side parser (FactoryZmqTransport._parse_reply), so the [meta, buffer]
framing, tensor round-trip and miss handling are all checked end to end.
"""

import asyncio
import json

import torch
import zmq
import zmq.asyncio

from sglang.srt.disaggregation.encoder.factory_transport import FactoryZmqTransport
from sglang.srt.disaggregation.encoder.pull_zmq_server import PullZmqServer
from sglang.srt.disaggregation.encoder.server import WorkflowTaskEmbedding


class _StubEncoder:
    """Just the task-embedding hold interface PullZmqServer needs."""

    def __init__(self, store):
        """Hold a task_id -> WorkflowTaskEmbedding map and record releases."""
        self._store = store
        self.released = []

    async def acquire_task_embedding(self, task_id):
        """Take a ref on the held embedding, mirroring the real encoder."""
        entry = self._store.get(task_id)
        if entry is not None:
            entry.refs += 1
        return entry

    async def release_task_embedding(self, task_id):
        """Drop a ref and record the release for the assertion."""
        self.released.append(task_id)
        entry = self._store.get(task_id)
        if entry is not None:
            entry.refs -= 1


async def _pull(ctx, port, task_id):
    """One DEALER pull, parsed with the real prefill-side reply parser."""
    sock = ctx.socket(zmq.DEALER)
    sock.connect(f"tcp://127.0.0.1:{port}")
    await sock.send_multipart(
        [b"", json.dumps({"task_id": task_id, "req_id": "r1"}).encode()]
    )
    frames = await asyncio.wait_for(sock.recv_multipart(), timeout=5)
    sock.close()
    # DEALER receives [empty, task_id, req_id, meta, (buffer)]
    return FactoryZmqTransport._parse_reply(frames[1].decode(), frames[3:])


async def _run():
    """Start the pull server, pull a held task and a missing one."""
    emb = torch.arange(502 * 8, dtype=torch.bfloat16).reshape(502, 8)
    store = {
        "t-hit": WorkflowTaskEmbedding(
            embedding=emb, image_grid_thw=[[1, 74, 55]], original_image_sizes=None
        )
    }
    encoder = _StubEncoder(store)
    ctx = zmq.asyncio.Context.instance()
    probe = ctx.socket(zmq.ROUTER)
    port = probe.bind_to_random_port("tcp://*")
    probe.close()
    server = PullZmqServer(encoder, ctx, port)
    server_task = asyncio.create_task(server.run())
    await asyncio.sleep(0.3)
    try:
        hit = await _pull(ctx, port, "t-hit")
        assert hit.success, hit.error_msg
        assert tuple(hit.embedding.shape) == (502, 8)
        assert torch.equal(hit.embedding, emb)
        assert hit.grid_thw.tolist() == [[1, 74, 55]]
        assert encoder.released == ["t-hit"]
        assert store["t-hit"].refs == 0

        miss = await _pull(ctx, port, "t-miss")
        assert not miss.success
        assert "not held" in miss.error_msg
    finally:
        server_task.cancel()


def test_pull_router_round_trip_and_miss():
    """A held embedding round-trips; an unknown task_id is a clean error."""
    asyncio.run(_run())
