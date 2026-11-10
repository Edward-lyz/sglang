"""Factory (厂内) /epd/transfer HTTP contract, exercised without RDMA.

Uses a stub encoder whose transfer_task_embedding records its args (no real
mooncake), drives the route with Starlette's TestClient, and parses the ack with
the real prefill-side epd_transfer_error so the request/ack wire is pinned.
"""

from http import HTTPStatus

import pytest
from starlette.testclient import TestClient

import sglang.srt.disaggregation.encoder.http_server as hs
from sglang.srt.disaggregation.encoder.factory_transport import epd_transfer_error
from sglang.srt.disaggregation.encoder.server import MMError


class _StubEncoder:
    """Records transfer args instead of doing a real mooncake RDMA write."""

    def __init__(self, missing=()):
        """``missing`` task ids raise NOT_FOUND, as a released hold would."""
        self.calls = []
        self._missing = set(missing)

    async def transfer_task_embedding(
        self, task_id, session_id, buffer_address, byte_size
    ):
        """Record the pull-triggered-push args; raise for a missing hold."""
        self.calls.append((task_id, session_id, buffer_address, byte_size))
        if task_id in self._missing:
            raise MMError(
                f"embedding not held for task_id={task_id}", code=HTTPStatus.NOT_FOUND
            )
        return 1.5, None  # (transfer_ms, embed_digest)


def _payload(task_id):
    """The /epd/transfer body prefill POSTs (factory mooncake pull)."""
    return {
        "req_id": "r-1",
        "task_id": task_id,
        "part_idx": 0,
        "num_parts": 1,
        "session_id": "10.0.0.9:62354",
        "buffer_address": 140234567890432,
        "byte_size": 502 * 4096 * 2,
        "dtype": "bfloat16",
        "shape": [502, 4096],
        "modality": "image",
    }


def test_epd_transfer_success_acks_and_forwards_args():
    """A held task transfers to prefill's session/buffer and acks success."""
    stub = _StubEncoder()
    hs.encoder = stub
    try:
        resp = TestClient(hs.app).post("/epd/transfer", json=_payload("t-hit"))
    finally:
        hs.encoder = None
    assert resp.status_code == 200
    assert epd_transfer_error(resp.json()) is None
    assert stub.calls == [("t-hit", "10.0.0.9:62354", 140234567890432, 502 * 4096 * 2)]


def test_epd_transfer_miss_is_a_parseable_error():
    """A missing hold surfaces as an error the prefill client can read."""
    stub = _StubEncoder(missing={"t-miss"})
    hs.encoder = stub
    try:
        resp = TestClient(hs.app).post("/epd/transfer", json=_payload("t-miss"))
    finally:
        hs.encoder = None
    assert resp.status_code == int(HTTPStatus.NOT_FOUND)
    assert "not held" in epd_transfer_error(resp.json())
