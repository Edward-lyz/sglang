from types import SimpleNamespace

from sglang.srt.mem_cache.as_trace import (
    ASCacheTraceRecorder,
    ASTraceContext,
    TransferDigestCapture,
)


def test_transfer_digest_capture_is_empty_without_transport_snapshot():
    assert TransferDigestCapture().value() == {}


def test_transfer_digest_capture_preserves_transport_snapshots():
    capture = TransferDigestCapture()
    page_snapshot = {
        "kind": "page_buffers",
        "action": "write",
        "transferred": 2,
        "components": [("kv", 0, 1, [(42, ["abc123"])])],
    }
    state_snapshot = {
        "kind": "kda_state",
        "action": "write",
        "transferred": 1,
        "slots": {"checkpoint": 7},
        "components": [("temporal", 0, "def456")],
    }

    capture(page_snapshot)
    capture(state_snapshot)

    assert capture.value() == {"transfers": [page_snapshot, state_snapshot]}


def test_recorder_creates_operation_local_digest_captures():
    req = SimpleNamespace(rid="request-1")
    recorder = ASCacheTraceRecorder(req, ASTraceContext(req, "trace-1"))

    first = recorder.new_digest_capture()
    second = recorder.new_digest_capture()
    first({"kind": "page_buffers"})

    assert first.value() == {"transfers": [{"kind": "page_buffers"}]}
    assert second.value() == {}


def test_trace_context_finalizes_after_all_as_writes_retire():
    req = SimpleNamespace(rid="request-1")
    context = ASTraceContext(req, "trace-1")
    finalized = []
    context.set_finalizer(lambda events, pending: finalized.append(events))

    context.retain_write()
    context.seal_registered()
    context.record({"event": "AS_WRITE"})

    assert finalized == []

    context.release_write()

    assert finalized == [[{"event": "AS_WRITE"}]]


def test_trace_context_seals_immediately_without_as_writes():
    req = SimpleNamespace(rid="request-1")
    context = ASTraceContext(req, "trace-1")
    finalized = []
    context.set_finalizer(lambda events, pending: finalized.append(events))
    context.record({"event": "AS_READ"})

    context.seal_registered()

    assert finalized == [[{"event": "AS_READ"}]]


def test_trace_context_ignores_seal_before_trace_fires():
    # Optimistic prefill retry releases KV and requeues before the trace fires.
    req = SimpleNamespace(rid="request-1")
    context = ASTraceContext(req, "trace-1")
    finalized = []

    context.seal_registered()
    context.record({"event": "AS_READ"})
    context.set_finalizer(lambda events, pending: finalized.append(events))
    assert finalized == []

    context.seal_registered()

    assert finalized == [[{"event": "AS_READ"}]]


def test_trace_context_force_finalize_reports_pending_writes():
    req = SimpleNamespace(rid="request-1")
    context = ASTraceContext(req, "trace-1")
    finalized = []
    context.set_finalizer(lambda events, pending: finalized.append((events, pending)))
    context.retain_write()
    context.seal_registered()

    assert context.force_finalize()
    assert finalized == [([], 1)]
    assert context.finalized

    # A late retirement must not finalize a second time.
    context.release_write()
    assert not context.force_finalize()
    assert finalized == [([], 1)]
