"""Factory (厂内) EPD encoder ops: hold reuse, K3 cache listing, flush, health gate.

Binds the real MMEncoder / http_server functions to minimal stand-ins (no GPU,
no model, no mooncake engine) so the behaviour that only shows up on real
traffic -- a repeated task_id served without re-encoding, the vit cache
collector's listing format, the preStop drain flag -- is checked directly.
"""

import asyncio
import os
import types
from collections import OrderedDict
from unittest import mock

import pytest
import torch

from sglang.srt.disaggregation import encode_health
from sglang.srt.disaggregation.encoder import http_server
from sglang.srt.disaggregation.encoder.server import MMEncoder


class _Holder:
    """Minimal object carrying just the state the hold methods touch."""

    def __init__(self):
        """Bind the real MMEncoder hold/listing methods onto this stand-in."""
        self.task_embeddings = OrderedDict()
        self.task_embeddings_lock = asyncio.Lock()
        self.task_embeddings_bytes = 0
        self.task_embeddings_max_bytes = 0
        self.task_embedding_hits = 0
        self.use_mooncake = True
        self.engine = None
        self.server_args = types.SimpleNamespace(port=8005)
        self.encode_calls = 0
        for name in (
            "store_task_embedding",
            "acquire_task_embedding",
            "release_task_embedding",
            "_drop_task_embedding_locked",
            "_evict_stale_task_embeddings_locked",
            "_evict_to_fit_locked",
            "_mark_hold_reserved_locked",
            "_check_hold_pool_budget",
            "_reuse_held_task_embedding",
            "_advertised_http_port",
            "list_held_embeddings",
            "flush_task_embeddings",
            "encode_and_store_for_workflow",
            "_encode_workflow_task",
            "_factory_batcher",
            "_record_workflow_cache",
            "_task_row_bytes",
        ):
            setattr(self, name, getattr(MMEncoder, name).__get__(self, _Holder))

    async def encode(
        self, mm_items, modality, req_id, num_parts, part_idx, out_staged, task_id=None
    ):
        """Stand-in for the vision tower: count calls, stage a fixed embedding."""
        self.encode_calls += 1
        out_staged.append(
            types.SimpleNamespace(
                embedding=torch.ones((6, 4), dtype=torch.bfloat16),
                grid_dim=torch.tensor([[1, 4, 4]]),
                original_image_sizes=None,
            )
        )

    async def release_request(self, req_id):
        """No push-side ReqState in the stand-in."""


def _run(coro):
    return asyncio.run(coro)


def _prefix_cache(enabled):
    return mock.patch(
        "sglang.srt.disaggregation.encoder.server.get_mm",
        return_value=types.SimpleNamespace(enable_prefix_mm_cache=enabled),
    )


def test_repeated_task_id_is_served_from_the_hold_without_re_encoding():
    """With --enable-prefix-mm-cache a held task_id must not re-run the ViT.

    Matches K3's mm_cache.get(task_id); the New EPD prefix cache is bypassed on
    the mooncake backend, so without this a repeated image re-encoded each time.
    """
    from sglang.srt.managers.schedule_batch import Modality

    async def run():
        h = _Holder()
        first = await h.encode_and_store_for_workflow(["u"], "r1", "t1", Modality.IMAGE)
        second = await h.encode_and_store_for_workflow(["u"], "r2", "t1", Modality.IMAGE)
        assert h.encode_calls == 1
        assert h.task_embedding_hits == 1
        assert first[0] == second[0] == 6 and second[3] == [[1, 4, 4]]
        await h.encode_and_store_for_workflow(["v"], "r3", "t2", Modality.IMAGE)
        assert h.encode_calls == 2

    with _prefix_cache(True):
        _run(run())


def test_hold_reuse_reports_k3_cache_metrics():
    """A task served from its hold is a workflow cache hit.

    The miss is recorded by encode()'s prefix mm_cache lookup, not here.
    """
    from sglang.srt.managers.schedule_batch import Modality

    calls = []
    collector = types.SimpleNamespace(
        record_cache_hit=lambda n, source: calls.append(("hit", n, source)),
        record_cache_miss=lambda n, source: calls.append(("miss", n, source)),
    )

    async def run():
        h = _Holder()
        h.metrics_collector = collector
        await h.encode_and_store_for_workflow(["u"], "r1", "t1", Modality.IMAGE)
        await h.encode_and_store_for_workflow(["u"], "r2", "t1", Modality.IMAGE)

    with _prefix_cache(True):
        _run(run())
    assert calls == [("hit", 6 * 4 * 2, "workflow")]

def test_without_prefix_mm_cache_every_task_is_encoded():
    """Cache off (no --enable-prefix-mm-cache): encode every time, like K3."""
    from sglang.srt.managers.schedule_batch import Modality

    async def run():
        h = _Holder()
        await h.encode_and_store_for_workflow(["u"], "r1", "t1", Modality.IMAGE)
        await h.encode_and_store_for_workflow(["u"], "r2", "t1", Modality.IMAGE)
        assert h.encode_calls == 2

    with _prefix_cache(False):
        _run(run())


def test_list_vit_caches_matches_k3_format_and_dialable_port():
    """The collector parses K3's shape; http_port must be the port prefill dials."""

    async def run():
        h = _Holder()
        await h.store_task_embedding(
            "t1", torch.zeros((3, 8), dtype=torch.bfloat16), image_grid_thw=[[1, 2, 2]]
        )
        with mock.patch.dict(
            os.environ, {"MODEL_ID": "m_test", "POD_IP": "10.0.0.7"}
        ), mock.patch(
            "sglang.srt.disaggregation.encoder.server.get_disagg",
            return_value=types.SimpleNamespace(encoder_transfer_backend="mooncake"),
        ):
            out = await h.list_held_embeddings()
        assert out["model_id"] == "m_test"
        (entry,) = out["caches"]
        for key in (
            "ip", "http_port", "transfer_type", "task_id", "model_id",
            "embedding_size", "embedding_len", "embedding_dim", "embedding_dtype",
            "img_grid_thw", "original_image_sizes", "rdma_port", "item_addr",
            "cache_size", "header_size", "mooncake_session_id", "modality",
            "chunk_timestamps",
        ):
            assert key in entry, key
        assert entry["ip"] == "10.0.0.7"
        assert entry["http_port"] == 8005
        assert entry["embedding_size"] == 3 * 8 * 2
        assert entry["img_grid_thw"] == [[1, 2, 2]]

    _run(run())


def test_flush_keeps_holds_with_an_in_flight_pull():
    """/flush_cache drops idle holds but never one that is still owed a pull."""

    async def run():
        h = _Holder()
        # Pulled once and idle: reclaimable.
        await h.store_task_embedding("idle", torch.zeros((2, 2)))
        await h.acquire_task_embedding("idle")
        await h.release_task_embedding("idle")
        # A transfer is reading it.
        await h.store_task_embedding("busy", torch.zeros((2, 2)))
        await h.acquire_task_embedding("busy")
        # Produced and handed to vit-server, not pulled yet.
        await h.store_task_embedding("fresh", torch.zeros((2, 2)))
        out = await h.flush_task_embeddings()
        assert out["flushed_entries"] == 1 and out["remaining_entries"] == 2
        assert list(h.task_embeddings) == ["busy", "fresh"]

    _run(run())


class _FakeWorkflow:
    def __init__(self, alive=True):
        self.is_alive = alive
        self.stopped = False

    def graceful_stop(self):
        self.stopped = True


def _gate(path, workflow, stop_flag=False):
    fake_encoder = types.SimpleNamespace(encode_workflow=workflow)
    with mock.patch.object(http_server, "encoder", fake_encoder), mock.patch(
        "sglang.srt.entrypoints.http_server.stopped_by_external",
        return_value=stop_flag,
    ):
        return http_server._factory_health_gate(path)


def test_ready_with_stop_flag_drains_and_stays_200():
    """preStop flag: stop fetching but keep /ready 200 so pulls can finish."""
    wf = _FakeWorkflow()
    resp = _gate("/ready", wf, stop_flag=True)
    assert resp is not None and resp.status_code == 200
    assert wf.stopped
    # /health is not the drain endpoint and falls through to the probe.
    assert _gate("/health", _FakeWorkflow(), stop_flag=True) is None


def test_dead_poll_loop_or_broken_transfer_fails_health():
    """A dead vit-server loop, or every pull failing, must surface as 503."""
    assert _gate("/health", _FakeWorkflow(alive=False)).status_code == 503
    state = encode_health.TransferHealthState()
    with mock.patch.object(encode_health, "transfer_health_state", state):
        state.record_attempt("s1")
        state._sessions["s1"].last_success = 1.0  # failing for a long time
        assert _gate("/health", _FakeWorkflow()).status_code == 503
    assert _gate("/health", None) is None


def test_tp0_only_receive_follows_the_broadcast_group():
    """One rank per broadcast group pulls: attn-tp, or attn-cp under prefill CP.

    Under --enable-prefill-cp attn_tp_size is 1, so the forward broadcast goes
    over attn-cp; the receiver must split ranks by the same group, otherwise a
    metadata-only rank has nothing to broadcast and crashes on concat.
    """
    from sglang.srt.disaggregation.encoder import receiver
    from sglang.srt.managers import mm_schedule

    tp_group, cp_group = object(), object()

    def parallel(tp, cp, cp_rank=0):
        return types.SimpleNamespace(
            attn_tp_size=tp, attn_tp_rank=0, attn_tp_group=tp_group,
            attn_cp_size=cp, attn_cp_rank=cp_rank, attn_cp_group=cp_group,
        )

    rx = types.SimpleNamespace(tp_size=4, tp_group=object())
    req = types.SimpleNamespace(task_results=[{"task_id": "t"}])
    fn = receiver.MMReceiverBase._tp0_only_receive
    with mock.patch.dict(os.environ, {"SGLANG_ENCODER_TP0_ONLY_RECEIVE": "1"}):
        with mock.patch.object(mm_schedule, "get_parallel", return_value=parallel(1, 1)):
            assert fn(rx, req) is False
        with mock.patch.object(mm_schedule, "get_parallel", return_value=parallel(4, 1)):
            assert fn(rx, req) is True
            assert fn(rx, types.SimpleNamespace(task_results=None)) is False
            assert mm_schedule.tp0_only_broadcast_group() == (tp_group, 0)
        # Prefill CP: attn-tp is 1, so the CP group shares one pull.
        with mock.patch.object(
            mm_schedule, "get_parallel", return_value=parallel(1, 4, cp_rank=2)
        ):
            assert fn(rx, req) is True
            assert mm_schedule.tp0_only_broadcast_group() == (cp_group, 2)


def test_preprocessed_token_count_corrects_the_fetch_budget():
    """K3 parity: the real token count replaces vit-server's estimate early."""
    corrections = []
    enc = types.SimpleNamespace(
        rank=0,
        encode_workflow=types.SimpleNamespace(
            update_task_tokens=lambda tid, n: corrections.append((tid, n))
        ),
    )
    ctx = types.SimpleNamespace(
        items_per_req=[2, 1],
        preprocess_result=types.SimpleNamespace(token_counts=[10, 20, 7]),
    )
    MMEncoder._correct_workflow_task_tokens(
        enc, ctx, [{"task_id": "a"}, {"task_id": None}]
    )
    assert corrections == [("a", 30)]


def test_stage_costs_are_recorded_and_popped_once():
    enc = types.SimpleNamespace(rank=0, encode_workflow=object(), task_time_costs={})
    pop = MMEncoder.pop_task_time_costs.__get__(enc)
    assert pop("t") == {"process_time_cost": 0.1235, "embed_time_cost": 0.5}
    assert pop("t") == {}


def test_embed_digest_fingerprints_the_rows():
    from sglang.srt.disaggregation.encoder.server import compute_embed_digest

    d = compute_embed_digest(torch.arange(12, dtype=torch.float32).reshape(3, 4))
    assert d["num_elements"] == 12 and d["min"] == 0 and d["max"] == 11
    assert d["first8"] == [float(i) for i in range(8)]


def test_v41_reports_original_image_sizes():
    from sglang.srt.disaggregation.encoder.server import _build_mm_aux_data

    aux = _build_mm_aux_data({"original_image_sizes": [[640, 360]]}, "deepseek_v41")
    assert aux["original_image_sizes"] == [[640, 360]]


def test_hold_pool_registers_once_and_reuses_pages():
    """K3 ScatterMemPool parity: holds live in one pre-registered pinned pool."""
    from sglang.srt.disaggregation.gpu_buffer_pool import GpuBufferPool

    registered = []
    pool = GpuBufferPool(4 * 4096, 4096, device="cpu", pin_memory=False)

    async def run():
        h = _Holder()
        for name in ("_alloc_from_hold_pool_locked",):
            setattr(h, name, getattr(MMEncoder, name).__get__(h, _Holder))
        h.engine = types.SimpleNamespace(
            register=lambda *a: registered.append(a), deregister=lambda *a: None
        )
        h.task_hold_pool = pool
        emb = torch.arange(3 * 1024, dtype=torch.bfloat16).reshape(1024, 3)  # 6 KB
        await h.store_task_embedding("t1", emb)
        held = h.task_embeddings["t1"]
        assert held.pool_addr is not None and held.mr_ptr is None
        assert torch.equal(held.embedding, emb)
        assert pool.data_ptr <= held.embedding.data_ptr() < pool.data_ptr + pool.total_bytes
        assert registered == []  # no per-hold MR register
        # Pool holds 4 pages; each 6 KB hold takes 2, so a third store must
        # evict the pulled, unreferenced t1 and reuse its pages.
        await h.store_task_embedding("t2", emb)
        await h.acquire_task_embedding("t1")
        await h.release_task_embedding("t1")
        await h.store_task_embedding("t3", emb)
        assert "t1" not in h.task_embeddings and "t3" in h.task_embeddings
        assert h.task_embeddings["t3"].pool_addr is not None
        # Nothing left to evict (t2/t3 never pulled): the pool is the hard cap,
        # so a further store fails retryably instead of growing host memory.
        from sglang.srt.disaggregation.encoder.server import MMError

        try:
            await h.store_task_embedding("t4", emb)
            assert False, "expected pool-full MMError"
        except MMError as e:
            assert int(e.code) == 503
        assert "t4" not in h.task_embeddings

    _run(run())


def _pool_holder(pages=4):
    from sglang.srt.disaggregation.gpu_buffer_pool import GpuBufferPool

    h = _Holder()
    h._alloc_from_hold_pool_locked = MMEncoder._alloc_from_hold_pool_locked.__get__(
        h, _Holder
    )
    h.engine = types.SimpleNamespace(register=lambda *a: None, deregister=lambda *a: None)
    h.task_hold_pool = GpuBufferPool(pages * 4096, 4096, device="cpu", pin_memory=False)
    return h


def _six_kb():
    return torch.arange(3 * 1024, dtype=torch.bfloat16).reshape(1024, 3)


def test_a_re_served_hold_is_not_evicted_before_its_new_pull():
    """A second TaskResult for a pulled hold must survive pool pressure."""
    from sglang.srt.managers.schedule_batch import Modality

    async def run():
        from sglang.srt.disaggregation.encoder.server import MMError

        h = _pool_holder()
        await h.store_task_embedding("t1", _six_kb())
        await h.acquire_task_embedding("t1")
        await h.release_task_embedding("t1")
        # Request B is served t1 from the hold: its prefill has not pulled yet.
        await h.encode_and_store_for_workflow(["u"], "rB", "t1", Modality.IMAGE)
        await h.store_task_embedding("t2", _six_kb())
        # Pool full and t1 is reserved (not pulled): the store fails retryably
        # rather than evicting t1, so B's pending pull cannot 404.
        try:
            await h.store_task_embedding("t3", _six_kb())
            assert False, "expected pool-full MMError"
        except MMError as e:
            assert int(e.code) == 503
        assert "t1" in h.task_embeddings

    with _prefix_cache(True):
        _run(run())


def test_one_pull_does_not_release_a_hold_another_request_awaits():
    """Two requests served the same hold: the first pull must not free it."""
    from sglang.srt.disaggregation.encoder.server import MMError
    from sglang.srt.managers.schedule_batch import Modality

    async def run():
        h = _pool_holder()
        await h.store_task_embedding("t1", _six_kb())
        # The task that produced t1 pulls it (consumes its outstanding result).
        await h.acquire_task_embedding("t1")
        await h.release_task_embedding("t1")
        # A and B are both handed a TaskResult for t1.
        await h.encode_and_store_for_workflow(["u"], "rA", "t1", Modality.IMAGE)
        await h.encode_and_store_for_workflow(["u"], "rB", "t1", Modality.IMAGE)
        await h.acquire_task_embedding("t1")  # A's prefill pulls
        await h.release_task_embedding("t1")
        # B has not pulled yet, so t1 must not be an eviction victim.
        await h.store_task_embedding("t2", _six_kb())
        try:
            await h.store_task_embedding("t3", _six_kb())
            assert False, "expected pool-full MMError"
        except MMError as e:
            assert int(e.code) == 503
        assert "t1" in h.task_embeddings
        # Once B pulls too, t1 becomes evictable again.
        await h.acquire_task_embedding("t1")
        await h.release_task_embedding("t1")
        await h.store_task_embedding("t3", _six_kb())
        assert "t3" in h.task_embeddings

    with _prefix_cache(True):
        _run(run())


def test_a_hold_larger_than_the_pool_is_registered_per_hold():
    async def run():
        h = _pool_holder()  # 4 x 4 KB pages
        big = torch.zeros((1024, 10), dtype=torch.bfloat16)  # 20 KB > 16 KB pool
        await h.store_task_embedding("big", big)
        held = h.task_embeddings["big"]
        assert held.pool_addr is None
        assert held.mr_ptr is not None

    _run(run())


def test_a_cancelled_store_gives_its_pool_pages_back():
    """Cancelled (task timeout) between pool allocation and insert: no leak."""

    async def run():
        h = _pool_holder()

        def cancelled_at_insert():
            raise asyncio.CancelledError()

        h._evict_stale_task_embeddings_locked = cancelled_at_insert
        with pytest.raises(asyncio.CancelledError):
            await h.store_task_embedding("t1", _six_kb())
        assert "t1" not in h.task_embeddings
        assert h.task_hold_pool.used_bytes() == 0

    _run(run())


def test_transfer_rejects_a_byte_size_other_than_the_hold():
    """The pool is one MR: an oversized read would return a neighbour's rows."""
    from sglang.srt.disaggregation.encoder.server import BadRequestError

    async def run():
        h = _pool_holder()
        h.metrics_collector = None
        h.transfer_task_embedding = MMEncoder.transfer_task_embedding.__get__(
            h, _Holder
        )
        h._run_mooncake_transfer = mock.AsyncMock(return_value=0)
        await h.store_task_embedding("t1", _six_kb())
        nbytes = h.task_embeddings["t1"].size_bytes
        with pytest.raises(BadRequestError, match="byte_size"):
            await h.transfer_task_embedding("t1", "s", 1, nbytes + 4096)
        h._run_mooncake_transfer.assert_not_called()
        await h.transfer_task_embedding("t1", "s", 1, 0)  # 0 = whole hold
        await h.transfer_task_embedding("t1", "s", 1, nbytes)
        assert h._run_mooncake_transfer.await_count == 2
        assert h.task_embeddings["t1"].refs == 0

    _run(run())


def test_a_hold_whose_mr_cannot_register_is_not_advertised():
    from sglang.srt.disaggregation.encoder.server import InternalError

    async def run():
        h = _Holder()

        def fail(*args):
            raise RuntimeError("register failed")

        h.engine = types.SimpleNamespace(register=fail, deregister=lambda *a: None)
        with pytest.raises(InternalError, match="MR register failed"):
            await h.store_task_embedding("t1", _six_kb())
        assert "t1" not in h.task_embeddings

    _run(run())


def test_a_full_pool_reaps_expired_holds_before_refusing():
    """Never-pulled holds past their TTL must not keep the pool full."""

    async def run():
        h = _pool_holder()
        await h.store_task_embedding("t1", _six_kb())
        await h.store_task_embedding("t2", _six_kb())  # pool full, both unpulled
        for entry in h.task_embeddings.values():
            entry.created -= 10_000  # past any TTL
        await h.store_task_embedding("t3", _six_kb())
        assert "t3" in h.task_embeddings
        assert h.task_embeddings["t3"].pool_addr is not None

    _run(run())


def test_hold_pool_budget_is_checked_before_allocating():
    """A pool that cannot fit the container limit fails with the env name."""
    from unittest import mock

    from sglang.srt.disaggregation.encoder import server as server_module

    h = _Holder()
    h.task_hold_pool = None
    h.engine = types.SimpleNamespace(register=lambda *a: None, deregister=lambda *a: None)
    h.init_task_hold_pool = MMEncoder.init_task_hold_pool.__get__(h, _Holder)
    h.task_embeddings_max_bytes = 96 * 1024**3

    with (
        mock.patch.object(server_module, "_cgroup_memory_limit_bytes", return_value=64 * 1024**3),
        mock.patch.dict(os.environ, {"SGLANG_VLM_CACHE_SIZE_MB": "8192"}),
    ):
        with pytest.raises(RuntimeError, match="SGLANG_ENCODER_TASK_EMB_CACHE_GB"):
            h.init_task_hold_pool()
    assert h.task_hold_pool is None


def test_hold_pool_allocation_failure_is_reported_and_leaves_no_pool():
    from unittest import mock

    from sglang.srt.disaggregation.encoder import server as server_module

    h = _Holder()
    h.task_hold_pool = None
    h.engine = types.SimpleNamespace(register=lambda *a: None, deregister=lambda *a: None)
    h.init_task_hold_pool = MMEncoder.init_task_hold_pool.__get__(h, _Holder)
    h.task_embeddings_max_bytes = 1024**3

    def no_memory(*args, **kwargs):
        raise MemoryError("cannot allocate pinned memory")

    import sglang.srt.disaggregation.gpu_buffer_pool as pool_module

    with (
        mock.patch.object(server_module, "_cgroup_memory_limit_bytes", return_value=0),
        mock.patch.object(pool_module, "GpuBufferPool", no_memory),
    ):
        with pytest.raises(RuntimeError, match="Failed to allocate/register"):
            h.init_task_hold_pool()
    assert h.task_hold_pool is None


def test_flush_keeps_a_hold_whose_result_is_still_outstanding():
    from sglang.srt.managers.schedule_batch import Modality

    async def run():
        h = _pool_holder()
        await h.store_task_embedding("t1", _six_kb())  # pending_pulls = 1
        await h.acquire_task_embedding("t1")
        await h.release_task_embedding("t1")
        await h.encode_and_store_for_workflow(["u"], "rB", "t1", Modality.IMAGE)
        await h.flush_task_embeddings()
        assert "t1" in h.task_embeddings  # B still has to pull it
        await h.acquire_task_embedding("t1")
        await h.release_task_embedding("t1")
        await h.flush_task_embeddings()
        assert "t1" not in h.task_embeddings

    with _prefix_cache(True):
        _run(run())


def test_per_hold_fallback_does_not_keep_the_fused_batch_alive():
    """A row slice stored outside the pool must own its storage."""
    import torch

    async def run():
        h = _Holder()
        batch = torch.zeros((8, 3), dtype=torch.bfloat16)
        await h.store_task_embedding("t1", batch[2:4])
        held = h.task_embeddings["t1"].embedding
        assert held.untyped_storage().nbytes() == held.nbytes

    _run(run())
