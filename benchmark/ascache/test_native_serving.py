#!/usr/bin/env python3
"""Validate inherited serving and compatibility with deferred allocation gate."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from sglang.srt.mem_cache.storage.asradix.checkpoint_coordinator import (
    SPACacheCoordinator,
    SPAGCacheCoordinator,
    _SPLoadBackTask,
)


@pytest.mark.parametrize("peer_ready", [False, True])
@pytest.mark.parametrize("peer_success", [False, True])
@pytest.mark.parametrize("read", [3, 2, RuntimeError("metadata failed")])
def test_prepared_slots_respect_both_CP_gates_and_diffuse_order(
    peer_ready, peer_success, read
):
    """Prepared slots respect both cp gates and diffuse order.

    Args:
        peer_ready: Whether the other ranks acknowledge read readiness.
        peer_success: Whether the other ranks acknowledge a successful read.
        read: Native block count or injected read exception.

    Returns:
        None.

    Raises:
        AssertionError: The implementation violates the tested contract.
    """
    coordinator = SPAGCacheCoordinator()
    calls = []
    native = SimpleNamespace(
        _completion=dict(
            prepared_pages=[10, 20, 30, 40],
            local_indices=[0, 1, 3],
            prepared_swa_page=7,
        )
    )
    future = SimpleNamespace(
        native=native,
        done=lambda: True,
        result=mock.Mock(
            return_value=None if isinstance(read, Exception) else read,
            side_effect=read if isinstance(read, Exception) else None,
        ),
    )
    req = SimpleNamespace(_as_pending_load_back=None)
    task = _SPLoadBackTask(
        rid="cp",
        req=req,
        load_info={"hashes": [1, 2, 3, 4]},
        loaded=torch.arange(16),
        requested=4,
        local_hit_len=0,
        started=0,
        lock=lambda: calls.append("unlock"),
        future=future,
        native_read_future=future,
        native_metadata=True,
    )
    task.native_cp_sharded = True
    req._as_pending_load_back = task
    coordinator._pending_load_backs = [task]
    coordinator._free_loaded_pages = mock.Mock(
        side_effect=lambda *a, **k: calls.append("free")
    )
    coordinator._insert_loaded_prefix = mock.Mock(
        side_effect=lambda *a, **k: calls.append("publish")
    )
    votes = []

    def reduce(tensor, op):
        """Inject remote readiness or failure into the rank-agreement tensor.

        Args:
            tensor: Mutable CPU vote tensor representing rank agreement.
            op: Collective reduction operation selected by the coordinator.

        Returns:
            None.
        """
        votes.append(tensor.tolist())
        if (len(votes) == 1 and not peer_ready) or (
            len(votes) == 2 and not peer_success
        ):
            tensor.zero_()

    cache = SimpleNamespace(
        page_size=4,
        _all_reduce_attn_groups=reduce,
        _log_as_result=mock.Mock(),
        _log_as_failure=mock.Mock(),
        _as_read_diffuser=SimpleNamespace(
            diffuse=mock.Mock(side_effect=lambda *a, **k: calls.append("diffuse"))
        ),
        ascli=SimpleNamespace(
            emit_content_digest=mock.Mock(
                side_effect=lambda *a, **k: calls.append("digest")
            )
        ),
    )
    coordinator.commit_load_backs(cache)
    if not peer_ready:
        assert votes == [[1]] and not calls
        future.result.assert_not_called()
        assert coordinator._pending_load_backs == [task]
    else:
        assert len(votes) == 2
        assert votes[1] == ([4] if read == 3 else [0])
        assert not coordinator._pending_load_backs
        assert req._as_pending_load_back is None
        if peer_success and read == 3:
            assert calls == ["diffuse", "digest", "publish", "unlock"]
            cache._as_read_diffuser.diffuse.assert_called_once_with(
                (10, 20, 30, 40), 4, extra_local_blocks=(3,), peers_ready=True
            )
        else:
            assert calls == ["free", "unlock"]


def test_direct_native_submit_bypasses_metadata_executor(monkeypatch):
    """Direct native submit bypasses metadata executor.

    Args:
        monkeypatch: Pytest fixture restoring temporary configuration and
            attribute changes.

    Returns:
        None.

    Raises:
        AssertionError: The implementation violates the tested contract.
    """
    coordinator = SPACacheCoordinator()
    native = SimpleNamespace(done=lambda: False)
    coordinator._try_native_metadata = mock.Mock(return_value=native)
    coordinator._ensure_load_back_executor = mock.Mock(
        side_effect=AssertionError("Python reader started")
    )
    loaded = torch.arange(16, dtype=torch.int64)
    allocator = SimpleNamespace(
        device="cpu",
        available_size=lambda: 100,
        alloc_extend_swa_tail=lambda *a: loaded,
    )
    cache = SimpleNamespace(
        page_size=4,
        token_to_kv_pool_allocator=allocator,
        _as_min=lambda v: v,
        full_evictable_size=lambda: 0,
        _as_read_diffuser=object(),
    )
    request = SimpleNamespace(
        rid="submit", prefix_indices=torch.empty(0, dtype=torch.int64)
    )
    from sglang.srt.mem_cache.storage.asradix import checkpoint_coordinator as module

    with mock.patch.object(module, "_acquire_node_release", return_value=lambda: None):
        task = coordinator._submit_load_back(
            cache,
            request,
            dict(block_count=4, start_block=0, hashes=[1, 2, 3, 4]),
            object(),
        )
    assert task.native_read_future is task.future is native
    assert task.native_metadata and task.native_cp_sharded
    assert coordinator._pending_load_backs == [task]
    coordinator._ensure_load_back_executor.assert_not_called()


def test_unsupported_mode_uses_existing_reader_executor():
    """Unsupported mode uses existing reader executor.

    Returns:
        None.

    Raises:
        AssertionError: The implementation violates the tested contract.
    """
    coordinator = SPACacheCoordinator()
    coordinator._try_native_metadata = mock.Mock(return_value=None)
    marker = object()
    executor = SimpleNamespace(submit=mock.Mock(return_value=marker))
    coordinator._ensure_load_back_executor = mock.Mock(return_value=executor)
    allocator = SimpleNamespace(
        device="cpu",
        available_size=lambda: 100,
        alloc_extend_swa_tail=lambda *a: torch.arange(16, dtype=torch.int64),
    )
    cache = SimpleNamespace(
        page_size=4,
        token_to_kv_pool_allocator=allocator,
        _as_min=lambda v: v,
        full_evictable_size=lambda: 0,
    )
    request = SimpleNamespace(
        rid="fallback", prefix_indices=torch.empty(0, dtype=torch.int64)
    )
    from sglang.srt.mem_cache.storage.asradix import checkpoint_coordinator as module

    with mock.patch.object(module, "_acquire_node_release", return_value=lambda: None):
        task = coordinator._submit_load_back(
            cache,
            request,
            dict(block_count=4, start_block=0, hashes=[1, 2, 3, 4]),
            object(),
        )
    assert not task.native_metadata and task.future is marker
    executor.submit.assert_called_once_with(
        coordinator._run_load_back_read, cache, task
    )


@pytest.mark.parametrize(
    "mode", ["unified", "ring", "custom_pool", "custom_coordinator"]
)
def test_virtual_ring_custom_layout_preserves_fallback(monkeypatch, mode):
    """Virtual ring custom layout preserves fallback.

    Args:
        monkeypatch: Pytest fixture restoring temporary configuration and
            attribute changes.
        mode: Supported or unsupported contract variant selected by
            parametrization.

    Returns:
        None.

    Raises:
        AssertionError: The implementation violates the tested contract.
    """
    monkeypatch.setenv("ATTENTIONSTORE_NATIVE_METADATA", "1")
    from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
    from sglang.srt.mem_cache.allocator.unified_hybrid_swa import (
        UnifiedSWATokenToKVPoolAllocator,
    )
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

    class CustomCoordinator(SPAGCacheCoordinator):
        """Represent an unreviewed coordinator that must use the reader fallback."""
        pass

    coordinator = (
        CustomCoordinator() if mode == "custom_coordinator" else SPAGCacheCoordinator()
    )
    policy = SimpleNamespace(block_step_for_position=lambda p: 2)
    coordinator._swa_policy = lambda c: policy
    allocator = object.__new__(
        UnifiedSWATokenToKVPoolAllocator
        if mode == "unified"
        else SWATokenToKVPoolAllocator
    )
    allocator._swa_req_ring = mode == "ring"
    pool = (
        object() if mode == "custom_pool" else object.__new__(DeepSeekV4TokenToKVPool)
    )
    allocator._kvcache = pool
    cache = SimpleNamespace(
        token_to_kv_pool_allocator=allocator,
        ascli=SimpleNamespace(submit_read_slots_native=mock.Mock()),
    )
    assert coordinator._try_native_metadata(cache, object()) is None
    cache.ascli.submit_read_slots_native.assert_not_called()
