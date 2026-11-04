"""Native writer queue, error and ownership checks; import overlay is a plugin."""

import concurrent.futures
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from sglang.srt.mem_cache.storage.asradix.checkpoint_coordinator import (
    SPAGCacheCoordinator,
    _NativeWriterResult,
    _NativeWriteSlots,
    _SPAWriteTask,
)


def task():
    """Create a write task with observable staging and node release callbacks.

    Returns:
        _SPAWriteTask: The task fixture with four requested blocks.
    """
    return _SPAWriteTask(
        rid="writer",
        page_endpoint=SimpleNamespace(store=object()),
        checkpoint_store=None,
        block_hash_ids=[1, 2, 3, 4],
        page_ids=None,
        checkpoint_snapshot=None,
        params={},
        release_after_write=mock.Mock(),
        release=mock.Mock(),
        request=SimpleNamespace(as_spag_submitted_write=None),
    )


def cache():
    """Create a CPU cache fixture with logging and rank-vote stubs.

    Returns:
        SimpleNamespace: Cache settings and observable logging hooks.
    """
    return SimpleNamespace(
        pp_size=1,
        device="cpu",
        page_size=4,
        _log_as_failure=mock.Mock(),
        _log_as_result=mock.Mock(),
        _tp_min=lambda v: v,
    )


class Native(object):
    """Model native write completion and delayed ownership draining."""
    def __init__(self, value):
        self.value = value
        self._drained = False

    def done(self):
        """Report the fixture as ready independently of result draining.

        Returns:
            bool: Always True for this completed fixture.
        """
        return True

    def result(self, timeout=None):
        """Drain the fixture once unless its wait is injected to time out.

        Args:
            timeout: Optional boundary wait in seconds; None uses the native
                wait limit.

        Returns:
            int: The configured successful result.

        Raises:
            Exception: The configured error or timeout is raised unchanged.
        """
        if isinstance(self.value, TimeoutError):
            raise self.value
        self._drained = True
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


@pytest.mark.parametrize("value", [4, 2, RuntimeError("copy failed")])
def test_native_writer_retirement_releases_staging_once_then_retained_node(value):
    """Native writer retirement releases staging once then retained node.

    Args:
        value: Configured fixture value or injected failure.

    Returns:
        None.

    Raises:
        AssertionError: The implementation violates the tested contract.
    """
    coordinator = SPAGCacheCoordinator()
    t, c = task(), cache()
    native = Native(value)
    wrapper = _NativeWriterResult(c, t, native)
    coordinator._submit_native_write = mock.Mock(return_value=wrapper)
    coordinator._ensure_executor = mock.Mock(
        side_effect=AssertionError("Python writer started")
    )
    coordinator._reuse_finished_write = False
    coordinator.submit_write(c, t)
    assert len(coordinator._pending_writes) == 1
    assert coordinator._pending_writes[0].future is wrapper
    assert not native._drained
    coordinator._retire(c, coordinator._pending_writes, None, synchronize=False)
    t.release_after_write.assert_called_once()
    t.release.assert_called_once()
    assert wrapper.result() == (-1 if isinstance(value, Exception) else value)
    t.release_after_write.assert_called_once()
    coordinator._ensure_executor.assert_not_called()


def test_boundary_timeout_retains_slots_and_staging_until_native_completes():
    """Boundary timeout retains slots and staging until native completes.

    Returns:
        None.

    Raises:
        AssertionError: The implementation violates the tested contract.
    """
    t, c = task(), cache()
    native = Native(TimeoutError("native still running"))
    wrapper = _NativeWriterResult(c, t, native)
    with pytest.raises(TimeoutError):
        wrapper.result(0.001)
    assert not wrapper._finalized
    t.release_after_write.assert_not_called()
    native.value = 4
    assert wrapper.result() == 4
    t.release_after_write.assert_called_once()


def test_unsupported_task_keeps_existing_ordered_executor():
    """Unsupported task keeps existing ordered executor.

    Returns:
        None.

    Raises:
        AssertionError: The implementation violates the tested contract.
    """
    coordinator = SPAGCacheCoordinator()
    coordinator._submit_native_write = mock.Mock(return_value=None)
    coordinator._reuse_finished_write = False
    future = concurrent.futures.Future()
    executor = SimpleNamespace(submit=mock.Mock(return_value=future))
    coordinator._ensure_executor = mock.Mock(return_value=executor)
    t, c = task(), cache()
    coordinator.submit_write(c, t)
    assert coordinator._pending_writes[0].future is future
    executor.submit.assert_called_once_with(coordinator._run_write, c, t)


def test_native_submit_failure_discards_staging_and_node_receipt_once():
    """Native submit failure discards staging and node receipt once.

    Returns:
        None.

    Raises:
        AssertionError: The implementation violates the tested contract.
    """
    coordinator = SPAGCacheCoordinator()
    coordinator._submit_native_write = mock.Mock(side_effect=RuntimeError("queue full"))
    t, c = task(), cache()
    with pytest.raises(RuntimeError, match="queue full"):
        coordinator.submit_write(c, t)
    assert not coordinator._pending_writes
    t.release_after_write.assert_called_once()
    t.release.assert_called_once()


def test_direct_submission_preserves_chunks_producer_and_retry_policy(monkeypatch):
    """Direct submission preserves chunks producer and retry policy.

    Args:
        monkeypatch: Pytest fixture restoring temporary configuration and
            attribute changes.

    Returns:
        None.

    Raises:
        AssertionError: The implementation violates the tested contract.
    """
    monkeypatch.setenv("ATTENTIONSTORE_NATIVE_WRITE", "1")
    coordinator = SPAGCacheCoordinator()
    coordinator._reuse_finished_write = True
    coordinator._write_chunks = mock.Mock(return_value=[(0, 4, (7, 8), 2)])
    t, c = task(), cache()
    t.producer_ready = object()
    t.page_ids = _NativeWriteSlots(4, torch.arange(16), object())
    native = Native(4)
    submit = mock.Mock(return_value=native)
    t.page_endpoint = SimpleNamespace(
        store=SimpleNamespace(submit_write_slots_native=submit)
    )
    wrapper = coordinator._submit_native_write(c, t)
    assert wrapper.native is native
    submit.assert_called_once_with(
        t.block_hash_ids,
        t.page_ids,
        [(0, 4, (7, 8), 2)],
        t.producer_ready,
        attempts=2,
        params={},
        timeout=1.0,
    )
    t.release_after_write.assert_not_called()


def test_native_preparation_reaps_previously_cancelled_copy_owners(monkeypatch):
    """Native preparation reaps previously cancelled copy owners.

    Args:
        monkeypatch: Pytest fixture restoring temporary configuration and
            attribute changes.

    Returns:
        None.

    Raises:
        AssertionError: The implementation violates the tested contract.
    """
    monkeypatch.setenv("ATTENTIONSTORE_NATIVE_WRITE", "1")
    from sglang.srt.mem_cache.storage.asradix import coordinator as module

    coordinator = SPAGCacheCoordinator()
    coordinator._native_physical_writer = lambda cache: True
    slots = torch.arange(16, dtype=torch.int64)
    event = SimpleNamespace(record=mock.Mock())
    with mock.patch.object(
        torch.Tensor, "is_cuda", new_callable=mock.PropertyMock, return_value=True
    ), mock.patch.object(torch.cuda, "Event", return_value=event), mock.patch.object(
        torch.cuda, "current_stream", return_value=object()
    ), mock.patch.object(
        module, "_reap_quarantined_copies"
    ) as reap:
        prepared = coordinator.prepare_page_ids(cache(), slots)
    reap.assert_called_once()
    assert prepared.slots is slots


@pytest.mark.parametrize("mode", ["unified", "ring", "custom_pool"])
def test_native_writer_declines_unreviewed_physical_address_spaces(mode):
    """Native writer declines unreviewed physical address spaces.

    Args:
        mode: Supported or unsupported contract variant selected by
            parametrization.

    Returns:
        None.

    Raises:
        AssertionError: The implementation violates the tested contract.
    """
    from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
    from sglang.srt.mem_cache.allocator.unified_hybrid_swa import (
        UnifiedSWATokenToKVPoolAllocator,
    )
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

    allocator = object.__new__(
        UnifiedSWATokenToKVPoolAllocator
        if mode == "unified"
        else SWATokenToKVPoolAllocator
    )
    allocator._swa_req_ring = mode == "ring"
    allocator._kvcache = (
        object() if mode == "custom_pool" else object.__new__(DeepSeekV4TokenToKVPool)
    )
    assert not SPAGCacheCoordinator._native_physical_writer(
        SimpleNamespace(token_to_kv_pool_allocator=allocator)
    )
