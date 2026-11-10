"""Unsupported modes keep allocation semantics under the feature flag."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from sglang.srt.mem_cache.storage.asradix import checkpoint_coordinator as module


@pytest.mark.parametrize(
    "mode", ["flag_off", "native_ineligible", "PP", "custom", "peer_ineligible"]
)
def test_compatibility_path_keeps_per_request_allocation_vote(monkeypatch, mode):
    """Compatibility path keeps per request allocation vote.

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
    monkeypatch.setenv(
        "SGLANG_AS_DEFER_ALLOCATION_GATE", "0" if mode == "flag_off" else "1"
    )

    class Custom(module.SPAGCacheCoordinator):
        """Keep a custom coordinator on the compatibility allocation path."""
        pass

    coordinator = Custom() if mode == "custom" else module.SPAGCacheCoordinator()
    coordinator._native_physical_writer = lambda cache: mode != "native_ineligible"
    future = SimpleNamespace(done=lambda: False)
    coordinator._try_native_metadata = mock.Mock(return_value=future)
    votes = []

    def vote(value):
        votes.append(value)
        return 0 if len(votes) == 1 and mode == "peer_ineligible" else value

    allocator = SimpleNamespace(
        device="cpu",
        available_size=lambda: 100,
        alloc_extend_swa_tail=lambda *a: torch.arange(12, dtype=torch.int64),
    )
    cache = SimpleNamespace(
        page_size=4,
        token_to_kv_pool_allocator=allocator,
        full_evictable_size=lambda: 0,
        pp_world_size=2 if mode == "PP" else 1,
        pp_size=2 if mode == "PP" else 1,
        _as_min=vote,
        _as_read_diffuser=None,
    )
    with mock.patch.object(module, "_acquire_node_release", return_value=lambda: None):
        for index in range(2):
            req = SimpleNamespace(
                rid=f"compat-{index}", prefix_indices=torch.empty(0, dtype=torch.int64)
            )
            task = coordinator._submit_load_back(
                cache, req, dict(block_count=3, start_block=0, hashes=[1, 2, 3]), None
            )
            assert task is not None
    expected = (
        [1, 1]
        if mode == "flag_off"
        else ([1, 1, 1] if mode == "peer_ineligible" else [0, 1, 1])
    )
    assert votes == expected
    assert len(coordinator._pending_load_backs) == 2


def test_compatibility_vote_failure_frees_allocation_without_submission(monkeypatch):
    """Compatibility vote failure frees allocation without submission.

    Args:
        monkeypatch: Pytest fixture restoring temporary configuration and
            attribute changes.

    Returns:
        None.

    Raises:
        AssertionError: The implementation violates the tested contract.
    """
    monkeypatch.setenv("SGLANG_AS_DEFER_ALLOCATION_GATE", "0")
    coordinator = module.SPAGCacheCoordinator()
    coordinator._try_native_metadata = mock.Mock()
    coordinator._free_loaded_pages = mock.Mock()
    loaded = torch.arange(12, dtype=torch.int64)
    allocator = SimpleNamespace(
        device="cpu",
        available_size=lambda: 100,
        alloc_extend_swa_tail=lambda *a: loaded,
    )
    cache = SimpleNamespace(
        page_size=4,
        token_to_kv_pool_allocator=allocator,
        full_evictable_size=lambda: 0,
        _as_min=lambda value: 0,
    )
    req = SimpleNamespace(
        rid="peer-allocation-failure", prefix_indices=torch.empty(0, dtype=torch.int64)
    )
    task = coordinator._submit_load_back(
        cache, req, dict(block_count=3, start_block=0, hashes=[1, 2, 3]), None
    )
    assert task is None
    coordinator._free_loaded_pages.assert_called_once_with(cache, loaded, start_pos=0)
    coordinator._try_native_metadata.assert_not_called()
    assert not coordinator._pending_load_backs


def test_initial_eligibility_vote_is_cached_on_enabled_supported_path(monkeypatch):
    """Initial eligibility vote is cached on enabled supported path.

    Args:
        monkeypatch: Pytest fixture restoring temporary configuration and
            attribute changes.

    Returns:
        None.

    Raises:
        AssertionError: The implementation violates the tested contract.
    """
    monkeypatch.setenv("SGLANG_AS_DEFER_ALLOCATION_GATE", "1")
    coordinator = module.SPAGCacheCoordinator()
    coordinator._native_physical_writer = mock.Mock(return_value=True)
    cache = SimpleNamespace(pp_world_size=1, _as_min=mock.Mock(return_value=1))
    assert coordinator._use_deferred_allocation(cache)
    assert coordinator._use_deferred_allocation(cache)
    cache._as_min.assert_called_once_with(1)
    coordinator._native_physical_writer.assert_called_once_with(cache)
