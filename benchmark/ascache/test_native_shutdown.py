"""Regression: native metadata direct submission must join without a Python executor."""

from types import SimpleNamespace
from unittest import mock

import pytest


@pytest.mark.parametrize("executor_present", [False, True])
@pytest.mark.parametrize("read_fails", [False, True])
def test_shutdown_joins_direct_native_read_before_dropping_owners(
    executor_present, read_fails
):
    """Shutdown joins direct native read before dropping owners.

    Args:
        executor_present: Whether a legacy Python reader executor was
            previously created.
        read_fails: Whether the native read completion raises during shutdown.

    Returns:
        None.

    Raises:
        AssertionError: The implementation violates the tested contract.
    """
    from sglang.srt.mem_cache.storage.asradix.checkpoint_coordinator import (
        SPAGCacheCoordinator,
    )

    coordinator = SPAGCacheCoordinator()
    order = []
    coordinator._drain = mock.Mock(
        side_effect=lambda *a, **k: order.append("write_drain")
    )
    coordinator._load_back_executor = (
        SimpleNamespace(shutdown=lambda **kwargs: order.append("metadata_join"))
        if executor_present
        else None
    )

    def read():
        """Complete the injected native read and record its cleanup order.

        Returns:
            None.

        Raises:
            RuntimeError: The fixture is configured to fail the read.
        """
        assert coordinator._pending_load_backs
        order.append("native_join")
        if read_fails:
            raise RuntimeError("cancelled fixture read")

    coordinator._pending_load_backs = [
        SimpleNamespace(native_read_future=SimpleNamespace(result=read))
    ]
    coordinator.shutdown(object())
    assert order == ["write_drain"] + (
        ["metadata_join"] if executor_present else []
    ) + ["native_join"]
    assert not coordinator._pending_load_backs
