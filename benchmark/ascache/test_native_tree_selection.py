"""Verify the opt-in native tree never replaces unsupported Python contracts."""

from types import SimpleNamespace
from unittest import mock

import pytest

from sglang.srt.mem_cache.unified_cache import tree_core_registry as registry
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType


def parameters(**overrides):
    """Create native-tree selection parameters with explicit test overrides.

    Args:
        overrides: Field overrides for the fixture or supported contract.

    Returns:
        SimpleNamespace: Default supported settings updated with overrides.
    """
    values = dict(
        enable_session_radix_cache=False,
        component_registry_override=None,
        eviction_policy_config=None,
        eviction_policy="lru",
        cache_ttl_seconds=None,
        pp_size=1,
        tree_components=(ComponentType.FULL, ComponentType.SWA),
        page_size=256,
        attn_cp_size=4,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "overrides",
    [
        {"enable_session_radix_cache": True},
        {"component_registry_override": {"custom": object()}},
        {"eviction_policy_config": {"custom": True}},
        {"eviction_policy": "lfu"},
        {"cache_ttl_seconds": 60},
        {"pp_size": 2},
        {"tree_components": ()},
        {"tree_components": (ComponentType.SWA,)},
        {"tree_components": (ComponentType.FULL, ComponentType.MAMBA)},
    ],
)
def test_unsupported_contract_keeps_python(overrides):
    """Unsupported contract keeps python.

    Args:
        overrides: Field overrides for the fixture or supported contract.

    Returns:
        None.

    Raises:
        AssertionError: The implementation violates the tested contract.
    """
    params = parameters(**overrides)
    with mock.patch.object(
        registry, "_python_tree_core_factory"
    ) as python, mock.patch.object(registry, "_rust_tree_core_factory") as rust:
        assert registry.create_tree_core("as_native", params, {}) is python.return_value
        python.assert_called_once_with(params, {})
        rust.assert_not_called()


def test_supported_contract_selects_native():
    """Supported contract selects native.

    Returns:
        None.

    Raises:
        AssertionError: The implementation violates the tested contract.
    """
    params = parameters()
    with mock.patch.object(
        registry, "_python_tree_core_factory"
    ) as python, mock.patch.object(registry, "_rust_tree_core_factory") as rust:
        assert registry.create_tree_core("as_native", params, {}) is rust.return_value
        rust.assert_called_once_with(params, {})
        python.assert_not_called()
