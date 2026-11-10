"""Registry for pluggable TreeCore implementations.

The unified cache constructs its TreeCore through `create_tree_core`, selected
by SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND (default "python"). To plug in a custom
implementation, register it under a string name via
`register_tree_core_backend(name, factory)`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.mem_cache.unified_cache.components import TreeComponent
    from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import (
        UnifiedTreeCoreInterface,
    )

TreeCoreFactory = Callable[
    ["CacheInitParams", "dict[ComponentType, TreeComponent]"],
    "UnifiedTreeCoreInterface",
]

_TREE_CORE_REGISTRY: dict[str, TreeCoreFactory] = {}


def register_tree_core_backend(name: str, factory: TreeCoreFactory) -> None:
    """Register a TreeCore factory under `name`."""
    if not name.strip():
        raise ValueError(
            f"register_tree_core_backend: name must be non-empty, got {name!r}"
        )
    if name in _TREE_CORE_REGISTRY:
        raise ValueError(f"register_tree_core_backend: {name!r} is already registered")
    _TREE_CORE_REGISTRY[name] = factory


def get_tree_core_factory(name: str) -> Optional[TreeCoreFactory]:
    return _TREE_CORE_REGISTRY.get(name)


def registered_tree_core_backends() -> list[str]:
    return list(_TREE_CORE_REGISTRY.keys())


def _python_tree_core_factory(
    params: CacheInitParams, components: dict[ComponentType, TreeComponent]
) -> UnifiedTreeCoreInterface:
    """The pure-Python TreeCore."""
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore

    return UnifiedTreeCore(params, components)


def _rust_tree_core_factory(
    params: CacheInitParams, components: dict[ComponentType, TreeComponent]
) -> UnifiedTreeCoreInterface:
    """Load and construct the in-tree Rust TreeCore only when selected."""
    from sglang.srt.mem_cache.rust_tree_core.adapter import RustUnifiedTreeCore

    return RustUnifiedTreeCore(params)


def _as_native_tree_core_factory(params, components):
    """Use the native CP-aware Full/SWA core inside its validated contract.

    Other components and optional Python-node APIs keep the original backend;
    this opt-in backend must not remove session/C128/Mamba/PP/custom features.
    """
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    if (
        params.enable_session_radix_cache
        or params.component_registry_override
        or params.eviction_policy_config
        or params.eviction_policy != "lru"
        or params.cache_ttl_seconds is not None
        or params.pp_size != 1
        or not params.tree_components
        or ComponentType.FULL not in params.tree_components
        or set(params.tree_components) - {ComponentType.FULL, ComponentType.SWA}
    ):
        return _python_tree_core_factory(params, components)
    import logging
    core = _rust_tree_core_factory(params, components)
    logging.getLogger(__name__).info(
        "Native tree selected: components=%s page_size=%d CP=%d",
        params.tree_components, params.page_size, params.attn_cp_size,
    )
    return core


register_tree_core_backend("python", _python_tree_core_factory)
register_tree_core_backend("rust", _rust_tree_core_factory)
register_tree_core_backend("as_native", _as_native_tree_core_factory)


def create_tree_core(
    name: str,
    params: CacheInitParams,
    components: dict[ComponentType, TreeComponent],
) -> UnifiedTreeCoreInterface:
    """Construct the TreeCore registered under `name`."""
    factory = get_tree_core_factory(name)
    if factory is None:
        raise ValueError(
            f"SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND={name!r} is not registered. "
            f"Registered backends: {registered_tree_core_backends()}. "
            "External backends must call register_tree_core_backend(...) at import time."
        )
    return factory(params, components)
