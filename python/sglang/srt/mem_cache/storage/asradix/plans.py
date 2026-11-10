"""AttentionStore cache capability composition and selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from sglang.srt.mem_cache.unified_cache_components import ComponentType

from .checkpoint_binding import (
    create_kda_binding,
    create_spa_binding,
    describe_spa_endpoint_schema,
)
from .checkpoint_coordinator import KDACacheCoordinator, SPACacheCoordinator
from .coordinator import AttentionStoreCoordinator
from .page_binding import create_dsa_binding, create_mha_binding, create_mla_binding
from .page_coordinator import FullPageCacheCoordinator
from .types import AttentionStoreBinding, AttentionStoreInitContext


def _empty_endpoint_schema(_kv_cache: Any) -> Mapping[str, int]:
    return {}


@dataclass(frozen=True)
class AttentionStoreCachePlan:
    """Compose endpoint binding and cache lifecycle capabilities."""

    name: str
    coordinator_type: type[AttentionStoreCoordinator]
    binding_factory: Callable[[AttentionStoreInitContext], AttentionStoreBinding]
    tree_components: tuple[ComponentType, ...]
    schema_factory: Callable[[Any], Mapping[str, int]] = _empty_endpoint_schema

    def __post_init__(self) -> None:
        """Validate the cache plan contract."""
        if not self.name:
            raise ValueError("AttentionStore cache plan name must not be empty")
        if (
            not self.tree_components
            or self.tree_components[0] is not ComponentType.FULL
        ):
            raise ValueError("AttentionStore plans must start with the FULL component")


_ATTENTION_STORE_CACHE_PLANS = {
    "MHA": AttentionStoreCachePlan(
        name="MHA",
        coordinator_type=FullPageCacheCoordinator,
        binding_factory=create_mha_binding,
        tree_components=(ComponentType.FULL,),
    ),
    "MLA": AttentionStoreCachePlan(
        name="MLA",
        coordinator_type=FullPageCacheCoordinator,
        binding_factory=create_mla_binding,
        tree_components=(ComponentType.FULL,),
    ),
    "DSA": AttentionStoreCachePlan(
        name="DSA",
        coordinator_type=FullPageCacheCoordinator,
        binding_factory=create_dsa_binding,
        tree_components=(ComponentType.FULL,),
    ),
    "SPA": AttentionStoreCachePlan(
        name="SPA",
        coordinator_type=SPACacheCoordinator,
        binding_factory=create_spa_binding,
        tree_components=(ComponentType.FULL, ComponentType.SWA),
        schema_factory=describe_spa_endpoint_schema,
    ),
    "KDA": AttentionStoreCachePlan(
        name="KDA",
        coordinator_type=KDACacheCoordinator,
        binding_factory=create_kda_binding,
        tree_components=(ComponentType.FULL, ComponentType.MAMBA),
    ),
}


def resolve_attention_store_plan(model_config, kv_cache) -> AttentionStoreCachePlan:
    """Resolve a cache plan from cache capabilities, not model names."""
    from sglang.srt.configs.mamba_utils import LinearStateKind
    from sglang.srt.configs.model_config import AttentionArch, is_deepseek_dsa
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
        DeepSeekV4TokenToKVPool,
    )
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

    if isinstance(kv_cache, DeepSeekV4TokenToKVPool):
        plan_name = "SPA"
    elif (
        isinstance(kv_cache, HybridLinearKVPool)
        and getattr(kv_cache.mamba_pool, "linear_state_kind", None)
        is LinearStateKind.KDA
    ):
        plan_name = "KDA"
    elif is_deepseek_dsa(model_config.hf_config):
        plan_name = "DSA"
    elif model_config.attention_arch == AttentionArch.MLA:
        plan_name = "MLA"
    else:
        plan_name = "MHA"
    return _ATTENTION_STORE_CACHE_PLANS[plan_name]
