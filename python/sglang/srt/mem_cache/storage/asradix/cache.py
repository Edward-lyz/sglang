"""AttentionStore-backed unified radix cache construction."""

from __future__ import annotations

import atexit
import logging
import time
from typing import Any, Optional, Tuple

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    InitLoadBackParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

from .coordinator import AttentionStoreCoordinator
from .plans import AttentionStoreCachePlan
from .types import AttentionStoreBinding, AttentionStoreInitContext

logger = logging.getLogger(__name__)


class AttentionStoreIntegrationMixin:
    """Connect UnifiedRadixCache to its selected AttentionStore plan."""

    _as_coordinator: Optional[AttentionStoreCoordinator] = None
    _as_plan: Optional[AttentionStoreCachePlan] = None

    _AS_RESULT_FIELDS = {
        "write": "written_blocks",
        "match": "matched_blocks",
        "load_back": "loaded_blocks",
    }

    def __init__(
        self,
        params: CacheInitParams,
        server_args,
        plan: Optional[AttentionStoreCachePlan] = None,
    ) -> None:
        """Initialize the local cache and its AttentionStore plan."""
        if plan is None:
            raise ValueError(f"{type(self).__name__} requires a cache plan")
        self.configure_attention_store(plan)
        self.binding: Optional[AttentionStoreBinding] = None
        self.ascli = None
        self.disable_finished_insert = params.disable_finished_insert
        self._disagg_mode = server_args.disaggregation_mode
        self.tp_group = params.tp_cache_group
        self.tp_world_size = (
            torch.distributed.get_world_size(group=self.tp_group)
            if self.tp_group is not None
            else 1
        )
        super().__init__(params)
        self.pp_world_size = (
            torch.distributed.get_world_size(group=self.pp_group)
            if self.pp_group is not None
            else 1
        )
        attention_group_sizes = [self.tp_world_size]
        for group in (self.attn_cp_group, self.attn_tp_group):
            if group is not None:
                attention_group_sizes.append(
                    torch.distributed.get_world_size(group=group)
                )
        self.as_world_size = max(attention_group_sizes)
        atexit.register(self._shutdown_attention_store)

    def _log_as_result(
        self,
        operation: str,
        *,
        rid: str,
        requested: int,
        result: int,
        started: float,
    ) -> None:
        result_field = self._AS_RESULT_FIELDS[operation]
        logger.info(
            "AS %s: cache_plan=%s rid=%s requested_blocks=%s "
            "%s=%s status=ok cost_ms=%.2f",
            operation,
            self.attention_store_plan.name,
            rid,
            requested,
            result_field,
            result,
            (time.perf_counter() - started) * 1000,
        )

    def _log_as_failure(
        self,
        operation: str,
        *,
        rid: str,
        requested: int,
        error: Exception,
        started: Optional[float] = None,
    ) -> None:
        result_field = self._AS_RESULT_FIELDS[operation]
        cost_ms = (
            (time.perf_counter() - started) * 1000 if started is not None else -1.0
        )
        logger.warning(
            "AS %s: cache_plan=%s rid=%s requested_blocks=%s "
            "%s=-1 status=error cost_ms=%.2f error=%s",
            operation,
            self.attention_store_plan.name,
            rid,
            requested,
            result_field,
            cost_ms,
            error,
        )

    def _as_min(self, value: int) -> int:
        """Reduce a runtime cache decision within the current PP stage."""
        tensor = torch.tensor(value, dtype=torch.int64)
        self._all_reduce_attn_groups(tensor, torch.distributed.ReduceOp.MIN)
        return int(tensor.item())

    def _as_topology_min(self, value: int) -> int:
        """Reduce an initialization decision across attention shards and PP."""
        tensor = torch.tensor(self._as_min(value), dtype=torch.int64)
        if self.pp_world_size > 1:
            torch.distributed.all_reduce(
                tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=self.pp_group,
            )
        return int(tensor.item())

    def _tp_min(self, value: int) -> int:
        """Reduce a write-retirement value across local attention shards."""
        return self._as_min(value)

    def _take_attention_store_load_context(
        self, params: InitLoadBackParams
    ) -> tuple[Any, Any, Any]:
        anchor = params.best_match_node
        req = params.req
        load_info = getattr(req, "_as_load_info", None) if req is not None else None
        if req is not None and hasattr(req, "_as_load_info"):
            del req._as_load_info
        return anchor, load_info, req

    def _match_checkpointed_page_prefix(
        self,
        params: MatchPrefixParams,
        local_result: MatchResult,
    ) -> Optional[MatchResult]:
        """Match page prefixes whose terminal hash has a separate checkpoint."""
        req = params.req
        if (
            req is None
            or req.ever_read_as
            or not req.as_block_hashs
            or self.ascli is None
        ):
            return None

        local_hit_len = len(local_result.device_indices)
        max_hit_len = len(params.key) // self.page_size * self.page_size
        if local_hit_len == max_hit_len:
            return None

        start_block = local_hit_len // self.page_size
        hashes = req.as_block_hashs.hash_ids[start_block:]
        # A checkpoint match is valid only for the hash at its returned endpoint.
        # Apply the request's prefix limit before matching so a later checkpoint
        # cannot be matched and then relabeled as an earlier prefix by truncation.
        max_blocks = max((max_hit_len - local_hit_len) // self.page_size, 0)
        hashes = hashes[:max_blocks]
        if not hashes:
            return None

        requested = len(hashes)
        started = time.perf_counter()
        match_error = None
        try:
            match_count = self.ascli.match(hashes, timeout=1.0)
        except Exception as exc:
            match_error = exc
            match_count = 0

        match_count = min(self._as_min(match_count), requested)
        if match_error is None:
            self._log_as_result(
                "match",
                rid=req.rid,
                requested=requested,
                result=match_count,
                started=started,
            )
        else:
            self._log_as_failure(
                "match",
                rid=req.rid,
                requested=requested,
                error=match_error,
                started=started,
            )
        if match_count <= 0:
            return None

        anchor = local_result.last_device_node
        req._as_load_info = {
            "rid": req.rid,
            "start_block": start_block,
            "block_count": match_count,
            "hashes": hashes[:match_count],
            "remote_addrs": (req.as_max_hit_host_ip or [] if self.pp_size <= 1 else []),
        }
        return MatchResult(
            device_indices=local_result.device_indices,
            last_device_node=anchor,
            last_host_node=anchor,
            best_match_node=anchor,
            host_hit_length=match_count * self.page_size,
        )

    def configure_attention_store(self, plan: AttentionStoreCachePlan) -> None:
        """Attach an AttentionStore plan and its payload coordinator."""
        if self._as_plan is not None:
            raise RuntimeError("AttentionStore cache plan is already configured")
        self._as_plan = plan
        self._as_coordinator = plan.coordinator_type()

    @property
    def attention_store_coordinator(self) -> AttentionStoreCoordinator:
        """Return the configured write coordinator."""
        if self._as_coordinator is None:
            raise RuntimeError("AttentionStore coordinator is not configured")
        return self._as_coordinator

    @property
    def attention_store_plan(self) -> AttentionStoreCachePlan:
        """Return the configured cache plan."""
        if self._as_plan is None:
            raise RuntimeError("AttentionStore cache plan is not configured")
        return self._as_plan

    def create_attention_store_binding(
        self, context: AttentionStoreInitContext
    ) -> AttentionStoreBinding:
        """Create the endpoint binding selected by the cache plan."""
        return self.attention_store_plan.binding_factory(context)

    def ref_as_binding(self, binding: AttentionStoreBinding) -> None:
        """Attach a validated runtime binding and its cache store."""
        if binding.plan_name != self.attention_store_plan.name:
            raise ValueError(
                "AttentionStore binding plan mismatch: "
                f"cache={self.attention_store_plan.name} "
                f"binding={binding.plan_name}"
            )
        self.binding = binding
        self.ascli = binding.cache_store

    def needs_req_in_match(self) -> bool:
        """Return whether prefix matching requires request context."""
        return True

    def _is_prefill_role(self) -> bool:
        return self._disagg_mode in ("null", "prefill")

    def ready_to_load_host_cache(self) -> int:
        """Report that AttentionStore loads need no host-cache readiness wait."""
        return 0

    def cache_finished_req(self, req, is_insert: bool = True, **kwargs) -> None:
        """Commit a finished request locally and schedule its remote write."""
        coordinator = self.attention_store_coordinator
        effective_is_insert = (
            is_insert
            and not self.disable
            and not getattr(self, "disable_finished_insert", False)
        )
        context = coordinator.capture_finished(self, req, effective_is_insert)
        try:
            if context is not None:
                coordinator.before_local_commit(self, context)
            super().cache_finished_req(req, effective_is_insert, **kwargs)
            if context is not None:
                coordinator.after_local_commit(self, context)
        except Exception:
            if context is not None:
                coordinator.abort_local_commit(self, context)
            raise

    def cache_unfinished_req(self, req, chunked=False) -> None:
        """Commit an unfinished request locally and schedule its remote write."""
        coordinator = self.attention_store_coordinator
        context = coordinator.capture_unfinished(self, req)
        try:
            if context is not None:
                coordinator.before_local_commit(self, context)
            super().cache_unfinished_req(req, chunked=chunked)
            if context is not None:
                coordinator.after_local_commit(self, context)
        except Exception:
            if context is not None:
                coordinator.abort_local_commit(self, context)
            raise

    def check_ascache_events(self) -> None:
        """Retire completed AttentionStore writes."""
        self.attention_store_coordinator.check_events(self)

    def drain_attention_store_writes(self) -> None:
        """Wait for all pending AttentionStore writes."""
        self.attention_store_coordinator.drain(self)

    def reset(self) -> None:
        """Drain remote writes before resetting the local cache."""
        self.attention_store_coordinator.before_reset(self)
        super().reset()

    def sanity_check(self) -> None:
        """Drain remote writes before checking local cache invariants."""
        self.attention_store_coordinator.before_sanity_check(self)
        super().sanity_check()

    def _shutdown_attention_store(self) -> None:
        self.attention_store_coordinator.shutdown(self)
        if self.ascli is not None:
            try:
                self.ascli.stop()
            except Exception as exc:
                logger.warning(
                    "Failed to stop AttentionStore: cache_plan=%s error=%s",
                    self.attention_store_plan.name,
                    exc,
                )
            self.ascli = None
            self.binding = None


class AttentionStoreUnifiedRadixCache(
    AttentionStoreIntegrationMixin, UnifiedRadixCache
):
    """Component-driven UnifiedRadixCache with an AttentionStore L2."""

    def __init__(
        self,
        params: CacheInitParams,
        server_args,
        plan: Optional[AttentionStoreCachePlan] = None,
    ) -> None:
        """Select the component set declared by the AttentionStore plan."""
        if plan is None:
            raise ValueError(f"{type(self).__name__} requires a cache plan")
        params.tree_components = plan.tree_components
        super().__init__(params=params, server_args=server_args, plan=plan)

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """Extend a local component-consistent match through its coordinator."""
        if params.req is not None and hasattr(params.req, "_as_load_info"):
            del params.req._as_load_info
        local_result = super().match_prefix(params)
        remote = self.attention_store_coordinator.extend_match(
            self,
            params,
            local_result,
        )
        return remote if remote is not None else local_result

    def init_load_back(self, params: InitLoadBackParams) -> Tuple[torch.Tensor, Any]:
        """Delegate type-specific payload restoration to the plan coordinator."""
        return self.attention_store_coordinator.init_load_back(self, params)


def create_attention_store_cache(
    plan: AttentionStoreCachePlan,
    *,
    params: CacheInitParams,
    server_args,
):
    """Create the component-driven UnifiedRadixCache for an AS plan."""
    return AttentionStoreUnifiedRadixCache(
        params=params,
        server_args=server_args,
        plan=plan,
    )
