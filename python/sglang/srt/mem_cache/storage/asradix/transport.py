"""Shared AttentionStore SDK transport adapters."""

from __future__ import annotations

import logging
import threading
import time
from numbers import Integral
from typing import Any, Callable, Mapping, Optional, Sequence

from .types import AttentionStoreInitContext

logger = logging.getLogger(__name__)


def _once(release: Callable[[], None]) -> Callable[[], None]:
    """Make a resource-release callback idempotent and thread-safe."""
    lock = threading.Lock()
    released = False

    def release_once() -> None:
        nonlocal released
        with lock:
            if released:
                return
            released = True
        release()

    return release_once


def _sdk_pod_config(
    ctx: AttentionStoreInitContext,
    namespace: str,
    *,
    endpoint_name: Optional[str] = None,
    model_version: Optional[str] = None,
    gpu_count: Optional[int] = None,
    pod_name: Optional[str] = None,
):
    from attentionstore_sdk.sdk import PodConfig
    from attentionstore_sdk.types import (
        PipelineCacheEndpointRole,
        PipelineCacheGroupDescriptor,
        PipelineCacheMember,
    )

    endpoint_registration = (
        None if endpoint_name is None else ctx.endpoint_registrations.get(endpoint_name)
    )
    pipeline_cache_group = None
    desired_block_count = 0
    pipeline_cache_member_id = 0
    if endpoint_registration is not None:
        desired_block_count = endpoint_registration.desired_block_count
        pipeline_cache_member_id = endpoint_registration.local_member_id
        pipeline_cache_group = PipelineCacheGroupDescriptor(
            group_id=endpoint_registration.group_id,
            generation=endpoint_registration.generation,
            endpoint_name=endpoint_registration.endpoint_name,
            endpoint_role=PipelineCacheEndpointRole[
                endpoint_registration.endpoint_role.name
            ],
            coordinator_member_id=endpoint_registration.coordinator_member_id,
            members=tuple(
                PipelineCacheMember(
                    member_id=member.member_id,
                    pp_stage_start=member.pp_stage_start,
                    pp_stage_end=member.pp_stage_end,
                    agent_address=member.agent_address,
                    bytes_per_block=member.bytes_per_block,
                    desired_block_count=member.desired_block_count,
                )
                for member in endpoint_registration.members
            ),
        )

    return PodConfig(
        namespace=namespace,
        pod_name=pod_name or ctx.pod_name,
        model_version=model_version or ctx.model_version,
        device_id=ctx.device_id,
        dp_id=ctx.dp_rank,
        gpu_count=ctx.gpu_count if gpu_count is None else gpu_count,
        desired_block_count=desired_block_count,
        pipeline_cache_group=pipeline_cache_group,
        pipeline_cache_member_id=pipeline_cache_member_id,
    )


def _standard_namespace(
    ctx: AttentionStoreInitContext,
    endpoint_name: str,
    shard_num: int,
) -> str:
    role = f"pp{ctx.node_pp_start}-{ctx.node_pp_end - 1}-of{ctx.pp_size}"
    return f"{ctx.namespace_base}:{shard_num}:{role}:{endpoint_name}"


class _PassiveStore:
    """Neutral endpoint used by PP stages with no local layers of one kind."""

    def __init__(self, layout=None):
        self.layout = layout

    @staticmethod
    def _count(block_hash_ids) -> int:
        return 1 if isinstance(block_hash_ids, Integral) else len(block_hash_ids)

    def match(self, block_hash_ids: Sequence[int], timeout: float = 1.0) -> int:
        """Treat every requested block as matched."""
        return self._count(block_hash_ids)

    def write(self, block_hash_ids: Sequence[int], *args, **kwargs) -> int:
        """Accept every passive write without storing data."""
        return self._count(block_hash_ids)

    def read(self, block_hash_ids: Sequence[int], *args, **kwargs) -> int:
        """Accept every passive read without loading data."""
        return self._count(block_hash_ids)

    def stop(self) -> None:
        """Stop the passive endpoint without performing cleanup."""
        return


class _BufferPageIO:
    """Transport page requests over SDK-owned typed buffer bindings."""

    def __init__(
        self,
        sdk,
        buffers,
        layer_ids: Sequence[int],
        *,
        buffer_owners: Sequence[Any] = (),
    ):
        self._sdk = sdk
        self._buffers = buffers
        self._layer_ids = tuple(map(int, layer_ids))
        self._buffer_owners = tuple(buffer_owners)

    def match(self, block_hash_ids: Sequence[int], timeout: float = 1.0) -> int:
        """Return the matched page prefix length."""
        from attentionstore_sdk.sdk import MatchRequest

        return self._sdk.match(MatchRequest(block_hash_ids, timeout=timeout))

    def write(
        self,
        block_hash_ids: Sequence[int],
        page_ids: Sequence[int],
        timeout: float = 1.0,
        params: Optional[dict] = None,
    ) -> int:
        """Write page blocks through the typed SDK buffers."""
        from attentionstore_sdk.sdk import BlockBatch, WriteRequest

        return self._sdk.write(
            WriteRequest(
                blocks=BlockBatch(
                    hash_ids=block_hash_ids,
                    data_ids=page_ids,
                    layer_ids=self._layer_ids,
                ),
                buffers=self._buffers,
                timeout=timeout,
                params=params or {},
                h2h_copy=False,
            )
        )

    def read(
        self,
        block_hash_ids: Sequence[int],
        page_ids: Sequence[int],
        timeout: float = 1.0,
        remote_addrs=None,
        logical_start: Optional[int] = None,
    ) -> int:
        """Read page blocks through the typed SDK buffers."""
        from attentionstore_sdk.sdk import BlockBatch, ReadRequest

        block_kwargs = (
            {"logical_start": logical_start} if logical_start is not None else {}
        )
        return self._sdk.read(
            ReadRequest(
                blocks=BlockBatch(
                    hash_ids=block_hash_ids,
                    data_ids=page_ids,
                    layer_ids=self._layer_ids,
                    **block_kwargs,
                ),
                buffers=self._buffers,
                timeout=timeout,
                remote_addrs=remote_addrs or (),
            )
        )

    def stop(self) -> None:
        """Stop the underlying page SDK client."""
        self._sdk.stop()


class CheckpointedPageStore:
    """Page stream plus an independently stored endpoint checkpoint."""

    def __init__(
        self, page_store, checkpoint_store, *, page_match_is_joint: bool = False
    ):
        """Combine a page store with its terminal checkpoint store."""
        self.page_store = page_store
        self.checkpoint_store = checkpoint_store
        self.page_match_is_joint = page_match_is_joint
        self.layout = checkpoint_store.layout

    def match(self, block_hash_ids: Sequence[int], timeout: float = 1.0) -> int:
        """Return the longest page prefix with a terminal checkpoint."""
        page_match_started = time.perf_counter()
        page_count = self.page_store.match(block_hash_ids, timeout=timeout)
        page_match_ms = (time.perf_counter() - page_match_started) * 1000

        # A group-bound page endpoint asks the Agent group for the common terminal
        # checkpoint and pins both endpoint payloads. This also applies to a
        # singleton non-PP group. Re-querying the checkpoint endpoint would create
        # a second transaction for the same logical Match and double lease pressure.
        if self.page_match_is_joint:
            return page_count

        checkpoint_match_started = time.perf_counter()
        checkpoint_probes = 0
        matched = 0
        for count in range(page_count, 0, -1):
            checkpoint_probes += 1
            if (
                self.checkpoint_store.match(
                    [block_hash_ids[count - 1]], timeout=timeout
                )
                == 1
            ):
                matched = count
                break
        checkpoint_match_ms = (time.perf_counter() - checkpoint_match_started) * 1000
        logger.info(
            "AS checkpoint match: requested_blocks=%s page_matched_blocks=%s "
            "matched_blocks=%s checkpoint_probes=%s page_match_ms=%.2f "
            "checkpoint_match_ms=%.2f",
            len(block_hash_ids),
            page_count,
            matched,
            checkpoint_probes,
            page_match_ms,
            checkpoint_match_ms,
        )
        return matched

    def read(
        self,
        block_hash_ids: Sequence[int],
        page_ids: Sequence[int],
        checkpoint_slot: Mapping[str, int] | int,
        timeout: float = 1.0,
        remote_addrs=None,
    ) -> int:
        """Read pages and their terminal checkpoint as one unit."""
        if not block_hash_ids:
            return 0
        page_count = self.page_store.read(
            block_hash_ids,
            page_ids,
            timeout=timeout,
            remote_addrs=remote_addrs,
        )
        if page_count != len(block_hash_ids):
            return page_count
        state_count = self.checkpoint_store.read(
            int(block_hash_ids[-1]),
            checkpoint_slot,
            timeout=timeout,
            remote_addrs=remote_addrs,
        )
        return page_count if state_count == 1 else 0

    def write(
        self,
        block_hash_ids: Sequence[int],
        page_ids: Sequence[int],
        checkpoint_slot: Mapping[str, int] | int,
        timeout: float = 1.0,
        params: Optional[dict] = None,
        producer_ready=None,
    ) -> int:
        """Write pages and their terminal checkpoint as one unit."""
        if not block_hash_ids:
            return 0
        wait_for_producer = getattr(self.checkpoint_store, "wait_for_producer", None)
        if wait_for_producer is not None:
            wait_for_producer(producer_ready, timeout=timeout)
        self.page_store.write(block_hash_ids, page_ids, timeout=timeout, params=params)
        checkpoint_result = self.checkpoint_store.write(
            int(block_hash_ids[-1]),
            checkpoint_slot,
            timeout=timeout,
            params=params,
            producer_ready=producer_ready,
        )
        return len(block_hash_ids) if checkpoint_result == 1 else -1

    def stop(self) -> None:
        """Stop both the page and checkpoint stores."""
        try:
            self.page_store.stop()
        finally:
            self.checkpoint_store.stop()
