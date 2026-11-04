"""Shared AttentionStore SDK transport adapters."""

from __future__ import annotations

import logging
import threading
import time
from numbers import Integral
from typing import Any, Callable, Mapping, Optional, Sequence

from .types import AttentionStoreInitContext

logger = logging.getLogger(__name__)
# Cache-content diagnostic. Separate logger so it can be silenced or routed to
# its own file without touching the normal AS logs.
# Runtime override for the digest log, set through /set_internal_state. None
# means "no override", so the env default applies. Each PP rank keeps its own
# copy: the control request walks the whole PP chain, so every rank gets it.






# Runtime override for the producer-event barrier diagnostic. Same on/off
# split as the cache digest: it logs whether an event fenced anything and,
# on read, whether the page content matches what the SDK gathered.








def wait_for_producer_event(
    event,
    timeout: Optional[float] = 1.0,
    operation: str = "AS page write",
    debug_tag: str = "",
) -> None:
    """Fence the HBM pages a transfer reads against the stream that fills them.

    Writes are executed on the coordinator's writer thread, whose current stream
    is not the one the forward pass runs on, so neither the digest snapshot nor
    the SDK gather is ordered against the producer. Without this wait either can
    read a page that is still being written.
    """
    if event is None:
        return
    query = getattr(event, "query", None)
    if query is None:
        raise RuntimeError(f"{operation} event does not support bounded polling")
    already_done = query()
    deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
    polls = 0
    while not query():
        polls += 1
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"{operation} producer wait timed out")
        time.sleep(0.001)


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
        buffer_tensors: Optional[Mapping[str, Sequence[Any]]] = None,
        buffer_pools: Optional[Mapping[str, Sequence[Any]]] = None,
        local_page_size: Optional[int] = None,
        block_size: Optional[int] = None,
    ):
        self._sdk = sdk
        self._buffers = buffers
        self._layer_ids = tuple(map(int, layer_ids))
        self._buffer_owners = tuple(buffer_owners)
        self._buffer_tensors = {
            name: tuple(tensors) for name, tensors in (buffer_tensors or {}).items()
        }
        self._buffer_pools = {
            name: tuple(pools) for name, pools in (buffer_pools or {}).items()
        }
        self._local_page_size = local_page_size
        self._block_size = block_size
        if (local_page_size is None) != (block_size is None):
            raise ValueError("local_page_size and block_size must be set together")
        if block_size is not None:
            if local_page_size <= 0 or block_size % local_page_size:
                raise ValueError(
                    f"block_size={block_size} must be divisible by "
                    f"local_page_size={local_page_size}"
                )
            if set(self._buffer_tensors) != set(self._buffer_pools):
                raise ValueError("buffer tensor/pool families must match")
            for name, tensors in self._buffer_tensors.items():
                if len(tensors) != len(self._buffer_pools[name]):
                    raise ValueError(f"{name} buffer tensor/pool counts differ")

    @property
    def _pages_per_block(self) -> int:
        return (
            1 if self._block_size is None else self._block_size // self._local_page_size
        )

    def _logical_blocks(self, block_hash_ids, page_ids):
        """Group the physical pages backing each logical AttentionStore block."""
        page_ids = tuple(map(int, page_ids))
        pages_per_block = self._pages_per_block
        if pages_per_block == 1:
            if len(block_hash_ids) != len(page_ids):
                raise ValueError("hash and page counts differ")
            return tuple(
                (int(hash_id), (page_id,))
                for hash_id, page_id in zip(block_hash_ids, page_ids)
            )
        expected = len(block_hash_ids) * pages_per_block
        if len(page_ids) != expected:
            raise ValueError(
                f"{len(block_hash_ids)} blocks of size {self._block_size} require "
                f"{expected} local pages, got {len(page_ids)}"
            )
        result = []
        for index, hash_id in enumerate(block_hash_ids):
            pages = page_ids[index * pages_per_block : (index + 1) * pages_per_block]
            result.append((int(hash_id), pages))
        return tuple(result)






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
        producer_ready=None,
    ) -> int:
        """Write page blocks through the typed SDK buffers."""
        from attentionstore_sdk.sdk import BlockBatch, WriteRequest

        # Both the digest snapshot and the SDK gather read these pages, so the
        # fence has to come before either of them.
        wait_for_producer_event(
            producer_ready, timeout, debug_tag="write hash=%s" % (block_hash_ids[:1],)
        )
        if self._pages_per_block == 1:
            self._logical_blocks(block_hash_ids, page_ids)
            # Snapshot before the SDK call: the D2H gather happens inside write(),
            # so this is the source content actually handed to the Agent.
            written = self._sdk.write(
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
            return written
        blocks = self._logical_blocks(block_hash_ids, page_ids)
        written = self._sdk.write(
            WriteRequest(
                blocks=BlockBatch(
                    hash_ids=tuple(block[0] for block in blocks),
                    page_ids=tuple(block[1] for block in blocks),
                    layer_ids=self._layer_ids,
                ),
                buffers=self._buffers,
                timeout=timeout,
                params=params or {},
                h2h_copy=False,
            )
        )
        return written

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

        if self._pages_per_block == 1:
            self._logical_blocks(block_hash_ids, page_ids)
            block_kwargs = (
                {"logical_start": logical_start} if logical_start is not None else {}
            )
            read = self._sdk.read(
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
            return read
        blocks = self._logical_blocks(block_hash_ids, page_ids)
        block_kwargs = (
            {"logical_start": logical_start} if logical_start is not None else {}
        )
        return self._sdk.read(
            ReadRequest(
                blocks=BlockBatch(
                    hash_ids=tuple(block[0] for block in blocks),
                    page_ids=tuple(block[1] for block in blocks),
                    layer_ids=self._layer_ids,
                    **block_kwargs,
                ),
                buffers=self._buffers,
                timeout=timeout,
                remote_addrs=remote_addrs or (),
            )
        )
        return read

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
        page_write_start: int = 0,
    ) -> int:
        """Write suffix page IDs and the full prefix's terminal checkpoint."""
        if not block_hash_ids:
            return 0
        if not 0 <= page_write_start <= len(block_hash_ids):
            raise ValueError(
                f"page_write_start={page_write_start} is outside "
                f"[0, {len(block_hash_ids)}]"
            )
        wait_for_producer = getattr(self.checkpoint_store, "wait_for_producer", None)
        if wait_for_producer is not None:
            wait_for_producer(producer_ready, timeout=timeout)
        page_hashes = block_hash_ids[page_write_start:]
        # SDK counts newly written blocks; deduplicated pages/checkpoints return
        # fewer blocks (possibly zero), while write failures raise an exception.
        if page_hashes:
            page_result = self.page_store.write(
                page_hashes, page_ids, timeout=timeout, params=params
            )
            if page_result < 0:
                return -1
        checkpoint_result = self.checkpoint_store.write(
            int(block_hash_ids[-1]),
            checkpoint_slot,
            timeout=timeout,
            params=params,
            producer_ready=producer_ready,
        )
        return len(block_hash_ids) if checkpoint_result in (0, 1) else -1

    def stop(self) -> None:
        """Stop both the page and checkpoint stores."""
        try:
            self.page_store.stop()
        finally:
            self.checkpoint_store.stop()
