"""Shared AttentionStore SDK transport adapters."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from numbers import Integral
from typing import Any, Callable, Mapping, Optional, Sequence

import torch

from .types import AttentionStoreInitContext

logger = logging.getLogger(__name__)
# Cache-content diagnostic. Separate logger so it can be silenced or routed to
# its own file without touching the normal AS logs.
cache_digest_logger = logging.getLogger("asradix.cache_digest")


def cache_digest_enabled() -> bool:
    """Report whether the AttentionStore cache-content digest log is on."""
    from sglang.srt.environ import envs

    return envs.SGLANG_LOG_AS_CACHE_DIGEST.get()


def page_digest(tensor, page_id: int, rows_per_page: int = 1) -> str:
    """MD5 the buffer rows one page occupies inside a layer buffer.

    A token-major KV buffer stores ``rows_per_page`` token rows per page, so the
    digest has to cover the whole row range the transfer moves, not just row
    ``page_id``. Bytes are taken through a uint8 view because fp8 and bf16 have
    no numpy dtype, and the digest is truncated to 12 hex chars: the log is only
    ever compared against another line from the same run.
    """
    rows_per_page = max(1, int(rows_per_page))
    start = int(page_id) * rows_per_page
    rows = tensor[start : start + rows_per_page]
    host = rows.detach().to("cpu", copy=True).contiguous()
    return hashlib.md5(host.view(torch.uint8).numpy()).hexdigest()[:12]


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
            1
            if self._block_size is None
            else self._block_size // self._local_page_size
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

    def _read_prefix(self, block_hash_ids, page_ids, read: int):
        """Narrow a read request to the blocks the Agent actually filled.

        A partial read leaves the tail buffers holding whatever was there
        before, so digesting them would report a mismatch that says nothing
        about the blocks that did come back.
        """
        count = max(0, min(int(read), len(block_hash_ids)))
        pages = count * self._pages_per_block
        return block_hash_ids[:count], page_ids[:pages]

    def _rows_per_page(self, name: str, position: int, tensor) -> int:
        """Return how many buffer rows one local page occupies.

        Mirrors the layout split in ``_uniform_page_bytes``: a token-major
        buffer (first dim covers every token slot in the pool) holds one row per
        token, while a page-major buffer already holds one row per local page.
        """
        page_size = self._local_page_size or 1
        pools = self._buffer_pools.get(name, ())
        pool = pools[position] if position < len(pools) else None
        pool_size = getattr(pool, "size", None)
        if pool_size is not None and int(tensor.shape[0]) >= int(pool_size):
            return int(page_size)
        pool_page_size = int(getattr(pool, "page_size", page_size) or page_size)
        return max(1, int(page_size) // pool_page_size)

    def _snapshot_digests(
        self,
        block_hash_ids: Sequence[int],
        page_ids: Sequence[int],
    ) -> Optional[list]:
        """Hash the HBM pages a transfer touches, grouped per logical block.

        Only the first and last local layer are hashed: hashing every layer costs
        a D2H copy per layer per page, and a corruption that spares both ends of
        the layer range is not a case worth paying for by default.
        """
        if not cache_digest_enabled() or not block_hash_ids:
            return None
        try:
            blocks = self._logical_blocks(block_hash_ids, page_ids)
            snapshot = []
            for name, tensors in self._buffer_tensors.items():
                if not tensors:
                    continue
                positions = [0] if len(tensors) == 1 else [0, len(tensors) - 1]
                for position in positions:
                    tensor = tensors[position]
                    rows_per_page = self._rows_per_page(name, position, tensor)
                    snapshot.append(
                        (
                            name,
                            position,
                            (
                                self._layer_ids[position]
                                if position < len(self._layer_ids)
                                else -1
                            ),
                            [
                                (
                                    hash_id,
                                    [
                                        page_digest(tensor, page, rows_per_page)
                                        for page in pages
                                    ],
                                )
                                for hash_id, pages in blocks
                            ],
                        )
                    )
            return snapshot
        except Exception as exc:  # diagnostic only, never break the transfer
            cache_digest_logger.warning("AS digest snapshot failed: %s", exc)
            return None

    @staticmethod
    def _emit_digests(action: str, snapshot: Optional[list], transferred: int) -> None:
        """Print a previously captured digest snapshot once the transfer is done."""
        if not snapshot:
            return
        for name, position, layer_id, digests in snapshot:
            cache_digest_logger.info(
                "AS %s digest: family=%s layer_pos=%d layer_id=%s "
                "blocks=%d transferred=%s digests=%s",
                action,
                name,
                position,
                layer_id,
                len(digests),
                transferred,
                digests,
            )

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

        if self._pages_per_block == 1:
            self._logical_blocks(block_hash_ids, page_ids)
            # Snapshot before the SDK call: the D2H gather happens inside write(),
            # so this is the source content actually handed to the Agent.
            snapshot = self._snapshot_digests(block_hash_ids, page_ids)
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
            self._emit_digests("write", snapshot, written)
            return written
        blocks = self._logical_blocks(block_hash_ids, page_ids)
        snapshot = self._snapshot_digests(block_hash_ids, page_ids)
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
        self._emit_digests("write", snapshot, written)
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
            self._emit_digests(
                "read",
                self._snapshot_digests(
                    *self._read_prefix(block_hash_ids, page_ids, read)
                ),
                read,
            )
            return read
        blocks = self._logical_blocks(block_hash_ids, page_ids)
        block_kwargs = (
            {"logical_start": logical_start} if logical_start is not None else {}
        )
        read = self._sdk.read(
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
        self._emit_digests(
            "read",
            self._snapshot_digests(*self._read_prefix(block_hash_ids, page_ids, read)),
            read,
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
