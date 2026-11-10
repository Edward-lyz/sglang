"""AttentionStore-backed radix caches assembled from typed cache endpoints."""

from __future__ import annotations

import atexit
import concurrent.futures
import hashlib
import json
import logging
import os
import threading
import time
from array import array
from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InitLoadBackParams,
    InsertParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import (
    UnifiedRadixCache,
    UnifiedTreeNode,
)

logger = logging.getLogger(__name__)

AS_MIN_MATCH_BLOCKS = 5
_KDA_PENDING_HASH = "as_kda_pending_hash"
_KDA_PUBLISHED_HASH = "as_kda_published_hash"


class AttentionStoreEndpointRole(str, Enum):
    """Identify whether an endpoint stores pages or checkpoints."""

    PAGE = "PAGE"
    CHECKPOINT = "CHECKPOINT"


@dataclass(frozen=True)
class AttentionStoreInitContext:
    """Provide runtime resources used to construct an AttentionStore binding."""

    kv_cache: Any
    req_to_token_pool: Any
    namespace_base: str
    pod_name: str
    model_version: str
    page_size: int
    tp_rank: int
    tp_size: int
    pp_rank: int
    dp_rank: int
    device_id: int
    gpu_count: int
    target_layer_ids: Sequence[int] = ()
    draft_kv_cache: Any = None
    draft_layer_ids: Sequence[int] = ()


@dataclass(frozen=True)
class AttentionStoreEndpointBinding:
    """One independently named page or checkpoint storage contract."""

    name: str
    role: AttentionStoreEndpointRole
    store: Any
    namespace: str
    layout_id: str
    bytes_per_block: int
    payload_fields: Sequence[str]

    def __post_init__(self) -> None:
        """Normalize and validate the endpoint contract."""
        object.__setattr__(self, "payload_fields", tuple(self.payload_fields))
        if not self.name:
            raise ValueError("AttentionStore endpoint name must not be empty")
        if not self.namespace:
            raise ValueError(f"AttentionStore endpoint {self.name} has no namespace")
        if self.bytes_per_block <= 0:
            raise ValueError(
                f"AttentionStore endpoint {self.name} has invalid "
                f"bytes_per_block={self.bytes_per_block}"
            )
        if not self.payload_fields:
            raise ValueError(
                f"AttentionStore endpoint {self.name} has no payload fields"
            )


@dataclass(frozen=True)
class AttentionStoreBinding:
    """Runtime stores and endpoint schemas selected for one attention plan."""

    plan_name: str
    cache_store: Any
    endpoints: Sequence[AttentionStoreEndpointBinding]

    def __post_init__(self) -> None:
        """Normalize and validate the endpoint collection."""
        if not self.plan_name:
            raise ValueError("AttentionStore cache plan name must not be empty")
        endpoints = tuple(self.endpoints)
        object.__setattr__(self, "endpoints", endpoints)
        names = [endpoint.name for endpoint in endpoints]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate AttentionStore endpoint names: {names}")
        page_count = sum(
            endpoint.role is AttentionStoreEndpointRole.PAGE for endpoint in endpoints
        )
        checkpoint_count = sum(
            endpoint.role is AttentionStoreEndpointRole.CHECKPOINT
            for endpoint in endpoints
        )
        if page_count != 1:
            raise ValueError(
                "AttentionStore binding requires exactly one public page endpoint, "
                f"got {page_count}"
            )
        if checkpoint_count > 1:
            raise ValueError(
                "AttentionStore binding supports at most one checkpoint endpoint, "
                f"got {checkpoint_count}"
            )

    @property
    def page_endpoint(self) -> AttentionStoreEndpointBinding:
        """Return the binding's page endpoint."""
        return next(
            endpoint
            for endpoint in self.endpoints
            if endpoint.role is AttentionStoreEndpointRole.PAGE
        )

    @property
    def checkpoint_endpoint(self) -> Optional[AttentionStoreEndpointBinding]:
        """Return the optional checkpoint endpoint."""
        return next(
            (
                endpoint
                for endpoint in self.endpoints
                if endpoint.role is AttentionStoreEndpointRole.CHECKPOINT
            ),
            None,
        )

    @property
    def namespace(self) -> str:
        """Return the public page namespace."""
        return self.page_endpoint.namespace

    @property
    def layout_id(self) -> str:
        """Return a stable composite identifier for all endpoint layouts."""
        if len(self.endpoints) == 1:
            return self.endpoints[0].layout_id
        return ":".join(
            f"{endpoint.name}={endpoint.layout_id}" for endpoint in self.endpoints
        )

    @property
    def bytes_per_block(self) -> int:
        """Return the page endpoint's bytes per block."""
        return self.page_endpoint.bytes_per_block


@dataclass
class AttentionStoreWriteContext:
    """Capture request state across a local cache commit."""

    req: Any
    token_ids: Sequence[int]
    snapshot: torch.Tensor
    is_insert: bool
    prepared_write: Any = None


def _noop_release() -> None:
    return


def _once(release: Callable[[], None]) -> Callable[[], None]:
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


def _acquire_node_release(cache, node) -> Callable[[], None]:
    lock_result = cache.inc_lock_ref(node)

    def release() -> None:
        cache.dec_lock_ref(node, lock_result.to_dec_params())

    return _once(release)


def _contiguous_page_ids(slot_ids: torch.Tensor, page_size: int) -> tuple[int, ...]:
    """Validate page-aligned contiguous slots with one device-to-host sync."""
    return _materialize_page_ids(_page_id_metadata(slot_ids, page_size), page_size)


def _page_id_metadata(slot_ids: torch.Tensor, page_size: int) -> torch.Tensor:
    """Build page bases and validity flags on the source device."""
    if page_size <= 0 or slot_ids.numel() % page_size:
        raise ValueError(
            f"slot count {slot_ids.numel()} is not aligned to page_size={page_size}"
        )
    if not slot_ids.numel():
        return torch.empty((0, 2), dtype=torch.int64, device=slot_ids.device)

    pages = slot_ids.reshape(-1, page_size)
    page_bases = pages[:, 0]
    offsets = torch.arange(page_size, dtype=pages.dtype, device=pages.device)
    valid = torch.all(pages == page_bases[:, None] + offsets, dim=1)
    valid &= page_bases.remainder(page_size) == 0
    return torch.stack((page_bases, valid.to(page_bases.dtype)), dim=1)


def _materialize_page_ids(page_meta: torch.Tensor, page_size: int) -> tuple[int, ...]:
    """Validate host page metadata and return physical page identifiers."""
    page_meta = page_meta.tolist()
    for page_index, (_, is_valid) in enumerate(page_meta):
        if not is_valid:
            raise ValueError(
                f"non-contiguous page at prefix {(page_index + 1) * page_size}"
            )
    return tuple(int(page_base) // page_size for page_base, _ in page_meta)


@dataclass
class _PreparedPageIds:
    page_size: int
    metadata: torch.Tensor
    ready_event: Any = None
    source_ready_event: Any = None
    source_metadata: Optional[torch.Tensor] = None

    def materialize(self) -> tuple[int, ...]:
        """Wait for an asynchronous copy and return validated page IDs."""
        if self.ready_event is not None:
            self.ready_event.synchronize()
        return _materialize_page_ids(self.metadata, self.page_size)

    def release(self) -> None:
        """Wait until asynchronous metadata users are quiescent."""
        if self.ready_event is not None:
            self.ready_event.synchronize()


def _request_block_hash_ids(
    req, token_ids: Sequence[int], page_size: int, page_count: int
) -> tuple[int, ...]:
    """Reuse request block hashes when they cover the written prefix."""
    # Padding preserves length, so a length check alone would let the scheduler's
    # multimodal substitution (which rebinds origin_input_ids) hash away the image.
    if (
        req.as_block_hashs is not None
        and req.origin_input_ids is req.origin_input_ids_unpadded
    ):
        request_hashes = req.as_block_hashs.hash_ids
        if len(request_hashes) >= page_count:
            return tuple(map(int, request_hashes[:page_count]))

    from attentionstore_sdk.sdk import Tokens

    aligned_tokens = page_count * page_size
    return tuple(map(int, Tokens(token_ids[:aligned_tokens], page_size).hash_ids))


@dataclass
class _PageWriteTask:
    rid: str
    endpoint: Optional[AttentionStoreEndpointBinding]
    block_hash_ids: Sequence[int]
    data_id_list: Any
    write_params: Any
    release_after_write: Callable[[], None] = _noop_release
    release: Callable[[], None] = _noop_release

    @property
    def requested_blocks(self) -> int:
        """Return the number of page blocks requested by this task."""
        return len(self.block_hash_ids)


@dataclass
class _KDAWriteTask:
    rid: str
    node: UnifiedTreeNode
    block_hash_ids: Sequence[int]
    page_ids: Any
    checkpoint_slot: int
    params: dict
    request: Any = None
    write_key: Optional[tuple[int, int]] = None
    release_after_write: Callable[[], None] = _noop_release
    release: Callable[[], None] = _noop_release

    @property
    def requested_blocks(self) -> int:
        """Return the number of KDA page blocks requested by this task."""
        return len(self.block_hash_ids)

    @property
    def endpoint_hash(self) -> int:
        """Return the terminal block hash associated with the checkpoint."""
        return int(self.block_hash_ids[-1]) if self.block_hash_ids else 0


@dataclass(frozen=True)
class _SPACheckpointSnapshot:
    tensor: torch.Tensor
    release: Callable[[], None]


@dataclass
class _SPAWriteTask:
    rid: str
    page_endpoint: Optional[AttentionStoreEndpointBinding]
    checkpoint_store: Any
    block_hash_ids: Sequence[int]
    page_ids: Any
    checkpoint_snapshot: Optional[_SPACheckpointSnapshot]
    params: dict
    release_after_write: Callable[[], None] = _noop_release
    release: Callable[[], None] = _noop_release

    @property
    def requested_blocks(self) -> int:
        """Return the number of SPA page blocks requested by this task."""
        return len(self.block_hash_ids)


@dataclass
class _PendingWrite:
    future: concurrent.futures.Future
    task: Any


class AttentionStoreCoordinator:
    """Own local-commit hooks and the complete asynchronous write lifecycle.

    Subclasses capture stable request inputs and implement type-specific I/O.
    The base class owns submission ordering and the two resource horizons:
    staging is released when the writer returns, while cache-node retention is
    released only after the task is retired across the TP completion frontier.
    """

    write_thread_name = "as_writer"

    def __init__(self) -> None:
        """Initialize an idle asynchronous write coordinator."""
        self._write_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._pending_writes: list[_PendingWrite] = []
        self._page_id_copy_stream = None
        self._closed = False

    def prepare_page_ids(self, cache, slot_ids: torch.Tensor) -> _PreparedPageIds:
        """Stage validated page metadata without blocking the scheduler."""
        metadata = _page_id_metadata(slot_ids, cache.page_size)
        if metadata.device.type != "cuda" or not metadata.numel():
            return _PreparedPageIds(cache.page_size, metadata.cpu())

        if self._page_id_copy_stream is None:
            self._page_id_copy_stream = torch.cuda.Stream(device=metadata.device)
        source_ready = torch.cuda.Event()
        source_ready.record(torch.cuda.current_stream(metadata.device))
        host_metadata = torch.empty(
            metadata.shape,
            dtype=metadata.dtype,
            device="cpu",
            pin_memory=True,
        )
        with torch.cuda.stream(self._page_id_copy_stream):
            self._page_id_copy_stream.wait_event(source_ready)
            host_metadata.copy_(metadata, non_blocking=True)
            copy_ready = torch.cuda.Event()
            copy_ready.record(self._page_id_copy_stream)
        return _PreparedPageIds(
            cache.page_size,
            host_metadata,
            ready_event=copy_ready,
            source_ready_event=source_ready,
            source_metadata=metadata,
        )

    def create_binding(
        self, _context: AttentionStoreInitContext
    ) -> AttentionStoreBinding:
        """Create this coordinator's storage binding."""
        raise NotImplementedError

    def capture_finished(
        self, _cache, _req, _is_insert: bool
    ) -> Optional[AttentionStoreWriteContext]:
        """Capture state before committing a finished request."""
        return None

    def capture_unfinished(self, _cache, _req) -> Optional[AttentionStoreWriteContext]:
        """Capture state before committing an unfinished request."""
        return None

    def before_local_commit(self, _cache, _context: AttentionStoreWriteContext) -> None:
        """Prepare an AttentionStore write before the local commit."""
        return

    def after_local_commit(self, _cache, _context: AttentionStoreWriteContext) -> None:
        """Publish a prepared write after the local commit succeeds."""
        return

    def abort_local_commit(self, cache, context: AttentionStoreWriteContext) -> None:
        """Release a prepared write after the local commit fails."""
        task = context.prepared_write
        context.prepared_write = None
        if task is not None:
            self._discard_unsubmitted(cache, task)

    def extend_match(
        self,
        cache,
        params: MatchPrefixParams,
        local_result: MatchResult,
    ) -> Optional[MatchResult]:
        """Extend a local component-consistent match with an L2 hit."""
        raise NotImplementedError

    def init_load_back(
        self,
        cache,
        params: InitLoadBackParams,
    ) -> Tuple[torch.Tensor, Any]:
        """Restore an L2 hit into the components declared by the cache plan."""
        raise NotImplementedError

    def do_write(self, cache, task) -> int:
        """Execute one payload-specific AttentionStore write."""
        raise NotImplementedError

    def finish(self, _cache, _task, _result: int) -> None:
        """Apply payload-specific state after a write retires."""
        return

    def _ensure_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        if self._closed:
            raise RuntimeError("AttentionStore coordinator is already closed")
        if self._write_executor is None:
            self._write_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=self.write_thread_name,
            )
        return self._write_executor

    def _run_write(self, cache, task) -> int:
        try:
            return self.do_write(cache, task)
        finally:
            task.release_after_write()

    def _release_task_resource(
        self,
        cache,
        task,
        release: Callable[[], None],
        resource: str,
    ) -> Optional[Exception]:
        try:
            release()
            return None
        except Exception as exc:
            cache._log_as_failure(
                "write",
                rid=task.rid,
                requested=task.requested_blocks,
                error=RuntimeError(f"write {resource} release failed: {exc}"),
            )
            return exc

    def _discard_unsubmitted(self, cache, task) -> None:
        self._release_task_resource(
            cache, task, task.release_after_write, "staging resource"
        )
        self._release_task_resource(cache, task, task.release, "retained resource")

    def submit_write(self, cache, task) -> None:
        """Submit a write task to the ordered background executor."""
        try:
            future = self._ensure_executor().submit(self._run_write, cache, task)
        except Exception:
            self._discard_unsubmitted(cache, task)
            raise
        self._pending_writes.append(_PendingWrite(future=future, task=task))

    def submit_completed(self, task, result: int = 0) -> None:
        """Queue an already completed task for ordered retirement."""
        if self._closed:
            raise RuntimeError("AttentionStore coordinator is already closed")
        future = concurrent.futures.Future()
        future.set_result(result)
        self._pending_writes.append(_PendingWrite(future=future, task=task))

    def _resolve_result(
        self, cache, pending: _PendingWrite, timeout: Optional[float]
    ) -> int:
        try:
            return pending.future.result(timeout=timeout)
        except Exception as exc:
            cache._log_as_failure(
                "write",
                rid=pending.task.rid,
                requested=pending.task.requested_blocks,
                error=exc,
            )
            return -1

    def _retire(
        self,
        cache,
        pending_writes: Sequence[_PendingWrite],
        timeout: Optional[float],
    ) -> None:
        first_error = None
        for pending in pending_writes:
            result = self._resolve_result(cache, pending, timeout)
            try:
                self.finish(cache, pending.task, result)
            except Exception as exc:
                cache._log_as_failure(
                    "write",
                    rid=pending.task.rid,
                    requested=pending.task.requested_blocks,
                    error=RuntimeError(f"write completion failed: {exc}"),
                )
                first_error = first_error or exc
            release_error = self._release_task_resource(
                cache,
                pending.task,
                pending.task.release,
                "retained resource",
            )
            first_error = first_error or release_error
        if first_error is not None:
            raise RuntimeError(
                "AttentionStore write retirement failed"
            ) from first_error

    def check_events(self, cache) -> None:
        """Retire the completed write prefix shared by all TP ranks."""
        completed = 0
        for pending in self._pending_writes:
            if not pending.future.done():
                break
            completed += 1
        if cache.tp_world_size > 1:
            completed = cache._tp_min(completed)
        if not completed:
            return
        pending_writes = self._pending_writes[:completed]
        del self._pending_writes[:completed]
        self._retire(cache, pending_writes, timeout=1.0)

    def drain(self, cache) -> None:
        """Wait for and retire every pending write."""
        pending_writes = self._pending_writes
        self._pending_writes = []
        # A timed-out writer is still allowed to touch its page buffers.  A
        # reset or shutdown therefore waits for quiescence before releasing
        # retained radix nodes.
        self._retire(cache, pending_writes, timeout=None)

    def before_reset(self, cache) -> None:
        """Drain pending writes before resetting the cache."""
        self.drain(cache)

    def before_sanity_check(self, cache) -> None:
        """Drain pending writes before checking cache invariants."""
        self.drain(cache)

    def shutdown(self, cache) -> None:
        """Drain writes and stop the background executor."""
        try:
            self.drain(cache)
        finally:
            self._closed = True
            if self._write_executor is not None:
                self._write_executor.shutdown(wait=True)
                self._write_executor = None


class CheckpointedPageCacheCoordinator(AttentionStoreCoordinator):
    """Provide the shared remote-match contract for hybrid component plans."""

    def extend_match(
        self,
        cache,
        params: MatchPrefixParams,
        local_result: MatchResult,
    ) -> Optional[MatchResult]:
        """Extend a local component-consistent match with remote pages/state."""
        return cache._match_checkpointed_page_prefix(params, local_result)

    def init_load_back(
        self,
        cache,
        params: InitLoadBackParams,
    ) -> Tuple[torch.Tensor, Any]:
        """Restore one hybrid checkpointed-page prefix into UnifiedRadixCache."""
        raise NotImplementedError


class FullPageCacheCoordinator(AttentionStoreCoordinator):
    """Coordinate remote match, load-back, and writes for typed page endpoints."""

    def capture_finished(
        self, cache, req, is_insert: bool
    ) -> AttentionStoreWriteContext:
        """Snapshot the committed page slots of a finished request."""
        token_count = req._cache_commit_len()
        token_ids = (req.origin_input_ids + req.output_ids)[:token_count]
        snapshot = cache.req_to_token_pool.req_to_token[
            req.req_pool_idx, :token_count
        ].clone()
        return AttentionStoreWriteContext(
            req=req,
            token_ids=token_ids,
            snapshot=snapshot,
            is_insert=is_insert,
        )

    def extend_match(
        self,
        cache,
        params: MatchPrefixParams,
        local_result: MatchResult,
    ) -> Optional[MatchResult]:
        """Extend a local FULL-component match with remote page hits."""
        req = params.req
        if (
            req is None
            or req.ever_read_as
            or not req.as_block_hashs
            or cache.ascli is None
        ):
            return None
        max_hit_length = len(params.key) // cache.page_size * cache.page_size
        if local_result.device_indices.numel() == max_hit_length:
            return None
        return self._match_remote(cache, req, local_result)

    def init_load_back(
        self,
        cache,
        params: InitLoadBackParams,
    ) -> Tuple[torch.Tensor, Any]:
        """Load remote FULL-component pages into UnifiedRadixCache."""
        anchor = params.best_match_node
        empty = torch.empty((0,), dtype=torch.int64, device=cache.device)
        if params.host_hit_length <= 0:
            return empty, anchor

        anchor, load_info, req = cache._take_attention_store_load_context(params)
        if cache.ascli is None or load_info is None or req is None:
            return empty, anchor

        loaded_indices = self._load_pages(cache, load_info, params.host_hit_length)
        if loaded_indices is None or len(loaded_indices) == 0:
            return empty, anchor

        req.ever_read_as = True
        return self._insert_loaded(cache, req, anchor, load_info, loaded_indices)

    def _match_remote(
        self,
        cache,
        req,
        local_result: MatchResult,
    ) -> Optional[MatchResult]:
        local_hit_len = len(local_result.device_indices)
        start_block = local_hit_len // cache.page_size
        remote_addrs = req.as_max_hit_host_ip
        remote_len = req.as_max_hit_token_len

        if remote_len and len(remote_len) == 1 and remote_len[0] > 0:
            as_hit_tokens = (
                (remote_len[0] - local_hit_len) // cache.page_size
            ) * cache.page_size
            if as_hit_tokens <= 0:
                return None
        else:
            requested_hashes = req.as_block_hashs.hash_ids[start_block:]
            requested = len(requested_hashes)
            started = time.perf_counter()
            match_error = None
            try:
                match_count = cache.ascli.match(requested_hashes, timeout=1.0)
            except Exception as exc:
                match_error = exc
                match_count = 0

            match_count = cache._tp_min(match_count)
            max_blocks = max(
                (len(req.full_untruncated_fill_ids) - local_hit_len - 1)
                // cache.page_size,
                0,
            )
            match_count = min(match_count, max_blocks)
            if match_count < AS_MIN_MATCH_BLOCKS:
                match_count = 0
            if match_error is None:
                cache._log_as_result(
                    "match",
                    rid=req.rid,
                    requested=requested,
                    result=match_count,
                    started=started,
                )
            else:
                cache._log_as_failure(
                    "match",
                    rid=req.rid,
                    requested=requested,
                    error=match_error,
                    started=started,
                )
            if match_count == 0:
                return None
            as_hit_tokens = match_count * cache.page_size

        max_as_tokens = (
            (len(req.full_untruncated_fill_ids) - local_hit_len - 1)
            // cache.page_size
            * cache.page_size
        )
        if max_as_tokens <= 0:
            return None
        as_hit_tokens = min(as_hit_tokens, max_as_tokens)

        anchor = local_result.last_device_node
        req._as_load_info = {
            "rid": req.rid,
            "start_block": start_block,
            "as_hit_tokens": as_hit_tokens,
            "as_block_hashs": req.as_block_hashs,
            "remote_addrs": remote_addrs,
            "remote_len": remote_len,
        }
        return MatchResult(
            device_indices=local_result.device_indices,
            last_device_node=anchor,
            last_host_node=anchor,
            best_match_node=anchor,
            host_hit_length=as_hit_tokens,
        )

    def _load_pages(
        self,
        cache,
        load_info: dict,
        host_hit_length: int,
    ) -> Optional[torch.Tensor]:
        rid = load_info.get("rid", "-")
        requested = host_hit_length // cache.page_size
        started = time.perf_counter()
        as_hit_tokens = load_info["as_hit_tokens"]
        available = cache.token_to_kv_pool_allocator.available_size()
        if available < as_hit_tokens and cache.evictable_size() > 0:
            evict_tokens = max(as_hit_tokens - available, 0) + cache.page_size
            cache.evict(EvictParams(num_tokens=evict_tokens))

        try:
            alloc_indices = cache.token_to_kv_pool_allocator.alloc(as_hit_tokens)
        except Exception as exc:
            cache._log_as_failure(
                "load_back",
                rid=rid,
                requested=requested,
                error=exc,
                started=started,
            )
            return None
        if alloc_indices is None or len(alloc_indices) == 0:
            cache._log_as_failure(
                "load_back",
                rid=rid,
                requested=requested,
                error=RuntimeError(
                    "allocation failed: "
                    f"need={as_hit_tokens}, "
                    f"available={cache.token_to_kv_pool_allocator.available_size()}, "
                    f"evictable={cache.evictable_size()}"
                ),
                started=started,
            )
            return None
        try:
            data_ids = list(_contiguous_page_ids(alloc_indices, cache.page_size))
        except ValueError as exc:
            cache.token_to_kv_pool_allocator.free(alloc_indices)
            cache._log_as_failure(
                "load_back",
                rid=rid,
                requested=requested,
                error=exc,
                started=started,
            )
            return None

        start_block = load_info["start_block"]
        block_hash_ids = load_info["as_block_hashs"].hash_ids[
            start_block : start_block + len(data_ids)
        ]
        requested = len(block_hash_ids)
        remote_addrs = load_info.get("remote_addrs", [])
        remote_addr = remote_addrs[0] if remote_addrs else ""
        read_error = None
        try:
            read_count = cache.ascli.read(
                block_hash_ids,
                data_ids,
                timeout=1.0,
                remote_addrs=(remote_addr,) if remote_addr else (),
                logical_start=start_block,
            )
        except Exception as exc:
            read_error = exc
            read_count = 0
        read_count = cache._tp_min(read_count)
        if read_error is None:
            cache._log_as_result(
                "load_back",
                rid=rid,
                requested=requested,
                result=read_count,
                started=started,
            )
        else:
            cache._log_as_failure(
                "load_back",
                rid=rid,
                requested=requested,
                error=read_error,
                started=started,
            )
        if read_count <= 0:
            cache.token_to_kv_pool_allocator.free(alloc_indices)
            return None

        loaded_tokens = read_count * cache.page_size
        loaded_indices = alloc_indices[:loaded_tokens]
        if loaded_tokens < len(alloc_indices):
            cache.token_to_kv_pool_allocator.free(alloc_indices[loaded_tokens:])
        return loaded_indices

    def _insert_loaded(
        self,
        cache,
        req,
        anchor,
        load_info,
        loaded_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, Any]:
        local_hit_len = load_info["start_block"] * cache.page_size
        total_prefix_len = local_hit_len + len(loaded_indices)
        assert len(req.prefix_indices) == local_hit_len, (
            f"prefix_indices length {len(req.prefix_indices)} != "
            f"local_hit_len {local_hit_len} (start_block={load_info['start_block']}, "
            f"page_size={cache.page_size})"
        )
        token_ids = req.full_untruncated_fill_ids[:total_prefix_len]
        bigram_len = max(0, len(token_ids) - 1) if cache.is_eagle else len(token_ids)
        final_key_len = bigram_len // cache.page_size * cache.page_size
        if final_key_len == 0:
            cache.token_to_kv_pool_allocator.free(loaded_indices)
            return loaded_indices[:0], anchor

        full_values = torch.cat([req.prefix_indices, loaded_indices])
        values = full_values[:final_key_len].to(dtype=torch.int64, copy=True)
        tail_start = max(final_key_len - local_hit_len, 0)
        if tail_start < len(loaded_indices):
            cache.token_to_kv_pool_allocator.free(loaded_indices[tail_start:])
            loaded_indices = loaded_indices[:tail_start]

        cache.insert(
            InsertParams(
                key=RadixKey(token_ids, req.extra_key, is_bigram=cache.is_eagle),
                value=values,
                prev_prefix_len=local_hit_len,
            )
        )
        match = UnifiedRadixCache.match_prefix(
            cache,
            MatchPrefixParams(
                key=RadixKey(token_ids, req.extra_key, is_bigram=cache.is_eagle),
            ),
        )
        if match.last_device_node is None:
            return loaded_indices, anchor

        tree_full_indices = match.device_indices
        if len(tree_full_indices) >= final_key_len:
            # _insert_helper already freed the slots the tree displaced.
            loaded_indices = tree_full_indices[
                local_hit_len : local_hit_len + len(loaded_indices)
            ].to(dtype=loaded_indices.dtype)
        return loaded_indices, match.last_device_node

    def after_local_commit(self, cache, context: AttentionStoreWriteContext) -> None:
        """Enqueue eligible page data after the local commit."""
        if context.is_insert and cache._is_prefill_role() and cache.ascli is not None:
            self.enqueue(
                cache,
                context.req,
                context.token_ids,
                context.snapshot,
            )

    def enqueue_dummy(self) -> None:
        """Queue a no-op task to preserve TP retirement ordering."""
        task = _PageWriteTask(
            rid="-",
            endpoint=None,
            block_hash_ids=(),
            data_id_list=[],
            write_params={},
        )
        self.submit_completed(task)

    def do_write(self, cache, task: _PageWriteTask) -> int:
        """Write a prepared page task to its endpoint."""
        started = time.perf_counter()
        try:
            endpoint = task.endpoint
            if endpoint is None:
                return 0
            data_ids = task.data_id_list.materialize()
            written = endpoint.store.write(
                task.block_hash_ids,
                data_ids,
                timeout=1.0,
                params=task.write_params,
            )
            cache._log_as_result(
                "write",
                rid=task.rid,
                requested=task.requested_blocks,
                result=written,
                started=started,
            )
            return written
        except Exception as exc:
            cache._log_as_failure(
                "write",
                rid=task.rid,
                requested=task.requested_blocks,
                error=exc,
                started=started,
            )
            return -1

    def enqueue(
        self,
        cache,
        req,
        token_ids: Sequence[int],
        snapshot: torch.Tensor,
    ) -> None:
        """Build and submit a page write for an eligible radix prefix."""
        if cache.binding is None:
            if cache.tp_world_size > 1:
                self.enqueue_dummy()
            return
        task = None
        num_pages = 0
        try:
            num_pages = len(token_ids) // cache.page_size
            if num_pages <= 0:
                if cache.tp_world_size > 1:
                    self.enqueue_dummy()
                return
            aligned_tokens = num_pages * cache.page_size
            hashes = _request_block_hash_ids(
                req,
                token_ids,
                cache.page_size,
                num_pages,
            )
            kv_indices = snapshot[:aligned_tokens]
            match = UnifiedRadixCache.match_prefix(
                cache,
                MatchPrefixParams(
                    key=RadixKey(token_ids, req.extra_key, is_bigram=cache.is_eagle)
                ),
            )
            node = match.last_device_node
            if node is None:
                cache._log_as_failure(
                    "write",
                    rid=req.rid,
                    requested=num_pages,
                    error=RuntimeError("no radix node matched"),
                )
                if cache.tp_world_size > 1:
                    self.enqueue_dummy()
                return
            data_ids = self.prepare_page_ids(cache, kv_indices)

            task = _PageWriteTask(
                rid=req.rid,
                endpoint=cache.binding.page_endpoint,
                block_hash_ids=hashes,
                data_id_list=data_ids,
                write_params=getattr(req, "as_kv_cache_params", None) or {},
                release_after_write=data_ids.release,
            )
            task.release = _acquire_node_release(cache, node)
            self.submit_write(cache, task)
        except Exception as exc:
            if task is not None:
                self._discard_unsubmitted(cache, task)
            cache._log_as_failure(
                "write",
                rid=req.rid,
                requested=num_pages,
                error=exc,
            )
            if cache.tp_world_size > 1:
                self.enqueue_dummy()


class MHACacheCoordinator(FullPageCacheCoordinator):
    """Coordinate MHA page cache I/O."""

    def create_binding(
        self, context: AttentionStoreInitContext
    ) -> AttentionStoreBinding:
        return create_mha_binding(context)


class MLACacheCoordinator(FullPageCacheCoordinator):
    """Coordinate MLA page cache I/O."""

    def create_binding(
        self, context: AttentionStoreInitContext
    ) -> AttentionStoreBinding:
        return create_mla_binding(context)


class DSACacheCoordinator(FullPageCacheCoordinator):
    """Coordinate DSA page cache I/O."""

    def create_binding(
        self, context: AttentionStoreInitContext
    ) -> AttentionStoreBinding:
        return create_dsa_binding(context)


class SPACacheCoordinator(CheckpointedPageCacheCoordinator):
    """Coordinate SPA page writes with a stable endpoint checkpoint snapshot."""

    def create_binding(
        self, context: AttentionStoreInitContext
    ) -> AttentionStoreBinding:
        return create_spa_binding(context)

    write_thread_name = "as_spa_writer"

    def capture_finished(
        self, cache, req, is_insert: bool
    ) -> AttentionStoreWriteContext:
        """Snapshot SPA slots for a finished request."""
        committed_len = req._cache_commit_len()
        token_ids = (req.origin_input_ids + req.output_ids)[:committed_len]
        snapshot = cache.req_to_token_pool.req_to_token[
            req.req_pool_idx, :committed_len
        ].clone()
        return AttentionStoreWriteContext(
            req=req,
            token_ids=token_ids,
            snapshot=snapshot,
            is_insert=is_insert,
        )

    def capture_unfinished(self, cache, req) -> AttentionStoreWriteContext:
        """Snapshot SPA slots for an unfinished request."""
        token_ids = req.get_fill_ids()
        snapshot = cache.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ].clone()
        return AttentionStoreWriteContext(
            req=req,
            token_ids=token_ids,
            snapshot=snapshot,
            is_insert=True,
        )

    def before_local_commit(self, cache, context: AttentionStoreWriteContext) -> None:
        """Pack a stable SPA checkpoint before the local commit."""
        if context.is_insert and cache._is_prefill_role() and cache.ascli is not None:
            context.prepared_write = self.prepare_write(
                cache,
                context.req,
                context.token_ids,
                context.snapshot,
            )

    def after_local_commit(self, cache, context: AttentionStoreWriteContext) -> None:
        """Retain the committed SPA node and submit its prepared write."""
        if not context.is_insert or not cache._is_prefill_role() or cache.ascli is None:
            return
        task = context.prepared_write
        if task is None:
            self.enqueue_dummy(cache)
            return

        try:
            key = RadixKey(
                context.token_ids,
                context.req.extra_key,
                is_bigram=cache.is_eagle,
            ).page_aligned(cache.page_size)
            match = UnifiedRadixCache.match_prefix(
                cache,
                MatchPrefixParams(key=key),
            )
            node = match.last_device_node
            if node is None or len(match.device_indices) < len(key):
                raise RuntimeError("SPA local commit did not publish the write prefix")
            task.release = _acquire_node_release(cache, node)
            self.submit_write(cache, task)
            context.prepared_write = None
        except Exception as exc:
            self._discard_unsubmitted(cache, task)
            context.prepared_write = None
            cache._log_as_failure(
                "write",
                rid=context.req.rid,
                requested=task.requested_blocks,
                error=exc,
            )
            self.enqueue_dummy(cache)

    def state_slots(self, cache, req_pool_idx: int, swa_page_id: int) -> dict[str, int]:
        """Return the logical SPA checkpoint slots for a request."""
        slots = {}
        for group in cache.ascli.layout.groups:
            if group in ("swa_page", "c4_state"):
                slots[group] = swa_page_id
            elif group in ("swa_request", "c128_state"):
                slots[group] = req_pool_idx
            else:
                raise ValueError(f"unsupported SPA checkpoint group: {group}")
        return slots

    def prepare_write(
        self,
        cache,
        req,
        token_ids: Sequence[int],
        snapshot: torch.Tensor,
    ) -> Optional[_SPAWriteTask]:
        """Prepare page identifiers and a stable SPA checkpoint snapshot."""
        if cache.ascli is None or not token_ids:
            return None
        cache_len = len(token_ids)
        if cache_len % cache.page_size != 0 or len(snapshot) < cache_len:
            return None
        requested = 0
        checkpoint_snapshot = None
        try:
            page_count = cache_len // cache.page_size
            swa_indices = (
                cache.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                    snapshot[cache_len - cache.page_size : cache_len]
                )
            )
            swa_page_id = int(swa_indices[0].item()) // cache.page_size
            hashes = _request_block_hash_ids(
                req,
                token_ids,
                cache.page_size,
                page_count,
            )
            requested = len(hashes)
            checkpoint_endpoint = cache.binding.checkpoint_endpoint
            if checkpoint_endpoint is None:
                raise ValueError(
                    "SPA AttentionStore binding has no checkpoint endpoint"
                )
            checkpoint_store = checkpoint_endpoint.store
            checkpoint_snapshot = checkpoint_store.snapshot(
                self.state_slots(cache, req.req_pool_idx, swa_page_id)
            )
            page_ids = self.prepare_page_ids(cache, snapshot[:cache_len])

            def release_staging() -> None:
                try:
                    page_ids.release()
                finally:
                    checkpoint_snapshot.release()

            return _SPAWriteTask(
                rid=req.rid,
                page_endpoint=cache.binding.page_endpoint,
                checkpoint_store=checkpoint_store,
                block_hash_ids=hashes,
                page_ids=page_ids,
                checkpoint_snapshot=checkpoint_snapshot,
                params=getattr(req, "as_kv_cache_params", None) or {},
                release_after_write=release_staging,
            )
        except Exception as exc:
            if checkpoint_snapshot is not None:
                checkpoint_snapshot.release()
            cache._log_as_failure(
                "write",
                rid=req.rid,
                requested=requested,
                error=exc,
            )
            return None

    def init_load_back(
        self,
        cache,
        params: InitLoadBackParams,
    ) -> Tuple[torch.Tensor, Any]:
        """Load matched SPA pages and checkpoint state into unified storage."""
        anchor = params.best_match_node
        empty = torch.empty((0,), dtype=torch.int64, device=cache.device)
        if params.host_hit_length <= 0:
            return empty, anchor

        anchor, load_info, req = cache._take_attention_store_load_context(params)
        if cache.ascli is None or load_info is None or req is None:
            return empty, anchor

        requested = int(load_info["block_count"])
        started = time.perf_counter()
        token_count = requested * cache.page_size
        available = cache.token_to_kv_pool_allocator.available_size()
        if available < token_count and cache.full_evictable_size() > 0:
            cache.evict(
                EvictParams(
                    num_tokens=token_count - available + cache.page_size,
                    swa_num_tokens=cache.page_size,
                )
            )
        local_hit_len = int(load_info["start_block"]) * cache.page_size
        device = cache.token_to_kv_pool_allocator.device
        prefix_cpu = torch.tensor([local_hit_len], dtype=torch.int64)
        seq_cpu = torch.tensor([local_hit_len + token_count], dtype=torch.int64)
        last_loc_value = int(req.prefix_indices[-1].item()) if local_hit_len else -1
        loaded = cache.token_to_kv_pool_allocator.alloc_extend_swa_tail(
            prefix_cpu.to(device),
            prefix_cpu,
            seq_cpu.to(device),
            seq_cpu,
            torch.tensor([last_loc_value], dtype=torch.int64, device=device),
            token_count,
            min(token_count, cache.page_size),
        )
        if loaded is None:
            cache._log_as_failure(
                "load_back",
                rid=req.rid,
                requested=requested,
                error=RuntimeError(f"cannot allocate {token_count} tokens"),
                started=started,
            )
            return empty, anchor
        try:
            page_ids = _contiguous_page_ids(loaded, cache.page_size)
        except ValueError as exc:
            cache.token_to_kv_pool_allocator.free(loaded)
            cache._log_as_failure(
                "load_back",
                rid=req.rid,
                requested=requested,
                error=exc,
                started=started,
            )
            return empty, anchor
        swa_indices = cache.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
            loaded[-cache.page_size :]
        )
        swa_page_id = int(swa_indices[0].item()) // cache.page_size
        state_slots = self.state_slots(cache, req.req_pool_idx, swa_page_id)
        read_error = None
        try:
            read_count = cache.ascli.read(
                load_info["hashes"],
                page_ids,
                state_slots,
                timeout=1.0,
                remote_addrs=load_info.get("remote_addrs") or None,
            )
        except Exception as exc:
            read_error = exc
            read_count = 0
        read_count = cache._tp_min(read_count)
        if read_error is None:
            cache._log_as_result(
                "load_back",
                rid=req.rid,
                requested=requested,
                result=read_count,
                started=started,
            )
        else:
            cache._log_as_failure(
                "load_back",
                rid=req.rid,
                requested=requested,
                error=read_error,
                started=started,
            )
        if read_count != requested:
            cache.token_to_kv_pool_allocator.free(loaded)
            return empty, anchor

        total_prefix_len = local_hit_len + token_count
        token_ids = req.full_untruncated_fill_ids[:total_prefix_len]
        values = torch.cat([req.prefix_indices, loaded]).to(
            dtype=torch.int64, copy=True
        )
        cache.insert(
            InsertParams(
                key=RadixKey(token_ids, req.extra_key, is_bigram=cache.is_eagle),
                value=values,
                prev_prefix_len=local_hit_len,
                swa_evicted_seqlen=local_hit_len
                + max(0, token_count - cache.page_size),
            )
        )
        match = UnifiedRadixCache.match_prefix(
            cache,
            MatchPrefixParams(
                key=RadixKey(token_ids, req.extra_key, is_bigram=cache.is_eagle),
                req=req,
            ),
        )
        tree_indices = match.device_indices[local_hit_len:total_prefix_len].to(
            dtype=loaded.dtype
        )
        if len(tree_indices) != token_count:
            logger.warning(
                "AS cache insert: cache_plan=%s rid=%s status=error "
                "reason=shortened_prefix expected_tokens=%s actual_tokens=%s",
                cache.attention_store_plan.name,
                req.rid,
                token_count,
                len(tree_indices),
            )
        req.ever_read_as = True
        return tree_indices, match.last_device_node

    def do_write(self, cache, task: _SPAWriteTask) -> int:
        """Write SPA pages followed by their terminal checkpoint."""
        if not task.block_hash_ids:
            return 0
        started = time.perf_counter()
        try:
            page_ids = task.page_ids.materialize()
            confirmed_pages = task.page_endpoint.store.match(
                task.block_hash_ids,
                timeout=1.0,
            )
            confirmed_pages = max(0, min(confirmed_pages, task.requested_blocks))
            if confirmed_pages < task.requested_blocks:
                task.page_endpoint.store.write(
                    task.block_hash_ids[confirmed_pages:],
                    page_ids[confirmed_pages:],
                    timeout=1.0,
                    params=task.params,
                )
                confirmed_pages = task.page_endpoint.store.match(
                    task.block_hash_ids,
                    timeout=1.0,
                )
                confirmed_pages = max(0, min(confirmed_pages, task.requested_blocks))
            if confirmed_pages != task.requested_blocks:
                result = confirmed_pages
            else:
                checkpoint_hash = int(task.block_hash_ids[-1])
                checkpoint_result = task.checkpoint_store.match(
                    [checkpoint_hash], timeout=1.0
                )
                if checkpoint_result != 1:
                    checkpoint_result = task.checkpoint_store.write_snapshot(
                        checkpoint_hash,
                        task.checkpoint_snapshot,
                        timeout=1.0,
                        params=task.params,
                    )
                result = confirmed_pages if checkpoint_result == 1 else -1
            cache._log_as_result(
                "write",
                rid=task.rid,
                requested=task.requested_blocks,
                result=result,
                started=started,
            )
            return result
        except Exception as exc:
            cache._log_as_failure(
                "write",
                rid=task.rid,
                requested=task.requested_blocks,
                error=exc,
                started=started,
            )
            return -1

    def enqueue_dummy(self, cache) -> None:
        """Queue a no-op SPA task to preserve TP retirement ordering."""
        if cache.tp_world_size <= 1:
            return
        self.submit_completed(
            _SPAWriteTask(
                rid="-",
                page_endpoint=None,
                checkpoint_store=None,
                block_hash_ids=(),
                page_ids=(),
                checkpoint_snapshot=None,
                params={},
            )
        )


class KDACacheCoordinator(CheckpointedPageCacheCoordinator):
    """Coordinate full-attention pages and recurrent KDA endpoint checkpoints."""

    def create_binding(
        self, context: AttentionStoreInitContext
    ) -> AttentionStoreBinding:
        return create_kda_binding(context)

    write_thread_name = "as_kda_writer"

    def capture_finished(
        self, cache, req, is_insert: bool
    ) -> AttentionStoreWriteContext:
        """Snapshot KDA page slots for a finished request."""
        token_ids = array("q", req.origin_input_ids)
        token_ids.extend(req.output_ids)
        snapshot = cache.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ].clone()
        return AttentionStoreWriteContext(
            req=req,
            token_ids=token_ids,
            snapshot=snapshot,
            is_insert=is_insert,
        )

    def capture_unfinished(self, cache, req) -> AttentionStoreWriteContext:
        """Snapshot KDA page slots for an unfinished request."""
        token_ids = req.get_fill_ids()
        snapshot = cache.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ].clone()
        return AttentionStoreWriteContext(
            req=req,
            token_ids=token_ids,
            snapshot=snapshot,
            is_insert=True,
        )

    def after_local_commit(self, cache, context: AttentionStoreWriteContext) -> None:
        """Enqueue eligible KDA state after the local commit."""
        if context.is_insert and cache._is_prefill_role() and cache.ascli is not None:
            self.enqueue(
                cache,
                context.req,
                context.token_ids,
                context.snapshot,
            )

    def enqueue(
        self,
        cache,
        req,
        token_ids: Sequence[int],
        snapshot: torch.Tensor,
    ) -> None:
        """Build and submit a KDA page-and-checkpoint write."""
        task = None
        cache_len = 0
        try:
            if not isinstance(token_ids, array) or token_ids.typecode != "q":
                token_ids = array("q", token_ids)
            match = UnifiedRadixCache.match_prefix(
                cache,
                MatchPrefixParams(key=RadixKey(token_ids, req.extra_key)),
            )
            cache_len = len(match.device_indices)
            node = match.last_device_node
            mamba_data = (
                node.component_data[ComponentType.MAMBA] if node is not None else None
            )
            if (
                cache_len < cache.page_size
                or cache_len % cache.page_size != 0
                or node is None
                or mamba_data.value is None
                or len(snapshot) < cache_len
            ):
                self.enqueue_dummy(cache)
                return

            page_count = cache_len // cache.page_size
            hashes = _request_block_hash_ids(
                req,
                token_ids,
                cache.page_size,
                page_count,
            )
            endpoint_hash = hashes[-1]
            if mamba_data.metadata.get(_KDA_PENDING_HASH) is not None or (
                endpoint_hash == mamba_data.metadata.get(_KDA_PUBLISHED_HASH)
            ):
                self.enqueue_dummy(cache)
                return
            write_key = (cache_len, endpoint_hash)
            if getattr(req, "_as_kda_last_submitted_write", None) == write_key:
                self.enqueue_dummy(cache)
                return
            checkpoint_slot = int(mamba_data.value.item())
            page_ids = self.prepare_page_ids(cache, snapshot[:cache_len])
            task = _KDAWriteTask(
                rid=req.rid,
                node=node,
                block_hash_ids=hashes,
                page_ids=page_ids,
                checkpoint_slot=checkpoint_slot,
                params=getattr(req, "as_kv_cache_params", None) or {},
                request=req,
                write_key=write_key,
                release_after_write=page_ids.release,
            )
            task.release = _acquire_node_release(cache, node)
            mamba_data.metadata[_KDA_PENDING_HASH] = endpoint_hash
            req._as_kda_last_submitted_write = write_key
            try:
                self.submit_write(cache, task)
            except Exception:
                if mamba_data.metadata.get(_KDA_PENDING_HASH) == endpoint_hash:
                    del mamba_data.metadata[_KDA_PENDING_HASH]
                if getattr(req, "_as_kda_last_submitted_write", None) == write_key:
                    del req._as_kda_last_submitted_write
                raise
        except Exception as exc:
            if task is not None:
                self._discard_unsubmitted(cache, task)
            cache._log_as_failure(
                "write",
                rid=req.rid,
                requested=cache_len // cache.page_size,
                error=exc,
            )
            self.enqueue_dummy(cache)

    def do_write(self, cache, task: _KDAWriteTask) -> int:
        """Write KDA pages and recurrent state through the composite store."""
        started = time.perf_counter()
        try:
            page_ids = task.page_ids.materialize()
            written = cache.ascli.write(
                task.block_hash_ids,
                page_ids,
                {"checkpoint": task.checkpoint_slot},
                timeout=1.0,
                params=task.params,
            )
            cache._log_as_result(
                "write",
                rid=task.rid,
                requested=task.requested_blocks,
                result=written,
                started=started,
            )
            return written
        except Exception as exc:
            cache._log_as_failure(
                "write",
                rid=task.rid,
                requested=task.requested_blocks,
                error=exc,
                started=started,
            )
            return -1

    def finish(self, _cache, task: _KDAWriteTask, result: int) -> None:
        """Publish or clear the endpoint hash after KDA write completion."""
        if task.endpoint_hash == 0:
            return
        metadata = task.node.component_data[ComponentType.MAMBA].metadata
        if metadata.get(_KDA_PENDING_HASH) == task.endpoint_hash:
            del metadata[_KDA_PENDING_HASH]
        if result == task.requested_blocks:
            metadata[_KDA_PUBLISHED_HASH] = task.endpoint_hash
        elif task.request is not None and (
            getattr(task.request, "_as_kda_last_submitted_write", None)
            == task.write_key
        ):
            del task.request._as_kda_last_submitted_write

    def init_load_back(
        self,
        cache,
        params: InitLoadBackParams,
    ) -> Tuple[torch.Tensor, Any]:
        """Load matched KDA pages and recurrent state into unified storage."""
        anchor = params.best_match_node
        empty = torch.empty((0,), dtype=torch.int64, device=cache.device)
        if params.host_hit_length <= 0:
            return empty, anchor

        anchor, load_info, req = cache._take_attention_store_load_context(params)
        if cache.ascli is None or load_info is None or req is None:
            return empty, anchor

        loaded = self._read_blocks(cache, load_info)
        if loaded is None:
            return empty, anchor
        loaded_indices, value_transaction, checkpoint_allocation = loaded
        req.ever_read_as = True
        return self._insert_loaded(
            cache,
            req,
            anchor,
            load_info,
            loaded_indices,
            value_transaction,
            checkpoint_allocation,
        )

    def _read_blocks(self, cache, load_info):
        block_count = int(load_info["block_count"])
        rid = load_info.get("rid", "-")
        started = time.perf_counter()
        token_count = block_count * cache.page_size
        value_transaction = cache.component_value_transaction()
        try:
            full_allocation = value_transaction.allocate(
                ComponentType.FULL, token_count
            )
            alloc_indices = full_allocation.value
        except Exception as exc:
            value_transaction.close()
            cache._log_as_failure(
                "load_back",
                rid=rid,
                requested=block_count,
                error=exc,
                started=started,
            )
            return None
        if len(alloc_indices) != token_count:
            value_transaction.close()
            cache._log_as_failure(
                "load_back",
                rid=rid,
                requested=block_count,
                error=RuntimeError(
                    f"allocated {len(alloc_indices)} of {token_count} tokens"
                ),
                started=started,
            )
            return None

        try:
            page_ids = _contiguous_page_ids(alloc_indices, cache.page_size)
        except ValueError as exc:
            value_transaction.close()
            cache._log_as_failure(
                "load_back",
                rid=rid,
                requested=block_count,
                error=exc,
                started=started,
            )
            return None

        try:
            checkpoint_allocation = value_transaction.allocate(ComponentType.MAMBA)
        except Exception as exc:
            value_transaction.close()
            cache._log_as_failure(
                "load_back",
                rid=rid,
                requested=block_count,
                error=exc,
                started=started,
            )
            return None
        checkpoint_slot = checkpoint_allocation.value
        keep_transaction = False
        try:
            state_id = int(checkpoint_slot.item())
            read_error = None
            try:
                read_count = cache.ascli.read(
                    load_info["hashes"],
                    page_ids,
                    {"checkpoint": state_id},
                    timeout=1.0,
                    remote_addrs=load_info.get("remote_addrs") or None,
                )
            except Exception as exc:
                read_error = exc
                read_count = 0

            read_count = cache._tp_min(read_count)

            if read_error is None:
                cache._log_as_result(
                    "load_back",
                    rid=rid,
                    requested=block_count,
                    result=read_count,
                    started=started,
                )
            else:
                cache._log_as_failure(
                    "load_back",
                    rid=rid,
                    requested=block_count,
                    error=read_error,
                    started=started,
                )

            if read_count != block_count:
                return None
            keep_transaction = True
            return alloc_indices, value_transaction, checkpoint_allocation
        finally:
            if not keep_transaction:
                value_transaction.close()

    def _insert_loaded(
        self,
        cache,
        req,
        anchor,
        load_info,
        loaded_indices: torch.Tensor,
        value_transaction,
        checkpoint_allocation,
    ) -> Tuple[torch.Tensor, Any]:
        local_hit_len = load_info["start_block"] * cache.page_size
        total_prefix_len = local_hit_len + len(loaded_indices)
        assert len(req.prefix_indices) == local_hit_len, (
            f"prefix_indices length {len(req.prefix_indices)} != "
            f"local_hit_len {local_hit_len} (start_block={load_info['start_block']}, "
            f"page_size={cache.page_size})"
        )
        token_ids = req.full_untruncated_fill_ids[:total_prefix_len]
        values = torch.cat([req.prefix_indices, loaded_indices]).to(
            dtype=torch.int64, copy=True
        )
        try:
            result = cache.insert(
                InsertParams(
                    key=RadixKey(token_ids, req.extra_key),
                    value=values,
                    mamba_value=checkpoint_allocation.value,
                    prev_prefix_len=local_hit_len,
                ),
                value_transaction=value_transaction,
            )
            value_transaction.commit_insert(result)
        finally:
            # Release values not adopted by the tree.
            value_transaction.close()

        match = UnifiedRadixCache.match_prefix(
            cache,
            MatchPrefixParams(
                key=RadixKey(token_ids, req.extra_key),
                cow_mamba=True,
                req=req,
            ),
        )
        tree_indices = match.device_indices[local_hit_len:total_prefix_len].to(
            dtype=loaded_indices.dtype
        )
        if len(tree_indices) != len(loaded_indices):
            logger.warning(
                "AS cache insert: cache_plan=%s rid=%s status=error "
                "reason=shortened_prefix expected_tokens=%s actual_tokens=%s",
                cache.attention_store_plan.name,
                req.rid,
                len(loaded_indices),
                len(tree_indices),
            )
        elif load_info["hashes"]:
            match.last_device_node.component_data[ComponentType.MAMBA].metadata[
                _KDA_PUBLISHED_HASH
            ] = int(load_info["hashes"][-1])
        return tree_indices, match.last_device_node

    def enqueue_dummy(self, cache) -> None:
        """Queue a no-op KDA task to preserve TP retirement ordering."""
        if cache.tp_world_size <= 1:
            return
        self.submit_completed(
            _KDAWriteTask(
                rid="-",
                node=cache.root_node,
                block_hash_ids=(),
                page_ids=(),
                checkpoint_slot=0,
                params={},
            )
        )


@dataclass(frozen=True)
class AttentionStoreCachePlan:
    """Map an attention capability to components and a coordinator."""

    name: str
    coordinator_type: type[AttentionStoreCoordinator]
    tree_components: tuple[ComponentType, ...]

    def __post_init__(self) -> None:
        """Validate the cache plan name."""
        if not self.name:
            raise ValueError("AttentionStore cache plan name must not be empty")
        if (
            not self.tree_components
            or self.tree_components[0] is not ComponentType.FULL
        ):
            raise ValueError("AttentionStore plans must start with the FULL component")


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

    def _tp_min(self, value: int) -> int:
        if self.tp_world_size <= 1:
            return value
        tensor = torch.tensor(value, dtype=torch.int)
        torch.distributed.all_reduce(
            tensor,
            op=torch.distributed.ReduceOp.MIN,
            group=self.tp_group,
        )
        return int(tensor.item())

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

        match_count = min(self._tp_min(match_count), requested)
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
            "remote_addrs": req.as_max_hit_host_ip or [],
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
        """Create the configured coordinator's binding."""
        return self.attention_store_coordinator.create_binding(context)

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


def _sdk_pod_config(
    ctx: AttentionStoreInitContext,
    namespace: str,
    *,
    model_version: Optional[str] = None,
    gpu_count: Optional[int] = None,
):
    from attentionstore_sdk.sdk import PodConfig

    return PodConfig(
        namespace=namespace,
        pod_name=ctx.pod_name,
        model_version=model_version or ctx.model_version,
        device_id=ctx.device_id,
        dp_id=ctx.dp_rank,
        gpu_count=ctx.gpu_count if gpu_count is None else gpu_count,
    )


def _key_buffer(pool, layer_id: int) -> torch.Tensor:
    getter = getattr(pool, "_get_key_buffer", None) or pool.get_key_buffer
    return getter(layer_id)


def _value_buffer(pool, layer_id: int) -> torch.Tensor:
    getter = getattr(pool, "_get_value_buffer", None) or pool.get_value_buffer
    return getter(layer_id)


def _uniform_page_bytes(
    buffers: Sequence[torch.Tensor],
    pools: Sequence[Any],
    page_size: int,
    name: str,
) -> int:
    if not buffers:
        raise ValueError(f"{name} requires at least one buffer")
    if len(buffers) != len(pools):
        raise ValueError(f"{name} buffer/pool metadata length mismatch")
    sizes = set()
    for buffer, pool in zip(buffers, pools):
        row_bytes = int(buffer[0].nbytes)
        pool_size = getattr(pool, "size", None)
        sizes.add(
            row_bytes * page_size
            if pool_size is not None and buffer.shape[0] >= int(pool_size)
            else row_bytes
        )
    if len(sizes) != 1:
        raise ValueError(f"{name} buffers have non-uniform block sizes: {sizes}")
    return sizes.pop()


def _page_layout_id(
    page_format: str,
    page_size: int,
    buffers: Mapping[str, Sequence[torch.Tensor]],
) -> str:
    payload = {
        "schema_version": 1,
        "endpoint_kind": page_format,
        "page_size": page_size,
        "buffers": {
            name: [
                {
                    "dtype": str(buffer.dtype),
                    "page_item_shape": tuple(buffer.shape[1:]),
                    "stride": tuple(buffer.stride()),
                    "storage_offset": buffer.storage_offset(),
                }
                for buffer in tensors
            ]
            for name, tensors in buffers.items()
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _standard_namespace(
    ctx: AttentionStoreInitContext,
    endpoint_name: str,
    shard_num: int,
) -> str:
    return f"{ctx.namespace_base}:{shard_num}:pp{ctx.pp_rank}:{endpoint_name}"


class _BufferPageIO:
    """Transport page requests over SDK-owned typed buffer bindings."""

    def __init__(self, sdk, buffers, layer_ids: Sequence[int]):
        self._sdk = sdk
        self._buffers = buffers
        self._layer_ids = tuple(map(int, layer_ids))

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


def _create_mha_page_endpoint(
    ctx: AttentionStoreInitContext,
    *,
    pool: Any,
    namespace: str,
    gpu_count: int,
) -> AttentionStoreEndpointBinding:
    """Bind an MHA K/V endpoint to a full-attention pool."""
    from attentionstore_sdk.sdk import (
        AttentionStoreSDK,
        BufferBindings,
        MHAConfig,
    )

    layer_ids = tuple(range(int(pool.layer_num)))
    if not layer_ids:
        raise ValueError("MHA page endpoint requires at least one attention layer")
    key_buffers = [_key_buffer(pool, layer_id) for layer_id in layer_ids]
    value_buffers = [_value_buffer(pool, layer_id) for layer_id in layer_ids]
    buffer_pools = [pool] * len(layer_ids)
    k_bytes = _uniform_page_bytes(key_buffers, buffer_pools, ctx.page_size, "MHA K")
    v_bytes = _uniform_page_bytes(value_buffers, buffer_pools, ctx.page_size, "MHA V")
    bindings = BufferBindings(
        k=[buffer.data_ptr() for buffer in key_buffers],
        v=[buffer.data_ptr() for buffer in value_buffers],
    )
    layout_id = _page_layout_id(
        "MHA",
        ctx.page_size,
        {"k": key_buffers, "v": value_buffers},
    )
    sdk = AttentionStoreSDK(
        _sdk_pod_config(
            ctx,
            namespace,
            model_version=f"{ctx.model_version}-layout-{layout_id[:16]}",
            gpu_count=gpu_count,
        ),
        MHAConfig(
            layer_num=len(layer_ids),
            shard_id=ctx.tp_rank,
            shard_num=ctx.tp_size,
            block_token_size=ctx.page_size,
            k_bytes=k_bytes,
            v_bytes=v_bytes,
        ),
        enable_as_kv_rw=True,
    )
    return AttentionStoreEndpointBinding(
        name="mha",
        role=AttentionStoreEndpointRole.PAGE,
        store=_BufferPageIO(sdk, bindings, layer_ids),
        namespace=namespace,
        layout_id=layout_id,
        bytes_per_block=(k_bytes + v_bytes) * len(layer_ids),
        payload_fields=("k", "v"),
    )


def _create_mla_page_endpoint(
    ctx: AttentionStoreInitContext,
    *,
    pool: Any,
    namespace: str,
    gpu_count: int,
) -> AttentionStoreEndpointBinding:
    """Bind an MLA latent-KV endpoint to a full-attention pool."""
    from attentionstore_sdk.sdk import (
        AttentionStoreSDK,
        BufferBindings,
        MLAConfig,
    )

    layer_ids = tuple(range(int(pool.layer_num)))
    if not layer_ids:
        raise ValueError("MLA page endpoint requires at least one attention layer")
    kv_buffers = [_key_buffer(pool, layer_id) for layer_id in layer_ids]
    kv_bytes = _uniform_page_bytes(
        kv_buffers, [pool] * len(layer_ids), ctx.page_size, "MLA latent KV"
    )
    bindings = BufferBindings(k=[buffer.data_ptr() for buffer in kv_buffers])
    layout_id = _page_layout_id(
        "MLA",
        ctx.page_size,
        {"kv_latent": kv_buffers},
    )
    sdk = AttentionStoreSDK(
        _sdk_pod_config(
            ctx,
            namespace,
            model_version=f"{ctx.model_version}-layout-{layout_id[:16]}",
            gpu_count=gpu_count,
        ),
        MLAConfig(
            layer_num=len(layer_ids),
            shard_id=ctx.tp_rank,
            shard_num=ctx.tp_size,
            block_token_size=ctx.page_size,
            kv_bytes=kv_bytes,
        ),
        enable_as_kv_rw=True,
    )
    return AttentionStoreEndpointBinding(
        name="mla",
        role=AttentionStoreEndpointRole.PAGE,
        store=_BufferPageIO(sdk, bindings, layer_ids),
        namespace=namespace,
        layout_id=layout_id,
        bytes_per_block=kv_bytes * len(layer_ids),
        payload_fields=("kv_latent",),
    )


def create_mha_binding(ctx: AttentionStoreInitContext) -> AttentionStoreBinding:
    """Create an MHA binding."""
    from attentionstore_sdk.sdk import (
        AttentionStoreSDK,
        BufferBindings,
        MHAConfig,
    )

    target_ids = tuple(ctx.target_layer_ids)
    layer_ids = tuple(range(len(target_ids) + len(ctx.draft_layer_ids)))
    key_buffers = [_key_buffer(ctx.kv_cache, layer_id) for layer_id in target_ids]
    value_buffers = [_value_buffer(ctx.kv_cache, layer_id) for layer_id in target_ids]
    buffer_pools = [ctx.kv_cache] * len(target_ids)
    if ctx.draft_kv_cache is not None:
        key_buffers.extend(
            _key_buffer(ctx.draft_kv_cache, layer_id)
            for layer_id in ctx.draft_layer_ids
        )
        value_buffers.extend(
            _value_buffer(ctx.draft_kv_cache, layer_id)
            for layer_id in ctx.draft_layer_ids
        )
        buffer_pools.extend([ctx.draft_kv_cache] * len(ctx.draft_layer_ids))
    k_bytes = _uniform_page_bytes(key_buffers, buffer_pools, ctx.page_size, "MHA K")
    v_bytes = _uniform_page_bytes(value_buffers, buffer_pools, ctx.page_size, "MHA V")
    namespace = _standard_namespace(ctx, "mha", ctx.tp_size)
    sdk = AttentionStoreSDK(
        _sdk_pod_config(ctx, namespace),
        MHAConfig(
            layer_num=len(layer_ids),
            shard_id=ctx.tp_rank,
            shard_num=ctx.tp_size,
            block_token_size=ctx.page_size,
            k_bytes=k_bytes,
            v_bytes=v_bytes,
        ),
        enable_as_kv_rw=True,
    )
    buffers = BufferBindings(
        k=[buffer.data_ptr() for buffer in key_buffers],
        v=[buffer.data_ptr() for buffer in value_buffers],
    )
    page_store = _BufferPageIO(sdk, buffers, layer_ids)
    endpoint = AttentionStoreEndpointBinding(
        name="mha",
        role=AttentionStoreEndpointRole.PAGE,
        store=page_store,
        namespace=namespace,
        layout_id="mha",
        bytes_per_block=(k_bytes + v_bytes) * len(layer_ids),
        payload_fields=("k", "v"),
    )
    return AttentionStoreBinding(
        plan_name="MHA",
        cache_store=page_store,
        endpoints=(endpoint,),
    )


def create_mla_binding(ctx: AttentionStoreInitContext) -> AttentionStoreBinding:
    """Create an MLA binding."""
    from attentionstore_sdk.sdk import (
        AttentionStoreSDK,
        BufferBindings,
        MLAConfig,
    )

    target_ids = tuple(ctx.target_layer_ids)
    layer_ids = tuple(range(len(target_ids) + len(ctx.draft_layer_ids)))
    kv_buffers = [_key_buffer(ctx.kv_cache, layer_id) for layer_id in target_ids]
    buffer_pools = [ctx.kv_cache] * len(target_ids)
    if ctx.draft_kv_cache is not None:
        draft_buffers = [
            _key_buffer(ctx.draft_kv_cache, layer_id)
            for layer_id in ctx.draft_layer_ids
        ]
        for phys_id, buffer in zip(ctx.draft_layer_ids, draft_buffers):
            # An MLA binding must never silently reinterpret an MHA draft layer.
            if (
                _value_buffer(ctx.draft_kv_cache, phys_id).data_ptr()
                != buffer.data_ptr()
            ):
                raise ValueError("MLA AttentionStore requires MLA-organized draft KV")
        kv_buffers.extend(draft_buffers)
        buffer_pools.extend([ctx.draft_kv_cache] * len(draft_buffers))
    kv_bytes = _uniform_page_bytes(kv_buffers, buffer_pools, ctx.page_size, "MLA KV")
    namespace = f"{_standard_namespace(ctx, 'mla', 1)}:tp{ctx.tp_rank}"
    sdk = AttentionStoreSDK(
        _sdk_pod_config(ctx, namespace),
        MLAConfig(
            layer_num=len(layer_ids),
            shard_id=0,
            shard_num=1,
            block_token_size=ctx.page_size,
            kv_bytes=kv_bytes,
        ),
        enable_as_kv_rw=True,
    )
    buffers = BufferBindings(k=[buffer.data_ptr() for buffer in kv_buffers])
    page_store = _BufferPageIO(sdk, buffers, layer_ids)
    endpoint = AttentionStoreEndpointBinding(
        name="mla",
        role=AttentionStoreEndpointRole.PAGE,
        store=page_store,
        namespace=namespace,
        layout_id="mla",
        bytes_per_block=kv_bytes * len(layer_ids),
        payload_fields=("kv_latent",),
    )
    return AttentionStoreBinding(
        plan_name="MLA",
        cache_store=page_store,
        endpoints=(endpoint,),
    )


def _dsa_indexer_layer_ids(
    kv_cache, target_layer_ids: Sequence[int]
) -> tuple[int, ...]:
    mapping = getattr(kv_cache, "indexer_layer_mapping", None)
    if mapping is None:
        return tuple(target_layer_ids)
    target_set = set(target_layer_ids)
    layer_ids = tuple(
        layer_id
        for layer_id, _ in sorted(mapping.items(), key=lambda item: item[1])
        if layer_id in target_set
    )
    if not layer_ids:
        raise ValueError("DSA compact indexer mapping contains no physical layers")
    return layer_ids


def create_dsa_binding(ctx: AttentionStoreInitContext) -> AttentionStoreBinding:
    """Create a DSA binding and preserve compact physical indexer ordering."""
    from attentionstore_sdk.sdk import (
        AttentionStoreSDK,
        BufferBindings,
        DSAConfig,
    )

    target_ids = tuple(ctx.target_layer_ids)
    layer_ids = tuple(range(len(target_ids) + len(ctx.draft_layer_ids)))
    latent_buffers = [_key_buffer(ctx.kv_cache, layer_id) for layer_id in target_ids]
    latent_pools = [ctx.kv_cache] * len(target_ids)
    physical_indexer_ids = _dsa_indexer_layer_ids(ctx.kv_cache, target_ids)
    indexer_buffers = [
        ctx.kv_cache.get_index_k_with_scale_buffer(layer_id)
        for layer_id in physical_indexer_ids
    ]
    indexer_pools = [ctx.kv_cache] * len(indexer_buffers)
    target_logical_ids = {layer_id: i for i, layer_id in enumerate(target_ids)}
    sdk_indexer_ids = [
        target_logical_ids[layer_id] for layer_id in physical_indexer_ids
    ]
    if ctx.draft_kv_cache is not None:
        draft_count = len(ctx.draft_layer_ids)
        latent_buffers.extend(
            _key_buffer(ctx.draft_kv_cache, layer_id)
            for layer_id in ctx.draft_layer_ids
        )
        latent_pools.extend([ctx.draft_kv_cache] * draft_count)
        for draft_offset, layer_id in enumerate(ctx.draft_layer_ids):
            if hasattr(ctx.draft_kv_cache, "get_index_k_with_scale_buffer"):
                buffer = ctx.draft_kv_cache.get_index_k_with_scale_buffer(layer_id)
            else:
                buffer = _value_buffer(ctx.draft_kv_cache, layer_id)
            indexer_buffers.append(buffer)
            indexer_pools.append(ctx.draft_kv_cache)
            sdk_indexer_ids.append(len(target_ids) + draft_offset)
    latent_bytes = _uniform_page_bytes(
        latent_buffers, latent_pools, ctx.page_size, "DSA latent KV"
    )
    indexer_bytes = _uniform_page_bytes(
        indexer_buffers, indexer_pools, ctx.page_size, "DSA indexer"
    )
    namespace = f"{_standard_namespace(ctx, 'dsa', 1)}:tp{ctx.tp_rank}"
    sdk = AttentionStoreSDK(
        _sdk_pod_config(ctx, namespace),
        DSAConfig(
            layer_num=len(layer_ids),
            shard_id=0,
            shard_num=1,
            block_token_size=ctx.page_size,
            latent_kv_bytes=latent_bytes,
            indexer_bytes=indexer_bytes,
            indexer_layer_ids=sdk_indexer_ids,
        ),
        enable_as_kv_rw=True,
    )
    buffers = BufferBindings(
        k=[buffer.data_ptr() for buffer in latent_buffers],
        v=[buffer.data_ptr() for buffer in indexer_buffers],
    )
    page_store = _BufferPageIO(sdk, buffers, layer_ids)
    endpoint = AttentionStoreEndpointBinding(
        name="dsa",
        role=AttentionStoreEndpointRole.PAGE,
        store=page_store,
        namespace=namespace,
        layout_id="dsa",
        bytes_per_block=(latent_bytes + indexer_bytes) * len(layer_ids),
        payload_fields=("kv_latent", "indexer"),
    )
    return AttentionStoreBinding(
        plan_name="DSA",
        cache_store=page_store,
        endpoints=(endpoint,),
    )


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class StateTensorSpec:
    """A tensor slice addressed by a logical slot group."""

    name: str
    tensor: torch.Tensor
    group: str = "state"
    slot_dim: int = 0
    items_per_slot: int = 1
    base_index: int = 0
    slot_count_limit: Optional[int] = None

    def __post_init__(self):
        """Normalize and validate the logical tensor slice."""
        if not self.name:
            raise ValueError("state tensor name must not be empty")
        if not self.group:
            raise ValueError(f"state tensor {self.name}: group must not be empty")
        if self.tensor.ndim == 0:
            raise ValueError(
                f"state tensor {self.name}: scalar tensors are unsupported"
            )
        slot_dim = self.slot_dim % self.tensor.ndim
        object.__setattr__(self, "slot_dim", slot_dim)
        if self.items_per_slot <= 0:
            raise ValueError(
                f"state tensor {self.name}: items_per_slot must be positive"
            )
        if self.base_index < 0 or self.base_index >= self.tensor.shape[slot_dim]:
            raise ValueError(
                f"state tensor {self.name}: invalid base_index={self.base_index}"
            )
        remaining = self.tensor.shape[slot_dim] - self.base_index
        if self.slot_count_limit is not None:
            if (
                self.slot_count_limit <= 0
                or self.slot_count_limit * self.items_per_slot > remaining
            ):
                raise ValueError(
                    f"state tensor {self.name}: invalid "
                    f"slot_count_limit={self.slot_count_limit}"
                )
        elif remaining % self.items_per_slot != 0:
            raise ValueError(
                f"state tensor {self.name}: remaining slot dimension {remaining} "
                f"is not divisible by items_per_slot={self.items_per_slot}"
            )

    @property
    def slot_count(self) -> int:
        """Return the number of addressable logical slots."""
        if self.slot_count_limit is not None:
            return self.slot_count_limit
        return (
            self.tensor.shape[self.slot_dim] - self.base_index
        ) // self.items_per_slot

    @property
    def item_shape(self) -> tuple[int, ...]:
        """Return the tensor shape occupied by one logical slot."""
        shape = list(self.tensor.shape)
        shape[self.slot_dim] = self.items_per_slot
        return tuple(shape)

    @property
    def bytes_per_slot(self) -> int:
        """Return the byte size of one logical slot."""
        count = 1
        for dim in self.item_shape:
            count *= int(dim)
        return count * self.tensor.element_size()

    def select(self, slot: int) -> torch.Tensor:
        """Return the tensor view for one logical slot."""
        if slot < 0 or slot >= self.slot_count:
            raise IndexError(
                f"state tensor {self.name}: slot {slot} outside [0, {self.slot_count})"
            )
        start = self.base_index + slot * self.items_per_slot
        return self.tensor.narrow(self.slot_dim, start, self.items_per_slot)


@dataclass(frozen=True)
class _LayoutEntry:
    spec: StateTensorSpec
    offset: int


class PackedStateLayout:
    """Deterministic byte layout for one self-contained cache block."""

    ALIGNMENT = 16

    def __init__(
        self,
        specs: Sequence[StateTensorSpec],
        *,
        layout_name: str,
        pack_validator: Optional[Callable[[Sequence[Mapping[str, int]]], None]] = None,
        unpack_hook: Optional[Callable[[Sequence[Mapping[str, int]]], None]] = None,
    ):
        """Build a deterministic packed layout from tensor specifications."""
        if not specs:
            raise ValueError("packed state layout requires at least one tensor")

        entries = []
        offset = 0
        group_slot_counts: dict[str, int] = {}
        device = specs[0].tensor.device
        for spec in specs:
            if spec.tensor.device != device:
                raise ValueError(
                    "all state tensors must reside on one device: "
                    f"expected {device}, got {spec.tensor.device} for {spec.name}"
                )
            previous = group_slot_counts.setdefault(spec.group, spec.slot_count)
            if previous != spec.slot_count:
                raise ValueError(
                    f"state group {spec.group!r} has inconsistent slot counts: "
                    f"{previous} and {spec.slot_count} ({spec.name})"
                )
            offset = _align_up(offset, max(self.ALIGNMENT, spec.tensor.element_size()))
            entries.append(_LayoutEntry(spec=spec, offset=offset))
            offset += spec.bytes_per_slot

        if not layout_name:
            raise ValueError("layout name must not be empty")
        self.layout_name = layout_name.upper()
        self.entries = tuple(entries)
        self.device = device
        self.group_slot_counts = group_slot_counts
        self.bytes_per_block = _align_up(offset, self.ALIGNMENT)
        self.layout_id = self._compute_layout_id()
        self._pack_validator = pack_validator
        self._unpack_hook = unpack_hook

    @property
    def groups(self) -> tuple[str, ...]:
        """Return the logical slot groups used by this layout."""
        return tuple(self.group_slot_counts)

    def _compute_layout_id(self) -> str:
        payload = [
            {
                "name": entry.spec.name,
                "group": entry.spec.group,
                "dtype": str(entry.spec.tensor.dtype),
                "shape": entry.spec.item_shape,
                "slot_dim": entry.spec.slot_dim,
                "items_per_slot": entry.spec.items_per_slot,
                "offset": entry.offset,
                "bytes": entry.spec.bytes_per_slot,
            }
            for entry in self.entries
        ]
        encoded = json.dumps(
            {
                "schema_version": 2,
                "layout_name": self.layout_name,
                "segments": payload,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def allocate_staging(self, capacity: int = 1) -> torch.Tensor:
        """Allocate contiguous byte staging for packed blocks."""
        if capacity <= 0:
            raise ValueError(f"staging capacity must be positive, got {capacity}")
        return torch.empty(
            (capacity, self.bytes_per_block),
            dtype=torch.uint8,
            device=self.device,
        )

    def _normalize_slots(self, slots: Mapping[str, int] | int) -> Mapping[str, int]:
        if isinstance(slots, torch.Tensor):
            if slots.numel() != 1:
                raise ValueError(
                    f"state slot tensor must contain one index, got {tuple(slots.shape)}"
                )
            slots = int(slots.item())
        if isinstance(slots, Integral):
            if len(self.groups) != 1:
                raise ValueError(
                    f"layout groups {self.groups} require an explicit slot mapping"
                )
            slots = {self.groups[0]: int(slots)}
        missing = set(self.groups) - set(slots)
        extra = set(slots) - set(self.groups)
        if missing or extra:
            raise ValueError(
                f"state slot groups mismatch: missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        return slots

    def _validate_staging(self, staging: torch.Tensor, count: int) -> None:
        if (
            staging.dtype != torch.uint8
            or staging.ndim != 2
            or staging.shape[0] < count
            or staging.shape[1] != self.bytes_per_block
            or staging.device != self.device
            or not staging.is_contiguous()
        ):
            raise ValueError(
                "invalid packed-state staging tensor: expected contiguous "
                f"uint8 [>= {count}, {self.bytes_per_block}] on {self.device}, "
                f"got dtype={staging.dtype}, shape={tuple(staging.shape)}, "
                f"device={staging.device}, contiguous={staging.is_contiguous()}"
            )

    def pack(
        self,
        slots: Sequence[Mapping[str, int] | int],
        staging: torch.Tensor,
    ) -> None:
        """Pack logical state slots into staging rows."""
        self._validate_staging(staging, len(slots))
        normalized_slots = [self._normalize_slots(raw_slots) for raw_slots in slots]
        if self._pack_validator is not None:
            self._pack_validator(normalized_slots)
        for row, slot_map in enumerate(normalized_slots):
            staging[row].zero_()
            for entry in self.entries:
                source = entry.spec.select(int(slot_map[entry.spec.group]))
                source_bytes = source.contiguous().view(torch.uint8).reshape(-1)
                end = entry.offset + entry.spec.bytes_per_slot
                staging[row, entry.offset : end].copy_(source_bytes)

    def unpack(
        self,
        slots: Sequence[Mapping[str, int] | int],
        staging: torch.Tensor,
    ) -> None:
        """Restore logical state slots from staging rows."""
        self._validate_staging(staging, len(slots))
        normalized_slots = [self._normalize_slots(raw_slots) for raw_slots in slots]
        for row, slot_map in enumerate(normalized_slots):
            for entry in self.entries:
                target = entry.spec.select(int(slot_map[entry.spec.group]))
                end = entry.offset + entry.spec.bytes_per_slot
                source = staging[row, entry.offset : end]
                typed_source = source.view(entry.spec.tensor.dtype).reshape(
                    entry.spec.item_shape
                )
                target.copy_(typed_source)
        if self._unpack_hook is not None:
            self._unpack_hook(normalized_slots)


class KDACheckpointStore:
    """Bind fixed KDA state buffers to the SDK pointer-array transport."""

    def __init__(
        self,
        sdk,
        binding,
        slot_group: str,
    ):
        self._sdk = sdk
        self.layout = binding
        self._state_ptrs = binding.state_ptrs
        self._layer_ids = tuple(range(binding.layer_num))
        self._device = binding.device
        self._slot_group = slot_group
        self._pack_validator = binding.pack_validator
        self._unpack_hook = binding.unpack_hook
        self._lock = threading.Lock()

    def _synchronize_device(self) -> None:
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        elif self._device.type == "xpu":
            torch.xpu.synchronize(self._device)

    def _slot(self, slot: Mapping[str, int] | int) -> tuple[dict[str, int], int]:
        slot_map = slot if isinstance(slot, Mapping) else {self._slot_group: int(slot)}
        if set(slot_map) != {self._slot_group}:
            raise ValueError(
                f"KDA checkpoint requires slot group {self._slot_group!r}, "
                f"got {sorted(slot_map)}"
            )
        return dict(slot_map), int(slot_map[self._slot_group])

    def match(self, block_hash_ids: Sequence[int], timeout: float = 1.0) -> int:
        """Return the matched KDA checkpoint prefix length."""
        from attentionstore_sdk.sdk import MatchRequest

        return self._sdk.match(MatchRequest(block_hash_ids, timeout=timeout))

    def write(
        self,
        block_hash_id: int,
        slot: Mapping[str, int] | int,
        timeout: float = 1.0,
        params: Optional[dict] = None,
    ) -> int:
        """Write one KDA checkpoint from a logical state slot."""
        from attentionstore_sdk.sdk import BlockBatch, BufferBindings, WriteRequest

        slot_map, slot_id = self._slot(slot)
        with self._lock:
            # Model updates can run on a model-owned stream. The SDK consumes
            # the registered layer-buffer pointers directly, so fence those
            # updates before queuing the native D2H transfer.
            self._synchronize_device()
            if self._pack_validator is not None:
                self._pack_validator([slot_map])
            return self._sdk.write(
                WriteRequest(
                    blocks=BlockBatch(
                        [block_hash_id],
                        [slot_id],
                        layer_ids=self._layer_ids,
                    ),
                    buffers=BufferBindings(state=self._state_ptrs),
                    timeout=timeout,
                    params=params or {},
                )
            )

    def read(
        self,
        block_hash_id: int,
        slot: Mapping[str, int] | int,
        timeout: float = 1.0,
        remote_addrs=None,
    ) -> int:
        """Read one KDA checkpoint into a logical state slot."""
        from attentionstore_sdk.sdk import BlockBatch, BufferBindings, ReadRequest

        slot_map, slot_id = self._slot(slot)
        with self._lock:
            count = self._sdk.read(
                ReadRequest(
                    blocks=BlockBatch(
                        [block_hash_id],
                        [slot_id],
                        layer_ids=self._layer_ids,
                    ),
                    buffers=BufferBindings(state=self._state_ptrs),
                    timeout=timeout,
                    remote_addrs=remote_addrs or (),
                )
            )
            if count and self._unpack_hook is not None:
                try:
                    self._unpack_hook([slot_map])
                finally:
                    # Fence ReplaySSM cursor resets before another model stream
                    # consumes the state restored by native H2D copies.
                    self._synchronize_device()
            return count

    def stop(self) -> None:
        """Stop the underlying checkpoint SDK client."""
        self._sdk.stop()


class CheckpointedPageStore:
    """Page stream plus an independently stored endpoint checkpoint."""

    def __init__(self, page_store, checkpoint_store):
        """Combine a page store with its terminal checkpoint store."""
        self.page_store = page_store
        self.checkpoint_store = checkpoint_store
        self.layout = checkpoint_store.layout

    def match(self, block_hash_ids: Sequence[int], timeout: float = 1.0) -> int:
        """Return the longest page prefix with a terminal checkpoint."""
        page_count = self.page_store.match(block_hash_ids, timeout=timeout)
        for count in range(page_count, 0, -1):
            if (
                self.checkpoint_store.match(
                    [block_hash_ids[count - 1]], timeout=timeout
                )
                == 1
            ):
                return count
        return 0

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
    ) -> int:
        """Write pages and their terminal checkpoint as one unit."""
        if not block_hash_ids:
            return 0
        self.page_store.write(block_hash_ids, page_ids, timeout=timeout, params=params)
        checkpoint_result = self.checkpoint_store.write(
            int(block_hash_ids[-1]),
            checkpoint_slot,
            timeout=timeout,
            params=params,
        )
        return len(block_hash_ids) if checkpoint_result == 1 else -1

    def stop(self) -> None:
        """Stop both the page and checkpoint stores."""
        try:
            self.page_store.stop()
        finally:
            self.checkpoint_store.stop()


@dataclass(frozen=True)
class _KDACheckpointBinding:
    layer_num: int
    temporal_bytes: tuple[int, ...]
    conv_bytes: tuple[int, ...]
    state_ptrs: tuple[int, ...]
    tensors: tuple[torch.Tensor, ...]
    payload_fields: tuple[str, ...]
    device: torch.device
    pack_validator: Optional[Callable] = None
    unpack_hook: Optional[Callable] = None

    @property
    def bytes_per_layer(self) -> int:
        """Return the checkpoint byte size for one KDA layer."""
        return sum(self.temporal_bytes) + sum(self.conv_bytes)

    @property
    def bytes_per_block(self) -> int:
        """Return the checkpoint byte size across all KDA layers."""
        return self.layer_num * self.bytes_per_layer


def _kda_checkpoint_tensors(checkpoint_pool):
    temporal = [
        ("temporal.qdata", checkpoint_pool.temporal.qdata),
        ("temporal.scale", checkpoint_pool.temporal.scale),
    ]
    conv = [
        (f"conv.{index}", tensor) for index, tensor in enumerate(checkpoint_pool.conv)
    ]
    return temporal, conv


def _make_kda_checkpoint_binding(
    temporal_tensors,
    conv_tensors,
    *,
    pack_validator=None,
    unpack_hook=None,
) -> _KDACheckpointBinding:
    """Bind per-layer KDA components as component-major pointer arrays."""
    temporal_tensors = tuple(temporal_tensors)
    conv_tensors = tuple(conv_tensors)
    if not temporal_tensors:
        raise ValueError("KDA AttentionStore requires temporal state buffers")
    if not conv_tensors:
        raise ValueError("KDA AttentionStore requires conv state buffers")
    named_tensors = temporal_tensors + conv_tensors

    names = tuple(name for name, _ in named_tensors)
    tensors = tuple(tensor for _, tensor in named_tensors)
    if len(set(names)) != len(names):
        raise ValueError(f"KDA state buffer names must be unique, got {names}")

    first = tensors[0]
    if first.ndim < 2:
        raise ValueError("KDA state buffers must have [layer, slot, ...] layout")
    layer_num, slot_num = first.shape[:2]
    if layer_num <= 0 or slot_num <= 0:
        raise ValueError("KDA state buffers require positive layer and slot counts")

    component_bytes = []
    state_ptrs = []
    for name, tensor in named_tensors:
        if tensor.ndim < 2 or tuple(tensor.shape[:2]) != (layer_num, slot_num):
            raise ValueError(
                f"KDA state buffer {name!r} must start with "
                f"[{layer_num}, {slot_num}], got {tuple(tensor.shape)}"
            )
        if tensor.device != first.device:
            raise ValueError(
                f"KDA state buffer {name!r} is on {tensor.device}, "
                f"expected {first.device}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"KDA state buffer {name!r} must be contiguous")

        bytes_per_slot = tensor[0, 0].numel() * tensor.element_size()
        component_bytes.append(bytes_per_slot)
        for layer in range(layer_num):
            layer_buffer = tensor[layer]
            if not layer_buffer.is_contiguous():
                raise ValueError(
                    f"KDA state buffer {name!r} layer {layer} must be contiguous"
                )
            state_ptrs.append(layer_buffer.data_ptr())

    temporal_count = len(temporal_tensors)
    return _KDACheckpointBinding(
        layer_num=layer_num,
        temporal_bytes=tuple(component_bytes[:temporal_count]),
        conv_bytes=tuple(component_bytes[temporal_count:]),
        state_ptrs=tuple(state_ptrs),
        tensors=tensors,
        payload_fields=names,
        device=first.device,
        pack_validator=pack_validator,
        unpack_hook=unpack_hook,
    )


def make_kda_checkpoint_layout(checkpoint_pool) -> _KDACheckpointBinding:
    """Bind the fixed recurrent-state buffers of an int8 KDA model."""
    temporal, conv = _kda_checkpoint_tensors(checkpoint_pool)
    return _make_kda_checkpoint_binding(temporal, conv)


def _kda_active_state_parts(mamba_pool):
    """Build full-precision KDA state with ReplaySSM boundary checks."""
    cache = mamba_pool.mamba_cache
    temporal_tensors = [("temporal", cache.temporal)]
    conv_tensors = list(
        (f"conv.{index}", tensor) for index, tensor in enumerate(cache.conv)
    )
    write_pos = getattr(mamba_pool, "replayssm_write_pos", None)

    def checkpoint_indices(slots: Sequence[Mapping[str, int]]) -> torch.Tensor:
        return torch.tensor(
            [int(slot["checkpoint"]) for slot in slots],
            dtype=torch.long,
            device=write_pos.device,
        )

    def require_flushed_replayssm(slots: Sequence[Mapping[str, int]]) -> None:
        if write_pos is None or not slots:
            return
        indices = checkpoint_indices(slots)
        pending = write_pos[indices]
        if torch.any(pending != 0).item():
            raise ValueError(
                "KDA AttentionStore checkpoints require fully flushed ReplaySSM "
                f"slots; write_pos={pending.tolist()} for slots={indices.tolist()}"
            )

    def reset_replayssm_cursor(slots: Sequence[Mapping[str, int]]) -> None:
        if write_pos is None or not slots:
            return
        write_pos[checkpoint_indices(slots)] = 0

    return (
        temporal_tensors,
        conv_tensors,
        require_flushed_replayssm,
        reset_replayssm_cursor,
    )


def make_kda_active_state_layout(mamba_pool) -> _KDACheckpointBinding:
    """Bind the fixed recurrent-state buffers of a full-precision KDA model."""
    temporal_tensors, conv_tensors, validator, hook = _kda_active_state_parts(
        mamba_pool
    )
    return _make_kda_checkpoint_binding(
        temporal_tensors,
        conv_tensors,
        pack_validator=validator,
        unpack_hook=hook,
    )


def _split_kda_namespace_chunks(total_chunks: int) -> tuple[int, int]:
    """Partition Agent chunks between KDA page and checkpoint namespaces."""
    if total_chunks < 2:
        raise ValueError(
            "KDA page/state namespaces require at least two Agent memory chunks"
        )
    raw_checkpoint_chunks = os.getenv("AS_KDA_CHECKPOINT_GPU_COUNT", "1")
    try:
        checkpoint_chunks = int(raw_checkpoint_chunks)
    except ValueError:
        raise ValueError(
            "AS_KDA_CHECKPOINT_GPU_COUNT must be an integer, "
            f"got {raw_checkpoint_chunks!r}"
        ) from None
    if not 1 <= checkpoint_chunks < total_chunks:
        raise ValueError(
            "AS_KDA_CHECKPOINT_GPU_COUNT must be in "
            f"[1, {total_chunks - 1}], got {checkpoint_chunks}"
        )
    return total_chunks - checkpoint_chunks, checkpoint_chunks


def create_kda_binding(ctx: AttentionStoreInitContext) -> AttentionStoreBinding:
    """Create independent KDA page and recurrent-checkpoint stores."""
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

    if not isinstance(ctx.kv_cache, HybridLinearKVPool):
        raise ValueError("KDA AttentionStore requires HybridLinearKVPool")
    checkpoint_pool = getattr(ctx.req_to_token_pool, "mamba_ckpt_pool", None)
    checkpoint_binding = (
        make_kda_checkpoint_layout(checkpoint_pool)
        if checkpoint_pool is not None
        else make_kda_active_state_layout(ctx.kv_cache.mamba_pool)
    )
    page_namespace = f"{ctx.namespace_base}:1"
    checkpoint_namespace = f"{ctx.namespace_base}:kda:1"
    page_chunks, checkpoint_chunks = _split_kda_namespace_chunks(ctx.gpu_count)
    logger.info(
        "KDA AttentionStore Agent chunk split: total=%d page=%d checkpoint=%d",
        ctx.gpu_count,
        page_chunks,
        checkpoint_chunks,
    )
    page_factory = (
        _create_mla_page_endpoint if ctx.kv_cache.use_mla else _create_mha_page_endpoint
    )
    page_endpoint = page_factory(
        ctx,
        pool=ctx.kv_cache.full_kv_pool,
        namespace=page_namespace,
        gpu_count=page_chunks,
    )
    try:
        from attentionstore_sdk.sdk import AttentionStoreSDK, KDAConfig, PodConfig

        checkpoint_sdk = AttentionStoreSDK(
            PodConfig(
                namespace=checkpoint_namespace,
                pod_name=ctx.pod_name,
                model_version=ctx.model_version,
                device_id=ctx.device_id,
                dp_id=ctx.dp_rank,
                gpu_count=checkpoint_chunks,
            ),
            KDAConfig(
                layer_num=checkpoint_binding.layer_num,
                shard_id=ctx.tp_rank,
                shard_num=ctx.tp_size,
                block_token_size=ctx.page_size,
                temporal_bytes=checkpoint_binding.temporal_bytes,
                conv_bytes=checkpoint_binding.conv_bytes,
            ),
            enable_as_kv_rw=True,
        )
    except Exception:
        page_endpoint.store.stop()
        raise
    checkpoint_store = KDACheckpointStore(
        checkpoint_sdk,
        checkpoint_binding,
        slot_group="checkpoint",
    )
    cache_store = CheckpointedPageStore(page_endpoint.store, checkpoint_store)
    try:
        checkpoint_endpoint = AttentionStoreEndpointBinding(
            name="checkpoint",
            role=AttentionStoreEndpointRole.CHECKPOINT,
            store=checkpoint_store,
            namespace=checkpoint_namespace,
            layout_id="kda-state",
            bytes_per_block=checkpoint_binding.bytes_per_block,
            payload_fields=checkpoint_binding.payload_fields,
        )
        return AttentionStoreBinding(
            plan_name="KDA",
            cache_store=cache_store,
            endpoints=(page_endpoint, checkpoint_endpoint),
        )
    except Exception:
        cache_store.stop()
        raise


def make_deepseek_v4_checkpoint_layout(
    pool, *, include_compressed_pages: bool = True
) -> PackedStateLayout:
    """Describe DSV4 KV, SWA, C4, and C128 checkpoint groups."""

    specs: list[StateTensorSpec] = []

    def add_buffers(
        prefix: str,
        buffers: Sequence[torch.Tensor],
        *,
        group: str,
        items_per_slot: int = 1,
        base_index: int = 0,
        slot_count_limit: Optional[int] = None,
    ) -> None:
        specs.extend(
            StateTensorSpec(
                f"{prefix}.{index}",
                tensor,
                group=group,
                slot_dim=0,
                items_per_slot=items_per_slot,
                base_index=base_index,
                slot_count_limit=slot_count_limit,
            )
            for index, tensor in enumerate(buffers)
        )

    if getattr(pool, "_unified_kv", False):
        unified = pool.unified_kv_pool
        stage_ratios = pool.compression_ratios[pool._stage_start : pool._stage_end]
        if include_compressed_pages:
            for local_layer_id, ratio in enumerate(stage_ratios):
                if ratio not in (4, 128):
                    continue
                add_buffers(
                    f"unified.compressed.{local_layer_id}",
                    [unified.kv_buffer[local_layer_id]],
                    group="kv_page",
                    items_per_slot=pool.page_size // ratio,
                    base_index=unified.swa_pages,
                )
            add_buffers(
                "c4.indexer",
                pool.c4_indexer_kv_pool.index_k_with_scale_buffer,
                group="kv_page",
            )
        add_buffers(
            "unified.swa",
            unified.kv_buffer,
            group="swa_request",
            items_per_slot=unified.swa_ring_size,
            slot_count_limit=unified.num_slots,
        )
    else:
        if include_compressed_pages:
            add_buffers("c4.kv", pool.c4_kv_pool.kv_buffer, group="kv_page")
            add_buffers(
                "c4.indexer",
                pool.c4_indexer_kv_pool.index_k_with_scale_buffer,
                group="kv_page",
            )
            add_buffers("c128.kv", pool.c128_kv_pool.kv_buffer, group="kv_page")
        add_buffers("swa", pool.swa_kv_pool.kv_buffer, group="swa_page")

    for layer_id, state_pool in enumerate(pool.compress_state_pools):
        if state_pool is None:
            continue
        group = "c4_state" if state_pool.ratio == 4 else "c128_state"
        state = state_pool.kv_score_buffer.kv_score
        is_online_c128 = state_pool.ratio == 128 and state_pool.ring_size == 1
        if is_online_c128:
            slot_count = pool.online_c128_state_num_req_slots
            add_buffers(
                f"attention_state.{layer_id}.committed",
                [state],
                group=group,
                slot_count_limit=slot_count,
            )
            for bank in range(state_pool.online_mtp_max_draft_tokens):
                add_buffers(
                    f"attention_state.{layer_id}.draft.{bank}",
                    [state],
                    group=group,
                    base_index=(bank + 1) * state_pool.online_mtp_state_slot_offset,
                    slot_count_limit=slot_count,
                )
        else:
            slot_count_limit = None
            if state_pool.ratio == 4 and hasattr(pool, "c4_state_pool_size"):
                slot_count_limit = pool.c4_state_pool_size // state_pool.ring_size
            elif state_pool.ratio == 128 and hasattr(pool, "c128_state_pool_size"):
                slot_count_limit = pool.c128_state_pool_size // state_pool.ring_size
            add_buffers(
                f"attention_state.{layer_id}",
                [state],
                group=group,
                items_per_slot=state_pool.ring_size,
                slot_count_limit=slot_count_limit,
            )

    for layer_id, state_pool in enumerate(pool.indexer_compress_state_pools):
        if state_pool is None:
            continue
        add_buffers(
            f"indexer_state.{layer_id}",
            [state_pool.kv_score_buffer.kv_score],
            group="c4_state",
            items_per_slot=state_pool.ring_size,
            slot_count_limit=(
                pool.c4_state_pool_size // state_pool.ring_size
                if hasattr(pool, "c4_state_pool_size")
                else None
            ),
        )

    pending_seq_lens = getattr(pool, "online_c128_mtp_pending_seq_lens", None)
    if pending_seq_lens is not None:
        specs.append(
            StateTensorSpec(
                "c128.pending_seq_len",
                pending_seq_lens,
                group="c128_state",
                slot_dim=0,
                slot_count_limit=pool.online_c128_state_num_req_slots,
            )
        )

    return PackedStateLayout(
        specs,
        layout_name=(
            "SPA_CHECKPOINT_WITH_PAGES"
            if include_compressed_pages
            else "SPA_CHECKPOINT"
        ),
    )


class SPACheckpointStore:
    """SPA-owned packed resume state carried by the SPA SWA segment."""

    def __init__(
        self,
        sdk,
        layout: PackedStateLayout,
        write_snapshot_capacity: int = 1,
    ):
        """Initialize packed SPA staging and snapshot capacity."""
        if write_snapshot_capacity <= 0:
            raise ValueError("SPA write snapshot capacity must be positive")
        self._sdk = sdk
        self.layout = layout
        # The second row matches the SDK's SPA read offset.
        self.staging = layout.allocate_staging(2)
        self._lock = threading.Lock()
        self._snapshot_condition = threading.Condition()
        self._available_write_snapshots = [
            layout.allocate_staging(1) for _ in range(write_snapshot_capacity)
        ]
        self._stopped = False

    def _synchronize_device(self) -> None:
        if self.layout.device.type == "cuda":
            torch.cuda.synchronize(self.layout.device)
        elif self.layout.device.type == "xpu" and hasattr(torch, "xpu"):
            torch.xpu.synchronize(self.layout.device)

    def match(self, block_hash_ids: Sequence[int], timeout: float = 1.0) -> int:
        """Return the matched SPA checkpoint prefix length."""
        from attentionstore_sdk.sdk import MatchRequest

        return self._sdk.match(MatchRequest(block_hash_ids, timeout=timeout))

    def write(
        self,
        block_hash_id: int,
        slots: Mapping[str, int],
        timeout: float = 1.0,
        params: Optional[dict] = None,
    ) -> int:
        """Snapshot and write one SPA checkpoint."""
        snapshot = self.snapshot(slots)
        try:
            return self.write_snapshot(
                block_hash_id,
                snapshot,
                timeout=timeout,
                params=params,
            )
        finally:
            snapshot.release()

    def snapshot(self, slots: Mapping[str, int]) -> _SPACheckpointSnapshot:
        """Pack state into an exclusively leased checkpoint snapshot."""
        with self._snapshot_condition:
            if self._stopped:
                raise RuntimeError("SPA checkpoint store is already stopped")
            while not self._available_write_snapshots:
                if self._stopped:
                    raise RuntimeError("SPA checkpoint store is already stopped")
                self._snapshot_condition.wait()
            tensor = self._available_write_snapshots.pop()

        def return_snapshot() -> None:
            with self._snapshot_condition:
                self._available_write_snapshots.append(tensor)
                self._snapshot_condition.notify()

        release = _once(return_snapshot)
        try:
            self._synchronize_device()
            self.layout.pack([slots], tensor)
            self._synchronize_device()
            return _SPACheckpointSnapshot(tensor=tensor, release=release)
        except Exception:
            release()
            raise

    def write_snapshot(
        self,
        block_hash_id: int,
        snapshot: _SPACheckpointSnapshot,
        timeout: float = 1.0,
        params: Optional[dict] = None,
    ) -> int:
        """Write a previously packed SPA checkpoint snapshot."""
        from attentionstore_sdk.sdk import (
            BlockBatch,
            BufferBindings,
            WriteRequest,
        )

        self.layout._validate_staging(snapshot.tensor, 1)
        with self._lock:
            self._synchronize_device()
            return self._sdk.write(
                WriteRequest(
                    blocks=BlockBatch(hash_ids=[block_hash_id], data_ids=[0]),
                    buffers=BufferBindings(
                        swa=(snapshot.tensor.data_ptr(),), swa_ids=(0,)
                    ),
                    timeout=timeout,
                    params=params or {},
                    swa_block_step=1,
                )
            )

    def read(
        self,
        block_hash_id: int,
        slots: Mapping[str, int],
        timeout: float = 1.0,
        remote_addrs=None,
    ) -> int:
        """Read and unpack one SPA checkpoint into logical slots."""
        from attentionstore_sdk.sdk import (
            BlockBatch,
            BufferBindings,
            ReadRequest,
        )

        with self._lock:
            count = self._sdk.read(
                ReadRequest(
                    blocks=BlockBatch(hash_ids=[block_hash_id], data_ids=[0]),
                    buffers=BufferBindings(
                        swa=(self.staging.data_ptr(),), swa_ids=(0,)
                    ),
                    timeout=timeout,
                    remote_addrs=remote_addrs or (),
                    swa_block_step=1,
                )
            )
            if count == 1:
                self.layout.unpack([slots], self.staging[1:2])
                self._synchronize_device()
            return count

    def stop(self) -> None:
        """Wake snapshot waiters and stop the underlying SDK client."""
        with self._snapshot_condition:
            if self._stopped:
                return
            self._stopped = True
            self._snapshot_condition.notify_all()
        self._sdk.stop()


def _uniform_nonzero(values: Sequence[int], name: str) -> int:
    if not values:
        return 0
    unique = set(map(int, values))
    if len(unique) != 1 or next(iter(unique)) <= 0:
        raise ValueError(f"SPA {name} block sizes are invalid: {sorted(unique)}")
    return unique.pop()


def create_spa_binding(ctx: AttentionStoreInitContext) -> AttentionStoreBinding:
    """Create the DeepSeek-V4 SPA page/checkpoint binding."""
    from attentionstore_sdk.sdk import (
        AttentionStoreSDK,
        BufferBindings,
        SPAConfig,
    )

    from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
        DeepSeekV4TokenToKVPool,
    )

    pool = ctx.kv_cache
    if not isinstance(pool, DeepSeekV4TokenToKVPool):
        raise ValueError("SPA AttentionStore requires DeepSeekV4TokenToKVPool")
    ratios = pool.compression_ratios[pool._stage_start : pool._stage_end]
    c4_count = sum(ratio == 4 for ratio in ratios)
    c128_count = sum(ratio == 128 for ratio in ratios)
    ptrs, _lens, item_lens = pool.get_contiguous_buf_infos()
    expected = c4_count * 2 + c128_count
    if len(ptrs) != expected or len(item_lens) != expected:
        raise ValueError(
            "SPA contiguous buffer order mismatch: "
            f"expected={expected}, ptrs={len(ptrs)}, items={len(item_lens)}"
        )
    c4_kv_end = c4_count
    c4_indexer_end = c4_kv_end + c4_count
    c4_kv_ptrs = ptrs[:c4_kv_end]
    c4_indexer_ptrs = ptrs[c4_kv_end:c4_indexer_end]
    c128_ptrs = ptrs[c4_indexer_end:]
    c4_kv_bytes = _uniform_nonzero(item_lens[:c4_kv_end], "c4 KV")
    c4_indexer_bytes = _uniform_nonzero(
        item_lens[c4_kv_end:c4_indexer_end], "c4 indexer"
    )
    c128_bytes = _uniform_nonzero(item_lens[c4_indexer_end:], "c128 KV")

    namespace_base = f"{ctx.namespace_base}:pp{ctx.pp_rank}:tp{ctx.tp_rank}:spa"
    page_namespace = f"{namespace_base}:page"
    page_sdk = AttentionStoreSDK(
        _sdk_pod_config(ctx, page_namespace),
        SPAConfig(
            block_token_size=ctx.page_size,
            c4_layer_count=c4_count,
            c128_layer_count=c128_count,
            c4_kv_bytes=c4_kv_bytes,
            c4_indexer_bytes=c4_indexer_bytes,
            c128_kv_bytes=c128_bytes,
            swa_read_layer_bytes=0,
            swa_write_layer_bytes=0,
            swa_layer_count=0,
            swa_step=ctx.page_size,
        ),
        enable_as_kv_rw=True,
    )
    page_buffers = BufferBindings(
        c4_kv=c4_kv_ptrs,
        c4_indexer=c4_indexer_ptrs,
        c128_kv=c128_ptrs,
    )
    page_store = _BufferPageIO(
        page_sdk,
        page_buffers,
        layer_ids=(0,),
    )

    try:
        state_layout = make_deepseek_v4_checkpoint_layout(
            pool, include_compressed_pages=False
        )
        state_namespace = f"{namespace_base}:checkpoint:{state_layout.layout_id[:16]}"
        state_sdk = AttentionStoreSDK(
            _sdk_pod_config(ctx, state_namespace),
            SPAConfig(
                block_token_size=ctx.page_size,
                c4_layer_count=0,
                c128_layer_count=0,
                c4_kv_bytes=0,
                c4_indexer_bytes=0,
                c128_kv_bytes=0,
                swa_read_layer_bytes=state_layout.bytes_per_block,
                swa_write_layer_bytes=state_layout.bytes_per_block,
                swa_layer_count=1,
                swa_step=ctx.page_size,
            ),
            enable_as_kv_rw=True,
        )
    except Exception:
        page_store.stop()
        raise
    try:
        state_store = SPACheckpointStore(state_sdk, state_layout)
    except Exception:
        state_sdk.stop()
        page_store.stop()
        raise
    page_bytes_per_block = (
        c4_kv_bytes + c4_indexer_bytes
    ) * c4_count + c128_bytes * c128_count
    cache_store = CheckpointedPageStore(page_store, state_store)
    try:
        page_endpoint = AttentionStoreEndpointBinding(
            name="spa",
            role=AttentionStoreEndpointRole.PAGE,
            store=page_store,
            namespace=page_namespace,
            layout_id="spa-page",
            bytes_per_block=page_bytes_per_block,
            payload_fields=("c4_kv", "c4_indexer", "c128_kv"),
        )
        checkpoint_endpoint = AttentionStoreEndpointBinding(
            name="checkpoint",
            role=AttentionStoreEndpointRole.CHECKPOINT,
            store=state_store,
            namespace=state_namespace,
            layout_id=state_layout.layout_id,
            bytes_per_block=state_layout.bytes_per_block,
            payload_fields=tuple(entry.spec.name for entry in state_layout.entries),
        )
        return AttentionStoreBinding(
            plan_name="SPA",
            cache_store=cache_store,
            endpoints=(page_endpoint, checkpoint_endpoint),
        )
    except Exception:
        cache_store.stop()
        raise


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


_ATTENTION_STORE_CACHE_PLANS = {
    "MHA": AttentionStoreCachePlan(
        "MHA",
        MHACacheCoordinator,
        tree_components=(ComponentType.FULL,),
    ),
    "MLA": AttentionStoreCachePlan(
        "MLA",
        MLACacheCoordinator,
        tree_components=(ComponentType.FULL,),
    ),
    "DSA": AttentionStoreCachePlan(
        "DSA",
        DSACacheCoordinator,
        tree_components=(ComponentType.FULL,),
    ),
    "SPA": AttentionStoreCachePlan(
        "SPA",
        SPACacheCoordinator,
        tree_components=(ComponentType.FULL, ComponentType.SWA),
    ),
    "KDA": AttentionStoreCachePlan(
        "KDA",
        KDACacheCoordinator,
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
