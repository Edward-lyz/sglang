"""Coordinator implementations for checkpointed-page cache plans."""

from __future__ import annotations

import concurrent.futures
import logging
import os
import time
from array import array
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence, Tuple

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InitLoadBackParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.chunk_tail_swa_pool import free_chunk_tail_swa_slots
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import (
    UnifiedRadixCache,
    UnifiedTreeNode,
)
from sglang.srt.utils.nvtx_utils import (
    NVTX_OPERATIONS_ENABLED,
    operations_nvtx_range,
    profile_method,
)

from .checkpoint_binding import PackedStateSnapshot
from .coordinator import (
    CheckpointedPageCacheCoordinator,
    _acquire_node_release,
    _contiguous_page_ids,
    _materialize_page_ids,
    _noop_release,
    _page_id_metadata,
    _request_block_hash_ids,
)
from .types import (
    AttentionStoreEndpointBinding,
    AttentionStoreWriteContext,
    AttentionStoreWriteOwner,
)
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_KDA_PENDING_HASH = "as_kda_pending_hash"
_KDA_PUBLISHED_HASH = "as_kda_published_hash"


@dataclass
class _SPLoadBackTask:
    """One deferred SPA/SPAG load-back, split across two threads.

    The scheduler thread builds this at admission (``_submit_load_back``:
    evict, allocate, hand over the node hold), a bounded background executor
    runs the SDK read (``_run_load_back_read``: prepare + plain full
    ``ascli.read``, no torch.distributed), and the scheduler thread commits
    the result at the next loop top (``commit_load_backs``: future.result, one
    ``_as_min``, insert). Submission order remains the commit order even when
    more than one read lane is enabled.

    - submit (scheduler): rid, load_info, loaded, requested, local_hit_len,
      started, lock (the node hold taken at submit and released at commit).
    - background read: future result = read_count on success, or the read error.
    - commit (scheduler): consumes the fields, then the task is dropped.
    """

    rid: str
    req: Any
    load_info: dict
    loaded: Optional[torch.Tensor]
    requested: int
    local_hit_len: int
    started: float
    lock: Callable[[], None]
    future: Any = None
    diffuse_page_slots: Optional[Tuple[int, ...]] = None
    diffuse_extra_blocks: Tuple[int, ...] = ()
    allocation_ready: Any = None
    native_read_future: Any = None
    native_sharded_blocks: Optional[int] = None
    native_metadata: bool = False


class _NativeWriteSlots:
    """Physical CUDA slots retained until native metadata/copy have joined."""
    def __init__(self, page_size, slots, ready_event):
        self.page_size, self.slots, self.ready_event = (
            page_size,
            slots,
            ready_event,
        )
        self.quarantined = False

    def materialize(self):
        """Wait for allocation metadata and return physical page identifiers.

        Returns:
            tuple[int, ...]: Contiguous physical page identifiers.

        Raises:
            TimeoutError: The allocation event did not complete in time.
            ValueError: The slots do not form aligned contiguous pages.
        """
        from .coordinator import _wait_copy_event, _COPY_EVENT_TIMEOUT_S

        _wait_copy_event(
            self.ready_event, _COPY_EVENT_TIMEOUT_S, "native slots"
        )
        return _contiguous_page_ids(self.slots, self.page_size)

    def release(self):
        """Quarantine slots whose allocation event has not completed.

        Returns:
            None.
        """
        from .coordinator import _event_completed, _quarantine_copy

        if not _event_completed(self.ready_event):
            _quarantine_copy(self, "native slots")


class _NativeWriterResult:
    """Scheduler drain owns staging release; completion alone takes no GIL."""
    def __init__(self, cache, task, native):
        import threading

        self.cache, self.task, self.native = cache, task, native
        self._lock = threading.Lock()
        self._finalized = False
        self._value = None
        self.started = time.perf_counter()

    def done(self):
        """Check native completion without releasing scheduler-owned resources.

        Returns:
            bool: Whether the result is finalized or native work is complete.
        """
        return self._finalized or self.native.done()

    def result(self, timeout=None):
        """Drain the native write once and release staging after native
        cleanup.

        Args:
            timeout: Optional boundary wait in seconds; None uses the native
                wait limit.

        Returns:
            int: Written block count, or -1 for a completed write failure.

        Raises:
            TimeoutError: Native work remains active after the boundary wait.
        """
        with self._lock:
            if not self._finalized:
                try:
                    value = self.native.result(timeout)
                except TimeoutError:
                    # The native job still borrows slots/staging after a
                    # boundary wait timeout; the caller may drain it again.
                    raise
                except Exception as exc:
                    self.cache._log_as_failure(
                        "write",
                        rid=self.task.rid,
                        requested=self.task.requested_blocks,
                        error=exc,
                        started=self.started,
                    )
                    self._value = -1
                else:
                    self._value = value
                    self.cache._log_as_result(
                        "write",
                        rid=self.task.rid,
                        requested=self.task.requested_blocks,
                        result=value,
                        started=self.started,
                    )
                finally:
                    if getattr(self.native, "_drained", False):
                        # Mark before callbacks so release cannot run twice
                        # even if a diagnostic or release hook raises.
                        self._finalized = True
                        self.task.release_after_write()
            return self._value


@dataclass
class _SPAWriteTask:
    rid: str
    page_endpoint: Optional[AttentionStoreEndpointBinding]
    checkpoint_store: Any
    block_hash_ids: Sequence[int]
    page_ids: Any
    checkpoint_snapshot: Optional[PackedStateSnapshot]
    params: dict
    release_after_write: Callable[[], None] = _noop_release
    release: Callable[[], None] = _noop_release
    owner: Optional[AttentionStoreWriteOwner] = None
    # SPAG only: one chunk-tail staging slot per checkpoint of this write's page
    # range, indexed by checkpoint number from the request's start. ``-1`` means
    # "no staged window for this checkpoint" -- either an AS-hit prefix, a
    # staging-pool shortage, or a checkpoint another task already owns. Purely
    # rank-local, so it must never reach the write key / logical id.
    swa_slot_ids: Sequence[int] = ()
    # Exclusive end of the checkpoint range this task owns (it must release those
    # slots when it retires). The owned range starts at the request's
    # ``as_written_count`` as it stood when the task was prepared.
    swa_owned_end: int = 0
    request: Any = None
    producer_ready: Any = None
    # Logical checkpoint slots of the terminal page, kept for the AS trace only.
    checkpoint_slots: dict[str, int] = field(default_factory=dict)
    trace_recorder: Any = None

    @property
    def requested_blocks(self) -> int:
        """Return the number of SPA page blocks requested by this task."""
        return len(self.block_hash_ids)


@dataclass
class _KDAWriteTask:
    rid: str
    node: UnifiedTreeNode
    block_hash_ids: Sequence[int]
    page_ids: Any
    checkpoint_slot: Any
    params: dict
    producer_ready: Any = None
    request: Any = None
    write_key: Optional[tuple[int, int]] = None
    release_after_write: Callable[[], None] = _noop_release
    release: Callable[[], None] = _noop_release
    owner: Optional[AttentionStoreWriteOwner] = None
    trace_recorder: Any = None

    @property
    def requested_blocks(self) -> int:
        """Return the number of KDA page blocks requested by this task."""
        return len(self.block_hash_ids)

    @property
    def endpoint_hash(self) -> int:
        """Return the terminal block hash associated with the checkpoint."""
        return int(self.block_hash_ids[-1]) if self.block_hash_ids else 0


class SPACacheCoordinator(CheckpointedPageCacheCoordinator):
    """Coordinate SPA page writes with a stable endpoint checkpoint snapshot."""

    write_thread_name = "as_spa_writer"
    load_back_thread_name = "as_load_back_reader"

    def _ensure_load_back_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        """Bounded local reader lanes for deferred load-back reads."""
        if self._closed:
            raise RuntimeError("AttentionStore coordinator is already closed")
        if self._load_back_executor is None:
            max_workers = max(1, int(envs.SGLANG_AS_LOAD_BACK_READ_WORKERS.get()))
            self._load_back_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix=self.load_back_thread_name,
            )
        return self._load_back_executor

    def _discard_load_back(
        self, cache, task: _SPLoadBackTask, error: Exception
    ) -> None:
        """Release a submitted load-back's resources and report the failure."""
        self._free_loaded_pages(cache, task.loaded, start_pos=task.local_hit_len)
        task.lock()
        cache._log_as_failure(
            "load_back",
            rid=task.rid,
            requested=task.requested,
            error=error,
            started=task.started,
        )

    def _use_deferred_allocation(self, cache):
        if not envs.SGLANG_AS_DEFER_ALLOCATION_GATE.get():
            return False
        if not hasattr(self, "_deferred_allocation_enabled"):
            eligible = (
                type(self) is SPAGCacheCoordinator
                and getattr(cache, "pp_world_size", getattr(cache, "pp_size", 1)) <= 1
                and self._native_physical_writer(cache)
            )
            # One eligibility vote per coordinator keeps unsupported local
            # layouts on the same compatibility path on every attention rank.
            self._deferred_allocation_enabled = bool(cache._as_min(int(eligible)))
            logger.info(
                "Deferred allocation selected: enabled=%s",
                self._deferred_allocation_enabled)
        return self._deferred_allocation_enabled

    def _failed_allocation(
        self, cache, req, load_info, local_hit_len, started, error, *, logged=False
    ):
        """Keep an identical queue slot on every rank after local failure."""
        requested = int(load_info["block_count"])
        if not logged:
            cache._log_as_failure(
                "load_back", rid=req.rid, requested=requested, error=error, started=started)
        task = _SPLoadBackTask(
            rid=req.rid, req=req, load_info=load_info, loaded=None,
            requested=requested, local_hit_len=local_hit_len, started=started,
            lock=_noop_release)
        task.future = concurrent.futures.Future()
        task.future.set_result(error)
        self._pending_load_backs.append(task)
        return task

    def _submit_load_back(
        self, cache, req, load_info, anchor
    ) -> Optional[_SPLoadBackTask]:
        """Evict, allocate, and hold the node; IO happens on the reader thread."""
        defer_vote = self._use_deferred_allocation(cache)
        requested = int(load_info["block_count"])
        started = time.perf_counter()
        token_count = requested * cache.page_size
        with operations_nvtx_range("radix.as.lb.evict"):
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
        # The allocator consumes a one-element tensor; keep an existing index on device.
        last_loc = req.prefix_indices[-1].reshape(1) if local_hit_len else None
        with operations_nvtx_range("radix.as.lb.alloc"):
            try:
                if torch.device(device).type == "cuda" and torch.version.hip is None:
                    # Keep existing last_loc on device; only pack Python scalars.
                    metadata_cpu = torch.tensor(
                        [local_hit_len, local_hit_len + token_count, -1],
                        dtype=torch.int64,
                        device="cpu",
                    )
                    metadata = metadata_cpu.to(device, non_blocking=True)
                    prefix_lens = metadata[:1]
                    seq_lens = metadata[1:2]
                    load_last_loc = (
                        last_loc.to(device=device, dtype=torch.int64)
                        if last_loc is not None
                        else metadata[2:3]
                    )
                else:
                    prefix_lens = prefix_cpu.to(device)
                    seq_lens = seq_cpu.to(device)
                    load_last_loc = (
                        last_loc.to(device=device, dtype=torch.int64)
                        if last_loc is not None
                        else torch.tensor([-1], dtype=torch.int64, device=device)
                    )
                loaded = cache.token_to_kv_pool_allocator.alloc_extend_swa_tail(
                    prefix_lens,
                    prefix_cpu,
                    seq_lens,
                    seq_cpu,
                    load_last_loc,
                    token_count,
                    # One whole SWA page, even though a plan may restore only its
                    # upper half: alloc_extend_swa_tail maps the *last* swa_tail_len
                    # full slots onto the freshly allocated SWA page, so asking for
                    # less than a page would land the restored window in the page's
                    # lower half and desynchronize it from the stored byte offset.
                    min(token_count, cache.page_size),
                )
            except Exception as exc:
                if defer_vote:
                    return self._failed_allocation(
                        cache, req, load_info, local_hit_len, started, exc)
                cache._log_as_failure(
                    "load_back",
                    rid=req.rid,
                    requested=requested,
                    error=exc,
                    started=started,
                )
                cache._as_min(0)
                return None
        if loaded is None and defer_vote:
            return self._failed_allocation(
                cache, req, load_info, local_hit_len, started,
                RuntimeError(f"cannot allocate {token_count} tokens"))
        if loaded is None:
            cache._log_as_failure(
                "load_back",
                rid=req.rid,
                requested=requested,
                error=RuntimeError(f"cannot allocate {token_count} tokens"),
                started=started,
            )
            cache._as_min(0)
            return None
        # Compatibility mode keeps048's per-request allocation vote.
        # Deferred mode queues every rank, including local failures;
        # existing CP readiness/read-count votes decide before publish.
        if not defer_vote and cache._as_min(1) == 0:
            self._free_loaded_pages(cache, loaded, start_pos=local_hit_len)
            return None
        lock = _acquire_node_release(cache, anchor)
        task = _SPLoadBackTask(
            rid=req.rid,
            req=req,
            load_info=load_info,
            loaded=loaded,
            requested=requested,
            local_hit_len=local_hit_len,
            started=started,
            lock=lock,
        )
        if loaded.is_cuda:
            task.allocation_ready = torch.cuda.Event()
            task.allocation_ready.record(torch.cuda.current_stream(loaded.device))
        try:
            native = self._try_native_metadata(cache, task)
            if native is not None:
                task.native_metadata = True
                task.native_cp_sharded = (
                    getattr(cache, "_as_read_diffuser", None) is not None
                    and task.requested > 1
                )
                task.native_read_future = native
                task.future = native
            else:
                task.future = self._ensure_load_back_executor().submit(
                    self._run_load_back_read, cache, task
                )
        except Exception as exc:
            self._discard_load_back(cache, task, exc)
            if defer_vote:
                return self._failed_allocation(
                    cache, req, load_info, local_hit_len, started, exc, logged=True)
            return None
        self._pending_load_backs.append(task)
        return task

    def _try_native_metadata(self, cache, task):
        """Only reviewed physical mappings may bypass the Python reader."""
        if os.environ.get("ATTENTIONSTORE_NATIVE_METADATA") != "1":
            return None
        submit = getattr(cache.ascli, "submit_read_slots_native", None)
        if submit is None:
            return None
        # Custom coordinators may override the page preparation semantics.
        if type(self) is not SPAGCacheCoordinator:
            return None
        policy = self._swa_policy(cache)
        mapping, step = None, None
        if policy is not None:
            from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
            from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
            from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
            allocator = cache.token_to_kv_pool_allocator
            # Unified and request-ring allocators use other address spaces.
            if type(allocator) is not SWATokenToKVPoolAllocator or allocator.swa_req_ring:
                return None
            pool = allocator.get_kvcache()
            if type(pool) not in (DeepSeekV4TokenToKVPool, SWAKVPool):
                return None
            if "translate_loc_from_full_to_swa" in vars(pool):
                return None
            mapping = pool.full_to_swa_index_mapping
            if mapping is None:
                return None
            position = (task.load_info["start_block"] + task.requested) * cache.page_size
            step = policy.block_step_for_position(position)
        diffuser = getattr(cache, "_as_read_diffuser", None)
        cp_rank, cp_size = (0, 1) if diffuser is None else (diffuser.cp_rank, diffuser.cp_size)
        return submit(
            task.load_info["hashes"], task.loaded, cache.page_size, mapping,
            task.allocation_ready, cp_rank=cp_rank, cp_size=cp_size,
            swa_block_step=step, timeout=5.0,
            remote_addrs=task.load_info.get("remote_addrs") or None)

    def _run_load_back_read(self, cache, task: _SPLoadBackTask):
        """Reader thread: local SDK IO only; no torch.distributed calls.

        Resolve allocator metadata here so the scheduler does not pay the D2H
        wait. A sharded read saves its physical slot map on the task; only the
        scheduler may agree on all-rank success, diffuse and publish those pages.
        """
        if task.loaded.is_cuda:
            with torch.cuda.device(task.loaded.device):
                stream = getattr(self, "_load_back_metadata_stream", None)
                if stream is None:
                    stream = torch.cuda.Stream(device=task.loaded.device)
                    self._load_back_metadata_stream = stream
                with torch.cuda.stream(stream):
                    if task.allocation_ready is not None:
                        stream.wait_event(task.allocation_ready)
                    return self._run_local_load_back_read(cache, task)
        return self._run_local_load_back_read(cache, task)

    def _run_local_load_back_read(self, cache, task: _SPLoadBackTask):
        with operations_nvtx_range("radix.as.lb.read"):
            try:
                page_ids, read_args, read_kwargs = self._prepare_read(
                    cache, task.req, task.load_info, task.loaded
                )
            except Exception as exc:
                return exc
            try:
                hashes = task.load_info["hashes"]
                diffuser = getattr(cache, "_as_read_diffuser", None)
                sharded = diffuser is not None and task.requested > 1
                if sharded:
                    terminal = task.requested - 1
                    indices = sorted(
                        set(diffuser.local_shard(task.requested)) | {terminal}
                    )
                    task.diffuse_page_slots = tuple(int(page) for page in page_ids)
                    task.diffuse_extra_blocks = (terminal,)
                    hashes = [hashes[index] for index in indices]
                    page_ids = [task.diffuse_page_slots[index] for index in indices]
                submit_native = (
                    getattr(cache.ascli, "submit_read_native", None)
                    if os.environ.get("ATTENTIONSTORE_NATIVE_CONTROL") == "1" else None
                )
                if submit_native is not None:
                    native = submit_native(
                        hashes, page_ids, *read_args, timeout=5.0,
                        remote_addrs=task.load_info.get("remote_addrs") or None,
                        **read_kwargs,
                    )
                    if native is not None:
                        task.native_sharded_blocks = len(hashes) if sharded else None
                        # Last task mutation on the reader. Native copy/release
                        # completion is now visible without executor.set_result.
                        task.native_read_future = native
                        return None
                got = cache.ascli.read(
                    hashes,
                    page_ids,
                    *read_args,
                    timeout=5.0,
                    remote_addrs=task.load_info.get("remote_addrs") or None,
                    **read_kwargs,
                )
                # This is a local readiness vote, not published host credit.
                # commit_load_backs MIN-reduces it across ranks, then diffuses.
                if sharded:
                    return task.requested if got == len(hashes) else 0
                return got
            except Exception as exc:
                return exc

    def _release_task_requests(self, tasks: list) -> None:
        """Drop the submit-side flags so the waiting request re-admits."""
        for task in tasks:
            req = task.req
            if getattr(req, "_as_pending_load_back", None) is task:
                req._as_pending_load_back = None

    @staticmethod
    def _drain_task_result(task: _SPLoadBackTask) -> Any:
        """Block on one read future; an executor-level failure becomes the error."""
        native = getattr(task, "native_read_future", None)
        if native is not None:
            try:
                count = native.result()
                if getattr(task, "native_metadata", False):
                    metadata = native.native._completion
                    pages, indices = metadata["prepared_pages"], metadata["local_indices"]
                    if len(pages) != task.requested:
                        raise RuntimeError("native prepared page count differs")
                    if getattr(task, "native_cp_sharded", False):
                        task.diffuse_page_slots = tuple(pages)
                        task.diffuse_extra_blocks = (task.requested - 1,)
                        return task.requested if count == len(indices) else 0
                    return count
                sharded_blocks = task.native_sharded_blocks
                return (task.requested if count == sharded_blocks else 0) if sharded_blocks is not None else count
            except Exception as exc:
                return exc
        if task.future is None:
            return None
        try:
            return task.future.result()
        except Exception as exc:
            return exc

    def commit_load_backs(self, cache) -> None:
        """Commit the completed prefix of deferred load-backs.

        A rank-local reader can finish out of order relative to another
        attention rank.  Use ``Future.done()`` only as a readiness hint, then
        commit the same contiguous ready prefix everywhere through a MIN vote.
        Tasks after the first not-ready entry stay queued for the next loop.
        This keeps submission order and the existing read-count correctness
        gate while allowing ready host hits to refill an underfilled batch.
        """
        if not self._pending_load_backs:
            return
        pending_all = self._pending_load_backs
        local_ready = 0
        for task in pending_all:
            completion = getattr(task, "native_read_future", None)
            if completion is None:
                completion = task.future
            if completion is not None and not completion.done():
                break
            local_ready += 1

        # Future completion can differ briefly across CP ranks.  Agree on the
        # largest prefix that is locally ready everywhere before removing any
        # task from the shared queue.  The existing result MIN below still
        # validates each read count.
        ready = torch.tensor([local_ready], dtype=torch.int64)
        cache._all_reduce_attn_groups(ready, torch.distributed.ReduceOp.MIN)
        ready_count = int(ready.item())
        if ready_count <= 0:
            return

        pending = pending_all[:ready_count]
        self._pending_load_backs = pending_all[ready_count:]
        # One collective for the whole drain: MIN over ranks per task slot.
        # Ranks submit identical task counts at the same loop position, so the
        # stacked vectors pair element-wise without a length handshake.
        drained = [self._drain_task_result(task) for task in pending]
        read_counts = [
            0 if isinstance(r, BaseException) else int(r or 0) for r in drained
        ]
        stacked = torch.tensor(read_counts, dtype=torch.int64)
        cache._all_reduce_attn_groups(stacked, torch.distributed.ReduceOp.MIN)
        confirmed_counts = stacked.tolist()
        grouped = []
        if envs.SGLANG_AS_READ_DIFFUSE_GROUP_COMMIT.get():
            grouped = [
                task for task, result, count in zip(pending, drained, confirmed_counts)
                if not isinstance(result, BaseException)
                and count == task.requested
                and task.diffuse_page_slots is not None
            ]
            if grouped:
                cache._as_read_diffuser.diffuse_many(
                    [
                        (task.diffuse_page_slots, task.requested, task.diffuse_extra_blocks)
                        for task in grouped
                    ],
                    peers_ready=True,
                )
        grouped_ids = {id(task) for task in grouped}
        for task, result, read_count in zip(pending, drained, confirmed_counts):
            read_error = result if isinstance(result, BaseException) else None
            read_count = int(read_count)
            if read_error is None:
                if read_count == task.requested and task.diffuse_page_slots is not None:
                    if id(task) not in grouped_ids:
                        cache._as_read_diffuser.diffuse(
                            task.diffuse_page_slots,
                            task.requested,
                            extra_local_blocks=task.diffuse_extra_blocks,
                            peers_ready=True,
                        )
                    cache.ascli.emit_content_digest(
                        "postdiffuse", task.load_info["hashes"],
                        task.diffuse_page_slots,
                    )
                cache._log_as_result(
                    "load_back", rid=task.rid, requested=task.requested,
                    result=read_count, started=task.started,
                )
            else:
                cache._log_as_failure(
                    "load_back",
                    rid=task.rid,
                    requested=task.requested,
                    error=read_error,
                    started=task.started,
                )
            if read_error is not None or read_count != task.requested:
                self._free_loaded_pages(
                    cache, task.loaded, start_pos=task.local_hit_len
                )
                task.lock()
                self._release_task_requests([task])
                continue
            token_count = task.requested * cache.page_size
            self._insert_loaded_prefix(cache, task, token_count)
            task.lock()
            self._release_task_requests([task])

    def _insert_loaded_prefix(
        self, cache, task: _SPLoadBackTask, token_count: int
    ) -> None:
        """Publish a finished read into the tree and arm the read latch."""
        req = task.req
        total_prefix_len = task.local_hit_len + token_count
        token_ids = req.full_untruncated_fill_ids[:total_prefix_len]
        values = torch.cat([req.prefix_indices, task.loaded]).to(dtype=torch.int64)
        restored_swa_len = self._restored_swa_len(cache, token_count)
        cache.insert(
            InsertParams(
                key=RadixKey(
                    token_ids, req.extra_key, is_bigram=cache.tree_core.is_eagle
                ),
                value=values,
                prev_prefix_len=task.local_hit_len,
                # Everything before the restored window is an SWA tombstone. This
                # must describe the bytes actually read back, not the SWA pages
                # allocated: over-claiming makes the tree serve uninitialized SWA.
                swa_evicted_seqlen=task.local_hit_len
                + max(0, token_count - restored_swa_len),
            )
        )
        match = UnifiedRadixCache.match_prefix(
            cache,
            MatchPrefixParams(
                key=RadixKey(
                    token_ids, req.extra_key, is_bigram=cache.tree_core.is_eagle
                ),
                req=req,
            ),
        )
        tree_indices = match.device_indices[task.local_hit_len:total_prefix_len]
        if len(tree_indices) != token_count:
            logger.warning(
                "AS cache insert: cache_plan=%s rid=%s status=error "
                "reason=shortened_prefix expected_tokens=%s actual_tokens=%s",
                cache.attention_store_plan.name,
                task.rid,
                token_count,
                len(tree_indices),
            )
        # Do NOT repoint req.last_node here: the req may hold a lock receipt
        # anchored on its admission-time node, and repointing without a new
        # receipt breaks dec_lock_ref's anchor assert. The next admission's
        # match_prefix repoints it with a fresh receipt.
        # This path is reached only after every attention rank confirms the
        # complete read. A later re-match sees device-resident pages, but their
        # origin for this request remains the completed AS host restore.
        req._as_materialized_host_span = (
            task.local_hit_len,
            task.local_hit_len + min(token_count, len(tree_indices)),
        )
        req.as_read_pending_lock = True

    def capture_finished(
        self, cache, req, is_insert: bool
    ) -> AttentionStoreWriteContext:
        """Snapshot SPA slots for a finished request."""
        committed_len = req.effective_kv_committed_len()
        token_ids = (req.origin_input_ids + req.output_ids)[:committed_len]
        snapshot = cache.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, :committed_len
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
            req.kv.req_pool_idx, : len(token_ids)
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
            self.enqueue_dummy(cache, context.req.rid)
            return

        try:
            key = RadixKey(
                context.token_ids,
                context.req.extra_key,
                is_bigram=cache.tree_core.is_eagle,
            )
            if len(key) % cache.page_size:
                key = key.page_aligned(cache.page_size)
            match = UnifiedRadixCache.match_prefix(
                cache,
                MatchPrefixParams(key=key),
            )
            node = match.last_device_node
            # The post-commit re-match only publishes the device-resident,
            # page-aligned front of the committed prefix. On a warm hit part of
            # the prefix can be offloaded to host/L2, so device_indices covers
            # fewer pages than the full key -- that is legitimate, not an error.
            published_len = (
                len(match.device_indices) // cache.page_size
            ) * cache.page_size
            if node is None or published_len == 0:
                logger.warning(
                    "SPA local commit did not publish the write prefix rid=%s "
                    "is_eagle=%s page=%s len_key=%s published_indices=%s "
                    "node_is_none=%s requested_blocks=%s",
                    context.req.rid,
                    cache.tree_core.is_eagle,
                    cache.page_size,
                    len(key),
                    len(match.device_indices),
                    node is None,
                    task.requested_blocks,
                )
                raise RuntimeError("SPA local commit did not publish the write prefix")
            # WAR fix: build page_ids from the tree's canonical, ref-protected
            # slots (post-commit), not the pre-commit snapshot. _acquire_node_release
            # below protects these device_indices for the async gather's lifetime.
            cache_len = min(len(key), published_len)
            if cache_len < len(key):
                logger.debug(
                    "SPA write floored to published prefix rid=%s len_key=%s "
                    "published_indices=%s published_pages=%s requested_blocks=%s",
                    context.req.rid,
                    len(key),
                    len(match.device_indices),
                    cache_len // cache.page_size,
                    task.requested_blocks,
                )
                self._refloor_task(cache, context, task, match, cache_len)
            page_ids = self.prepare_page_ids(cache, match.device_indices[:cache_len])
            task.page_ids = page_ids

            task.release_after_write = self._make_release_after_write(
                cache, task, page_ids
            )
            task.release = _acquire_node_release(cache, node)
            # Ownership moves to the task *before* submission: if submit_write
            # fails it runs release_after_write itself, so the resources must
            # already belong to the task for that release to be the only one.
            self._commit_write_ownership(cache, context, task)
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
            self.enqueue_dummy(cache, context.req.rid)

    def _refloor_task(self, cache, context, task, match, cache_len: int) -> None:
        """Shrink a prepared task to the ``cache_len`` prefix the tree published.

        Truncates the block hashes and re-derives the checkpoint for the new
        terminal page, so the endpoint hash (``block_hash_ids[-1]``) and the
        recurrent-state snapshot still describe the same page.
        """
        task.block_hash_ids = task.block_hash_ids[: cache_len // cache.page_size]
        swa_indices = cache.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
            match.device_indices[cache_len - cache.page_size : cache_len]
        )
        swa_page_id = int(swa_indices[0].item()) // cache.page_size
        task.checkpoint_slots = self.state_slots(
            cache, context.req.kv.req_pool_idx, swa_page_id
        )
        new_snapshot = task.checkpoint_store.snapshot(task.checkpoint_slots)
        if task.checkpoint_snapshot is not None:
            task.checkpoint_snapshot.release()
        task.checkpoint_snapshot = new_snapshot

    def _make_release_after_write(self, cache, task, page_ids) -> Callable[[], None]:
        """Build the callback that frees everything the write staged."""
        checkpoint_snapshot = task.checkpoint_snapshot

        def _release_staging() -> None:
            try:
                page_ids.release()
            finally:
                if checkpoint_snapshot is not None:
                    checkpoint_snapshot.release()

        return _release_staging

    def _commit_write_ownership(self, cache, context, task) -> None:
        """Hand the task's rank-local resources over to the writer thread.

        Called once the task is fully built, right before ``submit_write``, so
        the task's ``release_after_write`` is the single owner from here on.
        """

    def _read_payload_args(self, cache, req, load_info, loaded):
        """Derive the plan-specific extra arguments of the load-back read.

        Returns ``(positional, keyword)``. SPA appends the checkpoint slot map
        that names where the terminal blob must land. Plans that carry their
        state inside page blocks can specialize ``_prepare_read`` instead.
        """
        swa_indices = cache.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
            loaded[-cache.page_size :]
        )
        swa_page_id = int(swa_indices[0].item()) // cache.page_size
        return (self.state_slots(cache, req.kv.req_pool_idx, swa_page_id),), {}

    def _prepare_read(self, cache, req, load_info, loaded):
        """Resolve physical page IDs and plan-specific read arguments."""
        page_ids = _contiguous_page_ids(loaded, cache.page_size)
        read_args, read_kwargs = self._read_payload_args(cache, req, load_info, loaded)
        return page_ids, read_args, read_kwargs

    def _free_loaded_pages(self, cache, loaded, *, start_pos: int) -> None:
        """Release a validated, page-aligned alloc_extend_swa_tail result."""
        if loaded is None:
            return
        allocator = cache.token_to_kv_pool_allocator
        # Only the last page owns SWA slots. Free its peer before returning all
        # full pages; free_segment(loaded) would try to return unmapped SWA pages.
        # Use the paged allocator's fixed-shape segment release paths.
        tail_start = max(0, len(loaded) - cache.page_size)
        allocator.free_swa_segment(
            loaded[tail_start:], start_pos=start_pos + tail_start
        )
        allocator.free_full_segment(loaded, start_pos=start_pos)

    def _restored_swa_len(self, cache, token_count: int) -> int:
        """Tokens at the restored tail whose SWA state this read really fills.

        Must stay page-aligned: the tree asserts ``swa_evicted_seqlen`` is a
        multiple of ``page_size``, so a plan that restores only part of the final
        SWA page still reports the whole page here. That is safe because matches
        are page-aligned too, so no reachable prefix ends inside the page, and the
        only consumer of that page is the extend that follows the restore, whose
        sliding window never reaches past the restored part.
        """
        return min(token_count, cache.page_size)

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
        # Offload the page-aligned prefix even when the fill length carries a
        # partial tail page. The radix tree only publishes full pages and
        # after_local_commit already page-aligns the write key, so flooring
        # here (instead of bailing on any unaligned length) lets every request
        # with at least one complete page offload and emit its digests.
        cache_len = (len(token_ids) // cache.page_size) * cache.page_size
        if cache_len == 0 or len(snapshot) < cache_len:
            return None
        token_ids = token_ids[:cache_len]
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
                self.state_slots(cache, req.kv.req_pool_idx, swa_page_id)
            )

            # WAR fix (align KDA / commit 763d75172f): do NOT build page_ids from
            # this pre-commit snapshot. super().cache_*_req (run after this, before
            # the async gather) frees the duplicate prefix slots the snapshot
            # references, and a later forward can reuse/overwrite them. page_ids are
            # built in after_local_commit from the tree's ref-protected
            # match.device_indices instead. Keep only the (already staged, copied)
            # checkpoint snapshot here.
            def release_staging() -> None:
                if checkpoint_snapshot is not None:
                    checkpoint_snapshot.release()

            return _SPAWriteTask(
                rid=req.rid,
                page_endpoint=cache.binding.page_endpoint,
                checkpoint_store=checkpoint_store,
                block_hash_ids=hashes,
                page_ids=None,
                checkpoint_snapshot=checkpoint_snapshot,
                checkpoint_slots=self.state_slots(
                    cache, req.kv.req_pool_idx, swa_page_id
                ),
                params=getattr(req, "as_kv_cache_params", None) or {},
                release_after_write=release_staging,
                trace_recorder=cache.get_as_trace_recorder(req),
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

    @profile_method("radix.as.load_back", nvtx_enabled=NVTX_OPERATIONS_ENABLED)
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

        if (
            envs.SGLANG_ENABLE_AS_LOAD_BACK_ASYNC.get()
            # PP stages admit independently, so the deferred publish cannot keep
            # the wrapper's cross-stage length check satisfied; sync mode only.
            and cache.pp_world_size <= 1
        ):
            task = self._submit_load_back(cache, req, load_info, anchor)
            # The request stays in the waiting queue this round (admission sees
            # an empty splice); the committed prefix is published by
            # commit_load_backs at the next loop top, and a later admission
            # match_prefix picks it up as an ordinary device hit.
            if task is None:
                # Submit failed and already reported the failure; the request
                # re-prefills from its device-resident prefix this round.
                req._as_pending_load_back = None
            else:
                req._as_pending_load_back = task
            return empty, anchor

        requested = int(load_info["block_count"])
        started = time.perf_counter()
        token_count = requested * cache.page_size
        with operations_nvtx_range("radix.as.lb.evict"):
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
        # The allocator consumes a one-element tensor; keep an existing index on device.
        last_loc = req.prefix_indices[-1].reshape(1) if local_hit_len else None
        with operations_nvtx_range("radix.as.lb.alloc"):
            try:
                loaded = cache.token_to_kv_pool_allocator.alloc_extend_swa_tail(
                    prefix_cpu.to(device),
                    prefix_cpu,
                    seq_cpu.to(device),
                    seq_cpu,
                    (
                        last_loc.to(device=device, dtype=torch.int64)
                        if last_loc is not None
                        else torch.tensor([-1], dtype=torch.int64, device=device)
                    ),
                    token_count,
                    # One whole SWA page, even though a plan may restore only its
                    # upper half: alloc_extend_swa_tail maps the *last* swa_tail_len
                    # full slots onto the freshly allocated SWA page, so asking for
                    # less than a page would land the restored window in the page's
                    # lower half and desynchronize it from the stored byte offset.
                    min(token_count, cache.page_size),
                )
            except Exception as exc:
                cache._log_as_failure(
                    "load_back",
                    rid=req.rid,
                    requested=requested,
                    error=exc,
                    started=started,
                )
                cache._as_min(0)
                return empty, anchor
        if loaded is None:
            cache._log_as_failure(
                "load_back",
                rid=req.rid,
                requested=requested,
                error=RuntimeError(f"cannot allocate {token_count} tokens"),
                started=started,
            )
            cache._as_min(0)
            return empty, anchor
        with operations_nvtx_range("radix.as.lb.prepare"):
            try:
                page_ids, read_args, read_kwargs = self._prepare_read(
                    cache, req, load_info, loaded
                )
            except Exception as exc:
                # Validation may have failed on malformed indices or a missing SWA
                # mapping, so keep the arbitrary-index cleanup for this branch.
                cache.token_to_kv_pool_allocator.free(loaded)
                cache._log_as_failure(
                    "load_back",
                    rid=req.rid,
                    requested=requested,
                    error=exc,
                    started=started,
                )
                cache._as_min(0)
                return empty, anchor
        if cache._as_min(1) == 0:
            self._free_loaded_pages(cache, loaded, start_pos=local_hit_len)
            return empty, anchor
        recorder = cache.get_as_trace_recorder(req)
        digest_capture = recorder.new_digest_capture() if recorder is not None else None
        if digest_capture is not None:
            read_kwargs = {**read_kwargs, "on_digest_snapshot": digest_capture}
        diffuser = getattr(cache, "_as_read_diffuser", None)
        read_error = None
        if diffuser is not None and requested > 1:
            read_count, read_error = self._diffuse_read(
                cache, diffuser, load_info, page_ids, read_args, read_kwargs, requested
            )
        else:
            with operations_nvtx_range("radix.as.lb.read"):
                try:
                    read_count = cache.ascli.read(
                        load_info["hashes"],
                        page_ids,
                        *read_args,
                        timeout=5.0,
                        remote_addrs=load_info.get("remote_addrs") or None,
                        **read_kwargs,
                    )
                except Exception as exc:
                    read_error = exc
                    read_count = 0
        read_count = cache._as_min(read_count)
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
            self._free_loaded_pages(cache, loaded, start_pos=local_hit_len)
            return empty, anchor
        if recorder is not None:
            # SPA passes its checkpoint slot map positionally; SPAG carries the
            # restored SWA window in read_kwargs["swa_ids"] instead.
            state_slots = (
                read_args[0] if read_args and isinstance(read_args[0], dict) else {}
            )
            if "swa_page" in state_slots:
                swa_id_list = [state_slots["swa_page"]]
            else:
                swa_id_list = list(read_kwargs.get("swa_ids") or ())
            recorder.record(
                "AS_READ",
                phase="complete",
                hash_ids=recorder.hash_ids(load_info["hashes"]),
                data_id_list=page_ids,
                swa_id_list=swa_id_list,
                checkpoint_slots=state_slots,
                actual_blocks=read_count,
                start_transfer_block_index=load_info["start_block"],
                md5_after_read=digest_capture.value(),
                status="ok",
            )

        total_prefix_len = local_hit_len + token_count
        with operations_nvtx_range("radix.as.lb.insert"):
            token_ids = req.full_untruncated_fill_ids[:total_prefix_len]
            values = torch.cat([req.prefix_indices, loaded]).to(dtype=torch.int64)
            restored_swa_len = self._restored_swa_len(cache, token_count)
            cache.insert(
                InsertParams(
                    key=RadixKey(
                        token_ids, req.extra_key, is_bigram=cache.tree_core.is_eagle
                    ),
                    value=values,
                    prev_prefix_len=local_hit_len,
                    # Everything before the restored window is an SWA tombstone. This
                    # must describe the bytes actually read back, not the SWA pages
                    # allocated: over-claiming makes the tree serve uninitialized SWA.
                    swa_evicted_seqlen=local_hit_len
                    + max(0, token_count - restored_swa_len),
                )
            )
            match = UnifiedRadixCache.match_prefix(
                cache,
                MatchPrefixParams(
                    key=RadixKey(
                        token_ids, req.extra_key, is_bigram=cache.tree_core.is_eagle
                    ),
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
        req.as_read_pending_lock = True
        return tree_indices, match.last_device_node

    @profile_method("radix.as.diffuse", nvtx_enabled=NVTX_OPERATIONS_ENABLED)
    def _diffuse_read(
        self, cache, diffuser, load_info, page_ids, read_args, read_kwargs, requested
    ):
        """Sharded L2 read + NVLink diffuse (returns (read_count, error)).

        Each CP rank reads only its 1/N interleaved shard plus the terminal
        block (which carries the SWA window, so every rank restores it locally),
        then pulls the remaining peer-owned blocks over NVLink. The ``_as_min``
        that agrees on shard success is itself a CP-group collective *after* the
        host-synchronous shard reads, so it doubles as the pre-diffuse barrier
        (every peer's H2D has landed) -- diffuse runs with ``peers_ready=True``.
        """
        terminal = requested - 1
        read_idx = sorted(set(diffuser.local_shard(requested)) | {terminal})
        slots = [int(x) for x in page_ids]
        shard_hashes = [load_info["hashes"][k] for k in read_idx]
        shard_page_ids = [slots[k] for k in read_idx]
        read_error = None
        try:
            got = cache.ascli.read(
                shard_hashes,
                shard_page_ids,
                *read_args,
                timeout=5.0,
                remote_addrs=load_info.get("remote_addrs") or None,
                **read_kwargs,
            )
            shard_ok = int(got == len(read_idx))
        except Exception as exc:  # noqa: BLE001 -- reduced across the CP group
            read_error = exc
            shard_ok = 0
        if int(cache._as_min(shard_ok)) == 0:
            return 0, read_error
        diffuser.diffuse(
            slots, requested, extra_local_blocks=(terminal,), peers_ready=True
        )
        # Verify the peer-pulled blocks landed correctly (digest-gated no-op
        # otherwise): re-hash the FULL restored set, compared to write by hash.
        cache.ascli.emit_content_digest("postdiffuse", load_info["hashes"], slots)
        return requested, read_error

    def do_write(self, cache, task: _SPAWriteTask) -> int:
        """Write SPA pages followed by their terminal checkpoint."""
        if not task.block_hash_ids:
            return 0
        started = time.perf_counter()
        transfer_start = 0
        transfer_hashes = ()
        transfer_page_ids = ()
        try:
            page_ids = task.page_ids.materialize()
            # RAW fix (align with KDACacheCoordinator.do_write / commit 763d75172f):
            # record the producer fence here, on the writer thread right before the
            # gather, targeting the injected forward_stream -- not the scheduler's
            # current stream. _BufferPageIO.write waits on it before reading pages.
            producer_ready = self.record_device_event(
                cache.device, self._producer_stream(cache)
            )
            digest_capture = (
                task.trace_recorder.new_digest_capture()
                if task.trace_recorder is not None
                else None
            )
            confirmed_pages = task.page_endpoint.store.match(
                task.block_hash_ids,
                timeout=1.0,
            )
            confirmed_pages = max(0, min(confirmed_pages, task.requested_blocks))
            if confirmed_pages < task.requested_blocks:
                transfer_start = confirmed_pages
                transfer_hashes = task.block_hash_ids[transfer_start:]
                transfer_page_ids = page_ids[transfer_start:]
                task.page_endpoint.store.write(
                    transfer_hashes,
                    transfer_page_ids,
                    timeout=1.0,
                    params=task.params,
                    producer_ready=producer_ready,
                    on_digest_snapshot=digest_capture,
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
                        on_digest_snapshot=digest_capture,
                    )
                result = confirmed_pages if checkpoint_result == 1 else -1
            cache._log_as_result(
                "write",
                rid=task.rid,
                requested=task.requested_blocks,
                result=result,
                started=started,
            )
            if task.trace_recorder is not None and transfer_hashes:
                task.trace_recorder.record(
                    "AS_WRITE",
                    phase="complete",
                    task_id=id(task),
                    hash_ids=task.trace_recorder.hash_ids(transfer_hashes),
                    data_id_list=transfer_page_ids,
                    swa_id_list=[task.checkpoint_slots["swa_page"]]
                    if "swa_page" in task.checkpoint_slots
                    else [],
                    checkpoint_slots=task.checkpoint_slots,
                    start_transfer_block_index=transfer_start,
                    md5_before_write=digest_capture.value(),
                    status="ok" if result >= task.requested_blocks else "partial",
                )
            return result
        except Exception as exc:
            if task.trace_recorder is not None and transfer_hashes:
                task.trace_recorder.record(
                    "AS_WRITE",
                    phase="complete",
                    task_id=id(task),
                    hash_ids=task.trace_recorder.hash_ids(transfer_hashes),
                    data_id_list=[],
                    swa_id_list=[task.checkpoint_slots["swa_page"]]
                    if "swa_page" in task.checkpoint_slots
                    else [],
                    checkpoint_slots=task.checkpoint_slots,
                    start_transfer_block_index=transfer_start,
                    status="error",
                    error=str(exc),
                )
            cache._log_as_failure(
                "write",
                rid=task.rid,
                requested=task.requested_blocks,
                error=exc,
                started=started,
            )
            return -1

    def enqueue_dummy(self, cache, rid: str) -> None:
        """Queue a no-op SPA task to preserve TP retirement ordering."""
        if cache.as_world_size <= 1:
            return
        self.submit_completed(
            _SPAWriteTask(
                rid=rid,
                page_endpoint=None,
                checkpoint_store=None,
                block_hash_ids=(),
                page_ids=(),
                checkpoint_snapshot=None,
                checkpoint_slots={},
                params={},
            )
        )


class SPAGCacheCoordinator(SPACacheCoordinator):
    """Coordinate SPAG page writes whose blocks carry their own SWA window.

    SPAG has no checkpoint endpoint. Every page block reserves a trailing SWA
    region and the blocks that close a ``swa_step`` checkpoint fill it from the
    request's ``ChunkTailSWAPool`` slots, so a matched page range is directly
    restorable instead of depending on a separate blob keyed on the range's final
    page.

    Slot ownership (the reason no eager pre-commit snapshot is needed here, in
    contrast to SPA's WAR in ``prepare_write``): the window was already copied
    out to the staging pool at the chunk boundary that produced it
    (``Scheduler.detect_and_may_copy_as_swa``), so nothing the local commit frees
    can invalidate it. ``req.as_swa_kv_slots`` is split at
    ``req.as_written_count``: below it the slots belong to a submitted write task
    and are released when that task retires, at or above it they belong to the
    request and are released with it.
    """

    write_thread_name = "as_spag_writer"

    def prepare_page_ids(self, cache, slot_ids):
        """Borrow validated-layout GPU slots instead of Python metadata staging."""
        if (
            os.environ.get("ATTENTIONSTORE_NATIVE_WRITE") == "1"
            and type(self) is SPAGCacheCoordinator
            and getattr(cache, "pp_size", 1) == 1
            and self._native_physical_writer(cache)
            and isinstance(slot_ids, torch.Tensor)
            and slot_ids.is_cuda
            and slot_ids.dtype == torch.int64
            and slot_ids.ndim == 1
            and slot_ids.is_contiguous()
        ):
            from .coordinator import _reap_quarantined_copies

            _reap_quarantined_copies()
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(slot_ids.device))
            return _NativeWriteSlots(cache.page_size, slot_ids, ready)
        return super().prepare_page_ids(cache, slot_ids)

    @staticmethod
    def _native_physical_writer(cache):
        from sglang.srt.mem_cache.allocator.swa import (
            SWATokenToKVPoolAllocator,
        )
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
        )
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        allocator = cache.token_to_kv_pool_allocator
        if (
            type(allocator) is not SWATokenToKVPoolAllocator
            or allocator.swa_req_ring
        ):
            return False
        pool = allocator.get_kvcache()
        return (
            type(pool) in (DeepSeekV4TokenToKVPool, SWAKVPool)
            and "translate_loc_from_full_to_swa" not in vars(allocator)
            and "translate_loc_from_full_to_swa" not in vars(pool)
        )

    def _submit_native_write(self, cache, task):
        if (
            os.environ.get("ATTENTIONSTORE_NATIVE_WRITE") != "1"
            or type(self) is not SPAGCacheCoordinator
            or getattr(cache, "pp_size", 1) != 1
            or not isinstance(task.page_ids, _NativeWriteSlots)
        ):
            return None
        submit = getattr(
            task.page_endpoint.store, "submit_write_slots_native", None
        )
        if submit is None:
            return None
        producer = task.producer_ready
        if producer is None:
            producer = self.record_device_event(
                cache.device, self._producer_stream(cache)
            )
        chunks = self._write_chunks(cache, task, 0)
        future = submit(
            task.block_hash_ids,
            task.page_ids,
            chunks,
            producer,
            attempts=2 if self._reuse_finished_write else 1,
            params=task.params,
            timeout=1.0,
        )
        if future is None:
            return None
        return _NativeWriterResult(cache, task, future)

    def __init__(self) -> None:
        super().__init__()
        self._reuse_finished_write = envs.SGLANG_AS_SPAG_REUSE_FINISHED_WRITE.get()
        self._batch_write_fence = envs.SGLANG_AS_SPAG_BATCH_WRITE_FENCE.get()

    @staticmethod
    def _write_signature(task):
        return tuple(task.block_hash_ids), dict(task.params)

    def _can_reuse_finished_write(self, cache, req, token_ids) -> bool:
        if (
            not self._reuse_finished_write
            or cache.pp_size > 1
            or req.as_spag_submitted_write is None
        ):
            return False
        page_count = len(token_ids) // cache.page_size
        hashes = _request_block_hash_ids(req, token_ids, cache.page_size, page_count)
        return req.as_spag_submitted_write == (
            hashes,
            dict(req.as_kv_cache_params or {}),
        )

    def submit_write(self, cache, task) -> None:
        """Queue a write and remember its signature when reuse is enabled.

        Args:
            cache: Cache instance supplying allocation, rank gates and
                diagnostics.
            task: Write task holding its resource owners and requested block
                count.

        Returns:
            None.

        Raises:
            Exception: Submission failed; the base coordinator cleans up the
                task.
        """
        super().submit_write(cache, task)
        if self._reuse_finished_write and cache.pp_size == 1:
            task.request.as_spag_submitted_write = self._write_signature(task)

    def _run_write(self, cache, task) -> int:
        if not self._reuse_finished_write or cache.pp_size > 1:
            return super()._run_write(cache, task)
        try:
            result = self.do_write(cache, task)
            if result < task.requested_blocks:
                result = self.do_write(cache, task)
            return result
        finally:
            task.release_after_write()

    def finish(self, cache, task, result: int) -> None:
        """Clear the remembered write signature after an incomplete write.

        Args:
            cache: Cache instance supplying allocation, rank gates and
                diagnostics.
            task: Write task holding its resource owners and requested block
                count.
            result: Existing operation result to record or finalize.

        Returns:
            None.
        """
        if (
            task.request is not None
            and result < task.requested_blocks
            and task.request.as_spag_submitted_write == self._write_signature(task)
        ):
            task.request.as_spag_submitted_write = None

    # prepare_write only checks the captured slot count. The writer's page IDs
    # come from the protected tree indices after commit, so these views need no
    # copy even if the request's slot mapping changes during the local commit.
    def capture_finished(
        self, cache, req, is_insert: bool
    ) -> AttentionStoreWriteContext:
        """Capture the committed slot range without copying its indices."""
        committed_len = req.effective_kv_committed_len()
        token_ids = (req.origin_input_ids + req.output_ids)[:committed_len]
        snapshot = cache.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, :committed_len
        ]
        return AttentionStoreWriteContext(
            req=req,
            token_ids=token_ids,
            snapshot=snapshot,
            is_insert=is_insert
            and not self._can_reuse_finished_write(cache, req, token_ids),
        )

    def capture_unfinished(self, cache, req) -> AttentionStoreWriteContext:
        """Capture the filled slot range without copying its indices."""
        token_ids = req.get_fill_ids()
        snapshot = cache.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, : len(token_ids)
        ]
        return AttentionStoreWriteContext(
            req=req,
            token_ids=token_ids,
            snapshot=snapshot,
            is_insert=True,
        )

    @staticmethod
    def _swa_policy(cache):
        """Return the engine's ``SwaStepPolicy``, or ``None`` when SWA is off."""
        kvcache = cache.token_to_kv_pool_allocator.get_kvcache()
        if getattr(kvcache, "chunk_tail_swa_pool", None) is None:
            return None
        return getattr(kvcache, "swa_policy", None)

    def state_slots(self, cache, req_pool_idx: int, swa_page_id: int) -> dict[str, int]:
        """Unreachable: SPAG stores no separate checkpoint blob."""
        raise NotImplementedError(
            "SPAG has no checkpoint endpoint; SWA rides inside the page block"
        )

    def floor_remote_terminal(self, cache, terminal_block: int) -> int:
        """Shrink the match to the last page that closes a checkpoint.

        Only checkpoint blocks carry a SWA window, so a match ending mid-stride
        would restore pages whose trailing window is a sentinel hole. The tree's
        SWA validator would then reject the range anyway, after the read already
        paid for it.
        """
        policy = self._swa_policy(cache)
        if policy is None:
            return terminal_block
        return policy.floor_to_checkpoint(terminal_block * cache.page_size) // (
            cache.page_size
        )

    def _prepare_read(self, cache, req, load_info, loaded):
        """Resolve page IDs and the trailing SWA window with one D2H transfer.

        ``floor_remote_terminal`` guarantees the restored range ends on a
        checkpoint, so exactly one block of this read carries a window: the last
        one. ``swa_block_step`` only has to place the real offset on that block,
        and it is the same stride the write used for that checkpoint.
        """
        policy = self._swa_policy(cache)
        if policy is None:
            return _contiguous_page_ids(loaded, cache.page_size), (), {}
        page_meta = _page_id_metadata(loaded, cache.page_size)
        # Preserve validation before an unsafe mapping lookup: malformed pages
        # use the padding slot until the host validator reports the error.
        tail_slot = torch.where(
            page_meta[:, 1].all(), loaded[-cache.page_size :][:1], 0
        )
        swa_index = cache.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
            tail_slot
        )
        # Both computations stay on the current device stream. Materialize them
        # together so page validation and SWA lookup share one synchronization.
        host_meta = torch.cat((page_meta.reshape(-1), swa_index)).cpu()
        page_ids = _materialize_page_ids(host_meta[:-1].reshape(-1, 2), cache.page_size)
        swa_page_id = int(host_meta[-1].item()) // cache.page_size
        if swa_page_id == 0:
            # Page 0 is the padding sink every cleared full->swa entry maps to,
            # so a real fresh allocation never lands there. Reading into it would
            # silently corrupt the sink instead of restoring anything.
            raise RuntimeError(
                "SPAG load-back resolved SWA page 0 (the padding sink); "
                "the full->swa mapping for the restored tail was cleared"
            )
        end_pos = (
            int(load_info["start_block"]) + int(load_info["block_count"])
        ) * cache.page_size
        return (
            page_ids,
            (),
            {
                "swa_ids": (swa_page_id,),
                "swa_block_step": policy.block_step_for_position(end_pos),
            },
        )

    def _restored_swa_len(self, cache, token_count: int) -> int:
        """Keep the inherited page granularity.

        SPAG stores one ``swa_win`` window per checkpoint, i.e. only the upper
        half of the final SWA page comes back. Reporting the whole page is still
        correct -- and the only page-aligned option -- for the reasons in the base
        docstring.
        """
        return super()._restored_swa_len(cache, token_count)

    def _collect_swa_slots(self, cache, req, cache_len: int):
        """Map this write's checkpoints to staging slots, claiming the new ones.

        The returned tuple is indexed by checkpoint number from the *request's*
        start, because the SDK derives a block's checkpoint from its index alone.
        Entries this task does not own are ``-1``:

        * below ``as_written_count`` an earlier task owns (and will free) the
          slot, so reusing the id here could hand a recycled window to the Agent;
          the block itself is already stored with its window anyway.
        * at or above ``as_swa_copy_count`` nothing has been staged yet.
        """
        policy = self._swa_policy(cache)
        if policy is None:
            return (), 0
        total = policy.num_checkpoints(cache_len)
        if total <= 0:
            return (), 0
        staged = req.as_swa_kv_slots
        start = min(int(req.as_written_count), total, len(staged))
        end = min(int(req.as_swa_copy_count), total, len(staged))
        slots = [-1] * total
        for index in range(start, end):
            slots[index] = int(staged[index])
        return tuple(slots), max(start, end)

    def prepare_write(
        self,
        cache,
        req,
        token_ids: Sequence[int],
        snapshot: torch.Tensor,
    ) -> Optional[_SPAWriteTask]:
        """Prepare page identifiers and the per-checkpoint SWA slot map."""
        if cache.ascli is None or not token_ids:
            return None
        # Offload the page-aligned prefix even when the fill length carries a
        # partial tail page; the radix tree only publishes full pages.
        cache_len = (len(token_ids) // cache.page_size) * cache.page_size
        if cache_len == 0 or len(snapshot) < cache_len:
            return None
        requested = 0
        try:
            hashes = _request_block_hash_ids(
                req,
                token_ids,
                cache.page_size,
                cache_len // cache.page_size,
            )
            requested = len(hashes)
            swa_slot_ids, owned_end = self._collect_swa_slots(cache, req, cache_len)
            # page_ids are deliberately NOT built from this pre-commit snapshot:
            # super().cache_*_req frees the duplicate prefix slots it references
            # before the async gather runs. after_local_commit builds them from
            # the tree's ref-protected match.device_indices instead.
            return _SPAWriteTask(
                rid=req.rid,
                page_endpoint=cache.binding.page_endpoint,
                checkpoint_store=None,
                block_hash_ids=hashes,
                page_ids=None,
                checkpoint_snapshot=None,
                params=getattr(req, "as_kv_cache_params", None) or {},
                swa_slot_ids=swa_slot_ids,
                swa_owned_end=owned_end,
                request=req,
                trace_recorder=cache.get_as_trace_recorder(req),
            )
        except Exception as exc:
            cache._log_as_failure(
                "write",
                rid=req.rid,
                requested=requested,
                error=exc,
            )
            return None

    def _refloor_task(self, cache, context, task, match, cache_len: int) -> None:
        """Shrink the task to the published prefix, checkpoints included."""
        task.block_hash_ids = task.block_hash_ids[: cache_len // cache.page_size]
        policy = self._swa_policy(cache)
        if policy is None:
            return
        total = policy.num_checkpoints(cache_len)
        # Checkpoints past the published prefix stay with the request: their
        # staged windows are still valid, so the next chunk's write re-claims
        # them rather than dropping them.
        task.swa_slot_ids = task.swa_slot_ids[:total]
        task.swa_owned_end = min(task.swa_owned_end, total)

    def _commit_write_ownership(self, cache, context, task) -> None:
        """Advance the request's ownership frontier to this task's checkpoints."""
        req = context.req
        req.as_written_count = max(int(req.as_written_count), int(task.swa_owned_end))

    def _make_release_after_write(self, cache, task, page_ids) -> Callable[[], None]:
        """Also return the staged SWA windows this task owned."""
        base = super()._make_release_after_write(cache, task, page_ids)
        if self._batch_write_fence and task.request.as_kv_ready_event is not None:
            stream = torch.cuda.current_stream(cache.device)
            stream.wait_event(task.request.as_kv_ready_event)
            task.producer_ready = self.record_device_event(cache.device, stream)
        kvcache = cache.token_to_kv_pool_allocator.get_kvcache()
        # Only slots this task claimed are non-sentinel (see _collect_swa_slots).
        slots = tuple(slot for slot in task.swa_slot_ids if slot != -1)

        released = False

        def _release_staging() -> None:
            nonlocal released
            try:
                base()
            finally:
                if not released:
                    released = True
                    free_chunk_tail_swa_slots(kvcache, slots)

        return _release_staging

    def _write_chunks(self, cache, task, confirmed_pages: int):
        """Split the unwritten block range into batches the SDK can address.

        The SDK decides which block carries a checkpoint from the block's index
        *within the batch* (``(idx + 1) % swa_block_step == 0``) and takes a
        single scalar step per call. So a batch must start on the checkpoint grid
        and may not straddle a tier boundary. ``confirmed_pages`` is floored onto
        the grid of the run it lands in: re-storing a few already-stored pages is
        idempotent, whereas starting mid-stride would shift every checkpoint of
        the batch onto the wrong block.
        """
        page_count = task.requested_blocks
        policy = self._swa_policy(cache)
        if policy is None or not task.swa_slot_ids:
            return [(confirmed_pages, page_count, (), None)]
        page_size = cache.page_size
        chunks = []
        covered = 0
        for first, last, block_step in policy.segments(0, len(task.swa_slot_ids)):
            start = policy.checkpoint_pos(first) // page_size - block_step
            end = policy.checkpoint_pos(last - 1) // page_size
            covered = end
            if end <= confirmed_pages:
                continue
            if start < confirmed_pages:
                whole_strides = (confirmed_pages - start) // block_step
                first += whole_strides
                start += whole_strides * block_step
            chunks.append((start, end, task.swa_slot_ids[first:last], block_step))
        tail_start = max(covered, confirmed_pages)
        if tail_start < page_count:
            # Trailing blocks close no checkpoint, so they carry no SWA ids and
            # the SDK sentinels their whole SWA region.
            chunks.append((tail_start, page_count, (), None))
        return chunks

    @profile_method("radix.as.write", nvtx_enabled=NVTX_OPERATIONS_ENABLED)
    def do_write(self, cache, task: _SPAWriteTask) -> int:
        """Write SPAG page blocks together with their embedded SWA windows."""
        if not task.block_hash_ids:
            return 0
        started = time.perf_counter()
        transfer_start = 0
        transfer_page_ids = ()
        transfer_swa_ids = []
        try:
            page_ids = task.page_ids.materialize()
            producer_ready = task.producer_ready
            if producer_ready is None:
                producer_ready = self.record_device_event(
                    cache.device, self._producer_stream(cache)
                )
            digest_capture = (
                task.trace_recorder.new_digest_capture()
                if task.trace_recorder is not None
                else None
            )
            store = task.page_endpoint.store
            confirmed_pages = self._match_pages(store, task)
            if confirmed_pages < task.requested_blocks:
                chunks = self._write_chunks(cache, task, confirmed_pages)
                transfer_start = chunks[0][0] if chunks else confirmed_pages
                transfer_page_ids = page_ids[transfer_start:]
                for start, end, swa_ids, block_step in chunks:
                    transfer_swa_ids.extend(int(x) for x in swa_ids)
                    store.write(
                        task.block_hash_ids[start:end],
                        page_ids[start:end],
                        timeout=1.0,
                        params=task.params,
                        producer_ready=producer_ready,
                        swa_ids=swa_ids,
                        swa_block_step=block_step,
                        logical_start=start,
                        on_digest_snapshot=digest_capture,
                    )
                confirmed_pages = self._match_pages(store, task)
            cache._log_as_result(
                "write",
                rid=task.rid,
                requested=task.requested_blocks,
                result=confirmed_pages,
                started=started,
            )
            if task.trace_recorder is not None and transfer_page_ids:
                task.trace_recorder.record(
                    "AS_WRITE",
                    phase="complete",
                    task_id=id(task),
                    hash_ids=task.trace_recorder.hash_ids(
                        task.block_hash_ids[transfer_start:]
                    ),
                    data_id_list=transfer_page_ids,
                    swa_id_list=transfer_swa_ids,
                    start_transfer_block_index=transfer_start,
                    md5_before_write=digest_capture.value(),
                    status=(
                        "ok"
                        if confirmed_pages >= task.requested_blocks
                        else "partial"
                    ),
                )
            return confirmed_pages
        except Exception as exc:
            if task.trace_recorder is not None and transfer_page_ids:
                task.trace_recorder.record(
                    "AS_WRITE",
                    phase="complete",
                    task_id=id(task),
                    hash_ids=task.trace_recorder.hash_ids(
                        task.block_hash_ids[transfer_start:]
                    ),
                    data_id_list=[],
                    swa_id_list=[],
                    start_transfer_block_index=transfer_start,
                    status="error",
                    error=str(exc),
                )
            cache._log_as_failure(
                "write",
                rid=task.rid,
                requested=task.requested_blocks,
                error=exc,
                started=started,
            )
            return -1

    @staticmethod
    def _match_pages(store, task: _SPAWriteTask) -> int:
        """Return how much of this task's page range the store already holds."""
        matched = store.match(task.block_hash_ids, timeout=1.0)
        return max(0, min(matched, task.requested_blocks))


class KDACacheCoordinator(CheckpointedPageCacheCoordinator):
    """Coordinate full-attention pages and recurrent KDA endpoint checkpoints."""

    write_thread_name = "as_kda_writer"

    @staticmethod
    def _trace_write_skip(cache, req, reason: str, **fields) -> None:
        """Record why no KDA write was submitted, so the trace is not silent."""
        recorder = cache.get_as_trace_recorder(req)
        if recorder is not None:
            recorder.record("AS_WRITE", status="skipped", reason=reason, **fields)

    def capture_finished(
        self, cache, req, is_insert: bool
    ) -> AttentionStoreWriteContext:
        """Snapshot KDA page slots for a finished request."""
        token_ids = array("q", req.origin_input_ids)
        token_ids.extend(req.output_ids)
        snapshot = cache.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, : len(token_ids)
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
            req.kv.req_pool_idx, : len(token_ids)
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
        checkpoint_slot = None
        page_ids = None
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
                cache_len < cache.attention_store_block_size
                or cache_len % cache.attention_store_block_size != 0
                or node is None
                or mamba_data.value is None
                or len(snapshot) < cache_len
            ):
                self._trace_write_skip(
                    cache, req, "no_aligned_checkpoint", cache_len=cache_len
                )
                self.enqueue_dummy(cache, req.rid)
                return

            block_size = cache.attention_store_block_size
            page_count = cache_len // block_size
            hashes = _request_block_hash_ids(
                req,
                token_ids,
                block_size,
                page_count,
            )
            endpoint_hash = hashes[-1]
            if mamba_data.metadata.get(_KDA_PENDING_HASH) is not None or (
                endpoint_hash == mamba_data.metadata.get(_KDA_PUBLISHED_HASH)
            ):
                self._trace_write_skip(
                    cache,
                    req,
                    "write_pending"
                    if mamba_data.metadata.get(_KDA_PENDING_HASH) is not None
                    else "already_published",
                    cache_len=cache_len,
                )
                self.enqueue_dummy(cache, req.rid)
                return
            write_key = (cache_len, endpoint_hash)
            if getattr(req, "_as_kda_last_submitted_write", None) == write_key:
                self._trace_write_skip(
                    cache, req, "already_submitted", cache_len=cache_len
                )
                self.enqueue_dummy(cache, req.rid)
                return
            checkpoint_slot = self.prepare_checkpoint_slot(mamba_data.value)
            # WAR fix: gather from the tree's canonical, ref-protected slots
            # (match.device_indices), NOT the pre-commit request snapshot.
            # super().cache_finished_req (run between capture and this enqueue)
            # frees the duplicate prefix slots kv_indices[cache_protected_len:
            # prefix_len] that live in `snapshot`; a later forward reuses and
            # overwrites them before the async writer's gather runs, so gathering
            # from `snapshot` reads corrupted (reused) KV. inc_lock_ref(node)
            # below protects device_indices, so page_ids must come from there.
            page_ids = self.prepare_page_ids(cache, match.device_indices)
            # Producer fence is recorded in do_write right before the gather; an
            # enqueue-time event is too early to cover a still-draining write.
            producer_ready = None

            def release_staging() -> None:
                page_ids.release()
                checkpoint_slot.release()

            task = _KDAWriteTask(
                rid=req.rid,
                node=node,
                block_hash_ids=hashes,
                page_ids=page_ids,
                checkpoint_slot=checkpoint_slot,
                producer_ready=producer_ready,
                params=getattr(req, "as_kv_cache_params", None) or {},
                request=req,
                write_key=write_key,
                release_after_write=release_staging,
                trace_recorder=cache.get_as_trace_recorder(req),
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
            else:
                if page_ids is not None:
                    page_ids.release()
                if checkpoint_slot is not None:
                    checkpoint_slot.release()
            cache._log_as_failure(
                "write",
                rid=req.rid,
                requested=cache_len // cache.attention_store_block_size,
                error=exc,
            )
            self.enqueue_dummy(cache, req.rid)

    def do_write(self, cache, task: _KDAWriteTask) -> int:
        """Write KDA pages and recurrent state through the composite store."""
        started = time.perf_counter()
        transfer_hashes = ()
        transfer_page_ids = ()
        checkpoint_slot = None
        try:
            page_ids = task.page_ids.materialize()
            # Record the producer fence here (writer thread, right before the
            # gather) capturing the producer stream's current tail, so a late
            # page write not covered by an enqueue-time event is still fenced.
            producer_ready = self.record_device_event(
                cache.device, self._producer_stream(cache)
            )
            checkpoint_slot = task.checkpoint_slot.materialize()
            digest_capture = (
                task.trace_recorder.new_digest_capture()
                if task.trace_recorder is not None
                else None
            )
            written = cache.ascli.write(
                task.block_hash_ids,
                page_ids,
                {"checkpoint": checkpoint_slot},
                timeout=1.0,
                params=task.params,
                producer_ready=producer_ready,
                on_digest_snapshot=digest_capture,
            )
            transfer_hashes = task.block_hash_ids
            transfer_page_ids = page_ids
            if task.trace_recorder is not None:
                task.trace_recorder.record(
                    "AS_WRITE",
                    phase="complete",
                    task_id=id(task),
                    hash_ids=task.trace_recorder.hash_ids(transfer_hashes),
                    data_id_list=transfer_page_ids,
                    checkpoint_slots={"checkpoint": checkpoint_slot},
                    start_transfer_block_index=0,
                    md5_before_write=digest_capture.value(),
                    status="ok",
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
            if task.trace_recorder is not None:
                task.trace_recorder.record(
                    "AS_WRITE",
                    phase="complete",
                    task_id=id(task),
                    hash_ids=task.trace_recorder.hash_ids(task.block_hash_ids),
                    data_id_list=[],
                    checkpoint_slots=(
                        {"checkpoint": checkpoint_slot}
                        if checkpoint_slot is not None
                        else {}
                    ),
                    start_transfer_block_index=0,
                    status="error",
                    error=str(exc),
                )
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

        loaded = self._read_blocks(
            cache, load_info, cache.get_as_trace_recorder(req)
        )
        if loaded is None:
            return empty, anchor
        loaded_indices, value_transaction, checkpoint_allocation = loaded
        req.as_read_pending_lock = True
        return self._insert_loaded(
            cache,
            req,
            anchor,
            load_info,
            loaded_indices,
            value_transaction,
            checkpoint_allocation,
        )

    def _read_blocks(self, cache, load_info, recorder=None):
        block_count = int(load_info["block_count"])
        rid = load_info.get("rid", "-")
        started = time.perf_counter()
        block_size = cache.attention_store_block_size
        token_count = block_count * block_size
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
            cache._as_min(0)
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
            cache._as_min(0)
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
            cache._as_min(0)
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
            cache._as_min(0)
            return None
        keep_transaction = False
        try:
            host_ids = checkpoint_allocation.host_value
            if host_ids is None or len(host_ids) != 1:
                raise RuntimeError(
                    "AttentionStore KDA load-back requires a host-side checkpoint "
                    "slot ID; refusing to scalarize the scheduler GPU tensor"
                )
            state_id = int(host_ids[0])
            if cache._as_min(1) == 0:
                return None
            digest_capture = (
                recorder.new_digest_capture() if recorder is not None else None
            )
            read_error = None
            try:
                read_count = cache.ascli.read(
                    load_info["hashes"],
                    page_ids,
                    {"checkpoint": state_id},
                    timeout=5.0,
                    remote_addrs=load_info.get("remote_addrs") or None,
                    on_digest_snapshot=digest_capture,
                )
            except Exception as exc:
                read_error = exc
                read_count = 0

            read_count = cache._as_min(read_count)

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
            if recorder is not None:
                recorder.record(
                    "AS_READ",
                    phase="complete",
                    hash_ids=recorder.hash_ids(load_info["hashes"]),
                    data_id_list=page_ids,
                    checkpoint_slots={"checkpoint": state_id},
                    actual_blocks=read_count,
                    start_transfer_block_index=load_info["start_block"],
                    md5_after_read=digest_capture.value(),
                    status="ok",
                )
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
        local_hit_len = load_info["start_block"] * cache.attention_store_block_size
        total_prefix_len = local_hit_len + len(loaded_indices)
        if len(req.prefix_indices) != local_hit_len:
            logger.error(
                "AS cache insert rejected: cache_plan=%s rid=%s "
                "reason=prefix_length_mismatch expected=%d actual=%d",
                cache.attention_store_plan.name,
                req.rid,
                local_hit_len,
                len(req.prefix_indices),
            )
            value_transaction.close()
            return loaded_indices[:0], anchor
        try:
            token_ids = req.full_untruncated_fill_ids[:total_prefix_len]
            values = torch.cat([req.prefix_indices, loaded_indices]).to(
                dtype=torch.int64, copy=True
            )
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

    def enqueue_dummy(self, cache, rid: str) -> None:
        """Queue a no-op KDA task to preserve TP retirement ordering."""
        if cache.as_world_size <= 1:
            return
        self.submit_completed(
            _KDAWriteTask(
                rid=rid,
                node=cache.root_node,
                block_hash_ids=(),
                page_ids=(),
                checkpoint_slot=self.prepare_checkpoint_slot(0),
                params={},
            )
        )
