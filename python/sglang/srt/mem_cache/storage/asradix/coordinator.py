"""Shared asynchronous lifecycle for AttentionStore coordinators."""

from __future__ import annotations

import concurrent.futures
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    InitLoadBackParams,
    MatchPrefixParams,
    MatchResult,
)

from .transport import _once
from .types import AttentionStoreWriteContext

logger = logging.getLogger(__name__)


def _noop_release() -> None:
    return


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
        *,
        synchronize: bool = True,
    ) -> None:
        first_error = None
        for pending in pending_writes:
            result = self._resolve_result(cache, pending, timeout)
            if synchronize:
                result = cache._tp_min(result)
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
        """Retire the completed write prefix shared by local attention ranks."""
        if not self._pending_writes:
            return
        completed = 0
        for pending in self._pending_writes:
            if not pending.future.done():
                break
            completed += 1
        completed = cache._tp_min(completed)
        if not completed:
            return
        pending_writes = self._pending_writes[:completed]
        del self._pending_writes[:completed]
        self._retire(cache, pending_writes, timeout=1.0)

    def drain(self, cache, *, synchronize: bool = True) -> None:
        """Wait for and retire every pending write."""
        pending_writes = self._pending_writes
        self._pending_writes = []
        if not pending_writes:
            return
        common_count = (
            cache._tp_min(len(pending_writes)) if synchronize else len(pending_writes)
        )
        extra_writes = pending_writes[common_count:]
        pending_writes = pending_writes[:common_count]
        # A timed-out writer is still allowed to touch its page buffers.  A
        # reset or shutdown therefore waits for quiescence before releasing
        # retained radix nodes.
        self._retire(
            cache,
            pending_writes,
            timeout=None,
            synchronize=synchronize,
        )
        for pending in extra_writes:
            self._resolve_result(cache, pending, timeout=None)
            try:
                self.finish(cache, pending.task, -1)
            finally:
                self._release_task_resource(
                    cache,
                    pending.task,
                    pending.task.release,
                    "retained resource",
                )

    def before_reset(self, cache) -> None:
        """Drain pending writes before resetting the cache."""
        self.drain(cache)

    def before_sanity_check(self, cache) -> None:
        """Drain pending writes before checking cache invariants."""
        self.drain(cache)

    def shutdown(self, cache) -> None:
        """Drain writes and stop the background executor."""
        try:
            # atexit is not ordered across PP processes. Agent only publishes a
            # block after every shard is complete, so local retirement is safe.
            self.drain(cache, synchronize=False)
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
