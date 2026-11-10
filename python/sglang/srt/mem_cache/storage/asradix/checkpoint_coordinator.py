"""Coordinator implementations for checkpointed-page cache plans."""

from __future__ import annotations

import logging
import time
from array import array
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InitLoadBackParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import (
    UnifiedRadixCache,
    UnifiedTreeNode,
)

from .checkpoint_binding import PackedStateSnapshot
from .coordinator import (
    CheckpointedPageCacheCoordinator,
    _acquire_node_release,
    _contiguous_page_ids,
    _noop_release,
    _request_block_hash_ids,
)
from .types import (
    AttentionStoreEndpointBinding,
    AttentionStoreWriteContext,
)

logger = logging.getLogger(__name__)

_KDA_PENDING_HASH = "as_kda_pending_hash"
_KDA_PUBLISHED_HASH = "as_kda_published_hash"


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


class SPACacheCoordinator(CheckpointedPageCacheCoordinator):
    """Coordinate SPA page writes with a stable endpoint checkpoint snapshot."""

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
            self.enqueue_dummy(cache, context.req.rid)
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
            self.enqueue_dummy(cache, context.req.rid)

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
        try:
            loaded = cache.token_to_kv_pool_allocator.alloc_extend_swa_tail(
                prefix_cpu.to(device),
                prefix_cpu,
                seq_cpu.to(device),
                seq_cpu,
                torch.tensor([last_loc_value], dtype=torch.int64, device=device),
                token_count,
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
            cache._as_min(0)
            return empty, anchor
        try:
            swa_indices = (
                cache.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                    loaded[-cache.page_size :]
                )
            )
            swa_page_id = int(swa_indices[0].item()) // cache.page_size
            state_slots = self.state_slots(cache, req.req_pool_idx, swa_page_id)
        except Exception as exc:
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
            cache.token_to_kv_pool_allocator.free(loaded)
            return empty, anchor
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
                params={},
            )
        )


class KDACacheCoordinator(CheckpointedPageCacheCoordinator):
    """Coordinate full-attention pages and recurrent KDA endpoint checkpoints."""

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
                self.enqueue_dummy(cache, req.rid)
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
                self.enqueue_dummy(cache, req.rid)
                return
            write_key = (cache_len, endpoint_hash)
            if getattr(req, "_as_kda_last_submitted_write", None) == write_key:
                self.enqueue_dummy(cache, req.rid)
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
            self.enqueue_dummy(cache, req.rid)

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
        checkpoint_slot = checkpoint_allocation.value
        keep_transaction = False
        try:
            state_id = int(checkpoint_slot.item())
            if cache._as_min(1) == 0:
                return None
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
                checkpoint_slot=0,
                params={},
            )
        )
