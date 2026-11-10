"""
AsRadixCache: AttentionStore-backed secondary cache for RadixCache.

Extends RadixCache with AttentionStore (AS) integration, following the HiRadixCache
pattern of host_hit + load_back semantics. AS acts as an L2 cache layer
(analogous to HiCache's host memory tier).

Actual KV data DMA between GPU HBM and AS Host memory is always enabled.

Supported attention types:
- MLA: key and value share kv_buffer, v_buffer_ptrs is empty
- DSA (NSA): MLA-based key + separate index_k_with_scale_buffer as v
- MHA: separate k_buffer and v_buffer
"""

import atexit
import concurrent.futures
import logging
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    EvictResult,
    InitLoadBackParams,
    InsertParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode

logger = logging.getLogger(__name__)

# Minimum number of AS-matched blocks to justify a DMA read
AS_MIN_MATCH_BLOCKS = 5


@dataclass
class _AsWriteTask:
    """Encapsulates state for one async AS write operation."""

    node: TreeNode  # Last tree node covering write range (locked during DMA)
    layer_ids: List[int]
    as_block_hashs: Any  # attentionstore_sdk.sdk.Tokens object
    data_id_list: List[int]  # Page-boundary GPU buffer positions
    k_buffer_ptrs: List[int]
    v_buffer_ptrs: List[int]
    write_params: Any
    submit_time: float
    num_pages: int


class AsRadixCache(RadixCache):
    """RadixCache subclass with AttentionStore secondary cache integration."""

    def __init__(
        self,
        params: CacheInitParams,
        server_args,
        tp_rank: int = 0,
    ):
        self.enable_asradix_cache = server_args.enable_asradix_cache
        self.real_tp_rank = tp_rank
        self.per_node_gpu_num = server_args.per_node_gpu_num

        # AS SDK client reference (injected via ref_ascli)
        self.ascli = None

        # Attention type string: "MLA", "DSA", "MHA" (set by init_buffer_ptrs)
        self.attention_type: Optional[str] = None

        # Buffer pointers for AS SDK (initialized via init_buffer_ptrs)
        self.layer_ids: List[int] = []
        self.k_buffer_ptrs: List[int] = []
        self.v_buffer_ptrs: List[int] = []
        self._buffer_ptrs_ready = False

        # Disaggregation mode for PD role detection
        self._disagg_mode = server_args.disaggregation_mode

        # TP synchronization group (from CacheInitParams, same as HiRadixCache)
        self.tp_group = params.tp_cache_group
        if self.tp_group is not None:
            self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)
        else:
            self.tp_world_size = 1

        # Async write
        self._as_write_queue: List[Tuple[concurrent.futures.Future, _AsWriteTask]] = []
        self._write_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._max_pending_as_writes = 64  # Backpressure threshold

        # Call parent init (RadixCache)
        super().__init__(params)

        self._write_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="as_writer"
        )
        atexit.register(self._shutdown_as_writer)

        logger.info(
            f"AsRadixCache initialized: "
            f"tp_rank={self.real_tp_rank}, tp_world_size={self.tp_world_size}, "
            f"page_size={self.page_size}"
        )

    # ===== AS SDK injection =====

    def ref_ascli(self, ascli):
        """Inject AttentionStore SDK client reference."""
        self.ascli = ascli
        logger.info("AttentionStore SDK client injected into AsRadixCache")

    def init_buffer_ptrs(self, layer_ids: List[int], attention_type: str):
        """Extract K/V buffer data pointers from the KV pool for AS SDK.

        Buffer pointer extraction is attention-type aware:
        - MLA: k_buffer_ptrs from kv_buffer, v_buffer_ptrs empty
        - DSA: k_buffer_ptrs from kv_buffer, v_buffer_ptrs from index_k_with_scale_buffer
        - MHA: k_buffer_ptrs from k_buffer, v_buffer_ptrs from v_buffer

        Args:
            layer_ids: List of layer indices to extract buffer pointers for.
            attention_type: One of "MLA", "DSA", "MHA".
        """
        self.attention_type = attention_type.upper()
        self.layer_ids = layer_ids
        self.k_buffer_ptrs = []
        self.v_buffer_ptrs = []

        kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
        self._extract_buffer_ptrs(kv_cache, layer_ids)

        self._buffer_ptrs_ready = True
        logger.info(
            f"AS buffer pointers initialized: {len(self.layer_ids)} layers, "
            f"attention_type={self.attention_type}, "
            f"k_ptrs={len(self.k_buffer_ptrs)}, v_ptrs={len(self.v_buffer_ptrs)}"
        )

    def _extract_buffer_ptrs(self, kv_cache, layer_ids: List[int]):
        """Extract k/v buffer pointers from a KV cache according to attention_type.

        Uses internal (_get_*) methods when available to avoid triggering
        layer_transfer_counter synchronization hooks during initialization.
        """
        for layer_id in layer_ids:
            # K buffer: same for all types (get_key_buffer)
            if hasattr(kv_cache, "_get_key_buffer"):
                k_buf = kv_cache._get_key_buffer(layer_id)
            else:
                k_buf = kv_cache.get_key_buffer(layer_id)
            self.k_buffer_ptrs.append(k_buf.data_ptr())

            # V buffer: depends on attention type
            if self.attention_type == "MLA":
                # MLA: K and V share the same kv_buffer, no separate V pointer needed
                pass
            elif self.attention_type == "DSA":
                # DSA (NSA): V is the index_k_with_scale_buffer
                v_buf = kv_cache.get_index_k_with_scale_buffer(layer_id)
                self.v_buffer_ptrs.append(v_buf.data_ptr())
            else:
                # MHA: separate V buffer
                if hasattr(kv_cache, "_get_value_buffer"):
                    v_buf = kv_cache._get_value_buffer(layer_id)
                else:
                    v_buf = kv_cache.get_value_buffer(layer_id)
                self.v_buffer_ptrs.append(v_buf.data_ptr())

    def extend_with_draft_pool(self, draft_kv_cache, draft_layer_ids: List[int]):
        """Extend buffer pointers with draft model's KV pool layers.

        draft_layer_ids are virtual IDs (offset by target layer count) for AS SDK,
        but the draft KV pool is 0-indexed. Use physical IDs to access buffers.
        The draft model (EAGLE) may use a different attention type (e.g. MHA)
        than the target (e.g. DSA), so detect the pool type to extract V buffer.
        """
        physical_ids = list(range(len(draft_layer_ids)))
        has_index_k = hasattr(draft_kv_cache, "get_index_k_with_scale_buffer")

        for phys_id in physical_ids:
            # K buffer
            if hasattr(draft_kv_cache, "_get_key_buffer"):
                k_buf = draft_kv_cache._get_key_buffer(phys_id)
            else:
                k_buf = draft_kv_cache.get_key_buffer(phys_id)
            self.k_buffer_ptrs.append(k_buf.data_ptr())

            # V buffer: detect draft pool's actual type
            if self.attention_type == "MLA" and not has_index_k:
                # Draft pool is also MLA-like: no separate V pointer
                pass
            elif self.attention_type == "DSA" and has_index_k:
                # Draft pool is NSA: has index_k_with_scale_buffer
                v_buf = draft_kv_cache.get_index_k_with_scale_buffer(phys_id)
                self.v_buffer_ptrs.append(v_buf.data_ptr())
            else:
                # Draft pool is MHA (EAGLE) or fallback: use separate V buffer
                if hasattr(draft_kv_cache, "_get_value_buffer"):
                    v_buf = draft_kv_cache._get_value_buffer(phys_id)
                else:
                    v_buf = draft_kv_cache.get_value_buffer(phys_id)
                self.v_buffer_ptrs.append(v_buf.data_ptr())

        self.layer_ids = self.layer_ids + draft_layer_ids

    # ===== BasePrefixCache hooks =====

    def needs_req_in_match(self) -> bool:
        """Ensure schedule_batch passes req in MatchPrefixParams for AS parameter access."""
        return True

    # ===== Core overrides =====

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """Match prefix in local radix tree, then extend with AS match if applicable."""
        req = params.req

        # Step 1: Standard radix tree match (parent logic)
        result = super().match_prefix(params)

        # Step 2: Check if we should query AS
        if req is None or not self._should_as_match(req):
            return result

        max_hit_length = len(params.key) // self.page_size * self.page_size

        # Step 3: Skip AS match if HBM fully hit (all tokens matched locally)
        if result.device_indices.numel() == max_hit_length:
            logger.info(
                f"AS match_prefix: req:{req.rid} hbm_hit={max_hit_length} tokens, "
                f"as_hit=0 tokens, total={len(params.key)} tokens"
            )
            return result

        # Step 4: Query AS for additional prefix hits beyond local match
        as_result = self._as_match(req, result, params.key)
        if as_result is not None and as_result.host_hit_length > 0:
            logger.info(
                f"AS match_prefix: req:{req.rid} hbm_hit={len(result.device_indices)} tokens, "
                f"as_hit={as_result.host_hit_length} tokens, total={len(params.key)} tokens"
            )
            return as_result

        logger.info(
            f"AS match_prefix: req:{req.rid} hbm_hit={len(result.device_indices)} tokens, "
            f"as_hit=0 tokens, total={len(params.key)} tokens"
        )
        return result

    def cache_finished_req(self, req, is_insert: bool = True):
        """Cache finished request, then report to AS."""
        all_tokenids_len = len(req.origin_input_ids) + len(req.output_ids)
        snapshot = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :all_tokenids_len
        ].clone()
        # Standard radix tree insert
        super().cache_finished_req(req, is_insert)

        # Write KV to AS
        if self._is_prefill_role() and self.ascli is not None:
            self._write_kv_on_finish(req, snapshot)

    def evict(self, params: EvictParams) -> EvictResult:
        """Evict with AS index reporting."""
        if self.disable:
            return EvictResult()

        # Perform standard eviction
        result = super().evict(params)

        return result

    def reset(self):
        """Reset the radix tree.

        AsRadixCache holds node references in `_as_write_queue` whose lock_ref
        was incremented by `_write_kv_on_finish`. The base `RadixCache.reset()`
        replaces `self.root_node` with a brand-new TreeNode, which orphans
        every node referenced by the queue (their parent chain still points
        to the OLD root). A subsequent `check_ascache_events()` would then
        call `dec_lock_ref(old_node)` and fail in the base class with:
          AssertionError: This request holds the node from another tree
        because walking up the old chain reaches a parent=None node that
        is not the new `self.root_node`.

        Drain pending AS writes first (synchronously wait + dec_lock_ref
        while the old tree is still live), then call the base reset.
        """
        self._drain_as_write_queue()
        super().reset()

    def _drain_as_write_queue(self):
        """Synchronously drain `_as_write_queue`, waiting for every pending
        future and dec_lock_ref'ing each task node.

        Safe to call without TP all_reduce: every rank's queue length is
        kept identical by `_write_kv_on_finish` (real or dummy entry on
        each call), so all ranks drain the same number of entries.
        """
        if not self._as_write_queue:
            return

        for future, task in self._as_write_queue:
            try:
                future.result(timeout=30.0)
            except Exception as e:
                logger.warning(
                    f"AS write task failed during drain: "
                    f"node_id={id(task.node)}, error={e}"
                )
            self.dec_lock_ref(task.node)
        self._as_write_queue = []

    # ===== HiCache-compatible hooks =====

    def init_load_back(
        self,
        params: InitLoadBackParams,
    ) -> Tuple[torch.Tensor, Any]:
        """Allocate HBM and prepare for AS -> GPU DMA read.

        Called by schedule_policy when host_hit_length > 0.
        The last_host_node here is the node with _as_load_info attached by _as_match().
        """
        last_host_node = params.best_match_node
        host_hit_length = params.host_hit_length

        empty = torch.empty((0,), dtype=torch.int64, device=self.device)

        if self.ascli is None or host_hit_length <= 0:
            return empty, last_host_node

        load_info = getattr(last_host_node, "_as_load_info", None)
        req = getattr(last_host_node, "_as_load_req", None)

        # Clean up temporary attributes to avoid memory leaks on TreeNode
        if hasattr(last_host_node, "_as_load_info"):
            del last_host_node._as_load_info
        if hasattr(last_host_node, "_as_load_req"):
            del last_host_node._as_load_req

        if load_info is None or req is None:
            return empty, last_host_node

        # Perform the actual AS -> GPU DMA load
        loaded_indices = self._do_as_load(load_info, host_hit_length)
        if loaded_indices is not None and len(loaded_indices) > 0:
            req.ever_read_as = True

            # Insert AS-loaded tokens into the radix tree immediately so that
            # they are properly tracked by evictable_size_. This prevents the
            # memory leak where tokens are allocated but untracked if the
            # request is rejected by schedule_policy after init_load_back.
            # _insert_as_loaded returns truncated loaded_indices (tail freed
            # due to EAGLE bigram + page alignment) and the new last_node.
            loaded_indices, new_last_node = self._insert_as_loaded(
                req, last_host_node, load_info, loaded_indices
            )
            return loaded_indices, new_last_node

        return empty, last_host_node

    def ready_to_load_host_cache(self) -> int:
        """AS loads are synchronous, so return 0 (no async consumer to track)."""
        return 0

    def check_ascache_events(self):
        """Check for completed AS write tasks and clean up.

        TP-synchronized: all ranks agree on how many tasks to process
        using all_reduce(MIN), mirroring HiRadixCache.writing_check().
        Single-threaded executor (max_workers=1) guarantees FIFO
        completion order, so counting from the front is safe.

        This is called by the scheduler in each iteration.
        """
        # Single-card: skip sync, use simple early return
        if self.tp_world_size <= 1:
            if not self._as_write_queue:
                return
            finish_count = 0
            for future, _task in self._as_write_queue:
                if future.done():
                    finish_count += 1
                else:
                    break
            if finish_count > 0:
                for i in range(finish_count):
                    future, task = self._as_write_queue[i]
                    try:
                        future.result(timeout=1.0)
                    except Exception as e:
                        logger.warning(
                            f"AS write task failed: node_id={id(task.node)}, error={e}"
                        )
                    self.dec_lock_ref(task.node)
                self._as_write_queue = self._as_write_queue[finish_count:]
            return

        # Multi-card TP: must participate in all_reduce even if queue is empty,
        # otherwise other ranks will block on the collective.
        # Step 1: FIFO count of locally completed tasks
        finish_count = 0
        for future, _task in self._as_write_queue:
            if future.done():
                finish_count += 1
            else:
                break  # FIFO: later tasks can't be done before this one

        # Step 2: TP sync — take MIN so all ranks process the same count
        count_tensor = torch.tensor(finish_count, dtype=torch.int)
        torch.distributed.all_reduce(
            count_tensor,
            op=torch.distributed.ReduceOp.MIN,
            group=self.tp_group,
        )
        finish_count = int(count_tensor.item())

        if finish_count == 0:
            return

        # Step 3: Process exactly finish_count tasks — all ranks unlock
        # the same nodes in the same iteration, keeping evictable_size_ in sync
        for i in range(finish_count):
            future, task = self._as_write_queue[i]
            try:
                future.result(timeout=1.0)
            except Exception as e:
                logger.warning(
                    f"AS write task failed: node_id={id(task.node)}, error={e}"
                )
            self.dec_lock_ref(task.node)

        self._as_write_queue = self._as_write_queue[finish_count:]

    # ===== Internal: AS match =====

    def _insert_as_loaded(self, req, last_host_node, load_info, loaded_indices):
        """Insert AS-loaded KV data into the radix tree for proper memory tracking.

        After a successful AS DMA read, inserts the loaded prefix into the tree
        so the tokens are tracked by evictable_size_. This prevents the memory
        leak where tokens allocated by init_load_back are untracked if the
        request is rejected by schedule_policy (early return after init_load_back).

        If the request IS scheduled, _req_inc_lock_ref will lock the node.
        If the request is rejected, the tokens remain as evictable tree entries.

        Returns (truncated_loaded_indices, new_last_node). The returned indices
        only cover the portion actually inserted into the tree (page-aligned
        after EAGLE bigram conversion). The tail is freed back to the allocator.
        """
        local_hit_len = load_info["start_block"] * self.page_size
        total_prefix_len = local_hit_len + len(loaded_indices)

        assert len(req.prefix_indices) == local_hit_len, (
            f"prefix_indices length {len(req.prefix_indices)} != "
            f"local_hit_len {local_hit_len} (start_block={load_info['start_block']}, "
            f"page_size={self.page_size})"
        )

        # Build key: pass raw token_ids and let RadixKey/RadixCache handle the
        # bigram view + page alignment internally. Pre-bigram-converting here
        # and then setting is_bigram=True is a bug — RadixKey treats token_ids
        # as raw whenever is_bigram=True, so feeding it pre-paired tuples
        # double-applies the bigram transform (child_key becomes pairs-of-tuples,
        # creating a phantom sibling under root distinct from cache_finished_req's
        # correctly-keyed insert and double-counting the same KV slots).
        token_ids = req.full_untruncated_fill_ids[:total_prefix_len]

        # Compute the final tree-key length the way RadixCache.insert will:
        # bigram view shrinks raw N to N-1, then page-align.
        bigram_len = max(0, len(token_ids) - 1) if self.is_eagle else len(token_ids)
        final_key_len = bigram_len // self.page_size * self.page_size

        if final_key_len == 0:
            # No keys after page alignment — free all loaded indices
            self.token_to_kv_pool_allocator.free(loaded_indices)
            return (
                torch.empty(
                    (0,), dtype=loaded_indices.dtype, device=loaded_indices.device
                ),
                last_host_node,
            )

        # Build value: local prefix kv_indices + AS loaded kv_indices, sized
        # to the actual tree-key length so RadixCache.insert won't truncate.
        full_values = torch.cat([req.prefix_indices, loaded_indices])
        values = full_values[:final_key_len].to(dtype=torch.int64, copy=True)

        # Free the tail of loaded_indices that won't be tracked by the tree
        # (EAGLE bigram drops 1 token + page alignment may drop up to one page).
        tail_start = max(final_key_len - local_hit_len, 0)
        if tail_start < len(loaded_indices):
            self.token_to_kv_pool_allocator.free(loaded_indices[tail_start:])
            loaded_indices = loaded_indices[:tail_start]

        # Insert into tree — _insert_helper walks existing local prefix nodes
        # and creates new nodes for the AS portion only.
        self.insert(
            InsertParams(
                key=RadixKey(token_ids, req.extra_key, is_bigram=self.is_eagle),
                value=values,
            )
        )

        # Re-match to get the correct last_node + the tree-owned slot indices
        # for every key we just (tried to) insert. Needed because if the tree
        # already held part of the AS portion, _insert_helper walked past
        # those keys without adopting our freshly allocated loaded_indices —
        # they would leak. Switch loaded_indices to the tree's slot indices
        # and free our displaced slots.
        match_result = super().match_prefix(
            MatchPrefixParams(
                key=RadixKey(token_ids, req.extra_key, is_bigram=self.is_eagle),
            )
        )

        if match_result.last_device_node is None:
            logger.warning(
                "AS insert_loaded: re-match returned no last_node, "
                f"loaded={len(loaded_indices)}, final_key_len={final_key_len}"
            )
            return loaded_indices, last_host_node

        # Reconcile loaded_indices with the tree's actually-owned slots.
        # tree_full_indices covers `final_key_len` tokens (prefix + AS portion).
        # The AS portion is `tree_full_indices[local_hit_len:]`.
        # _insert_helper walks matched nodes WITHOUT replacing their values,
        # creating new nodes for the unmatched suffix using our loaded_indices.
        # Free only positions where the tree's slot differs from ours.
        tree_full_indices = match_result.device_indices
        if len(tree_full_indices) >= final_key_len:
            tree_as_indices = tree_full_indices[local_hit_len:].to(
                dtype=loaded_indices.dtype
            )
            if tree_as_indices.numel() == loaded_indices.numel():
                displaced_mask = tree_as_indices != loaded_indices
                if bool(displaced_mask.any()):
                    self.token_to_kv_pool_allocator.free(loaded_indices[displaced_mask])
                    loaded_indices = tree_as_indices

        logger.debug(
            f"AS insert_loaded: inserted {len(loaded_indices)} tokens into radix tree"
        )
        return loaded_indices, match_result.last_device_node

    def _should_as_match(self, req) -> bool:
        """Determine if AS match should be attempted for this request."""
        if req.ever_read_as:
            return False
        if not req.as_block_hashs:
            return False
        if self.ascli is None:
            return False
        return True

    def _as_match(self, req, local_result: MatchResult, key) -> Optional[MatchResult]:
        """Query AS for additional prefix hits beyond local radix tree match.

        Maps AS hit to host_hit_length semantics so the scheduler's existing
        host-hit processing path can handle it transparently.
        """
        local_hit_len = len(local_result.device_indices)
        as_block_hashs = req.as_block_hashs
        start_block = local_hit_len // self.page_size

        # Check for remote hit info (from upstream router/IC)
        remote_addrs = req.as_max_hit_host_ip
        remote_len = req.as_max_hit_token_len

        # If remote hit info indicates a specific hit length, use it directly
        if remote_len and len(remote_len) == 1 and remote_len[0] > 0:
            as_hit_tokens = (
                (remote_len[0] - local_hit_len) // self.page_size
            ) * self.page_size
            if as_hit_tokens <= 0:
                return None
        else:
            # Each TP rank queries AS independently
            try:
                t0 = time.perf_counter()
                as_match_count = self.ascli.match(as_block_hashs, start_block, 1)
                t1 = time.perf_counter()
                logger.info(
                    f"AS match: req:{req.rid} count={as_match_count}, cost={(t1 - t0) * 1000:.2f}ms"
                )
            except Exception as e:
                logger.warning(f"AS match failed: {e}")
                as_match_count = 0

            # TP sync: all_reduce(MIN) so all ranks agree on the same match count
            if self.tp_world_size > 1:
                match_tensor = torch.tensor(as_match_count, dtype=torch.int)
                torch.distributed.all_reduce(
                    match_tensor,
                    op=torch.distributed.ReduceOp.MIN,
                    group=self.tp_group,
                )
                as_match_count = int(match_tensor.item())

            if as_match_count < AS_MIN_MATCH_BLOCKS:
                return None

            as_hit_tokens = as_match_count * self.page_size

        # Cap AS hit so that at least 1 token remains for extend forward pass.
        # Without this, local_hit + AS_hit could cover all fill_ids, making
        # extend_input_len=0 and causing an empty-tensor crash in model forward.
        max_as_tokens = (
            (len(req.full_untruncated_fill_ids) - local_hit_len - 1)
            // self.page_size
            * self.page_size
        )
        if max_as_tokens <= 0:
            return None
        as_hit_tokens = min(as_hit_tokens, max_as_tokens)

        # Store load info on the last_device_node for init_load_back to use
        last_node = local_result.last_device_node
        last_node._as_load_req = req
        last_node._as_load_info = {
            "start_block": start_block,
            "as_hit_tokens": as_hit_tokens,
            "as_block_hashs": as_block_hashs,
            "remote_addrs": remote_addrs,
            "remote_len": remote_len,
        }

        return MatchResult(
            device_indices=local_result.device_indices,
            last_device_node=local_result.last_device_node,
            last_host_node=last_node,  # used as load_back anchor
            best_match_node=last_node,
            host_hit_length=as_hit_tokens,
        )

    # ===== Internal: AS load (DMA read) =====

    def _do_as_load(
        self, load_info: dict, host_hit_length: int
    ) -> Optional[torch.Tensor]:
        """Execute the actual AS -> GPU DMA read.

        Allocates HBM slots and calls ascli.read() to DMA load KV data.
        """
        if not self._buffer_ptrs_ready:
            logger.warning("AS load: buffer pointers not initialized, skipping")
            return None

        as_block_hashs = load_info["as_block_hashs"]
        start_block = load_info["start_block"]
        as_hit_tokens = load_info["as_hit_tokens"]
        remote_addrs = load_info.get("remote_addrs", [])

        # Check available space; evict proactively if needed
        available = self.token_to_kv_pool_allocator.available_size()
        if available < as_hit_tokens:
            evictable = self.evictable_size()
            if evictable > 0:
                evict_tokens = max(as_hit_tokens - available, 0) + self.page_size
                logger.debug(
                    f"AS load: proactive evict {evict_tokens} tokens "
                    f"(need={as_hit_tokens}, available={available}, evictable={evictable})"
                )
                self.evict(EvictParams(num_tokens=evict_tokens))

        # Single alloc attempt after ensuring space
        try:
            alloc_indices = self.token_to_kv_pool_allocator.alloc(as_hit_tokens)
        except Exception:
            logger.warning(f"AS load: failed to allocate {as_hit_tokens} tokens")
            return None

        if alloc_indices is None or len(alloc_indices) == 0:
            logger.warning(
                f"AS load: alloc failed, "
                f"need={as_hit_tokens}, "
                f"available={self.token_to_kv_pool_allocator.available_size()}, "
                f"evictable={self.evictable_size()}"
            )
            return None

        # Prepare data_id_list: one page-boundary index per block for AS SDK
        data_id_list = (alloc_indices[:: self.page_size] // self.page_size).tolist()

        # Determine remote address – SDK rejects None; pass empty string for local read
        remote_addr = remote_addrs[0] if remote_addrs else ""

        # Execute DMA read via AS SDK
        try:
            t0 = time.perf_counter()
            read_count = self.ascli.read(
                layer_ids=self.layer_ids,
                tokens=as_block_hashs,
                start_transfer_block_index=start_block,
                k_dest_ptrs=self.k_buffer_ptrs,
                v_dest_ptrs=self.v_buffer_ptrs,
                data_id_list=data_id_list,
                timeout=1,
                remote_addrs=remote_addr,
            )
            t1 = time.perf_counter()
            logger.info(f"AS read: count={read_count}, cost={(t1 - t0) * 1000:.2f}ms")
        except Exception as e:
            logger.warning(f"AS read failed: {e}")
            read_count = 0

        # TP sync: all_reduce(MIN) so all ranks agree on the same read count
        if self.tp_world_size > 1:
            read_tensor = torch.tensor(read_count, dtype=torch.int)
            torch.distributed.all_reduce(
                read_tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
            read_count = int(read_tensor.item())

        if read_count <= 0:
            self.token_to_kv_pool_allocator.free(alloc_indices)
            return None

        loaded_tokens = read_count * self.page_size
        loaded_indices = alloc_indices[:loaded_tokens]

        # Free unused allocation if partial read
        if loaded_tokens < len(alloc_indices):
            self.token_to_kv_pool_allocator.free(alloc_indices[loaded_tokens:])

        return loaded_indices

    def _shutdown_as_writer(self):
        """Shutdown the async AS writer executor."""
        if self._write_executor is not None:
            self._write_executor.shutdown(wait=True)

    # ===== Internal: AS KV write (Full R/W mode) =====

    def _enqueue_dummy_write(self):
        """Enqueue a no-op write task to keep _as_write_queue aligned across TP ranks.

        When a real write submission fails on this rank but succeeds on others,
        the dummy ensures all ranks have the same queue length so that
        check_ascache_events's all_reduce(MIN) stays correct.

        Uses root_node as the task node: dec_lock_ref(root_node) walks
        'while node != self.root_node' which is immediately false, making
        it a safe no-op with no lock_ref / evictable_size_ side effects.
        """
        dummy_future = concurrent.futures.Future()
        dummy_future.set_result(0)
        dummy_task = _AsWriteTask(
            node=self.root_node,
            layer_ids=[],
            as_block_hashs=None,
            data_id_list=[],
            k_buffer_ptrs=[],
            v_buffer_ptrs=[],
            write_params={},
            submit_time=time.perf_counter(),
            num_pages=0,
        )
        self._as_write_queue.append((dummy_future, dummy_task))

    def _do_as_write(self, task: _AsWriteTask) -> int:
        """Execute AS write operation in background thread.

        Args:
            task: _AsWriteTask containing all necessary write parameters

        Returns:
            Number of pages written (or 0 on failure)
        """
        try:
            t0 = time.perf_counter()
            w_res = self.ascli.write(
                layer_ids=task.layer_ids,
                tokens=task.as_block_hashs,
                start_transfer_block_index=0,
                k_data_ptrs=task.k_buffer_ptrs,
                v_data_ptrs=task.v_buffer_ptrs,
                data_id_list=task.data_id_list,
                timeout=1,
                h2h_copy=False,
                params=task.write_params,
            )
            t1 = time.perf_counter()

            logger.info(
                f"AS write: type={self.attention_type}, "
                f"pages num={task.num_pages}, result={w_res}, "
                f"cost={(t1 - t0) * 1000:.2f}ms"
            )
            return w_res
        except Exception as e:
            logger.warning(f"AS write failed: {e}")
            return 0

    def _write_kv_on_finish(self, req, snapshot: Optional[torch.Tensor] = None):
        """Submit KV write task to async executor after request completion.

        Key points:
        - Called after super().cache_finished_req(), so KV is already inserted in radix tree
        - Finds the node holding the KV data using match_prefix for lock_ref management
        - Locks the node with inc_lock_ref() to prevent eviction during async write
        - Submits async write to background thread pool
        - Node is unlocked in check_ascache_events() when write completes

        TP alignment: every call must append exactly one entry to _as_write_queue
        (real or dummy) so that queue length stays consistent across TP ranks.
        check_ascache_events uses all_reduce(MIN) on FIFO position, so misaligned
        queue lengths would cause it to dec_lock_ref on mismatched nodes.
        """
        if not self._buffer_ptrs_ready:
            if self.tp_world_size > 1:
                self._enqueue_dummy_write()
            return

        if not self._write_executor:
            logger.warning("AS write: executor not initialized, skipping")
            if self.tp_world_size > 1:
                self._enqueue_dummy_write()
            return

        try:
            from attentionstore_sdk.sdk import Tokens

            kv_committed_len = len(req.origin_input_ids) + len(req.output_ids)
            token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]

            # Align to page boundary (only write complete pages)
            num_tokens = len(token_ids)
            num_pages = num_tokens // self.page_size
            if num_pages <= 0:
                if self.tp_world_size > 1:
                    self._enqueue_dummy_write()
                return
            aligned_tokens = num_pages * self.page_size

            as_block_hashs = Tokens(token_ids[:aligned_tokens], self.page_size)

            # Get KV indices for this request (token-level pool slot indices)
            kv_indices = snapshot[:aligned_tokens]

            # Convert to page-boundary indices: one index per block for AS SDK
            data_id_list = (kv_indices[:: self.page_size] // self.page_size).tolist()

            # Get write_params from request's AS parameters (passed from upstream)
            # Default to empty dict if None – SDK requires a python/numpy/pytorch object
            write_params = getattr(req, "as_kv_cache_params", None) or {}

            # Find the node covering the written range for lock_ref management.
            # NOTE: pass RAW token_ids to RadixKey with is_bigram=self.is_eagle; do NOT
            # pre-convert via convert_to_bigram_key — RadixKey assumes raw tokens when
            # is_bigram=True and computes bigrams on-the-fly. Pre-converting causes a
            # double-bigram (pairs-of-pairs) child_key namespace mismatch with
            # cache_finished_req's insert path, leaking KV slots.
            match_result = super().match_prefix(
                MatchPrefixParams(
                    key=RadixKey(token_ids, req.extra_key, is_bigram=self.is_eagle),
                )
            )
            last_node = match_result.last_device_node

            if last_node is None:
                logger.warning("AS write: no matching node found, skipping")
                if self.tp_world_size > 1:
                    self._enqueue_dummy_write()
                return

            # Create write task
            task = _AsWriteTask(
                node=last_node,
                layer_ids=self.layer_ids,
                as_block_hashs=as_block_hashs,
                data_id_list=data_id_list,
                k_buffer_ptrs=self.k_buffer_ptrs,
                v_buffer_ptrs=self.v_buffer_ptrs,
                write_params=write_params,
                submit_time=time.perf_counter(),
                num_pages=num_pages,
            )

            # Lock the node before submitting write task
            # This prevents the node from being evicted during async write
            self.inc_lock_ref(last_node)

            # Submit to executor and track
            future = self._write_executor.submit(self._do_as_write, task)
            self._as_write_queue.append((future, task))

            logger.debug(
                f"AS write task submitted: num_pages={num_pages}, "
                f"node_id={id(last_node)}"
            )

        except Exception as e:
            logger.warning(f"AS write: failed to submit task: {e}")
            if self.tp_world_size > 1:
                self._enqueue_dummy_write()

    # ===== Internal: utilities =====

    def _is_prefill_role(self) -> bool:
        """Check if current node is in PREFILL role (for PD disaggregation)."""
        return self._disagg_mode in ("null", "prefill")

    def _reconstruct_token_ids(self, node: TreeNode) -> list:
        """Reconstruct the full token ID sequence from root to this node."""
        parts = []
        current = node
        while current.parent is not None:
            if current.key is not None:
                parts.append(list(current.key.token_ids))
            current = current.parent
        parts.reverse()
        result = []
        for p in parts:
            result.extend(p)
        return result
