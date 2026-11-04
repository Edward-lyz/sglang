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
import os
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    EvictResult,
    InsertParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import (
    RadixCache,
    RadixKey,
    TreeNode,
    page_align_keys,
)
from sglang.srt.mem_cache.as_load_kernels import (
    AsLoadSegment,
    _BLOCK_BYTES,
    _DESCRIPTOR_COLUMNS,
    _METADATA_COLUMNS,
    build_as_load_descriptors,
    make_inplace_allgather_pair,
    make_as_load_segment,
    pack_as_load_segments,
    select_as_load_vector_bytes,
    unpack_as_load_segments,
    warmup_as_load_kernels,
)

logger = logging.getLogger(__name__)


@dataclass
class _AsLoadBatchPlan:
    layer_ids: List[int]
    segments: List[AsLoadSegment]
    metadata: Optional[torch.Tensor]
    descriptors: Optional[torch.Tensor]
    local_nbytes: int
    vector_bytes: int = 1
    inplace_pairs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    local_scratch: Optional[torch.Tensor] = None
    gathered_scratch: Optional[torch.Tensor] = None


@dataclass
class _AsLoadResult:
    indices: torch.Tensor
    page_ids_cpu: torch.Tensor
    ready_event: Optional[Any]
    resources: Tuple[Any, ...] = ()


@dataclass
class _AsPendingLoad:
    ready_event: Any
    node: Optional[TreeNode]
    deferred_page_ids: torch.Tensor
    resources: Tuple[Any, ...]
    pages_released: bool = False
    lock_released: bool = False


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
    trace_recorder: Optional[Any] = None


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
        self.indexer_layer_ids: Optional[List[int]] = None
        self._indexer_layer_id_set = None
        self.k_buffer_ptrs: List[int] = []
        self.v_buffer_ptrs: List[int] = []
        self._buffer_ptrs_ready = False

        # KV pool refs kept for sharded-load allgather (need real tensors, not just data_ptr)
        self._main_kv_cache = None
        self._draft_kv_cache = None
        self._target_layer_count = 0

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

        # read/allgather overlap: per-batch layer count. Larger values trade
        # fewer collectives for more scratch and coarser overlap.
        self._load_batch_layers = max(
            1, int(os.environ.get("AS_LOAD_ALLGATHER_BATCH", "5"))
        )

        # Shared-fetch read timeout (s): must outlast all batches' scatter +
        # allgather, else the agent's timeout checker auto-releases SHM mid-load.
        self._as_shard_fetch_timeout = float(
            os.environ.get("AS_LOAD_FETCH_TIMEOUT", "30")
        )

        # Dedicated NCCL stream for load allgather (lazily created on device) so
        # it runs concurrently with the SDK's scatter streams.
        self._as_load_stream: Optional[torch.cuda.Stream] = None

        # Three-stream overlap: unpack (Triton unpack of gathered scratch) and a
        # page-id D2H copy stream run concurrently with the load/allgather stream.
        self._as_unpack_stream: Optional[torch.cuda.Stream] = None
        self._as_page_id_copy_stream: Optional[torch.cuda.Stream] = None

        # Pinned host buffer for async page-ID D2H staging (reused across loads).
        self._as_page_ids_host: Optional[torch.Tensor] = None

        # Persistent pack/unpack scratch, reused once the prior request's unpack
        # stream drains (fenced by _as_scratch_ready_event). Gathered scratch is
        # double-buffered (2 slots) for back-to-back batches.
        self._as_local_scratch: Optional[torch.Tensor] = None
        self._as_gathered_scratch: List[torch.Tensor] = []
        self._as_scratch_ready_event: Optional[Any] = None

        # Pending AS loads awaiting async-completion cleanup (deferred page-ID
        # release + lock_ref release). Drained FIFO in check_ascache_events.
        self._as_pending_loads: List[_AsPendingLoad] = []

        # Allgather mode: packed (default/fallback), auto (in-place when pages
        # are contiguous + pynccl available), inplace (force, fall back to packed).
        self._as_allgather_mode = os.environ.get(
            "AS_LOAD_ALLGATHER_MODE", "packed"
        ).lower()
        if self._as_allgather_mode not in {"packed", "auto", "inplace"}:
            raise ValueError(
                "AS_LOAD_ALLGATHER_MODE must be one of: packed, auto, inplace"
            )

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

    def init_buffer_ptrs(
        self,
        layer_ids: List[int],
        attention_type: str,
        indexer_layer_ids: Optional[List[int]] = None,
    ):
        """Extract K/V buffer data pointers from the KV pool for AS SDK.

        Buffer pointer extraction is attention-type aware:
        - MLA: k_buffer_ptrs from kv_buffer, v_buffer_ptrs empty
        - DSA: k_buffer_ptrs from kv_buffer, v_buffer_ptrs from index_k_with_scale_buffer
        - MHA: k_buffer_ptrs from k_buffer, v_buffer_ptrs from v_buffer

        Args:
            layer_ids: List of layer indices to extract buffer pointers for.
            attention_type: One of "MLA", "DSA", "MHA".
            indexer_layer_ids: DSA logical layers that own physical indexer
                buffers. None preserves the legacy one-V-buffer-per-layer mode.
        """
        self.attention_type = attention_type.upper()
        self.layer_ids = layer_ids
        self.indexer_layer_ids = indexer_layer_ids
        self._indexer_layer_id_set = (
            set(indexer_layer_ids) if indexer_layer_ids is not None else None
        )
        self.k_buffer_ptrs = []
        self.v_buffer_ptrs = []

        kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
        self._main_kv_cache = kv_cache
        self._target_layer_count = len(layer_ids)
        self._extract_buffer_ptrs(kv_cache, layer_ids)

        self._buffer_ptrs_ready = True
        logger.info(
            f"AS buffer pointers initialized: {len(self.layer_ids)} layers, "
            f"attention_type={self.attention_type}, "
            f"indexer_layers={self.indexer_layer_ids}, "
            f"k_ptrs={len(self.k_buffer_ptrs)}, v_ptrs={len(self.v_buffer_ptrs)}"
        )
        self._warmup_as_load_kernels()

    def _warmup_as_load_kernels(self) -> None:
        """Pre-compile pack/unpack Triton kernels at init so the first AS load
        does not pay JIT compilation inside the shared-memory timeout window.

        Triton caches per-constexpr binary (VECTOR_BYTES, BLOCK_ROWS,
        BLOCK_BYTES, world_size); shapes/pointers don't matter. ``.warmup()``
        only compiles, so dummy tensors suffice. VECTOR_BYTES is per-request, so
        pre-compile all widths.
        """
        from sglang.srt.layers.dp_attention import get_attention_tp_group

        try:
            attn_tp = get_attention_tp_group()
            world = attn_tp.world_size
        except Exception as e:
            logger.warning(
                f"AS load kernel warmup skipped: attn TP group unavailable ({e})"
            )
            return

        device = self.device
        # One descriptor block -> non-empty grid; contents irrelevant since
        # .warmup() does not launch. torch.zeros is aligned for every VECTOR_BYTES.
        metadata = torch.zeros(1, _METADATA_COLUMNS, dtype=torch.int64, device=device)
        descriptors = torch.zeros(
            1, _DESCRIPTOR_COLUMNS, dtype=torch.int64, device=device
        )
        scratch = _BLOCK_BYTES
        local = torch.zeros(scratch, dtype=torch.uint8, device=device)
        gathered = torch.zeros(
            scratch * max(world, 1), dtype=torch.uint8, device=device
        )

        try:
            warmed = []
            for vb in (8, 4, 2, 1):
                warmup_as_load_kernels(
                    metadata,
                    descriptors,
                    local,
                    gathered,
                    packed_rank_bytes=scratch,
                    world_size=world,
                    vector_bytes=vb,
                )
                warmed.append(vb)
            logger.info(
                f"AS load kernels warmed up: world_size={world}, "
                f"vector_bytes={warmed}, device=cuda:{torch.cuda.current_device()}"
            )
        except Exception as e:
            # Warmup is an optimization only; the runtime path still compiles
            # lazily on first use, so never fail init over it.
            logger.warning(f"AS load kernel warmup failed (will lazy-compile): {e}")

    def _extract_buffer_ptrs(self, kv_cache, layer_ids: List[int]):
        """Extract k/v buffer pointers from a KV cache according to attention_type.

        Uses internal (_get_*) methods when available to avoid triggering
        layer_transfer_counter synchronization hooks during initialization.
        """
        # K buffers exist for every logical layer.
        for layer_id in layer_ids:
            if hasattr(kv_cache, "_get_key_buffer"):
                k_buf = kv_cache._get_key_buffer(layer_id)
            else:
                k_buf = kv_cache.get_key_buffer(layer_id)
            self.k_buffer_ptrs.append(k_buf.data_ptr())

        if self.attention_type == "MLA":
            # MLA: K and V share the same kv_buffer, no separate V pointer needed.
            return

        if self.attention_type == "DSA":
            # Under indexer-cache optimization only non-shared logical layers
            # own a physical indexer buffer. Preserve that compact pointer order;
            # the SDK expands it to the unchanged full DSA block layout.
            v_layer_ids = (
                self.indexer_layer_ids
                if self.indexer_layer_ids is not None
                else layer_ids
            )
            for layer_id in v_layer_ids:
                v_buf = kv_cache.get_index_k_with_scale_buffer(layer_id)
                self.v_buffer_ptrs.append(v_buf.data_ptr())
            return

        # MHA: separate V buffer for every logical layer.
        for layer_id in layer_ids:
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
        self._draft_kv_cache = draft_kv_cache

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
        if self.indexer_layer_ids is not None:
            self.indexer_layer_ids = self.indexer_layer_ids + draft_layer_ids
            self._indexer_layer_id_set = set(self.indexer_layer_ids)

        logger.info(
            f"AS buffer pointers extended with draft pool: "
            f"+{len(draft_layer_ids)} draft layers, "
            f"total_layers={len(self.layer_ids)}, "
            f"k_ptrs={len(self.k_buffer_ptrs)}, v_ptrs={len(self.v_buffer_ptrs)}"
        )

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

        # Forward completion proves every pending load from this scheduling
        # epoch is consumable. Drain the FIFO so every TP rank advances the
        # same count and no in-flight allgather touches snapshotted/written rows.
        self._synchronize_pending_as_loads()
        self._complete_as_loads(len(self._as_pending_loads))
        if self._as_pending_loads:
            raise RuntimeError(
                "Failed to drain AS-load state during request finalization"
            )

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
        """Drain AS I/O before clearing tree, allocator-facing, and scratch state."""
        write_tasks = self._as_write_queue
        self._as_write_queue = []
        for future, task in write_tasks:
            try:
                future.result()
            except Exception:
                logger.exception("AS write failed while draining cache reset")
            finally:
                try:
                    self.dec_lock_ref(task.node)
                except Exception:
                    logger.exception("Failed to release AS-write lock during reset")

        self._synchronize_pending_as_loads()
        self._complete_as_loads(len(self._as_pending_loads))
        if self._as_pending_loads:
            raise RuntimeError("Failed to drain AS-load state during cache reset")

        super().reset()

        # Drop persistent overlap scratch so the next load reallocates cleanly.
        self._as_local_scratch = None
        self._as_gathered_scratch = []
        self._as_scratch_ready_event = None

    # ===== HiCache-compatible hooks =====

    def init_load_back(
        self,
        last_host_node: Any,
        host_hit_length: int,
    ) -> Tuple[torch.Tensor, Any]:
        """Allocate HBM and prepare for AS -> GPU DMA read.

        Called by schedule_policy when host_hit_length > 0.
        The last_host_node here is the node with _as_load_info attached by _as_match().
        """
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
        load_result = self._do_as_load(load_info, host_hit_length)
        if load_result is not None and len(load_result.indices) > 0:
            req.ever_read_as = True

            # Insert AS-loaded tokens into the radix tree immediately so that
            # they are properly tracked by evictable_size_. This prevents the
            # memory leak where tokens are allocated but untracked if the
            # request is rejected by schedule_policy after init_load_back.
            # _insert_as_loaded returns truncated loaded_indices (tail deferred
            # for release until the async load completes) and the new last_node.
            loaded_indices, new_last_node = self._insert_as_loaded(
                req, last_host_node, load_info, load_result
            )
            return loaded_indices, new_last_node

        return empty, last_host_node

    def ready_to_load_host_cache(self) -> int:
        """AS loads are synchronous, so return 0 (no async consumer to track)."""
        return 0

    def check_ascache_events(self):
        """Check for completed AS load/write tasks and clean up.

        TP-synchronized via all_reduce(MIN), mirroring HiRadixCache.writing_check().
        The single-threaded executor (max_workers=1) guarantees FIFO completion,
        so counting from the front is safe. Pending AS loads drain the same way;
        their ready_event fences the deferred page-ID and lock_ref release.

        Called by the scheduler each iteration.
        """
        load_finish_count = self._finished_as_load_count()

        # Single-card: skip sync, use simple early return
        if self.tp_world_size <= 1:
            if not self._as_write_queue and load_finish_count == 0:
                return
            finish_count = 0
            for future, _task in self._as_write_queue:
                if future.done():
                    finish_count += 1
                else:
                    break
            self._complete_as_loads(load_finish_count)
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
        count_tensor = torch.tensor([finish_count, load_finish_count], dtype=torch.int)
        torch.distributed.all_reduce(
            count_tensor,
            op=torch.distributed.ReduceOp.MIN,
            group=self.tp_group,
        )
        finish_count, load_finish_count = map(int, count_tensor.tolist())

        if finish_count == 0 and load_finish_count == 0:
            return

        # Step 3: Process exactly finish_count tasks — all ranks unlock
        # the same nodes in the same iteration, keeping evictable_size_ in sync
        self._complete_as_loads(load_finish_count, counts_are_synchronized=True)
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

    def _finished_as_load_count(self) -> int:
        finish_count = 0
        for task in self._as_pending_loads:
            if not task.ready_event.query():
                break
            finish_count += 1
        return finish_count

    def _synchronize_pending_as_loads(self) -> None:
        first_error = None
        for task in self._as_pending_loads:
            try:
                task.ready_event.synchronize()
            except Exception as error:
                logger.exception("Failed to synchronize pending AS load")
                first_error = error
                break

        sync_ok = first_error is None
        if self.tp_world_size > 1:
            sync_state = torch.tensor(int(sync_ok), dtype=torch.int)
            torch.distributed.all_reduce(
                sync_state,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
            sync_ok = bool(sync_state.item())

        if not sync_ok:
            raise RuntimeError(
                "AS-load event synchronization failed on at least one TP rank"
            ) from first_error

    def _complete_as_loads(
        self, finish_count: int, *, counts_are_synchronized: bool = False
    ) -> None:
        if self.tp_world_size > 1 and not counts_are_synchronized:
            count_state = torch.tensor([finish_count, -finish_count], dtype=torch.int)
            torch.distributed.all_reduce(
                count_state,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
            min_count = int(count_state[0].item())
            max_count = -int(count_state[1].item())
            if min_count != max_count:
                raise RuntimeError(
                    "AS-load cleanup queue diverged across TP ranks: "
                    f"min={min_count}, max={max_count}"
                )

        if finish_count <= 0:
            return
        if finish_count > len(self._as_pending_loads):
            raise RuntimeError(
                "AS-load cleanup count exceeds the local pending queue: "
                f"count={finish_count}, pending={len(self._as_pending_loads)}"
            )

        tasks = self._as_pending_loads[:finish_count]
        first_error = None
        for task in tasks:
            if not task.pages_released:
                try:
                    self._free_deferred_page_ids(task.deferred_page_ids)
                    task.pages_released = True
                except Exception as error:
                    logger.exception("Failed to release completed AS-load pages")
                    first_error = first_error or error
            if not task.lock_released:
                try:
                    if task.node is not None:
                        self.dec_lock_ref(task.node)
                    task.lock_released = True
                except Exception as error:
                    logger.exception("Failed to release completed AS-load lock")
                    first_error = first_error or error
            if first_error is not None:
                break

        cleanup_ok = first_error is None
        if self.tp_world_size > 1:
            cleanup_state = torch.tensor(int(cleanup_ok), dtype=torch.int)
            torch.distributed.all_reduce(
                cleanup_state,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
            cleanup_ok = bool(cleanup_state.item())

        if not cleanup_ok:
            raise RuntimeError(
                "AS-load cleanup failed; refusing to advance the TP FIFO or cache epoch"
            ) from first_error

        del self._as_pending_loads[:finish_count]

    def _queue_as_load_completion(
        self,
        result: _AsLoadResult,
        node: Optional[TreeNode],
        deferred_page_ids: List[torch.Tensor],
    ) -> None:
        if deferred_page_ids:
            page_ids = torch.cat(deferred_page_ids).contiguous()
        else:
            page_ids = torch.empty(0, dtype=torch.int64, device="cpu")

        if result.ready_event is None:
            # Synchronous load (e.g. single-rank): release immediately.
            self._free_deferred_page_ids(page_ids)
            return

        locked = False
        try:
            if node is not None:
                self.inc_lock_ref(node)
                locked = True
            self._as_pending_loads.append(
                _AsPendingLoad(
                    ready_event=result.ready_event,
                    node=node,
                    deferred_page_ids=page_ids,
                    resources=result.resources,
                )
            )
        except Exception:
            # Queue insertion is the ownership handoff. If it fails, make the
            # asynchronous writes quiescent before undoing temporary state.
            result.ready_event.synchronize()
            self._free_deferred_page_ids(page_ids)
            if locked:
                self.dec_lock_ref(node)
            raise

    def _free_deferred_page_ids(self, page_ids: torch.Tensor) -> None:
        """Release deferred page IDs back to the allocator.

        Uses free_page_ids when the allocator supports it (paged allocator);
        otherwise the overlap path is not active and there is nothing to defer.
        """
        if page_ids.numel() == 0:
            return
        free_page_ids = getattr(self.token_to_kv_pool_allocator, "free_page_ids", None)
        if free_page_ids is not None:
            free_page_ids(page_ids)
        else:
            logger.warning(
                "AS load: allocator lacks free_page_ids; "
                f"leaking {page_ids.numel()} deferred pages"
            )

    # ===== Internal: AS match =====

    def _insert_as_loaded(self, req, last_host_node, load_info, load_result):
        """Insert AS-loaded KV data into the radix tree for proper memory tracking.

        After a successful AS DMA read, inserts the loaded prefix into the tree
        so the tokens are tracked by evictable_size_. This prevents the memory
        leak where tokens allocated by init_load_back are untracked if the
        request is rejected by schedule_policy (early return after init_load_back).

        If the request IS scheduled, _req_inc_lock_ref will lock the node.
        If the request is rejected, the tokens remain as evictable tree entries.

        Returns (inserted_loaded_indices, new_last_node). Any excluded tail pages
        are released after the load's ready_event.
        """
        loaded_indices = load_result.indices
        loaded_page_ids = load_result.page_ids_cpu
        assert loaded_page_ids.device.type == "cpu"
        assert loaded_page_ids.numel() * self.page_size == loaded_indices.numel()
        deferred_page_ids = []

        local_hit_len = load_info["start_block"] * self.page_size
        total_prefix_len = local_hit_len + len(loaded_indices)

        assert len(req.prefix_indices) == local_hit_len, (
            f"prefix_indices length {len(req.prefix_indices)} != "
            f"local_hit_len {local_hit_len} (start_block={load_info['start_block']}, "
            f"page_size={self.page_size})"
        )

        # Include the boundary token reserved for EAGLE bigram key construction.
        token_ids = req.fill_ids[: total_prefix_len + int(self.is_eagle)]
        keys = page_align_keys(token_ids, self.page_size, is_bigram=self.is_eagle)
        radix_key = RadixKey(keys, req.extra_key, is_bigram=self.is_eagle)
        if len(radix_key) == 0:
            # No keys after page alignment. Keep the pages alive until the
            # asynchronous load has stopped writing them.
            deferred_page_ids.append(loaded_page_ids)
            self._queue_as_load_completion(load_result, None, deferred_page_ids)
            return (
                torch.empty(
                    (0,), dtype=loaded_indices.dtype, device=loaded_indices.device
                ),
                last_host_node,
            )

        # Build value: local prefix kv_indices + AS loaded kv_indices
        full_values = torch.cat([req.prefix_indices, loaded_indices])
        values = full_values[: len(radix_key)].to(dtype=torch.int64, copy=True)

        # Defer any loaded pages excluded by key alignment.
        tail_start = max(len(radix_key) - local_hit_len, 0)
        if tail_start < len(loaded_indices):
            deferred_page_ids.append(loaded_page_ids[tail_start // self.page_size :])
            loaded_indices = loaded_indices[:tail_start]
            loaded_page_ids = loaded_page_ids[: tail_start // self.page_size]

        # Insert into tree — _insert_helper walks existing local prefix nodes
        # and creates new nodes for the AS portion only.
        # evictable_size_ is updated automatically for new nodes.
        self.insert(
            InsertParams(
                key=radix_key,
                value=values,
            )
        )

        # Re-match to get the correct last_node after insertion.
        match_result = super().match_prefix(
            MatchPrefixParams(
                key=RadixKey(keys, req.extra_key, is_bigram=self.is_eagle),
            )
        )

        if match_result.last_device_node is None:
            logger.warning(
                "AS insert_loaded: re-match returned no last_node, "
                f"loaded={len(loaded_indices)}, keys={len(radix_key)}"
            )
            # Still queue the deferred tail release; no node to lock.
            self._queue_as_load_completion(load_result, None, deferred_page_ids)
            return loaded_indices, last_host_node

        logger.debug(
            f"AS insert_loaded: inserted {len(loaded_indices)} tokens into radix tree"
        )
        # Lock the inserted suffix node (and its ancestors) until the async
        # load completes so eviction cannot reclaim the freshly loaded rows.
        self._queue_as_load_completion(
            load_result, match_result.last_device_node, deferred_page_ids
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

    def _get_as_trace(self, req):
        if req is None:
            return None
        from sglang.srt.mem_cache.as_trace import get_recorder

        return get_recorder(req, self)

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

        as_match_count = 0
        match_error = None
        remote_match = bool(remote_len and len(remote_len) == 1 and remote_len[0] > 0)

        # If remote hit info indicates a specific hit length, use it directly
        if remote_match:
            as_match_count = max((remote_len[0] - local_hit_len) // self.page_size, 0)
            as_hit_tokens = as_match_count * self.page_size
            if as_hit_tokens <= 0:
                recorder = self._get_as_trace(req)
                if recorder is not None:
                    recorder.record(
                        "AS_MATCH",
                        raw_match_blocks=as_match_count,
                        final_match_blocks=0,
                        start_block=start_block,
                        hash_ids=recorder.hash_ids(as_block_hashs),
                        status="miss",
                    )
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
                match_error = str(e)
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
                recorder = self._get_as_trace(req)
                if recorder is not None:
                    recorder.record(
                        "AS_MATCH",
                        raw_match_blocks=as_match_count,
                        final_match_blocks=0,
                        start_block=start_block,
                        hash_ids=recorder.hash_ids(as_block_hashs),
                        status="error" if match_error else "miss",
                        error=match_error,
                    )
                return None

            as_hit_tokens = as_match_count * self.page_size

        # Reserve one matched page for EAGLE's shared target/draft bigram boundary.
        if self.is_eagle:
            as_hit_tokens -= self.page_size
            if as_hit_tokens <= 0:
                recorder = self._get_as_trace(req)
                if recorder is not None:
                    recorder.record(
                        "AS_MATCH",
                        raw_match_blocks=as_match_count,
                        final_match_blocks=0,
                        start_block=start_block,
                        hash_ids=recorder.hash_ids(as_block_hashs),
                        status="miss",
                    )
                return None

        # Cap AS hit so that at least 1 token remains for extend forward pass.
        # Without this, local_hit + AS_hit could cover all fill_ids, making
        # extend_input_len=0 and causing an empty-tensor crash in model forward.
        max_as_tokens = (
            (len(req.fill_ids) - local_hit_len - 1) // self.page_size * self.page_size
        )
        if max_as_tokens <= 0:
            recorder = self._get_as_trace(req)
            if recorder is not None:
                recorder.record(
                    "AS_MATCH",
                    raw_match_blocks=as_match_count,
                    final_match_blocks=0,
                    start_block=start_block,
                    hash_ids=recorder.hash_ids(as_block_hashs),
                    status="miss",
                )
            return None
        as_hit_tokens = min(as_hit_tokens, max_as_tokens)

        recorder = self._get_as_trace(req)
        if recorder is not None:
            recorder.record(
                "AS_MATCH",
                raw_match_blocks=as_match_count,
                final_match_blocks=as_hit_tokens // self.page_size,
                start_block=start_block,
                hash_ids=recorder.hash_ids(as_block_hashs),
                status="hit",
            )

        # Store load info on the last_device_node for init_load_back to use
        last_node = local_result.last_device_node
        last_node._as_load_req = req
        last_node._as_load_info = {
            "start_block": start_block,
            "as_hit_tokens": as_hit_tokens,
            "as_block_hashs": as_block_hashs,
            "req": req,
            "remote_addrs": remote_addrs,
            "remote_len": remote_len,
        }

        return MatchResult(
            device_indices=local_result.device_indices,
            last_device_node=local_result.last_device_node,
            last_host_node=last_node,  # used as load_back anchor
            host_hit_length=as_hit_tokens,
        )

    # ===== Internal: AS load (DMA read) =====

    def _resolve_layer_pool(self, layer_id: int):
        """Map virtual layer_id back to (kv_cache, phys_id). Draft layers are offset by target count."""
        if layer_id < self._target_layer_count:
            return self._main_kv_cache, layer_id
        return self._draft_kv_cache, layer_id - self._target_layer_count

    def _get_layer_kv_tensors(self, layer_id: int):
        """Fetch (k_buf, v_buf, v_is_page_indexed) for allgather. v_buf is None in MLA.

        DSA's V (index_k_with_scale_buffer) is page-indexed; MHA's V is token-indexed.
        """
        kv_cache, phys_id = self._resolve_layer_pool(layer_id)

        if hasattr(kv_cache, "_get_key_buffer"):
            k_buf = kv_cache._get_key_buffer(phys_id)
        else:
            k_buf = kv_cache.get_key_buffer(phys_id)

        v_buf = None
        v_is_page_indexed = False
        if self.attention_type != "MLA" and (
            self._indexer_layer_id_set is None or layer_id in self._indexer_layer_id_set
        ):
            if hasattr(kv_cache, "get_index_k_with_scale_buffer"):
                v_buf = kv_cache.get_index_k_with_scale_buffer(phys_id)
                v_is_page_indexed = True
            elif hasattr(kv_cache, "_get_value_buffer"):
                v_buf = kv_cache._get_value_buffer(phys_id)
            else:
                v_buf = kv_cache.get_value_buffer(phys_id)

        return k_buf, v_buf, v_is_page_indexed

    def _get_as_load_stream(self) -> torch.cuda.Stream:
        """Dedicated NCCL stream so load allgather runs concurrently with the
        SDK's scatter streams (true read/allgather overlap)."""
        if self._as_load_stream is None:
            self._as_load_stream = torch.cuda.Stream()
        return self._as_load_stream

    def _get_as_unpack_stream(self) -> torch.cuda.Stream:
        """Dedicated stream for Triton unpack of the gathered scratch, so it
        overlaps the next batch's NCCL allgather on the load stream."""
        if self._as_unpack_stream is None:
            self._as_unpack_stream = torch.cuda.Stream()
        return self._as_unpack_stream

    def _get_as_page_id_copy_stream(self) -> torch.cuda.Stream:
        """Dedicated stream for the pinned page-ID D2H copy."""
        if self._as_page_id_copy_stream is None:
            self._as_page_id_copy_stream = torch.cuda.Stream()
        return self._as_page_id_copy_stream

    def _stage_page_ids_to_host(self, page_rows: torch.Tensor, source_ready: Any):
        """Start a pinned D2H copy without blocking the scheduler thread."""
        count = page_rows.numel()
        if self._as_page_ids_host is None or self._as_page_ids_host.numel() < count:
            self._as_page_ids_host = torch.empty(
                count, dtype=torch.int64, device="cpu", pin_memory=True
            )

        copy_stream = self._get_as_page_id_copy_stream()
        copy_done = torch.cuda.Event()
        with torch.cuda.stream(copy_stream):
            copy_stream.wait_event(source_ready)
            self._as_page_ids_host[:count].copy_(page_rows, non_blocking=True)
            copy_done.record(copy_stream)
        return self._as_page_ids_host[:count], copy_done

    def _ensure_as_load_scratch(
        self,
        max_local_nbytes: int,
        world: int,
        device: torch.device,
        load_stream: torch.cuda.Stream,
    ) -> None:
        # Reusing gathered buffers is safe only after the previous request's
        # asynchronous unpack stream has finished with them.
        if self._as_scratch_ready_event is not None:
            load_stream.wait_event(self._as_scratch_ready_event)

        if (
            self._as_local_scratch is None
            or self._as_local_scratch.numel() < max_local_nbytes
        ):
            self._as_local_scratch = torch.empty(
                max_local_nbytes, dtype=torch.uint8, device=device
            )

        gathered_nbytes = max_local_nbytes * world
        if len(self._as_gathered_scratch) != 2 or any(
            buf.numel() < gathered_nbytes for buf in self._as_gathered_scratch
        ):
            self._as_gathered_scratch = [
                torch.empty(gathered_nbytes, dtype=torch.uint8, device=device)
                for _ in range(2)
            ]

    def _prepare_packed_as_load_plans(
        self,
        plans: List[_AsLoadBatchPlan],
        world: int,
        device: torch.device,
        load_stream: torch.cuda.Stream,
    ) -> torch.Tensor:
        """Build one metadata slab and bind cached descriptors/scratch."""
        metadata_rows = []
        metadata_spans = []
        for plan in plans:
            plan.inplace_pairs = None
            start = len(metadata_rows)
            metadata_rows.extend(
                [
                    segment.buffer.data_ptr(),
                    segment.local_rows.data_ptr(),
                    segment.full_rows.data_ptr(),
                    segment.row_stride_bytes,
                    segment.row_bytes,
                    segment.local_row_count,
                    segment.packed_offset,
                ]
                for segment in plan.segments
            )
            metadata_spans.append((start, len(metadata_rows)))
            plan.descriptors = build_as_load_descriptors(plan.segments, device)

        metadata = torch.tensor(metadata_rows, dtype=torch.int64, device=device)
        max_local_nbytes = max(plan.local_nbytes for plan in plans)
        self._ensure_as_load_scratch(max_local_nbytes, world, device, load_stream)

        for bi, (plan, (start, end)) in enumerate(zip(plans, metadata_spans)):
            plan.metadata = metadata[start:end]
            plan.local_scratch = self._as_local_scratch[: plan.local_nbytes]
            plan.gathered_scratch = self._as_gathered_scratch[bi % 2][
                : plan.local_nbytes * world
            ]

        warmup_plans = {}
        for plan in plans:
            current = warmup_plans.get(plan.vector_bytes)
            if current is None or plan.local_nbytes > current.local_nbytes:
                warmup_plans[plan.vector_bytes] = plan
        for warmup_plan in warmup_plans.values():
            warmup_as_load_kernels(
                warmup_plan.metadata,
                warmup_plan.descriptors,
                warmup_plan.local_scratch,
                warmup_plan.gathered_scratch,
                warmup_plan.local_nbytes,
                world,
                warmup_plan.vector_bytes,
            )
        return metadata

    def _prepare_as_load_plans(
        self,
        alloc_indices: torch.Tensor,
        page_rows: torch.Tensor,
        first_page: Optional[int],
        n_blocks: int,
        batches: List[List[int]],
        world: int,
        rank: int,
        allow_inplace: bool,
    ) -> List[_AsLoadBatchPlan]:
        """Build all GPU-visible metadata before request-level preflight."""
        chunk = n_blocks // world
        local_token_count = chunk * self.page_size
        local_token_start = rank * local_token_count
        local_token_rows = alloc_indices[
            local_token_start : local_token_start + local_token_count
        ]
        local_page_start = rank * chunk
        local_page_rows = page_rows[local_page_start : local_page_start + chunk]
        plans = []
        for layer_ids in batches:
            segments = []
            packed_offset = 0

            def append_segment(buffer, local_rows, full_rows):
                nonlocal packed_offset
                segment = make_as_load_segment(
                    buffer, local_rows, full_rows, packed_offset
                )
                segments.append(segment)
                packed_offset += segment.local_nbytes

            for layer_id in layer_ids:
                k_buf, v_buf, v_is_page_indexed = self._get_layer_kv_tensors(layer_id)
                append_segment(k_buf, local_token_rows, alloc_indices)
                if v_buf is not None:
                    append_segment(
                        v_buf,
                        local_page_rows if v_is_page_indexed else local_token_rows,
                        page_rows if v_is_page_indexed else alloc_indices,
                    )

            inplace_pairs = None
            if allow_inplace and first_page is not None:
                inplace_pairs = []
                for segment in segments:
                    pair = make_inplace_allgather_pair(
                        segment,
                        first_page,
                        n_blocks,
                        self.page_size,
                        world,
                        rank,
                    )
                    if pair is None:
                        inplace_pairs = None
                        break
                    inplace_pairs.append(pair)

            vector_bytes = select_as_load_vector_bytes(segments)
            plans.append(
                _AsLoadBatchPlan(
                    layer_ids=layer_ids,
                    segments=segments,
                    metadata=None,
                    descriptors=None,
                    local_nbytes=packed_offset,
                    vector_bytes=vector_bytes,
                    inplace_pairs=inplace_pairs,
                )
            )
        return plans

    @staticmethod
    def _pack_loaded_kv(plan: _AsLoadBatchPlan):
        if plan.inplace_pairs is not None:
            return
        pack_as_load_segments(
            plan.metadata,
            plan.descriptors,
            plan.local_scratch,
            plan.vector_bytes,
        )

    @staticmethod
    def _allgather_loaded_kv(
        nccl_tp,
        world: int,
        plan: _AsLoadBatchPlan,
        stream: Optional[torch.cuda.Stream] = None,
        unpack_stream: Optional[torch.cuda.Stream] = None,
    ):
        """Run an in-place grouped allgather or the packed fallback.

        For the packed fallback, Triton unpack is submitted on a dedicated
        unpack stream chained behind an ag_done event so it overlaps the next
        batch's NCCL launch. Returns (nccl_ms, unpack_ms, unpack_done_event).
        """
        _t0 = time.perf_counter()
        pynccl_comm = nccl_tp.pynccl_comm
        if stream is not None and pynccl_comm is not None and pynccl_comm.available:
            # PyNccl is disabled by default outside graph capture; enable it only
            # for this AS enqueue so the load stream stays async, then restore state.
            with pynccl_comm.change_state(enable=True, stream=stream):
                if plan.inplace_pairs is not None:
                    nccl_tp.cp_batch_all_gather_into_tensor_async(
                        plan.inplace_pairs, stream=stream
                    )
                else:
                    nccl_tp.cp_all_gather_into_tensor_async(
                        plan.gathered_scratch,
                        plan.local_scratch,
                        stream=stream,
                    )
        else:
            nccl_tp.all_gather_into_tensor(plan.gathered_scratch, plan.local_scratch)
        _nccl_ms = (time.perf_counter() - _t0) * 1000

        _t1 = time.perf_counter()
        unpack_done = None
        if plan.inplace_pairs is None:
            if unpack_stream is None:
                unpack_stream = stream or torch.cuda.current_stream()
            ag_done = torch.cuda.Event()
            ag_done.record(stream or torch.cuda.current_stream())
            unpack_done = torch.cuda.Event()
            with torch.cuda.stream(unpack_stream):
                unpack_stream.wait_event(ag_done)
                unpack_as_load_segments(
                    plan.metadata,
                    plan.descriptors,
                    plan.gathered_scratch,
                    plan.local_nbytes,
                    world,
                    plan.vector_bytes,
                )
                unpack_done.record(unpack_stream)
                plan.metadata.record_stream(unpack_stream)
                plan.descriptors.record_stream(unpack_stream)
                plan.gathered_scratch.record_stream(unpack_stream)
                for segment in plan.segments:
                    segment.local_rows.record_stream(unpack_stream)
                    segment.full_rows.record_stream(unpack_stream)
        _unpack_ms = (time.perf_counter() - _t1) * 1000
        return _nccl_ms, _unpack_ms, unpack_done

    def _do_as_load(
        self, load_info: dict, host_hit_length: int
    ) -> Optional[_AsLoadResult]:
        """Execute the actual AS -> GPU DMA read.

        Allocates HBM slots and calls ascli.read() to DMA load KV data.
        """
        if not self._buffer_ptrs_ready:
            logger.warning("AS load: buffer pointers not initialized, skipping")
            return None

        _t_load0 = time.perf_counter()  # Temporary diag: per-request load breakdown.
        _diag = {
            "indices": 0.0,
            "prepare": 0.0,
            "preflight_submit": 0.0,
            "fetch": 0.0,
            "acquire": 0.0,
            "preflight_wait": 0.0,
            "pipeline": 0.0,
            "sw": 0.0,
            "pack": 0.0,
            "nccl": 0.0,
            "unpack": 0.0,
            "remainder": 0.0,
            "handoff": 0.0,
            "tp_sync": 0.0,
        }

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

        # Prefer one physical run so NCCL can allgather directly into the KV
        # buffers. Custom/non-paged allocators retain the original behavior.
        used_contiguous_alloc = False
        data_id_list = None
        alloc_t0 = time.perf_counter()
        try:
            alloc_contiguous_with_page_ids = getattr(
                self.token_to_kv_pool_allocator,
                "alloc_contiguous_with_page_ids",
                None,
            )
            if alloc_contiguous_with_page_ids is not None:
                contiguous_result = alloc_contiguous_with_page_ids(as_hit_tokens)
                if contiguous_result is None:
                    alloc_indices = None
                else:
                    alloc_indices, data_id_list = contiguous_result
            else:
                alloc_contiguous = getattr(
                    self.token_to_kv_pool_allocator, "alloc_contiguous", None
                )
                alloc_indices = (
                    alloc_contiguous(as_hit_tokens)
                    if alloc_contiguous is not None
                    else None
                )
            if alloc_indices is not None:
                used_contiguous_alloc = True
            else:
                alloc_indices = self.token_to_kv_pool_allocator.alloc(as_hit_tokens)
        except Exception:
            logger.warning(f"AS load: failed to allocate {as_hit_tokens} tokens")
            return None
        alloc_ms = (time.perf_counter() - alloc_t0) * 1000

        if alloc_indices is None or len(alloc_indices) == 0:
            logger.warning(
                f"AS load: alloc failed, "
                f"need={as_hit_tokens}, "
                f"available={self.token_to_kv_pool_allocator.available_size()}, "
                f"evictable={self.evictable_size()}"
            )
            return None

        # Keep page rows on-device for pack/unpack metadata. The SDK still needs
        # host IDs, so stage them through pinned memory and overlap that wait with
        # plan construction instead of Tensor.tolist() on a CUDA tensor.
        used_cpu_page_ids = data_id_list is not None
        _t_indices = time.perf_counter()
        page_rows = alloc_indices[:: self.page_size] // self.page_size
        indices_ready_event = torch.cuda.Event()
        indices_ready_event.record(torch.cuda.current_stream(page_rows.device))
        page_ids_host = None
        page_ids_ready_event = None
        page_ids_failed = False
        if data_id_list is None:
            try:
                page_ids_host, page_ids_ready_event = self._stage_page_ids_to_host(
                    page_rows, indices_ready_event
                )
            except Exception:
                logger.exception(
                    "AS pinned page-ID staging failed; using synchronous fallback"
                )
                try:
                    data_id_list = page_rows.tolist()
                except Exception:
                    logger.exception("AS synchronous page-ID fallback failed")
                    data_id_list = []
                    page_ids_failed = True
        _diag["indices"] += (time.perf_counter() - _t_indices) * 1000

        def _materialize_data_id_list():
            nonlocal data_id_list, page_ids_failed
            if data_id_list is None:
                wait_t0 = time.perf_counter()
                try:
                    page_ids_ready_event.synchronize()
                    data_id_list = page_ids_host.tolist()
                except Exception:
                    logger.exception(
                        "AS asynchronous page-ID copy failed; using synchronous fallback"
                    )
                    try:
                        data_id_list = page_rows.tolist()
                    except Exception:
                        logger.exception("AS synchronous page-ID fallback failed")
                        data_id_list = []
                        page_ids_failed = True
                _diag["indices"] += (time.perf_counter() - wait_t0) * 1000
            return data_id_list

        # SDK main expects List[str]; pass None for local read.
        remote_addrs_arg = remote_addrs if remote_addrs else None

        # Split into floor (sharded across TP) + remainder (full-load on every rank).
        # SDK read() only supports contiguous slices, so we issue up to 2 reads per rank.
        # world/rank MUST come from the attention-TP group (same group used by the
        # _allgather_loaded_kv collective), otherwise the per-rank shard offsets and
        # the all-gather layout disagree when dp_size > 1 (global tp_rank 0..15 with
        # an attn-TP world of 8 would over-shard and index out of range).
        from sglang.srt.layers.dp_attention import get_attention_tp_group

        attn_tp = get_attention_tp_group()
        n_blocks = page_rows.numel()
        world = attn_tp.world_size
        rank = attn_tp.rank_in_group
        if world > 1 and n_blocks > 0:
            chunk = n_blocks // world
            n_shard = chunk * world
            n_remainder = n_blocks - n_shard
        else:
            chunk = 0
            n_shard = 0
            n_remainder = n_blocks

        # ---- read/allgather overlap pipeline ----
        # gRPC is layer-independent, so fetch once and replay the offsets. Each
        # batch issues one in-place allgather (contiguous pages) or one packed
        # allgather. The next scatter is submitted before the current NCCL launch,
        # placing it inside that NCCL window.
        # eff_k = n_layers disables batching (single shot): remote (avoid K x IB
        # fetch) or single-rank (no allgather to overlap).
        n_layers = len(self.layer_ids)
        if remote_addrs_arg is not None or world == 1 or chunk == 0:
            eff_k = n_layers
        else:
            eff_k = min(self._load_batch_layers, n_layers)
        batches = [self.layer_ids[i : i + eff_k] for i in range(0, n_layers, eff_k)]

        # Shard (floor) slice each rank reads — identical for every batch.
        local_lo = rank * chunk
        local_start_block = start_block + local_lo
        shard_indices = alloc_indices[: n_shard * self.page_size]
        shard_page_rows = page_rows[:n_shard]

        pynccl_comm = attn_tp.pynccl_comm
        pynccl_available = bool(
            pynccl_comm is not None and getattr(pynccl_comm, "available", False)
        )
        load_stream = self._get_as_load_stream()
        load_stream.wait_event(indices_ready_event)
        unpack_stream = self._get_as_unpack_stream()
        first_page = None
        pages_contiguous = False
        allow_inplace = False
        if self._as_allgather_mode != "packed" and n_shard > 0:
            candidate_ids = _materialize_data_id_list()[:n_shard]
            pages_contiguous = len(candidate_ids) == n_shard and all(
                candidate_ids[i] == candidate_ids[0] + i
                for i in range(1, len(candidate_ids))
            )
            allow_inplace = pynccl_available and pages_contiguous
            first_page = candidate_ids[0] if allow_inplace else None

        failed = page_ids_failed
        shard_offsets = None
        acquired_refs = 0
        released_refs = 0
        handles = [None] * len(batches)
        plans = []
        candidate_inplace_batches = 0
        local_preflight_state = 0
        preflight_state = 0
        packed_metadata = None
        unpack_slot_events = [None, None]
        last_unpack_event = None
        scatter_handles_quiescent = True

        def _prepare_packed_fallback():
            nonlocal packed_metadata
            packed_metadata = self._prepare_packed_as_load_plans(
                plans, world, alloc_indices.device, load_stream
            )

        # One gRPC for the whole sharded read; replayed per layer batch below.
        # Timeout must cover all batches' scatter + allgather, else the agent's
        # timeout checker auto-releases the SHM mid-load.
        if chunk > 0:
            # Prepare all plans, scratch, and first-use Triton kernels before fetch;
            # JIT compilation must not consume the shared-memory timeout window.
            _t_prepare = time.perf_counter()
            try:
                with torch.cuda.stream(load_stream):
                    plans = self._prepare_as_load_plans(
                        shard_indices,
                        shard_page_rows,
                        first_page,
                        n_shard,
                        batches,
                        world,
                        rank,
                        allow_inplace,
                    )
                    candidate_inplace_batches = sum(
                        plan.inplace_pairs is not None for plan in plans
                    )
                    if any(plan.inplace_pairs is None for plan in plans):
                        _prepare_packed_fallback()
                alloc_indices.record_stream(load_stream)
            except Exception as e:
                logger.exception(f"AS allgather preparation failed: {e}")
                failed = True
            _diag["prepare"] += (time.perf_counter() - _t_prepare) * 1000

            all_data_ids = _materialize_data_id_list()
            failed = failed or page_ids_failed or len(all_data_ids) != n_blocks
            local_data_ids = all_data_ids[local_lo : local_lo + chunk]
            shard_data_ids = all_data_ids[:n_shard]
            if self._as_allgather_mode == "packed":
                pages_contiguous = bool(shard_data_ids) and all(
                    shard_data_ids[i] == shard_data_ids[0] + i
                    for i in range(1, len(shard_data_ids))
                )
            elif self._as_allgather_mode == "inplace" and not allow_inplace:
                logger.debug(
                    "AS forced in-place mode unavailable; falling back to packed"
                )

            local_preflight_state = (
                0
                if failed
                else 2 if all(plan.inplace_pairs is not None for plan in plans) else 1
            )
            _t_preflight_submit = time.perf_counter()
            preflight_t = torch.tensor(preflight_state, dtype=torch.int)
            preflight_t.fill_(local_preflight_state)
            preflight_work = torch.distributed.all_reduce(
                preflight_t,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
                async_op=True,
            )
            _diag["preflight_submit"] += (
                time.perf_counter() - _t_preflight_submit
            ) * 1000

            # Overlap control-plane fetch with the cross-rank mode negotiation.
            # The collective completes before scatter, so every rank uses the
            # same NCCL sequence without waiting for the slowest RPC first.
            if local_preflight_state:
                try:
                    _t_fetch = time.perf_counter()
                    shard_offsets = self.ascli.read_fetch(
                        tokens=as_block_hashs,
                        start_transfer_block_index=local_start_block,
                        k_dest_ptrs=self.k_buffer_ptrs,
                        v_dest_ptrs=self.v_buffer_ptrs,
                        data_id_list=local_data_ids,
                        timeout=self._as_shard_fetch_timeout,
                        remote_addrs=remote_addrs_arg,
                    )
                    _diag["fetch"] += (time.perf_counter() - _t_fetch) * 1000
                except Exception as e:
                    logger.warning(f"AS shard fetch failed: {e}")
                    failed = True

                if shard_offsets is None:
                    failed = True

            _t_preflight_wait = time.perf_counter()
            preflight_work.wait()
            preflight_state = int(preflight_t.item())
            _diag["preflight_wait"] += (time.perf_counter() - _t_preflight_wait) * 1000
            if preflight_state == 0:
                failed = True
            elif preflight_state == 1:
                if any(plan.inplace_pairs is not None for plan in plans):
                    _t_prepare = time.perf_counter()
                    try:
                        with torch.cuda.stream(load_stream):
                            _prepare_packed_fallback()
                    except Exception as e:
                        logger.exception(
                            f"AS negotiated packed fallback preparation failed: {e}"
                        )
                        failed = True
                    _diag["prepare"] += (time.perf_counter() - _t_prepare) * 1000

            # Do not acquire SHM refs unless every rank can enter the pipeline.
            pipeline_state = torch.tensor(
                int(preflight_state != 0 and not failed), dtype=torch.int
            )
            torch.distributed.all_reduce(
                pipeline_state,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
            if not pipeline_state.item():
                failed = True
                preflight_state = 0

            # Each batch needs one ref released through its scatter handle.
            if preflight_state:
                try:
                    _t_acquire = time.perf_counter()
                    self.ascli.read_acquire(shard_offsets, len(batches))
                    _diag["acquire"] += (time.perf_counter() - _t_acquire) * 1000
                    acquired_refs = len(batches)
                except Exception as e:
                    logger.warning(f"AS shard acquire failed: {e}")
                    failed = True

            def _scatter(bi, dependency_stream=0):
                return self.ascli.read_scatter(
                    shard_offsets,
                    layer_ids=batches[bi],
                    refcounted=True,
                    dependency_stream=dependency_stream,
                )

            load_stream_ptr = load_stream.cuda_stream
            _t_pipeline = time.perf_counter()
            try:
                if preflight_state and not failed:
                    try:
                        handles[0] = _scatter(0)
                    except Exception as e:
                        logger.warning(f"AS shard scatter b0 failed: {e}")
                        failed = True

                for bi, plan in enumerate(plans if preflight_state else []):
                    with torch.cuda.stream(load_stream):
                        handle = handles[bi]
                        stream_wait_ok = False
                        _tsw = time.perf_counter()
                        if handle is None:
                            rd_cur = -1
                        else:
                            try:
                                rd_cur = self.ascli.read_stream_wait(
                                    handle, load_stream_ptr
                                )
                                stream_wait_ok = True
                            except Exception as e:
                                logger.warning(
                                    f"AS shard stream_wait b{bi} failed: {e}"
                                )
                                rd_cur = -1
                        _diag["sw"] += (time.perf_counter() - _tsw) * 1000
                        if rd_cur != chunk:
                            failed = True

                        # Queue pack after scatter(b), then submit scatter(b+1)
                        # before NCCL(b). SDK scatter uses its own slot streams.
                        _tpack = time.perf_counter()
                        try:
                            self._pack_loaded_kv(plan)
                        except Exception as e:
                            logger.exception(f"AS shard pack b{bi} failed: {e}")
                            failed = True
                        _diag["pack"] += (time.perf_counter() - _tpack) * 1000

                        nxt = bi + 1
                        # Submit all acquired refs so cleanup receives every handle.
                        if nxt < len(plans) and acquired_refs > 0:
                            try:
                                handles[nxt] = _scatter(nxt, load_stream_ptr)
                            except Exception as e:
                                logger.warning(f"AS shard scatter b{nxt} failed: {e}")
                                failed = True

                        # Always preserve the preflighted NCCL sequence, even
                        # after a local scatter/pack failure. Terminal MIN discards
                        # any data gathered from an invalid batch.
                        try:
                            slot = bi % 2
                            if (
                                plan.inplace_pairs is None
                                and unpack_slot_events[slot] is not None
                            ):
                                load_stream.wait_event(unpack_slot_events[slot])
                            _nc, _unpack, unpack_done = self._allgather_loaded_kv(
                                attn_tp,
                                world,
                                plan,
                                stream=load_stream,
                                unpack_stream=unpack_stream,
                            )
                            if unpack_done is not None:
                                unpack_slot_events[slot] = unpack_done
                                last_unpack_event = unpack_done
                            _diag["nccl"] += _nc
                            _diag["unpack"] += _unpack
                        except Exception as e:
                            logger.exception(f"AS shard allgather b{bi} failed: {e}")
                            failed = True
                        finally:
                            # Release needs the per-batch scatter handle (paired
                            # with read_stream_wait above), not the fetch offsets.
                            if (
                                handle is not None
                                and stream_wait_ok
                                and released_refs < acquired_refs
                            ):
                                try:
                                    self.ascli.read_release_on_stream(
                                        handle, load_stream_ptr
                                    )
                                    released_refs += 1
                                    handles[bi] = None
                                except Exception as e:
                                    logger.warning(
                                        f"AS shard stream release b{bi} failed: {e}"
                                    )
                                    failed = True
            finally:
                # Drain any scatter left unconsumed by an early loop exit; the
                # successful path never reaches here. sync_wait + release pairs.
                for bi, handle in enumerate(handles):
                    if handle is None:
                        continue
                    handles[bi] = None
                    try:
                        self.ascli.read_sync_wait(handle)
                    except Exception as e:
                        logger.warning(f"AS shard cleanup wait b{bi} failed: {e}")
                        failed = True
                        scatter_handles_quiescent = False
                        continue
                    if released_refs < acquired_refs:
                        try:
                            self.ascli.read_release(handle)
                            released_refs += 1
                            handles[bi] = None
                        except Exception as e:
                            logger.warning(
                                f"AS shard cleanup release b{bi} failed: {e}"
                            )
                            failed = True
            _diag["pipeline"] += (time.perf_counter() - _t_pipeline) * 1000

            if preflight_state == 1 and plans:
                # One request-level event covers every unpack queued above and
                # fences persistent scratch reuse by the next AS load.
                with torch.cuda.stream(unpack_stream):
                    last_unpack_event = torch.cuda.Event()
                    last_unpack_event.record(unpack_stream)
                self._as_scratch_ready_event = last_unpack_event

        # Remainder (replicated on every rank, no allgather).
        if n_remainder > 0 and not failed:
            _materialize_data_id_list()
            _t_remainder = time.perf_counter()
            try:
                rh = self.ascli.read_submit(
                    layer_ids=self.layer_ids,
                    tokens=as_block_hashs,
                    start_transfer_block_index=start_block + n_shard,
                    k_dest_ptrs=self.k_buffer_ptrs,
                    v_dest_ptrs=self.v_buffer_ptrs,
                    data_id_list=data_id_list[n_shard:],
                    timeout=1,
                    remote_addrs=remote_addrs_arg,
                )
                rem_rd = self.ascli.read_wait(rh)
            except Exception as e:
                logger.warning(f"AS remainder read failed: {e}")
                rem_rd = -1
            if rem_rd != n_remainder:
                failed = True
            _diag["remainder"] += (time.perf_counter() - _t_remainder) * 1000

        # Make the default/compute stream wait for all allgathers before it
        # consumes the loaded KV, without host-blocking here.
        ready_event = None
        if world > 1 and chunk > 0:
            _t_handoff = time.perf_counter()
            if preflight_state == 1 and last_unpack_event is not None:
                ready_event = last_unpack_event
            else:
                ready_event = torch.cuda.Event()
                ready_event.record(load_stream)
            torch.cuda.current_stream().wait_event(ready_event)
            _diag["handoff"] += (time.perf_counter() - _t_handoff) * 1000

        # Single all-or-nothing barrier across ranks.
        ok = 0 if failed else 1
        reclaim_safe = int(scatter_handles_quiescent)
        if world > 1:
            _t_tp_sync = time.perf_counter()
            ok_t = torch.tensor([ok, reclaim_safe], dtype=torch.int)
            torch.distributed.all_reduce(
                ok_t, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
            )
            ok, reclaim_safe = map(int, ok_t.tolist())
            _diag["tp_sync"] += (time.perf_counter() - _t_tp_sync) * 1000

        # Temporary diag: attribute the full host time, including terminal sync.
        logger.info(
            "AS do_as_load breakdown: total=%.1fms alloc=%.1f indices=%.1f "
            "prepare=%.1f preflight_submit=%.1f fetch=%.1f acquire=%.1f "
            "preflight_wait=%.1f pipeline=%.1f (sw=%.1f pack=%.1f nccl=%.1f "
            "unpack=%.1f) remainder=%.1f handoff=%.1f tp_sync=%.1f "
            "(n_batches=%d, inplace=%d, candidate=%d, chunk=%d, pynccl=%d, "
            "pages_contig=%d, contig_alloc=%d, cpu_page_ids=%d, mode=%s, "
            "preflight=%d->%d)",
            (time.perf_counter() - _t_load0) * 1000,
            alloc_ms,
            _diag["indices"],
            _diag["prepare"],
            _diag["preflight_submit"],
            _diag["fetch"],
            _diag["acquire"],
            _diag["preflight_wait"],
            _diag["pipeline"],
            _diag["sw"],
            _diag["pack"],
            _diag["nccl"],
            _diag["unpack"],
            _diag["remainder"],
            _diag["handoff"],
            _diag["tp_sync"],
            len(batches),
            sum(plan.inplace_pairs is not None for plan in plans),
            candidate_inplace_batches,
            chunk,
            pynccl_available,
            pages_contiguous,
            used_contiguous_alloc,
            used_cpu_page_ids,
            self._as_allgather_mode,
            local_preflight_state,
            preflight_state,
        )

        if ok == 0:
            # Both streams may still be writing the freed rows. Drain them
            # before returning any physical KV pages to the allocator.
            if world > 1 and chunk > 0:
                load_stream.synchronize()
                unpack_stream.synchronize()
            if not reclaim_safe:
                # Keep pages quarantined if an SDK scatter may still write them.
                raise RuntimeError(
                    "AS shard cleanup could not quiesce all scatter handles; "
                    "allocated KV pages were not returned to the allocator"
                )
            free_page_ids = getattr(
                self.token_to_kv_pool_allocator, "free_page_ids", None
            )
            failed_data_ids = _materialize_data_id_list()
            if free_page_ids is not None and len(failed_data_ids) == n_blocks:
                free_page_ids(
                    torch.tensor(
                        failed_data_ids,
                        dtype=torch.int64,
                        device="cpu",
                    )
                )
            else:
                self.token_to_kv_pool_allocator.free(alloc_indices)
            return None

        page_ids_cpu = torch.tensor(
            _materialize_data_id_list(), dtype=torch.int64, device="cpu"
        ).contiguous()
        req = load_info.get("req")
        recorder = self._get_as_trace(req)
        if recorder is not None:
            page_ids = page_ids_cpu.tolist()
            recorder.record(
                "AS_READ",
                hash_ids=recorder.hash_ids(as_block_hashs),
                data_id_list=page_ids,
                actual_blocks=len(page_ids),
                start_transfer_block_index=start_block,
                md5_after_read=recorder.collect_page_md5(page_ids),
                status="ok",
            )
        return _AsLoadResult(
            indices=alloc_indices,
            page_ids_cpu=page_ids_cpu,
            ready_event=ready_event,
            resources=(plans, packed_metadata, page_rows, alloc_indices),
        )

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
            md5_before_write = (
                task.trace_recorder.collect_page_md5(task.data_id_list, write=True)
                if task.trace_recorder is not None
                else None
            )
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
            if task.trace_recorder is not None:
                task.trace_recorder.record(
                    "AS_WRITE",
                    hash_ids=task.trace_recorder.hash_ids(task.as_block_hashs),
                    data_id_list=task.data_id_list,
                    start_transfer_block_index=0,
                    md5_before_write=md5_before_write,
                    status="ok",
                )
            return w_res
        except Exception as e:
            logger.warning(f"AS write failed: {e}")
            if task.trace_recorder is not None:
                task.trace_recorder.record(
                    "AS_WRITE",
                    hash_ids=task.trace_recorder.hash_ids(task.as_block_hashs),
                    data_id_list=task.data_id_list,
                    start_transfer_block_index=0,
                    status="error",
                    error=str(e),
                )
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
            kv_committed_len = len(req.origin_input_ids) + len(req.output_ids)
            token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]

            # TODO(@duanbing02): input_ids + output_ids 是 page_size 整倍数时写入会越界，暂时只写 input
            num_pages = min(
                len(token_ids) // self.page_size,
                len(req.origin_input_ids) // self.page_size,
            )
            if num_pages <= 0:
                if self.tp_world_size > 1:
                    self._enqueue_dummy_write()
                return
            aligned_tokens = num_pages * self.page_size

            # Get KV indices for this request (token-level pool slot indices)
            kv_indices = snapshot[:aligned_tokens]

            # Convert to page-boundary indices: one index per block for AS SDK
            data_id_list = (kv_indices[:: self.page_size] // self.page_size).tolist()

            # Get write_params from request's AS parameters (passed from upstream)
            # Default to empty dict if None – SDK requires a python/numpy/pytorch object
            write_params = getattr(req, "as_kv_cache_params", None) or {}

            # Find the node covering the written range for lock_ref management
            # Use match_prefix to find the node that holds our KV data
            keys = page_align_keys(token_ids, self.page_size, is_bigram=self.is_eagle)
            match_result = super().match_prefix(
                MatchPrefixParams(
                    key=RadixKey(keys, req.extra_key, is_bigram=self.is_eagle),
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
                as_block_hashs=req.as_block_hashs,
                data_id_list=data_id_list,
                k_buffer_ptrs=self.k_buffer_ptrs,
                v_buffer_ptrs=self.v_buffer_ptrs,
                write_params=write_params,
                submit_time=time.perf_counter(),
                num_pages=num_pages,
                trace_recorder=self._get_as_trace(req),
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
