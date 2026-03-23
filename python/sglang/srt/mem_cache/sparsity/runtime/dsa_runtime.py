"""DSA runtime for decode disaggregation workers.

Prompt pages land on the host KV pool directly during PD transfer. Page-aligned
history can later be canonicalized from the live device tail into host, while
the current partial tail page stays on device so decode append can keep using
the paged main allocator.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.mem_cache.sparsity.core.sparse_cache_allocator import (
    SparseCacheAllocator,
)
from sglang.srt.mem_cache.sparsity.kernel.prepare_dsa_h2d import (
    DSAGraphWorkspace,
    detect_dsa_misses,
    remap_dsa_topk,
)
from sglang.srt.mem_cache.sparsity.runtime.indexer_namespace import (
    IndexerSlotNamespace,
)
from sglang.srt.mem_cache.sparsity.core.page_table import PageTable

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.chunk_cache import ChunkCache
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


@dataclass
class DSAPrecomputedMetadata:
    """Per-step metadata computed OUTSIDE the cuda graph and consumed INSIDE it.

    The cuda graph replays a fixed kernel sequence — it cannot run Python or call
    IndexerSlotNamespace.  So we materialize these tensors beforehand in
    prepare_decode_step() and stash them on ForwardBatch.dsa_precomputed for
    the graph-captured kernels to read.
    """

    indexer_out_cache_loc: Optional[torch.Tensor] = None
    indexer_page_table_1: Optional[torch.Tensor] = None
    indexer_real_page_table: Optional[torch.Tensor] = None
    return_logical_topk: bool = True


@dataclass
class DSAGraphDecodeBuffers:
    """Pre-allocated device buffers whose addresses are baked into the cuda graph.

    During graph capture we record kernels that read from these buffers.  On each
    replay we copy fresh metadata into the SAME tensors so the captured addresses
    stay valid — this is the standard cuda-graph "pointer-stable buffer" pattern.
    """

    max_bs: int
    max_context_len: int
    max_real_cols: int
    indexer_out_cache_loc: torch.Tensor
    indexer_page_table_1: torch.Tensor
    indexer_real_page_table: torch.Tensor

    @classmethod
    def create(
        cls,
        *,
        max_bs: int,
        max_context_len: int,
        page_size: int,
        device: torch.device,
    ) -> "DSAGraphDecodeBuffers":
        max_real_cols = (max_context_len + page_size - 1) // page_size
        return cls(
            max_bs=max_bs,
            max_context_len=max_context_len,
            max_real_cols=max_real_cols,
            indexer_out_cache_loc=torch.zeros(max_bs, dtype=torch.int64, device=device),
            indexer_page_table_1=torch.zeros(
                (max_bs, max_context_len), dtype=torch.int32, device=device
            ),
            indexer_real_page_table=torch.zeros(
                (max_bs, max_real_cols), dtype=torch.int32, device=device
            ),
        )


class DSARuntime:
    """Orchestrate DSA offload on decode workers.

    Lifecycle: after the prefill worker sends the KV cache via disaggregation,
    on_disagg_cache_received() finalizes prompt residency on the host KV pool,
    materializes the trailing partial page back to device, and initializes the
    page table. During each decode step:

      1. prepare_decode_step()  — precompute indexer metadata outside the graph
      2. remap_selected_indices() — detect misses, pull pages from host, remap
         topk indices to physical addresses (runs inside the captured graph)

    The sparse cache is a fixed carve-out from the main KV pool.  Eviction is
    LRU-based (via _page_last_access), with pages that are currently selected
    ("protected") excluded from eviction candidates.
    """

    def __init__(
        self,
        page_table: PageTable,
        indexer_slots: IndexerSlotNamespace,
        main_allocator,
        sparse_cache_allocator: SparseCacheAllocator,
        host_kv_pool,
        req_to_token_pool,
        kv_pool,
        workspace: DSAGraphWorkspace,
        page_size: int,
        start_layer: int,
        end_layer: int,
        device: torch.device,
    ):
        self.page_table = page_table
        self.indexer_slots = indexer_slots
        self.main_allocator = main_allocator
        self.sparse_cache_allocator = sparse_cache_allocator
        self.host_kv_pool = host_kv_pool
        self.req_to_token_pool = req_to_token_pool
        self.kv_pool = kv_pool
        self.workspace = workspace
        self.page_size = page_size
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.device = device
        self.max_context_len = req_to_token_pool.max_context_len

        self.released_prefix_len = torch.zeros(
            req_to_token_pool.req_to_token.shape[0],
            dtype=torch.int64,
            device=device,
        )
        self._hosted_page_count = torch.zeros(
            req_to_token_pool.req_to_token.shape[0],
            dtype=torch.int64,
            device=device,
        )
        self._page_offsets = torch.arange(page_size, dtype=torch.int64, device=device)
        self._active_indexer_out_cache_loc: Optional[torch.Tensor] = None
        self._cache_step = torch.zeros(1, dtype=torch.int64, device=device)
        self._miss_position_cache = torch.empty(0, dtype=torch.int64, device=device)
        self._graph_decode_buffers: Optional[DSAGraphDecodeBuffers] = None
        self._reserved_page_starts = sparse_cache_allocator.reserved_page_starts
        self._page_owner_req = torch.full(
            (sparse_cache_allocator.capacity,), -1, dtype=torch.int32, device=device
        )
        self._page_owner_logical = torch.full(
            (sparse_cache_allocator.capacity,), -1, dtype=torch.int32, device=device
        )
        self._page_last_access = torch.full(
            (sparse_cache_allocator.capacity,), -1, dtype=torch.int64, device=device
        )
        self._protected_slot_mask = torch.zeros(
            (sparse_cache_allocator.capacity,), dtype=torch.bool, device=device
        )
        self._protected_slot_counts = torch.zeros(
            (sparse_cache_allocator.capacity,), dtype=torch.int32, device=device
        )
        self._dense_page_to_slot = torch.full(
            (kv_pool.size // page_size + 1,),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self._dense_page_to_slot[
            torch.div(self._reserved_page_starts, page_size, rounding_mode="floor")
        ] = torch.arange(
            sparse_cache_allocator.capacity,
            dtype=torch.int32,
            device=device,
        )
        self._slot_indices = torch.arange(
            sparse_cache_allocator.capacity, dtype=torch.int64, device=device
        )
        # Sentinel base for invalid miss sort keys.  Valid keys are
        # req_idx * max_pages + logical_page — always < this value.  Invalid
        # entries get keys >= _invalid_miss_key_base so they sort to the end
        # without colliding with any real (req, page) pair.
        self._invalid_miss_key_base = (
            req_to_token_pool.req_to_token.shape[0] * self.page_table.max_pages_per_req + 1
        )
        if not hasattr(host_kv_pool, "data_ptrs") or not hasattr(kv_pool, "data_ptrs"):
            raise RuntimeError("DSA runtime requires layer-first MLA KV pools with data_ptrs")

    @classmethod
    def create(
        cls,
        *,
        req_to_token_pool,
        main_allocator,
        sparse_cache_allocator: SparseCacheAllocator,
        host_kv_pool,
        kv_pool,
        workspace: DSAGraphWorkspace,
        page_size: int,
        start_layer: int,
        end_layer: int,
        device: torch.device,
    ) -> "DSARuntime":
        max_pages_per_req = (
            req_to_token_pool.max_context_len + page_size - 1
        ) // page_size
        page_table = PageTable(
            max_pool_size=req_to_token_pool.req_to_token.shape[0],
            max_pages_per_req=max_pages_per_req,
            device=device,
        )
        indexer_slots = IndexerSlotNamespace(
            indexer_pool=kv_pool,
            page_size=page_size,
            device=device,
            req_to_token_shape=req_to_token_pool.req_to_token.shape,
        )
        return cls(
            page_table=page_table,
            indexer_slots=indexer_slots,
            main_allocator=main_allocator,
            sparse_cache_allocator=sparse_cache_allocator,
            host_kv_pool=host_kv_pool,
            req_to_token_pool=req_to_token_pool,
            kv_pool=kv_pool,
            workspace=workspace,
            page_size=page_size,
            start_layer=start_layer,
            end_layer=end_layer,
            device=device,
        )

    def on_disagg_cache_received(self, req: "Req", token_to_kv_pool_allocator) -> None:
        """Finalize prompt residency after the disagg transfer lands on the host pool."""
        del token_to_kv_pool_allocator
        if req.req_pool_idx is None:
            return
        if getattr(req, "_dsa_prompt_finalized", False):
            return

        token_count = len(req.origin_input_ids)
        page_aligned_prefix_len = (token_count // self.page_size) * self.page_size
        host_indices = getattr(req, "_dsa_prompt_host_indices", None)
        host_backing_len = int(getattr(req, "_dsa_prompt_host_backing_len", 0))
        if host_indices is None or host_backing_len < token_count:
            raise RuntimeError(
                "DSA prompt finalize requires host-backed transfer indices prepared in decode pre-allocation"
            )

        num_host_pages = host_backing_len // self.page_size
        if num_host_pages > 0:
            full_host_pages = page_aligned_prefix_len // self.page_size
            if full_host_pages > 0:
                logical_pages = torch.arange(
                    full_host_pages, dtype=torch.int64, device=self.device
                )
                host_page_starts = host_indices[:: self.page_size][:full_host_pages].to(
                    device=self.device, dtype=torch.int64
                )
                self.page_table.set_host_slots(
                    req.req_pool_idx, logical_pages, host_page_starts
                )
                self.page_table.gpu_slot[req.req_pool_idx, logical_pages] = -1

        partial_prompt_len = token_count - page_aligned_prefix_len
        if partial_prompt_len > 0:
            partial_loc = self._alloc_device_tokens(partial_prompt_len)
            self._copy_host_tokens_to_device(
                host_indices=host_indices[page_aligned_prefix_len:token_count],
                device_indices=partial_loc,
            )
            self.req_to_token_pool.write(
                (
                    req.req_pool_idx,
                    slice(page_aligned_prefix_len, token_count),
                ),
                partial_loc.to(torch.int32),
            )
            partial_host_indices = host_indices[
                page_aligned_prefix_len:host_backing_len
            ].to(torch.int64)
            self.host_kv_pool.free(partial_host_indices)

        self.indexer_slots.prepare_received_slots(req.req_pool_idx, token_count)
        self.released_prefix_len[req.req_pool_idx] = page_aligned_prefix_len
        self._hosted_page_count[req.req_pool_idx] = page_aligned_prefix_len // self.page_size
        req._dsa_prompt_host_indices = None
        req._dsa_prompt_host_backing_len = 0
        req._dsa_prompt_finalized = True
        logger.info(
            "DSA host-finalize req=%s req_pool_idx=%s prompt_tokens=%d hosted_prefix_tokens=%d",
            getattr(req, "rid", "unknown"),
            req.req_pool_idx,
            token_count,
            page_aligned_prefix_len,
        )

    def on_request_begin(self, req: "Req") -> None:
        """Initialize page table and indexer slots for a new request.

        Must be called after req_pool_idx is assigned so all per-request
        tensors can be indexed.
        """
        if req.req_pool_idx is None:
            return
        self.page_table.register(req.req_pool_idx)
        self.indexer_slots.register_request(req.req_pool_idx)
        self.released_prefix_len[req.req_pool_idx] = 0
        self._hosted_page_count[req.req_pool_idx] = 0
        self._active_indexer_out_cache_loc = None
        req._dsa_prompt_host_indices = None
        req._dsa_prompt_host_backing_len = 0
        req._dsa_prompt_finalized = False

    def remap_selected_indices(
        self,
        selected_indices: torch.Tensor,
        forward_batch: "ForwardBatch",
    ) -> torch.Tensor:
        """Detect cache misses, materialize them, then remap to physical addresses.

        Execution order invariant (do NOT reorder):
          1. detect_dsa_misses  — kernel scans topk, records misses via gpu_slot
          2. _materialize_current_misses — Python loads pages, updates gpu_slot
          3. remap_dsa_topk — kernel reads the now-updated gpu_slot for remapping
        """
        if not forward_batch.forward_mode.is_decode():
            raise RuntimeError("DSA remap only supports decode mode")

        self._cache_step.add_(1)
        selected_i32, req_pool_indices_i32, seq_lens_i32 = detect_dsa_misses(
            selected_indices=selected_indices,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            page_table=self.page_table,
            page_size=self.page_size,
            workspace=self.workspace,
        )
        self._materialize_current_misses(
            selected_i32,
            req_pool_indices_i32,
            seq_lens_i32,
        )
        remapped = remap_dsa_topk(
            selected_indices_i32=selected_i32,
            req_pool_indices_i32=req_pool_indices_i32,
            seq_lens_i32=seq_lens_i32,
            page_table=self.page_table,
            req_to_token=self.req_to_token_pool.req_to_token,
            page_size=self.page_size,
            workspace=self.workspace,
        )
        self._touch_sparse_token_indices(remapped)
        return remapped

    def on_request_end(self, req: "Req") -> None:
        """Release all DSA resources: sparse cache slots, host memory, page table, indexer slots."""
        if req.req_pool_idx is None:
            return

        prompt_host_indices = getattr(req, "_dsa_prompt_host_indices", None)
        prompt_host_backing_len = int(getattr(req, "_dsa_prompt_host_backing_len", 0))
        if prompt_host_indices is not None and prompt_host_backing_len > 0:
            self.host_kv_pool.free(prompt_host_indices[:prompt_host_backing_len].cpu())
            req._dsa_prompt_host_indices = None
            req._dsa_prompt_host_backing_len = 0

        gpu_slots = self.page_table.get_cached_gpu_slots(req.req_pool_idx)
        if gpu_slots is not None and gpu_slots.numel() > 0:
            self._release_sparse_pages(gpu_slots)

        host_slots = self.page_table.get_all_host_slots(req.req_pool_idx)
        if host_slots is not None and host_slots.numel() > 0:
            host_token_indices = self._expand_page_starts_to_token_indices(
                host_slots
            ).cpu()
            self.host_kv_pool.free(host_token_indices)

        self.page_table.clear(req.req_pool_idx)
        self.indexer_slots.clear_request(req.req_pool_idx)
        self.released_prefix_len[req.req_pool_idx] = 0
        self._hosted_page_count[req.req_pool_idx] = 0
        self._active_indexer_out_cache_loc = None
        req._dsa_prompt_finalized = False

    def on_request_end_finish(
        self, req: "Req", tree_cache: "ChunkCache"
    ) -> tuple[bool, Optional[int]]:
        """Read the released prefix length before on_request_end frees the page table.

        Returns (bypass_tree_cache, released_prefix_len).  The caller (coordinator)
        must call this BEFORE on_request_end to read the still-valid state.
        """
        from sglang.srt.mem_cache.chunk_cache import ChunkCache

        unwrapped_tree_cache = self._unwrap_tree_cache(tree_cache)
        if req.req_pool_idx is None or not isinstance(unwrapped_tree_cache, ChunkCache):
            return False, None

        released_prefix_len = int(self.released_prefix_len[req.req_pool_idx].item())
        return True, released_prefix_len

    def init_cuda_graph_state(self, max_bs: int, max_topk: int) -> None:
        """Freeze workspace and allocate graph-stable buffers before graph capture.

        After this call, workspace capacity is frozen and all DSA buffer addresses
        are stable — the cuda graph can record kernels that read/write them.
        """
        if max_bs <= 0:
            return
        if max_bs * max_topk > self.sparse_cache_allocator.capacity:
            raise RuntimeError(
                "DSA cuda graph requires sparse cache capacity to cover the capture-time "
                f"worst-case miss set: {max_bs=} {max_topk=} "
                f"capacity={self.sparse_cache_allocator.capacity}"
            )
        self._graph_decode_buffers = DSAGraphDecodeBuffers.create(
            max_bs=max_bs,
            max_context_len=self.max_context_len,
            page_size=self.page_size,
            device=self.device,
        )
        self.workspace.freeze_capacity(max_bs, max_topk)
        self._ensure_miss_position_cache(self.workspace.max_misses)

    def prepare_decode_step(self, forward_batch: "ForwardBatch") -> None:
        """Precompute indexer metadata OUTSIDE the cuda graph for the current step.

        Python-level IndexerSlotNamespace operations cannot run inside a captured
        graph, so we materialize the results here and stash them on forward_batch
        for graph-captured kernels to consume.
        """
        if not forward_batch.forward_mode.is_decode():
            return
        self._canonicalize_completed_tail_pages(forward_batch)
        precomputed = self._build_decode_precomputed(forward_batch)
        forward_batch.dsa_precomputed = precomputed
        self._active_indexer_out_cache_loc = (
            None if precomputed is None else precomputed.indexer_out_cache_loc
        )

    def get_indexer_metadata(
        self,
        page_table_1: torch.Tensor,
        forward_batch: "ForwardBatch",
    ) -> tuple[bool, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not forward_batch.forward_mode.is_decode():
            return False, None, None
        if forward_batch.dsa_precomputed is not None:
            return (
                forward_batch.dsa_precomputed.return_logical_topk,
                forward_batch.dsa_precomputed.indexer_page_table_1,
                forward_batch.dsa_precomputed.indexer_real_page_table,
            )
        indexer_page_table_1, indexer_real_page_table = (
            self.indexer_slots.get_page_tables(page_table_1, forward_batch)
        )
        return True, indexer_page_table_1, indexer_real_page_table

    def get_indexer_write_loc(self, forward_batch: "ForwardBatch") -> Optional[torch.Tensor]:
        if not forward_batch.forward_mode.is_decode():
            return None
        if forward_batch.dsa_precomputed is not None:
            return forward_batch.dsa_precomputed.indexer_out_cache_loc
        return self._active_indexer_out_cache_loc

    def get_disagg_state_indices(
        self,
        req_pool_idx: int,
        token_count: int,
        fallback_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del fallback_indices
        return self.indexer_slots.prepare_received_slots(req_pool_idx, token_count)

    def get_reserved_token_count(self) -> int:
        return self.sparse_cache_allocator.capacity * self.page_size

    def get_indexer_available_token_count(self) -> int:
        return self.indexer_slots.available_size()

    def get_host_available_token_count(self) -> int:
        return self.host_kv_pool.available_size()

    def get_host_kv_pool(self):
        return self.host_kv_pool

    def _alloc_device_tokens(self, token_count: int) -> torch.Tensor:
        if token_count <= 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        if self.main_allocator.page_size == 1:
            device_indices = self.main_allocator.alloc(token_count)
        else:
            device_indices = self.main_allocator.alloc_extend(
                prefix_lens=torch.tensor([0], dtype=torch.int64, device=self.device),
                prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
                seq_lens=torch.tensor([token_count], dtype=torch.int64, device=self.device),
                seq_lens_cpu=torch.tensor([token_count], dtype=torch.int64),
                last_loc=torch.tensor([-1], dtype=torch.int64, device=self.device),
                extend_num_tokens=token_count,
            )
        if device_indices is None:
            raise RuntimeError(f"Failed to allocate device tokens for DSA: {token_count=}")
        return device_indices.to(torch.int64)

    def _copy_host_tokens_to_device(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
    ) -> None:
        if host_indices.numel() == 0:
            return
        from sgl_kernel.kvcacheio import transfer_kv_per_layer_mla

        host_indices_i64 = host_indices.to(device=self.device, dtype=torch.int64)
        device_indices_i64 = device_indices.to(device=self.device, dtype=torch.int64)
        for layer_id in range(self.start_layer, self.end_layer):
            local_layer = layer_id - self.start_layer
            transfer_kv_per_layer_mla(
                src=self.host_kv_pool.kv_buffer[local_layer],
                dst=self.kv_pool.kv_buffer[local_layer],
                src_indices=host_indices_i64,
                dst_indices=device_indices_i64,
                item_size=self.host_kv_pool.token_stride_size,
            )

    def _copy_device_tokens_to_host(
        self,
        device_indices: torch.Tensor,
        host_indices: torch.Tensor,
    ) -> None:
        if device_indices.numel() == 0:
            return
        from sgl_kernel.kvcacheio import transfer_kv_per_layer_mla

        device_indices_i64 = device_indices.to(torch.int64)
        host_indices_i64 = host_indices.to(device=self.device, dtype=torch.int64)
        for layer_id in range(self.start_layer, self.end_layer):
            local_layer = layer_id - self.start_layer
            transfer_kv_per_layer_mla(
                src=self.kv_pool.kv_buffer[local_layer],
                dst=self.host_kv_pool.kv_buffer[local_layer],
                src_indices=device_indices_i64,
                dst_indices=host_indices_i64,
                item_size=self.host_kv_pool.token_stride_size,
            )

    def _canonicalize_completed_tail_pages(self, forward_batch: "ForwardBatch") -> None:
        if forward_batch.seq_lens_cpu is None:
            return
        req_pool_indices = forward_batch.req_pool_indices.to(torch.long)
        for batch_idx, req_pool_idx in enumerate(req_pool_indices.tolist()):
            seq_len = int(forward_batch.seq_lens_cpu[batch_idx].item())
            if seq_len <= 0:
                continue
            current_hosted_pages = int(self._hosted_page_count[req_pool_idx].item())
            target_hosted_pages = (seq_len - 1) // self.page_size
            if target_hosted_pages <= current_hosted_pages:
                continue

            start_token = current_hosted_pages * self.page_size
            end_token = target_hosted_pages * self.page_size
            device_indices = self.req_to_token_pool.req_to_token[
                req_pool_idx, start_token:end_token
            ].to(torch.int64)
            if device_indices.numel() == 0:
                continue
            if bool((device_indices <= 0).any().item()):
                raise RuntimeError(
                    f"Cannot canonicalize non-device tail page: req_pool_idx={req_pool_idx}, "
                    f"start_token={start_token}, end_token={end_token}"
                )

            host_indices = self.host_kv_pool.alloc(end_token - start_token)
            if host_indices is None:
                raise RuntimeError(
                    f"Failed to allocate host slots for DSA tail canonicalize: req_pool_idx={req_pool_idx}, "
                    f"num_tokens={end_token - start_token}"
                )
            self._copy_device_tokens_to_host(device_indices=device_indices, host_indices=host_indices)

            logical_pages = torch.arange(
                current_hosted_pages,
                target_hosted_pages,
                dtype=torch.int64,
                device=self.device,
            )
            host_page_starts = host_indices[:: self.page_size].to(
                device=self.device, dtype=torch.int64
            )
            self.page_table.set_host_slots(req_pool_idx, logical_pages, host_page_starts)
            self.page_table.gpu_slot[req_pool_idx, logical_pages] = -1
            self.req_to_token_pool.req_to_token[req_pool_idx, start_token:end_token].zero_()
            self.main_allocator.free(device_indices)
            self.released_prefix_len[req_pool_idx] = end_token
            self._hosted_page_count[req_pool_idx] = target_hosted_pages

    def _materialize_current_misses(
        self,
        selected_indices_i32: torch.Tensor,
        req_pool_indices_i32: torch.Tensor,
        seq_lens_i32: torch.Tensor,
    ) -> None:
        """Pull missing pages from host into the sparse device cache.

        Pipeline: validate misses → sort by (req, page) → deduplicate →
        exclude already-cached → assign eviction victims → apply plan + DMA.
        """
        ws = self.workspace
        if ws.max_misses == 0:
            return

        protected_slot_mask = self._get_current_hit_slot_mask(
            selected_indices_i32,
            req_pool_indices_i32,
            seq_lens_i32,
        )
        valid_miss_mask, miss_positions = self._get_valid_miss_mask(ws)
        sorted_req, sorted_pages, sorted_hosts, sorted_valid_mask, sorted_keys = (
            self._sort_misses(ws, valid_miss_mask, miss_positions)
        )
        load_miss_mask = self._get_load_miss_mask(
            ws,
            sorted_req,
            sorted_pages,
            sorted_valid_mask,
            sorted_keys,
        )
        self._assign_slot_indices(ws, load_miss_mask, protected_slot_mask)
        self._apply_load_plan_and_transfer(ws, load_miss_mask, sorted_hosts)

    def _get_valid_miss_mask(
        self, ws: DSAGraphWorkspace
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build a boolean mask selecting positions < miss_count (valid entries).

        The detect_misses kernel writes a variable number of entries but the
        workspace has a fixed max_misses capacity; positions beyond miss_count
        are padding and must be excluded from downstream processing.
        """
        self._ensure_miss_position_cache(ws.max_misses)
        miss_positions = self._miss_position_cache[: ws.max_misses]
        miss_count = ws.miss_count_buf[0].to(torch.int64)
        valid_miss_mask = miss_positions < miss_count
        ws.miss_valid_mask_buf.copy_(valid_miss_mask)
        return valid_miss_mask, miss_positions

    def _sort_misses(
        self,
        ws: DSAGraphWorkspace,
        valid_miss_mask: torch.Tensor,
        miss_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sort misses by composite key (req_idx * max_pages + logical_page).

        Sorting brings identical (req, page) pairs together so that the next
        stage can deduplicate with a simple adjacent-pair comparison.  Invalid
        entries get keys > _invalid_miss_key_base to sort last.
        """
        miss_req = torch.where(valid_miss_mask, ws.miss_req_indices_buf[: ws.max_misses], 0)
        miss_pages = torch.where(
            valid_miss_mask,
            ws.miss_logical_pages_buf[: ws.max_misses],
            0,
        )
        miss_hosts = torch.where(valid_miss_mask, ws.miss_host_slots_buf[: ws.max_misses], 0)

        miss_keys = miss_req.to(torch.int64) * self.page_table.max_pages_per_req + miss_pages.to(
            torch.int64
        )
        miss_keys = torch.where(
            valid_miss_mask,
            miss_keys,
            self._invalid_miss_key_base + miss_positions,
        )
        miss_order = torch.argsort(miss_keys, stable=True)

        sorted_req = miss_req[miss_order]
        sorted_pages = miss_pages[miss_order]
        sorted_hosts = miss_hosts[miss_order]
        sorted_valid_mask = valid_miss_mask[miss_order]
        sorted_keys = miss_keys[miss_order]

        ws.sorted_miss_req_indices_buf.copy_(sorted_req)
        ws.sorted_miss_logical_pages_buf.copy_(sorted_pages)
        ws.sorted_miss_host_slots_buf.copy_(sorted_hosts)
        return sorted_req, sorted_pages, sorted_hosts, sorted_valid_mask, sorted_keys

    def _get_load_miss_mask(
        self,
        ws: DSAGraphWorkspace,
        sorted_req: torch.Tensor,
        sorted_pages: torch.Tensor,
        sorted_valid_mask: torch.Tensor,
        sorted_keys: torch.Tensor,
    ) -> torch.Tensor:
        """Identify unique misses that actually need a host→device transfer.

        Two filters: (1) deduplicate — adjacent pairs with the same key keep only
        the first; (2) exclude pages already in the GPU sparse cache (gpu_slot >= 0).
        """
        unique_miss_mask = sorted_valid_mask.clone()
        if ws.max_misses > 1:
            unique_miss_mask[1:] &= sorted_keys[1:] != sorted_keys[:-1]
        ws.unique_miss_mask_buf.copy_(unique_miss_mask)

        current_gpu_slot = self.page_table.gpu_slot[
            sorted_req.to(torch.int64), sorted_pages.to(torch.int64)
        ]
        load_miss_mask = unique_miss_mask & (current_gpu_slot < 0)
        ws.load_miss_mask_buf.copy_(load_miss_mask)
        return load_miss_mask

    def _assign_slot_indices(
        self,
        ws: DSAGraphWorkspace,
        load_miss_mask: torch.Tensor,
        protected_slot_mask: torch.Tensor,
    ) -> None:
        """Assign sparse-cache slot indices to pages that need loading.

        Eviction priority: free slots (owner < 0) first (biased to sort before
        occupied slots via -(1<<60)), then occupied slots by ascending last_access
        (LRU).  Protected slots (currently selected by the batch) are excluded by
        setting their priority to INT64_MAX.
        """
        slot_priority = torch.where(
            self._page_owner_req < 0,
            self._slot_indices - (1 << 60),
            self._page_last_access,
        )
        slot_priority = torch.where(
            protected_slot_mask,
            torch.full_like(slot_priority, torch.iinfo(slot_priority.dtype).max),
            slot_priority,
        )
        candidate_slots = torch.argsort(slot_priority, stable=True)
        load_ranks = torch.cumsum(load_miss_mask.to(torch.int64), dim=0) - 1
        load_ranks = torch.where(load_miss_mask, load_ranks, 0)

        if self._graph_decode_buffers is None:
            num_loads = int(load_miss_mask.sum().item())
            num_protected_slots = int(protected_slot_mask.sum().item())
            available_candidate_slots = (
                self.sparse_cache_allocator.capacity - num_protected_slots
            )
            if num_loads > available_candidate_slots:
                raise RuntimeError(
                    "DSA eager decode is not supported when the sparse-cache miss set "
                    "exceeds the unprotected eviction candidates. "
                    f"current_loads={num_loads}, protected_current_hits={num_protected_slots}, "
                    f"sparse_cache_capacity={self.sparse_cache_allocator.capacity}. "
                    "Enable cuda graph for DSA decode."
                )

        gathered_candidate_slots = candidate_slots[load_ranks]
        ws.assigned_slot_indices_buf.copy_(
            torch.where(
                load_miss_mask,
                gathered_candidate_slots,
                torch.zeros_like(gathered_candidate_slots),
            )
        )

    def _apply_load_plan_and_transfer(
        self,
        ws: DSAGraphWorkspace,
        load_miss_mask: torch.Tensor,
        sorted_hosts: torch.Tensor,
    ) -> None:
        """Execute the load plan: update page table ownership, then DMA pages.

        dsa_apply_load_plan runs a single-thread kernel that atomically updates
        gpu_slot, page_owner, and page_last_access.  dsa_transfer_mla_pages then
        copies data from host to the assigned device page starts.
        """
        ws.transfer_active_pages_buf.copy_(load_miss_mask.to(torch.int32))
        from sglang.jit_kernel.dsa_apply_load_plan import dsa_apply_load_plan

        dsa_apply_load_plan(
            active_pages=ws.transfer_active_pages_buf,
            req_indices=ws.sorted_miss_req_indices_buf,
            logical_pages=ws.sorted_miss_logical_pages_buf,
            assigned_slot_indices=ws.assigned_slot_indices_buf,
            assigned_page_starts=ws.assigned_page_starts_buf,
            reserved_page_starts=self._reserved_page_starts,
            page_owner_req=self._page_owner_req,
            page_owner_logical=self._page_owner_logical,
            page_last_access=self._page_last_access,
            gpu_slot=self.page_table.gpu_slot,
            cache_step=self._cache_step,
        )
        ws.transfer_host_page_starts_buf.copy_(torch.where(load_miss_mask, sorted_hosts, 0))
        ws.transfer_device_page_starts_buf.copy_(
            torch.where(load_miss_mask, ws.assigned_page_starts_buf, 0)
        )
        from sglang.jit_kernel.dsa_transfer_mla import dsa_transfer_mla_pages

        dsa_transfer_mla_pages(
            src_ptrs=self.host_kv_pool.data_ptrs,
            dst_ptrs=self.kv_pool.data_ptrs,
            src_page_starts=ws.transfer_host_page_starts_buf,
            dst_page_starts=ws.transfer_device_page_starts_buf,
            active_pages=ws.transfer_active_pages_buf,
            page_size=self.page_size,
            token_stride_bytes=self.host_kv_pool.token_stride_size,
        )

    def _expand_page_starts_to_token_indices(self, page_starts: torch.Tensor) -> torch.Tensor:
        if page_starts.numel() == 0:
            return torch.empty(0, dtype=torch.int64, device=page_starts.device)
        return (
            page_starts.to(torch.int64).unsqueeze(1)
            + self._page_offsets.unsqueeze(0)
        ).reshape(-1)

    def _ensure_miss_position_cache(self, max_misses: int) -> None:
        if max_misses <= self._miss_position_cache.numel():
            return
        self._miss_position_cache = torch.arange(
            max_misses, dtype=torch.int64, device=self.device
        )

    def _build_decode_precomputed(
        self, forward_batch: "ForwardBatch"
    ) -> Optional[DSAPrecomputedMetadata]:
        indexer_out_cache_loc = self.indexer_slots.prepare_decode_slots(forward_batch)
        if indexer_out_cache_loc is None:
            return None

        max_len = self._get_decode_max_seq_len(forward_batch)
        dense_page_table_1 = self.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices.to(torch.long), :max_len
        ]
        indexer_page_table_1, indexer_real_page_table = self.indexer_slots.get_page_tables(
            dense_page_table_1,
            forward_batch,
        )

        graph_buffers = self._graph_decode_buffers
        if (
            graph_buffers is None
            or forward_batch.batch_size > graph_buffers.max_bs
            or indexer_page_table_1 is None
            or indexer_real_page_table is None
        ):
            return DSAPrecomputedMetadata(
                indexer_out_cache_loc=indexer_out_cache_loc,
                indexer_page_table_1=indexer_page_table_1,
                indexer_real_page_table=indexer_real_page_table,
                return_logical_topk=True,
            )

        bs = forward_batch.batch_size
        graph_buffers.indexer_out_cache_loc[:bs].copy_(indexer_out_cache_loc)
        graph_buffers.indexer_page_table_1[:bs].zero_()
        graph_buffers.indexer_real_page_table[:bs].zero_()

        page_cols = indexer_page_table_1.shape[1]
        real_cols = indexer_real_page_table.shape[1]
        graph_buffers.indexer_page_table_1[:bs, :page_cols].copy_(indexer_page_table_1)
        graph_buffers.indexer_real_page_table[:bs, :real_cols].copy_(
            indexer_real_page_table
        )
        return DSAPrecomputedMetadata(
            indexer_out_cache_loc=graph_buffers.indexer_out_cache_loc[:bs],
            indexer_page_table_1=graph_buffers.indexer_page_table_1[:bs],
            indexer_real_page_table=graph_buffers.indexer_real_page_table[:bs],
            return_logical_topk=True,
        )

    def _get_decode_max_seq_len(self, forward_batch: "ForwardBatch") -> int:
        if forward_batch.seq_lens_cpu is not None and forward_batch.seq_lens_cpu.numel() > 0:
            return int(forward_batch.seq_lens_cpu.max().item())
        if forward_batch.seq_lens.numel() == 0:
            return 0
        return int(forward_batch.seq_lens.max().item())

    @staticmethod
    def _unwrap_tree_cache(tree_cache):
        current = tree_cache
        while hasattr(current, "inner"):
            current = current.inner
        return current

    def _slot_indices_from_page_starts(self, page_starts: torch.Tensor) -> torch.Tensor:
        page_numbers = torch.div(
            page_starts.to(torch.int64), self.page_size, rounding_mode="floor"
        )
        return self._dense_page_to_slot[page_numbers].to(torch.int64)

    def _touch_sparse_token_indices(self, token_indices: torch.Tensor) -> None:
        if token_indices.numel() == 0:
            return
        from sglang.jit_kernel.dsa_touch_slots import dsa_touch_slots

        dsa_touch_slots(
            token_indices=token_indices,
            dense_page_to_slot=self._dense_page_to_slot,
            page_last_access=self._page_last_access,
            page_size=self.page_size,
            cache_step=self._cache_step,
        )

    def _release_sparse_pages(self, page_starts: torch.Tensor) -> None:
        slot_indices = self._slot_indices_from_page_starts(page_starts)
        slot_indices = slot_indices[slot_indices >= 0]
        if slot_indices.numel() == 0:
            return
        self._page_owner_req[slot_indices] = -1
        self._page_owner_logical[slot_indices] = -1
        self._page_last_access[slot_indices] = -1

    def _get_current_hit_slot_mask(
        self,
        selected_indices_i32: torch.Tensor,
        req_pool_indices_i32: torch.Tensor,
        seq_lens_i32: torch.Tensor,
    ) -> torch.Tensor:
        """Build a mask of sparse-cache slots currently referenced by the batch.

        These "protected" slots must not be evicted in the same step — evicting
        a page that the batch's topk still points to would corrupt the decode.
        """
        self._protected_slot_mask.zero_()
        self._protected_slot_counts.zero_()
        if selected_indices_i32.numel() == 0:
            return self._protected_slot_mask

        selected_indices_i64 = selected_indices_i32.to(torch.int64)
        valid_selected_mask = (
            (selected_indices_i64 >= 0)
            & (selected_indices_i64 < self.max_context_len)
            & (selected_indices_i64 < seq_lens_i32.to(torch.int64).unsqueeze(1))
        )
        logical_pages = torch.div(
            torch.where(valid_selected_mask, selected_indices_i64, 0),
            self.page_size,
            rounding_mode="floor",
        )
        expanded_req_pool_indices = req_pool_indices_i32.to(torch.long).unsqueeze(1).expand_as(
            logical_pages
        )
        current_page_starts = self.page_table.gpu_slot[
            expanded_req_pool_indices, logical_pages
        ]
        current_page_starts = torch.where(
            valid_selected_mask & (current_page_starts >= 0),
            current_page_starts,
            0,
        )

        slot_indices = self._slot_indices_from_page_starts(current_page_starts.reshape(-1))
        valid_slot_indices = (
            valid_selected_mask.reshape(-1)
            & (current_page_starts.reshape(-1) >= 0)
            & (slot_indices >= 0)
        )
        safe_slot_indices = torch.where(valid_slot_indices, slot_indices, 0)
        self._protected_slot_counts.index_add_(
            0,
            safe_slot_indices,
            valid_slot_indices.to(torch.int32),
        )
        self._protected_slot_mask.copy_(self._protected_slot_counts > 0)
        return self._protected_slot_mask
