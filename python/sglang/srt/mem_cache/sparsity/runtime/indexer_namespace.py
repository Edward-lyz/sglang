"""Indexer slot namespace for DSA decode.

IndexerSlotNamespace maintains a stable indexer-only slot namespace
independent from the main KV pool, so that NSA indexer writes land in
a dedicated device buffer while the underlying KV pages may be offloaded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class IndexerSlotNamespace:
    """Stable indexer-only KV slot namespace for DSA decode.

    The NSA indexer needs to write new KV entries each decode step, but the
    underlying prompt pages may have been offloaded to host.  This namespace
    owns a separate paged allocator backed by the same NSATokenToKVPool, so
    indexer writes land in device-resident slots that survive page eviction.

    req_to_indexer mirrors req_to_token but tracks indexer slot ids instead of
    main-pool token indices — this is how the attention backend resolves
    indexer-aware page tables without touching the (partially offloaded) main
    pool mapping.
    """

    def __init__(
        self,
        indexer_pool: "NSATokenToKVPool",
        page_size: int,
        device: torch.device,
        req_to_token_shape: tuple[int, int],
    ):
        self.page_size = page_size
        self.device = device
        self.slot_allocator = PagedTokenToKVPoolAllocator(
            size=indexer_pool.size,
            page_size=page_size,
            dtype=torch.int32,
            device=device,
            kvcache=indexer_pool,
            need_sort=False,
        )
        # This tensor mirrors req_to_token but stores indexer slot ids instead.
        self.req_to_indexer = torch.zeros(
            req_to_token_shape,
            dtype=torch.int32,
            device=device,
        )
        self._received_prompt_tokens_cpu = torch.zeros(
            req_to_token_shape[0],
            dtype=torch.int64,
        )
        self._zero_i64_cpu = torch.zeros((1,), dtype=torch.int64)
        self._zero_i64 = torch.zeros((1,), dtype=torch.int64, device=device)
        self._minus_one_i64 = torch.full((1,), -1, dtype=torch.int64, device=device)
        self._tmp_token_count_i64 = torch.zeros((1,), dtype=torch.int64, device=device)
        self._tmp_token_count_i64_cpu = torch.zeros((1,), dtype=torch.int64)

    def register_request(self, req_pool_idx: int) -> None:
        """Reset all decode-side indexer slots for a new request."""
        self.req_to_indexer[req_pool_idx].zero_()
        self._received_prompt_tokens_cpu[req_pool_idx] = 0

    def clear_request(self, req_pool_idx: int) -> None:
        """Free all indexer slots currently owned by the request."""
        used = self.req_to_indexer[req_pool_idx]
        used_slots = used[used > 0].to(torch.int64)
        if used_slots.numel() > 0:
            self.slot_allocator.free(used_slots)
        self.req_to_indexer[req_pool_idx].zero_()
        self._received_prompt_tokens_cpu[req_pool_idx] = 0

    def available_size(self) -> int:
        return self.slot_allocator.available_size()

    def prepare_received_slots(
        self, req_pool_idx: int, token_count: int
    ) -> torch.Tensor:
        """Allocate stable indexer slots for a received prompt."""
        if token_count <= 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)

        if int(self._received_prompt_tokens_cpu[req_pool_idx].item()) >= token_count:
            return self.req_to_indexer[req_pool_idx, :token_count].to(torch.int64)

        self._tmp_token_count_i64.fill_(token_count)
        self._tmp_token_count_i64_cpu.fill_(token_count)
        indexer_loc = self.slot_allocator.alloc_extend(
            prefix_lens=self._zero_i64,
            prefix_lens_cpu=self._zero_i64_cpu,
            seq_lens=self._tmp_token_count_i64,
            seq_lens_cpu=self._tmp_token_count_i64_cpu,
            last_loc=self._minus_one_i64,
            extend_num_tokens=token_count,
        )
        assert indexer_loc is not None, (
            f"Indexer pool is full while allocating received prompt slots: {token_count=}"
        )
        self.req_to_indexer[req_pool_idx, :token_count] = indexer_loc.to(torch.int32)
        self._received_prompt_tokens_cpu[req_pool_idx] = token_count
        return indexer_loc.to(torch.int64)

    def prepare_decode_slots(self, forward_batch: "ForwardBatch") -> Optional[torch.Tensor]:
        """Attach decode-step indexer slots onto the batch.

        Vectorized: batch-gather existing slots, then only call alloc_decode
        for the subset that needs new allocations.
        """
        if not forward_batch.forward_mode.is_decode():
            return None
        out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
        if out_cache_loc is None or out_cache_loc.numel() == 0:
            return None

        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens
        max_tokens = self.req_to_indexer.shape[1]
        logical_pos = seq_lens.to(torch.long) - 1
        assert bool(((logical_pos >= 0) & (logical_pos < max_tokens)).all().item()), (
            f"Decode indexer logical positions out of range: max_tokens={max_tokens}, "
            f"logical_pos_min={int(logical_pos.min().item())}, logical_pos_max={int(logical_pos.max().item())}"
        )

        # Batch-gather existing indexer slots
        existing = self.req_to_indexer[
            req_pool_indices.to(torch.long), logical_pos
        ]  # [bs], int32
        need_alloc = existing <= 0  # [bs], bool

        indexer_out_cache_loc = existing.clone()

        alloc_indices = torch.where(need_alloc)[0]
        if alloc_indices.numel() == 0:
            return indexer_out_cache_loc.to(torch.int64)

        alloc_req_pool = req_pool_indices[alloc_indices]
        alloc_seq_lens = seq_lens[alloc_indices]
        alloc_logical_pos = logical_pos[alloc_indices]

        # Build last_loc: gather from (logical_pos - 1) where logical_pos > 0
        prev_pos = (alloc_logical_pos - 1).clamp(min=0)
        prev_slots = self.req_to_indexer[
            alloc_req_pool.to(torch.long), prev_pos
        ].to(torch.int64)
        last_loc = torch.where(
            alloc_logical_pos > 0,
            prev_slots,
            torch.full_like(prev_slots, -1),
        )

        alloc_seq_lens_i64 = alloc_seq_lens.to(torch.int64)
        alloc = self.slot_allocator.alloc_decode(
            seq_lens=alloc_seq_lens_i64,
            seq_lens_cpu=None,
            last_loc=last_loc,
        )
        assert alloc is not None, (
            f"Indexer pool is full while allocating decode slots: n_alloc={int(alloc_indices.numel())}"
        )
        self.req_to_indexer[
            alloc_req_pool.to(torch.long), alloc_logical_pos
        ] = alloc.to(torch.int32)
        indexer_out_cache_loc[alloc_indices] = alloc.to(torch.int32)

        return indexer_out_cache_loc.to(torch.int64)

    def get_page_tables(
        self,
        page_table_1: torch.Tensor,
        forward_batch: "ForwardBatch",
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Build indexer page tables from the stable indexer namespace."""
        if not forward_batch.forward_mode.is_decode():
            return None, None

        bs = forward_batch.req_pool_indices.shape[0]
        max_seq = page_table_1.shape[1]
        assert max_seq <= self.req_to_indexer.shape[1], (
            f"Indexer page table width exceeds namespace capacity: {max_seq=} "
            f"req_to_indexer_width={self.req_to_indexer.shape[1]}"
        )

        # Lazy init: page_table_1 shape is unknown at __init__ time (depends on
        # max_context_len from forward metadata). Create/resize on first call.
        if not hasattr(self, "_col_idx_cache") or self._col_idx_cache.shape[0] < max_seq:
            self._col_idx_cache = torch.arange(max_seq, device=self.device)
            self._real_page_col_idx = torch.arange(
                0, max_seq, self.page_size, device=self.device, dtype=torch.int32
            )
        col_idx = self._col_idx_cache[:max_seq].unsqueeze(0).expand(bs, -1)

        seq_lens_2d = forward_batch.seq_lens.unsqueeze(1)
        mask = col_idx < seq_lens_2d

        req_pool_indices = forward_batch.req_pool_indices.to(torch.long)
        gathered = self.req_to_indexer[req_pool_indices][:, :max_seq]

        # torch.where creates a new tensor — no clone of page_table_1 needed.
        indexer_page_table_1 = torch.where(mask & (gathered > 0), gathered, page_table_1)

        num_real_cols = (max_seq + self.page_size - 1) // self.page_size
        real_page_col_idx = self._real_page_col_idx[:num_real_cols]
        indexer_real_page_table = (
            indexer_page_table_1[:, real_page_col_idx] // self.page_size
        )
        assert indexer_real_page_table.stride(1) == 1, (
            "indexer_real_page_table must be contiguous on the last dimension"
        )
        return indexer_page_table_1, indexer_real_page_table
