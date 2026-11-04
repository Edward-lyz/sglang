"""Chunk tail SWA KV cache pool for DeepSeekV4 + chunked prefill + AS disaggregation.

For each swa_step boundary crossed in a long request, we back up the last swa_win
tokens' SWA-layer KV data so that cross-chunk SWA attention can refer back to
them even after the original SWA pool slots have been freed/evicted.

Buffer layout mirrors DeepSeekV4SingleKVPool:
  kv_buffer[layer_idx]: Tensor[num_pages, slot_bytes], dtype=uint8
  where slot_bytes = swa_win * bytes_per_token = 128 * 584 = 74752
        (the lower or upper half of one SWA page, without page padding)

Lifecycle:
  alloc(need_pages)                           — called in extend_alloc per chunk
  copy_swa_page_half(slot_id, pool, page, half) — called in detect_and_may_copy_as_swa
  free(slot_ids)                              — called by SDK after async write completes
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_USE_TRITON_COPY = os.environ.get("SGLANG_COPY_SWA_PAGE_HALF_TRITON", "0") == "1"


@triton.jit
def _copy_swa_page_half_kernel(
    dst_ptrs,  # pointer to array of dst base pointers (one per layer)
    src_ptrs,  # pointer to array of src base pointers (one per layer)
    dst_offset,  # byte offset into dst buffer for slot_id
    src_offset,  # byte offset into src buffer for swa_page_idx + half
    slot_elems: tl.constexpr,  # slot_bytes // 4, number of float32 elements
    BLOCK_SIZE: tl.constexpr,  # elements per iteration (in float32 units)
):
    """Each program instance copies one layer. pid = layer_idx.
    Uses float32 (4-byte) vectorized load/store for memory throughput."""
    layer_idx = tl.program_id(0)

    dst_base = tl.load(dst_ptrs + layer_idx).to(tl.pointer_type(tl.float32))
    src_base = tl.load(src_ptrs + layer_idx).to(tl.pointer_type(tl.float32))

    # offset is in bytes, convert to float32 (4-byte) units
    dst_start = dst_base + dst_offset // 4
    src_start = src_base + src_offset // 4

    for off in range(0, slot_elems, BLOCK_SIZE):
        idxs = off + tl.arange(0, BLOCK_SIZE)
        mask = idxs < slot_elems
        data = tl.load(src_start + idxs, mask=mask)
        tl.store(dst_start + idxs, data, mask=mask)


@triton.jit
def _copy_swa_page_half_batch_kernel(
    dst_ptrs,  # [layer_num] dst base pointers
    src_ptrs,  # [layer_num] src base pointers
    dst_offsets,  # [num_copies] byte offset of each dst slot
    src_offsets,  # [num_copies] byte offset of each src half-page
    copy_idx,  # [num_rows, 2] (layer, slot) pairs, flattened
    num_rows,  # rows in copy_idx (= grid size)
    slot_elems: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Each program instance copies one (slot, layer) pair. pid flattens the
    num_copies x layer_num grid via a precomputed (layer, slot) index table,
    so one launch covers a whole request's checkpoints."""
    row = tl.program_id(0)
    if row >= num_rows:
        return
    layer_idx = tl.load(copy_idx + row * 2)
    slot_idx = tl.load(copy_idx + row * 2 + 1)

    dst_base = tl.load(dst_ptrs + layer_idx).to(tl.pointer_type(tl.float32))
    src_base = tl.load(src_ptrs + layer_idx).to(tl.pointer_type(tl.float32))
    dst_start = dst_base + tl.load(dst_offsets + slot_idx) // 4
    src_start = src_base + tl.load(src_offsets + slot_idx) // 4

    for off in range(0, slot_elems, BLOCK_SIZE):
        idxs = off + tl.arange(0, BLOCK_SIZE)
        mask = idxs < slot_elems
        data = tl.load(src_start + idxs, mask=mask)
        tl.store(dst_start + idxs, data, mask=mask)


class ChunkTailSWAPool:
    """Backup storage for chunk-tail SWA KV data in DeepSeekV4 chunked prefill.

    For every swa_step boundary in a request, we copy the last swa_win tokens'
    SWA-layer KV data into an independent fixed-size buffer. The original SWA
    pool slots are not touched (no extra free/alloc there).

    Buffer format:
      kv_buffer  : List[Tensor]  — one Tensor per SWA layer
      Tensor shape: [num_pages, slot_bytes], dtype uint8
      slot_bytes = swa_win * bytes_per_token

    Field naming:
      bytes_per_page_padded — mirrors DeepSeekV4SingleKVPool (source pool's value)
      slot_bytes            — actual bytes stored per slot (= swa_win * bytes_per_token)

    Capacity:
      num_pages — total backup slots; sized per chunk (not per request), see
                  DeepSeekV4TokenToKVPool.init_as_swa_pool():
                    slots_per_chunk = chunked_prefill_size
                                      // (swa_page_size // swa_win) // swa_win
                    num_pages = int(slots_per_chunk * as_swa_buffer_coef)

    Slot management is external: callers hold slot IDs in Req.as_swa_kv_slots.
    No internal rid→slot dicts — this keeps the pool simple and thread-safe
    (SDK calls free() from its own thread; engine only calls alloc()).
    """

    def __init__(
        self,
        num_pages: int,
        layer_num: int,
        bytes_per_token: int,
        src_swa_page_size: int,
        swa_win: int,
        device: str,
    ):
        """
        Args:
            num_pages:            Total number of backup slots.
            layer_num:            Number of SWA layers.
            bytes_per_token:      Bytes per token in SWA pool (= kv_cache_total_dim = 584).
            src_swa_page_size:    SWA pool's page_size in tokens (= 256).
            swa_win:              Number of tokens per slot = model_config.window_size (= 128).
            device:               Torch device string.
        """
        self.num_pages = num_pages
        self.layer_num = layer_num
        self.src_swa_page_size = src_swa_page_size  # = 256
        self.swa_win = swa_win  # = 128
        self.slot_bytes = swa_win * bytes_per_token  # = 128 * 584 = 74752
        self.store_dtype = torch.uint8
        self.device = device

        # kv_buffer[layer_idx]: Tensor[num_pages, slot_bytes], dtype=uint8
        self.kv_buffer: List[torch.Tensor] = [
            torch.zeros(
                num_pages,
                self.slot_bytes,
                dtype=self.store_dtype,
                device=device,
            )
            for _ in range(layer_num)
        ]

        # Flat free-list allocator. Engine only calls alloc(); SDK calls free().
        # No lock needed: alloc() is single-threaded (engine loop), free() is
        # called by SDK after async write — races are acceptable because the
        # freed slots are not reallocated until the engine's next alloc() call.
        self.free_slots: List[int] = list(range(num_pages))

    # ------------------------------------------------------------------
    # Allocator
    # ------------------------------------------------------------------

    def alloc(self, need_pages: int) -> Optional[List[int]]:
        """Allocate need_pages slots in one call.

        Returns a list of slot IDs on success, or None if the pool does not
        have enough free slots (graceful degradation — caller should log/skip).

        Example: 32k input, swa_step=4k → need_pages=8 → returns [s0..s7].
        The caller (extend_alloc) appends the returned list to req.as_swa_kv_slots.
        """
        if len(self.free_slots) < need_pages:
            logger.warning(
                "ChunkTailSWAPool: need %d slots but only %d available (num_pages=%d)",
                need_pages,
                len(self.free_slots),
                self.num_pages,
            )
            return None
        return [self.free_slots.pop() for _ in range(need_pages)]

    def free(self, slot_ids: List[int]) -> None:
        """Return slot_ids to the free pool.

        Called by the SDK after async write completes, or by the engine on abort.
        Accepting a list (not a single slot) matches alloc()'s batch semantics.
        Callers guarantee no -1 sentinels in slot_ids.
        """
        # Leak/double-free guard: detect if free would push free_slots beyond
        # capacity (double-free) — a strong signal of a slot accounting bug.
        if len(self.free_slots) + len(slot_ids) > self.num_pages:
            logger.error(
                "ChunkTailSWAPool double-free suspected: free_slots=%d + "
                "freeing=%d > num_pages=%d. slot_ids=%s",
                len(self.free_slots),
                len(slot_ids),
                self.num_pages,
                slot_ids,
            )
        self.free_slots.extend(slot_ids)

    # ------------------------------------------------------------------
    # Copy
    # ------------------------------------------------------------------

    def _init_copy_swa_page_half_triton_ptrs(self, swa_kv_pool):
        """Lazily initialize pointer tensors for the triton copy kernel (once)."""
        if hasattr(self, "_copy_swa_dst_ptrs"):
            return
        self._copy_swa_dst_ptrs = torch.tensor(
            [self.kv_buffer[i].data_ptr() for i in range(self.layer_num)],
            dtype=torch.int64,
            device=self.device,
        )
        self._copy_swa_src_ptrs = torch.tensor(
            [swa_kv_pool.kv_buffer[i].data_ptr() for i in range(self.layer_num)],
            dtype=torch.int64,
            device=self.device,
        )

    def copy_swa_page_half(
        self,
        slot_id: int,
        swa_kv_pool,
        swa_page_idx: int,
        half: int,
    ) -> None:
        """Copy one half of a SWA page into slot_id.

        If SGLANG_COPY_SWA_PAGE_HALF_TRITON=1, uses triton kernel (61 layers parallel).
        Otherwise falls back to per-layer copy_().
        """
        if _USE_TRITON_COPY:
            self._copy_swa_page_half_triton(slot_id, swa_kv_pool, swa_page_idx, half)
        else:
            byte_start = half * self.slot_bytes
            byte_end = byte_start + self.slot_bytes
            for layer_idx in range(self.layer_num):
                self.kv_buffer[layer_idx][slot_id].copy_(
                    swa_kv_pool.kv_buffer[layer_idx][swa_page_idx, byte_start:byte_end]
                )

    def copy_swa_page_half_batch(
        self, slot_ids, swa_page_idxs, swa_kv_pool, half: int = 1
    ) -> None:
        """Stage many slots with one triton launch (see the kernel docstring).

        Falls back to the per-slot path when triton copy is disabled. The
        caller must have validated every full->swa mapping already.
        """
        if not slot_ids:
            return
        if not _USE_TRITON_COPY:
            for slot_id, swa_page_idx in zip(slot_ids, swa_page_idxs):
                self.copy_swa_page_half(slot_id, swa_kv_pool, swa_page_idx, half)
            return
        n = len(slot_ids)
        self._init_copy_swa_page_half_triton_ptrs(swa_kv_pool)
        if n == 1:
            self._copy_swa_page_half_triton(
                slot_ids[0], swa_kv_pool, swa_page_idxs[0], half
            )
            return
        device = self.device
        # One pinned, async upload carries both offset tables. torch.tensor(
        # ..., device=...) would cudaMemcpyAsync + cudaStreamSynchronize the
        # current stream, which sits behind the scheduler's WAR barrier during
        # process_batch_result: the sync drained the in-flight forward,
        # ~18ms per swa_stage call before this fix.
        host = torch.tensor(
            [s * self.slot_bytes for s in slot_ids]
            + [
                p * swa_kv_pool.bytes_per_page_padded + half * self.slot_bytes
                for p in swa_page_idxs
            ],
            dtype=torch.int64,
            pin_memory=True,
        )
        dev = host.to(device, non_blocking=True)
        dst_offsets = dev[:n]
        src_offsets = dev[n:]
        layer_col = torch.arange(self.layer_num, device=device).repeat(n)
        slot_col = torch.arange(n, device=device).repeat_interleave(self.layer_num)
        # Row r = (layer = r % layer_num, slot = r // layer_num): one dense
        # int32 table so every program id maps to a real (slot, layer) copy.
        copy_idx = (
            torch.stack((layer_col, slot_col), dim=1).reshape(-1).to(torch.int32)
        )
        _copy_swa_page_half_batch_kernel[(n * self.layer_num,)](
            self._copy_swa_dst_ptrs,
            self._copy_swa_src_ptrs,
            dst_offsets,
            src_offsets,
            copy_idx,
            n * self.layer_num,
            slot_elems=self.slot_bytes // 4,
            BLOCK_SIZE=1024,
        )

    def _copy_swa_page_half_triton(self, slot_id, swa_kv_pool, swa_page_idx, half):
        """Triton path: single launch, one block per layer."""
        self._init_copy_swa_page_half_triton_ptrs(swa_kv_pool)
        _copy_swa_page_half_kernel[(self.layer_num,)](
            self._copy_swa_dst_ptrs,
            self._copy_swa_src_ptrs,
            slot_id * self.slot_bytes,
            swa_page_idx * swa_kv_pool.bytes_per_page_padded + half * self.slot_bytes,
            slot_elems=self.slot_bytes // 4,
            BLOCK_SIZE=1024,
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        """Return the full kv_buffer Tensor for layer_id.

        Used by CUDA kernels to obtain data pointers, mirroring the
        DeepSeekV4SingleKVPool.get_key_buffer() convention:

            self.chunk_tail_read_ptrs = [
                kv_cache.chunk_tail_swa_pool.get_key_buffer(i).data_ptr()
                for i in range(len(self._swa_layer_ids))
            ]
        """
        return self.kv_buffer[layer_id]

    def available_size(self) -> int:
        """Return the number of available free slots.

        Returns:
            int: The number of available free slots in the pool.
        """
        return len(self.free_slots)


def free_chunk_tail_swa_slots(kvcache, slot_ids) -> None:
    """Return ``slot_ids`` to the target pool, which owns their allocation.

    Draft staging buffers use the target's slot ids without allocating from
    their own free lists. The AS write covers both workers' buffers before
    releasing these ids; also freeing them on the draft would duplicate entries
    in a free list that was never depleted.

    ``-1`` entries are sentinels ("no SWA data for this checkpoint") and are
    never owned by any pool, so they are dropped here rather than at each caller.
    """
    pool = getattr(kvcache, "chunk_tail_swa_pool", None)
    if pool is None:
        return
    live = [s for s in slot_ids if s != -1]
    if not live:
        return
    pool.free(live)
