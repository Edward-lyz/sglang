"""DSA helpers for miss detection, topk remap, and workspace management.

This module contains the Python-side orchestration for DSA cache operations.
Functions that have real logic (type conversion, workspace management) live here.
Pure JIT kernel wrappers live in sglang.jit_kernel.dsa_* and should be imported
directly by callers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.srt.mem_cache.sparsity.core.page_table import PageTable


@dataclass
class DSAGraphWorkspace:
    """Pre-allocated buffers shared across the miss-detect → sort → load pipeline.

    All DSA kernels in a decode step read/write these buffers.  In eager mode
    ensure_capacity() grows them on demand.  Once freeze_capacity() is called
    (before cuda graph capture), the buffer addresses are baked into the graph
    and any attempt to resize becomes a hard error — this is why the frozen
    flag exists.
    """

    device: torch.device
    max_rows: int
    max_topk: int
    max_misses: int
    miss_req_indices_buf: torch.Tensor
    miss_logical_pages_buf: torch.Tensor
    miss_host_slots_buf: torch.Tensor
    miss_count_buf: torch.Tensor
    remapped_buf: torch.Tensor
    miss_valid_mask_buf: torch.Tensor
    sorted_miss_req_indices_buf: torch.Tensor
    sorted_miss_logical_pages_buf: torch.Tensor
    sorted_miss_host_slots_buf: torch.Tensor
    unique_miss_mask_buf: torch.Tensor
    load_miss_mask_buf: torch.Tensor
    assigned_slot_indices_buf: torch.Tensor
    assigned_page_starts_buf: torch.Tensor
    transfer_active_pages_buf: torch.Tensor
    transfer_host_page_starts_buf: torch.Tensor
    transfer_device_page_starts_buf: torch.Tensor
    frozen: bool = False
    _MISS_BUFFER_SPECS = (
        ("miss_req_indices_buf", torch.int32),
        ("miss_logical_pages_buf", torch.int32),
        ("miss_host_slots_buf", torch.int64),
        ("miss_valid_mask_buf", torch.bool),
        ("sorted_miss_req_indices_buf", torch.int32),
        ("sorted_miss_logical_pages_buf", torch.int32),
        ("sorted_miss_host_slots_buf", torch.int64),
        ("unique_miss_mask_buf", torch.bool),
        ("load_miss_mask_buf", torch.bool),
        ("assigned_slot_indices_buf", torch.int64),
        ("assigned_page_starts_buf", torch.int64),
        ("transfer_active_pages_buf", torch.int32),
        ("transfer_host_page_starts_buf", torch.int64),
        ("transfer_device_page_starts_buf", torch.int64),
    )

    @classmethod
    def create(cls, *, device: torch.device) -> "DSAGraphWorkspace":
        return cls(
            device=device,
            max_rows=0,
            max_topk=0,
            max_misses=0,
            miss_req_indices_buf=torch.empty(0, dtype=torch.int32, device=device),
            miss_logical_pages_buf=torch.empty(0, dtype=torch.int32, device=device),
            miss_host_slots_buf=torch.empty(0, dtype=torch.int64, device=device),
            miss_count_buf=torch.zeros(1, dtype=torch.int32, device=device),
            remapped_buf=torch.empty((0, 0), dtype=torch.int32, device=device),
            miss_valid_mask_buf=torch.empty(0, dtype=torch.bool, device=device),
            sorted_miss_req_indices_buf=torch.empty(0, dtype=torch.int32, device=device),
            sorted_miss_logical_pages_buf=torch.empty(0, dtype=torch.int32, device=device),
            sorted_miss_host_slots_buf=torch.empty(0, dtype=torch.int64, device=device),
            unique_miss_mask_buf=torch.empty(0, dtype=torch.bool, device=device),
            load_miss_mask_buf=torch.empty(0, dtype=torch.bool, device=device),
            assigned_slot_indices_buf=torch.empty(0, dtype=torch.int64, device=device),
            assigned_page_starts_buf=torch.empty(0, dtype=torch.int64, device=device),
            transfer_active_pages_buf=torch.empty(0, dtype=torch.int32, device=device),
            transfer_host_page_starts_buf=torch.empty(
                0, dtype=torch.int64, device=device
            ),
            transfer_device_page_starts_buf=torch.empty(
                0, dtype=torch.int64, device=device
            ),
        )

    def ensure_capacity(self, rows: int, topk: int) -> None:
        target_rows = max(rows, self.max_rows)
        target_topk = max(topk, self.max_topk)
        target_misses = target_rows * target_topk
        if (
            target_rows == self.max_rows
            and target_topk == self.max_topk
            and target_misses == self.max_misses
        ):
            return
        if self.frozen:
            raise RuntimeError(
                "DSA graph workspace capacity is frozen, but replay requested "
                f"{rows=} {topk=} beyond allocated "
                f"{self.max_rows=} {self.max_topk=}"
            )

        self.max_rows = target_rows
        self.max_topk = target_topk
        self.max_misses = target_misses
        for name, dtype in self._MISS_BUFFER_SPECS:
            setattr(
                self,
                name,
                torch.empty(self.max_misses, dtype=dtype, device=self.device),
            )
        self.remapped_buf = torch.empty(
            (self.max_rows, self.max_topk), dtype=torch.int32, device=self.device
        )

    def freeze_capacity(self, rows: int, topk: int) -> None:
        self.frozen = False
        self.ensure_capacity(rows, topk)
        self.frozen = True


def _to_int32(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.int32) if x.dtype != torch.int32 else x


def detect_dsa_misses(
    *,
    selected_indices: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: PageTable,
    page_size: int,
    workspace: DSAGraphWorkspace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scan topk selections and record which logical pages are not in the GPU cache.

    The JIT kernel writes miss entries (req_idx, logical_page, host_slot) into the
    workspace buffers and increments miss_count_buf.  This function additionally
    handles int32 conversion — the JIT kernels require int32 inputs to match the
    CUDA thread indexing, while upstream tensors may be int64.
    """
    from sglang.jit_kernel.dsa_detect_misses import dsa_detect_misses

    rows, topk = selected_indices.shape
    if rows == 0 or topk == 0:
        return _to_int32(selected_indices), _to_int32(req_pool_indices), _to_int32(seq_lens)
    workspace.ensure_capacity(rows, topk)

    max_context_len = page_table.max_pages_per_req * page_size
    assert max_context_len % page_size == 0, (
        f"max_context_len ({max_context_len}) must be divisible by page_size ({page_size}) "
        "to match kernel-side integer division"
    )
    seq_lens_i32 = _to_int32(seq_lens)
    req_pool_indices_i32 = _to_int32(req_pool_indices)
    selected_i32 = _to_int32(selected_indices)
    workspace.miss_count_buf.fill_(0)
    dsa_detect_misses(
        topk=selected_i32,
        req_pool_indices=req_pool_indices_i32,
        seq_lens=seq_lens_i32,
        gpu_slot=page_table.gpu_slot,
        host_slot=page_table.host_slot,
        miss_req_indices=workspace.miss_req_indices_buf,
        miss_logical_pages=workspace.miss_logical_pages_buf,
        miss_host_slots=workspace.miss_host_slots_buf,
        miss_count=workspace.miss_count_buf,
        page_size=page_size,
        max_context_len=max_context_len,
        max_misses=workspace.max_misses,
        gpu_slot_stride_0=page_table.gpu_slot.stride(0),
        host_slot_stride_0=page_table.host_slot.stride(0),
    )
    return selected_i32, req_pool_indices_i32, seq_lens_i32


def remap_dsa_topk(
    *,
    selected_indices_i32: torch.Tensor,
    req_pool_indices_i32: torch.Tensor,
    seq_lens_i32: torch.Tensor,
    page_table: PageTable,
    req_to_token: torch.Tensor,
    page_size: int,
    workspace: DSAGraphWorkspace,
) -> torch.Tensor:
    """Translate logical topk token indices to physical KV buffer addresses.

    For tokens in the offloaded prefix, the kernel looks up the sparse cache
    gpu_slot to find the physical page start.  For tokens in the live decode
    tail (not offloaded), it falls through to req_to_token for the original
    dense address.  The output is a [rows, topk] tensor of physical addresses
    ready for the attention backend.
    """
    from sglang.jit_kernel.dsa_remap_topk import dsa_remap_topk

    rows, topk = selected_indices_i32.shape
    workspace.ensure_capacity(rows, topk)
    max_context_len = page_table.max_pages_per_req * page_size
    assert max_context_len % page_size == 0, (
        f"max_context_len ({max_context_len}) must be divisible by page_size ({page_size}) "
        "to match kernel-side integer division"
    )
    req_to_token_i32 = _to_int32(req_to_token)
    remapped = workspace.remapped_buf[:rows, :topk]
    dsa_remap_topk(
        topk=selected_indices_i32,
        req_pool_indices=req_pool_indices_i32,
        seq_lens=seq_lens_i32,
        gpu_slot=page_table.gpu_slot,
        host_slot=page_table.host_slot,
        req_to_token=req_to_token_i32,
        remapped=remapped,
        page_size=page_size,
        max_context_len=max_context_len,
        host_slot_stride_0=page_table.host_slot.stride(0),
    )
    return remapped
