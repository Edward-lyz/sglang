"""[fusion-opt] Prefill-CP SWA KV store, local-first (SGLANG_FUSE_CP_KV_STORE_LOCAL_FIRST).

The stock CP path all-gathers the raw bf16 kv [T, 512] (1024 B/row), reorders it into
global token order, then every rank runs RMSNorm + RoPE + fp8 quant on all T rows into
its SWA cache. All three steps are row-wise, so each rank instead
  1. norm + RoPE + quantizes only its local rows, with the same fused kernel, into a
     staging buffer laid out exactly like the paged cache (528 B/row for V4.1 fp8),
  2. all-gathers the staging pages,
  3. copies every token's 512 B data row + 16 B scale row into its cache slot.
Every stored byte is produced by the same kernel on the same inputs (row, position),
so the cache contents are bitwise identical to the stock path. Paged layouts V4 (576+8 B)
and V4.1 fp8 (512+16 B) are supported.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.elementwise import fused_k_norm_rope_flashmla


@triton.jit
def _scatter_kv_rows_kernel(
    src_ptr,  # int32 view of the gathered staging pages [cp * npages, PB / 4]
    dst_ptr,  # int32 view of the SWA cache [num_pages, PB / 4]
    swa_loc_ptr,  # [T] dst slot per global token (< 0: no write)
    src_row_ptr,  # [T] int64 flat staging row (rank * P + local row) per global token
    rank_len,  # P: rows per rank in the staging layout
    npages_per_rank,
    page_words,  # PB / 4
    PAGE_SIZE: tl.constexpr,
    DATA_WORDS: tl.constexpr,  # data bytes per token / 4 (V4: 144, V4.1: 128)
    SCALE_WORDS: tl.constexpr,  # scale bytes per token / 4 (V4: 2, V4.1: 4)
    DATA_P2: tl.constexpr,
    SCALE_P2: tl.constexpr,
):
    t = tl.program_id(0)
    dst = tl.load(swa_loc_ptr + t).to(tl.int64)
    if dst < 0:
        return
    m = tl.load(src_row_ptr + t).to(tl.int64)
    r = m // rank_len
    j = m % rank_len
    s_page = r * npages_per_rank + j // PAGE_SIZE
    s_slot = j % PAGE_SIZE
    d_page = dst // PAGE_SIZE
    d_slot = dst % PAGE_SIZE
    w = tl.arange(0, DATA_P2)
    wm = w < DATA_WORDS
    v = tl.load(src_ptr + s_page * page_words + s_slot * DATA_WORDS + w, mask=wm)
    tl.store(dst_ptr + d_page * page_words + d_slot * DATA_WORDS + w, v, mask=wm)
    sw = tl.arange(0, SCALE_P2)
    sm = sw < SCALE_WORDS
    scale_base = PAGE_SIZE * DATA_WORDS
    sv = tl.load(src_ptr + s_page * page_words + scale_base + s_slot * SCALE_WORDS + sw, mask=sm)
    tl.store(dst_ptr + d_page * page_words + scale_base + d_slot * SCALE_WORDS + sw, sv, mask=sm)


def cp_store_swa_kv_local_first(
    *,
    kvcache: torch.Tensor,
    page_size: int,
    layout,
    kv_local: torch.Tensor,
    positions_local: torch.Tensor,
    swa_loc: torch.Tensor,
    forward_batch,
    kv_weight: torch.Tensor,
    eps: float,
    freqs_cis: torch.Tensor,
    cp_rank: int,
    cp_size: int,
) -> None:
    from sglang.srt.layers.dp_attention import attn_cp_all_gather_into_tensor

    meta = forward_batch.attn_cp_metadata
    total = int(meta.total_seq_lens)
    logical = meta.per_rank_logical_token or meta.per_rank_actual_token
    rank_len = max(meta.per_rank_actual_token)
    if rank_len == 0 or total == 0:
        return
    local_len = logical[cp_rank]
    page_bytes = kvcache.shape[1]
    npages = (rank_len + page_size - 1) // page_size
    dev = kv_local.device

    # 1. local rows -> staging pages (same kernel as the cache store)
    staging = torch.empty((npages, page_bytes), dtype=torch.uint8, device=dev)
    stage_loc = torch.arange(local_len, dtype=torch.int32, device=dev)
    fused_k_norm_rope_flashmla(
        kv=kv_local[:local_len],
        kv_weight=kv_weight,
        eps=eps,
        freqs_cis=freqs_cis,
        positions=positions_local[:local_len],
        out_loc=stage_loc,
        kvcache=staging,
        page_size=page_size,
        layout=layout,
    )
    # 2. all-gather the staging pages (rank-major)
    gathered = torch.empty((cp_size * npages, page_bytes), dtype=torch.uint8, device=dev)
    attn_cp_all_gather_into_tensor(gathered, staging)

    # 3. global token t -> flat staging row; same map as the interleave gather
    if meta.gather_index is not None:
        src_row = meta.gather_index[:total].to(torch.int64)
    else:
        t = torch.arange(total, device=dev, dtype=torch.int64)
        src_row = (t % cp_size) * rank_len + t // cp_size
    loc = swa_loc[:total]
    _scatter_kv_rows_kernel[(total,)](
        gathered.view(torch.int32),
        kvcache.view(torch.int32),
        loc,
        src_row,
        rank_len,
        npages,
        page_bytes // 4,
        PAGE_SIZE=page_size,
        DATA_WORDS=layout.data_bytes // 4,
        SCALE_WORDS=layout.scale_bytes // 4,
        DATA_P2=triton.next_power_of_2(layout.data_bytes // 4),
        SCALE_P2=max(2, triton.next_power_of_2(layout.scale_bytes // 4)),
        num_warps=1,
    )
