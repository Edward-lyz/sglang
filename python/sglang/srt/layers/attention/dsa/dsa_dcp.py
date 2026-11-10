"""Decode context parallelism (DCP) helper for the DSA trtllm sparse MLA path.

Layout (ported from upstream sglang #43735): the target MLA KV is striped by
widened slot (rank ``slot % dcp_size`` stores row ``slot // dcp_size``), while
the index-K cache is replicated, so every rank computes the global top-k. Each
rank keeps the top-k slots it owns, attends with the DCP group's gathered query
heads and returns the LSE; the MLA DCP merge combines the partials.
"""

from __future__ import annotations

import torch

from sglang.kernels.ops.attention.fixup_zero_kv import fixup_zero_kv_rows

TRTLLM_MAX_BATCH_ROWS = 65535
# FlashInfer carves heads * rows * 256 float2 softmax slots out of the workspace
# per call; the rest is kernel scratch.
_SOFTMAX_BYTES_PER_ROW_HEAD = 256 * 8
_SCRATCH_RESERVE_BYTES = 128 << 20


def trtllm_rows_per_call(*, workspace_bytes: int, heads: int) -> int:
    """trtllm_rows_per_call"""
    rows = (workspace_bytes - _SCRATCH_RESERVE_BYTES) // (
        heads * _SOFTMAX_BYTES_PER_ROW_HEAD
    )
    return max(1, min(TRTLLM_MAX_BATCH_ROWS, rows))


def dcp_trtllm_sparse_attention(
    *,
    query: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    **attention_kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse trtllm MLA on this rank's KV shard.

    query: [rows, 1, heads, head_dim]; block_tables: [rows, 1, topk] local rows
    packed to the front (-1 tail); seq_lens: per-row local top-k count;
    cu_seqlens_q: arange(rows + 1). Returns (out [rows, heads, kv_lora_rank],
    lse [rows, heads]); rows with no owned slot get out=0, lse=-inf.
    """
    import flashinfer.decode

    rows, _, heads, _ = query.shape
    if rows == 0:
        return (
            torch.empty(
                (0, heads, attention_kwargs["kv_lora_rank"]),
                dtype=torch.bfloat16,
                device=query.device,
            ),
            torch.empty((0, heads), dtype=torch.float32, device=query.device),
        )
    lse = torch.empty((rows, heads), dtype=torch.float32, device=query.device)
    workspace = attention_kwargs["workspace_buffer"]
    step = trtllm_rows_per_call(
        workspace_bytes=workspace.numel() * workspace.element_size(), heads=heads
    )
    if rows > step:
        attention_kwargs["enable_pdl"] = False
    outputs = []
    for start in range(0, rows, step):
        end = min(start + step, rows)
        output, _ = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=query[start:end],
            block_tables=block_tables[start:end],
            seq_lens=seq_lens[start:end],
            return_lse=True,
            lse=lse[start:end],
            **attention_kwargs,
        )
        output = output.view(end - start, heads, -1)
        fixup_zero_kv_rows(
            output,
            lse[start:end],
            seq_lens[start:end],
            cu_seqlens_q[: end - start + 1],
            1,
        )
        outputs.append(output)
    output = outputs[0] if len(outputs) == 1 else torch.cat(outputs)
    return output, lse


def dcp_flashmla_rope8_sparse_attention(
    *,
    attention,
    query: torch.Tensor,
    local_seq_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    kv_lora_rank: int,
    **attention_kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``dcp_trtllm_sparse_attention`` served by FlashMLA's rope-FP8 kernel.

    Same contract: query [rows, 1, heads, head_dim]; block_tables [rows, 1, topk]
    with this rank's owned slots packed to the front and a -1 tail;
    ``local_seq_lens`` the per-row owned count; returns
    (out [rows, heads, kv_lora_rank], lse [rows, heads]) with out=0 / lse=-inf
    for rows that own nothing.

    Two differences from the trtllm version: the query heads are not zero-padded
    (FlashMLA takes the real count), and there is no row chunking because
    FlashMLA carries no softmax workspace to run out of.
    """
    rows, _, heads, _ = query.shape
    if rows == 0:
        return (
            torch.empty(
                (0, heads, kv_lora_rank), dtype=torch.bfloat16, device=query.device
            ),
            torch.empty((0, heads), dtype=torch.float32, device=query.device),
        )

    out, lse = attention(query=query, return_lse=True, **attention_kwargs)
    out = out.view(rows, heads, -1)
    # FlashMLA reports lse=+inf (and out=0) for a row whose every slot is -1,
    # while the DCP merge expects -inf. fixup_zero_kv_rows writes exactly that
    # for the rows local_seq_lens marks as empty.
    fixup_zero_kv_rows(out, lse, local_seq_lens, cu_seqlens_q[: rows + 1], 1)
    return out, lse
