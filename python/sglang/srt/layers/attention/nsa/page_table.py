"""Triton-based gather kernels for building page/block tables for NSA attention."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_page_table_kernel(
    req_to_token_ptr,
    req_pool_indices_ptr,
    output_ptr,
    req_to_token_stride_0: tl.constexpr,
    req_to_token_stride_1: tl.constexpr,
    req_pool_indices_stride: tl.constexpr,
    output_stride_0: tl.constexpr,
    output_stride_1: tl.constexpr,
    max_len,
    REPEATS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel: gather token slots for one request into output rows.

    Grid is (num_source_rows, cdiv(max_len, BLOCK_SIZE)). Each program handles
    one request row (program_id(0)) and one column block (program_id(1)):
    it resolves the request's pool index, loads that request's token slots from
    ``req_to_token``, and writes them to ``REPEATS`` consecutive output rows
    (each source row expands to ``REPEATS`` output rows for speculative decoding).

    Args:
        req_to_token_ptr: Base pointer to the [num_reqs, max_tokens] token table.
        req_pool_indices_ptr: Pointer to the per-source-row request pool indices.
        output_ptr: Base pointer to the output token table.
        req_to_token_stride_0/1: Row/column strides of ``req_to_token``.
        req_pool_indices_stride: Stride of the ``req_pool_indices`` array.
        output_stride_0/1: Row/column strides of ``output``.
        max_len: Number of token columns to copy per row (bounds mask).
        REPEATS: Number of output rows produced per source row (compile-time).
        BLOCK_SIZE: Column block size processed per program (compile-time).
    """
    source_row = tl.program_id(0)
    req_idx = tl.load(
        req_pool_indices_ptr + source_row * req_pool_indices_stride
    )

    cols = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < max_len
    values = tl.load(
        req_to_token_ptr
        + req_idx * req_to_token_stride_0
        + cols * req_to_token_stride_1,
        mask=mask,
    )
    for repeat_idx in tl.static_range(REPEATS):
        output_row = source_row * REPEATS + repeat_idx
        tl.store(
            output_ptr + output_row * output_stride_0 + cols * output_stride_1,
            values,
            mask=mask,
        )


@triton.jit
def _gather_block_table_kernel(
    req_to_token_ptr,
    req_pool_indices_ptr,
    output_ptr,
    req_to_token_stride_0: tl.constexpr,
    req_to_token_stride_1: tl.constexpr,
    req_pool_indices_stride: tl.constexpr,
    output_stride_0: tl.constexpr,
    output_stride_1: tl.constexpr,
    num_blocks,
    REPEATS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel: gather page IDs for one request into output rows.

    Grid is (num_source_rows, cdiv(num_blocks, BLOCK_SIZE)). Each program handles
    one request row (program_id(0)) and one block chunk (program_id(1)): it loads
    the token slot at the start of each page (``blocks * PAGE_SIZE``), converts it
    to a page ID via integer division by ``PAGE_SIZE``, and writes the page IDs to
    ``REPEATS`` consecutive output rows.

    Args:
        req_to_token_ptr: Base pointer to the [num_reqs, max_tokens] token table.
        req_pool_indices_ptr: Pointer to the per-source-row request pool indices.
        output_ptr: Base pointer to the output block (page) table.
        req_to_token_stride_0/1: Row/column strides of ``req_to_token``.
        req_pool_indices_stride: Stride of the ``req_pool_indices`` array.
        output_stride_0/1: Row/column strides of ``output``.
        num_blocks: Number of pages/blocks to emit per row (bounds mask).
        REPEATS: Number of output rows produced per source row (compile-time).
        PAGE_SIZE: Number of tokens per page (compile-time).
        BLOCK_SIZE: Number of blocks processed per program (compile-time).
    """
    source_row = tl.program_id(0)
    req_idx = tl.load(
        req_pool_indices_ptr + source_row * req_pool_indices_stride
    )

    blocks = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = blocks < num_blocks
    token_cols = blocks * PAGE_SIZE
    token_slots = tl.load(
        req_to_token_ptr
        + req_idx * req_to_token_stride_0
        + token_cols * req_to_token_stride_1,
        mask=mask,
    )
    page_ids = token_slots // PAGE_SIZE
    for repeat_idx in tl.static_range(REPEATS):
        output_row = source_row * REPEATS + repeat_idx
        tl.store(
            output_ptr + output_row * output_stride_0 + blocks * output_stride_1,
            page_ids,
            mask=mask,
        )


def gather_page_table_into(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    output: torch.Tensor,
    max_len: int,
    repeats: int = 1,
) -> torch.Tensor:
    """Gather request rows directly into an existing token-granularity table.

    For each request in ``req_pool_indices``, copies its first ``max_len`` token
    slots from ``req_to_token`` into ``output``. Each source row is expanded into
    ``repeats`` consecutive output rows (used by speculative decoding). The write
    happens in place; a view sized to the populated region is returned.

    Args:
        req_to_token: [num_reqs, max_tokens] int32 token table.
        req_pool_indices: 1-D int32/int64 request pool indices to gather.
        output: Pre-allocated int32 output table to write into.
        max_len: Number of token columns to copy per row.
        repeats: Output rows produced per source row (default 1).

    Returns:
        View of ``output`` shaped [num_source_rows * repeats, max_len].
    """
    num_source_rows = req_pool_indices.shape[0]
    num_output_rows = num_source_rows * repeats
    assert req_to_token.dtype == output.dtype == torch.int32
    assert req_pool_indices.dtype in (torch.int32, torch.int64)
    assert repeats > 0 and max_len >= 0
    assert output.shape[0] >= num_output_rows and output.shape[1] >= max_len

    if num_output_rows == 0 or max_len == 0:
        return output[:num_output_rows, :max_len]

    block_size = 4096
    grid = (num_source_rows, triton.cdiv(max_len, block_size))
    _gather_page_table_kernel[grid](
        req_to_token,
        req_pool_indices,
        output,
        req_to_token.stride(0),
        req_to_token.stride(1),
        req_pool_indices.stride(0),
        output.stride(0),
        output.stride(1),
        max_len,
        REPEATS=repeats,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return output[:num_output_rows, :max_len]


def gather_block_table_into(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    output: torch.Tensor,
    max_len: int,
    page_size: int,
    repeats: int = 1,
) -> torch.Tensor:
    """Gather page IDs directly into an existing page-granularity table.

    For each request in ``req_pool_indices``, derives the page ID of each page by
    reading the token slot at the page boundary and dividing by ``page_size``,
    writing ``cdiv(max_len, page_size)`` page IDs into ``output``. Each source row
    expands into ``repeats`` consecutive output rows. Writes in place and returns
    a view sized to the populated region.

    Args:
        req_to_token: [num_reqs, max_tokens] int32 token table.
        req_pool_indices: 1-D int32/int64 request pool indices to gather.
        output: Pre-allocated int32 output block table to write into.
        max_len: Max token length used to compute the number of pages.
        page_size: Number of tokens per page (must be > 1).
        repeats: Output rows produced per source row (default 1).

    Returns:
        View of ``output`` shaped [num_source_rows * repeats, num_blocks].
    """
    assert page_size > 1
    num_source_rows = req_pool_indices.shape[0]
    num_output_rows = num_source_rows * repeats
    num_blocks = triton.cdiv(max_len, page_size)
    assert req_to_token.dtype == output.dtype == torch.int32
    assert req_pool_indices.dtype in (torch.int32, torch.int64)
    assert repeats > 0 and max_len >= 0
    assert output.shape[0] >= num_output_rows and output.shape[1] >= num_blocks

    if num_output_rows == 0 or num_blocks == 0:
        return output[:num_output_rows, :num_blocks]

    block_size = 256
    grid = (num_source_rows, triton.cdiv(num_blocks, block_size))
    _gather_block_table_kernel[grid](
        req_to_token,
        req_pool_indices,
        output,
        req_to_token.stride(0),
        req_to_token.stride(1),
        req_pool_indices.stride(0),
        output.stride(0),
        output.stride(1),
        num_blocks,
        REPEATS=repeats,
        PAGE_SIZE=page_size,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return output[:num_output_rows, :num_blocks]
