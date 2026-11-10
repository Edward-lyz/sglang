"""[fusion-opt] Candidate-source block scoring for the dense (CP) prefill indexer.

One pass over a request's dense index logits replaces the torch chain in
``_select_tile`` / ``_candidate_block_topk``:
  masked_fill(col >= compress_len, -inf)  (in place, kept for the later top-k)
  -> pad to a block multiple -> amax over each block -> newest block := +inf
Bit-identical block scores, so the following torch.topk picks the same blocks.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _candidate_block_score_kernel(
    logits_ptr,
    stride_l,
    lens_ptr,
    out_ptr,
    stride_o,
    width,
    num_blocks,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_PROG: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    blk = tl.program_id(1) * BLOCKS_PER_PROG + tl.arange(0, BLOCKS_PER_PROG)
    length = tl.load(lens_ptr + row)
    col = blk[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    in_width = col < width
    ptrs = logits_ptr + row * stride_l + col
    vals = tl.load(ptrs, mask=in_width, other=float("-inf"))
    live = col < length
    vals = tl.where(live, vals, float("-inf"))
    # same side effect as the torch masked_fill_: columns past the row's length
    tl.store(ptrs, tl.full(vals.shape, float("-inf"), tl.float32), mask=in_width & ~live)
    score = tl.max(vals, axis=1)
    newest = tl.where(length > 0, (length - 1) // BLOCK_SIZE, -1)
    score = tl.where(blk == newest, float("inf"), score)
    tl.store(out_ptr + row * stride_o + blk, score, mask=blk < num_blocks)


def candidate_block_scores_(
    logits: torch.Tensor, lengths: torch.Tensor, width: int, block_size: int
) -> torch.Tensor:
    """logits: fp32 [R, >=width] row-strided view (masked in place past lengths);
    lengths: int [R]. Returns fp32 [R, ceil(width / block_size)] block scores."""
    assert logits.dtype == torch.float32 and logits.stride(-1) == 1
    rows = logits.shape[0]
    num_blocks = (width + block_size - 1) // block_size
    out = torch.empty((rows, num_blocks), dtype=torch.float32, device=logits.device)
    if rows == 0 or num_blocks == 0:
        return out
    blocks_per_prog = 128
    grid = (rows, triton.cdiv(num_blocks, blocks_per_prog))
    _candidate_block_score_kernel[grid](
        logits,
        logits.stride(0),
        lengths.contiguous(),
        out,
        out.stride(0),
        width,
        num_blocks,
        BLOCK_SIZE=block_size,
        BLOCKS_PER_PROG=blocks_per_prog,
        num_warps=4,
    )
    return out


def fused_select_candidate_block_ids(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    width: int,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Drop-in for ``masked_fill_ + select_candidate_block_ids`` on the publish path."""
    scores = candidate_block_scores_(logits, lengths, width, block_size)
    top = scores.topk(min(topk_blocks, scores.shape[-1]), dim=-1)
    return top.indices.to(torch.int32).masked_fill_(~(top.values > -torch.inf), -1)


@triton.jit
def _publish_block_membership(
    IDS, KEEP, ID_ROW: tl.constexpr, ID_COL: tl.constexpr,
    COUNT: tl.constexpr, BLOCKS: tl.constexpr, TILE: tl.constexpr,
):
    row = tl.program_id(0)
    k = tl.arange(0, TILE)
    block = tl.load(IDS + row * ID_ROW + k * ID_COL, k < COUNT, -1)
    tl.store(KEEP + row * BLOCKS + block, True, (k < COUNT) & (block >= 0) & (block < BLOCKS))


@triton.jit
def _mask_candidate_scores_inplace(
    SCORES, KEEP, ROW_STRIDE: tl.constexpr, WIDTH: tl.constexpr,
    BLOCKS: tl.constexpr, GROUP: tl.constexpr, TILE: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * TILE + tl.arange(0, TILE)
    live = col < WIDTH
    keep = tl.load(KEEP + row * BLOCKS + col // GROUP, live, False)
    # Kept NaNs and signed zeros are unchanged, just like masked_fill_.
    tl.store(SCORES + row * ROW_STRIDE + col, -float("inf"), live & ~keep)


def mask_candidate_logits_(logits, blocks, block_size):
    """Apply exactly the block-id visibility mask, without a full-width bool mask."""
    assert logits.dtype == torch.float32 and logits.stride(-1) == 1
    assert blocks.ndim == 2 and blocks.shape[0] == logits.shape[0]
    assert block_size > 0
    rows, width = logits.shape
    nblocks = triton.cdiv(width, block_size)
    if not rows or not width:
        return logits
    keep = torch.zeros((rows, nblocks), dtype=torch.bool, device=logits.device)
    if blocks.shape[1]:
        _publish_block_membership[(rows,)](
            blocks, keep, blocks.stride(0), blocks.stride(1),
            blocks.shape[1], nblocks, triton.next_power_of_2(blocks.shape[1]),
        )
    _mask_candidate_scores_inplace[(rows, triton.cdiv(width, 1024))](
        logits, keep, logits.stride(0), width, nblocks, block_size, 1024,
    )
    return logits
