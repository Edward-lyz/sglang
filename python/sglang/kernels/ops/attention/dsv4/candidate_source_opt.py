"""Candidate-SOURCE (publish_decode / layer 20) optimization for DeepSeek-V4.1 decode.

Two optimizations for candidate_block_logits(published=None), the encoder->decoder-
boundary block scoring + top-2048-block selection + publish:

1. `_candidate_scores_kernel` -> row-loop `candidate_scores_rowloop`: one program per
   row, writes only the live block-tiles of `output[rows,width]` (dead page-capacity tail
   is never read by the downstream vendor top-k), scores pre-filled -inf so dead blocks
   stay -inf. 3-5x on the scores kernel.
2. `scores.topk(2048, sorted=False)` (torch.topk == the slow at::native::mbtopk cluster,
   ~60% of the source path) -> the vendor `topk_transform_paged_v2` block top-k +
   membership publish. 33-64x on the selection stage. keep-mask is bit-exact vs torch.topk
   when block scores are distinct at the top-2048 threshold (always when live blocks <=
   2048, the traced regime); at extreme context a boundary tie may pick a different
   equally-scored block (inherent top-k tie ambiguity; torch.topk's tie order is arbitrary
   too).

Together ~5x on the candidate-source path. Graph-safe by construction (static grids, no
host sync). Flag-gated (SGLANG_FUSE_CANDIDATE_SOURCE_OPT), default OFF.
"""
import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2, topk_transform_paged_v2


@triton.jit
def _maximum_with_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _scores_rowloop_kernel(
    X, LENS, OUT, SCORES,
    WIDTH: tl.constexpr, STRIDE: tl.constexpr, BLOCKS: tl.constexpr,
    GROUP: tl.constexpr, GROUP_PAD: tl.constexpr, TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    length = tl.load(LENS + row)
    nblocks_live = (length + GROUP - 1) // GROUP
    ntiles = (nblocks_live + TILE - 1) // TILE
    for t in range(ntiles):
        blocks = t * TILE + tl.arange(0, TILE)
        offsets = tl.arange(0, GROUP_PAD)
        cols = blocks[:, None] * GROUP + offsets[None, :]
        in_bounds = (cols < WIDTH) & (offsets[None, :] < GROUP)
        values = tl.load(X + row * STRIDE + cols, in_bounds & (cols < length), other=-float("inf")).to(tl.float32)
        tl.store(OUT + row * WIDTH + cols, values, in_bounds)
        sc = tl.reduce(values, axis=1, combine_fn=_maximum_with_nan)
        sc = tl.where((length > 0) & (blocks == (length - 1) // GROUP), float("inf"), sc)
        tl.store(SCORES + row * BLOCKS + blocks, sc, blocks < BLOCKS)


@triton.jit
def _publish_members_kernel(IDX, KEEP, 
                            WIDTH: tl.constexpr, 
                            GROUP: tl.constexpr, 
                            TOPK: tl.constexpr, 
                            TILE: tl.constexpr):
    """Publish the topk block members (col indices) into the keep buffer."""
    row = tl.program_id(0).to(tl.int64)
    i = tl.program_id(1) * TILE + tl.arange(0, TILE)
    sel = tl.load(IDX + row * TOPK + i // GROUP, i < TOPK * GROUP, -1)
    cols = sel * GROUP + i % GROUP
    tl.store(KEEP + row * WIDTH + cols, sel >= 0, (i < TOPK * GROUP) & (sel >= 0) & (cols < WIDTH))


@triton.jit
def _memzero_u8_kernel(P, N, TILE: tl.constexpr):
    """Full-width bool-zero via wide uint8 stores (~1.9x vs torch.zeros' FillFunctor,
    which runs the bool fill well under HBM bandwidth). Zeroes the WHOLE keep buffer,
    so it stays valid for every consumer read (including the non-Triton fallback that
    reads published[:, :width] unmasked) — same semantics as torch.zeros, just faster."""
    # int64: N = rows * width passes 2^31 for large decode batches x long contexts.
    off = tl.program_id(0).to(tl.int64) * TILE + tl.arange(0, TILE)
    tl.store(P + off, tl.zeros((TILE,), tl.uint8), off < N)


def candidate_scores_rowloop(logits, seq_lens, out, scores, block_size=8, num_warps=4):
    """Row-loop block scoring: writes live-tile masked logits into `out[rows,width]`
    and block-max into `scores[rows,blocks]` (blocks of `block_size` columns, the size
    `scores` was allocated for). `scores` must be pre-filled with -inf so dead blocks
    stay -inf; `out`'s dead tail is left unwritten (never read downstream)."""
    rows, width = logits.shape
    blocks = scores.shape[1]
    group_pad = triton.next_power_of_2(block_size)
    tile = max(1, 1024 // group_pad)
    scores.fill_(-float("inf"))
    _scores_rowloop_kernel[(rows,)](
        logits, seq_lens, out, scores, width, logits.stride(0), blocks, block_size, group_pad, tile,
        num_warps=num_warps,
    )


def block_topk_keep(scores, seq_lens, width, topk_blocks, block_size):
    """Top-`topk_blocks` block selection via the vendor top-k (replaces torch.topk),
    then a membership publish -> keep[rows,width] bool. `scores[rows,blocks]` block-max."""
    rows, blocks = scores.shape
    k = min(topk_blocks, blocks)
    nblocks = ((seq_lens + block_size - 1) // block_size).to(torch.int32)
    meta = plan_topk_v2(nblocks)
    bidx = torch.empty(rows, k, dtype=torch.int32, device=scores.device)
    topk_transform_paged_v2(scores, nblocks, None, bidx, block_size, meta, None)
    keep = torch.empty(rows, width, dtype=torch.bool, device=scores.device)
    n = keep.numel()
    ztile = 8192
    _memzero_u8_kernel[(triton.cdiv(n, ztile),)](keep.view(torch.uint8).view(-1), n, ztile, num_warps=8)
    _publish_members_kernel[(rows, triton.cdiv(k * block_size, 256))](
        bidx, keep, width, block_size, k, 256, num_warps=4,
    )
    return keep


def candidate_source_scores_and_keep(logits, seq_lens, topk_blocks, block_size, out=None):
    """Optimized replacement for candidate_block_logits(published=None): returns
    (output masked logits [rows,width] fp32, keep [rows,width] bool)."""
    rows, width = logits.shape
    blocks = triton.cdiv(width, block_size)
    if out is None:
        out = torch.empty((rows, width), dtype=torch.float32, device=logits.device)
    scores = torch.empty((rows, blocks), dtype=torch.float32, device=logits.device)
    candidate_scores_rowloop(logits, seq_lens, out, scores, block_size)
    keep = block_topk_keep(scores, seq_lens, width, topk_blocks, block_size)
    return out, keep
