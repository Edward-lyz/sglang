"""Row-loop candidate-mask for the DeepSeek-V4.1 NSA-indexer decode consumer.

Optimizes OP244 ``_candidate_mask_kernel`` (candidate_blocks.py) on the decode
``select_decode`` path. The baseline fills the ENTIRE ``[rows, width]`` fp32 output
(width = allocated compressed page-table capacity), but the downstream vendor paged
top-k reads only each row's live ``[0:seq_len]`` prefix (capacity >> live length), so
the ``[seq_len:width]`` tail writes are dead work. This kernel launches one program per
ROW (grid = (rows,)) and each program loops ONLY over its live tiles, so no dead blocks
are launched and no dead tail is written.

Output semantics on the live region are byte-identical to the baseline:
    OUT[r, c] = X[r, c]  if (c < seq_len[r] and KEEP[r, c]) else -inf,   for c < ceil(seq_len[r]/TILE)*TILE
The region past ``ceil(seq_len[r]/TILE)*TILE`` is intentionally left UNWRITTEN; the
vendor top-k never reads it (verified: 0 selected indices >= seq_len for aligned and
unaligned seq_len), so the final page_indices / raw_indices are bit-exact as a set.

Graph-safe by construction: static grid (rows), no host sync; the runtime ``seq_len``
value only bounds an in-kernel loop. NOTE: not yet confirmed under a live decode
CUDA-graph capture (needs an in-pod boot gate).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _candidate_mask_rowloop_kernel(
    X, LENS, KEEP, OUT,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    KEEP_STRIDE: tl.constexpr,
    KEEP_COL_STRIDE: tl.constexpr,
    TILE: tl.constexpr,
    CONTIG_KEEP: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    seqlen = tl.load(LENS + row)
    ntiles = (seqlen + TILE - 1) // TILE  # only the live tiles; dead tail skipped
    for t in range(ntiles):
        cols = t * TILE + tl.arange(0, TILE)
        in_width = cols < WIDTH
        visible = in_width & (cols < seqlen)
        if CONTIG_KEEP:
            keep = tl.load(KEEP + row * KEEP_STRIDE + cols, visible, other=0)
        else:
            keep = tl.load(KEEP + row * KEEP_STRIDE + cols * KEEP_COL_STRIDE, visible, other=0)
        values = tl.load(X + row * STRIDE + cols, visible & keep, other=-float("inf"))
        tl.store(OUT + row * WIDTH + cols, values, in_width)


def candidate_mask_rowloop(X, seqlens, keep, out=None, TILE=1024, num_warps=4):
    """OUT[r,c] = X[r,c] if (c<seqlens[r] and keep[r,c]) else -inf, live prefix only.

    ``X`` [rows,width] fp32; ``seqlens`` [rows] int32/int64 non-negative; ``keep``
    [rows,>=width] bool (contiguous or column-strided). Returns ``out`` (allocated when
    None) whose live prefix matches the reference; the dead tail is left unwritten.
    """
    rows, width = X.shape
    if out is None:
        out = torch.empty((rows, width), dtype=torch.float32, device=X.device)
    contig = (keep.stride(1) == 1)
    _candidate_mask_rowloop_kernel[(rows,)](
        X, seqlens, keep, out,
        width, X.stride(0), keep.stride(0), keep.stride(1),
        TILE, contig, num_warps=num_warps,
    )
    return out
