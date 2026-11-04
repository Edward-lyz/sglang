"""Optimized _candidate_mask_kernel: same exact fp32 output, better launch config.

OUT[r,c] = X[r,c] if (c < seqlen[r] and KEEP[r,c]) else -inf,  for all c < WIDTH.

Memory-bound. Traffic per launch (full visibility): OUT write (rows*width*4) +
KEEP read (rows*seqlen*1) + X read (sparse, where kept). We keep traffic minimal
(X stays masked/sparse; keep/x skipped for fully-out-of-range tiles) and raise
achieved bandwidth via a contiguous-keep specialization + tuned TILE/num_warps.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _candidate_mask_opt_kernel(
    X, LENS, KEEP, OUT,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    KEEP_STRIDE: tl.constexpr,
    KEEP_COL_STRIDE: tl.constexpr,
    TILE: tl.constexpr,
    CONTIG_KEEP: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col0 = tl.program_id(1) * TILE
    cols = col0 + tl.arange(0, TILE)
    seqlen = tl.load(LENS + row)
    in_width = cols < WIDTH
    visible = in_width & (cols < seqlen)
    if CONTIG_KEEP:
        keep = tl.load(KEEP + row * KEEP_STRIDE + cols, visible, other=0)
    else:
        keep = tl.load(KEEP + row * KEEP_STRIDE + cols * KEEP_COL_STRIDE, visible, other=0)
    values = tl.load(X + row * STRIDE + cols, visible & keep, other=-float("inf"))
    tl.store(OUT + row * WIDTH + cols, values, in_width)


def candidate_mask_opt(X, seqlens, keep, out=None, TILE=2048, num_warps=8):
    """
    X: [rows, width]
    seqlens: [rows]
    keep: [rows, width]
    out: [rows, width]
    """
    rows, width = X.shape
    if out is None:
        out = torch.empty((rows, width), dtype=torch.float32, device=X.device)
    contig = (keep.stride(1) == 1)
    grid = (rows, triton.cdiv(width, TILE))
    _candidate_mask_opt_kernel[grid](
        X, seqlens, keep, out,
        width, X.stride(0), keep.stride(0), keep.stride(1),
        TILE, contig, num_warps=num_warps,
    )
    return out


@triton.jit
def _candidate_mask_opt2_kernel(
    X, LENS, KEEP, OUT,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    KEEP_STRIDE: tl.constexpr,
    KEEP_COL_STRIDE: tl.constexpr,
    TILE: tl.constexpr,
    CONTIG_KEEP: tl.constexpr,
):
    # Store-bound: page-table capacity WIDTH >> live seqlen, so most tiles are
    # pure -inf tail. Uniform-branch those tiles into a bare vectorized store and
    # skip the keep/x masked loads entirely (removes issue + latency overhead).
    row = tl.program_id(0).to(tl.int64)
    col0 = tl.program_id(1) * TILE
    cols = col0 + tl.arange(0, TILE)
    in_width = cols < WIDTH
    seqlen = tl.load(LENS + row)
    if col0 >= seqlen:
        tl.store(OUT + row * WIDTH + cols, tl.full((TILE,), -float("inf"), tl.float32), in_width)
    else:
        visible = in_width & (cols < seqlen)
        if CONTIG_KEEP:
            keep = tl.load(KEEP + row * KEEP_STRIDE + cols, visible, other=0)
        else:
            keep = tl.load(KEEP + row * KEEP_STRIDE + cols * KEEP_COL_STRIDE, visible, other=0)
        values = tl.load(X + row * STRIDE + cols, visible & keep, other=-float("inf"))
        tl.store(OUT + row * WIDTH + cols, values, in_width)


def candidate_mask_opt2(X, seqlens, keep, out=None, TILE=1024, num_warps=4):
    """
    X: [rows, width]
    seqlens: [rows]
    keep: [rows, width]
    out: [rows, width]
    """
    rows, width = X.shape
    if out is None:
        out = torch.empty((rows, width), dtype=torch.float32, device=X.device)
    contig = (keep.stride(1) == 1)
    grid = (rows, triton.cdiv(width, TILE))
    _candidate_mask_opt2_kernel[grid](
        X, seqlens, keep, out,
        width, X.stride(0), keep.stride(0), keep.stride(1),
        TILE, contig, num_warps=num_warps,
    )
    return out
