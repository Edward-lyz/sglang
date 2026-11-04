"""Triton gather of DeepSeek-V4.1 engram rows: fp8 e4m3 payload, e8m0 block scales.

The table pointers arrive as raw addresses so one kernel serves a device table, a
pinned host table, or (on Grace-Blackwell, through ATS) a plain host mapping. The
output is bf16 computed as fp32(row) * 2**(exp - 127) then rounded once, which is
the arithmetic of the torch lookup it replaces.
"""

from typing import Callable, Optional, Tuple

import torch
import triton
import triton.language as tl

# e8m0 has no zero: the exponent byte 0 encodes 2**-127.
_E8M0_ZERO = 2.0**-127


@triton.jit(do_not_specialize=["row_lo", "row_hi"])
def _engram_gather_kernel(
    w_ptr,
    s_ptr,
    ids_ptr,
    out_ptr,
    row_lo,
    row_hi,
    DIM: tl.constexpr,
    BLK: tl.constexpr,
    E8M0_ZERO: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    idx = tl.load(ids_ptr + row).to(tl.int64)
    # The table holds rows [row_lo, row_hi); an id outside it is not read and
    # comes out as zeros, which is what the sharded all-reduce sums.
    owned = (idx >= row_lo) & (idx < row_hi)
    local = tl.where(owned, idx - row_lo, 0)
    w = w_ptr.to(tl.int64).to(tl.pointer_type(tl.float8e4nv))
    s = s_ptr.to(tl.int64).to(tl.pointer_type(tl.uint8))
    offs = tl.arange(0, DIM)
    vals = tl.load(w + local * DIM + offs, mask=owned, other=0.0).to(tl.float32)
    exps = tl.load(s + local * (DIM // BLK) + offs // BLK, mask=owned, other=0).to(
        tl.int32
    )
    # 2**(e - 127) from the exponent bits: exact, no exp2 rounding or denormal flush.
    scale = (exps << 23).to(tl.float32, bitcast=True)
    scale = tl.where(exps == 0, E8M0_ZERO, scale)
    out = tl.where(owned, vals * scale, 0.0)
    tl.store(out_ptr + row * DIM + offs, out.to(tl.bfloat16))


def engram_gather(
    weight_ptr: int,
    scale_ptr: int,
    ids: torch.Tensor,
    out: torch.Tensor,
    dim: int,
    block_size: int,
    row_lo: int = 0,
    row_hi: int = 2**62,
) -> torch.Tensor:
    """Gather rows ``ids`` ([N] int) into ``out`` ([N, dim] bf16, contiguous).

    ``weight_ptr`` addresses [rows, dim] fp8 e4m3 bytes and ``scale_ptr``
    [rows, dim // block_size] e8m0 bytes for global rows [row_lo, row_hi); both
    may live in device or host memory. Ids outside the range produce zero rows.
    """
    assert dim & (dim - 1) == 0 and dim % block_size == 0, (dim, block_size)
    assert out.dtype == torch.bfloat16 and out.is_contiguous()
    n = ids.numel()
    if n:
        _engram_gather_kernel[(n,)](
            weight_ptr,
            scale_ptr,
            ids,
            out,
            row_lo,
            row_hi,
            DIM=dim,
            BLK=block_size,
            E8M0_ZERO=_E8M0_ZERO,
        )
    return out


@triton.jit(do_not_specialize=["num_pairs", "layer", "row_lo", "row_hi"])
def _engram_gather_packed_kernel(
    w_ptr,
    s_ptr,
    ids_ptr,
    out_ptr,
    num_pairs,
    layer,
    row_lo,
    row_hi,
    LAYERS: tl.constexpr,
    COLS: tl.constexpr,
    DIM: tl.constexpr,
    BLK: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    # A pair is one (row, local column) of the [rows, LAYERS, COLS] id layout.
    pair = tl.program_id(0).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R)
    valid = pair < num_pairs
    flat = (pair // COLS * LAYERS + layer) * COLS + pair % COLS
    idx = tl.load(ids_ptr + flat, mask=valid, other=-1).to(tl.int64)
    owned = valid & (idx >= row_lo) & (idx < row_hi)
    local = tl.where(owned, idx - row_lo, 0)
    cols = tl.arange(0, DIM)
    scale_cols = tl.arange(0, DIM // BLK)
    vals = tl.load(
        w_ptr + local[:, None] * DIM + cols[None, :], mask=owned[:, None], other=0
    )
    exps = tl.load(
        s_ptr + local[:, None] * (DIM // BLK) + scale_cols[None, :],
        mask=owned[:, None],
        other=0,
    )
    dst = out_ptr + flat[:, None] * (DIM + DIM // BLK)
    tl.store(dst + cols[None, :], vals, mask=valid[:, None])
    tl.store(dst + DIM + scale_cols[None, :], exps, mask=valid[:, None])


def engram_gather_packed(
    weight: torch.Tensor,
    scale: torch.Tensor,
    ids: torch.Tensor,
    out: torch.Tensor,
    layer: int,
    row_lo: int,
) -> torch.Tensor:
    """Copy table rows as raw bytes: DIM fp8 values, then DIM // 32 e8m0 scales.

    ``ids`` is [..., LAYERS, COLS] (contiguous) and ``out`` [..., LAYERS, COLS,
    DIM + DIM // 32] uint8; only ``layer`` is written. ``weight``/``scale`` hold
    global rows [row_lo, row_lo + weight.shape[0]); other ids (the -1 padding)
    become zero bytes.
    """
    dim = weight.shape[1]
    block = dim // scale.shape[1]
    layers, cols = ids.shape[-2:]
    assert ids.is_contiguous() and out.is_contiguous() and out.dtype == torch.uint8
    assert out.shape == (*ids.shape, dim + dim // block), (out.shape, ids.shape)
    num_pairs = ids.numel() // layers
    if num_pairs:
        block_r = 16
        _engram_gather_packed_kernel[(triton.cdiv(num_pairs, block_r),)](
            weight.view(torch.uint8),
            scale.view(torch.uint8),
            ids,
            out,
            num_pairs,
            layer,
            row_lo,
            row_lo + weight.shape[0],
            LAYERS=layers,
            COLS=cols,
            DIM=dim,
            BLK=block,
            BLOCK_R=block_r,
        )
    return out


@triton.jit(do_not_specialize=["num_pairs", "slot", "n_real", "n_rows"])
def _engram_unpack_kernel(
    recv_ptr,
    fill_ptr,
    val_ptr,
    scale_ptr,
    num_pairs,
    slot,
    n_real,
    n_rows,
    val_stride,
    scale_stride,
    RANKS: tl.constexpr,
    COLS: tl.constexpr,
    LAYERS: tl.constexpr,
    DIM: tl.constexpr,
    BLK: tl.constexpr,
    HAS_FILL: tl.constexpr,
    OUT_BF16: tl.constexpr,
    E8M0_ZERO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """recv [RANKS, slot, LAYERS, COLS, WIDTH] -> per layer [n_rows, HEADS * DIM].

    Rows below n_real come from recv; rows in [n_real, n_rows) repeat
    fill[layer] in every column (HAS_FILL) or are zero. MXFP8 scales land in
    the 128x4 swizzled layout with the 128-row tile pad zeroed."""
    HEADS: tl.constexpr = RANKS * COLS
    WIDTH: tl.constexpr = DIM + DIM // BLK
    GROUPS: tl.constexpr = HEADS * (DIM // BLK)
    pair = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    layer = tl.program_id(1).to(tl.int64)
    valid = pair < num_pairs
    token, head = pair // HEADS, pair % HEADS
    real = valid & (token < n_real)
    rows_ok = valid & (token < n_rows)
    src = (
        recv_ptr
        + (((head // COLS * slot + token) * LAYERS + layer) * COLS + head % COLS)
        * WIDTH
    )
    if HAS_FILL:
        fill = rows_ok & (token >= n_real)
        src = tl.where(fill, fill_ptr + layer * WIDTH, src)
        real = real | fill
    cols = tl.arange(0, DIM)
    data = tl.load(src[:, None] + cols[None, :], mask=real[:, None], other=0)
    dst = val_ptr + layer * val_stride + token * (HEADS * DIM) + head * DIM
    if OUT_BF16:
        exps = tl.load(
            src[:, None] + DIM + cols[None, :] // BLK, mask=real[:, None], other=0
        ).to(tl.int32)
        # The arithmetic of _engram_gather_kernel.
        scale = (exps << 23).to(tl.float32, bitcast=True)
        scale = tl.where(exps == 0, E8M0_ZERO, scale)
        vals = data.to(tl.float8e4nv, bitcast=True).to(tl.float32) * scale
        vals = tl.where(real[:, None], vals, 0.0)
        tl.store(
            dst[:, None] + cols[None, :], vals.to(tl.bfloat16), mask=rows_ok[:, None]
        )
    else:
        tl.store(dst[:, None] + cols[None, :], data, mask=rows_ok[:, None])
        scale_cols = tl.arange(0, DIM // BLK)
        exps = tl.load(
            src[:, None] + DIM + scale_cols[None, :], mask=real[:, None], other=0
        )
        # F8_128x4: [row / 128, group / 4, row % 32, row % 128 / 32, group % 4].
        group = head[:, None] * (DIM // BLK) + scale_cols[None, :]
        row = token[:, None]
        offset = ((row // 128 * (GROUPS // 4) + group // 4) * 32 + row % 32) * 16
        offset += row % 128 // 32 * 4 + group % 4
        tl.store(scale_ptr + layer * scale_stride + offset, exps, mask=valid[:, None])


def mxfp8_swizzled_scale_bytes(rows: int, width: int) -> int:
    """Bytes of 128x4 swizzled e8m0 scales for [rows, width] MXFP8 values."""
    return triton.cdiv(rows, 128) * 128 * triton.cdiv(width // 32, 4) * 4


def engram_unpack(
    recv: torch.Tensor,
    n_real: int,
    n_rows: int,
    fill: Optional[torch.Tensor] = None,
    out_bf16: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Unpack exchanged rows ([RANKS, slot, LAYERS, COLS, DIM + DIM // 32] uint8,
    column-owner major) into every layer's wkv input.

    Returns (values [LAYERS, n_rows, RANKS * COLS * DIM] fp8 e4m3, scales
    [LAYERS, swizzled bytes] uint8) for a prequantized MXFP8 GEMM, or with
    ``out_bf16`` (bf16 values dequantized like ``engram_gather``, None).
    ``fill`` ([LAYERS, DIM + DIM // 32] uint8) fills rows [n_real, n_rows) in
    every column; without it those rows are zero.
    """
    ranks, slot, layers, cols, width = recv.shape
    assert recv.is_contiguous() and recv.dtype == torch.uint8
    assert n_rows == 0 or 0 <= n_real <= min(n_rows, slot), (n_real, n_rows, slot)
    dim = width * 32 // 33
    assert dim + dim // 32 == width, width
    heads = ranks * cols
    device = recv.device
    if out_bf16:
        values = torch.empty(
            layers, n_rows, heads * dim, dtype=torch.bfloat16, device=device
        )
        scales = None
        num_pairs = n_rows * heads
    else:
        values = torch.empty(
            layers, n_rows, heads * dim, dtype=torch.float8_e4m3fn, device=device
        )
        scales = torch.empty(
            layers,
            mxfp8_swizzled_scale_bytes(n_rows, heads * dim),
            dtype=torch.uint8,
            device=device,
        )
        num_pairs = triton.cdiv(n_rows, 128) * 128 * heads
    if num_pairs:
        block = 16
        _engram_unpack_kernel[(triton.cdiv(num_pairs, block), layers)](
            recv,
            fill if fill is not None else recv,
            values if out_bf16 else values.view(torch.uint8),
            scales if scales is not None else recv,
            num_pairs,
            slot,
            n_real,
            n_rows,
            values.stride(0),
            scales.stride(0) if scales is not None else 0,
            RANKS=ranks,
            COLS=cols,
            LAYERS=layers,
            DIM=dim,
            BLK=32,
            HAS_FILL=fill is not None,
            OUT_BF16=out_bf16,
            E8M0_ZERO=_E8M0_ZERO,
            BLOCK=block,
        )
    return values, scales


def engram_decode_send_ids(
    hash_ids: torch.Tensor, ranks: int, slot: int, n_send: int
) -> torch.Tensor:
    """[T, LAYERS, ranks * COLS] ids of this rank's tokens -> [ranks (column
    owner), slot, LAYERS, COLS] int32; rows past n_send are -1."""
    tokens, layers, heads = hash_ids.shape
    cols = heads // ranks
    send = hash_ids.new_full((ranks, slot, layers, cols), -1, dtype=torch.int32)
    if n_send:
        send[:, :n_send] = (
            hash_ids[:n_send].view(n_send, layers, ranks, cols).permute(2, 0, 1, 3)
        )
    return send


def engram_cp_send_ids(
    hash_ids: torch.Tensor, ranks: int, rank: int, slot: int
) -> torch.Tensor:
    """Whole-prompt ids [T, LAYERS, ranks * COLS] -> this rank's columns per
    interleave owner: [ranks (token owner), slot, LAYERS, COLS] int32, where
    token t is row t // ranks of owner t % ranks; tokens past T are -1."""
    tokens, layers, heads = hash_ids.shape
    cols = heads // ranks
    padded = hash_ids.new_full((slot * ranks, layers, heads), -1)
    padded[:tokens] = hash_ids
    return (
        padded.view(slot, ranks, layers, ranks, cols)[:, :, :, rank]
        .transpose(0, 1)
        .to(torch.int32)
        .contiguous()
    )


def engram_packed_exchange(
    tables,
    send_ids: torch.Tensor,
    all_to_all: Callable[[torch.Tensor, torch.Tensor], None],
    n_real: int,
    n_rows: int,
    exchange_ids: bool,
    fill: Optional[torch.Tensor] = None,
    out_bf16: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
    """Look up this rank's columns and move every row to its token's owner.

    ``tables`` is one (weight, scale, row_start) per layer. ``send_ids`` is
    [ranks, slot, LAYERS, COLS]: indexed by column owner when ``exchange_ids``
    (the ids go out first), else already by token owner. ``all_to_all(out, in)``
    is an equal-split all-to-all over flat tensors. Returns engram_unpack's
    output and the bytes this rank sent.
    """
    ranks, slot, layers, cols = send_ids.shape
    ids = send_ids
    sent = 0
    if exchange_ids:
        ids = torch.empty_like(send_ids)
        if slot:
            all_to_all(ids.view(-1), send_ids.view(-1))
        sent += send_ids.numel() * send_ids.element_size()
    weight, scale, _ = tables[0]
    width = weight.shape[1] + scale.shape[1]
    send_rows = torch.empty(
        (ranks, slot, layers, cols, width), dtype=torch.uint8, device=send_ids.device
    )
    for layer, (weight, scale, row_start) in enumerate(tables):
        engram_gather_packed(weight, scale, ids, send_rows, layer, row_start)
    recv_rows = torch.empty_like(send_rows)
    if slot:
        all_to_all(recv_rows.view(-1), send_rows.view(-1))
    sent += send_rows.numel()
    values, scales = engram_unpack(
        recv_rows, n_real, n_rows, fill=fill, out_bf16=out_bf16
    )
    return values, scales, sent


@triton.jit(do_not_specialize=["num_pairs", "layer"])
def _engram_peer_gather_packed_kernel(
    bases_ptr,
    starts_ptr,
    ids_ptr,
    out_ptr,
    num_pairs,
    layer,
    LAYERS: tl.constexpr,
    COLS: tl.constexpr,
    DIM: tl.constexpr,
    BLK: tl.constexpr,
    SHARDS: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """Gather packed engram rows from a table row-sharded across peer GPUs.

    Like _engram_gather_packed_kernel, but the table is split into SHARDS row
    ranges [starts[s], starts[s + 1]) that live at raw addresses bases[s] (this
    GPU's shard or a peer GPU's, mapped over CUDA IPC). Shard s stores
    [rows_s, DIM] fp8 bytes followed by [rows_s, DIM // BLK] e8m0 bytes.
    """
    pair = tl.program_id(0).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R)
    valid = pair < num_pairs
    flat = (pair // COLS * LAYERS + layer) * COLS + pair % COLS
    idx = tl.load(ids_ptr + flat, mask=valid, other=-1).to(tl.int64)
    total = tl.load(starts_ptr + SHARDS)
    owned = valid & (idx >= 0) & (idx < total)
    shard = tl.zeros([BLOCK_R], dtype=tl.int64)
    for s in tl.static_range(1, SHARDS):
        shard += (idx >= tl.load(starts_ptr + s)).to(tl.int64)
    start = tl.load(starts_ptr + shard, mask=owned, other=0)
    end = tl.load(starts_ptr + shard + 1, mask=owned, other=0)
    base = tl.load(bases_ptr + shard, mask=owned, other=0)
    local = tl.where(owned, idx - start, 0)
    w = base.to(tl.pointer_type(tl.uint8))
    s_base = (base + (end - start) * DIM).to(tl.pointer_type(tl.uint8))
    cols = tl.arange(0, DIM)
    scale_cols = tl.arange(0, DIM // BLK)
    vals = tl.load(
        w[:, None] + local[:, None] * DIM + cols[None, :], mask=owned[:, None], other=0
    )
    exps = tl.load(
        s_base[:, None] + local[:, None] * (DIM // BLK) + scale_cols[None, :],
        mask=owned[:, None],
        other=0,
    )
    dst = out_ptr + flat[:, None] * (DIM + DIM // BLK)
    tl.store(dst + cols[None, :], vals, mask=valid[:, None])
    tl.store(dst + DIM + scale_cols[None, :], exps, mask=valid[:, None])


def engram_peer_gather_packed(
    bases: torch.Tensor,
    starts: torch.Tensor,
    ids: torch.Tensor,
    out: torch.Tensor,
    layer: int,
    dim: int,
    block: int = 32,
) -> torch.Tensor:
    """engram_gather_packed over a row-sharded table: ``bases`` [S] int64 device
    addresses of the shards, ``starts`` [S + 1] int64 global row boundaries.
    ``ids`` [..., LAYERS, COLS], ``out`` [..., LAYERS, COLS, DIM + DIM // block]
    uint8; only ``layer`` is written, out-of-range ids become zero bytes."""
    layers, cols = ids.shape[-2:]
    assert ids.is_contiguous() and out.is_contiguous() and out.dtype == torch.uint8
    assert out.shape == (*ids.shape, dim + dim // block), (out.shape, ids.shape)
    assert bases.dtype == torch.int64 and starts.dtype == torch.int64
    num_pairs = ids.numel() // layers
    if num_pairs:
        block_r = 16
        _engram_peer_gather_packed_kernel[(triton.cdiv(num_pairs, block_r),)](
            bases,
            starts,
            ids,
            out,
            num_pairs,
            layer,
            LAYERS=layers,
            COLS=cols,
            DIM=dim,
            BLK=block,
            SHARDS=bases.numel(),
            BLOCK_R=block_r,
        )
    return out
