"""Fused MLA decode FP8 prepare: concat absorb-q + BF16 -> FP8 casts in one Triton launch.

Replaces the four-launch chain used by the MLA FP8 decode paths::

    q = concat_mla_absorb_q(q_nope, q_rope).to(float8_e4m3fn)   # 2 launches
    k_nope_fp8 = k_nope.to(float8_e4m3fn)                       # 1 launch
    k_rope_fp8 = k_rope.to(float8_e4m3fn)                       # 1 launch

The chain materialises a BF16 ``[tokens, heads, 576]`` concat buffer and reads it
straight back to cast it, so a Kimi-K3 decode step pays two extra full-size HBM
trips (11.06 MB written + 11.07 MB read at ``cuda-graph-bs=200``) plus three extra
kernel launches. This kernel reads the four BF16 sources once and writes the three
FP8 destinations once.

``q_nope`` is normally the transpose view of the absorb BMM output ``[heads,
tokens, kv_lora]``, so the head stride is ``tokens * kv_lora`` rather than
``kv_lora``; both strides are passed explicitly and only ``stride(-1) == 1`` is
required.

Numerics are **bit-identical** to ``Tensor.to(torch.float8_e4m3fn)``, not merely
close. The hardware conversion (``cvt.rn.satfinite.e4m3``) saturates ``|x| > 448``
to +-448 (encodings 0x7E / 0xFE), while PyTorch emits the NaN encoding 0x7F / 0xFF
for anything that overflows. Since the saturated encodings differ from the NaN
encodings only in bit 0 and already carry the correct sign, a single OR with 0x7F
on the overflow lanes reproduces PyTorch exactly. The overflow test is
``not (|x| <= 464.0)``: 464.0 is the largest BF16 magnitude that still rounds to
448, and the inverted form keeps NaN on the overflow side, because an IEEE
comparison against NaN is false.

Verified bit-exact against the reference chain on B200 (sm_100) for
``tokens x heads`` of ``800x12``, ``256x12`` and ``4x12``, including injected
``|x| > 464``, ``+-inf``, ``nan`` and ``-nan`` inputs.

The launch is split into four program classes, each covering exactly one
(source tensor, width) pair: q_nope and k_nope use ``8`` rows of 512 columns,
q_rope and k_rope use ``64`` rows of 64 columns, so every program moves the same
4096 elements. Giving each class a single tile width - rather than letting one
program handle both a 512-wide and a 64-wide slice - lowered register use and
raised occupancy on sm_100; the conversion also takes an explicit FP32
intermediate, matching what ``cvt.rn.satfinite.e4m3x2.f32`` consumes.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

FP8_DTYPE = torch.float8_e4m3fn

# Largest BF16 magnitude that still rounds to 448 under round-to-nearest-even.
_BF16_ABS_OVERFLOW_VALUE = tl.constexpr(464.0)
_E4M3_NAN_BITS = tl.constexpr(0x7F)

# Tuned on B200 (sm_100) for Kimi-K3 decode shapes; see the benchmark under
# test/registered/jit/benchmark/. Both blocks move 4096 elements per program.
_WIDE_ROW_BLOCK = 8  # 512-column classes (q_nope, k_nope)
_NARROW_ROW_BLOCK = 64  # 64-column classes (q_rope, k_rope)
_NUM_WARPS = 4

_KV_LORA_RANK = 512
_QK_ROPE_HEAD_DIM = 64


@triton.jit
def _to_fp8_e4m3_bits(x):
    """BF16 value -> uint8 bits identical to ``Tensor.to(torch.float8_e4m3fn)``.

    The FP32 step is explicit because the hardware conversion consumes FP32;
    BF16 -> FP32 is exact and 464.0 is representable in both formats, so neither
    the conversion nor the overflow test can change.
    """
    x32 = x.to(tl.float32)
    packed = x32.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    overflow = (~(tl.abs(x32) <= _BF16_ABS_OVERFLOW_VALUE)).to(tl.uint8)
    return packed | (overflow * _E4M3_NAN_BITS)


@triton.jit
def _mla_prepare_fp8_kernel(
    q_nope_ptr,
    q_rope_ptr,
    k_nope_ptr,
    k_rope_ptr,
    q_out_ptr,
    k_nope_out_ptr,
    k_rope_out_ptr,
    tokens,
    q_rows,
    q_nope_blocks,
    q_rope_blocks,
    k_nope_blocks,
    heads,
    q_nope_stride_token,
    q_nope_stride_head,
    q_rope_stride_token,
    q_rope_stride_head,
    k_nope_stride_token,
    k_rope_stride_token,
    DNOPE: tl.constexpr,
    DROPE: tl.constexpr,
    DOUT: tl.constexpr,
    WIDE: tl.constexpr,
    NARROW: tl.constexpr,
):
    """Four program classes in one grid, in pid order: q_nope rows, q_rope rows,
    k_nope rows, k_rope rows. Each class has a single tile width."""
    pid = tl.program_id(0)

    if pid < q_nope_blocks:
        inner_qn = tl.arange(0, DNOPE)
        rows_qn = pid * WIDE + tl.arange(0, WIDE)
        mask_qn = rows_qn < q_rows
        token_qn = rows_qn // heads
        head_qn = rows_qn % heads
        value_qn = tl.load(
            q_nope_ptr
            + token_qn[:, None].to(tl.int64) * q_nope_stride_token
            + head_qn[:, None].to(tl.int64) * q_nope_stride_head
            + inner_qn[None, :],
            mask=mask_qn[:, None],
            other=0.0,
        )
        tl.store(
            q_out_ptr + rows_qn[:, None].to(tl.int64) * DOUT + inner_qn[None, :],
            _to_fp8_e4m3_bits(value_qn).to(tl.float8e4nv, bitcast=True),
            mask=mask_qn[:, None],
        )
    elif pid < q_nope_blocks + q_rope_blocks:
        inner_qr = tl.arange(0, DROPE)
        rows_qr = (pid - q_nope_blocks) * NARROW + tl.arange(0, NARROW)
        mask_qr = rows_qr < q_rows
        token_qr = rows_qr // heads
        head_qr = rows_qr % heads
        value_qr = tl.load(
            q_rope_ptr
            + token_qr[:, None].to(tl.int64) * q_rope_stride_token
            + head_qr[:, None].to(tl.int64) * q_rope_stride_head
            + inner_qr[None, :],
            mask=mask_qr[:, None],
            other=0.0,
        )
        tl.store(
            q_out_ptr + rows_qr[:, None].to(tl.int64) * DOUT + DNOPE + inner_qr[None, :],
            _to_fp8_e4m3_bits(value_qr).to(tl.float8e4nv, bitcast=True),
            mask=mask_qr[:, None],
        )
    elif pid < q_nope_blocks + q_rope_blocks + k_nope_blocks:
        inner_kn = tl.arange(0, DNOPE)
        rows_kn = (pid - q_nope_blocks - q_rope_blocks) * WIDE + tl.arange(0, WIDE)
        mask_kn = rows_kn < tokens
        value_kn = tl.load(
            k_nope_ptr
            + rows_kn[:, None].to(tl.int64) * k_nope_stride_token
            + inner_kn[None, :],
            mask=mask_kn[:, None],
            other=0.0,
        )
        tl.store(
            k_nope_out_ptr + rows_kn[:, None].to(tl.int64) * DNOPE + inner_kn[None, :],
            _to_fp8_e4m3_bits(value_kn).to(tl.float8e4nv, bitcast=True),
            mask=mask_kn[:, None],
        )
    else:
        inner_kr = tl.arange(0, DROPE)
        rows_kr = (
            pid - q_nope_blocks - q_rope_blocks - k_nope_blocks
        ) * NARROW + tl.arange(0, NARROW)
        mask_kr = rows_kr < tokens
        value_kr = tl.load(
            k_rope_ptr
            + rows_kr[:, None].to(tl.int64) * k_rope_stride_token
            + inner_kr[None, :],
            mask=mask_kr[:, None],
            other=0.0,
        )
        tl.store(
            k_rope_out_ptr + rows_kr[:, None].to(tl.int64) * DROPE + inner_kr[None, :],
            _to_fp8_e4m3_bits(value_kr).to(tl.float8e4nv, bitcast=True),
            mask=mask_kr[:, None],
        )


def can_use_fused_mla_prepare_fp8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
) -> bool:
    """Whether the fused kernel covers these inputs; callers must fall back to the
    original chain when this returns ``False``."""
    tensors = (q_nope, q_rope, k_nope, k_rope)
    if any(t.dtype != torch.bfloat16 for t in tensors):
        return False
    if any(not t.is_cuda for t in tensors):
        return False
    if any(t.stride(-1) != 1 for t in tensors):
        return False
    if q_nope.dim() != 3 or q_rope.dim() != 3:
        return False
    if k_nope.dim() != 2 or k_rope.dim() != 2:
        return False
    if q_nope.shape[-1] != _KV_LORA_RANK or k_nope.shape[-1] != _KV_LORA_RANK:
        return False
    if q_rope.shape[-1] != _QK_ROPE_HEAD_DIM or k_rope.shape[-1] != _QK_ROPE_HEAD_DIM:
        return False
    if q_nope.shape[:2] != q_rope.shape[:2]:
        return False
    if k_nope.shape[0] != k_rope.shape[0] or k_nope.shape[0] != q_nope.shape[0]:
        return False
    return True


def fused_mla_prepare_fp8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(concat(q_nope, q_rope), k_nope, k_rope)`` cast to FP8 E4M3 in one launch.

    Shapes: ``q_nope [tokens, heads, 512]``, ``q_rope [tokens, heads, 64]``,
    ``k_nope [tokens, 512]``, ``k_rope [tokens, 64]``. Returns
    ``(q_fp8 [tokens, heads, 576], k_nope_fp8 [tokens, 512], k_rope_fp8 [tokens, 64])``.
    Results are bit-identical to the unfused chain. Guard with
    :func:`can_use_fused_mla_prepare_fp8` first.
    """
    tokens, heads, dnope = q_nope.shape
    drope = q_rope.shape[-1]
    dout = dnope + drope

    q_out = torch.empty((tokens, heads, dout), dtype=FP8_DTYPE, device=q_nope.device)
    k_nope_out = torch.empty((tokens, dnope), dtype=FP8_DTYPE, device=k_nope.device)
    k_rope_out = torch.empty((tokens, drope), dtype=FP8_DTYPE, device=k_rope.device)
    if tokens == 0:
        return q_out, k_nope_out, k_rope_out

    q_rows = tokens * heads
    q_nope_blocks = triton.cdiv(q_rows, _WIDE_ROW_BLOCK)
    q_rope_blocks = triton.cdiv(q_rows, _NARROW_ROW_BLOCK)
    k_nope_blocks = triton.cdiv(tokens, _WIDE_ROW_BLOCK)
    k_rope_blocks = triton.cdiv(tokens, _NARROW_ROW_BLOCK)
    grid = q_nope_blocks + q_rope_blocks + k_nope_blocks + k_rope_blocks
    _mla_prepare_fp8_kernel[(grid,)](
        q_nope,
        q_rope,
        k_nope,
        k_rope,
        q_out,
        k_nope_out,
        k_rope_out,
        tokens,
        q_rows,
        q_nope_blocks,
        q_rope_blocks,
        k_nope_blocks,
        heads,
        q_nope.stride(0),
        q_nope.stride(1),
        q_rope.stride(0),
        q_rope.stride(1),
        k_nope.stride(0),
        k_rope.stride(0),
        DNOPE=dnope,
        DROPE=drope,
        DOUT=dout,
        WIDE=_WIDE_ROW_BLOCK,
        NARROW=_NARROW_ROW_BLOCK,
        num_warps=_NUM_WARPS,
    )
    return q_out, k_nope_out, k_rope_out
