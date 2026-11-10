"""Bit-exact bf16 -> fp8_e4m3fn cast for the TRTLLM MLA prefill Q/K/V triple.

``trtllm_mla_backend._quantize_fp8_qkv`` casts Q, K and V with three
``Tensor.to(torch.float8_e4m3fn)`` calls before
``trtllm_ragged_attention_deepseek``. Two things make that expensive on
Kimi-K3 prefill (measured on B200, T=32768, H=12, 192/256 head dims):

* ``v`` is ``kv[..., qk_nope_head_dim:]`` of a contiguous ``[T, H, 256]``
  tensor, so its head stride is 256 while only the inner 128 elements are
  contiguous. That strided source blocks TensorIterator vectorization and the
  cast degenerates to ``gpu_kernel_impl_nocast`` at 0.667 TB/s -- 188.9 us of
  the group's 397.6 us.
* even the contiguous q/k casts only reach 1.91 TB/s at 74% sm__throughput,
  moving 8 bytes per thread over 31 waves per SM, so per-element instruction
  cost sets the pace rather than HBM.

This module splits the work by layout -- a flat 1D kernel for contiguous q/k, a
row-per-program kernel for the strided v -- and reproduces PyTorch's conversion
semantics exactly. That second part is not automatic: ``Tensor.to(float8_e4m3fn)``
is **not** the hardware saturating conversion. For ``|x| > 464`` and for every
inf/nan it emits the e4m3 NaN encoding ``0x7f`` with the sign preserved, while
``cvt.rn.satfinite.e4m3`` (what ``x.to(tl.float8e4nv)`` compiles to) clamps to
448. 30639 of the 65536 bf16 bit patterns differ. The patch is three integer
operations on the raw bf16 bits: 464.0 is ``0x43e8`` in bf16 and every bf16
inf/nan satisfies ``bits & 0x7fff >= 0x7f80 > 0x43e8``, so one unsigned compare
covers overflow, inf and nan together, and the sign byte ``(bits >> 8) & 0x80``
is already positioned as the fp8 sign bit.

Measured on B200 for the same shape: q/k 84.7/83.6 us at 2.35/2.38 TB/s and v
72.4 us at 1.715 TB/s, i.e. 1.58x on the whole triple, bit-identical output.
"""

import torch
import triton
import triton.language as tl

FLAT_BLOCK = 4096
FLAT_WARPS = 8
ROW_BLOCK = 16
ROW_WARPS = 8

E4M3_NAN_BITS = tl.constexpr(0x7F)
# 464.0 in bf16. Every bf16 inf/nan pattern compares above it, so a single
# unsigned compare against this value covers overflow, inf and nan at once.
BF16_ABS_OVERFLOW_BITS = tl.constexpr(0x43E8)

# Below this element count the operator is host-launch bound and three Triton
# launches lose to plain torch: measured crossover is between 25,171,968
# elements (torch faster) and 50,331,648 elements (Triton faster) on B200.
TRITON_MIN_ELEMENTS = 40 * 1024 * 1024


@triton.jit
def _exact_fp8_bits(x):
    """bf16 value -> uint8 bits identical to ``torch.Tensor.to(float8_e4m3fn)``."""
    packed = x.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    bits = x.to(tl.uint16, bitcast=True)
    sign = ((bits >> 8) & 0x80).to(tl.uint8)
    overflow = (bits & 0x7FFF) > BF16_ABS_OVERFLOW_BITS
    return tl.where(overflow, sign | E4M3_NAN_BITS, packed)


@triton.jit
def cast_bf16_to_fp8_flat_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    """Cast a contiguous tensor as one flat 1D stream (used for q and k)."""
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(src_ptr + offs, mask=mask, other=0.0)
    tl.store(
        dst_ptr + offs,
        _exact_fp8_bits(x).to(tl.float8e4nv, bitcast=True),
        mask=mask,
    )


@triton.jit
def cast_bf16_to_fp8_rows_kernel(
    src_ptr,
    dst_ptr,
    n_rows,
    head_cnt,
    src_stride0,
    src_stride1,
    D: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """Cast a strided ``[T, H, D]`` view row by row (used for the v slice).

    One program owns ``BLOCK_R`` consecutive ``(token, head)`` rows. Each row is
    ``D`` contiguous elements in the source even though the rows themselves are
    ``src_stride1`` apart, so every load is a full coalesced burst instead of the
    2-bytes-per-thread trickle TensorIterator falls back to.
    """
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    row_mask = rows < n_rows
    t = rows // head_cnt
    h = rows % head_cnt

    offs = tl.arange(0, D)
    src = (
        src_ptr
        + t[:, None].to(tl.int64) * src_stride0
        + h[:, None].to(tl.int64) * src_stride1
        + offs[None, :]
    )
    # The destination is a freshly allocated contiguous tensor, so row r starts
    # at r * D.
    dst = dst_ptr + rows[:, None].to(tl.int64) * D + offs[None, :]

    x = tl.load(src, mask=row_mask[:, None], other=0.0)
    tl.store(
        dst,
        _exact_fp8_bits(x).to(tl.float8e4nv, bitcast=True),
        mask=row_mask[:, None],
    )


def _cast_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    out = torch.empty(tensor.shape, dtype=torch.float8_e4m3fn, device=tensor.device)
    n_elements = tensor.numel()
    cast_bf16_to_fp8_flat_kernel[(triton.cdiv(n_elements, FLAT_BLOCK),)](
        tensor,
        out,
        n_elements,
        BLOCK=FLAT_BLOCK,
        num_warps=FLAT_WARPS,
    )
    return out


def _cast_rowwise(tensor: torch.Tensor) -> torch.Tensor:
    out = torch.empty(tensor.shape, dtype=torch.float8_e4m3fn, device=tensor.device)
    n_rows = tensor.shape[0] * tensor.shape[1]
    cast_bf16_to_fp8_rows_kernel[(triton.cdiv(n_rows, ROW_BLOCK),)](
        tensor,
        out,
        n_rows,
        tensor.shape[1],
        tensor.stride(0),
        tensor.stride(1),
        D=tensor.shape[2],
        BLOCK_R=ROW_BLOCK,
        num_warps=ROW_WARPS,
    )
    return out


def quantize_mla_qkv_fp8_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> bool:
    """Cheap host-side guard; return False to keep the caller's original path.

    Unsupported (falls back): non-CUDA, non-bf16, not 3-D, a non-contiguous
    ``q``/``k``, a non-unit stride on the last dimension of ``v``, an inner dim
    that is not a power-of-two multiple of 8 elements (``tl.arange`` bound), or
    an empty tensor. The ``k_scale``/``v_scale`` != 1.0 ``scaled_fp8_quant``
    branches are handled by the caller, not here.
    """
    tensors = (q, k, v)
    return (
        all(t.is_cuda and t.dtype == torch.bfloat16 and t.dim() == 3 for t in tensors)
        and q.is_contiguous()
        and k.is_contiguous()
        and v.stride(2) == 1
        and v.shape[2] % 8 == 0
        and (v.shape[2] & (v.shape[2] - 1)) == 0
        and all(t.numel() > 0 for t in tensors)
    )


def quantize_mla_qkv_fp8_profitable(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> bool:
    """Measured size gate: below it the three launches lose to plain torch."""
    return q.numel() + k.numel() + v.numel() >= TRITON_MIN_ELEMENTS


def quantize_mla_qkv_fp8_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    force_kernel: bool = False,
):
    """Cast ``q``/``k``/``v`` from bf16 to fp8_e4m3fn, bit-exactly.

    Caller must have checked :func:`quantize_mla_qkv_fp8_supported`. Small
    shapes are routed to the torch cast by :func:`quantize_mla_qkv_fp8_profitable`
    unless ``force_kernel`` is set, which tests and profiling use to reach the
    Triton path on any size.
    """
    if not force_kernel and not quantize_mla_qkv_fp8_profitable(q, k, v):
        return (
            q.to(torch.float8_e4m3fn),
            k.to(torch.float8_e4m3fn),
            v.to(torch.float8_e4m3fn),
        )
    return _cast_contiguous(q), _cast_contiguous(k), _cast_rowwise(v)
