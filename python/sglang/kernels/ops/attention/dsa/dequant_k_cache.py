from typing import Optional

import torch
import triton
import triton.language as tl


def dequantize_k_cache(quant_k_cache):
    return _dequantize_k_cache_fast_wrapped(quant_k_cache)


def _dequantize_k_cache_ref(
    quant_k_cache: torch.Tensor,  # (num_blocks, block_size, 1, bytes_per_token)
    dv: int = 512,
    tile_size: int = 128,
    d: int = 576,
) -> torch.Tensor:
    """
    De-quantize the k-cache
    """
    assert dv % tile_size == 0
    original_ndim = quant_k_cache.ndim
    if original_ndim == 3:
        # set block_size = 1
        quant_k_cache = quant_k_cache.unsqueeze(1)
    num_tiles = dv // tile_size
    num_blocks, block_size, h_k, _ = quant_k_cache.shape
    assert h_k == 1
    result = torch.empty(
        (num_blocks, block_size, d), dtype=torch.bfloat16, device=quant_k_cache.device
    )

    quant_k_cache = quant_k_cache.view(num_blocks, block_size, -1)

    input_nope = quant_k_cache[..., :dv]
    input_scale = quant_k_cache[..., dv : dv + num_tiles * 4].view(torch.float32)
    input_rope = quant_k_cache[..., dv + num_tiles * 4 :].view(torch.bfloat16)
    result[..., dv:] = input_rope

    for tile_idx in range(0, num_tiles):
        cur_nope = input_nope[
            ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
        ].to(torch.float32)
        cur_scales = input_scale[..., tile_idx].unsqueeze(-1)
        result[..., tile_idx * tile_size : (tile_idx + 1) * tile_size] = (
            cur_nope * cur_scales
        )

    if original_ndim == 3:
        return result.view(num_blocks, 1, -1)
    else:
        return result.view(num_blocks, block_size, 1, -1)


def _dequantize_k_cache_fast_wrapped(
    quant_k_cache: torch.Tensor,
    dv: int = 512,
    tile_size: int = 128,
) -> torch.Tensor:
    original_ndim = quant_k_cache.ndim
    if original_ndim == 3:
        # set block_size = 1
        quant_k_cache = quant_k_cache.unsqueeze(1)
    num_blocks, block_size, _, dim_quant = quant_k_cache.shape
    assert dv == 512
    assert dim_quant == 656
    assert tile_size == 128
    quant_k_cache = quant_k_cache.view((-1, dim_quant))

    output = _dequantize_k_cache_fast(quant_k_cache)

    if original_ndim == 3:
        return output.view(num_blocks, 1, -1)
    else:
        return output.view(num_blocks, block_size, 1, -1)


def _dequantize_k_cache_fast(quant_k_cache, group_size: int = 128):
    num_tokens, dim_quant = quant_k_cache.shape

    assert quant_k_cache.dtype == torch.float8_e4m3fn
    dim_nope = 512
    dim_rope = 64
    num_tiles = dim_nope // group_size
    assert dim_quant == 656

    output = torch.empty(
        (num_tokens, dim_nope + dim_rope),
        dtype=torch.bfloat16,
        device=quant_k_cache.device,
    )

    num_blocks_per_token = triton.cdiv(dim_nope + dim_rope, group_size)
    assert num_blocks_per_token == 5

    assert dim_nope % group_size == 0

    input_nope_q = quant_k_cache[:, :dim_nope]
    input_nope_s = quant_k_cache[:, dim_nope : dim_nope + num_tiles * 4].view(
        torch.float32
    )
    input_rope = quant_k_cache[:, dim_nope + num_tiles * 4 :].view(torch.bfloat16)

    _dequantize_k_cache_fast_kernel[(num_tokens, num_blocks_per_token)](
        output,
        input_nope_q,
        input_nope_s,
        input_rope,
        output.stride(0),
        input_nope_q.stride(0),
        input_nope_s.stride(0),
        input_rope.stride(0),
        NUM_NOPE_BLOCKS=num_tiles,
        GROUP_SIZE=group_size,
        DIM_NOPE=dim_nope,
        DIM_ROPE=dim_rope,
    )

    return output


@triton.jit
def _dequantize_k_cache_fast_kernel(
    output_ptr,
    input_nope_q_ptr,
    input_nope_s_ptr,
    input_rope_ptr,
    output_stride_0: int,
    input_nope_q_stride_0: int,
    input_nope_s_stride_0: int,
    input_rope_stride_0: int,
    NUM_NOPE_BLOCKS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
):
    token_id = tl.program_id(0)
    raw_block_id = tl.program_id(1)

    if raw_block_id < NUM_NOPE_BLOCKS:
        # a. dequant nope
        effective_block_id = raw_block_id

        offs_q = effective_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs_q < DIM_NOPE
        ptr_q = input_nope_q_ptr + token_id * input_nope_q_stride_0 + offs_q
        ptr_s = input_nope_s_ptr + token_id * input_nope_s_stride_0 + effective_block_id

        y_q = tl.load(ptr_q, mask=mask, other=0.0).to(tl.float32)
        y_s = tl.load(ptr_s)

        y = (y_q * y_s).to(output_ptr.dtype.element_ty)

        dst_ptr = output_ptr + token_id * output_stride_0 + offs_q
        tl.store(dst_ptr, y, mask=mask)
    else:
        # b. copy rope
        effective_block_id = raw_block_id - NUM_NOPE_BLOCKS

        offs = effective_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs < DIM_ROPE

        src_ptr = input_rope_ptr + token_id * input_rope_stride_0 + offs
        dst_ptr = output_ptr + token_id * output_stride_0 + DIM_NOPE + offs

        data = tl.load(src_ptr, mask=mask).to(tl.bfloat16)
        tl.store(dst_ptr, data, mask=mask)


def dequantize_k_cache_paged(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """
    De-quantize the k-cache with paged layout
    Args:
        quant_k_cache: [total_num_tokens, 1, dim_quant] or [num_blocks, block_size, 1, dim_quant], the quantized k-cache in paged layout
        page_table_1_flattened: [num_tokens], the flattened page_table_1 with the page indices in each requests concatenated together
    Returns:
        output: [num_tokens, 1, dim_nope + dim_rope], the de-quantized k-cache
    """
    dim_quant = quant_k_cache.shape[-1]
    assert (
        dim_quant == 656
    ), f"dim_quant: {dim_quant} != 656 detected in dequantize_k_cache_paged"
    quant_k_cache = quant_k_cache.view((-1, dim_quant))

    # num_tokens can exceed kv_cache_size due to prefix sharing (multiple seqs share same KV slots)
    # Index bounds validated in dsa_backend.init_forward_metadata
    num_tokens = page_table_1_flattened.shape[0]
    assert quant_k_cache.dtype == torch.float8_e4m3fn
    dim_nope = 512
    dim_rope = 64
    num_tiles = dim_nope // group_size  # 512 // 128 = 4

    output = torch.empty(
        (num_tokens, 1, dim_nope + dim_rope),
        dtype=torch.bfloat16,
        device=quant_k_cache.device,
    )

    # cdiv(512 + 64, 128) = 5
    num_blocks_per_token = triton.cdiv(dim_nope + dim_rope, group_size)
    assert num_blocks_per_token == 5

    assert dim_nope % group_size == 0

    input_nope_q = quant_k_cache[:, :dim_nope]
    # [:, 512:512+4*4] = [:, 512:528]
    input_nope_s = quant_k_cache[:, dim_nope : dim_nope + num_tiles * 4].view(
        torch.float32
    )
    # [:, 528:]
    input_rope = quant_k_cache[:, dim_nope + num_tiles * 4 :].view(torch.bfloat16)

    # Process multiple tokens per CTA for long cached prefixes. The original
    # kernel launches five tiny CTAs per token, which dominates large tables.
    # Keep the existing small-input path; both produce the same BF16 values.
    # 中文说明：
    # - 原 kernel 每个 token 要起 num_blocks_per_token（5）个很小的 CTA，
    #   命中前缀很长（token 数上万）时，启动的 CTA 数量成了主要开销。
    # - 新 kernel 一个 CTA 处理 BLOCK_TOKENS=8 个 token，CTA 数降为原来的约 1/40。
    # - 只在 num_tokens >= 4096 时启用；小输入沿用原路径，避免影响短请求。
    # - 两条路径只做 FP8*scale -> BF16 与 rope 拷贝，输出数值逐位一致。
    if num_tokens >= 4096:
        _dequantize_k_cache_paged_tiled_kernel[(triton.cdiv(num_tokens, 8),)](
            output, quant_k_cache, input_nope_s, input_rope,
            page_table_1_flattened, num_tokens, quant_k_cache.stride(0),
            input_nope_s.stride(0), input_rope.stride(0),
            BLOCK_TOKENS=8, num_warps=4,
        )
        return output

    _dequantize_k_cache_paged_kernel[(num_tokens, num_blocks_per_token)](
        output,
        input_nope_q,
        input_nope_s,
        input_rope,
        page_table_1_flattened,
        output.stride(0),
        input_nope_q.stride(0),
        input_nope_s.stride(0),
        input_rope.stride(0),
        NUM_NOPE_BLOCKS=num_tiles,
        GROUP_SIZE=group_size,
        DIM_NOPE=dim_nope,
        DIM_ROPE=dim_rope,
    )

    return output


@triton.jit
def _dequantize_k_cache_paged_kernel(
    output_ptr,
    input_nope_q_ptr,
    input_nope_s_ptr,
    input_rope_ptr,
    page_table_1_ptr,
    output_stride_0: int,
    input_nope_q_stride_0: int,
    input_nope_s_stride_0: int,
    input_rope_stride_0: int,
    NUM_NOPE_BLOCKS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
):
    token_id = tl.program_id(0)
    token_id_paged = tl.load(page_table_1_ptr + token_id).to(tl.int32)
    raw_block_id = tl.program_id(1)

    if raw_block_id < NUM_NOPE_BLOCKS:
        # a. dequant nope
        effective_block_id = raw_block_id

        offs_q = effective_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs_q < DIM_NOPE
        ptr_q = input_nope_q_ptr + token_id_paged * input_nope_q_stride_0 + offs_q
        ptr_s = (
            input_nope_s_ptr
            + token_id_paged * input_nope_s_stride_0
            + effective_block_id
        )

        y_q = tl.load(ptr_q, mask=mask, other=0.0).to(tl.float32)
        y_s = tl.load(ptr_s)

        y = (y_q * y_s).to(output_ptr.dtype.element_ty)

        dst_ptr = output_ptr + token_id * output_stride_0 + offs_q
        tl.store(dst_ptr, y, mask=mask)
    else:
        # b. copy rope
        effective_block_id = raw_block_id - NUM_NOPE_BLOCKS

        offs = effective_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs < DIM_ROPE

        src_ptr = input_rope_ptr + token_id_paged * input_rope_stride_0 + offs
        dst_ptr = output_ptr + token_id * output_stride_0 + DIM_NOPE + offs

        data = tl.load(src_ptr, mask=mask).to(tl.bfloat16)
        tl.store(dst_ptr, data, mask=mask)


# Tokens handled by one program of the vectorized gather kernel.  4 tokens
# x 512 fp8 nope elements = 2048 elements per program: with num_warps=4
# (128 threads) that is 16 fp8 elements per thread, which Triton emits as
# a single 16-byte vectorized load/store per thread.
_GATHER_TOKENS_PER_PROG = 4


def gather_dequant_requant_fp8_paged(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    group_size: int = 128,
    extra_rows: int = 0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gather paged fp8 KV tokens and re-pack into flat [576] fp8 layout.

    The paged KV cache stores 656 bytes per token:
        [512 nope_fp8 | 16 scales_f32 (4 groups) | 128 rope_bf16_bytes]
    This kernel gathers the requested tokens, de-quantises nope with the
    per-group scales, and re-quantises to per-tensor fp8 (scale=1.0).
    Rope is cast bf16->fp8.  The whole operation is fused into a single
    Triton kernel to avoid allocating an intermediate bf16 buffer.

    The kernel writes EVERY byte of rows [0, num_tokens) and zero-fills
    rows [num_tokens, num_tokens + extra_rows) (the -1-sentinel landing
    pad required by the SM90 sparse MLA Q8KV8 kernel, which clamps each
    -1 topk slot ``offs`` to distinct row ``num_tokens + offs``).  The
    destination therefore needs no pre-zeroing, which allows passing a
    persistent (dirty) buffer via ``out``.

    Args:
        quant_k_cache: [total_num_tokens, 1, 656] fp8_e4m3fn
        page_table_1_flattened: [num_tokens] int32
        group_size: per-group dequant tile size (default 128)
        extra_rows: number of zero-filled landing-pad rows to append at
            the end of the output (used by the SM90 sparse MLA Q8KV8
            kernel which over-reads past end-of-buffer for masked
            indices)
        out: optional pre-allocated destination of shape
            [num_tokens + extra_rows, 1, 576] (or [.., 576]) fp8_e4m3fn,
            contiguous.  Contents may be arbitrary (fully overwritten).
    Returns:
        output: [num_tokens + extra_rows, 1, 576] fp8_e4m3fn
    """
    dim_quant = quant_k_cache.shape[-1]
    assert dim_quant == 656
    quant_k_cache = quant_k_cache.view((-1, dim_quant))

    num_tokens = page_table_1_flattened.shape[0]
    assert quant_k_cache.dtype == torch.float8_e4m3fn
    dim_nope = 512
    dim_rope = 64
    num_tiles = dim_nope // group_size  # 4
    out_dim = dim_nope + dim_rope  # 576
    assert num_tiles * group_size == dim_nope

    total_rows = num_tokens + extra_rows
    if out is None:
        # No zero-fill needed: the kernel overwrites every byte of the
        # data rows and zero-fills the pad rows itself.
        output = torch.empty(
            (total_rows, 1, out_dim),
            dtype=torch.float8_e4m3fn,
            device=quant_k_cache.device,
        )
    else:
        assert out.dtype == torch.float8_e4m3fn
        assert out.device == quant_k_cache.device
        assert out.is_contiguous()
        assert out.numel() == total_rows * out_dim, (
            f"out buffer has {out.numel()} elements, expected "
            f"{total_rows} x {out_dim} = {total_rows * out_dim}"
        )
        output = out.view(total_rows, 1, out_dim)

    if total_rows == 0:
        return output

    input_nope_q = quant_k_cache[:, :dim_nope]
    input_nope_s = quant_k_cache[:, dim_nope : dim_nope + num_tiles * 4].view(
        torch.float32
    )
    input_rope = quant_k_cache[:, dim_nope + num_tiles * 4 :].view(torch.bfloat16)

    grid = (triton.cdiv(total_rows, _GATHER_TOKENS_PER_PROG),)
    _gather_dequant_requant_fp8_paged_vec_kernel[grid](
        output,
        input_nope_q,
        input_nope_s,
        input_rope,
        page_table_1_flattened,
        num_tokens,
        total_rows,
        output.stride(0),
        input_nope_q.stride(0),
        input_nope_s.stride(0),
        input_rope.stride(0),
        NUM_NOPE_BLOCKS=num_tiles,
        GROUP_SIZE=group_size,
        DIM_NOPE=dim_nope,
        DIM_ROPE=dim_rope,
        TOKENS_PER_PROG=_GATHER_TOKENS_PER_PROG,
        num_warps=4,
    )

    return output


@triton.jit
def _gather_dequant_requant_fp8_paged_vec_kernel(
    output_ptr,
    input_nope_q_ptr,
    input_nope_s_ptr,
    input_rope_ptr,
    page_table_1_ptr,
    num_tokens: int,
    total_rows: int,
    output_stride_0: int,
    input_nope_q_stride_0: int,
    input_nope_s_stride_0: int,
    input_rope_stride_0: int,
    NUM_NOPE_BLOCKS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    TOKENS_PER_PROG: tl.constexpr,
):
    """Vectorized fused gather + dequant(per-group) + requant(per-tensor).

    One program handles TOKENS_PER_PROG consecutive output rows (full
    576-byte rows each), instead of the legacy one-program-per-(token,
    128-elem-slice) layout, so each thread moves 16 contiguous fp8 bytes
    per load/store.  Rows >= num_tokens (the -1-sentinel landing pad) are
    zero-filled without touching the KV cache.  Per-element math is
    bit-identical to the legacy kernel: fp8 -> f32, * f32 group scale,
    -> fp8 (nope); bf16 -> fp8 (rope).
    """
    pid = tl.program_id(0)
    offs_t = pid * TOKENS_PER_PROG + tl.arange(0, TOKENS_PER_PROG)  # [T]
    row_in_range = offs_t < total_rows
    is_real = offs_t < num_tokens
    # Masked lanes (pad rows) never touch memory; `other=0` keeps the
    # address arithmetic in-bounds-irrelevant.
    paged = tl.load(page_table_1_ptr + offs_t, mask=is_real, other=0).to(tl.int64)
    # 64-bit output row offsets: total_rows * 576 can exceed int32 for
    # very large gathered buffers.
    offs_t64 = offs_t.to(tl.int64)

    offs_g = tl.arange(0, NUM_NOPE_BLOCKS)  # [G] dequant groups
    offs_i = tl.arange(0, GROUP_SIZE)  # [I] elems within a group

    # a. nope: [T, G, I] fp8 block; the (G, I) plane spans the contiguous
    #    DIM_NOPE bytes of one cache row.
    ptr_q = (
        input_nope_q_ptr
        + paged[:, None, None] * input_nope_q_stride_0
        + offs_g[None, :, None] * GROUP_SIZE
        + offs_i[None, None, :]
    )
    y_q = tl.load(ptr_q, mask=is_real[:, None, None], other=0.0).to(tl.float32)
    ptr_s = input_nope_s_ptr + paged[:, None] * input_nope_s_stride_0 + offs_g[None, :]
    y_s = tl.load(ptr_s, mask=is_real[:, None], other=0.0)
    # dequant -> f32 -> requant to fp8; pad rows: (0 * 0) -> +0 -> byte 0x00
    y = (y_q * y_s[:, :, None]).to(tl.float8e4nv)
    dst_q = (
        output_ptr
        + offs_t64[:, None, None] * output_stride_0
        + offs_g[None, :, None] * GROUP_SIZE
        + offs_i[None, None, :]
    )
    tl.store(dst_q, y, mask=row_in_range[:, None, None])

    # b. rope: [T, R] bf16 -> fp8; pad rows: 0.0 -> byte 0x00
    offs_r = tl.arange(0, DIM_ROPE)
    src_r = input_rope_ptr + paged[:, None] * input_rope_stride_0 + offs_r[None, :]
    data = tl.load(src_r, mask=is_real[:, None], other=0.0).to(tl.float8e4nv)
    dst_r = (
        output_ptr + offs_t64[:, None] * output_stride_0 + DIM_NOPE + offs_r[None, :]
    )
    tl.store(dst_r, data, mask=row_in_range[:, None])


def gather_dequant_requant_fp8_paged_legacy(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    group_size: int = 128,
    extra_rows: int = 0,
) -> torch.Tensor:
    """Legacy (pre-vectorization) gather + dequant + requant.

    Kept as the bit-exactness / performance reference for
    ``gather_dequant_requant_fp8_paged`` (see
    ``benchmark/kernels/deepseek/benchmark_q8kv8_kv_gather.py``).  Allocates and zero-fills the
    full destination each call, then launches one program per
    (token, 128-elem slice).
    """
    dim_quant = quant_k_cache.shape[-1]
    assert dim_quant == 656
    quant_k_cache = quant_k_cache.view((-1, dim_quant))

    num_tokens = page_table_1_flattened.shape[0]
    assert quant_k_cache.dtype == torch.float8_e4m3fn
    dim_nope = 512
    dim_rope = 64
    num_tiles = dim_nope // group_size  # 4
    out_dim = dim_nope + dim_rope  # 576
    assert num_tiles * group_size == dim_nope

    total_rows = num_tokens + extra_rows
    # Allocate a fresh zero-filled buffer.  The extra landing-pad rows at
    # the tail must read as zeros (the kernel may over-read past
    # num_tokens for masked indices).
    output = torch.zeros(
        (total_rows, 1, out_dim),
        dtype=torch.float8_e4m3fn,
        device=quant_k_cache.device,
    )

    num_blocks_per_token = triton.cdiv(dim_nope + dim_rope, group_size)  # 5
    assert num_blocks_per_token == 5

    input_nope_q = quant_k_cache[:, :dim_nope]
    input_nope_s = quant_k_cache[:, dim_nope : dim_nope + num_tiles * 4].view(
        torch.float32
    )
    input_rope = quant_k_cache[:, dim_nope + num_tiles * 4 :].view(torch.bfloat16)

    _gather_dequant_requant_fp8_paged_kernel[(num_tokens, num_blocks_per_token)](
        output,
        input_nope_q,
        input_nope_s,
        input_rope,
        page_table_1_flattened,
        output.stride(0),
        input_nope_q.stride(0),
        input_nope_s.stride(0),
        input_rope.stride(0),
        NUM_NOPE_BLOCKS=num_tiles,
        GROUP_SIZE=group_size,
        DIM_NOPE=dim_nope,
        DIM_ROPE=dim_rope,
    )

    return output


@triton.jit
def _gather_dequant_requant_fp8_paged_kernel(
    output_ptr,
    input_nope_q_ptr,
    input_nope_s_ptr,
    input_rope_ptr,
    page_table_1_ptr,
    output_stride_0: int,
    input_nope_q_stride_0: int,
    input_nope_s_stride_0: int,
    input_rope_stride_0: int,
    NUM_NOPE_BLOCKS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
):
    """Fused gather + dequant(per-group) + requant(per-tensor) -> fp8."""
    token_id = tl.program_id(0)
    token_id_paged = tl.load(page_table_1_ptr + token_id).to(tl.int32)
    raw_block_id = tl.program_id(1)

    if raw_block_id < NUM_NOPE_BLOCKS:
        # nope: read fp8, mul group scale -> f32, cast to fp8_e4m3fn
        effective_block_id = raw_block_id
        offs_q = effective_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs_q < DIM_NOPE

        ptr_q = input_nope_q_ptr + token_id_paged * input_nope_q_stride_0 + offs_q
        ptr_s = (
            input_nope_s_ptr
            + token_id_paged * input_nope_s_stride_0
            + effective_block_id
        )

        y_q = tl.load(ptr_q, mask=mask, other=0.0).to(tl.float32)
        y_s = tl.load(ptr_s)

        # dequant -> f32 -> requant to fp8
        y = (y_q * y_s).to(tl.float8e4nv)

        dst_ptr = output_ptr + token_id * output_stride_0 + offs_q
        tl.store(dst_ptr, y, mask=mask)
    else:
        # rope: read bf16, cast to fp8
        effective_block_id = raw_block_id - NUM_NOPE_BLOCKS
        offs = effective_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs < DIM_ROPE

        src_ptr = input_rope_ptr + token_id_paged * input_rope_stride_0 + offs
        dst_ptr = output_ptr + token_id * output_stride_0 + DIM_NOPE + offs

        data = tl.load(src_ptr, mask=mask).to(tl.float8e4nv)
        tl.store(dst_ptr, data, mask=mask)


@triton.jit
def _dequantize_k_cache_paged_tiled_kernel(
    out, q, scales, rope, table,
    N, QS: tl.constexpr, SS: tl.constexpr, RS: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    # 分块去量化 kernel：一个 CTA 处理 BLOCK_TOKENS 个 token 的整行 KV（576 维）。
    # 输入每个 token 的布局：[0:512) 为 FP8 的 nope，[512:528) 为 4 个 fp32 scale
    # （每 128 维共用一个），其后为 64 维 bf16 的 rope；输出为 [N, 576] 的 bf16。
    # 当前 token 行号；超出 N 的尾部行用 mask 屏蔽，避免越界读写
    rows = tl.program_id(0) * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    # 通过 page table 把逻辑 token 行映射成 KV 缓存里的物理槽位
    ids = tl.load(table + rows, mask=rows < N, other=0)
    cols = tl.arange(0, 512)
    # nope 部分：加载 FP8 数据并转 fp32，乘以所在 128 维分组对应的 scale，再转 bf16 写出
    x = tl.load(q + ids[:, None] * QS + cols[None, :],
                mask=rows[:, None] < N, other=0.0).to(tl.float32)
    s = tl.load(scales + ids[:, None] * SS + (cols[None, :] // 128),
                mask=rows[:, None] < N, other=0)
    tl.store(out + rows[:, None] * 576 + cols[None, :],
             (x * s).to(tl.bfloat16), mask=rows[:, None] < N)
    # rope 部分本身就是 bf16，不需要去量化，直接拷贝到输出的 [512:576)
    rc = tl.arange(0, 64)
    r = tl.load(rope + ids[:, None] * RS + rc[None, :],
                mask=rows[:, None] < N, other=0)
    tl.store(out + rows[:, None] * 576 + 512 + rc[None, :],
             r, mask=rows[:, None] < N)


if __name__ == "__main__":
    raise Exception("UT is in quant_k_cache.py")


@triton.jit
def _concat_cast_kv_fp8_pad_kernel(
    out_ptr,
    k_ptr,
    kr_ptr,
    num_tokens,
    k_stride,
    kr_stride,
    NOPE: tl.constexpr,
    ROPE: tl.constexpr,
):
    """Row program: real rows write cast(k)||cast(k_rope); pad-band rows
    write zeros (the -1-sentinel landing pad the kernel's clamp maps to)."""
    row = tl.program_id(0).to(tl.int64)
    offs_n = tl.arange(0, NOPE)
    offs_r = tl.arange(0, ROPE)
    head = NOPE + ROPE
    if row < num_tokens:
        v_n = tl.load(k_ptr + row * k_stride + offs_n)
        tl.store(out_ptr + row * head + offs_n, v_n.to(tl.float8e4nv))
        v_r = tl.load(kr_ptr + row * kr_stride + offs_r)
        tl.store(out_ptr + row * head + NOPE + offs_r, v_r.to(tl.float8e4nv))
    else:
        zero_n = tl.zeros([NOPE], dtype=tl.float32).to(tl.float8e4nv)
        zero_r = tl.zeros([ROPE], dtype=tl.float32).to(tl.float8e4nv)
        tl.store(out_ptr + row * head + offs_n, zero_n)
        tl.store(out_ptr + row * head + NOPE + offs_r, zero_r)


def concat_cast_kv_fp8_pad(
    out: torch.Tensor,
    k: torch.Tensor,
    k_rope: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """Fused non-prefix Q8KV8 KV prep: cast-concat k (nope latent) and k_rope
    directly into the persistent fp8 kv_buf and zero the trailing pad band —
    replaces the bf16 `_cat` materialization + `.copy_` cast + `.zero_()`
    tail (3 kernels + one [tokens, 576] bf16 alloc).  Same bf16->fp8
    store-cast the gather kernel uses (bit-identical bytes).

    ``out``: [total_rows, 576] fp8 slice (total_rows = num_tokens + pad band);
    ``k``: [num_tokens, NOPE] bf16 view; ``k_rope``: [num_tokens, ROPE] bf16.
    """
    total_rows, head = out.shape
    nope = k.shape[-1]
    rope = k_rope.shape[-1]
    assert head == nope + rope and out.dtype == torch.float8_e4m3fn
    k2 = k.view(num_tokens, nope)
    kr2 = k_rope.view(num_tokens, rope)
    assert k2.stride(-1) == 1 and kr2.stride(-1) == 1
    _concat_cast_kv_fp8_pad_kernel[(total_rows,)](
        out,
        k2,
        kr2,
        num_tokens,
        k2.stride(0),
        kr2.stride(0),
        NOPE=nope,
        ROPE=rope,
    )
    return out
