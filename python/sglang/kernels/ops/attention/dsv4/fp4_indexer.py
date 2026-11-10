from __future__ import annotations

import torch
import triton
import triton.language as tl

# One index-K slot: 64 packed e2m1 bytes and four ue8m0 block exponents.
INDEX_K_SLOT_BYTES = 64 + 4


@triton.jit
def _select_group_value(group, v0, v1, v2, v3):
    return tl.where(
        group == 0,
        v0,
        tl.where(group == 1, v1, tl.where(group == 2, v2, v3)),
    )


@triton.jit
def _ceil_ue8m0_exp(x):
    bits = x.to(tl.int32, bitcast=True)
    exp = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    exp += mantissa != 0
    return tl.minimum(tl.maximum(exp, 1), 254)


@triton.jit
def _fp4_e2m1_code(x):
    ax = tl.minimum(tl.abs(x), 6.0)
    idx = (ax > 0.25).to(tl.uint8)
    idx += (ax > 0.75).to(tl.uint8)
    idx += (ax > 1.25).to(tl.uint8)
    idx += (ax > 1.75).to(tl.uint8)
    idx += (ax > 2.5).to(tl.uint8)
    idx += (ax > 3.5).to(tl.uint8)
    idx += (ax > 5.0).to(tl.uint8)
    sign = ((x < 0) & (idx != 0)).to(tl.uint8)
    return idx | (sign << 3)


@triton.jit
def _fp4_e2m1_code_rne(x):
    """Round-to-nearest-even e2m1 code, matching the reference rounding: at an
    exact half-way value the even grid index wins."""
    ax = tl.minimum(tl.abs(x), 6.0)
    idx = (ax >= 0.25).to(tl.uint8)
    idx += (ax >= 0.75).to(tl.uint8)
    idx += (ax >= 1.25).to(tl.uint8)
    idx += (ax >= 1.75).to(tl.uint8)
    idx += (ax >= 2.5).to(tl.uint8)
    idx += (ax >= 3.5).to(tl.uint8)
    idx += (ax >= 5.0).to(tl.uint8)
    # Round-half-to-even: an odd index at an exact boundary drops to the even one.
    is_boundary = (
        (ax == 0.25)
        | (ax == 0.75)
        | (ax == 1.25)
        | (ax == 1.75)
        | (ax == 2.5)
        | (ax == 3.5)
        | (ax == 5.0)
    )
    idx = tl.where(is_boundary & ((idx & 1) == 1), idx - 1, idx)
    sign = ((x < 0) & (idx != 0)).to(tl.uint8)
    return idx | (sign << 3)


@triton.jit
def _quantize_fp4_indexer_kernel(
    x,
    x_fp4,
    x_sf,
    BLOCK_N: tl.constexpr,
    GROUP_N: tl.constexpr,
    RNE: tl.constexpr,
):
    token_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    values = tl.load(x + token_id * BLOCK_N + offs).to(tl.float32)
    abs_values = tl.abs(values)

    amax0 = tl.max(tl.where(offs < GROUP_N, abs_values, 0.0), axis=0)
    amax1 = tl.max(
        tl.where((GROUP_N <= offs) & (offs < 2 * GROUP_N), abs_values, 0.0),
        axis=0,
    )
    amax2 = tl.max(
        tl.where((2 * GROUP_N <= offs) & (offs < 3 * GROUP_N), abs_values, 0.0),
        axis=0,
    )
    amax3 = tl.max(tl.where(3 * GROUP_N <= offs, abs_values, 0.0), axis=0)

    sf0 = tl.maximum(amax0 / 6.0, 1.0e-4)
    sf1 = tl.maximum(amax1 / 6.0, 1.0e-4)
    sf2 = tl.maximum(amax2 / 6.0, 1.0e-4)
    sf3 = tl.maximum(amax3 / 6.0, 1.0e-4)

    exp0 = _ceil_ue8m0_exp(sf0)
    exp1 = _ceil_ue8m0_exp(sf1)
    exp2 = _ceil_ue8m0_exp(sf2)
    exp3 = _ceil_ue8m0_exp(sf3)

    packed_sf = exp0 | (exp1 << 8) | (exp2 << 16) | (exp3 << 24)
    tl.store(x_sf + token_id, packed_sf)

    pair_offsets = tl.arange(0, BLOCK_N // 2)
    offs0 = pair_offsets * 2
    offs1 = offs0 + 1
    group0 = offs0 // GROUP_N
    group1 = offs1 // GROUP_N
    scale_exp0 = _select_group_value(group0, exp0, exp1, exp2, exp3)
    scale_exp1 = _select_group_value(group1, exp0, exp1, exp2, exp3)
    scale0 = (scale_exp0 << 23).to(tl.float32, bitcast=True)
    scale1 = (scale_exp1 << 23).to(tl.float32, bitcast=True)

    v0 = tl.load(x + token_id * BLOCK_N + offs0).to(tl.float32) / scale0
    v1 = tl.load(x + token_id * BLOCK_N + offs1).to(tl.float32) / scale1
    if RNE:
        code0 = _fp4_e2m1_code_rne(v0)
        code1 = _fp4_e2m1_code_rne(v1)
    else:
        code0 = _fp4_e2m1_code(v0)
        code1 = _fp4_e2m1_code(v1)
    packed = (code0 & 0x0F) | ((code1 & 0x0F) << 4)
    tl.store(x_fp4 + token_id * (BLOCK_N // 2) + pair_offsets, packed)


@triton.jit
def _quantize_fp4_indexer_rows(
    x,
    x_fp4,
    x_sf,
    M,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_N: tl.constexpr,
    RNE: tl.constexpr,
):
    tl.static_assert(BLOCK_N == 128 and GROUP_N == 32)
    # Each reduction covers one scale group. Keep its values for packing,
    # avoiding four masked full-row reductions and a second input load.
    group = tl.program_id(0) * BLOCK_M * 4 + tl.arange(0, BLOCK_M * 4)
    offs = tl.arange(0, GROUP_N)
    values = tl.load(
        x + group[:, None].to(tl.int64) * GROUP_N + offs[None, :],
        group[:, None] < M * 4,
        0,
    ).to(tl.float32)
    amax = tl.max(tl.abs(values), axis=1)
    exp = _ceil_ue8m0_exp(tl.maximum(amax / 6.0, 1.0e-4))
    scale = (exp << 23).to(tl.float32, bitcast=True)
    v0, v1 = tl.split(
        tl.reshape(values / scale[:, None], (BLOCK_M * 4, GROUP_N // 2, 2))
    )
    if RNE:
        code0 = _fp4_e2m1_code_rne(v0)
        code1 = _fp4_e2m1_code_rne(v1)
    else:
        code0 = _fp4_e2m1_code(v0)
        code1 = _fp4_e2m1_code(v1)
    packed = (code0 & 0x0F) | ((code1 & 0x0F) << 4)
    tl.store(
        x_fp4
        + group[:, None].to(tl.int64) * (GROUP_N // 2)
        + tl.arange(0, GROUP_N // 2)[None, :],
        packed,
        group[:, None] < M * 4,
    )
    # The four exponents occupy disjoint bytes, so integer sum packs them.
    shifts = tl.arange(0, 4) * 8
    packed_sf = tl.sum(
        tl.reshape(exp.to(tl.uint32), (BLOCK_M, 4)) << shifts[None, :], axis=1
    )
    token_id = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(x_sf + token_id, packed_sf.to(tl.int32), token_id < M)


@triton.jit
def _store_fp4_index_k_cache_kernel(
    k_fp4,
    k_sf,
    cache,
    loc,
    page_size: tl.constexpr,
    cache_stride: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    cache_loc = tl.load(loc + token_id)
    page = cache_loc // page_size
    page_offset = cache_loc - page * page_size

    k = tl.load(k_fp4 + token_id * BLOCK + offsets)
    tl.store(cache + page * cache_stride + page_offset * BLOCK + offsets, k)

    sf = tl.load(k_sf + token_id)
    sf_offsets = tl.arange(0, 4)
    sf_bytes = (sf >> (sf_offsets * 8)) & 0xFF
    tl.store(
        cache + page * cache_stride + page_size * BLOCK + page_offset * 4 + sf_offsets,
        sf_bytes,
    )


def quantize_fp4_indexer_tensor(
    x: torch.Tensor, rne: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-32 ue8m0 fp4 quantize. rne=True uses round-to-nearest-even (the dsv41
    reference rounding); the default keeps the c4 threshold behavior."""
    assert x.shape[-1] == 128
    x = x.contiguous().view(-1, x.shape[-1])
    x_fp4 = torch.empty((x.shape[0], 64), device=x.device, dtype=torch.int8)
    x_sf = torch.empty((x.shape[0],), device=x.device, dtype=torch.int32)
    from sglang.srt.runtime_context import get_platform

    if x.shape[0] >= 4096 and get_platform().is_blackwell:
        from sglang.srt.utils.community_opt import log_community_opt_once

        log_community_opt_once(
            "#39704 quantize_fp4_indexer_rows",
            "row-group fp4 indexer quant kernel enabled (rows=%d)",
            x.shape[0],
        )
        # Independent rows share a CTA to avoid one block per 128 values.
        _quantize_fp4_indexer_rows[(triton.cdiv(x.shape[0], 8),)](
            x,
            x_fp4,
            x_sf,
            x.shape[0],
            8,
            BLOCK_N=128,
            GROUP_N=32,
            RNE=rne,
            num_warps=4,
        )
    elif x.shape[0] > 0:
        _quantize_fp4_indexer_kernel[(x.shape[0],)](
            x,
            x_fp4,
            x_sf,
            BLOCK_N=128,
            GROUP_N=32,
            RNE=rne,
        )
    return x_fp4, x_sf


def store_fp4_index_k_cache(
    input: torch.Tensor,
    cache: torch.Tensor,
    loc: torch.Tensor,
    *,
    page_size: int,
    rne: bool = False,
) -> None:
    assert input.shape[-1] == 128
    k_fp4, k_sf = quantize_fp4_indexer_tensor(input.contiguous(), rne=rne)
    n_tokens = input.numel() // input.shape[-1]
    assert k_fp4.shape == (n_tokens, 64)
    assert k_sf.shape == (n_tokens,)
    assert cache.shape[1] == page_size * (64 + 4)

    if n_tokens == 0:
        return
    _store_fp4_index_k_cache_kernel[(n_tokens,)](
        k_fp4.view(torch.uint8),
        k_sf,
        cache,
        loc,
        page_size,
        cache.stride(0),
        BLOCK=64,
    )


@triton.jit
def _gather_fp4_index_k_kernel(
    cache,
    cache_stride,
    slots,
    payload,
    scales,
    n,
    page_size: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    # int32 view of a page: page_size * 16 payload words, then page_size scale words.
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    valid = rows < n
    slot = tl.load(slots + rows, valid, 0).to(tl.int64)
    page = slot // page_size
    offset = slot - page * page_size
    page_base = cache + page * cache_stride
    words = tl.arange(0, 16)
    payload_words = tl.load(
        page_base[:, None] + offset[:, None] * 16 + words[None, :], valid[:, None]
    )
    tl.store(
        payload + rows[:, None].to(tl.int64) * 16 + words[None, :],
        payload_words,
        valid[:, None],
    )
    tl.store(scales + rows, tl.load(page_base + page_size * 16 + offset, valid), valid)


def gather_fp4_index_k(
    cache: torch.Tensor, slots: torch.Tensor, *, page_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Packed fp4 rows at ``slots`` of the index-K ``cache`` (``[pages, page_size *
    68]`` uint8, ``store_fp4_index_k_cache``'s layout): (payload int8 ``[n, 64]``,
    ue8m0 scales packed int32 ``[n]``), the layout ``quantize_fp4_indexer_tensor``
    returns."""
    assert cache.shape[1] == page_size * INDEX_K_SLOT_BYTES and cache.stride(1) == 1
    # The kernel addresses slots as `slots + rows`, so a strided view would be read
    # as if its elements were packed.
    assert slots.is_contiguous()
    words = cache.view(torch.int32)
    n = slots.shape[0]
    payload = torch.empty((n, 16), device=cache.device, dtype=torch.int32)
    scales = torch.empty((n,), device=cache.device, dtype=torch.int32)
    if n > 0:
        block_rows = 64
        _gather_fp4_index_k_kernel[(triton.cdiv(n, block_rows),)](
            words,
            words.stride(0),
            slots,
            payload,
            scales,
            n,
            page_size,
            BLOCK_ROWS=block_rows,
            num_warps=4,
        )
    return payload.view(torch.int8), scales


@triton.jit(do_not_specialize=["NUM_SLOTS"])
def _finish_flat_indexer_topk_kernel(
    Indices,
    Slots,
    Starts,
    Pages,
    Raw,
    K: tl.constexpr,
    NUM_SLOTS,
    IDX_STRIDE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    RAW_STRIDE: tl.constexpr,
    WRITE_RAW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    unselected = 2147483647
    idx = tl.load(Indices + row * IDX_STRIDE + col, col < K, unselected)
    idx = tl.sort(tl.where(idx < 0, unselected, idx), descending=False)
    chosen = (col < K) & (idx != unselected)
    slot = tl.load(Slots + tl.minimum(idx, NUM_SLOTS - 1), chosen, -1)
    tl.store(Pages + row * PAGE_STRIDE + col, slot, col < K)
    if WRITE_RAW:
        start = tl.load(Starts + row)
        tl.store(
            Raw + row * RAW_STRIDE + col, tl.where(chosen, idx - start, -1), col < K
        )


def finish_flat_indexer_topk(
    indices: torch.Tensor,
    slots: torch.Tensor,
    request_starts: torch.Tensor,
    page_indices: torch.Tensor,
    raw_indices: torch.Tensor | None,
) -> None:
    """Sort flattened prefill positions and map them to KV slots in one kernel.

    Negative positions and INT32_MAX are unselected. Write only the selected
    rows and columns; callers retain responsibility for resetting output padding.
    """
    rows, k = indices.shape
    if not rows or not k:
        return
    assert indices.dtype == torch.int32 and indices.stride(1) == 1
    assert slots.ndim == 1 and slots.is_contiguous() and slots.numel() > 0
    assert request_starts.shape == (rows,) and request_starts.is_contiguous()
    assert page_indices.shape[0] >= rows and page_indices.shape[1] >= k
    assert page_indices.dtype == torch.int32 and page_indices.stride(1) == 1
    if raw_indices is not None:
        assert raw_indices.shape[0] >= rows and raw_indices.shape[1] >= k
        assert raw_indices.dtype == torch.int32 and raw_indices.stride(1) == 1
    _finish_flat_indexer_topk_kernel[(rows,)](
        indices,
        slots,
        request_starts,
        page_indices,
        raw_indices,
        k,
        slots.numel(),
        indices.stride(0),
        page_indices.stride(0),
        raw_indices.stride(0) if raw_indices is not None else 0,
        raw_indices is not None,
        triton.next_power_of_2(k),
        num_warps=4,
    )
