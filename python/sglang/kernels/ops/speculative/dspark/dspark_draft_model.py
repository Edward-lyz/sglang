from __future__ import annotations

from typing import Optional

import msgspec
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.kernels.ops.speculative.dspark.dispatch import inputs_on_cuda

_BLOCK_V = 1024
_IDX_SENTINEL = tl.constexpr(2147483647)


class SampleStepTokens:
    @classmethod
    def execute(
        cls,
        *,
        step_logits: torch.Tensor,
        temperatures: torch.Tensor,
        greedy_mask: torch.Tensor,
        exp_noise: Optional[torch.Tensor] = None,
        rng_counter: Optional[torch.Tensor] = None,
        rng_step: int = 0,
        rng_num_steps: int = 1,
    ) -> torch.Tensor:
        if step_logits.is_cuda:
            return cls.triton(
                step_logits=step_logits,
                temperatures=temperatures,
                greedy_mask=greedy_mask,
                exp_noise=exp_noise,
                rng_counter=rng_counter,
                rng_step=rng_step,
                rng_num_steps=rng_num_steps,
            )
        return cls.torch(
            step_logits=step_logits,
            temperatures=temperatures,
            greedy_mask=greedy_mask,
            exp_noise=exp_noise,
            rng_counter=rng_counter,
            rng_step=rng_step,
            rng_num_steps=rng_num_steps,
        )

    @classmethod
    def torch(
        cls,
        *,
        step_logits: torch.Tensor,
        temperatures: torch.Tensor,
        greedy_mask: torch.Tensor,
        exp_noise: Optional[torch.Tensor] = None,
        rng_counter: Optional[torch.Tensor] = None,
        rng_step: int = 0,
        rng_num_steps: int = 1,
    ) -> torch.Tensor:
        return sample_step_tokens(
            step_logits=step_logits,
            temperatures=temperatures,
            greedy_mask=greedy_mask,
            exp_noise=exp_noise,
            rng_counter=rng_counter,
            rng_step=rng_step,
            rng_num_steps=rng_num_steps,
        )

    @classmethod
    def triton(
        cls,
        *,
        step_logits: torch.Tensor,
        temperatures: torch.Tensor,
        greedy_mask: torch.Tensor,
        exp_noise: Optional[torch.Tensor] = None,
        rng_counter: Optional[torch.Tensor] = None,
        rng_step: int = 0,
        rng_num_steps: int = 1,
    ) -> torch.Tensor:
        return sample_step_tokens_triton(
            step_logits=step_logits,
            temperatures=temperatures,
            greedy_mask=greedy_mask,
            exp_noise=exp_noise,
            rng_counter=rng_counter,
            rng_step=rng_step,
            rng_num_steps=rng_num_steps,
        )


def _inline_exp_noise_torch(
    *,
    shape: torch.Size,
    device: torch.device,
    rng_counter: torch.Tensor,
    rng_step: int,
    rng_num_steps: int,
) -> torch.Tensor:
    """CPU/eager reference for the inline philox noise: Exp(1) drawn as
    ``-log(u)``, u ~ Uniform(0,1). Distribution-equivalent to the Triton path
    (both draw iid Exp(1)); the seed folds the per-step index exactly as the
    kernel does, but values are not bit-identical because torch and Triton use
    different philox layouts. Only used off CUDA / in the kernel's torch
    fallback -- the graph path never touches this.
    """
    base = int(rng_counter.reshape(-1)[0].item())
    seed = (base * int(rng_num_steps) + int(rng_step)) & 0xFFFFFFFF
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    u = torch.rand(shape, generator=gen, device=device, dtype=torch.float32)
    return -torch.log(u.clamp_min(torch.finfo(torch.float32).tiny))


def sample_step_tokens(
    *,
    step_logits: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
    exp_noise: Optional[torch.Tensor] = None,
    rng_counter: Optional[torch.Tensor] = None,
    rng_step: int = 0,
    rng_num_steps: int = 1,
) -> torch.Tensor:
    probs = torch.softmax(step_logits.float() / temperatures[:, None], dim=-1)
    if rng_counter is not None:
        # Inline RNG: ignore exp_noise entirely and draw the Exp(1) noise here.
        exp_noise = _inline_exp_noise_torch(
            shape=step_logits.shape,
            device=step_logits.device,
            rng_counter=rng_counter,
            rng_step=rng_step,
            rng_num_steps=rng_num_steps,
        )
    else:
        assert exp_noise is not None, (
            "sample_step_tokens needs exp_noise unless rng_counter is set"
        )
    noise = torch.where(greedy_mask[:, None], 1.0, exp_noise)
    return probs.div_(noise).argmax(dim=-1)


@triton.jit
def _online_partial_kernel(
    logits_ptr,
    temperatures_ptr,
    greedy_mask_ptr,
    exp_noise_ptr,
    rng_seed_ptr,
    tile_max_ptr,
    partial_key_ptr,
    partial_idx_ptr,
    V,
    stride_row,
    n_tiles,
    rng_step,
    rng_num_steps,
    BLOCK_V: tl.constexpr,
    INLINE_RNG: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK_V + tl.arange(0, BLOCK_V)
    mask = offs < V
    logits = tl.load(
        logits_ptr + row * stride_row + offs, mask=mask, other=float("-inf")
    ).to(tl.float32)
    temperature = tl.load(temperatures_ptr + row)
    s = logits / temperature
    tile_max = tl.max(s, axis=0)
    greedy = tl.load(greedy_mask_ptr + row) != 0
    if INLINE_RNG:
        # Inline Gumbel noise: draw uniforms with philox and map u -> -log(u),
        # i.e. Exp(1) -- the exact noise exponential_() precomputes. fp32
        # throughout, matching the precomputed buffer's dtype. The seed is read
        # live from a device counter (graph-safe: the host bumps it each decode
        # step, so replays redraw), and folds in the per-block step index so the
        # gamma draft steps -- all baked into one captured graph -- draw
        # independently. The offset row * V + offs keys noise per (row, vocab).
        base = tl.load(rng_seed_ptr)
        seed = (base * rng_num_steps + rng_step).to(tl.uint32)
        u = tl.rand(seed, row * V + offs)
        noise = -tl.log(u)
    else:
        noise = tl.load(exp_noise_ptr + row * V + offs, mask=mask, other=1.0)
    denom = tl.where(greedy, 1.0, noise)
    key = tl.exp(s - tile_max) / denom
    key = tl.where(mask, key, -1.0)
    tile_best = tl.max(key, axis=0)
    idx = tl.where(key == tile_best, offs, _IDX_SENTINEL)
    tl.store(tile_max_ptr + row * n_tiles + tile, tile_max)
    tl.store(partial_key_ptr + row * n_tiles + tile, tile_best)
    tl.store(partial_idx_ptr + row * n_tiles + tile, tl.min(idx, axis=0))


@triton.jit
def _online_combine_kernel(
    tile_max_ptr,
    partial_key_ptr,
    partial_idx_ptr,
    next_tokens_ptr,
    n_tiles,
    BLOCK_TILES: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_TILES)
    mask = offs < n_tiles
    tile_max = tl.load(
        tile_max_ptr + row * n_tiles + offs, mask=mask, other=float("-inf")
    )
    keys = tl.load(partial_key_ptr + row * n_tiles + offs, mask=mask, other=-1.0)
    idxs = tl.load(
        partial_idx_ptr + row * n_tiles + offs, mask=mask, other=_IDX_SENTINEL
    )
    global_max = tl.max(tile_max, axis=0)
    rescaled = keys * tl.exp(tile_max - global_max)
    rescaled = tl.where(mask, rescaled, -1.0)
    best = tl.max(rescaled, axis=0)
    cand = tl.where(rescaled == best, idxs, _IDX_SENTINEL)
    next_token = tl.min(cand, axis=0)
    # Degenerate rows (e.g. all -inf logits) leave cand all-sentinel; clamp to 0
    # so a valid token id is emitted instead of an out-of-range 2147483647.
    next_token = tl.where(next_token == _IDX_SENTINEL, 0, next_token)
    tl.store(next_tokens_ptr + row, next_token.to(tl.int64))


def sample_step_tokens_triton(
    *,
    step_logits: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
    exp_noise: Optional[torch.Tensor] = None,
    rng_counter: Optional[torch.Tensor] = None,
    rng_step: int = 0,
    rng_num_steps: int = 1,
) -> torch.Tensor:
    bs, V = step_logits.shape
    device = step_logits.device
    assert step_logits.stride(1) == 1, "step_logits rows must be contiguous"
    stride_row = step_logits.stride(0)
    temperatures = temperatures.to(torch.float32).contiguous()
    greedy_mask = greedy_mask.to(torch.int32).contiguous()

    inline_rng = rng_counter is not None
    if inline_rng:
        # Graph-safe seed: a device counter the host bumps each decode step, so
        # replays redraw. No [bs, V] noise buffer, no exponential_() launch, and
        # no read of exp_noise -- its pointer below is dead (the load is behind a
        # constexpr-False branch) and only kept so the launch signature matches.
        rng_seed = rng_counter
        exp_noise_arg = rng_counter
    else:
        assert exp_noise is not None, (
            "sample_step_tokens_triton needs exp_noise unless rng_counter is set"
        )
        exp_noise_arg = exp_noise.to(torch.float32).contiguous()
        # Dead pointer for the (constexpr-False) inline branch; never loaded.
        rng_seed = greedy_mask

    n_tiles = triton.cdiv(V, _BLOCK_V)
    block_tiles = triton.next_power_of_2(n_tiles)

    tile_max = torch.empty((bs, n_tiles), dtype=torch.float32, device=device)
    partial_key = torch.empty((bs, n_tiles), dtype=torch.float32, device=device)
    partial_idx = torch.empty((bs, n_tiles), dtype=torch.int32, device=device)
    next_tokens = torch.empty((bs,), dtype=torch.int64, device=device)

    tile_grid = (bs, n_tiles)
    row_grid = (bs,)

    _online_partial_kernel[tile_grid](
        step_logits,
        temperatures,
        greedy_mask,
        exp_noise_arg,
        rng_seed,
        tile_max,
        partial_key,
        partial_idx,
        V,
        stride_row,
        n_tiles,
        int(rng_step),
        int(rng_num_steps),
        BLOCK_V=_BLOCK_V,
        INLINE_RNG=inline_rng,
    )
    _online_combine_kernel[row_grid](
        tile_max,
        partial_key,
        partial_idx,
        next_tokens,
        n_tiles,
        BLOCK_TILES=block_tiles,
    )
    return next_tokens


_MARKOV_BLOCK_V = 256
_MARKOV_BLOCK_R = 32


class MarkovGreedyStep:
    """One greedy markov draft step, fused.

    Computes ``argmax_v(base_logits[:, v] + dot(w2_weight[v, :], prev_embeds))``
    in a single pass over the (vocab x rank) weight: no full-vocab bias or
    step-logits materialization, no separate GEMV / add / two-pass argmax
    launches. Numerics: the dot and the add accumulate in fp32, while the eager
    path rounds the GEMV output and the add to bf16 before argmax — near-tie
    winners can differ on rare steps. Drafts are proposals only (target verify
    guards output correctness), so the impact is bounded to accept-rate noise
    on exact ties.
    """

    @classmethod
    def execute(
        cls,
        *,
        base_logits: torch.Tensor,
        prev_embeds: torch.Tensor,
        w2_weight: torch.Tensor,
    ) -> torch.Tensor:
        if base_logits.is_cuda:
            return cls.triton(
                base_logits=base_logits, prev_embeds=prev_embeds, w2_weight=w2_weight
            )
        return cls.torch(
            base_logits=base_logits, prev_embeds=prev_embeds, w2_weight=w2_weight
        )

    @classmethod
    def torch(
        cls,
        *,
        base_logits: torch.Tensor,
        prev_embeds: torch.Tensor,
        w2_weight: torch.Tensor,
    ) -> torch.Tensor:
        return markov_greedy_step(
            base_logits=base_logits, prev_embeds=prev_embeds, w2_weight=w2_weight
        )

    @classmethod
    def triton(
        cls,
        *,
        base_logits: torch.Tensor,
        prev_embeds: torch.Tensor,
        w2_weight: torch.Tensor,
    ) -> torch.Tensor:
        return markov_greedy_step_triton(
            base_logits=base_logits, prev_embeds=prev_embeds, w2_weight=w2_weight
        )


def markov_greedy_step(
    *,
    base_logits: torch.Tensor,
    prev_embeds: torch.Tensor,
    w2_weight: torch.Tensor,
) -> torch.Tensor:
    step_logits = base_logits + F.linear(prev_embeds, w2_weight)
    return torch.argmax(step_logits, dim=-1)


@triton.jit
def _markov_greedy_partial_kernel(
    base_ptr,
    embed_ptr,
    w2_ptr,
    tile_val_ptr,
    tile_idx_ptr,
    V,
    R,
    stride_base_row,
    stride_embed_row,
    stride_w2_v,
    n_tiles,
    BLOCK_V: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs_v = tile * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_v = offs_v < V
    acc = tl.zeros([BLOCK_V], dtype=tl.float32)
    for r0 in range(0, R, BLOCK_R):
        offs_r = r0 + tl.arange(0, BLOCK_R)
        mask_r = offs_r < R
        embed = tl.load(
            embed_ptr + row * stride_embed_row + offs_r, mask=mask_r, other=0.0
        ).to(tl.float32)
        w2 = tl.load(
            w2_ptr + offs_v[:, None] * stride_w2_v + offs_r[None, :],
            mask=mask_v[:, None] & mask_r[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(w2 * embed[None, :], axis=1)
    base = tl.load(
        base_ptr + row * stride_base_row + offs_v, mask=mask_v, other=float("-inf")
    ).to(tl.float32)
    score = tl.where(mask_v, base + acc, float("-inf"))
    tile_best = tl.max(score, axis=0)
    # First-index tie-break within the tile, matching torch.argmax.
    idx = tl.where(score == tile_best, offs_v, _IDX_SENTINEL)
    tl.store(tile_val_ptr + row * n_tiles + tile, tile_best)
    tl.store(tile_idx_ptr + row * n_tiles + tile, tl.min(idx, axis=0))


@triton.jit
def _markov_greedy_combine_kernel(
    tile_val_ptr,
    tile_idx_ptr,
    next_tokens_ptr,
    n_tiles,
    BLOCK_TILES: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_TILES)
    mask = offs < n_tiles
    vals = tl.load(tile_val_ptr + row * n_tiles + offs, mask=mask, other=float("-inf"))
    idxs = tl.load(tile_idx_ptr + row * n_tiles + offs, mask=mask, other=_IDX_SENTINEL)
    best = tl.max(vals, axis=0)
    # Lowest global index among equal-valued tiles, matching torch.argmax.
    cand = tl.where(vals == best, idxs, _IDX_SENTINEL)
    tl.store(next_tokens_ptr + row, tl.min(cand, axis=0).to(tl.int64))


def markov_greedy_step_triton(
    *,
    base_logits: torch.Tensor,
    prev_embeds: torch.Tensor,
    w2_weight: torch.Tensor,
) -> torch.Tensor:
    bs, vocab = base_logits.shape
    rank = w2_weight.shape[1]
    device = base_logits.device
    assert base_logits.stride(1) == 1, "base_logits rows must be contiguous"
    assert w2_weight.stride(1) == 1, "markov_w2 weight rows must be contiguous"
    prev_embeds = prev_embeds.contiguous()

    n_tiles = triton.cdiv(vocab, _MARKOV_BLOCK_V)
    tile_vals = torch.empty((bs, n_tiles), dtype=torch.float32, device=device)
    tile_idxs = torch.empty((bs, n_tiles), dtype=torch.int32, device=device)
    next_tokens = torch.empty((bs,), dtype=torch.int64, device=device)

    _markov_greedy_partial_kernel[(bs, n_tiles)](
        base_logits,
        prev_embeds,
        w2_weight,
        tile_vals,
        tile_idxs,
        vocab,
        rank,
        base_logits.stride(0),
        prev_embeds.stride(0),
        w2_weight.stride(0),
        n_tiles,
        BLOCK_V=_MARKOV_BLOCK_V,
        BLOCK_R=_MARKOV_BLOCK_R,
    )
    _markov_greedy_combine_kernel[(bs,)](
        tile_vals,
        tile_idxs,
        next_tokens,
        n_tiles,
        BLOCK_TILES=triton.next_power_of_2(n_tiles),
    )
    return next_tokens


_STACKED_WEIGHT_CACHE: dict[int, _StackedWkvWeight] = {}


class CommitKvProj:
    @classmethod
    def execute(
        cls,
        *,
        main_x: torch.Tensor,
        wkv_linears: list[torch.nn.Module],
    ) -> list[torch.Tensor]:
        if main_x.is_cuda and _fused_commit_kv_proj_supported(wkv_linears=wkv_linears):
            return cls.triton(main_x=main_x, wkv_linears=wkv_linears)
        return cls.torch(main_x=main_x, wkv_linears=wkv_linears)

    @classmethod
    def torch(
        cls,
        *,
        main_x: torch.Tensor,
        wkv_linears: list[torch.nn.Module],
    ) -> list[torch.Tensor]:
        return commit_kv_proj(main_x=main_x, wkv_linears=wkv_linears)

    @classmethod
    def triton(
        cls,
        *,
        main_x: torch.Tensor,
        wkv_linears: list[torch.nn.Module],
    ) -> list[torch.Tensor]:
        return commit_kv_proj_fused(main_x=main_x, wkv_linears=wkv_linears)


def commit_kv_proj(
    *,
    main_x: torch.Tensor,
    wkv_linears: list[torch.nn.Module],
) -> list[torch.Tensor]:
    return [linear(main_x)[0] for linear in wkv_linears]


def commit_kv_proj_fused(
    *,
    main_x: torch.Tensor,
    wkv_linears: list[torch.nn.Module],
) -> list[torch.Tensor]:
    num_stages = len(wkv_linears)
    stacked = _stacked_wkv_weight(wkv_linears=wkv_linears)

    if stacked.deepgemm_scale is not None:
        # [fusion-opt] SGLANG_FUSE_DSPARK_COMMIT_KV_DEEPGEMM
        kv_all = wkv_linears[0].quant_method.w8a8_mxfp8_linear(
            input=main_x,
            weight=stacked.weight,
            weight_scale=stacked.deepgemm_scale,
            input_scale=None,
            bias=None,
            weight_scale_swizzled=None,
        )
        _selfcheck_deepgemm_commit_once(main_x, kv_all, stacked, wkv_linears)
    elif stacked.mxfp8_scale is not None:
        kv_all = wkv_linears[0].quant_method.w8a8_mxfp8_linear(
            input=main_x,
            weight=stacked.weight,
            weight_scale=stacked.mxfp8_scale,
            input_scale=None,
            bias=None,
        )
    elif stacked.fp8_scale is not None:
        quant_method = wkv_linears[0].quant_method
        kv_all = quant_method.w8a8_block_fp8_linear(
            input=main_x,
            weight=stacked.weight,
            block_size=quant_method.quant_config.weight_block_size,
            weight_scale=stacked.fp8_scale,
            input_scale=None,
            bias=None,
        )
    else:
        kv_all = torch.nn.functional.linear(main_x, stacked.weight)

    head_dim = kv_all.shape[-1] // num_stages
    return [
        kv_all[:, i * head_dim : (i + 1) * head_dim].contiguous()
        for i in range(num_stages)
    ]


class _StackedWkvWeight(msgspec.Struct):
    weight: torch.Tensor
    fp8_scale: Optional[torch.Tensor]
    mxfp8_scale: Optional[torch.Tensor] = None
    # [fusion-opt] DeepGEMM-packed per-row ue8m0 scale of the stacked weight
    deepgemm_scale: Optional[torch.Tensor] = None


_DEEPGEMM_COMMIT_CHECKED = False


def _selfcheck_deepgemm_commit_once(main_x, kv_all, stacked, wkv_linears) -> None:
    """First call only: compare against the Triton block-fp8 path it replaces."""
    global _DEEPGEMM_COMMIT_CHECKED
    if _DEEPGEMM_COMMIT_CHECKED or stacked.fp8_scale is None or main_x.shape[0] == 0:
        return
    if torch.cuda.is_current_stream_capturing():
        return  # .item() would break graph capture (decode / target-verify)
    _DEEPGEMM_COMMIT_CHECKED = True
    import logging

    quant_method = wkv_linears[0].quant_method
    ref = quant_method.w8a8_block_fp8_linear(
        input=main_x,
        weight=stacked.weight,
        block_size=quant_method.quant_config.weight_block_size,
        weight_scale=stacked.fp8_scale,
        input_scale=None,
        bias=None,
    )
    err = (kv_all.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-12)
    logging.getLogger(__name__).info(
        "[fusion-opt] DSPARK commit wkv DeepGEMM vs Triton rel_err=%.3e rows=%d",
        err.item(),
        main_x.shape[0],
    )


def _deepgemm_stack_applies(*, wkv_linears: list[torch.nn.Module]) -> bool:
    from sglang.srt.environ import envs

    if not envs.SGLANG_FUSE_DSPARK_COMMIT_KV_DEEPGEMM.get():
        return False
    quant_method = wkv_linears[0].quant_method
    backend = getattr(quant_method, "mxfp8_dense_backend", None)
    return (
        backend is not None
        and backend.is_deep_gemm()
        and all(getattr(l, "block_fp8_mxfp8_ready", False) for l in wkv_linears)
    )


def _build_deepgemm_scale(
    *, wkv_linears: list[torch.nn.Module], weight: torch.Tensor, sf_fp32: torch.Tensor
) -> Optional[torch.Tensor]:
    """Per-row e8m0 of the stacked block scales, packed the way the DeepGEMM MXFP8
    linear of the per-stage layers is (Fp8LinearMethod._process_mxfp8_linear_weight_scale)."""
    import logging

    quant_method = wkv_linears[0].quant_method
    try:
        n, k = weight.shape
        sf = sf_fp32.detach().float().contiguous()
        if sf.shape[-1] != k // 32:
            raise ValueError(f"scale {tuple(sf.shape)} is not 32-wide-K for {n}x{k}")
        bits = sf.view(torch.int32)
        if not bool(torch.all((bits & 0x7FFFFF) == 0)) or not bool(torch.all(sf > 0)):
            raise ValueError("scales are not positive powers of two")
        scale_u8 = (bits >> 23).to(torch.uint8)
        if scale_u8.shape[0] != n:  # block rows -> per-row
            block_n = quant_method.quant_config.weight_block_size[0]
            scale_u8 = scale_u8.repeat_interleave(block_n, dim=0)[:n]
        scale_u8 = scale_u8.contiguous()
        if scale_u8.shape != (n, k // 32):
            raise ValueError(f"per-row scale shape {tuple(scale_u8.shape)}")
        holder = torch.nn.Module()
        holder.weight = torch.nn.Parameter(weight, requires_grad=False)
        quant_method._process_mxfp8_linear_weight_scale(holder, scale_u8=scale_u8)
        if getattr(holder, "weight_scale_inv_swizzled", None) is not None:
            return None
        return holder.weight_scale_inv_deepgemm.data
    except Exception as e:  # keep the Triton path on any layout surprise
        logging.getLogger(__name__).warning(
            "[fusion-opt] DSPARK commit DeepGEMM stack unavailable: %s", e
        )
        return None


def _stacked_wkv_weight(*, wkv_linears: list[torch.nn.Module]) -> _StackedWkvWeight:
    key = id(wkv_linears[0])
    cached = _STACKED_WEIGHT_CACHE.get(key)
    if cached is None:
        cached = _build_stacked_wkv_weight(wkv_linears=wkv_linears)
        _STACKED_WEIGHT_CACHE[key] = cached
    return cached


def _block_quant_stack_applies(*, wkv_linears: list[torch.nn.Module]) -> bool:
    quant_method = wkv_linears[0].quant_method
    block_quant = hasattr(quant_method, "block_quant") and quant_method.block_quant
    if not (block_quant and hasattr(quant_method, "w8a8_block_fp8_linear")):
        return False
    block_out = quant_method.quant_config.weight_block_size[0]
    return all(
        linear.weight.dtype == torch.float8_e4m3fn
        and linear.weight.shape[0] % block_out == 0
        for linear in wkv_linears
    )


def _dequant_supported(linear: torch.nn.Module) -> bool:
    """Mirrors the preconditions asserted in _dequant_linear_weight."""
    weight = linear.weight
    if weight.dtype in (torch.bfloat16, torch.float16, torch.float32):
        return True
    if weight.dtype != torch.float8_e4m3fn:
        return False
    block = 128
    out_dim, in_dim = weight.shape
    expected_scale_shape = (
        (out_dim + block - 1) // block,
        (in_dim + block - 1) // block,
    )
    return tuple(linear.weight_scale_inv.shape) == expected_scale_shape


def _fused_commit_kv_proj_supported(*, wkv_linears: list[torch.nn.Module]) -> bool:
    """Whether _build_stacked_wkv_weight can handle these weights; unsupported
    quant schemes fall back to the per-linear torch path in execute()."""
    if _block_quant_stack_applies(wkv_linears=wkv_linears):
        return True
    return all(_dequant_supported(linear) for linear in wkv_linears)


def _build_stacked_wkv_weight(
    *, wkv_linears: list[torch.nn.Module]
) -> _StackedWkvWeight:
    if _block_quant_stack_applies(wkv_linears=wkv_linears):
        weight = torch.cat([linear.weight for linear in wkv_linears], dim=0)
        backend = getattr(wkv_linears[0].quant_method, "mxfp8_dense_backend", None)
        if (
            backend is not None
            and (backend.is_flashinfer_cutlass() or backend.is_flashinfer_cutedsl())
            and all(
                getattr(linear, "block_fp8_mxfp8_ready", False)
                and linear.weight.shape[0] % 128 == 0
                for linear in wkv_linears
            )
        ):
            # Each projection ends on a 128-row scale tile, so concatenating
            # the prepared tiles preserves FlashInfer's swizzled scale layout.
            scale = torch.cat(
                [linear.weight_scale_inv_swizzled.reshape(-1) for linear in wkv_linears]
            )
            return _StackedWkvWeight(weight=weight, fp8_scale=None, mxfp8_scale=scale)
        if wkv_linears[0].weight_scale_inv.dtype == torch.int32:
            from sglang.srt.layers.quantization.fp8_utils import (
                inverse_transform_scale_ue8m0,
                transform_scale_ue8m0,
            )

            sf_fp32 = torch.cat(
                [
                    inverse_transform_scale_ue8m0(
                        linear.weight_scale_inv, mn=linear.weight.shape[0]
                    )
                    for linear in wkv_linears
                ],
                dim=0,
            )
            scale = transform_scale_ue8m0(sf_fp32, mn=weight.shape[0])
            dg_scale = (
                _build_deepgemm_scale(
                    wkv_linears=wkv_linears, weight=weight, sf_fp32=sf_fp32
                )
                if _deepgemm_stack_applies(wkv_linears=wkv_linears)
                else None
            )
            return _StackedWkvWeight(
                weight=weight, fp8_scale=scale, deepgemm_scale=dg_scale
            )
        scale = torch.cat([linear.weight_scale_inv for linear in wkv_linears], dim=0)
        if scale.dim() >= 2 and scale.stride(-2) != 1:
            scale = scale.transpose(-2, -1).contiguous().transpose(-2, -1)
        dg_scale = (
            _build_deepgemm_scale(
                wkv_linears=wkv_linears, weight=weight, sf_fp32=scale.float()
            )
            if _deepgemm_stack_applies(wkv_linears=wkv_linears)
            else None
        )
        return _StackedWkvWeight(weight=weight, fp8_scale=scale, deepgemm_scale=dg_scale)
    weight = torch.cat(
        [_dequant_linear_weight(linear) for linear in wkv_linears], dim=0
    )
    return _StackedWkvWeight(weight=weight, fp8_scale=None)


def _dequant_linear_weight(linear: torch.nn.Module) -> torch.Tensor:
    weight = linear.weight
    if weight.dtype in (torch.bfloat16, torch.float16, torch.float32):
        return weight.to(torch.bfloat16)
    assert weight.dtype == torch.float8_e4m3fn, (
        f"unsupported wkv weight dtype {weight.dtype} for the fused commit kv proj; "
        f"execute() should have routed this to the torch path "
        f"(_fused_commit_kv_proj_supported)"
    )
    block = 128
    scale = linear.weight_scale_inv
    out_dim, in_dim = weight.shape
    expected_scale_shape = (
        (out_dim + block - 1) // block,
        (in_dim + block - 1) // block,
    )
    assert tuple(scale.shape) == expected_scale_shape, (
        f"wkv weight_scale_inv shape {tuple(scale.shape)} does not match the "
        f"128x128 block grid {expected_scale_shape} for weight {tuple(weight.shape)}; "
        f"execute() should have routed this to the torch path "
        f"(_fused_commit_kv_proj_supported)"
    )
    scale_full = scale.repeat_interleave(block, dim=0)[:out_dim]
    scale_full = scale_full.repeat_interleave(block, dim=1)[:, :in_dim]
    return (weight.to(torch.float32) * scale_full.to(torch.float32)).to(torch.bfloat16)


_BLOCK = 1024


class BuildStepLocal:
    @classmethod
    def execute(cls, *args, **kwargs) -> torch.Tensor:
        if inputs_on_cuda(*args, **kwargs):
            return cls.triton(*args, **kwargs)
        return cls.torch(*args, **kwargs)

    @classmethod
    def torch(cls, *, bias: torch.Tensor, base_local: torch.Tensor) -> torch.Tensor:
        return build_step_local(bias=bias, base_local=base_local)

    @classmethod
    def triton(cls, *, bias: torch.Tensor, base_local: torch.Tensor) -> torch.Tensor:
        return build_step_local_triton(bias=bias, base_local=base_local)


def build_step_local(*, bias: torch.Tensor, base_local: torch.Tensor) -> torch.Tensor:
    per_partition = base_local.shape[-1]
    pad = per_partition - bias.shape[-1]
    padded = (
        F.pad(bias.to(torch.float32), (0, pad)) if pad > 0 else bias.to(torch.float32)
    )
    return base_local + padded


@triton.jit
def _build_step_local_kernel(
    bias_ptr,
    base_ptr,
    out_ptr,
    org_width,
    per_partition,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK + tl.arange(0, BLOCK)
    mask = offs < per_partition
    base = tl.load(base_ptr + row * per_partition + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    bias = tl.load(
        bias_ptr + row * org_width + offs, mask=offs < org_width, other=0.0
    ).to(tl.float32)
    tl.store(out_ptr + row * per_partition + offs, base + bias, mask=mask)


def build_step_local_triton(
    *, bias: torch.Tensor, base_local: torch.Tensor
) -> torch.Tensor:
    bs, per_partition = base_local.shape
    org_width = bias.shape[-1]
    base_local = base_local.contiguous()
    bias = bias.contiguous()
    out = torch.empty(
        (bs, per_partition), dtype=torch.float32, device=base_local.device
    )
    grid = (bs, triton.cdiv(per_partition, _BLOCK))
    _build_step_local_kernel[grid](
        bias, base_local, out, org_width, per_partition, BLOCK=_BLOCK
    )
    return out
