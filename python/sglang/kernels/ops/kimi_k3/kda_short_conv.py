"""CUDA JIT Kimi-K3 KDA short_conv: fused q/k/v causal conv1d + SiLU.

Replaces the three serial Triton `causal_conv1d_fn` calls in
`srt/layers/attention/linear/kda_backend.py::KDAAttnBackend.forward_extend`
with one fused launch plus a small conv-state update launch. Bit-exact
against the Triton path (`torch.equal`) and 1.787x faster aggregated over
5 shapes on sm_100; see `kimi_k3/kda_short_conv.cuh` for why the arithmetic
(even/odd fma asymmetry, inline-asm SiLU) must not be touched.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.ops.mamba.causal_conv1d_triton import PAD_SLOT_ID

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)

# Shipped tiling (reference variant 200): 4 channels/thread, 20 tokens/thread,
# block (32, 4), default cache policy (HINT=0), prefetch depth 5, and the
# Triton-matching even/odd fma asymmetry (MODE=1). A measured local optimum
# whose bit-exactness gate must be re-run if any of these change.
_CH: int = 4
_NT: int = 20
_TX: int = 32
_TG: int = 4
_MODE: int = 1
_HINT: int = 0
_PF: int = 5
_STATE_THREADS: int = 256

# Kimi-K3 KDA per-projection channel count, num_heads * head_dim divided by
# attn_tp_size: 12288 / 6144 / 3072 / 1536 for attn_tp 1 / 2 / 4 / 8. The dim is
# a runtime kernel argument, not a template parameter, so the only structural
# requirement is that the channel tile below divides it -- every attn_tp above
# satisfies that. q_dim == k_dim == v_dim holds at every TP size because the
# three head counts are divided by the same factor and head_v_dim == head_k_dim.
SUPPORTED_WIDTH: int = 4
# One block covers TX * CH channels; both `dim` and `3 * dim` must be multiples.
CHANNEL_TILE: int = _TX * _CH
# x is read CH channels at a time with one CH*2-byte vector load, at an offset
# that is a multiple of CH within the row, so the row pitch must also be a
# multiple of CH or every token after the first is misaligned. 3*dim and the
# production 4*dim both satisfy this.
X_ROW_STRIDE_ALIGN: int = _CH


@cache_once
def _jit_kda_short_conv_module() -> Module:
    """Compile and cache the fused short_conv module.

    SM100+ only: the even/odd accumulate asymmetry this kernel reproduces is
    exactly the Blackwell packed-fp32 (`mul.f32x2`) vectorization ptxas
    applies to the Triton kernel, so the bit-exactness contract does not hold
    on earlier architectures.
    """
    major, _ = torch.cuda.get_device_capability()
    if major < 10:
        raise RuntimeError("kimi_k3_kda_short_conv requires SM100+ (Blackwell)")
    args = make_cpp_args(_CH, _NT, _TX, _TG, _MODE, _HINT, _PF, _STATE_THREADS)
    return load_jit(
        "kimi_k3_kda_short_conv_" + "_".join(args),
        *args,
        cuda_files=["kimi_k3/kda_short_conv.cuh"],
        cuda_wrappers=[("run", f"KdaShortConvKernel<{args}>::run")],
        # No --use_fast_math: SiLU is inline asm, but fast-math would still be
        # free to rewrite the surrounding fp32 accumulate.
        extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-O3"],
    )


def reject_reason(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states_pool: torch.Tensor,
    bias: torch.Tensor | None,
    splits: Sequence[int],
) -> str | None:
    """`None` if this kernel can replace the three causal_conv1d_fn calls,
    otherwise the name of the FIRST predicate that failed (with the offending
    value), for the warn-once path in `covered`.

    Anything rejected must stay on the original Triton branch. `dim` is taken
    from `splits`, so every attn_tp_size whose per-projection dim is a multiple
    of CHANNEL_TILE is covered; the pool and the weight shard the same way,
    each rank holding one contiguous [q | k | v] run.

    `mixed_qkv` only has to be ROW-MAJOR WITH PADDING, not contiguous: in
    production it is the first of
    `torch.split(fused_qkvg_states, [3 * dim, dim], dim=-1)`, i.e. shape
    `(T, 3 * dim)` with stride `(4 * dim, 1)` and `is_contiguous() is False`
    (true at attn_tp 1 as well as 8). Requiring contiguity here made the whole
    integration dead code. The kernel reads the row pitch from
    `mixed_qkv.stride(0)`; the only structural needs are a unit-stride channel
    axis, a pitch wide enough that consecutive token rows do not overlap, and a
    pitch that keeps the CH-channel vector load aligned.
    """
    if len(splits) != 3:
        return f"len(splits) == 3 (got {len(splits)})"
    dim = int(splits[0])
    total = 3 * dim
    if splits[1] != dim or splits[2] != dim:
        return f"q/k/v splits equal (got {list(splits)})"
    if dim <= 0:
        return f"dim > 0 (got {dim})"
    if dim % CHANNEL_TILE != 0:
        return f"dim % CHANNEL_TILE == 0 (dim={dim}, CHANNEL_TILE={CHANNEL_TILE})"
    if bias is not None:
        return "bias is None (got a bias tensor)"
    if not mixed_qkv.is_cuda:
        return f"mixed_qkv.is_cuda (got device {mixed_qkv.device})"
    if mixed_qkv.dtype != torch.bfloat16:
        return f"mixed_qkv.dtype == bfloat16 (got {mixed_qkv.dtype})"
    if mixed_qkv.dim() != 2:
        return f"mixed_qkv.dim() == 2 (got {mixed_qkv.dim()})"
    if mixed_qkv.shape[1] != total:
        return f"mixed_qkv.shape[1] == 3 * dim (got {tuple(mixed_qkv.shape)}, dim={dim})"
    # row-major with padding: unit-stride channels, non-overlapping token rows.
    if mixed_qkv.stride(1) != 1:
        return f"mixed_qkv.stride(1) == 1 (got strides {tuple(mixed_qkv.stride())})"
    if mixed_qkv.stride(0) < total:
        return (
            f"mixed_qkv.stride(0) >= 3 * dim (got strides {tuple(mixed_qkv.stride())}"
            f", 3 * dim = {total})"
        )
    if mixed_qkv.stride(0) % X_ROW_STRIDE_ALIGN != 0:
        return (
            f"mixed_qkv.stride(0) % {X_ROW_STRIDE_ALIGN} == 0 "
            f"(got strides {tuple(mixed_qkv.stride())})"
        )
    if conv_weights.dtype != torch.float32:
        return f"conv_weights.dtype == float32 (got {conv_weights.dtype})"
    if not conv_weights.is_contiguous():
        return "conv_weights.is_contiguous() (got a non-contiguous weight)"
    if tuple(conv_weights.shape) != (total, SUPPORTED_WIDTH):
        return (
            f"conv_weights.shape == (3 * dim, {SUPPORTED_WIDTH}) "
            f"(got {tuple(conv_weights.shape)})"
        )
    if conv_states_pool.dtype != torch.bfloat16:
        return f"conv_states_pool.dtype == bfloat16 (got {conv_states_pool.dtype})"
    if conv_states_pool.dim() != 3:
        return f"conv_states_pool.dim() == 3 (got {conv_states_pool.dim()})"
    if not conv_states_pool.is_contiguous():
        return "conv_states_pool.is_contiguous() (got a non-contiguous pool)"
    if conv_states_pool.shape[1] != SUPPORTED_WIDTH - 1:
        return (
            f"conv_states_pool.shape[1] == {SUPPORTED_WIDTH - 1} "
            f"(got {tuple(conv_states_pool.shape)})"
        )
    if conv_states_pool.shape[2] != total:
        return (
            f"conv_states_pool.shape[2] == 3 * dim "
            f"(got {tuple(conv_states_pool.shape)}, 3 * dim = {total})"
        )
    if torch.cuda.get_device_capability()[0] < 10:
        return (
            "SM100+ (got compute capability "
            f"{'.'.join(map(str, torch.cuda.get_device_capability()))})"
        )
    return None


# Warn at most once per process: a silent fallback is how the row-stride
# mismatch above stayed invisible for a whole integration cycle.
_warned_rejected: bool = False


def covered(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states_pool: torch.Tensor,
    bias: torch.Tensor | None,
    splits: Sequence[int],
    warn_if_rejected: bool = False,
) -> bool:
    """Whether this kernel can replace the three causal_conv1d_fn calls.

    See `reject_reason` for the predicates. Pass `warn_if_rejected=True` from a
    call site that has already established the opt-in flag is on, so a rejected
    input is reported once instead of silently falling back forever.
    """
    reason = reject_reason(mixed_qkv, conv_weights, conv_states_pool, bias, splits)
    if reason is None:
        return True
    if warn_if_rejected:
        global _warned_rejected
        if not _warned_rejected:
            _warned_rejected = True
            logger.warning(
                "SGLANG_K3_KDA_SHORT_CONV is enabled but the fused KDA short_conv "
                "kernel cannot be used: failed predicate %s. Falling back to the "
                "Triton causal_conv1d_fn path for the rest of this process.",
                reason,
            )
    return False


def kimi_k3_kda_short_conv(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states_pool: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    seq_lens_cpu: Sequence[int],
    pad_slot_id: int = PAD_SLOT_ID,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused width-4 depthwise causal conv1d + SiLU over q, k and v.

    Caller must check `covered(...)` first.

    `conv_states_pool` (the raw `[num_lines, width - 1, 3 * dim]` cache pool)
    is updated IN PLACE: after each non-padded sequence its slot holds that
    sequence's last 3 pre-activation input columns, shifted left. That update
    is part of the output; a decode step reading a stale pool is a silent
    correctness bug. `cache_indices[s] == pad_slot_id` sequences are skipped
    entirely (no output columns written, state untouched), matching the
    Triton path.

    Returns (out_q, out_k, out_v), each a `(dim, num_tokens)` bf16 tensor with
    compact channel-last strides `(1, dim)` -- i.e. `.transpose(0, 1)` yields
    a row-contiguous `(num_tokens, dim)` view, as the call site expects.
    """
    # Per-projection dim comes from the tensor, not a constant: it shrinks with
    # attn_tp_size (12288 / 6144 / 3072 / 1536 for attn_tp 1 / 2 / 4 / 8).
    dim = mixed_qkv.shape[1] // 3
    splits = [dim] * 3
    # Metadata-only views: `x` must stay the interleaved (dim, T) stride
    # (1, mixed_qkv.stride(0)) view of mixed_qkv -- never .contiguous(). The row
    # pitch is 3*dim for a contiguous mixed_qkv and 4*dim for the production
    # torch.split view of the fused q/k/v/g projection; the kernel reads it from
    # mixed_qkv.stride(0), so both work unchanged. torch.empty_like keeps the
    # stride ORDER of a non-dense view (infer_dense_strides), so `outs` is
    # compact (1, dim) either way -- asserted by test_output_layout.
    xs = mixed_qkv.transpose(0, 1).split(splits, dim=0)
    outs = [torch.empty_like(x) for x in xs]
    max_seqlen = int(max(seq_lens_cpu)) if len(seq_lens_cpu) else 0
    _jit_kda_short_conv_module().run(
        mixed_qkv,
        conv_weights,
        conv_states_pool,
        outs[0],
        outs[1],
        outs[2],
        query_start_loc,
        cache_indices,
        has_initial_state,
        max_seqlen,
        pad_slot_id,
    )
    return outs[0], outs[1], outs[2]
