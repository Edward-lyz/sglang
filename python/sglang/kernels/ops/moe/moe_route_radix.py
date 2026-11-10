"""Native-CUDA radix-select router for K3 routing (all batch sizes).

Keys and activations stay in registers (224 threads, 4 experts each), the
split-bin search runs on warp scans instead of cub, rounds exit early when the
top-k separates on a byte boundary, and the (biased desc, id asc) output sort
is optional. Consumers that only gather by expert id can pass sorted=False and
skip the epilogue rank-sort entirely.

Dispatched automatically from moe_fused_gate for covered inputs; the production
dispatch uses sorted=False. It is 3.1-3.5x faster than the Triton router at
[1..8192, 896] top-16 on B200. Correctness and benchmark coverage against the
Triton baseline lives in test/registered/jit/test_moe_route_radix.py and
test/registered/jit/benchmark/bench_moe_route_radix.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_NUM_EXPERTS = 896
_TOPK = 16

# int64 id sibling for consumers that widen the ids first thing (DeepEP
# dispatch). When requested, route_radix also stores the winners as int64 from
# its epilogue and records the (int32 ids, int64 ids) pair; int64_ids_for()
# hands the copy back only for that very int32 tensor, so any post-processing
# that replaces the ids falls back to the caller's own cast. One slot.
_EMIT_INT64_IDS = False
_INT64_IDS_SLOT = None


def request_int64_ids(enabled: bool = True) -> None:
    """Make route_radix also store its top-k ids as int64."""
    global _EMIT_INT64_IDS
    _EMIT_INT64_IDS = bool(enabled)


def int64_ids_for(ids: torch.Tensor):
    """The int64 copy route_radix emitted for exactly this int32 ids tensor,
    or None if there is none."""
    slot = _INT64_IDS_SLOT
    if slot is not None and slot[0] is ids:
        return slot[1]
    return None


@cache_once
def _jit_route_radix_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "moe_route_radix",
        *args,
        cuda_files=["moe/route_radix.cuh"],
        cuda_wrappers=[("run", f"RouteRadixKernel<{args}>::run")],
        # No fast-math: expert-id selection must stay bit-identical to the
        # Triton router under ties/NaN.
        extra_cuda_cflags=["-O3"],
    )


def covered(scores: torch.Tensor, bias: torch.Tensor, topk: int) -> bool:
    """Specialized for K3 decode routing: [M, 896] bf16 or fp32
    row-contiguous scores (8B/16B-aligned rows), fp32 bias, top-16."""
    return (
        scores.dim() == 2
        and scores.size(1) == _NUM_EXPERTS
        and int(topk) == _TOPK
        and scores.dtype in (torch.bfloat16, torch.float32)
        and bias.dtype == torch.float32
        and scores.stride(1) == 1
        and scores.stride(0) % 4 == 0
        and bias.is_contiguous()
    )


def route_radix(
    scores: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    apply_scale: bool,
    sorted: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (weights [M, topk] fp32, ids [M, topk] int32). Caller must have
    checked covered().

    Default sorted=False: winners come out in expert-id-ascending order
    (downstream MoE kernels are order-insensitive) and the epilogue rank-sort
    is skipped. sorted=True restores the Triton router's (biased desc, id asc)
    output order. Either way the winner set matches Triton exactly; the renorm
    sum is taken in the respective output order, so weights may differ by
    <= ~1 ulp."""
    global _INT64_IDS_SLOT
    M = scores.shape[0]
    out_w = torch.empty((M, topk), dtype=torch.float32, device=scores.device)
    out_i = torch.empty((M, topk), dtype=torch.int32, device=scores.device)
    out_i64 = (
        torch.empty((M, topk), dtype=torch.int64, device=scores.device)
        if _EMIT_INT64_IDS
        else None
    )
    _jit_route_radix_module().run(
        scores,
        bias,
        out_w,
        out_i,
        out_i64,
        topk,
        float(routed_scaling_factor),
        bool(renormalize),
        bool(apply_scale),
        bool(sorted),
    )
    if out_i64 is not None:
        _INT64_IDS_SLOT = (out_i, out_i64)
    return out_w, out_i
