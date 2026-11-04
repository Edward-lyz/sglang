"""CUDA JIT KDA strided gated RMSNorm: fuses the non-contiguous layout copy
and the gated RMSNorm that follow KDA's extend/prefill attention output.

Replaces, on the covered fast path:
    core_attn_out = self.o_norm(core_attn_out, norm_gate)   # Triton gated RMSNorm
    core_attn_out = core_attn_out.squeeze(0).flatten(-2)     # forces an ATen copy
with a single kernel call that reads core_attn_out through its real strides
and writes a fresh contiguous [T, H*D] tensor -- the caller must NOT flatten
its output again.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_ACT_SWISH: int = 0
_ACT_SIGMOID: int = 1
_ACT_ID = {"swish": _ACT_SWISH, "silu": _ACT_SWISH, "sigmoid": _ACT_SIGMOID}

_SUPPORTED_D = (64, 128)


@cache_once
def _jit_kda_strided_onorm_module(dtype: torch.dtype, head_dim: int) -> Module:
    args = make_cpp_args(dtype, head_dim, is_arch_support_pdl())
    return load_jit(
        "kda_strided_onorm_" + str(head_dim),
        *args,
        cuda_files=["kimi_k3/kda_strided_onorm.cuh"],
        cuda_wrappers=[("run", f"KdaStridedOnormKernel<{args}>::run")],
        # --use_fast_math lowers sqrtf/expf (RMS norm + sigmoid/swish gate) to
        # single-instruction MUFU.RSQ/MUFU.EX2 hardware approximations instead
        # of nvcc's IEEE-safe software fallback (range-checked, CALL.REL.NOINC
        # to an out-of-line correction routine). Confirmed via SASS diff and
        # NCU against the reference build (which passes this same flag) that
        # omitting it turns this kernel compute-bound and costs ~26% wall
        # time; see record_kernel_opt/kimik3-kda-o-zulu/profiles/p02-gap-diagnosis/final_report.md.
        # The precision trade-off is already covered by this kernel's bf16
        # atol=rtol=8e-3 tolerance policy.
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )


def covered(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    activation: str,
    eps: float,
) -> bool:
    """Attempt-and-verify predicate: True iff `kda_strided_onorm` can serve
    this call. Callers must fall back to the reference `rms_norm_gated` /
    `FusedRMSNormGated` path when this returns False.

    Limitations (mirrors the kernel's two dispatch paths):
      - x, g must be [1, T, H, D], D in {64, 128}.
      - x.dtype == g.dtype == weight.dtype, in {bfloat16, float16}.
      - g's head_dim axis must be contiguous (stride 1); its token/head axes
        may have any stride -- both dispatch paths read g through runtime
        (token, head) strides rather than assuming it is row-major. This
        matters because g (norm_gate) is often a column slice of a wider
        fused projection tensor (e.g. TP-packed g_proj), so its token-axis
        stride is a multiple of H*D rather than exactly H*D.
      - x must be EITHER (a) token-axis stride 1, i.e. x.stride(1) == 1
        (the KDA extend case: x aliases the transposed conv-output buffer),
        with the head_dim and head axis strides divisible by 8 and T
        divisible by 8 (the SMEM path's vectorization requirement), OR
        (b) fully contiguous [1, T, H, D] (the flat path; only x needs to be
        contiguous here, not g).
      - weight is [D], contiguous, same dtype as x.
      - bias is always None / is_rms_norm always True in this kernel: no
        residual, no prenorm, no residual_in_fp32 -- unsupported cases must
        use the reference path.
      - activation in {"swish", "silu", "sigmoid"}.
    """
    if activation not in _ACT_ID:
        return False
    if x.dim() != 4 or g.dim() != 4 or weight.dim() != 1:
        return False
    if x.shape[0] != 1 or x.shape != g.shape:
        return False
    D = x.shape[3]
    if D not in _SUPPORTED_D:
        return False
    if weight.shape[0] != D:
        return False
    if x.dtype not in (torch.bfloat16, torch.float16):
        return False
    if x.dtype != g.dtype or x.dtype != weight.dtype:
        return False
    if not weight.is_contiguous():
        return False
    if g.stride(3) != 1:
        return False

    T, H = x.shape[1], x.shape[2]
    if T == 0 or H == 0:
        return False

    sx_t, sx_h, sx_d = x.stride(1), x.stride(2), x.stride(3)
    smem_ok = (
        sx_t == 1
        and sx_d % 8 == 0
        and sx_h % 8 == 0
        and T % 8 == 0
    )
    # Only x needs to be contiguous here -- g is read through runtime
    # (token, head) strides by both dispatch paths, so a non-row-major g
    # (e.g. a TP column slice of a wider fused g_proj tensor) does not force
    # a fallback to the reference path.
    flat_ok = x.is_contiguous()
    return smem_ok or flat_ok


def kda_strided_onorm(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    activation: str,
    eps: float,
) -> torch.Tensor:
    """out = fused gated RMSNorm over x's real strides, returning a fresh
    contiguous [T, H*D] tensor of x's dtype. Caller checks covered() first;
    this function does not fall back or validate beyond what the C++
    launcher's TensorMatcher/RuntimeCheck enforce."""
    T, H, D = x.shape[1], x.shape[2], x.shape[3]
    out = torch.empty((T, H * D), dtype=x.dtype, device=x.device)
    module = _jit_kda_strided_onorm_module(x.dtype, D)
    module.run(x, g, weight, out, _ACT_ID[activation], float(eps))
    return out
