# Copyright 2026 KDA(Kernel Design Agents) Team
# SPDX-License-Identifier: Apache-2.0
"""FlashKDA-compatible interface backed by the direct CuTe DSL KDA forward.

Vendored drop-in for the external ``flash_kda`` package. Enable it in SGLang via
``SGLANG_OPT_FLASH_KDA_CUTE=1``; ``kda_flashkda._load_flash_kda`` then returns
this module instead of importing the MoonshotAI CUTLASS ``flash_kda``.

Like ``flash_kda.fwd`` this writes ``out`` (and ``final_state`` when given) in
place and returns ``None``. Tensor contracts match ``flash_kda.fwd``:

- q/k/v/g : bf16 ``[B, T, H, 128]``
- beta    : bf16 ``[B, T, H]`` (pre-sigmoid)
- out     : bf16 ``[B, T, H, 128]`` output buffer (written in place)
- A_log   : fp32 ``[H]``
- dt_bias : fp32 ``[H, 128]`` (flat ``[H*128]`` also accepted)
- initial_state / final_state : V-first recurrent state ``[N, H, 128, 128]``
  (``None`` initial_state starts from zero; the CuTe kernel runs its state in
  fp32, so a non-fp32 state buffer is bridged through an fp32 scratch)
- cu_seqlens : int64 ``[N+1]`` (B must be 1 when provided)

Differences handled here vs the underlying CuTe ``impl.forward.fwd``:
dt_bias reshaping, fp32-state bridging, zero initial_state, and FlashKDA's
batched mode (``cu_seqlens is None`` with ``B > 1``), which is folded into a
packed single row with synthetic ``cu_seqlens`` because the CuTe kernel is
packed/B=1.
"""

from __future__ import annotations

import torch

from .impl.forward import fwd as _cute_fwd
from .impl.forward import precompile

D = 128


def get_workspace_size(T_total, H, N):  # noqa: ARG001 - signature parity
    """FlashKDA API parity. The CuTe kernel owns its scratch (host planner +
    TMEM/SMEM), so no caller-provided workspace is needed."""
    return 0


def fwd(
    q,
    k,
    v,
    g,
    beta,
    scale,
    out,
    A_log,
    dt_bias,
    lower_bound,
    initial_state=None,
    final_state=None,
    cu_seqlens=None,
):
    """FlashKDA-compatible KDA forward on the CuTe DSL kernel (writes in place)."""
    if q.dtype != torch.bfloat16:
        raise TypeError(f"FlashKDA requires bf16 q/k/v/g, got {q.dtype}")
    if q.shape[-1] != D or v.shape[-1] != D:
        raise ValueError(
            f"FlashKDA/CuTe require K=V={D}, got K={q.shape[-1]} V={v.shape[-1]}"
        )

    B, T, H = q.shape[0], q.shape[1], q.shape[2]
    dt_bias = dt_bias.reshape(H, D)

    # FlashKDA batched mode: each of B rows is an independent length-T sequence.
    # The CuTe kernel is packed/B=1, so fold B into one packed row.
    out_view = out
    if cu_seqlens is None and B > 1:
        cu_seqlens = torch.arange(B + 1, device=q.device, dtype=torch.int64) * T
        q = q.reshape(1, B * T, H, D)
        k = k.reshape(1, B * T, H, D)
        v = v.reshape(1, B * T, H, D)
        g = g.reshape(1, B * T, H, D)
        beta = beta.reshape(1, B * T, H)
        out_view = out.reshape(1, B * T, H, D)  # shares storage -> in-place

    nseq = 1 if cu_seqlens is None else int(cu_seqlens.numel()) - 1

    # CuTe runs the state in fp32 V-first [N, H, 128, 128].
    if initial_state is None:
        init_fp32 = torch.zeros(nseq, H, D, D, device=q.device, dtype=torch.float32)
    else:
        init_fp32 = initial_state.to(torch.float32).contiguous()

    if final_state is None:
        final_fp32 = None
    elif final_state.dtype == torch.float32 and final_state.is_contiguous():
        final_fp32 = final_state
    else:
        final_fp32 = torch.empty(nseq, H, D, D, device=q.device, dtype=torch.float32)

    _cute_fwd(
        q,
        k,
        v,
        g,
        beta,
        float(scale),
        out_view,
        A_log,
        dt_bias,
        float(lower_bound),
        initial_state=init_fp32,
        final_state=final_fp32,
        cu_seqlens=cu_seqlens,
    )

    if final_state is not None and final_fp32 is not final_state:
        final_state.copy_(final_fp32)
    return None


__all__ = ["fwd", "get_workspace_size", "precompile", "D"]
