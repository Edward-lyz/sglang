"""Explicit CuTe prefix/suffix snapshots, independent of the external op ABI."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True, slots=True)
class CuTeCheckpointPlan:
    """Input metadata shared across layers; contains no recurrent state."""

    lengths: tuple[int, ...]
    offsets: tuple[int, ...]
    prefix_tokens: torch.Tensor
    suffix_tokens: torch.Tensor
    prefix_cu: torch.Tensor
    suffix_cu: torch.Tensor
    active_rows: torch.Tensor


def build_cute_checkpoint_plan(
    lengths_cpu: torch.Tensor, offsets_cpu: torch.Tensor, *, device
) -> Optional[CuTeCheckpointPlan]:
    """Build gather plans once from the exact, existing CPU checkpoint inputs.

    Unsupported C32 boundaries return no plan before any kernel is invoked.
    They retain the whole-batch fallback; they are never rounded to a boundary.
    Zero offsets mean no interior snapshot (including final-only tracked rows).
    """
    for value in (lengths_cpu, offsets_cpu):
        if value.device.type != "cpu" or value.ndim != 1:
            raise ValueError("CuTe checkpoint inputs must be existing CPU vectors")
        if value.dtype not in (torch.int32, torch.int64):
            raise ValueError("CuTe checkpoint inputs must have integer dtype")
    if lengths_cpu.shape != offsets_cpu.shape:
        raise ValueError("CuTe checkpoint input counts differ")
    lengths = tuple(lengths_cpu.tolist())
    offsets = tuple(offsets_cpu.tolist())
    if not lengths or any(
        n <= 0 or offset < 0 or offset >= n or offset % 32
        for n, offset in zip(lengths, offsets)
    ):
        return None
    active = [row for row, offset in enumerate(offsets) if offset]
    if not active:
        return None
    prefix_tokens, suffix_tokens = [], []
    prefix_cu, suffix_cu = [0], [0]
    start = 0
    for n, offset in zip(lengths, offsets):
        cut = offset or n
        prefix_tokens.append(torch.arange(start, start + cut, dtype=torch.int64))
        prefix_cu.append(prefix_cu[-1] + cut)
        if offset:
            suffix_tokens.append(
                torch.arange(start + cut, start + n, dtype=torch.int64)
            )
            suffix_cu.append(suffix_cu[-1] + n - cut)
        start += n
    return CuTeCheckpointPlan(
        lengths=lengths,
        offsets=offsets,
        prefix_tokens=torch.cat(prefix_tokens).to(device=device),
        suffix_tokens=torch.cat(suffix_tokens).to(device=device),
        prefix_cu=torch.tensor(prefix_cu, dtype=torch.int64, device=device),
        suffix_cu=torch.tensor(suffix_cu, dtype=torch.int64, device=device),
        active_rows=torch.tensor(active, dtype=torch.int64, device=device),
    )


@dataclass(frozen=True, slots=True)
class CuTeCheckpointAdapter:
    """Two real packed CuTe forwards, with FP32 state retained on the GPU."""

    fwd: object

    def invoke(
        self,
        *,
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
        initial_state,
        final_state,
        checkpoint_state,
        plan: CuTeCheckpointPlan,
    ):
        """Run the two packed CuTe forwards, writing outputs and FP32 states in place."""
        # All validation precedes output writes or calls. No launch-error retry.
        if not isinstance(plan, CuTeCheckpointPlan):
            raise ValueError("CuTe checkpoint invocation requires an exact CPU plan")
        n = len(plan.lengths)
        if (
            q.ndim != 4
            or q.shape[0] != 1
            or q.shape[1] != sum(plan.lengths)
            or q.shape[-1] != 128
            or any(t.shape != q.shape for t in (k, v, g, out))
            or beta.shape != q.shape[:3]
            or any(t.dtype != torch.bfloat16 for t in (q, k, v, g, beta, out))
        ):
            raise ValueError("CuTe checkpoint packed input geometry/dtype mismatch")
        state_shape = (n, q.shape[2], 128, 128)
        for state in (initial_state, final_state, checkpoint_state):
            if state.shape != state_shape or state.dtype != torch.float32:
                raise ValueError("CuTe checkpoint states must be dense FP32")
        if any(
            tensor.device != q.device
            for tensor in (
                k,
                v,
                g,
                beta,
                out,
                A_log,
                dt_bias,
                initial_state,
                final_state,
                checkpoint_state,
                plan.prefix_tokens,
                plan.suffix_tokens,
                plan.prefix_cu,
                plan.suffix_cu,
                plan.active_rows,
            )
        ):
            raise ValueError("CuTe checkpoint tensors must share a device")

        def compute(indices, cu, initial, final):
            inputs = [
                tensor.index_select(1, indices).contiguous()
                for tensor in (q, k, v, g, beta)
            ]
            output = torch.empty_like(inputs[2])
            self.fwd(
                *inputs,
                scale,
                output,
                A_log,
                dt_bias,
                lower_bound,
                initial_state=initial,
                final_state=final,
                cu_seqlens=cu,
            )
            out.index_copy_(1, indices, output)

        compute(
            plan.prefix_tokens,
            plan.prefix_cu,
            initial_state,
            checkpoint_state,
        )
        # Inactive rows are complete after the first call. Only active rows
        # have suffixes; their prefix results are the actual checkpoint states.
        final_state.copy_(checkpoint_state)
        suffix_initial = checkpoint_state.index_select(0, plan.active_rows)
        suffix_final = torch.empty_like(suffix_initial)
        compute(
            plan.suffix_tokens,
            plan.suffix_cu,
            suffix_initial,
            suffix_final,
        )
        final_state.index_copy_(0, plan.active_rows, suffix_final)
