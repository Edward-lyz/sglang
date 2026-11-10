"""Opt-in exact KDA snapshot inputs/results and one explicitly supported ABI.

Metadata contains input instructions only; state buffers are per-layer results.
No extension import, capability probe, device readback, or state write at import.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True, slots=True)
class KDANativeCheckpointResult:
    """Exact KDA snapshot inputs/results and one explicitly supported ABI."""
    output: torch.Tensor
    checkpoint_states: torch.Tensor


@dataclass(slots=True)
class KDANativePlanSink:
    """Temporary builder output; never retain layer state/output/workspace here."""

    offsets: Optional[torch.Tensor] = None
    owner_rows: Optional[torch.Tensor] = None


def _cpu_vector(value, name):
    if value.device.type != "cpu" or value.ndim != 1:
        raise ValueError(f"native checkpoint {name} must be an existing CPU vector")


def validate_native_checkpoint_cpu_inputs(
    *, prefixes, extends, targets, track_mask, track_indices, cache_indices
) -> None:
    """Validate exact targets before any forward can mutate pool/conv state."""
    values = {
        "prefixes": prefixes,
        "extends": extends,
        "targets": targets,
        "track_mask": track_mask,
        "track_indices": track_indices,
        "cache_indices": cache_indices,
    }
    for name, value in values.items():
        _cpu_vector(value, name)
        if value.numel() != extends.numel():
            raise ValueError("native checkpoint CPU metadata count mismatch")
        if name != "track_mask" and value.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"native checkpoint {name} must have integer dtype")
    if track_mask.dtype != torch.bool:
        raise ValueError("native checkpoint track_mask must be boolean")
    delta = (targets - prefixes)[track_mask]
    if torch.any((delta <= 0) | (delta > extends[track_mask])):
        raise ValueError(
            "tracked checkpoint must satisfy prefix < target <= extend end"
        )


def fill_native_plan_from_existing_cpu_data(
    sink: Optional[KDANativePlanSink],
    *,
    exact_checkpoint_tracking: bool,
    prefixes: torch.Tensor,
    extends: torch.Tensor,
    targets: torch.Tensor,
    track_mask: torch.Tensor,
    is_final_masked: torch.Tensor,
    h_dst: torch.Tensor,
    device,
) -> None:
    """Reuse validated CPU inputs in _init_track_ssm_indices, once per forward.

    Offsets are relative to each sequence's extend start; zero disables native
    snapshot output. Owner rows are sequence indices in original h_dst order,
    never packed Triton h indices. Only active owners may be read from output.
    """
    if sink is None:
        return
    sink.offsets = sink.owner_rows = None
    if not exact_checkpoint_tracking:
        raise ValueError("native checkpoint plans require exact KDA tracking")
    for name, value in (
        ("prefixes", prefixes),
        ("extends", extends),
        ("targets", targets),
        ("track_mask", track_mask),
        ("is_final_masked", is_final_masked),
        ("h_dst", h_dst),
    ):
        _cpu_vector(value, name)
    if not (
        prefixes.numel() == extends.numel() == targets.numel() == track_mask.numel()
    ):
        raise ValueError("native checkpoint CPU metadata count mismatch")
    if track_mask.dtype != torch.bool or is_final_masked.dtype != torch.bool:
        raise ValueError("native checkpoint owner masks must be boolean")
    tracked_rows = track_mask.nonzero(as_tuple=True)[0]
    if tracked_rows.numel() != is_final_masked.numel():
        raise ValueError("native checkpoint track owner/final mask mismatch")
    tracked_delta = (targets - prefixes)[tracked_rows]
    tracked_n = extends[tracked_rows]
    if torch.any((tracked_delta <= 0) | (tracked_delta > tracked_n)):
        raise ValueError(
            "tracked checkpoint must satisfy prefix < target <= extend end"
        )
    if not torch.equal(is_final_masked, tracked_delta == tracked_n):
        raise ValueError(
            "native checkpoint final flags differ from exact target positions"
        )
    owner_rows = tracked_rows[~is_final_masked]
    if owner_rows.numel() != h_dst.numel():
        raise ValueError("native checkpoint owner count differs from legacy h_dst")
    if owner_rows.numel() == 0:
        return  # Final-only calls retain the old FlashKDA/Triton return contract.
    delta = (targets - prefixes)[owner_rows]
    if torch.any((delta <= 0) | (delta >= extends[owner_rows]) | (delta % 16 != 0)):
        # Do not permit positive offsets in native's padded final tile.
        raise ValueError(
            "native interior offsets require 0 < delta < N and delta % 16 == 0"
        )
    dense_offsets = torch.zeros_like(extends, dtype=torch.int64)
    dense_offsets[owner_rows] = delta.to(torch.int64)
    sink.offsets = dense_offsets.to(device=device, non_blocking=True)
    sink.owner_rows = owner_rows.to(device=device, dtype=torch.int64, non_blocking=True)


# Schema of vllm-project/FlashKDA pin 17a037d98da546deb4591e967cf961a43c034d8b.
# This recognizes an ABI, not binary provenance or numerical correctness.
_FWD_SCHEMA = (
    "flash_kda::fwd(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, "
    "float scale, Tensor(a!) out, Tensor(c!) workspace, Tensor A_log, "
    "Tensor dt_bias, float lower_bound, Tensor? initial_state=None, "
    "Tensor(b!)? final_state=None, Tensor? cu_seqlens=None, "
    "Tensor(d!)? checkpoint_state=None, Tensor? checkpoint_offsets=None) -> ()"
)
_WORKSPACE_SCHEMA = "flash_kda::get_workspace_size(int T_total, int H, int N=1) -> int"


def _canonical_schema(schema):
    return "".join(str(schema).split())


@dataclass(frozen=True, slots=True)
class PinnedStyleNativeAdapter:
    """Avoid Python wrapper overhead by calling the native op directly."""
    fwd: object
    get_workspace_size: object
    schema: str

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
        workspace,
        A_log,
        dt_bias,
        lower_bound,
        initial_state,
        final_state,
        cu_seqlens,
        checkpoint_state,
        checkpoint_offsets,
    ):
        """ 
        Explicit known op, not speculative kwargs on a Python wrapper. Any
        launch error propagates: never retry after a possibly mutating call.
        """
        return self.fwd(
            q,
            k,
            v,
            g,
            beta,
            scale,
            out,
            workspace,
            A_log,
            dt_bias,
            lower_bound,
            initial_state,
            final_state,
            cu_seqlens,
            checkpoint_state,
            checkpoint_offsets,
        )


def probe_pinned_style_native(torch_module, *, enabled):
    """Fail closed in reflection, before any native call or state mutation."""
    if not enabled:
        return None, "disabled"
    try:
        namespace = torch_module.ops.flash_kda
        fwd = namespace.fwd.default
        workspace = namespace.get_workspace_size.default
        schema = fwd._schema
        # Full ABI including alias/mutation, optional defaults, order, return,
        # namespace and workspace contract. No version/signature inference.
        if _canonical_schema(schema) != _canonical_schema(_FWD_SCHEMA):
            return None, "unsupported_fwd_schema"
        if _canonical_schema(workspace._schema) != _canonical_schema(_WORKSPACE_SCHEMA):
            return None, "unsupported_workspace_schema"
        if not torch_module._C._dispatch_has_kernel_for_dispatch_key(
            "flash_kda::fwd", "CUDA"
        ):
            return None, "missing_cuda_registration"
    except (AttributeError, RuntimeError, TypeError):
        # Reflection only. Do not broaden this to wrap native invocation.
        return None, "native_schema_unavailable"
    return (
        PinnedStyleNativeAdapter(fwd, workspace, str(schema)),
        "supported_schema_only",
    )
