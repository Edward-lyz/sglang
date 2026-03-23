from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, run_jit_kernel

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_dsa_apply_load_plan_module() -> Module:
    return load_jit(
        "dsa_apply_load_plan",
        cuda_files=["nsa/dsa_apply_load_plan.cuh"],
        cuda_wrappers=[
            ("run", "DSAApplyLoadPlan::run"),
        ],
    )


def dsa_apply_load_plan(
    *,
    active_pages: torch.Tensor,
    req_indices: torch.Tensor,
    logical_pages: torch.Tensor,
    assigned_slot_indices: torch.Tensor,
    assigned_page_starts: torch.Tensor,
    reserved_page_starts: torch.Tensor,
    page_owner_req: torch.Tensor,
    page_owner_logical: torch.Tensor,
    page_last_access: torch.Tensor,
    gpu_slot: torch.Tensor,
    cache_step: torch.Tensor,
) -> None:
    run_jit_kernel(
        _jit_dsa_apply_load_plan_module,
        active_pages,
        req_indices,
        logical_pages,
        assigned_slot_indices,
        assigned_page_starts,
        reserved_page_starts,
        page_owner_req,
        page_owner_logical,
        page_last_access,
        gpu_slot,
        cache_step,
    )
