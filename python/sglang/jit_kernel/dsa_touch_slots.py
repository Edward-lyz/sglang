from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, run_jit_kernel

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_dsa_touch_slots_module() -> Module:
    return load_jit(
        "dsa_touch_slots",
        cuda_files=["nsa/dsa_touch_slots.cuh"],
        cuda_wrappers=[
            ("run", "DSATouchSlots::run"),
        ],
    )


def dsa_touch_slots(
    *,
    token_indices: torch.Tensor,
    dense_page_to_slot: torch.Tensor,
    page_last_access: torch.Tensor,
    page_size: int,
    cache_step: torch.Tensor,
) -> None:
    run_jit_kernel(
        _jit_dsa_touch_slots_module,
        token_indices,
        dense_page_to_slot,
        page_last_access,
        page_size,
        cache_step,
    )
