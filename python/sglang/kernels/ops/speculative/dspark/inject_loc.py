"""Single-launch commit gate + full->SWA slot translation for DSpark target-hidden injection.

``DSparkKVInjector._inject_mla`` builds the draft SWA slot of every verify token with a
chain of tiny eager torch ops:

    swa_loc = mapping[cache_loc].to(int32)                       # gather + cast
    committed = (arange(verify_len) < commit_lens[:, None]).reshape(-1)   # arange, cast, compare
    swa_loc = where(committed, swa_loc, -1)                      # full_like + where
    swa_loc = where(row_valid, swa_loc, -1)                      # optional

~7 kernels of 1-3 us each (plus a launch gap apiece) per step, outside any CUDA graph.
This kernel does all of it in one launch; integer-only, so the result is identical.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _commit_swa_loc_kernel(
    mapping_ptr,
    cache_loc_ptr,
    commit_lens_ptr,
    row_valid_ptr,
    out_ptr,
    n,
    verify_len,
    HAS_COMMIT: tl.constexpr,
    HAS_ROW_VALID: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_range = offs < n
    loc = tl.load(cache_loc_ptr + offs, mask=in_range, other=0).to(tl.int64)
    swa = tl.load(mapping_ptr + loc, mask=in_range, other=0).to(tl.int32)
    if HAS_COMMIT:
        row = offs // verify_len
        col = offs - row * verify_len
        committed = col < tl.load(commit_lens_ptr + row, mask=in_range, other=0).to(
            tl.int64
        )
        swa = tl.where(committed, swa, -1)
    if HAS_ROW_VALID:
        valid = tl.load(row_valid_ptr + offs, mask=in_range, other=0) != 0
        swa = tl.where(valid, swa, -1)
    tl.store(out_ptr + offs, swa, mask=in_range)


def can_commit_swa_loc(
    mapping: Optional[torch.Tensor],
    cache_loc: torch.Tensor,
    commit_lens: Optional[torch.Tensor],
    verify_len: Optional[int],
    row_valid: Optional[torch.Tensor],
) -> bool:
    """Fusion commit_swa_loc into one kernel launch."""
    if mapping is None or not (mapping.is_cuda and cache_loc.is_cuda):
        return False
    if mapping.dim() != 1 or cache_loc.dim() != 1 or cache_loc.numel() == 0:
        return False
    if not (cache_loc.is_contiguous() and mapping.is_contiguous()):
        return False
    if cache_loc.dtype not in (torch.int32, torch.int64):
        return False
    if (commit_lens is None) != (verify_len is None):
        return False
    if commit_lens is not None and (
        not commit_lens.is_cuda
        or not commit_lens.is_contiguous()
        or verify_len <= 0
        or cache_loc.numel() != commit_lens.numel() * verify_len
    ):
        return False
    if row_valid is not None and (
        not row_valid.is_cuda
        or not row_valid.is_contiguous()
        or row_valid.numel() != cache_loc.numel()
    ):
        return False
    return True


def commit_swa_loc(
    mapping: torch.Tensor,
    cache_loc: torch.Tensor,
    commit_lens: Optional[torch.Tensor],
    verify_len: Optional[int],
    row_valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``where(row_valid, where(col < commit_lens, mapping[cache_loc], -1), -1)`` as int32."""
    assert can_commit_swa_loc(mapping, cache_loc, commit_lens, verify_len, row_valid)
    n = cache_loc.numel()
    out = torch.empty(n, dtype=torch.int32, device=cache_loc.device)
    block = 256
    _commit_swa_loc_kernel[(triton.cdiv(n, block),)](
        mapping,
        cache_loc,
        commit_lens if commit_lens is not None else cache_loc,
        row_valid if row_valid is not None else cache_loc,
        out,
        n,
        verify_len if verify_len is not None else 1,
        HAS_COMMIT=commit_lens is not None,
        HAS_ROW_VALID=row_valid is not None,
        BLOCK=block,
        num_warps=4,
    )
    return out
