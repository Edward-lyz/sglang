"""Fused KV-latent AllGather-input pack (GLM 5.2 NSA prefill context-parallel).

Registers the ``kv_latent_pack`` CUDA kernel as an in-tree JIT module and
exposes :func:`pack_kv_latent_for_allgather`, a drop-in for the pack step of
``sglang.srt.models.deepseek_v2.DeepseekV2AttentionMLA.rebuild_cp_kv_cache``:
it collapses 2 ``elementwise_kernel`` slice-assigns (into a strided view) +
1 ``.contiguous()`` copy into one fused CUDA launch that writes directly into
a fresh contiguous AllGather-input buffer.

    out[i, :dim_nope] = k_nope[i]
    out[i, dim_nope:] = k_pe[i]

Three kernel variants are dispatched by shape (see
project/op/glm52_kv_latent_pack/readme.md for the ncu-driven derivation):
  v1 (N<=1024), v2 (1024<N<=4096), v3 persistent (N>4096).
"""
from __future__ import annotations

from typing import Optional

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

_DEFAULT_BLK = 128
_DEFAULT_ROWS_PER_BLOCK = 4
_V1_MAX_N = 1024
_V2_MAX_N = 4096


@cache_once
def _jit_module_v1(dtype: torch.dtype, vec_size: int, block_threads: int):
    args = make_cpp_args(dtype, vec_size, block_threads)
    return load_jit(
        f"kv_latent_pack_v1_{vec_size}_{block_threads}",
        *args,
        cuda_files=["elementwise/kv_latent_pack.cuh"],
        cuda_wrappers=[("run", f"PackKvLatentKernel<{args}>::run")],
    )


@cache_once
def _jit_module_v2(dtype: torch.dtype, vec_size: int, rows_per_block: int):
    args = make_cpp_args(dtype, vec_size, rows_per_block)
    return load_jit(
        f"kv_latent_pack_v2_{vec_size}_{rows_per_block}",
        *args,
        cuda_files=["elementwise/kv_latent_pack.cuh"],
        cuda_wrappers=[("run", f"PackKvLatentKernelV2<{args}>::run")],
    )


@cache_once
def _jit_module_v3(dtype: torch.dtype, vec_size: int, rows_per_block: int):
    args = make_cpp_args(dtype, vec_size, rows_per_block)
    return load_jit(
        f"kv_latent_pack_v3_{vec_size}_{rows_per_block}",
        *args,
        cuda_files=["elementwise/kv_latent_pack.cuh"],
        cuda_wrappers=[("run", f"PackKvLatentKernelV3Persist<{args}>::run")],
    )


def _best_vec_size(dim_nope: int, dim_rope: int) -> int:
    for v in (8, 4, 2, 1):
        if dim_nope % v == 0 and dim_rope % v == 0:
            return v
    return 1  # unreachable, v=1 always divides


def pack_kv_latent_for_allgather(
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    version: str = "auto",
    block_threads: int = _DEFAULT_BLK,
    rows_per_block: int = _DEFAULT_ROWS_PER_BLOCK,
) -> torch.Tensor:
    """Drop-in fused replacement for the pack step of
    ``DeepseekV2AttentionMLA.rebuild_cp_kv_cache``:

        latent_cache[..., :kv_lora_rank] = k_nope.squeeze(1)
        latent_cache[..., kv_lora_rank:] = k_pe.squeeze(1)
        latent_cache = latent_cache.contiguous()

    Writes k_nope / k_pe directly into a fresh contiguous
    ``[N, dim_nope + dim_rope]`` tensor in a single kernel launch, ready to
    feed straight into ``cp_all_gather_rerange_output``. k_nope / k_pe may
    carry an optional singleton middle dim (as produced by ``.unsqueeze(1)``
    upstream) and may themselves be strided views; only the innermost dim is
    required to be contiguous (stride 1).

    Args:
        k_nope: ``[N, dim_nope]`` or ``[N, 1, dim_nope]``.
        k_pe: ``[N, dim_rope]`` or ``[N, 1, dim_rope]``, same dtype/device as
            k_nope.
        out: optional preallocated contiguous ``[N, dim_nope + dim_rope]``
            output.

    Returns:
        ``[N, dim_nope + dim_rope]`` contiguous tensor, dtype/device of
        k_nope.
    """
    if k_nope.dim() == 3:
        assert k_nope.shape[1] == 1, "expected a squeezable singleton dim"
        k_nope = k_nope.squeeze(1)
    if k_pe.dim() == 3:
        assert k_pe.shape[1] == 1, "expected a squeezable singleton dim"
        k_pe = k_pe.squeeze(1)

    assert k_nope.dim() == 2 and k_pe.dim() == 2, "k_nope/k_pe must be 2D after squeeze"
    assert k_nope.dtype == k_pe.dtype, "k_nope/k_pe must share dtype"
    assert k_nope.device == k_pe.device, "k_nope/k_pe must share device"
    assert k_nope.shape[0] == k_pe.shape[0], "k_nope/k_pe must share row count"
    assert k_nope.stride(-1) == 1 and k_pe.stride(-1) == 1, (
        "k_nope/k_pe must be contiguous along the last dim"
    )

    n, dim_nope = k_nope.shape
    dim_rope = k_pe.shape[1]

    if out is None:
        out = torch.empty(
            n, dim_nope + dim_rope, dtype=k_nope.dtype, device=k_nope.device
        )
    else:
        assert out.shape == (n, dim_nope + dim_rope)
        assert out.is_contiguous()

    if n == 0:
        return out

    vec = _best_vec_size(dim_nope, dim_rope)
    if version == "auto":
        version = "v1" if n <= _V1_MAX_N else ("v2" if n <= _V2_MAX_N else "v3")
    if version == "v1":
        module = _jit_module_v1(k_nope.dtype, vec, block_threads)
    elif version == "v2":
        module = _jit_module_v2(k_nope.dtype, vec, rows_per_block)
    elif version == "v3":
        module = _jit_module_v3(k_nope.dtype, vec, rows_per_block)
    else:
        raise ValueError(f"unknown version {version!r}, expected 'auto', 'v1', 'v2', or 'v3'")
    module.run(k_nope, k_pe, out)
    return out
