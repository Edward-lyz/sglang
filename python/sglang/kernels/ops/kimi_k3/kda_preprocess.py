"""CUDA-JIT kernels for Kimi-K3 KDA extend preprocessing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_INDEX_DTYPES = (torch.int32, torch.int64)


@cache_once
def _jit_beta_sigmoid_module() -> Module:
    return load_jit(
        "kimi_k3_kda_beta_sigmoid",
        cuda_files=["kimi_k3/kda_beta_sigmoid.cuh"],
        cuda_wrappers=[("run", "KdaBetaSigmoidKernel::run")],
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def _jit_track_conv_state_module(
    window_size: int,
    track_dtype: torch.dtype,
    destination_dtype: torch.dtype,
) -> Module:
    args = make_cpp_args(window_size, track_dtype, destination_dtype)
    return load_jit(
        "kimi_k3_kda_track_conv_state",
        *args,
        cuda_files=["kimi_k3/kda_track_conv_state.cuh"],
        cuda_wrappers=[("run", f"KdaTrackConvStateKernel<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def _jit_track_conv_state_flag_module(
    window_size: int,
    track_dtype: torch.dtype,
    destination_dtype: torch.dtype,
    prefix_dtype: torch.dtype,
) -> Module:
    args = make_cpp_args(window_size, track_dtype, destination_dtype, prefix_dtype)
    return load_jit(
        "kimi_k3_kda_track_conv_state_flag",
        *args,
        cuda_files=["kimi_k3/kda_track_conv_state_flag.cuh"],
        cuda_wrappers=[("run", f"KdaTrackConvStateFlagKernel<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def can_fuse_beta_sigmoid(beta: torch.Tensor) -> bool:
    """Check if a strided BF16 beta slice can be converted to contiguous FP32"""
    return (
        beta.is_cuda
        and beta.dtype == torch.bfloat16
        and beta.ndim == 2
        and beta.shape[1] > 0
        and beta.shape[1] % 4 == 0
        and beta.stride(1) == 1
        and beta.stride(0) % 4 == 0
        and beta.data_ptr() % 8 == 0
    )


def fused_beta_sigmoid(beta: torch.Tensor, output: torch.Tensor) -> None:
    """Convert a strided BF16 beta slice to contiguous FP32 sigmoid."""
    if not can_fuse_beta_sigmoid(beta):
        raise ValueError("beta does not satisfy the CUDA fused-kernel contract")
    if (
        output.shape != beta.shape
        or output.dtype != torch.float32
        or output.device != beta.device
        or not output.is_contiguous()
    ):
        raise ValueError("output must be contiguous FP32 with beta's shape/device")
    if beta.numel() == 0:
        return
    _jit_beta_sigmoid_module().run(beta, output)


def can_fuse_track_conv_state(
    mixed_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    track_indices: torch.Tensor,
    destination_indices: torch.Tensor,
) -> bool:
    """Check if a strided BF16 mixed_qkv slice can be converted to contiguous"""
    if (
        not mixed_qkv.is_cuda
        or not conv_state.is_cuda
        or not track_indices.is_cuda
        or not destination_indices.is_cuda
        or mixed_qkv.dtype != torch.bfloat16
        or conv_state.dtype != torch.bfloat16
        or track_indices.dtype not in _INDEX_DTYPES
        or destination_indices.dtype not in _INDEX_DTYPES
        or mixed_qkv.ndim != 2
        or conv_state.ndim != 3
        or track_indices.ndim != 2
        or destination_indices.ndim != 1
    ):
        return False

    num_tracks, window_size = track_indices.shape
    state_dim = mixed_qkv.shape[1]
    return (
        destination_indices.shape[0] == num_tracks
        and conv_state.shape[1] == window_size
        and conv_state.shape[2] == state_dim
        and state_dim % 8 == 0
        and mixed_qkv.stride(1) == 1
        and mixed_qkv.stride(0) % 8 == 0
        and conv_state.stride(2) == 1
        and conv_state.stride(0) % 8 == 0
        and conv_state.stride(1) % 8 == 0
        and mixed_qkv.data_ptr() % 16 == 0
        and conv_state.data_ptr() % 16 == 0
    )


def fused_track_conv_state(
    mixed_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    track_indices: torch.Tensor,
    destination_indices: torch.Tensor,
) -> None:
    """Gather tracked rows directly into persistent convolution states."""
    if not can_fuse_track_conv_state(
        mixed_qkv, conv_state, track_indices, destination_indices
    ):
        raise ValueError("track tensors do not satisfy the CUDA fused-kernel contract")
    if destination_indices.numel() == 0:
        return

    _jit_track_conv_state_module(
        track_indices.shape[1],
        track_indices.dtype,
        destination_indices.dtype,
    ).run(mixed_qkv, conv_state, track_indices, destination_indices)


def can_fuse_track_conv_state_flag(
    mixed_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    track_indices: torch.Tensor,
    destination_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> bool:
    """Check if a strided BF16 mixed_qkv slice can be converted to contiguous"""
    if not can_fuse_track_conv_state(
        mixed_qkv, conv_state, track_indices, destination_indices
    ):
        return False
    return (
        prefix_lens.is_cuda
        and has_initial_state.is_cuda
        and prefix_lens.dtype in _INDEX_DTYPES
        and has_initial_state.dtype == torch.bool
        and prefix_lens.ndim == 1
        and has_initial_state.ndim == 1
        and prefix_lens.shape == has_initial_state.shape
        and has_initial_state.stride(0) == 1
    )


def fused_track_conv_state_flag(
    mixed_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    track_indices: torch.Tensor,
    destination_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> None:
    """Track conv states and emit ``has_initial_state`` in a single launch.

    Fuses the extend-path trio ``extend_prefix_lens > 0`` + advanced-index gather
    + ``index_put`` into one kernel: the copy is bit-exact with the indexing
    version and the flag is written by block (0, 0), which is otherwise idle.
    """
    if not can_fuse_track_conv_state_flag(
        mixed_qkv,
        conv_state,
        track_indices,
        destination_indices,
        prefix_lens,
        has_initial_state,
    ):
        raise ValueError("tensors do not satisfy the CUDA fused-kernel contract")
    if destination_indices.numel() == 0 and prefix_lens.numel() == 0:
        return

    _jit_track_conv_state_flag_module(
        track_indices.shape[1],
        track_indices.dtype,
        destination_indices.dtype,
        prefix_lens.dtype,
    ).run(
        mixed_qkv,
        conv_state,
        track_indices,
        destination_indices,
        prefix_lens,
        # tvm-ffi keeps torch.bool as `bool`; hand the kernel the byte view of
        # the same storage, which it fills with 0/1.
        has_initial_state.view(torch.uint8),
    )
