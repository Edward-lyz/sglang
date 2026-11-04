# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch
import triton
import triton.language as tl

# https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/modelopt/torch/quantization/qtensor/mxfp4_tensor.py
class MXFP4QuantizeUtil:
    E2M1_max = 6.0

    E2M1_values = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    E2M1_bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])

    @classmethod
    def quantize(cls, input: torch.Tensor, block_size: Optional[int]) -> tuple:
        """Converting a tensor to a quantized format based on MXFP4 quantization. Only E4M3 is supported.
        Args:
            input (torch.Tensor): The input tensor to be quantized.
            block_sizes (dict | None): The block sizes for quantization.
        """

        def cast_fp4(x):
            sign = torch.sign(x)
            sign_bit = (2 - sign) // 2
            ord_ = torch.sum(
                (x.abs().unsqueeze(-1) - cls.E2M1_bounds.to(x.device)) > 0, dim=-1
            )
            fp4_val = (sign_bit * 0b1000 + ord_).to(torch.uint8)
            return fp4_val

        def fuse_uint4_to_uint8(x):
            # If the last dimension is odd, pad with zeros
            # If this behavior is not desired, please modify the code accordingly
            left_side = x[..., 0::2]  # Even indices (0, 2, 4...)
            right_side = x[..., 1::2]  # Odd indices (1, 3, 5...)
            new_data = (
                right_side.clone() << 4
            )  # Put odd indices (higher addresses) in high bits
            new_data[
                ..., : left_side.shape[-1]
            ] += left_side  # Put even indices in low bits
            return new_data

        if block_size is None:
            block_size = 32

        original_shape = input.shape
        original_dtype = input.dtype
        input = input.view(-1, block_size)
        # get scales
        input_amax = input.abs().max(dim=-1, keepdim=True).values
        descale = input_amax / cls.E2M1_max
        min_value = torch.tensor(-127.0, device=descale.device)
        e8m0_scale = torch.ceil(torch.maximum(torch.log2(descale), min_value))

        input = (input / torch.exp2(e8m0_scale)).view(original_shape)
        input_q = cast_fp4(input)
        input_q = fuse_uint4_to_uint8(input_q)
        e8m0_scale = (e8m0_scale + 127).to(torch.uint8)
        return cls(original_shape, original_dtype, input_q), e8m0_scale

    @classmethod
    def dequantize(cls, quantized_data, dtype: torch.dtype, scale, block_sizes):
        """Dequantze MXFP4 packed tensor to a target dtype."""

        def unfuse_uint8_to_uint4(x):
            """Unfuse uint8 values back to uint4 values.
            This is the inverse operation of fuse_uint4_to_uint8.
            """
            # Extract the lower 4 bits (even indices)
            left_side = x & 0x0F

            # Extract the upper 4 bits (odd indices)
            right_side = (x >> 4) & 0x0F

            # Create a new tensor with alternating values
            shape = list(x.shape)
            shape[-1] = shape[-1] * 2
            result = torch.zeros(shape, dtype=torch.uint8, device=x.device)

            # Fill in the values - even indices get low bits, odd indices get high bits
            result[..., 0::2] = left_side  # Even indices from low bits
            result[..., 1::2] = right_side  # Odd indices from high bits

            return result

        e8m0_scale = scale
        block_size = block_sizes[-1]

        # Unfuse the uint8 values back to uint4
        x_unfused = unfuse_uint8_to_uint4(quantized_data)
        # Extract sign and magnitude
        sign = 1 - 2 * ((x_unfused & 0b1000) >> 3).to(
            torch.float32
        )  # Extract sign bit and convert to +1/-1
        magnitude = x_unfused & 0b0111  # Extract magnitude bits
        magnitude = magnitude.to(torch.long)

        # Create a tensor with the E2M1 values
        values = torch.tensor(cls.E2M1_values, device=quantized_data.device)

        # Use gather to index the values tensor properly
        # We need to reshape magnitude to match the dimensions we want to gather along
        original_shape = magnitude.shape
        x_float = values[magnitude.reshape(-1)].reshape(original_shape)

        # Apply sign and scale
        x_float = sign.float() * x_float

        # Reshape to apply block-wise scaling
        x_float = x_float.reshape(-1, block_size)

        # Apply the E8M0 scale
        scale_factor = torch.exp2(e8m0_scale.float() - 127)
        scale_factor = scale_factor.reshape(-1, 1)  # Reshape for proper broadcasting

        # Apply scaling and reshape back to original shape
        x_float = x_float * scale_factor

        # Reshape back to the original shape
        return x_float.reshape(original_shape).to(dtype)

@triton.jit
def _swizzle_sf_kernel(
    x_ptr, y_ptr,
    N: tl.constexpr,              # total elements = R*C
    C: tl.constexpr,              # original_col
    num_k_tiles: tl.constexpr,    # C // 4
    stride_x0: tl.constexpr,      # x row stride (elements)
    stride_x1: tl.constexpr,      # x col stride (elements)
    stride_y0: tl.constexpr,      # y row stride (elements)
    stride_y1: tl.constexpr,      # y col stride (elements)
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # L = r_out * C + c_out (output is row-major contiguous)
    L = offs.to(tl.int64)

    # Unravel L into T[m_tile, k_tile, b, a, d] where
    # shape is (R/128, C/4, 32, 4, 4)
    d = (L % 4).to(tl.int32)
    L1 = L // 4
    a = (L1 % 4).to(tl.int32)
    L2 = L1 // 4
    b = (L2 % 32).to(tl.int32)
    L3 = L2 // 32
    k_tile = (L3 % num_k_tiles).to(tl.int32)
    m_tile = (L3 // num_k_tiles).to(tl.int32)

    # Map back to input (unswizzled) indices:
    # input is sf_reshaped[m_tile, a, b, k_tile, d]  => unswizzled[r_in, c_in]
    r_in = m_tile * 128 + a * 32 + b
    c_in = k_tile * 4 + d

    # Convert 2D -> linear with strides (elements)
    x_offs = r_in.to(tl.int64) * stride_x0 + c_in.to(tl.int64) * stride_x1
    y_offs = offs.to(tl.int64)  # output contiguous => linear

    x = tl.load(x_ptr + x_offs, mask=mask, other=0)
    tl.store(y_ptr + y_offs, x, mask=mask)


def swizzle_sf_triton(unswizzled_sf: torch.Tensor) -> torch.Tensor:
    """Swizzle scale factor tensor using Triton kernel."""
    if unswizzled_sf.ndim != 2:
        raise ValueError(f"not support unswizzled_sf's shape {unswizzled_sf.shape}, ndim: {unswizzled_sf.ndim}")

    x = unswizzled_sf
    if not x.is_cuda:
        raise ValueError("unswizzled_sf must be a CUDA tensor")

    # match your original behavior
    x = x.contiguous()

    R, C = x.shape
    factor = 4
    if R % 128 != 0:
        raise ValueError(f"R must be multiple of 128, got R={R}")
    if C % factor != 0:
        raise ValueError(f"C must be multiple of {factor}, got C={C}")

    num_k_tiles = C // factor
    y = torch.empty((R, C), device=x.device, dtype=x.dtype)  # final is contiguous

    # strides in elements
    stride_x0, stride_x1 = x.stride()
    stride_y0, stride_y1 = y.stride()

    N = R * C
    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)

    _swizzle_sf_kernel[grid](
        x, y,
        N=N,
        C=C,
        num_k_tiles=num_k_tiles,
        stride_x0=stride_x0, stride_x1=stride_x1,
        stride_y0=stride_y0, stride_y1=stride_y1,
        BLOCK=BLOCK,
        num_warps=4,
    )
    return y

@triton.jit
def _repeat_interleave_lastdim_kernel(
    x_ptr, y_ptr,
    M: tl.constexpr,         # rows (product of prefix dims)
    K: tl.constexpr,         # last dim of input
    stride_xm: tl.constexpr, # x stride for row dim (in elements)
    stride_xk: tl.constexpr, # x stride for col dim (in elements)
    stride_ym: tl.constexpr, # y stride for row dim (in elements)
    stride_yk: tl.constexpr, # y stride for col dim (in elements)
    BLOCK: tl.constexpr,     # number of output elements processed per program
    factor: tl.constexpr,    # number of times to repeat each element along last dim
):
    pid_m = tl.program_id(0)     # which row
    pid_b = tl.program_id(1)     # which block along last dim (output)

    # output column indices this program handles: [pid_b*BLOCK, pid_b*BLOCK+BLOCK)
    out_cols = pid_b * BLOCK + tl.arange(0, BLOCK)

    # map output col -> input col (repeat factor R=4)
    in_cols = out_cols // factor

    # bounds: out_cols < K*4  <=> in_cols < K
    mask = in_cols < K

    # base pointers for this row
    x_row = x_ptr + pid_m * stride_xm
    y_row = y_ptr + pid_m * stride_ym

    # load x[in_cols], store to y[out_cols]
    x_vals = tl.load(x_row + in_cols * stride_xk, mask=mask, other=0)
    tl.store(y_row + out_cols * stride_yk, x_vals, mask=mask)


def repeat_interleave_lastdim(x: torch.Tensor, repeats: int) -> torch.Tensor:
    """
    Equivalent to torch.repeat_interleave(x, repeats=repeats, dim=-1)
    Only for CUDA tensors. Works for any rank >= 1.
    """
    assert x.ndim >= 1, "x must have at least 1 dimension"

    # Flatten prefix dims -> [M, K]
    K = x.shape[-1]
    M = x.numel() // K
    x2 = x.reshape(-1, K)
    M = x2.shape[0]

    # Allocate output [M, K*repeats], then reshape back
    # y2 = torch.empty((M, K * repeats), device=x.device, dtype=x.dtype)
    y2 = torch.zeros((M, K * repeats), device=x.device, dtype=x.dtype)

    # Strides in elements (not bytes)
    stride_xm, stride_xk = x2.stride()
    stride_ym, stride_yk = y2.stride()

    # Launch config
    BLOCK = 1024  # output elements per program; tune if needed
    grid = (M, triton.cdiv(K * repeats, BLOCK))

    _repeat_interleave_lastdim_kernel[grid](
        x2, y2,
        M=M, K=K,
        stride_xm=stride_xm, stride_xk=stride_xk,
        stride_ym=stride_ym, stride_yk=stride_yk,
        BLOCK=BLOCK,
        num_warps=8, # hard code here, only for bzz2
        factor=repeats,
    )
    return y2.reshape(*x.shape[:-1], K * repeats)

@triton.jit
def _fused_repeat_swizzle_kernel(
    x_ptr,            # input scale [R, C_in_elem]
    y_ptr,            # output swizzled scale [R, C_out] (uint8)
    N: tl.constexpr,              # total output elements = R * C_out
    C_out: tl.constexpr,          # output columns (needed_sf)
    num_k_tiles: tl.constexpr,    # C_out // 4
    stride_x0: tl.constexpr,      # input row stride (in elements)
    stride_x1: tl.constexpr,      # input col stride (in elements)
    repeat_factor: tl.constexpr,  # C_out // C_in_uint8
    is_int32_packed: tl.constexpr, # whether input is int32 (4 uint8 per element)
    BLOCK: tl.constexpr,
):
    """Fused int32-unpack + repeat_interleave + swizzle kernel.

    For each output position in the swizzled layout, this kernel:
    1. Decomposes the linear index into swizzle 5D coordinates
    2. Maps the swizzled column back through repeat_interleave to the source column
    3. If int32 packed: loads int32 and extracts the correct byte
    4. Otherwise: loads uint8 directly
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < N

    L = offs.to(tl.int64)

    # Decompose into swizzle 5D: T[m_tile, k_tile, b, a, d]
    d = (L % 4).to(tl.int32)
    L1 = L // 4
    a = (L1 % 4).to(tl.int32)
    L2 = L1 // 4
    b = (L2 % 32).to(tl.int32)
    L3 = L2 // 32
    k_tile = (L3 % num_k_tiles).to(tl.int32)
    m_tile = (L3 // num_k_tiles).to(tl.int32)

    r_in = m_tile * 128 + a * 32 + b
    c_out = k_tile * 4 + d
    src_col = c_out // repeat_factor  # uint8-equivalent column index

    if is_int32_packed:
        int32_col = src_col // 4
        byte_idx = src_col % 4
        x_offs = r_in.to(tl.int64) * stride_x0 + int32_col.to(tl.int64) * stride_x1
        x_val_i32 = tl.load(x_ptr + x_offs, mask=valid, other=0)
        x_val = ((x_val_i32 >> (byte_idx.to(tl.int32) * 8)) & 0xFF).to(tl.uint8)
    else:
        x_offs = r_in.to(tl.int64) * stride_x0 + src_col.to(tl.int64) * stride_x1
        x_val = tl.load(x_ptr + x_offs, mask=valid, other=0)

    tl.store(y_ptr + offs.to(tl.int64), x_val, mask=valid)

def mxfp8_quantize(
    scale_factor: torch.Tensor,
    repeats: int,
    is_int32_packed: bool = False,
) -> torch.Tensor:
    """Fused int32-unpack + repeat_interleave + swizzle for MXFP8 scale factors.

    Args:
        scale_factor: scale tensor (uint8 or int32 packed)
        repeats: repeat factor for last dim (needed_sf // actual_sf_uint8)
        is_int32_packed: if True, input is int32 with 4 packed uint8 bytes per element

    Returns:
        Swizzled scale tensor as float8_e8m0fnu
    """
    C_in_elem = scale_factor.shape[-1]
    C_in_uint8 = C_in_elem * 4 if is_int32_packed else C_in_elem
    C_out = C_in_uint8 * repeats
    prefix_shape = scale_factor.shape[:-1]
    R = scale_factor.numel() // C_in_elem

    # Explicit contiguous() to avoid implicit copy inside reshape,
    # which can break CUDA graph capture.
    x = scale_factor.contiguous().reshape(R, C_in_elem)
    y = torch.empty((R, C_out), device=x.device, dtype=torch.uint8)

    assert R % 128 == 0, f"R must be multiple of 128, got R={R}"
    assert C_out % 4 == 0, f"C_out must be multiple of 4, got C_out={C_out}"

    N = R * C_out
    num_k_tiles = C_out // 4
    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)

    _fused_repeat_swizzle_kernel[grid](
        x, y,
        N=N, C_out=C_out,
        num_k_tiles=num_k_tiles,
        stride_x0=x.stride(0), stride_x1=x.stride(1),
        repeat_factor=repeats,
        is_int32_packed=is_int32_packed,
        BLOCK=BLOCK,
        num_warps=4,
    )
    return y.reshape(*prefix_shape, C_out).view(torch.float8_e8m0fnu)

@triton.jit
def _fused_repeat_swizzle_masked_kernel(
    x_ptr,            # input scale [R, C_in_elem]
    y_ptr,            # output swizzled scale [R, C_out] (uint8)
    masked_m_ptr,     # [num_experts] actual token count per expert
    M_STRIDE,         # actual per-expert row stride in flattened tensor (may be TMA-aligned)
    N: tl.constexpr,              # total output elements = R * C_out
    C_out: tl.constexpr,          # output columns (needed_sf)
    num_k_tiles: tl.constexpr,    # C_out // 4
    stride_x0: tl.constexpr,      # input row stride (in elements)
    stride_x1: tl.constexpr,      # input col stride (in elements)
    repeat_factor: tl.constexpr,  # C_out // C_in_uint8
    is_int32_packed: tl.constexpr, # whether input is int32 (4 uint8 per element)
    BLOCK: tl.constexpr,
):
    """Fused int32-unpack + repeat_interleave + swizzle kernel with per-expert token masking.

    For each output position in the swizzled layout, this kernel:
    1. Decomposes the linear index into swizzle 5D coordinates
    2. Checks if the row belongs to an active token (token_idx < masked_m[expert])
    3. Maps the swizzled column back through repeat_interleave to the source column
    4. If int32 packed: loads int32 and extracts the correct byte
    5. Otherwise: loads uint8 directly
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < N

    L = offs.to(tl.int64)

    # Decompose into swizzle 5D: T[m_tile, k_tile, b, a, d]
    d = (L % 4).to(tl.int32)
    L1 = L // 4
    a = (L1 % 4).to(tl.int32)
    L2 = L1 // 4
    b = (L2 % 32).to(tl.int32)
    L3 = L2 // 32
    k_tile = (L3 % num_k_tiles).to(tl.int32)
    m_tile = (L3 // num_k_tiles).to(tl.int32)

    # Map to source row
    r_in = m_tile * 128 + a * 32 + b

    # Mask check: determine expert and token index using actual tensor stride
    expert_id = r_in // M_STRIDE
    token_idx = r_in % M_STRIDE
    expert_m = tl.load(masked_m_ptr + expert_id, mask=valid, other=0)
    is_active = token_idx < expert_m

    # Fused repeat_interleave reverse mapping: swizzled col -> source col
    c_out = k_tile * 4 + d
    src_col = c_out // repeat_factor  # uint8-equivalent column index

    combined_mask = valid & is_active

    if is_int32_packed:
        int32_col = src_col // 4
        byte_idx = src_col % 4
        x_offs = r_in.to(tl.int64) * stride_x0 + int32_col.to(tl.int64) * stride_x1
        x_val_i32 = tl.load(x_ptr + x_offs, mask=combined_mask, other=0)
        x_val = ((x_val_i32 >> (byte_idx.to(tl.int32) * 8)) & 0xFF).to(tl.uint8)
    else:
        x_offs = r_in.to(tl.int64) * stride_x0 + src_col.to(tl.int64) * stride_x1
        x_val = tl.load(x_ptr + x_offs, mask=combined_mask, other=0)

    # Always store: active rows get real values, inactive rows get 0
    tl.store(y_ptr + offs.to(tl.int64), x_val, mask=valid)


def mxfp8_quantize_masked(
    scale_factor: torch.Tensor,
    repeats: int,
    masked_m: torch.Tensor,
    is_int32_packed: bool = False,
) -> torch.Tensor:
    """Fused int32-unpack + repeat_interleave + swizzle + mask for MXFP8 scale factors.

    Optimized for decode phase where many experts have few or zero tokens.
    Combines all operations into a single Triton kernel, eliminates intermediate
    tensor allocation, and skips computation for masked-out token rows.

    Args:
        scale_factor: scale tensor [num_experts, m, C] (uint8 or int32 packed)
        repeats: repeat factor for last dim (needed_sf // actual_sf_uint8)
        masked_m: [num_experts] actual token count per expert
        is_int32_packed: if True, input is int32 with 4 packed uint8 bytes per element

    Returns:
        Swizzled scale tensor as float8_e8m0fnu
    """
    C_in_elem = scale_factor.shape[-1]
    C_in_uint8 = C_in_elem * 4 if is_int32_packed else C_in_elem
    C_out = C_in_uint8 * repeats
    prefix_shape = scale_factor.shape[:-1]
    R = scale_factor.numel() // C_in_elem

    # Per-expert row stride derived directly from tensor shape.
    m_stride = scale_factor.shape[-2]

    # Explicit contiguous() to avoid implicit copy inside reshape,
    # which can break CUDA graph capture.
    x = scale_factor.contiguous().reshape(R, C_in_elem)
    y = torch.zeros((R, C_out), device=x.device, dtype=torch.uint8)

    assert R % 128 == 0, f"R must be multiple of 128, got R={R}"
    assert C_out % 4 == 0, f"C_out must be multiple of 4, got C_out={C_out}"

    N = R * C_out
    num_k_tiles = C_out // 4
    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)

    _fused_repeat_swizzle_masked_kernel[grid](
        x, y, masked_m,
        M_STRIDE=m_stride,
        N=N, C_out=C_out,
        num_k_tiles=num_k_tiles,
        stride_x0=x.stride(0), stride_x1=x.stride(1),
        repeat_factor=repeats,
        is_int32_packed=is_int32_packed,
        BLOCK=BLOCK,
        num_warps=4,
    )
    return y.reshape(*prefix_shape, C_out).view(torch.float8_e8m0fnu)