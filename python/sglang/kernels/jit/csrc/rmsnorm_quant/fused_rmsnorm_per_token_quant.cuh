#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <algorithm>
#include <cstdint>
#include <cuda_fp8.h>

namespace {

struct RMSNormQuantParams {
  const bf16_t* __restrict__ input;
  bf16_t* __restrict__ residual;
  const bf16_t* __restrict__ weight;
  bf16_t* __restrict__ norm_output;
  fp8_e4m3_t* __restrict__ output;
  void* __restrict__ output_scale;
  int64_t input_stride;
  int64_t residual_stride;
  int64_t norm_output_stride;
  int64_t output_stride;
  bool has_residual;
  float eps;
  float min_8bit;
  float max_8bit;
  float weight_bias;
  uint32_t hidden_dim;
  uint32_t output_scale_stride;
};

SGL_DEVICE float rsqrt_approx_ftz(float x) {
  float y;
  asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

SGL_DEVICE float shfl_xor_sync_f32(float x, int lane_mask) {
  float y;
  asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;"
               : "=f"(y)
               : "f"(x), "r"(lane_mask));
  return y;
}

SGL_DEVICE float fast_pow2(int x) {
  return __uint_as_float(static_cast<uint32_t>(x + 127) << 23);
}

SGL_DEVICE int fast_log2_ceil(float x) {
  const uint32_t bits = __float_as_uint(x);
  const int exp = static_cast<int>((bits >> 23) & 0xffu);
  const uint32_t mant = bits & 0x7fffffu;
  return exp - 127 + (mant != 0u);
}

SGL_DEVICE float2 fmul2_rn(float2 a, float2 b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  return __fmul2_rn(a, b);
#else
  return float2{a.x * b.x, a.y * b.y};
#endif
}

template <uint32_t kGroupThreads>
SGL_DEVICE float group_reduce_max_float(float val) {
  static_assert(kGroupThreads > 0 && kGroupThreads <= 32);
  static_assert((kGroupThreads & (kGroupThreads - 1)) == 0);
#pragma unroll
  for (uint32_t offset = kGroupThreads / 2; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset, kGroupThreads));
  }
  return val;
}

template <bool kScaleUE8M0>
SGL_DEVICE void calculate_scales(
    float amax, float& quant_scale, float& dequant_scale, uint32_t& ue8m0_exp) {
  constexpr float kMax8Bit = 448.0f;
  constexpr float kMax8BitInv = 1.0f / kMax8Bit;
  const float raw_dequant_scale = amax * kMax8BitInv;
  if constexpr (kScaleUE8M0) {
    const int exp_dequant_scale = fast_log2_ceil(raw_dequant_scale);
    quant_scale = fast_pow2(-exp_dequant_scale);
    dequant_scale = 0.0f;
    ue8m0_exp = static_cast<uint32_t>(exp_dequant_scale + 127);
  } else {
    quant_scale = kMax8Bit / amax;
    dequant_scale = raw_dequant_scale;
    ue8m0_exp = 0;
  }
}

SGL_DEVICE __nv_fp8x2_storage_t pack_fp8(
    float x, float y, float min_8bit, float max_8bit) {
  float2 outputx2 = {
      fmaxf(min_8bit, fminf(x, max_8bit)),
      fmaxf(min_8bit, fminf(y, max_8bit))};
  return __nv_cvt_float2_to_fp8x2(outputx2, __NV_SATFINITE, __NV_E4M3);
}

template <uint32_t VEC_SIZE, typename T>
SGL_DEVICE void load_vec_scalar(const T* ptr, T (&vec)[VEC_SIZE]) {
#pragma unroll
  for (uint32_t i = 0; i < VEC_SIZE; ++i) {
    vec[i] = ptr[i];
  }
}

template <uint32_t VEC_SIZE, typename T>
SGL_DEVICE void load_vec(const T* ptr, T (&vec)[VEC_SIZE]) {
  constexpr uint32_t kBytes = VEC_SIZE * sizeof(T);
  if constexpr (kBytes == sizeof(uint4) * 2) {
    auto* dst = reinterpret_cast<uint4*>(vec);
    const auto* src = reinterpret_cast<const uint4*>(ptr);
    dst[0] = src[0];
    dst[1] = src[1];
  } else if constexpr (kBytes == sizeof(uint4)) {
    *reinterpret_cast<uint4*>(vec) = *reinterpret_cast<const uint4*>(ptr);
  } else {
#pragma unroll
    for (uint32_t i = 0; i < VEC_SIZE; ++i) {
      vec[i] = ptr[i];
    }
  }
}

template <uint32_t VEC_SIZE, typename T>
SGL_DEVICE void store_vec_scalar(T* ptr, const T (&vec)[VEC_SIZE]) {
#pragma unroll
  for (uint32_t i = 0; i < VEC_SIZE; ++i) {
    ptr[i] = vec[i];
  }
}

template <uint32_t VEC_SIZE, typename T>
SGL_DEVICE void store_vec(T* ptr, const T (&vec)[VEC_SIZE]) {
  constexpr uint32_t kBytes = VEC_SIZE * sizeof(T);
  if constexpr (kBytes == sizeof(uint4) * 2) {
    auto* dst = reinterpret_cast<uint4*>(ptr);
    const auto* src = reinterpret_cast<const uint4*>(vec);
    dst[0] = src[0];
    dst[1] = src[1];
  } else if constexpr (kBytes == sizeof(uint4)) {
    *reinterpret_cast<uint4*>(ptr) = *reinterpret_cast<const uint4*>(vec);
  } else {
#pragma unroll
    for (uint32_t i = 0; i < VEC_SIZE; ++i) {
      ptr[i] = vec[i];
    }
  }
}

template <uint32_t VEC_SIZE>
SGL_DEVICE void store_vec(fp8_e4m3_t* ptr, const fp8_e4m3_t (&vec)[VEC_SIZE]) {
  constexpr uint32_t kBytes = VEC_SIZE * sizeof(fp8_e4m3_t);
  if constexpr (kBytes == sizeof(uint4)) {
    *reinterpret_cast<uint4*>(ptr) = *reinterpret_cast<const uint4*>(vec);
  } else if constexpr (kBytes == sizeof(uint2)) {
    *reinterpret_cast<uint2*>(ptr) = *reinterpret_cast<const uint2*>(vec);
  } else {
    static_assert(kBytes == sizeof(uint4) || kBytes == sizeof(uint2));
  }
}

template <uint32_t VEC_SIZE, uint32_t kGroupSize, bool kScaleUE8M0, bool kUsePDL>
__global__ void rmsnorm_per_token_quant_kernel(const RMSNormQuantParams __grid_constant__ params) {
  using namespace device;
  static_assert(kGroupSize % VEC_SIZE == 0);
  constexpr uint32_t kQuantGroupThreads = kGroupSize / VEC_SIZE;

  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x;
  const uint32_t ty = threadIdx.y;
  const uint32_t num_warps = blockDim.y;
  const uint32_t thread_id = tx + ty * kWarpThreads;
  const uint32_t offset = thread_id * VEC_SIZE;
  const bool valid_offset = offset < params.hidden_dim;
  extern __shared__ float smem[];

  PDLWaitPrimary<kUsePDL>();

  const auto input_row = params.input + static_cast<int64_t>(bx) * params.input_stride;
  const auto residual_row =
      params.has_residual ? params.residual + static_cast<int64_t>(bx) * params.residual_stride : nullptr;
  const auto norm_output_row = params.norm_output + static_cast<int64_t>(bx) * params.norm_output_stride;
  const auto output_row = params.output + static_cast<int64_t>(bx) * params.output_stride;
  const bool contiguous_input = params.input_stride == static_cast<int64_t>(params.hidden_dim);
  const bool contiguous_residual = params.residual_stride == static_cast<int64_t>(params.hidden_dim);
  const bool contiguous_norm_output = params.norm_output_stride == static_cast<int64_t>(params.hidden_dim);
  const bool contiguous_output = params.output_stride == static_cast<int64_t>(params.hidden_dim);

  alignas(16) bf16_t input_vec[VEC_SIZE];
  alignas(16) bf16_t residual_vec[VEC_SIZE];
  float sum_sq = 0.0f;
  if (valid_offset) {
    if (contiguous_input) {
      load_vec(input_row + offset, input_vec);
    } else {
      load_vec_scalar(input_row + offset, input_vec);
    }
    if (params.has_residual) {
      if (contiguous_residual) {
        load_vec(residual_row + offset, residual_vec);
      } else {
        load_vec_scalar(residual_row + offset, residual_vec);
      }
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        input_vec[j] = cast<bf16_t>(static_cast<float>(input_vec[j]) + static_cast<float>(residual_vec[j]));
      }
      if (contiguous_residual) {
        store_vec(residual_row + offset, input_vec);
      } else {
        store_vec_scalar(residual_row + offset, input_vec);
      }
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      const float val = static_cast<float>(input_vec[j]);
      sum_sq += val * val;
    }
  }

#pragma unroll
  for (uint32_t offset = kWarpThreads / 2; offset > 0; offset /= 2) {
    sum_sq += shfl_xor_sync_f32(sum_sq, offset);
  }

  smem[ty] = sum_sq;
  __syncthreads();

  if (ty == 0) {
    sum_sq = tx < num_warps ? smem[tx] : 0.0f;
#pragma unroll
    for (uint32_t offset = kWarpThreads / 2; offset > 0; offset /= 2) {
      sum_sq += shfl_xor_sync_f32(sum_sq, offset);
    }
    smem[0] = sum_sq;
  }
  __syncthreads();

  const float rms_rcp = rsqrt_approx_ftz(smem[0] / static_cast<float>(params.hidden_dim) + params.eps);
  const uint32_t output_scale_stride = params.output_scale_stride;

  alignas(16) bf16_t weight_vec[VEC_SIZE];
  if (valid_offset) {
    load_vec(params.weight + offset, weight_vec);
  }

  float local_absmax = 1e-10f;
  if (valid_offset) {
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      const float norm_val =
          static_cast<float>(input_vec[j]) * rms_rcp * (params.weight_bias + static_cast<float>(weight_vec[j]));
      input_vec[j] = cast<bf16_t>(norm_val);
      const float val = static_cast<float>(input_vec[j]);
      local_absmax = fmaxf(local_absmax, fabsf(val));
    }
    if (contiguous_norm_output) {
      store_vec(norm_output_row + offset, input_vec);
    } else {
      store_vec_scalar(norm_output_row + offset, input_vec);
    }
  }

  const uint32_t lane_id = thread_id % kQuantGroupThreads;
  const float group_max = group_reduce_max_float<kQuantGroupThreads>(local_absmax);

  float quant_scale = 0.0f;
  float dequant_scale = 0.0f;
  uint32_t ue8m0_exp = 0;
  if (lane_id == 0) {
    calculate_scales<kScaleUE8M0>(group_max, quant_scale, dequant_scale, ue8m0_exp);
  }
  quant_scale = __shfl_sync(kFullMask, quant_scale, 0, kQuantGroupThreads);
  if constexpr (kScaleUE8M0) {
    ue8m0_exp = __shfl_sync(kFullMask, ue8m0_exp, 0, kQuantGroupThreads);
  } else {
    dequant_scale = __shfl_sync(kFullMask, dequant_scale, 0, kQuantGroupThreads);
  }

  alignas(16) fp8_e4m3_t fp8[VEC_SIZE];
  if (valid_offset) {
    const float2 quant_scale_x2 = {quant_scale, quant_scale};
#pragma unroll
    for (uint32_t k = 0; k < VEC_SIZE; k += 2) {
      const float x = static_cast<float>(input_vec[k]);
      const float y = static_cast<float>(input_vec[k + 1]);
      const float2 outputx2 = fmul2_rn(float2{x, y}, quant_scale_x2);
      reinterpret_cast<__nv_fp8x2_storage_t*>(fp8)[k / 2] =
          pack_fp8(outputx2.x, outputx2.y, params.min_8bit, params.max_8bit);
    }
  }

  if (valid_offset) {
    if (lane_id == 0) {
      const uint32_t group_id = offset / kGroupSize;
      if constexpr (kScaleUE8M0) {
        const uint32_t scale_offset = (group_id / 4u) * output_scale_stride * 4u + bx * 4u + group_id % 4u;
        auto* output_scale = reinterpret_cast<uint8_t*>(params.output_scale);
        output_scale[scale_offset] = static_cast<uint8_t>(ue8m0_exp);
        const uint32_t num_groups = params.hidden_dim / kGroupSize;
        const uint32_t remainder = num_groups % 4u;
        if (remainder != 0u && group_id == num_groups - 1u) {
#pragma unroll
          for (uint32_t shift = 1u; shift < 4u; ++shift) {
            if (shift <= 4u - remainder) {
              output_scale[scale_offset + shift] = 0;
            }
          }
        }
      } else {
        reinterpret_cast<float*>(params.output_scale)[group_id * output_scale_stride + bx] = dequant_scale;
      }
    }
    if (contiguous_output) {
      store_vec(output_row + offset, fp8);
    } else {
      store_vec_scalar(output_row + offset, fp8);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kGroupSize, bool kScaleUE8M0, bool kUsePDL>
struct RMSNormPerTokenQuantKernel {
  static_assert(kGroupSize == 32 || kGroupSize == 128);
  static constexpr uint32_t kVecSize = 16;
  static constexpr auto kernel =
      rmsnorm_per_token_quant_kernel<kVecSize, static_cast<uint32_t>(kGroupSize), kScaleUE8M0, kUsePDL>;

  static void run(
      const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView norm_output,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView output_scale,
      const double eps,
      const double min_8bit,
      const double max_8bit) {
    using namespace host;

    auto device = SymbolicDevice{};
    auto M = SymbolicSize{"num_tokens"};
    auto D = SymbolicSize{"hidden_dim"};
    auto G = SymbolicSize{"num_groups"};
    auto IS = SymbolicSize{"input_stride"};
    auto NS = SymbolicSize{"norm_output_stride"};
    auto OS = SymbolicSize{"output_stride"};
    auto S = SymbolicSize{"output_scale_stride"};
    device.set_options<kDLCUDA>();

    TensorMatcher({M, D}).with_strides({IS, 1}).with_dtype<bf16_t>().with_device(device).verify(input);
    TensorMatcher({D}).with_dtype<bf16_t>().with_device(device).verify(weight);
    TensorMatcher({M, D}).with_strides({NS, 1}).with_dtype<bf16_t>().with_device(device).verify(norm_output);
    TensorMatcher({M, D}).with_strides({OS, 1}).with_dtype<fp8_e4m3_t>().with_device(device).verify(output);
    if constexpr (kScaleUE8M0) {
      TensorMatcher({M, G}).with_strides({1, S}).with_dtype<int32_t>().with_device(device).verify(output_scale);
    } else {
      TensorMatcher({M, G}).with_strides({1, S}).with_dtype<fp32_t>().with_device(device).verify(output_scale);
    }

    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    const auto hidden_dim = static_cast<uint32_t>(D.unwrap());
    const auto num_groups = static_cast<uint32_t>(G.unwrap());
    const auto output_scale_stride = static_cast<uint32_t>(S.unwrap());

    RuntimeCheck(hidden_dim > 0, "hidden_dim must be positive");
    RuntimeCheck(hidden_dim % kGroupSize == 0, "hidden_dim must be divisible by group_size");
    RuntimeCheck(hidden_dim % kVecSize == 0, "hidden_dim must be divisible by vector size");
    RuntimeCheck(hidden_dim <= kVecSize * 1024u, "hidden_dim is too large for register-resident fast path");
    const auto hidden_dim_num_groups = hidden_dim / static_cast<uint32_t>(kGroupSize);
    if constexpr (kScaleUE8M0) {
      RuntimeCheck(num_groups == div_ceil(hidden_dim_num_groups, 4u), "invalid packed scale groups");
    } else {
      RuntimeCheck(num_groups == hidden_dim_num_groups, "invalid scale groups");
    }
    RuntimeCheck(
        static_cast<float>(min_8bit) == -448.0f && static_cast<float>(max_8bit) == 448.0f,
        "invalid fp8_e4m3 quant range");

    const auto params = RMSNormQuantParams{
        .input = static_cast<const bf16_t*>(input.data_ptr()),
        .residual = nullptr,
        .weight = static_cast<const bf16_t*>(weight.data_ptr()),
        .norm_output = static_cast<bf16_t*>(norm_output.data_ptr()),
        .output = static_cast<fp8_e4m3_t*>(output.data_ptr()),
        .output_scale = output_scale.data_ptr(),
        .input_stride = IS.unwrap(),
        .residual_stride = 0,
        .norm_output_stride = NS.unwrap(),
        .output_stride = OS.unwrap(),
        .has_residual = false,
        .eps = static_cast<float>(eps),
        .min_8bit = static_cast<float>(min_8bit),
        .max_8bit = static_cast<float>(max_8bit),
        .weight_bias = 0.0f,
        .hidden_dim = hidden_dim,
        .output_scale_stride = output_scale_stride,
    };
    if (num_tokens == 0) return;

    const uint32_t block_size = std::min<uint32_t>(1024, hidden_dim / kVecSize);
    const uint32_t num_warps = div_ceil(block_size, device::kWarpThreads);
    const dim3 block(device::kWarpThreads, num_warps);
    const uint32_t smem_size = num_warps * sizeof(float);
    LaunchKernel(num_tokens, block, device.unwrap(), smem_size).enable_pdl(kUsePDL)(kernel, params);
  }
};

template <int64_t kGroupSize, bool kScaleUE8M0, bool kUsePDL>
struct FusedAddRMSNormPerTokenQuantKernel {
  static_assert(kGroupSize == 32 || kGroupSize == 128);
  static constexpr uint32_t kVecSize = 16;
  static constexpr auto kernel =
      rmsnorm_per_token_quant_kernel<kVecSize, static_cast<uint32_t>(kGroupSize), kScaleUE8M0, kUsePDL>;

  static void run(
      const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView norm_output,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView output_scale,
      const double eps,
      const double min_8bit,
      const double max_8bit) {
    using namespace host;

    auto device = SymbolicDevice{};
    auto M = SymbolicSize{"num_tokens"};
    auto D = SymbolicSize{"hidden_dim"};
    auto G = SymbolicSize{"num_groups"};
    auto IS = SymbolicSize{"input_stride"};
    auto RS = SymbolicSize{"residual_stride"};
    auto NS = SymbolicSize{"norm_output_stride"};
    auto OS = SymbolicSize{"output_stride"};
    auto S = SymbolicSize{"output_scale_stride"};
    device.set_options<kDLCUDA>();

    TensorMatcher({M, D}).with_strides({IS, 1}).with_dtype<bf16_t>().with_device(device).verify(input);
    TensorMatcher({M, D}).with_strides({RS, 1}).with_dtype<bf16_t>().with_device(device).verify(residual);
    TensorMatcher({D}).with_dtype<bf16_t>().with_device(device).verify(weight);
    TensorMatcher({M, D}).with_strides({NS, 1}).with_dtype<bf16_t>().with_device(device).verify(norm_output);
    TensorMatcher({M, D}).with_strides({OS, 1}).with_dtype<fp8_e4m3_t>().with_device(device).verify(output);
    if constexpr (kScaleUE8M0) {
      TensorMatcher({M, G}).with_strides({1, S}).with_dtype<int32_t>().with_device(device).verify(output_scale);
    } else {
      TensorMatcher({M, G}).with_strides({1, S}).with_dtype<fp32_t>().with_device(device).verify(output_scale);
    }

    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    const auto hidden_dim = static_cast<uint32_t>(D.unwrap());
    const auto num_groups = static_cast<uint32_t>(G.unwrap());
    const auto output_scale_stride = static_cast<uint32_t>(S.unwrap());

    RuntimeCheck(hidden_dim > 0, "hidden_dim must be positive");
    RuntimeCheck(hidden_dim % kGroupSize == 0, "hidden_dim must be divisible by group_size");
    RuntimeCheck(hidden_dim % kVecSize == 0, "hidden_dim must be divisible by vector size");
    RuntimeCheck(hidden_dim <= kVecSize * 1024u, "hidden_dim is too large for register-resident fast path");
    const auto hidden_dim_num_groups = hidden_dim / static_cast<uint32_t>(kGroupSize);
    if constexpr (kScaleUE8M0) {
      RuntimeCheck(num_groups == div_ceil(hidden_dim_num_groups, 4u), "invalid packed scale groups");
    } else {
      RuntimeCheck(num_groups == hidden_dim_num_groups, "invalid scale groups");
    }
    RuntimeCheck(
        static_cast<float>(min_8bit) == -448.0f && static_cast<float>(max_8bit) == 448.0f,
        "invalid fp8_e4m3 quant range");

    const auto params = RMSNormQuantParams{
        .input = static_cast<const bf16_t*>(input.data_ptr()),
        .residual = static_cast<bf16_t*>(residual.data_ptr()),
        .weight = static_cast<const bf16_t*>(weight.data_ptr()),
        .norm_output = static_cast<bf16_t*>(norm_output.data_ptr()),
        .output = static_cast<fp8_e4m3_t*>(output.data_ptr()),
        .output_scale = output_scale.data_ptr(),
        .input_stride = IS.unwrap(),
        .residual_stride = RS.unwrap(),
        .norm_output_stride = NS.unwrap(),
        .output_stride = OS.unwrap(),
        .has_residual = true,
        .eps = static_cast<float>(eps),
        .min_8bit = static_cast<float>(min_8bit),
        .max_8bit = static_cast<float>(max_8bit),
        .weight_bias = 0.0f,
        .hidden_dim = hidden_dim,
        .output_scale_stride = output_scale_stride,
    };
    if (num_tokens == 0) return;

    const uint32_t block_size = std::min<uint32_t>(1024, hidden_dim / kVecSize);
    const uint32_t num_warps = div_ceil(block_size, device::kWarpThreads);
    const dim3 block(device::kWarpThreads, num_warps);
    const uint32_t smem_size = num_warps * sizeof(float);
    LaunchKernel(num_tokens, block, device.unwrap(), smem_size).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
