// ===========================================================================
// Fused Q-latent RMSNorm + per-token group FP8 quant  +  KV-latent RMSNorm
// ---------------------------------------------------------------------------
// GLM 5.2 / DeepSeek-style MLA `forward_absorb_prepare` hot path.
//
// Baseline (nsys): three back-to-back launches touching the same token row:
//     RMSNormKernel                    (q  = q_a_layernorm(q))
//     RMSNormKernel                    (k_nope = kv_a_layernorm(k_nope))
//     per_token_group_quant_8bit_kernel(q -> fp8 for q_b_proj block-fp8 GEMM)
//
// This kernel collapses all three into ONE launch, one CTA per token:
//   * q  : RMSNorm -> bf16 norm_output  AND  per-token group (128) FP8 quant
//          with UE8M0 column-major TMA-aligned scale (deep_gemm drop-in).
//   * k  : RMSNorm -> bf16 norm_output (unquantized).
//
// The q half is bit-identical to the upstream `RMSNormPerTokenQuantKernel`
// (csrc/deepseek_v4/fused_rmsnorm_per_token_quant.cuh): same rsqrt.approx.ftz
// reduction, same UE8M0 scale math, same scale layout. We only add a second
// RMSNorm phase for k_nope inside the same CTA so the third launch (and its
// extra HBM round-trip of the k row) disappears.
//
// iter3+: the kernel is specialised on compile-time q/k hidden dims so the
// block size and occupancy (`__launch_bounds__`) are known at compile time and
// the inner loops fully unroll.
// ===========================================================================
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <algorithm>
#include <cstdint>
#include <cuda_fp8.h>

namespace {

struct QKVNormQuantParams {
  // q side (norm + fp8 group quant)
  const bf16_t* __restrict__ q_input;
  const bf16_t* __restrict__ q_weight;
  bf16_t* __restrict__ q_norm_output;
  fp8_e4m3_t* __restrict__ q_output;
  void* __restrict__ q_output_scale;
  int64_t q_input_stride;
  int64_t q_norm_output_stride;
  int64_t q_output_stride;
  uint32_t q_output_scale_stride;
  float q_eps;
  // k side (norm only)
  const bf16_t* __restrict__ k_input;
  const bf16_t* __restrict__ k_weight;
  bf16_t* __restrict__ k_norm_output;
  int64_t k_input_stride;
  int64_t k_norm_output_stride;
  float k_eps;
  // shared fp8 clamp range
  float min_8bit;
  float max_8bit;
};

// ---------------------------------------------------------------------------
// Device helpers (mirrors upstream fused_rmsnorm_per_token_quant.cuh so the q
// half stays numerically bit-identical to the shipping kernel).
// ---------------------------------------------------------------------------
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
SGL_DEVICE void calculate_scales(float amax, float& quant_scale, float& dequant_scale, uint32_t& ue8m0_exp) {
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

SGL_DEVICE __nv_fp8x2_storage_t pack_fp8(float x, float y, float min_8bit, float max_8bit) {
  float2 outputx2 = {fmaxf(min_8bit, fminf(x, max_8bit)), fmaxf(min_8bit, fminf(y, max_8bit))};
  return __nv_cvt_float2_to_fp8x2(outputx2, __NV_SATFINITE, __NV_E4M3);
}

template <uint32_t VEC_SIZE, typename T>
SGL_DEVICE void load_vec_scalar(const T* ptr, T (&vec)[VEC_SIZE]) {
#pragma unroll
  for (uint32_t i = 0; i < VEC_SIZE; ++i) vec[i] = ptr[i];
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
    for (uint32_t i = 0; i < VEC_SIZE; ++i) vec[i] = ptr[i];
  }
}

template <uint32_t VEC_SIZE, typename T>
SGL_DEVICE void store_vec_scalar(T* ptr, const T (&vec)[VEC_SIZE]) {
#pragma unroll
  for (uint32_t i = 0; i < VEC_SIZE; ++i) ptr[i] = vec[i];
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
    for (uint32_t i = 0; i < VEC_SIZE; ++i) ptr[i] = vec[i];
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

// Block-wide sum-of-squares -> reciprocal RMS (smem holds >= num_warps floats).
SGL_DEVICE float block_rms_rcp(float sum_sq, float* smem, uint32_t ty, uint32_t tx, uint32_t num_warps,
                               uint32_t hidden_dim, float eps) {
  using namespace device;
#pragma unroll
  for (uint32_t offset = kWarpThreads / 2; offset > 0; offset /= 2) {
    sum_sq += shfl_xor_sync_f32(sum_sq, offset);
  }
  smem[ty] = sum_sq;
  __syncthreads();
  if (ty == 0) {
    float v = tx < num_warps ? smem[tx] : 0.0f;
#pragma unroll
    for (uint32_t offset = kWarpThreads / 2; offset > 0; offset /= 2) {
      v += shfl_xor_sync_f32(v, offset);
    }
    smem[0] = v;
  }
  __syncthreads();
  return rsqrt_approx_ftz(smem[0] / static_cast<float>(hidden_dim) + eps);
}

// ---------------------------------------------------------------------------
// Compile-time block geometry helpers.
// ---------------------------------------------------------------------------
constexpr uint32_t kMaxU(uint32_t a, uint32_t b) { return a > b ? a : b; }
constexpr uint32_t kMinU(uint32_t a, uint32_t b) { return a < b ? a : b; }
constexpr uint32_t div_ceil_c(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

// Block threads = ceil-to-warp of max(q,k) row / VEC_SIZE, capped at 1024.
template <uint32_t VEC_SIZE, uint32_t kQHidden, uint32_t kKHidden>
constexpr uint32_t kBlockThreads() {
  constexpr uint32_t raw = kMinU(1024u, kMaxU(kQHidden, kKHidden) / VEC_SIZE);
  return div_ceil_c(raw, 32u) * 32u;
}

// Occupancy target (threads/SM) used for the __launch_bounds__ min-blocks hint.
// iter3 introduced this (2048 = full occupancy on sm_100) and it removed the
// large-N regression. iter5 tried pushing to 3072 but that spilled registers
// and regressed the default path at mid/large N, so 2048 is the locked value.
#ifndef QKV_OCC_TARGET
#define QKV_OCC_TARGET 2048u
#endif

template <uint32_t kBlock>
constexpr uint32_t kMinBlocksPerSM() {
  constexpr uint32_t b = kMinU(32u, kMaxU(1u, QKV_OCC_TARGET / kBlock));
  return b;
}

// ---------------------------------------------------------------------------
// Fused kernel: one CTA per token. Phase Q (norm + fp8 group quant), then
// Phase K (norm only). Specialised on compile-time hidden dims.
// ---------------------------------------------------------------------------
template <uint32_t VEC_SIZE, uint32_t kGroupSize, uint32_t kQHidden, uint32_t kKHidden, bool kScaleUE8M0,
          bool kUsePDL, bool kWriteQNorm>
__launch_bounds__(kBlockThreads<VEC_SIZE, kQHidden, kKHidden>(),
                  kMinBlocksPerSM<kBlockThreads<VEC_SIZE, kQHidden, kKHidden>()>())
__global__ void qkv_norm_quant_kernel(const QKVNormQuantParams __grid_constant__ params) {
  using namespace device;
  static_assert(kGroupSize % VEC_SIZE == 0);
  static_assert(kQHidden % kGroupSize == 0 && kQHidden % VEC_SIZE == 0);
  static_assert(kKHidden % VEC_SIZE == 0);
  constexpr uint32_t kQuantGroupThreads = kGroupSize / VEC_SIZE;
  constexpr bool kKSingleWarp = kKHidden <= kWarpThreads * VEC_SIZE;

  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x;
  const uint32_t ty = threadIdx.y;
  const uint32_t num_warps = blockDim.y;
  const uint32_t thread_id = tx + ty * kWarpThreads;
  const uint32_t offset = thread_id * VEC_SIZE;
  extern __shared__ float smem[];

  PDLWaitPrimary<kUsePDL>();

  // ---------------------------- Phase Q ------------------------------------
  {
    const bool valid = offset < kQHidden;
    const auto input_row = params.q_input + static_cast<int64_t>(bx) * params.q_input_stride;
    const auto norm_row = params.q_norm_output + static_cast<int64_t>(bx) * params.q_norm_output_stride;
    const auto out_row = params.q_output + static_cast<int64_t>(bx) * params.q_output_stride;
    const bool cont_in = params.q_input_stride == static_cast<int64_t>(kQHidden);
    const bool cont_norm = params.q_norm_output_stride == static_cast<int64_t>(kQHidden);
    const bool cont_out = params.q_output_stride == static_cast<int64_t>(kQHidden);

    alignas(16) bf16_t input_vec[VEC_SIZE];
    float sum_sq = 0.0f;
    if (valid) {
      if (cont_in) load_vec(input_row + offset, input_vec);
      else load_vec_scalar(input_row + offset, input_vec);
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        const float v = static_cast<float>(input_vec[j]);
        sum_sq += v * v;
      }
    }

    const float rms_rcp = block_rms_rcp(sum_sq, smem, ty, tx, num_warps, kQHidden, params.q_eps);

    alignas(16) bf16_t weight_vec[VEC_SIZE];
    if (valid) load_vec(params.q_weight + offset, weight_vec);

    float local_absmax = 1e-10f;
    if (valid) {
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        const float norm_val = static_cast<float>(input_vec[j]) * rms_rcp * static_cast<float>(weight_vec[j]);
        input_vec[j] = cast<bf16_t>(norm_val);
        local_absmax = fmaxf(local_absmax, fabsf(static_cast<float>(input_vec[j])));
      }
      if constexpr (kWriteQNorm) {
        if (cont_norm) store_vec(norm_row + offset, input_vec);
        else store_vec_scalar(norm_row + offset, input_vec);
      }
    }

    const uint32_t lane_id = thread_id % kQuantGroupThreads;
    const float group_max = group_reduce_max_float<kQuantGroupThreads>(local_absmax);

    float quant_scale = 0.0f, dequant_scale = 0.0f;
    uint32_t ue8m0_exp = 0;
    if (lane_id == 0) calculate_scales<kScaleUE8M0>(group_max, quant_scale, dequant_scale, ue8m0_exp);
    quant_scale = __shfl_sync(kFullMask, quant_scale, 0, kQuantGroupThreads);
    if constexpr (kScaleUE8M0) {
      ue8m0_exp = __shfl_sync(kFullMask, ue8m0_exp, 0, kQuantGroupThreads);
    } else {
      dequant_scale = __shfl_sync(kFullMask, dequant_scale, 0, kQuantGroupThreads);
    }

    alignas(16) fp8_e4m3_t fp8[VEC_SIZE];
    if (valid) {
      const float2 qsx2 = {quant_scale, quant_scale};
#pragma unroll
      for (uint32_t k = 0; k < VEC_SIZE; k += 2) {
        const float x = static_cast<float>(input_vec[k]);
        const float y = static_cast<float>(input_vec[k + 1]);
        const float2 o = fmul2_rn(float2{x, y}, qsx2);
        reinterpret_cast<__nv_fp8x2_storage_t*>(fp8)[k / 2] = pack_fp8(o.x, o.y, params.min_8bit, params.max_8bit);
      }
    }

    if (valid) {
      if (lane_id == 0) {
        const uint32_t group_id = offset / kGroupSize;
        if constexpr (kScaleUE8M0) {
          const uint32_t so = (group_id / 4u) * params.q_output_scale_stride * 4u + bx * 4u + group_id % 4u;
          auto* osc = reinterpret_cast<uint8_t*>(params.q_output_scale);
          osc[so] = static_cast<uint8_t>(ue8m0_exp);
          constexpr uint32_t num_groups = kQHidden / kGroupSize;
          constexpr uint32_t remainder = num_groups % 4u;
          if constexpr (remainder != 0u) {
            if (group_id == num_groups - 1u) {
#pragma unroll
              for (uint32_t shift = 1u; shift < 4u; ++shift)
                if (shift <= 4u - remainder) osc[so + shift] = 0;
            }
          }
        } else {
          reinterpret_cast<float*>(params.q_output_scale)[group_id * params.q_output_scale_stride + bx] =
              dequant_scale;
        }
      }
      if (cont_out) store_vec(out_row + offset, fp8);
      else store_vec_scalar(out_row + offset, fp8);
    }
  }

  // Reuse smem for the k reduction only when k spans >1 warp.
  if constexpr (!kKSingleWarp) __syncthreads();

  // ---------------------------- Phase K ------------------------------------
  {
    const bool valid = offset < kKHidden;
    const auto input_row = params.k_input + static_cast<int64_t>(bx) * params.k_input_stride;
    const auto norm_row = params.k_norm_output + static_cast<int64_t>(bx) * params.k_norm_output_stride;
    const bool cont_in = params.k_input_stride == static_cast<int64_t>(kKHidden);
    const bool cont_norm = params.k_norm_output_stride == static_cast<int64_t>(kKHidden);

    alignas(16) bf16_t input_vec[VEC_SIZE];
    float sum_sq = 0.0f;
    if (valid) {
      if (cont_in) load_vec(input_row + offset, input_vec);
      else load_vec_scalar(input_row + offset, input_vec);
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        const float v = static_cast<float>(input_vec[j]);
        sum_sq += v * v;
      }
    }

    float rms_rcp;
    if constexpr (kKSingleWarp) {
#pragma unroll
      for (uint32_t o = kWarpThreads / 2; o > 0; o /= 2) sum_sq += shfl_xor_sync_f32(sum_sq, o);
      rms_rcp = rsqrt_approx_ftz(sum_sq / static_cast<float>(kKHidden) + params.k_eps);
    } else {
      rms_rcp = block_rms_rcp(sum_sq, smem, ty, tx, num_warps, kKHidden, params.k_eps);
    }

    if (valid) {
      alignas(16) bf16_t weight_vec[VEC_SIZE];
      load_vec(params.k_weight + offset, weight_vec);
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        const float norm_val = static_cast<float>(input_vec[j]) * rms_rcp * static_cast<float>(weight_vec[j]);
        input_vec[j] = cast<bf16_t>(norm_val);
      }
      if (cont_norm) store_vec(norm_row + offset, input_vec);
      else store_vec_scalar(norm_row + offset, input_vec);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

// ---------------------------------------------------------------------------
// Host launcher. Template args: <group_size, scale_ue8m0, pdl, q_hidden, k_hidden>.
// ---------------------------------------------------------------------------
template <int64_t kGroupSize, bool kScaleUE8M0, bool kUsePDL, bool kWriteQNorm, int64_t kQHidden, int64_t kKHidden>
struct QKVNormQuantKernel {
  static_assert(kGroupSize == 32 || kGroupSize == 128);
  static constexpr uint32_t kVecSize = 16;
  static constexpr uint32_t kQ = static_cast<uint32_t>(kQHidden);
  static constexpr uint32_t kK = static_cast<uint32_t>(kKHidden);
  static constexpr auto kernel =
      qkv_norm_quant_kernel<kVecSize, static_cast<uint32_t>(kGroupSize), kQ, kK, kScaleUE8M0, kUsePDL, kWriteQNorm>;
  static constexpr uint32_t kBlock = kBlockThreads<kVecSize, kQ, kK>();
  static constexpr uint32_t kNumWarps = kBlock / 32u;

  static void run(
      const tvm::ffi::TensorView q_input,
      const tvm::ffi::TensorView q_weight,
      const tvm::ffi::TensorView q_norm_output,
      const tvm::ffi::TensorView q_output,
      const tvm::ffi::TensorView q_output_scale,
      const tvm::ffi::TensorView k_input,
      const tvm::ffi::TensorView k_weight,
      const tvm::ffi::TensorView k_norm_output,
      const double q_eps,
      const double k_eps,
      const double min_8bit,
      const double max_8bit) {
    using namespace host;

    auto device = SymbolicDevice{};
    auto M = SymbolicSize{"num_tokens"};
    auto GQ = SymbolicSize{"q_num_groups"};
    auto QIS = SymbolicSize{"q_input_stride"};
    auto QNS = SymbolicSize{"q_norm_output_stride"};
    auto QOS = SymbolicSize{"q_output_stride"};
    auto QSS = SymbolicSize{"q_output_scale_stride"};
    auto KIS = SymbolicSize{"k_input_stride"};
    auto KNS = SymbolicSize{"k_norm_output_stride"};
    device.set_options<kDLCUDA>();

    TensorMatcher({M, kQHidden}).with_strides({QIS, 1}).with_dtype<bf16_t>().with_device(device).verify(q_input);
    TensorMatcher({kQHidden}).with_dtype<bf16_t>().with_device(device).verify(q_weight);
    TensorMatcher({M, kQHidden}).with_strides({QNS, 1}).with_dtype<bf16_t>().with_device(device).verify(q_norm_output);
    TensorMatcher({M, kQHidden}).with_strides({QOS, 1}).with_dtype<fp8_e4m3_t>().with_device(device).verify(q_output);
    if constexpr (kScaleUE8M0) {
      TensorMatcher({M, GQ}).with_strides({1, QSS}).with_dtype<int32_t>().with_device(device).verify(q_output_scale);
    } else {
      TensorMatcher({M, GQ}).with_strides({1, QSS}).with_dtype<fp32_t>().with_device(device).verify(q_output_scale);
    }
    TensorMatcher({M, kKHidden}).with_strides({KIS, 1}).with_dtype<bf16_t>().with_device(device).verify(k_input);
    TensorMatcher({kKHidden}).with_dtype<bf16_t>().with_device(device).verify(k_weight);
    TensorMatcher({M, kKHidden}).with_strides({KNS, 1}).with_dtype<bf16_t>().with_device(device).verify(k_norm_output);

    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    const auto q_num_groups = static_cast<uint32_t>(GQ.unwrap());
    const auto q_scale_stride = static_cast<uint32_t>(QSS.unwrap());

    constexpr uint32_t q_hidden_num_groups = kQ / static_cast<uint32_t>(kGroupSize);
    if constexpr (kScaleUE8M0) {
      RuntimeCheck(q_num_groups == div_ceil(q_hidden_num_groups, 4u), "invalid packed scale groups");
    } else {
      RuntimeCheck(q_num_groups == q_hidden_num_groups, "invalid scale groups");
    }
    RuntimeCheck(
        static_cast<float>(min_8bit) == -448.0f && static_cast<float>(max_8bit) == 448.0f,
        "invalid fp8_e4m3 quant range");

    const auto params = QKVNormQuantParams{
        .q_input = static_cast<const bf16_t*>(q_input.data_ptr()),
        .q_weight = static_cast<const bf16_t*>(q_weight.data_ptr()),
        .q_norm_output = static_cast<bf16_t*>(q_norm_output.data_ptr()),
        .q_output = static_cast<fp8_e4m3_t*>(q_output.data_ptr()),
        .q_output_scale = q_output_scale.data_ptr(),
        .q_input_stride = QIS.unwrap(),
        .q_norm_output_stride = QNS.unwrap(),
        .q_output_stride = QOS.unwrap(),
        .q_output_scale_stride = q_scale_stride,
        .q_eps = static_cast<float>(q_eps),
        .k_input = static_cast<const bf16_t*>(k_input.data_ptr()),
        .k_weight = static_cast<const bf16_t*>(k_weight.data_ptr()),
        .k_norm_output = static_cast<bf16_t*>(k_norm_output.data_ptr()),
        .k_input_stride = KIS.unwrap(),
        .k_norm_output_stride = KNS.unwrap(),
        .k_eps = static_cast<float>(k_eps),
        .min_8bit = static_cast<float>(min_8bit),
        .max_8bit = static_cast<float>(max_8bit),
    };
    if (num_tokens == 0) return;

    const dim3 block(device::kWarpThreads, kNumWarps);
    const uint32_t smem_size = kNumWarps * sizeof(float);
    LaunchKernel(num_tokens, block, device.unwrap(), smem_size).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
