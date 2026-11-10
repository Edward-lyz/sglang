#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <cstddef>
#include <cstdint>

namespace {

// FP4 E2M1 quantization kernel
// Input: [M, 128] bfloat16, Output: [M, 64] int8 (packed), [M, 1] int32 (scale)
// gran_k=32, 4 groups per row, UE8M0 scale format
//
// Design: 1 warp (32 threads) per row, 8 threads per group
// Thread layout within warp:
//   Group 0: lane 0-7   -> elements 0-31
//   Group 1: lane 8-15  -> elements 32-63
//   Group 2: lane 16-23 -> elements 64-95
//   Group 3: lane 24-31 -> elements 96-127
// Each thread handles 4 consecutive elements, produces 2 packed bytes.

static constexpr int kGranK = 32;
static constexpr int kNumGroups = 4;  // 128 / 32
static constexpr int kRowDim = 128;
static constexpr int kThreadsPerGroup = 8;  // 32 / 4 groups
static constexpr int kElemsPerThread = 4;   // 32 / 8 threads

// FP4 E2M1 midpoint boundaries: {0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0}
// Values: {0, 0.5, 1, 1.5, 2, 3, 4, 6}
__device__ __forceinline__ int fp4_bucketize(float ax) {
  int idx = 0;
  idx += (ax > 0.25f);
  idx += (ax > 0.75f);
  idx += (ax > 1.25f);
  idx += (ax > 1.75f);
  idx += (ax > 2.5f);
  idx += (ax > 3.5f);
  idx += (ax > 5.0f);
  return idx;
}

__device__ __forceinline__ uint8_t ceil_to_ue8m0_exp(float amax) {
  // ceil_to_ue8m0: round scale up to nearest power of 2
  // scale = max(amax / 6.0, 1e-4), matching deep_gemm's per_token_cast_to_fp4.
  float raw_sf = fmaxf(amax / 6.0f, 1.0e-4f);
  uint32_t bits = __float_as_uint(raw_sf);
  uint32_t mantissa = bits & 0x7FFFFFu;
  uint32_t exp = (bits >> 23) & 0xFFu;
  // If mantissa != 0, round exponent up
  exp += (mantissa != 0) ? 1u : 0u;
  // Clamp to [1, 254]
  exp = max(exp, 1u);
  exp = min(exp, 254u);
  return static_cast<uint8_t>(exp);
}

__device__ __forceinline__ float ue8m0_exp_to_float(uint8_t exp) {
  uint32_t bits = static_cast<uint32_t>(exp) << 23;
  return __uint_as_float(bits);
}

template <int kWarpsPerBlock>
__global__ void per_token_group_quant_fp4_kernel(
    const __nv_bfloat16* __restrict__ input,  // [M, 128]
    int8_t* __restrict__ output_packed,       // [M, 64]
    int32_t* __restrict__ output_sf,          // [M, 1]
    const int64_t M) {
  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  const int64_t row_idx = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_id;

  if (row_idx >= M) return;

  // Determine which group this lane belongs to and position within group
  const int group_id = lane_id / kThreadsPerGroup;       // 0-3
  const int lane_in_group = lane_id % kThreadsPerGroup;  // 0-7

  // Load 4 consecutive elements for this thread
  const int elem_offset = group_id * kGranK + lane_in_group * kElemsPerThread;
  const int64_t base_addr = row_idx * kRowDim + elem_offset;

  float vals[kElemsPerThread];
#pragma unroll
  for (int i = 0; i < kElemsPerThread; ++i) {
    vals[i] = __bfloat162float(input[base_addr + i]);
  }

  // Compute local absolute max
  float local_amax = 0.0f;
#pragma unroll
  for (int i = 0; i < kElemsPerThread; ++i) {
    local_amax = fmaxf(local_amax, fabsf(vals[i]));
  }

  // Reduce amax within 8-thread group using warp shuffles
  // Mask for 8-thread subgroup
  const unsigned group_mask = 0xFFu << (group_id * kThreadsPerGroup);
  local_amax = fmaxf(local_amax, __shfl_xor_sync(group_mask, local_amax, 4));
  local_amax = fmaxf(local_amax, __shfl_xor_sync(group_mask, local_amax, 2));
  local_amax = fmaxf(local_amax, __shfl_xor_sync(group_mask, local_amax, 1));

  // Clamp to avoid zero
  local_amax = fmaxf(local_amax, 1e-4f);

  // Compute UE8M0 scale
  uint8_t exp_byte = ceil_to_ue8m0_exp(local_amax);
  float sf_float = ue8m0_exp_to_float(exp_byte);
  float inv_sf = 1.0f / sf_float;

  // Quantize to FP4 E2M1 and pack
  int codes[kElemsPerThread];
#pragma unroll
  for (int i = 0; i < kElemsPerThread; ++i) {
    float x_scaled = vals[i] * inv_sf;
    float ax = fminf(fabsf(x_scaled), 6.0f);
    int mag_idx = fp4_bucketize(ax);
    int sign_bit = (x_scaled < 0.0f && mag_idx != 0) ? 1 : 0;
    codes[i] = (mag_idx & 0x7) | (sign_bit << 3);
  }

  // Pack: even element = low nibble, odd element = high nibble
  // Each thread produces 2 packed bytes from 4 codes
  int8_t packed_bytes[kElemsPerThread / 2];
#pragma unroll
  for (int i = 0; i < kElemsPerThread / 2; ++i) {
    packed_bytes[i] = static_cast<int8_t>(
        (codes[i * 2] & 0x0F) | ((codes[i * 2 + 1] & 0x0F) << 4));
  }

  // Store packed output: each thread writes 2 bytes
  // Output layout: row_idx * 64 + (elem_offset / 2) for this thread's 2 bytes
  const int64_t out_base = row_idx * (kRowDim / 2) + elem_offset / 2;
#pragma unroll
  for (int i = 0; i < kElemsPerThread / 2; ++i) {
    output_packed[out_base + i] = packed_bytes[i];
  }

  // Store UE8M0 scale: 4 groups -> 4 bytes packed into 1 int32 (little-endian).
  // Every lane takes part in the shuffles (a full-warp mask requires it); lane
  // 0 then assembles and stores the packed int32. All 8 lanes of a group hold
  // the same group exponent after the xor reduction above.
  uint32_t my_exp = static_cast<uint32_t>(exp_byte);
  uint32_t exp_from_g0 = __shfl_sync(0xFFFFFFFF, my_exp, 0);
  uint32_t exp_from_g1 = __shfl_sync(0xFFFFFFFF, my_exp, 8);
  uint32_t exp_from_g2 = __shfl_sync(0xFFFFFFFF, my_exp, 16);
  uint32_t exp_from_g3 = __shfl_sync(0xFFFFFFFF, my_exp, 24);
  if (lane_id == 0) {
    uint32_t packed_sf = (exp_from_g0 & 0xFF) | ((exp_from_g1 & 0xFF) << 8) |
                         ((exp_from_g2 & 0xFF) << 16) | ((exp_from_g3 & 0xFF) << 24);
    output_sf[row_idx] = static_cast<int32_t>(packed_sf);
  }
}

template <typename DType>
void per_token_group_quant_fp4(
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView output_packed,
    tvm::ffi::TensorView output_sf) {
  using namespace host;

  auto device = SymbolicDevice{};
  auto M = SymbolicSize{"M"};
  auto N = SymbolicSize{"N"};
  auto N_half = SymbolicSize{"N_half"};
  device.set_options<kDLCUDA>();

  TensorMatcher({M, N}).with_dtype<DType>().with_device(device).verify(input);
  TensorMatcher({M, N_half}).with_dtype<int8_t>().with_device(device).verify(output_packed);
  TensorMatcher({M, 1}).with_dtype<int32_t>().with_device(device).verify(output_sf);

  const auto m = M.unwrap();
  const auto n = N.unwrap();
  RuntimeCheck(n == 128, "N must be 128 for FP4 quantization with gran_k=32");
  RuntimeCheck(N_half.unwrap() == 64, "output_packed N must be 64");

  constexpr int kWarpsPerBlock = 8;
  constexpr int kThreadsPerBlock = kWarpsPerBlock * 32;  // 256

  const int num_blocks = static_cast<int>((m + kWarpsPerBlock - 1) / kWarpsPerBlock);

  LaunchKernel(dim3(num_blocks), dim3(kThreadsPerBlock), device.unwrap())(
      per_token_group_quant_fp4_kernel<kWarpsPerBlock>,
      static_cast<const __nv_bfloat16*>(input.data_ptr()),
      static_cast<int8_t*>(output_packed.data_ptr()),
      static_cast<int32_t*>(output_sf.data_ptr()),
      m);
}

}  // namespace
