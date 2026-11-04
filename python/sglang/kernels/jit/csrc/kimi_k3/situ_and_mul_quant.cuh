#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

// ----------------------------------------------------------------
// Dense SiTU-and-mul fused with the per-token-group FP8 quant that feeds the
// DeepGEMM down projection (UE8M0 scales, column-major, packed 4 per int32).
//
// Bit-identical to situ_and_mul (bf16 out) followed by per_token_group_quant
// (fp8_e4m3, group 128, scale_ue8m0, column-major TMA-aligned): the activation
// is computed with the same formula and rounded to bf16 before the absmax and
// the bf16 scale multiply, exactly as the two kernels do back to back. The
// module is built without --use_fast_math so tanhf matches situ_and_mul.
//
// Input: bf16 [T, 2*D] (gate = [:, :D], up = [:, D:]), rows may be strided.
// Output: fp8 [T, D] contiguous; int32 scale view [T, D/128/4] with token
// stride 1 (the transposed TMA-aligned buffer).
// ----------------------------------------------------------------

constexpr uint32_t kGroupSize = 128;
constexpr uint32_t kElemsPerLane = 16;                       // 32 B of bf16 per lane
constexpr uint32_t kNumLanes = kGroupSize / kElemsPerLane;  // 8 lanes per group
constexpr uint32_t kBlockSize = 256;

struct SituMulQuantParams {
  const bf16_t* __restrict__ input;
  fp8_e4m3_t* __restrict__ output;
  uint8_t* __restrict__ scale;  // packed ue8m0 bytes
  float beta;
  float inv_beta;
  float linear_beta;
  float inv_linear_beta;
  uint32_t hidden_dim;    // D
  uint32_t num_tokens;    // T
  uint32_t num_groups;    // D / 128
  int64_t input_stride;   // elements per input row
  uint32_t scale_group_stride;  // bytes between packed-group rows
};

template <bool kHasLinearBeta>
SGL_DEVICE bf16_t situ_mul(const bf16_t gate, const bf16_t up, const SituMulQuantParams& p) {
  using namespace device;
  const float g = cast<fp32_t>(gate);
  const float u = cast<fp32_t>(up);
  const float gate_out = p.beta * tanhf(g * p.inv_beta) * (1.0f / (1.0f + __expf(-g)));
  float up_out;
  if constexpr (kHasLinearBeta) {
    up_out = p.linear_beta * tanhf(u * p.inv_linear_beta);
  } else {
    up_out = u;
  }
  return cast<bf16_t>(gate_out * up_out);
}

template <bool kHasLinearBeta, bool kUsePDL>
__global__ __launch_bounds__(kBlockSize) void situ_mul_quant_kernel(const __grid_constant__ SituMulQuantParams p) {
  using namespace device;
  using deepseek_v4::fp8::cast_to_ue8m0;
  using deepseek_v4::fp8::inv_scale_ue8m0;
  using vec_t = AlignedVector<bf16x2_t, 4>;  // 16 B
  using out_vec_t = AlignedVector<fp8x2_e4m3_t, kElemsPerLane / 2>;
  constexpr float kMaxValue = DTypeTrait<fp8_e4m3_t>::kFloatMax;
  constexpr float kMaxValueInv = 1.f / kMaxValue;
  constexpr uint32_t kWorkPerWarp = kWarpThreads / kNumLanes;

  const auto global_tid = blockIdx.x * kBlockSize + threadIdx.x;
  const auto total_work = p.num_tokens * p.num_groups;
  if ((global_tid / kWarpThreads) * kWorkPerWarp >= total_work) return;
  // the last partial warp duplicates the tail work (identical-byte stores)
  const auto work_id = min(global_tid / kNumLanes, total_work - 1);
  const auto lane_id = threadIdx.x % kNumLanes;
  const auto token_idx = work_id / p.num_groups;
  const auto group_idx = work_id % p.num_groups;

  const bf16_t* row = p.input + token_idx * p.input_stride;
  const uint32_t col = group_idx * kGroupSize + lane_id * kElemsPerLane;

  PDLWaitPrimary<kUsePDL>();

  vec_t gate[2], up[2];
  gate[0].load(row + col, 0);
  gate[1].load(row + col, 1);
  up[0].load(row + p.hidden_dim + col, 0);
  up[1].load(row + p.hidden_dim + col, 1);

  bf16x2_t act[kElemsPerLane / 2];
#pragma unroll
  for (uint32_t i = 0; i < kElemsPerLane / 2; ++i) {
    const bf16x2_t g2 = gate[i / 4][i % 4];
    const bf16x2_t u2 = up[i / 4][i % 4];
    act[i] = bf16x2_t{situ_mul<kHasLinearBeta>(g2.x, u2.x, p), situ_mul<kHasLinearBeta>(g2.y, u2.y, p)};
  }

  // absmax in the packed bf16 domain (abs/max of bf16 values are exact)
  bf16x2_t local_amax2 = math::abs(act[0]);
#pragma unroll
  for (uint32_t i = 1; i < kElemsPerLane / 2; ++i) {
    local_amax2 = math::max(local_amax2, math::abs(act[i]));
  }
  const auto amax2 = cast<fp32x2_t>(warp::reduce_max<kNumLanes>(local_amax2));
  const auto amax = math::max(math::max(amax2.x, amax2.y), 1e-10f);
  const auto exp = cast_to_ue8m0(amax * kMaxValueInv);
  const float quant_scale = inv_scale_ue8m0(exp);
  const auto scale2 = cast<bf16x2_t>(fp32x2_t{quant_scale, quant_scale});
  const auto max_clip = cast<bf16_t>(kMaxValue);
  const auto max_clip2 = bf16x2_t{max_clip, max_clip};

  out_vec_t out;
#pragma unroll
  for (uint32_t i = 0; i < kElemsPerLane / 2; ++i) {
    out[i] = static_cast<fp8x2_e4m3_t>(__hmin2(__hmul2(act[i], scale2), max_clip2));
  }

  PDLTriggerSecondary<kUsePDL>();

  out.store(p.output + static_cast<uint64_t>(token_idx) * p.hidden_dim + col, 0);
  if (lane_id == 0) {
    p.scale[(group_idx / 4) * p.scale_group_stride + token_idx * 4 + group_idx % 4] = static_cast<uint8_t>(exp);
  }
}

template <bool kUsePDL>
struct SituMulQuantKernel {
  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView output_q,
      const tvm::ffi::TensorView output_s,
      const double beta,
      const double linear_beta,
      const bool has_linear_beta) {
    using namespace host;

    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    auto N = SymbolicSize{"num_tokens"};
    auto D = SymbolicSize{"hidden_dim"};
    auto D2 = SymbolicSize{"hidden_dim x 2"};
    auto G = SymbolicSize{"num_groups // 4"};

    TensorMatcher({N, D2}).with_strides({-1, 1}).with_dtype<bf16_t>().with_device(device_).verify(input);
    TensorMatcher({N, D}).with_strides({-1, 1}).with_dtype<fp8_e4m3_t>().with_device(device_).verify(output_q);
    TensorMatcher({N, G}).with_strides({1, -1}).with_dtype<int32_t>().with_device(device_).verify(output_s);

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto hidden_dim = static_cast<uint32_t>(D.unwrap());
    RuntimeCheck(D2.unwrap() == 2 * D.unwrap(), "input width must be 2 * hidden_dim");
    RuntimeCheck(output_q.stride(0) == D.unwrap(), "output must be contiguous");
    RuntimeCheck(hidden_dim % (kGroupSize * 4) == 0, "hidden_dim must be a multiple of 512 (aligned ue8m0 packs)");
    RuntimeCheck(G.unwrap() * 4 * kGroupSize == hidden_dim, "invalid scale width");
    RuntimeCheck(reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0, "input must be 16B-aligned");
    RuntimeCheck(input.stride(0) % 8 == 0, "input row stride must keep rows 16B-aligned");
    if (num_tokens == 0) return;

    const float beta_f = static_cast<float>(beta);
    const float linear_beta_f = static_cast<float>(linear_beta);
    const auto params = SituMulQuantParams{
        .input = static_cast<const bf16_t*>(input.data_ptr()),
        .output = static_cast<fp8_e4m3_t*>(output_q.data_ptr()),
        .scale = static_cast<uint8_t*>(output_s.data_ptr()),
        .beta = beta_f,
        .inv_beta = 1.0f / beta_f,
        .linear_beta = linear_beta_f,
        .inv_linear_beta = linear_beta_f != 0.0f ? 1.0f / linear_beta_f : 0.0f,
        .hidden_dim = hidden_dim,
        .num_tokens = num_tokens,
        .num_groups = hidden_dim / kGroupSize,
        .input_stride = input.stride(0),
        .scale_group_stride = static_cast<uint32_t>(output_s.stride(1) * 4),
    };
    const int64_t total_threads = int64_t{num_tokens} * params.num_groups * kNumLanes;
    const auto num_blocks = static_cast<uint32_t>(div_ceil(total_threads, int64_t{kBlockSize}));
    const auto kernel = has_linear_beta ? situ_mul_quant_kernel<true, kUsePDL> : situ_mul_quant_kernel<false, kUsePDL>;
    LaunchKernel(num_blocks, kBlockSize, device_.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
