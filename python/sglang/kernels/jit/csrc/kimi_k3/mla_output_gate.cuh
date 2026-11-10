// K3 MLA output gate: out = bf16(bf16(x) * bf16(sigmoid(gate))), replacing
// the torch.sigmoid + mul elementwise pair (two launches, two memory passes)
// with one kernel. sigmoid is computed in fp32 and rounded to bf16 before the
// multiply, reproducing the unfused pair's double rounding bit-for-bit.

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck

#include <sgl_kernel/type.cuh>   // For bf16_t, fp32_t, device::cast
#include <sgl_kernel/utils.cuh>  // For LaunchKernel
#include <sgl_kernel/vec.cuh>    // For AlignedVector

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

struct MlaOutputGateParams {
  const bf16_t* __restrict__ x;     // [N] contiguous (flattened [T, H])
  const bf16_t* __restrict__ gate;  // [N] contiguous
  bf16_t* __restrict__ out;         // [N] contiguous
  uint32_t n_vecs;
  // Optional fused DeepGEMM input quant of `out` viewed as [T, H] (nullptr =
  // off): fp8 [T, H] and group-128 UE8M0 bytes at
  // (g / 4) * q_scale_group_stride + token * 4 + g % 4.
  fp8_e4m3_t* __restrict__ q_out;
  uint8_t* __restrict__ q_scale;
  uint32_t q_scale_group_stride;
  uint32_t row_vecs;  // H / 8
};

// Per-token-group FP8 quant of one thread's 8 outputs, bit-identical to
// per_token_group_quant (group 128, UE8M0, column-major) run on `out`: 16
// consecutive threads hold one group; absmax and the scale multiply stay in
// the bf16 domain as in that kernel. All lanes of a half-warp share validity
// (H % 128 == 0), so the shuffle never mixes live and dead lanes.
SGL_DEVICE void quantize_out(const MlaOutputGateParams& params, uint32_t v, const device::AlignedVector<bf16x2_t, 4>& o) {
  using namespace device;
  using deepseek_v4::fp8::cast_to_ue8m0;
  using deepseek_v4::fp8::inv_scale_ue8m0;
  constexpr uint32_t kGroupLanes = 16;
  constexpr float kMaxValue = DTypeTrait<fp8_e4m3_t>::kFloatMax;
  constexpr float kMaxValueInv = 1.f / kMaxValue;
  bf16x2_t amax2 = math::abs(o[0]);
#pragma unroll
  for (uint32_t j = 1; j < 4; ++j) {
    amax2 = math::max(amax2, math::abs(o[j]));
  }
  const auto amax_f2 = cast<fp32x2_t>(warp::reduce_max<kGroupLanes>(amax2));
  const auto amax = math::max(math::max(amax_f2.x, amax_f2.y), 1e-10f);
  const auto exp = cast_to_ue8m0(amax * kMaxValueInv);
  const float quant_scale = inv_scale_ue8m0(exp);
  const auto scale2 = cast<bf16x2_t>(fp32x2_t{quant_scale, quant_scale});
  const auto max_clip = cast<bf16_t>(kMaxValue);
  const auto max_clip2 = bf16x2_t{max_clip, max_clip};
  AlignedVector<fp8x2_e4m3_t, 4> q;
#pragma unroll
  for (uint32_t j = 0; j < 4; ++j) {
    q[j] = static_cast<fp8x2_e4m3_t>(__hmin2(__hmul2(o[j], scale2), max_clip2));
  }
  if (v >= params.n_vecs) return;
  q.store(params.q_out, v);
  if (v % kGroupLanes == 0) {
    const uint32_t token = v / params.row_vecs;
    const uint32_t g = (v % params.row_vecs) / kGroupLanes;
    params.q_scale[(g / 4) * params.q_scale_group_stride + token * 4 + g % 4] = static_cast<uint8_t>(exp);
  }
}

template <int kThreads, bool kUsePDL>
__global__ void mla_output_gate_kernel(const MlaOutputGateParams __grid_constant__ params) {
  using namespace device;

  constexpr int kVecN = 8;
  using vec_bf16_t = AlignedVector<bf16_t, kVecN>;

  const uint32_t v = blockIdx.x * kThreads + threadIdx.x;
  // With the fused quant every lane of a live half-warp must reach the
  // shuffle; whole dead half-warps may leave.
  if (v >= params.n_vecs && (params.q_out == nullptr || (v & ~15u) >= params.n_vecs)) return;
  const bool valid = v < params.n_vecs;

  PDLWaitPrimary<kUsePDL>();

  vec_bf16_t xv, gv, ov;
  if (valid) {
    xv.load(params.x, v);
    gv.load(params.gate, v);
  }
#pragma unroll
  for (int i = 0; i < kVecN; ++i) {
    // Match torch.sigmoid(bf16): fp32 sigmoid, round to bf16, then the bf16
    // multiply upcasts both operands to fp32 and rounds once more.
    const float g = cast<fp32_t>(gv[i]);
    const bf16_t s = cast<bf16_t>(1.0f / (1.0f + expf(-g)));
    ov[i] = cast<bf16_t>(cast<fp32_t>(xv[i]) * cast<fp32_t>(s));
  }
  if (valid) ov.store(params.out, v);
  if (params.q_out != nullptr) {
    quantize_out(params, v, reinterpret_cast<const AlignedVector<bf16x2_t, 4>&>(ov));
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int kThreads, bool kUsePDL>
struct MlaOutputGateKernel {
  static constexpr auto kernel = mla_output_gate_kernel<kThreads, kUsePDL>;

  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView gate, const tvm::ffi::TensorView out) {
    launch(x, gate, out, nullptr, nullptr, 0, 0);
  }

  // run() plus the fused FP8 quant of `out` viewed as [T, H]: q_out fp8
  // [T, H] contiguous, q_scale the int32 [T, H / 512] column-major TMA-aligned
  // UE8M0 view (token stride 1).
  static void run_quant(
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView gate,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView q_out,
      const tvm::ffi::TensorView q_scale) {
    using namespace host;
    auto T_ = SymbolicSize{"num_tokens"};
    auto H_ = SymbolicSize{"hidden"};
    auto G_ = SymbolicSize{"num_scale_packs"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({T_, H_}).with_strides({-1, 1}).with_dtype<fp8_e4m3_t>().with_device(device).verify(q_out);
    TensorMatcher({T_, G_}).with_strides({1, -1}).with_dtype<int32_t>().with_device(device).verify(q_scale);
    const auto H = static_cast<uint32_t>(H_.unwrap());
    RuntimeCheck(q_out.stride(0) == H, "q_out must be contiguous");
    RuntimeCheck(H % 512 == 0 && G_.unwrap() * 512 == H, "H must be a multiple of 512 (aligned ue8m0 packs)");
    RuntimeCheck(static_cast<int64_t>(T_.unwrap()) * H == x.size(0), "q_out must cover out");
    launch(
        x,
        gate,
        out,
        static_cast<fp8_e4m3_t*>(q_out.data_ptr()),
        static_cast<uint8_t*>(q_scale.data_ptr()),
        static_cast<uint32_t>(q_scale.stride(1) * 4),
        H / 8);
  }

  static void launch(
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView gate,
      const tvm::ffi::TensorView out,
      fp8_e4m3_t* q_out,
      uint8_t* q_scale,
      uint32_t q_scale_group_stride,
      uint32_t row_vecs) {
    using namespace host;

    auto N_ = SymbolicSize{"numel"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({N_}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({N_}).with_dtype<bf16_t>().with_device(device).verify(gate);
    TensorMatcher({N_}).with_dtype<bf16_t>().with_device(device).verify(out);

    const auto N = static_cast<uint32_t>(N_.unwrap());
    RuntimeCheck(N % 8 == 0, "numel must be divisible by 8");
    if (N == 0) return;

    const auto params = MlaOutputGateParams{
        .x = static_cast<const bf16_t*>(x.data_ptr()),
        .gate = static_cast<const bf16_t*>(gate.data_ptr()),
        .out = static_cast<bf16_t*>(out.data_ptr()),
        .n_vecs = N / 8,
        .q_out = q_out,
        .q_scale = q_scale,
        .q_scale_group_stride = q_scale_group_stride,
        .row_vecs = row_vecs,
    };
    const uint32_t n_blocks = (params.n_vecs + kThreads - 1) / kThreads;
    LaunchKernel(n_blocks, kThreads, device.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
