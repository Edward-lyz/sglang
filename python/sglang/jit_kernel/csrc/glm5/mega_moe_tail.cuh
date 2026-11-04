// Fused Mega-MoE tail operator for GLM5 MoE.
//
// Fuses the two trailing elementwise ops on the routed-experts output `y`
// (bfloat16, shape [num_tokens, hidden_size]) at the end of the Mega-MoE
// path:
//
//   routed output scaling: y.mul_(routed_scaling_factor)
//   shared expert merge:   y.add_(shared_output)
//
// Compute model: y = y * alpha + (has_shared ? shared : 0). When alpha == 1
// the multiply degenerates to identity; when shared is null the add is skipped.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <cuda_bf16.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

constexpr uint32_t kThreadsPerBlock = 256;
constexpr uint32_t kElemsPerThread = 8;  // 8 bf16 = 16B = float4 vector load.

struct MegaMoETailParams {
  bf16_t* __restrict__ y;
  const bf16_t* __restrict__ shared;
  float alpha;
  uint64_t num_elems;
};

template <bool kHasShared>
__global__ void mega_moe_tail_kernel(const MegaMoETailParams __grid_constant__ params) {
  const auto& [y, shared, alpha, num_elems] = params;

  const uint64_t base = (static_cast<uint64_t>(blockIdx.x) * kThreadsPerBlock + threadIdx.x) * kElemsPerThread;
  if (base >= num_elems) return;

  const bool full = (base + kElemsPerThread) <= num_elems;
  if (full) {
    float4 yv = *reinterpret_cast<const float4*>(y + base);
    bf16_t* yh = reinterpret_cast<bf16_t*>(&yv);
    float4 sv;
    const bf16_t* sh = nullptr;
    if constexpr (kHasShared) {
      sv = *reinterpret_cast<const float4*>(shared + base);
      sh = reinterpret_cast<const bf16_t*>(&sv);
    }
#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
      float v = __bfloat162float(yh[i]) * alpha;
      if constexpr (kHasShared) {
        v += __bfloat162float(sh[i]);
      }
      yh[i] = __float2bfloat16_rn(v);
    }
    *reinterpret_cast<float4*>(y + base) = yv;
  } else {
    for (uint64_t i = base; i < num_elems; ++i) {
      float v = __bfloat162float(y[i]) * alpha;
      if constexpr (kHasShared) {
        v += __bfloat162float(shared[i]);
      }
      y[i] = __float2bfloat16_rn(v);
    }
  }
}

}  // namespace

class MegaMoETailKernel {
 public:
  static void run(
      tvm::ffi::TensorView y,
      tvm::ffi::Optional<tvm::ffi::TensorView> shared,
      double alpha) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto H = SymbolicSize{"hidden"};
    SymbolicDevice device;

    TensorMatcher({N, H}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(y);

    const bf16_t* shared_ptr = nullptr;
    if (shared.has_value()) {
      auto s = shared.value();
      TensorMatcher({N, H}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(s);
      shared_ptr = static_cast<const bf16_t*>(s.data_ptr());
    }

    const uint64_t num_elems = static_cast<uint64_t>(N.unwrap()) * static_cast<uint64_t>(H.unwrap());
    if (num_elems == 0) return;

    const auto params = MegaMoETailParams{
        .y = static_cast<bf16_t*>(y.data_ptr()),
        .shared = shared_ptr,
        .alpha = static_cast<float>(alpha),
        .num_elems = num_elems,
    };

    const uint64_t total_threads = (num_elems + kElemsPerThread - 1) / kElemsPerThread;
    const uint32_t grid_size = static_cast<uint32_t>((total_threads + kThreadsPerBlock - 1) / kThreadsPerBlock);

    if (shared_ptr != nullptr) {
      LaunchKernel(grid_size, kThreadsPerBlock, device.unwrap())(mega_moe_tail_kernel<true>, params);
    } else {
      LaunchKernel(grid_size, kThreadsPerBlock, device.unwrap())(mega_moe_tail_kernel<false>, params);
    }
  }
};
