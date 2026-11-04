// Warm L2 with a weight that a later kernel streams, issued while the device
// waits on communication. One cp.async.bulk.prefetch.L2 per chunk; the kernel
// only issues the prefetches and exits, it does not wait for them.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

constexpr uint32_t kChunkBytes = 16384;  // per bulk prefetch (multiple of 16)

__global__ void l2_prefetch_kernel(const uint8_t* __restrict__ base, uint64_t bytes) {
  const uint64_t num_chunks = (bytes + kChunkBytes - 1) / kChunkBytes;
  for (uint64_t c = blockIdx.x * blockDim.x + threadIdx.x; c < num_chunks; c += gridDim.x * blockDim.x) {
    const uint64_t off = c * kChunkBytes;
    const uint32_t size = static_cast<uint32_t>(bytes - off < kChunkBytes ? bytes - off : kChunkBytes);
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(base + off), "r"(size) : "memory");
  }
}

struct L2PrefetchKernel {
  static void run(const tvm::ffi::TensorView tensor) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto N = SymbolicSize{"bytes"};
    TensorMatcher({N}).with_dtype<uint8_t>().with_device(device).verify(tensor);
    const uint64_t bytes = N.unwrap();
    RuntimeCheck(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0, "l2_prefetch: base must be 16B aligned");
    RuntimeCheck(bytes % 16 == 0, "l2_prefetch: size must be a multiple of 16 bytes");
    if (bytes == 0) return;
    const uint64_t chunks = (bytes + kChunkBytes - 1) / kChunkBytes;
    const uint32_t threads = 128;
    const uint32_t blocks = static_cast<uint32_t>((chunks + threads - 1) / threads);
    LaunchKernel(blocks, threads, device.unwrap())(l2_prefetch_kernel, static_cast<const uint8_t*>(tensor.data_ptr()), bytes);
  }
};

}  // namespace
