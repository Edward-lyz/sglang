// DSA miss detection: scan topk, check gpu_slot, record misses.
//
// Pure read on gpu_slot (no CAS, no exchange). Writes only to workspace
// miss buffers via atomicAdd on miss_count. CUDA graph safe.

#include <sgl_kernel/tensor.h>   // TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>    // RuntimeCheck, div_ceil
#include <sgl_kernel/utils.cuh>  // LaunchKernel, SGL_DEVICE

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

struct DSADetectMissesParams {
  const int32_t* __restrict__ topk;              // [bs, topk_dim]
  const int32_t* __restrict__ req_pool_indices;  // [bs]
  const int32_t* __restrict__ seq_lens;          // [bs]
  const int64_t* __restrict__ gpu_slot;          // [max_reqs, max_pages]
  const int64_t* __restrict__ host_slot;         // [max_reqs, max_pages]
  int32_t* __restrict__ miss_req_indices;        // [max_misses] output
  int32_t* __restrict__ miss_logical_pages;      // [max_misses] output
  int64_t* __restrict__ miss_host_slots;         // [max_misses] output
  int32_t* __restrict__ miss_count;              // [1] output (atomic counter)
  uint32_t bs;
  uint32_t topk_dim;
  uint32_t page_size;
  uint32_t max_context_len;
  uint32_t max_misses;
  uint32_t gpu_slot_stride_0;
  uint32_t host_slot_stride_0;
};

constexpr uint32_t kBlockSize = 256;

template <bool kUsePDL>
__global__ void dsa_detect_misses_kernel(
    const __grid_constant__ DSADetectMissesParams p) {
  using namespace device;

  PDLWaitPrimary<kUsePDL>();

  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t total = p.bs * p.topk_dim;
  if (tid >= total) return;

  const uint32_t row = tid / p.topk_dim;
  const uint32_t col = tid % p.topk_dim;

  const int32_t logical = p.topk[row * p.topk_dim + col];
  const int32_t req_idx = p.req_pool_indices[row];
  const int32_t seq_len = p.seq_lens[row];

  // Invalid entry
  if (logical < 0 || logical >= static_cast<int32_t>(p.max_context_len) ||
      logical >= seq_len) {
    return;
  }

  const uint32_t logical_page = static_cast<uint32_t>(logical) / p.page_size;
  const uint32_t slot_offset =
      static_cast<uint32_t>(req_idx) * p.gpu_slot_stride_0 + logical_page;

  // Read gpu_slot (pure read, no CAS)
  int64_t gpu_slot_val = p.gpu_slot[slot_offset];
  if (gpu_slot_val >= 0) {
    return;  // Already cached, not a miss
  }

  // Read host_slot; skip if page has no host mapping (e.g. local-only request)
  int64_t host_slot_val =
      p.host_slot[static_cast<uint32_t>(req_idx) * p.host_slot_stride_0 +
                   logical_page];
  if (host_slot_val < 0) {
    return;  // No host data available, cannot prefetch
  }

  // Miss with valid host slot — record to workspace.
  int32_t idx = atomicAdd(p.miss_count, 1);
  if (static_cast<uint32_t>(idx) >= p.max_misses) {
    return;  // Overflow guard
  }
  p.miss_req_indices[idx] = req_idx;
  p.miss_logical_pages[idx] = static_cast<int32_t>(logical_page);
  p.miss_host_slots[idx] = host_slot_val;

  PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL>
struct DSADetectMisses {
  static constexpr auto kernel = dsa_detect_misses_kernel<kUsePDL>;

  static void run(
      const tvm::ffi::TensorView topk,
      const tvm::ffi::TensorView req_pool_indices,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView gpu_slot,
      const tvm::ffi::TensorView host_slot,
      const tvm::ffi::TensorView miss_req_indices,
      const tvm::ffi::TensorView miss_logical_pages,
      const tvm::ffi::TensorView miss_host_slots,
      const tvm::ffi::TensorView miss_count,
      int64_t page_size,
      int64_t max_context_len,
      int64_t max_misses,
      int64_t gpu_slot_stride_0,
      int64_t host_slot_stride_0) {
    using namespace host;

    auto BS = SymbolicSize{"bs"};
    auto TOPK = SymbolicSize{"topk"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({BS, TOPK}).with_dtype<int32_t>().with_device(device).verify(topk);
    TensorMatcher({BS}).with_dtype<int32_t>().with_device(device).verify(req_pool_indices);
    TensorMatcher({BS}).with_dtype<int32_t>().with_device(device).verify(seq_lens);

    const auto bs = static_cast<uint32_t>(BS.unwrap());
    const auto topk_dim = static_cast<uint32_t>(TOPK.unwrap());
    const auto total = bs * topk_dim;
    if (total == 0) return;

    const auto params = DSADetectMissesParams{
        .topk = static_cast<const int32_t*>(topk.data_ptr()),
        .req_pool_indices = static_cast<const int32_t*>(req_pool_indices.data_ptr()),
        .seq_lens = static_cast<const int32_t*>(seq_lens.data_ptr()),
        .gpu_slot = static_cast<const int64_t*>(gpu_slot.data_ptr()),
        .host_slot = static_cast<const int64_t*>(host_slot.data_ptr()),
        .miss_req_indices = static_cast<int32_t*>(const_cast<void*>(miss_req_indices.data_ptr())),
        .miss_logical_pages = static_cast<int32_t*>(const_cast<void*>(miss_logical_pages.data_ptr())),
        .miss_host_slots = static_cast<int64_t*>(const_cast<void*>(miss_host_slots.data_ptr())),
        .miss_count = static_cast<int32_t*>(const_cast<void*>(miss_count.data_ptr())),
        .bs = bs,
        .topk_dim = topk_dim,
        .page_size = static_cast<uint32_t>(page_size),
        .max_context_len = static_cast<uint32_t>(max_context_len),
        .max_misses = static_cast<uint32_t>(max_misses),
        .gpu_slot_stride_0 = static_cast<uint32_t>(gpu_slot_stride_0),
        .host_slot_stride_0 = static_cast<uint32_t>(host_slot_stride_0),
    };

    const auto num_blocks = div_ceil(total, kBlockSize);
    LaunchKernel(num_blocks, kBlockSize, device.unwrap())
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
