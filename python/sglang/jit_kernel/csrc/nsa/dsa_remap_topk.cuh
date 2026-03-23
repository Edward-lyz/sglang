// DSA remap topk: convert logical topk indices to physical device addresses.
//
// For each topk entry:
//   - If gpu_slot[req, page] >= 0: remapped = cache_slot + page_offset
//   - Else if host_slot[req, page] < 0: remapped = req_to_token[req, logical_pos]
//   - Else: remapped = -1 because host-backed pages must be materialized first
//   - Invalid entries: remapped = -1

#include <sgl_kernel/tensor.h>   // TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>    // RuntimeCheck, div_ceil
#include <sgl_kernel/utils.cuh>  // LaunchKernel, SGL_DEVICE

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

struct DSARemapTopkParams {
  const int32_t* __restrict__ topk;              // [bs, topk_dim]
  const int32_t* __restrict__ req_pool_indices;  // [bs]
  const int32_t* __restrict__ seq_lens;          // [bs]
  const int64_t* __restrict__ gpu_slot;          // [max_reqs, max_pages]
  const int64_t* __restrict__ host_slot;         // [max_reqs, max_pages]
  const int32_t* __restrict__ req_to_token;      // [max_reqs, max_context_len]
  int32_t* __restrict__ remapped;                // [bs, topk_dim]
  uint32_t bs;
  uint32_t topk_dim;
  uint32_t page_size;
  uint32_t max_context_len;
  uint32_t gpu_slot_stride_0;
  uint32_t req_to_token_stride_0;
  uint32_t host_slot_stride_0;
};

constexpr uint32_t kBlockSize = 256;

template <bool kUsePDL>
__global__ void dsa_remap_topk_kernel(
    const __grid_constant__ DSARemapTopkParams p) {
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
    p.remapped[tid] = -1;
    return;
  }

  const uint32_t logical_page = static_cast<uint32_t>(logical) / p.page_size;
  const uint32_t page_offset = static_cast<uint32_t>(logical) % p.page_size;

  int64_t cache_slot = p.gpu_slot[
      static_cast<uint32_t>(req_idx) * p.gpu_slot_stride_0 + logical_page];

  if (cache_slot >= 0) {
    // Page is in GPU sparse cache: return cache_slot + page_offset
    p.remapped[tid] = static_cast<int32_t>(cache_slot + page_offset);
  } else {
    const int64_t host_slot = p.host_slot[
        static_cast<uint32_t>(req_idx) * p.host_slot_stride_0 + logical_page];
    if (host_slot < 0) {
      // This page still lives in the main KV pool (e.g. decode tail).
      p.remapped[tid] = p.req_to_token[
          static_cast<uint32_t>(req_idx) * p.req_to_token_stride_0 +
          static_cast<uint32_t>(logical)];
    } else {
      // Host-backed pages must have been materialized into sparse cache already.
      p.remapped[tid] = -1;
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL>
struct DSARemapTopk {
  static constexpr auto kernel = dsa_remap_topk_kernel<kUsePDL>;

  static void run(
      const tvm::ffi::TensorView topk,
      const tvm::ffi::TensorView req_pool_indices,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView gpu_slot,
      const tvm::ffi::TensorView host_slot,
      const tvm::ffi::TensorView req_to_token,
      const tvm::ffi::TensorView remapped,
      int64_t page_size,
      int64_t max_context_len,
      int64_t gpu_slot_stride_0,
      int64_t req_to_token_stride_0,
      int64_t host_slot_stride_0) {
    using namespace host;

    auto BS = SymbolicSize{"bs"};
    auto TOPK = SymbolicSize{"topk"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({BS, TOPK}).with_dtype<int32_t>().with_device(device).verify(topk);
    TensorMatcher({BS}).with_dtype<int32_t>().with_device(device).verify(req_pool_indices);
    TensorMatcher({BS}).with_dtype<int32_t>().with_device(device).verify(seq_lens);
    TensorMatcher({BS, TOPK}).with_dtype<int32_t>().with_device(device).verify(remapped);

    const auto bs = static_cast<uint32_t>(BS.unwrap());
    const auto topk_dim = static_cast<uint32_t>(TOPK.unwrap());
    const auto total = bs * topk_dim;
    if (total == 0) return;

    const auto params = DSARemapTopkParams{
        .topk = static_cast<const int32_t*>(topk.data_ptr()),
        .req_pool_indices = static_cast<const int32_t*>(req_pool_indices.data_ptr()),
        .seq_lens = static_cast<const int32_t*>(seq_lens.data_ptr()),
        .gpu_slot = static_cast<const int64_t*>(gpu_slot.data_ptr()),
        .host_slot = static_cast<const int64_t*>(host_slot.data_ptr()),
        .req_to_token = static_cast<const int32_t*>(req_to_token.data_ptr()),
        .remapped = static_cast<int32_t*>(const_cast<void*>(remapped.data_ptr())),
        .bs = bs,
        .topk_dim = topk_dim,
        .page_size = static_cast<uint32_t>(page_size),
        .max_context_len = static_cast<uint32_t>(max_context_len),
        .gpu_slot_stride_0 = static_cast<uint32_t>(gpu_slot_stride_0),
        .req_to_token_stride_0 = static_cast<uint32_t>(req_to_token_stride_0),
        .host_slot_stride_0 = static_cast<uint32_t>(host_slot_stride_0),
    };

    const auto num_blocks = div_ceil(total, kBlockSize);
    LaunchKernel(num_blocks, kBlockSize, device.unwrap())
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
