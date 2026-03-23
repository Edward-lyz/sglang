#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

struct DSATransferMLAPagesParams {
  const uint64_t* __restrict__ src_ptrs;
  const uint64_t* __restrict__ dst_ptrs;
  const int64_t* __restrict__ src_page_starts;
  const int64_t* __restrict__ dst_page_starts;
  const int32_t* __restrict__ active_pages;
  uint32_t num_layers;
  uint32_t num_pages;
  uint32_t page_size;
  uint32_t token_stride_bytes;
};

constexpr uint32_t kBlockSize = 256;

SGL_DEVICE void copy_token_bytes(
    char* __restrict__ dst,
    const char* __restrict__ src,
    uint32_t num_bytes) {
  uintptr_t src_addr = reinterpret_cast<uintptr_t>(src);
  uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst);
  uint32_t offset = 0;

  while (offset < num_bytes &&
         (((src_addr + offset) & 0x7) != 0 || ((dst_addr + offset) & 0x7) != 0)) {
    dst[offset] = src[offset];
    ++offset;
  }

  for (; offset + sizeof(uint64_t) <= num_bytes; offset += sizeof(uint64_t)) {
    *reinterpret_cast<uint64_t*>(dst + offset) =
        *reinterpret_cast<const uint64_t*>(src + offset);
  }

  for (; offset < num_bytes; ++offset) {
    dst[offset] = src[offset];
  }
}

__global__ void dsa_transfer_mla_pages_kernel(
    const __grid_constant__ DSATransferMLAPagesParams p) {
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t items_per_page = p.num_layers * p.page_size;
  const uint32_t total = p.num_pages * items_per_page;
  if (tid >= total) return;

  const uint32_t page_idx = tid / items_per_page;
  if (p.active_pages[page_idx] == 0) {
    return;
  }

  const uint32_t page_item_idx = tid % items_per_page;
  const uint32_t layer_idx = page_item_idx / p.page_size;
  const uint32_t token_offset = page_item_idx % p.page_size;

  const auto src_token_idx =
      static_cast<uint64_t>(p.src_page_starts[page_idx]) + token_offset;
  const auto dst_token_idx =
      static_cast<uint64_t>(p.dst_page_starts[page_idx]) + token_offset;

  const auto src_base = reinterpret_cast<const char*>(
      static_cast<uintptr_t>(p.src_ptrs[layer_idx]));
  auto dst_base =
      reinterpret_cast<char*>(static_cast<uintptr_t>(p.dst_ptrs[layer_idx]));

  const auto src = src_base + src_token_idx * p.token_stride_bytes;
  auto dst = dst_base + dst_token_idx * p.token_stride_bytes;
  copy_token_bytes(dst, src, p.token_stride_bytes);
}

struct DSATransferMLAPages {
  static void run(
      const tvm::ffi::TensorView src_ptrs,
      const tvm::ffi::TensorView dst_ptrs,
      const tvm::ffi::TensorView src_page_starts,
      const tvm::ffi::TensorView dst_page_starts,
      const tvm::ffi::TensorView active_pages,
      int64_t page_size,
      int64_t token_stride_bytes) {
    using namespace host;

    auto NUM_LAYERS = SymbolicSize{"num_layers"};
    auto NUM_PAGES = SymbolicSize{"num_pages"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({NUM_LAYERS})
        .with_dtype<uint64_t>()
        .with_device(device)
        .verify(src_ptrs)
        .verify(dst_ptrs);
    TensorMatcher({NUM_PAGES})
        .with_dtype<int64_t>()
        .with_device(device)
        .verify(src_page_starts)
        .verify(dst_page_starts);
    TensorMatcher({NUM_PAGES})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(active_pages);

    const auto num_layers = static_cast<uint32_t>(NUM_LAYERS.unwrap());
    const auto num_pages = static_cast<uint32_t>(NUM_PAGES.unwrap());
    if (num_layers == 0 || num_pages == 0) return;

    const auto params = DSATransferMLAPagesParams{
        .src_ptrs = static_cast<const uint64_t*>(src_ptrs.data_ptr()),
        .dst_ptrs = static_cast<const uint64_t*>(dst_ptrs.data_ptr()),
        .src_page_starts =
            static_cast<const int64_t*>(src_page_starts.data_ptr()),
        .dst_page_starts =
            static_cast<const int64_t*>(dst_page_starts.data_ptr()),
        .active_pages = static_cast<const int32_t*>(active_pages.data_ptr()),
        .num_layers = num_layers,
        .num_pages = num_pages,
        .page_size = static_cast<uint32_t>(page_size),
        .token_stride_bytes = static_cast<uint32_t>(token_stride_bytes),
    };

    const auto total = num_layers * num_pages * static_cast<uint32_t>(page_size);
    const auto num_blocks = div_ceil(total, kBlockSize);
    LaunchKernel(num_blocks, kBlockSize, device.unwrap())(
        dsa_transfer_mla_pages_kernel, params);
  }
};

}  // namespace
