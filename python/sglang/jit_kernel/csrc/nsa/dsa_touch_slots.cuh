#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

struct DSATouchSlotsParams {
  const int32_t* __restrict__ token_indices;
  const int32_t* __restrict__ dense_page_to_slot;
  int64_t* __restrict__ page_last_access;
  const int64_t* __restrict__ cache_step;
  uint32_t total;
  uint32_t page_size;
};

constexpr uint32_t kBlockSize = 256;

__global__ void dsa_touch_slots_kernel(
    const __grid_constant__ DSATouchSlotsParams p) {
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= p.total) return;

  const int32_t token_idx = p.token_indices[tid];
  if (token_idx < 0) return;

  const uint32_t dense_page = static_cast<uint32_t>(token_idx) / p.page_size;
  const int32_t slot_idx = p.dense_page_to_slot[dense_page];
  if (slot_idx < 0) return;

  p.page_last_access[static_cast<uint32_t>(slot_idx)] = p.cache_step[0];
}

struct DSATouchSlots {
  static void run(
      const tvm::ffi::TensorView token_indices,
      const tvm::ffi::TensorView dense_page_to_slot,
      const tvm::ffi::TensorView page_last_access,
      int64_t page_size,
      const tvm::ffi::TensorView cache_step) {
    using namespace host;

    auto ROWS = SymbolicSize{"rows"};
    auto COLS = SymbolicSize{"cols"};
    auto dense_pages = SymbolicSize{"dense_pages"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({ROWS, COLS})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(token_indices);
    TensorMatcher({dense_pages})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(dense_page_to_slot);
    TensorMatcher({-1})
        .with_dtype<int64_t>()
        .with_device(device)
        .verify(page_last_access);
    TensorMatcher({1}).with_dtype<int64_t>().with_device(device).verify(cache_step);

    const auto rows = static_cast<uint32_t>(ROWS.unwrap());
    const auto cols = static_cast<uint32_t>(COLS.unwrap());
    const auto total = rows * cols;
    if (total == 0) return;

    const auto params = DSATouchSlotsParams{
        .token_indices = static_cast<const int32_t*>(token_indices.data_ptr()),
        .dense_page_to_slot =
            static_cast<const int32_t*>(dense_page_to_slot.data_ptr()),
        .page_last_access =
            static_cast<int64_t*>(const_cast<void*>(page_last_access.data_ptr())),
        .cache_step = static_cast<const int64_t*>(cache_step.data_ptr()),
        .total = total,
        .page_size = static_cast<uint32_t>(page_size),
    };

    const auto num_blocks = div_ceil(total, kBlockSize);
    LaunchKernel(num_blocks, kBlockSize, device.unwrap())(
        dsa_touch_slots_kernel, params);
  }
};

}  // namespace
