#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

struct DSAApplyLoadPlanParams {
  const int32_t* __restrict__ active_pages;
  const int32_t* __restrict__ req_indices;
  const int32_t* __restrict__ logical_pages;
  const int64_t* __restrict__ assigned_slot_indices;
  int64_t* __restrict__ assigned_page_starts;
  const int64_t* __restrict__ reserved_page_starts;
  int32_t* __restrict__ page_owner_req;
  int32_t* __restrict__ page_owner_logical;
  int64_t* __restrict__ page_last_access;
  int64_t* __restrict__ gpu_slot;
  const int64_t* __restrict__ cache_step;
  uint32_t num_pages;
  uint32_t gpu_slot_stride_0;
};

constexpr uint32_t kBlockSize = 256;

__global__ void dsa_apply_load_plan_kernel(
    const __grid_constant__ DSAApplyLoadPlanParams p) {
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= p.num_pages) return;

  if (p.active_pages[tid] == 0) {
    p.assigned_page_starts[tid] = 0;
    return;
  }

  const uint32_t slot_idx = static_cast<uint32_t>(p.assigned_slot_indices[tid]);
  const int32_t req_idx = p.req_indices[tid];
  const int32_t logical_page = p.logical_pages[tid];

  const int32_t old_req_idx = p.page_owner_req[slot_idx];
  const int32_t old_logical_page = p.page_owner_logical[slot_idx];
  if (old_req_idx >= 0 && old_logical_page >= 0) {
    p.gpu_slot[static_cast<uint32_t>(old_req_idx) * p.gpu_slot_stride_0 +
               static_cast<uint32_t>(old_logical_page)] = -1;
  }

  const int64_t page_start = p.reserved_page_starts[slot_idx];
  p.gpu_slot[static_cast<uint32_t>(req_idx) * p.gpu_slot_stride_0 +
             static_cast<uint32_t>(logical_page)] = page_start;
  p.page_owner_req[slot_idx] = req_idx;
  p.page_owner_logical[slot_idx] = logical_page;
  p.page_last_access[slot_idx] = p.cache_step[0];
  p.assigned_page_starts[tid] = page_start;
}

struct DSAApplyLoadPlan {
  static void run(
      const tvm::ffi::TensorView active_pages,
      const tvm::ffi::TensorView req_indices,
      const tvm::ffi::TensorView logical_pages,
      const tvm::ffi::TensorView assigned_slot_indices,
      const tvm::ffi::TensorView assigned_page_starts,
      const tvm::ffi::TensorView reserved_page_starts,
      const tvm::ffi::TensorView page_owner_req,
      const tvm::ffi::TensorView page_owner_logical,
      const tvm::ffi::TensorView page_last_access,
      const tvm::ffi::TensorView gpu_slot,
      const tvm::ffi::TensorView cache_step) {
    using namespace host;

    auto NUM_PAGES = SymbolicSize{"num_pages"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({NUM_PAGES})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(active_pages)
        .verify(req_indices)
        .verify(logical_pages);
    TensorMatcher({NUM_PAGES})
        .with_dtype<int64_t>()
        .with_device(device)
        .verify(assigned_slot_indices)
        .verify(assigned_page_starts);
    TensorMatcher({-1})
        .with_dtype<int64_t>()
        .with_device(device)
        .verify(reserved_page_starts);
    TensorMatcher({-1})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(page_owner_req)
        .verify(page_owner_logical);
    TensorMatcher({-1})
        .with_dtype<int64_t>()
        .with_device(device)
        .verify(page_last_access);
    TensorMatcher({-1, -1}).with_dtype<int64_t>().with_device(device).verify(gpu_slot);
    TensorMatcher({1}).with_dtype<int64_t>().with_device(device).verify(cache_step);

    const auto num_pages = static_cast<uint32_t>(NUM_PAGES.unwrap());
    if (num_pages == 0) return;

    const auto params = DSAApplyLoadPlanParams{
        .active_pages = static_cast<const int32_t*>(active_pages.data_ptr()),
        .req_indices = static_cast<const int32_t*>(req_indices.data_ptr()),
        .logical_pages = static_cast<const int32_t*>(logical_pages.data_ptr()),
        .assigned_slot_indices =
            static_cast<const int64_t*>(assigned_slot_indices.data_ptr()),
        .assigned_page_starts =
            static_cast<int64_t*>(const_cast<void*>(assigned_page_starts.data_ptr())),
        .reserved_page_starts =
            static_cast<const int64_t*>(reserved_page_starts.data_ptr()),
        .page_owner_req =
            static_cast<int32_t*>(const_cast<void*>(page_owner_req.data_ptr())),
        .page_owner_logical =
            static_cast<int32_t*>(const_cast<void*>(page_owner_logical.data_ptr())),
        .page_last_access =
            static_cast<int64_t*>(const_cast<void*>(page_last_access.data_ptr())),
        .gpu_slot = static_cast<int64_t*>(const_cast<void*>(gpu_slot.data_ptr())),
        .cache_step = static_cast<const int64_t*>(cache_step.data_ptr()),
        .num_pages = num_pages,
        .gpu_slot_stride_0 = static_cast<uint32_t>(gpu_slot.stride(0)),
    };

    const auto num_blocks = div_ceil(num_pages, kBlockSize);
    LaunchKernel(num_blocks, kBlockSize, device.unwrap())(
        dsa_apply_load_plan_kernel, params);
  }
};

}  // namespace
