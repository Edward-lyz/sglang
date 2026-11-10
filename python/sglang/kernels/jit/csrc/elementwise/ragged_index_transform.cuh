// ===========================================================================
// Fused ragged index transform  (GLM 5.2 / DeepSeek-V3.2 NSA stable-TopK path)
// ---------------------------------------------------------------------------
// Baseline (nsys) on the NSA TopK transform hot path
// (python/sglang/srt/layers/attention/nsa/topk_stable.py::transform_index_ragged_stable):
//
//     mask = topk_indices != -1                              # elementwise_kernel
//     out  = torch.where(mask,                               # elementwise_kernel
//                        topk_indices + offset.unsqueeze(1), # elementwise_kernel (+ broadcast)
//                        topk_indices).to(torch.int32)       # (cast)
//
// For a stable-TopK index block `topk_indices : [M, C] int32` (C == 2048,
// -1 padded) and a per-row ragged offset `offset : [M]` (int32 or int64):
//
//     out[i, j] = (topk_indices[i, j] == -1)
//                 ? -1
//                 : (int32_t)(topk_indices[i, j] + offset[i])
//
// The eager path launches 3 `elementwise_kernel`s + a cast, each streaming the
// whole [M, C] block through HBM. This kernel collapses them into ONE launch.
//
// This op is pure memory movement. ncu on the naive 1-block-per-row / VEC=4 /
// BLK=256 kernel showed only ~48% DRAM (SM ~37%) at full occupancy: the blocks
// were too short (2 int4 ld/st per thread) -> latency/launch-overhead bound, and
// the default write path wasted L2 write-allocate traffic. The knobs below let
// us tune toward the DRAM ceiling:
//   * kRowsPerBlock : each block quantises R consecutive rows (longer blocks,
//                     more in-flight loads, fewer block launches).
//   * kPersist      : persistent grid-stride over rows (grid capped to
//                     SMs * maxBlocksPerSM) instead of one block per row.
//   * kCachePolicy  : 0 default, 1 __ldg read-only loads, 2 = 1 + __stcs
//                     streaming stores (write-once output bypasses L2).
//
// Accumulation is int64 then truncated to int32 -> bit-identical to torch for
// BOTH int32 (2's-complement wrap) and int64 (add-then-truncate) offsets.
// The vectorized path requires C % VEC == 0 (Python entry falls back to VEC=1).
// Target arch B200 / sm_100.
// ===========================================================================
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <cstdint>
#include <cuda_runtime.h>

namespace {

template <typename OffT>
struct RaggedIndexParams {
  const int32_t* __restrict__ indices;  // [M, C] int32, -1 padded
  int32_t* __restrict__ output;         // [M, C] int32
  const OffT* __restrict__ offset;      // [M] int32 or int64
  int64_t in_stride;                    // row stride (elements) of indices
  int64_t out_stride;                   // row stride (elements) of output
  uint32_t num_rows;                    // M
  uint32_t num_cols;                    // C
};

// Vectorized int32 load with cache policy: CACHE>=1 -> __ldg read-only.
template <uint32_t VEC, int CACHE>
SGL_DEVICE void load_i32_vec(const int32_t* ptr, int32_t (&v)[VEC]) {
  if constexpr (VEC == 4) {
    auto* s = reinterpret_cast<const int4*>(ptr);
    if constexpr (CACHE >= 1) *reinterpret_cast<int4*>(v) = __ldg(s);
    else *reinterpret_cast<int4*>(v) = *s;
  } else if constexpr (VEC == 2) {
    auto* s = reinterpret_cast<const int2*>(ptr);
    if constexpr (CACHE >= 1) *reinterpret_cast<int2*>(v) = __ldg(s);
    else *reinterpret_cast<int2*>(v) = *s;
  } else if constexpr (VEC == 1) {
    if constexpr (CACHE >= 1) v[0] = __ldg(ptr);
    else v[0] = *ptr;
  } else {
    static_assert(VEC == 4 || VEC == 2 || VEC == 1, "unsupported VEC");
  }
}

// Vectorized int32 store with cache policy: CACHE>=2 -> __stcs streaming.
template <uint32_t VEC, int CACHE>
SGL_DEVICE void store_i32_vec(int32_t* ptr, const int32_t (&v)[VEC]) {
  if constexpr (VEC == 4) {
    int4 val = *reinterpret_cast<const int4*>(v);
    if constexpr (CACHE >= 2) __stcs(reinterpret_cast<int4*>(ptr), val);
    else *reinterpret_cast<int4*>(ptr) = val;
  } else if constexpr (VEC == 2) {
    int2 val = *reinterpret_cast<const int2*>(v);
    if constexpr (CACHE >= 2) __stcs(reinterpret_cast<int2*>(ptr), val);
    else *reinterpret_cast<int2*>(ptr) = val;
  } else if constexpr (VEC == 1) {
    if constexpr (CACHE >= 2) __stcs(ptr, v[0]);
    else *ptr = v[0];
  } else {
    static_assert(VEC == 4 || VEC == 2 || VEC == 1, "unsupported VEC");
  }
}

// out = (x == -1) ? -1 : (int32_t)((int64)x + off)  (bit-identical to torch)
SGL_DEVICE int32_t apply_offset(int32_t x, int64_t off) {
  return (x == -1) ? -1 : static_cast<int32_t>(static_cast<int64_t>(x) + off);
}

// One row = C int32; the whole block strides it by VEC. offset read once.
// ILP: each thread issues ILP independent int4 loads before compute/store, to
// raise memory-level parallelism (this op is L1TEX/latency bound, not DRAM
// bound -- see ncu notes in history.md).
template <typename OffT, uint32_t VEC, uint32_t BLK, int CACHE, uint32_t ILP>
SGL_DEVICE void process_row(const RaggedIndexParams<OffT>& p, uint32_t row) {
  const int64_t off = static_cast<int64_t>(p.offset[row]);
  const int32_t* in_row = p.indices + static_cast<int64_t>(row) * p.in_stride;
  int32_t* out_row = p.output + static_cast<int64_t>(row) * p.out_stride;
  const uint32_t C = p.num_cols;
  const uint32_t nvec = C / VEC;
  const uint32_t step = BLK * ILP;

  uint32_t base = threadIdx.x;
  // full ILP tiles: all ILP lanes in-bounds -> issue all loads, then stores.
  for (; base + (ILP - 1u) * BLK < nvec; base += step) {
    int32_t xv[ILP][VEC];
#pragma unroll
    for (uint32_t k = 0; k < ILP; ++k)
      load_i32_vec<VEC, CACHE>(in_row + (base + k * BLK) * VEC, xv[k]);
#pragma unroll
    for (uint32_t k = 0; k < ILP; ++k) {
      int32_t yv[VEC];
#pragma unroll
      for (uint32_t j = 0; j < VEC; ++j) yv[j] = apply_offset(xv[k][j], off);
      store_i32_vec<VEC, CACHE>(out_row + (base + k * BLK) * VEC, yv);
    }
  }
  // remainder vecs (< ILP*BLK left), one BLK-stride at a time.
  for (uint32_t v = base; v < nvec; v += BLK) {
    const uint32_t c = v * VEC;
    int32_t xv[VEC];
    load_i32_vec<VEC, CACHE>(in_row + c, xv);
    int32_t yv[VEC];
#pragma unroll
    for (uint32_t j = 0; j < VEC; ++j) yv[j] = apply_offset(xv[j], off);
    store_i32_vec<VEC, CACHE>(out_row + c, yv);
  }
  // scalar tail (only when C % VEC != 0).
  for (uint32_t c = nvec * VEC + threadIdx.x; c < C; c += BLK) {
    out_row[c] = apply_offset(in_row[c], off);
  }
}

// ---- non-persistent: block handles ROWS_PER_BLK consecutive rows -----------
template <typename OffT, uint32_t VEC, uint32_t BLK, uint32_t ROWS_PER_BLK, int CACHE, uint32_t ILP>
__launch_bounds__(BLK) __global__
void ragged_index_transform_kernel(const RaggedIndexParams<OffT> __grid_constant__ p) {
  const uint32_t base_row = blockIdx.x * ROWS_PER_BLK;
#pragma unroll
  for (uint32_t r = 0; r < ROWS_PER_BLK; ++r) {
    const uint32_t row = base_row + r;
    if (row < p.num_rows) process_row<OffT, VEC, BLK, CACHE, ILP>(p, row);
  }
}

// ---- persistent: grid-stride over rows (grid capped to SMs*blocksPerSM) -----
template <typename OffT, uint32_t VEC, uint32_t BLK, int CACHE, uint32_t ILP>
__launch_bounds__(BLK) __global__
void ragged_index_transform_persist_kernel(const RaggedIndexParams<OffT> __grid_constant__ p) {
  for (uint32_t row = blockIdx.x; row < p.num_rows; row += gridDim.x) {
    process_row<OffT, VEC, BLK, CACHE, ILP>(p, row);
  }
}

// ---------------------------------------------------------------------------
// Host launcher. Template args:
//   <offset_is_i64, vec_size, block_threads, rows_per_block, cache_policy,
//    persist, ilp>
// ---------------------------------------------------------------------------
template <bool kOffset64, int64_t kVecSize, int64_t kBlockThreads,
          int64_t kRowsPerBlock = 1, int64_t kCachePolicy = 0, bool kPersist = false,
          int64_t kIlp = 1>
struct RaggedIndexTransformKernel {
  using OffT = std::conditional_t<kOffset64, int64_t, int32_t>;
  static constexpr uint32_t kVEC = static_cast<uint32_t>(kVecSize);
  static constexpr uint32_t kBLK = static_cast<uint32_t>(kBlockThreads);
  static constexpr uint32_t kRPB = static_cast<uint32_t>(kRowsPerBlock);
  static constexpr int kPOL = static_cast<int>(kCachePolicy);
  static constexpr uint32_t kILP = static_cast<uint32_t>(kIlp);
  static constexpr auto kernel = ragged_index_transform_kernel<OffT, kVEC, kBLK, kRPB, kPOL, kILP>;
  static constexpr auto kernel_persist = ragged_index_transform_persist_kernel<OffT, kVEC, kBLK, kPOL, kILP>;
  static_assert(kVEC == 4 || kVEC == 2 || kVEC == 1);
  static_assert(kBLK % 32 == 0 && kBLK <= 1024);
  static_assert(kRPB >= 1 && kRPB <= 32);
  static_assert(kILP >= 1 && kILP <= 8);

  static void run(
      const tvm::ffi::TensorView indices,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView offset) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto M = SymbolicSize{"m"};
    auto C = SymbolicSize{"c"};
    auto IS = SymbolicSize{"in_stride"};
    auto OS = SymbolicSize{"out_stride"};
    device.set_options<kDLCUDA>();

    TensorMatcher({M, C}).with_strides({IS, 1}).with_dtype<int32_t>().with_device(device).verify(indices);
    TensorMatcher({M, C}).with_strides({OS, 1}).with_dtype<int32_t>().with_device(device).verify(output);
    TensorMatcher({M}).with_dtype<OffT>().with_device(device).verify(offset);

    const auto m = static_cast<uint32_t>(M.unwrap());
    const auto c = static_cast<uint32_t>(C.unwrap());
    RuntimeCheck(kVEC == 1 || (c % kVEC) == 0, "num_cols must be divisible by vec_size");
    if (m == 0) return;

    const auto params = RaggedIndexParams<OffT>{
        .indices = static_cast<const int32_t*>(indices.data_ptr()),
        .output = static_cast<int32_t*>(output.data_ptr()),
        .offset = static_cast<const OffT*>(offset.data_ptr()),
        .in_stride = IS.unwrap(),
        .out_stride = OS.unwrap(),
        .num_rows = m,
        .num_cols = c,
    };
    const auto dev = device.unwrap();

    if constexpr (kPersist) {
      const uint32_t sm = runtime::get_sm_count(dev.device_id);
      const uint32_t bpsm = runtime::get_blocks_per_sm(kernel_persist, static_cast<int32_t>(kBLK));
      uint32_t grid = sm * (bpsm == 0u ? 1u : bpsm);
      if (grid > m) grid = m;
      if (grid == 0u) grid = 1u;
      LaunchKernel(dim3(grid), dim3(kBLK), dev)(kernel_persist, params);
    } else {
      const uint32_t grid = (m + kRPB - 1u) / kRPB;
      LaunchKernel(dim3(grid), dim3(kBLK), dev)(kernel, params);
    }
  }
};

}  // namespace
