// ===========================================================================
// Fused KV-latent AllGather-input pack  (GLM 5.2 NSA prefill context-parallel)
// ---------------------------------------------------------------------------
// Baseline (nsys) on the CP pre-AllGather hot path
// (python/sglang/srt/models/deepseek_v2.py::DeepseekV2AttentionMLA.rebuild_cp_kv_cache):
//
//     latent_cache[..., :kv_lora_rank] = k_nope.squeeze(1)   # elementwise_kernel
//     latent_cache[..., kv_lora_rank:] = k_pe.squeeze(1)     # elementwise_kernel
//     latent_cache = latent_cache.contiguous()               # copy kernel
//     latent_cache_output = cp_all_gather_rerange_output(latent_cache, ...)
//
// `latent_cache` here is a `dim=-1` narrow of the wider fused qkv_latent
// tensor (row-stride == qkv_latent width, not kv_lora_rank+qk_rope_head_dim),
// so the two slice-assigns write into a *strided* view; the eager path then
// pays a 3rd full-tensor copy (`.contiguous()`) just to materialize a packed
// buffer for the AllGather. This kernel fuses all three into ONE launch: it
// reads `k_nope` / `k_pe` (each may itself be a strided view) and writes
// directly into a FRESH contiguous `[N, kv_lora_rank + qk_rope_head_dim]`
// buffer that is immediately ready as the AllGather input.
//
// This op is pure memory movement (no arithmetic). Three kernel variants,
// selected by shape-regime dispatch in the Python wrapper (see
// project/op/glm52_kv_latent_pack/readme.md for the full ncu-driven
// derivation of each):
//   v1: one CTA/row, BLK threads cooperate on both segments. Fastest at
//       small N (decode, N<=1024) where grid is already launch-floor bound.
//   v2: one WARP/row, kRowsPerBlock warps/block. Gives each lane 2 back-
//       to-back vectorized loads (vs <=1 in v1) -> Achieved Occupancy
//       47.9%->77.2% at N=16384. Best for 1024<N<=4096.
//   v3: persistent grid-stride wrapper around v2, grid capped to
//       SMs*blocksPerSM (<=1 full wave). Eliminates the partial-wave tail
//       ncu flagged on v2 (Waves Per SM 1.73->1.00). Best for N>4096.
// Target arch B200 / sm_100.
// ===========================================================================
#include <sgl_kernel/tensor.h>   // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>    // For RuntimeCheck, div_ceil

#include <sgl_kernel/runtime.cuh>  // For runtime::get_sm_count, get_blocks_per_sm (v3 persistent dispatch)
#include <sgl_kernel/type.cuh>     // For dtype_trait / bf16_t / fp16_t / fp32_t aliases
#include <sgl_kernel/utils.cuh>    // For LaunchKernel, SGL_DEVICE

#include <cstdint>
#include <cuda_runtime.h>

namespace {

// Self-contained, alignment-tagged raw vector of VEC elements of T. Used
// purely as a vectorized-copy vehicle (no arithmetic), so this is safe for
// any dtype (bf16 / fp16 / fp32) without depending on a math-oriented packed
// vector type.
template <typename T, uint32_t VEC>
struct alignas(sizeof(T) * VEC) PackVec {
  T data[VEC];
};

template <typename T>
struct PackKvLatentParams {
  const T* __restrict__ k_nope;  // [num_rows, dim_nope], row-stride = nope_stride
  const T* __restrict__ k_pe;    // [num_rows, dim_rope], row-stride = pe_stride
  T* __restrict__ out;           // [num_rows, dim_nope + dim_rope], row-stride = out_stride
  int64_t nope_stride;           // elements
  int64_t pe_stride;             // elements
  int64_t out_stride;            // elements (== dim_nope + dim_rope for a fresh contiguous out)
  uint32_t dim_nope;
  uint32_t dim_rope;
  uint32_t num_rows;
};

// Copies `n` contiguous-per-row elements from src to dst using VEC-wide
// vectorized ld/st, with a scalar tail for n % VEC != 0. `NTHREADS` is the
// number of threads cooperating on this one segment (may be < blockDim.x
// when multiple rows share a block, e.g. one warp per row); `tid` is the
// caller-supplied cooperating-thread index (threadIdx.x, or lane id).
template <typename T, uint32_t VEC, uint32_t NTHREADS>
SGL_DEVICE void copy_segment(const T* __restrict__ src, T* __restrict__ dst, uint32_t n, uint32_t tid) {
  using V = PackVec<T, VEC>;
  const uint32_t nvec = n / VEC;
  const auto* vsrc = reinterpret_cast<const V*>(src);
  auto* vdst = reinterpret_cast<V*>(dst);
  for (uint32_t v = tid; v < nvec; v += NTHREADS) {
    vdst[v] = vsrc[v];
  }
  for (uint32_t c = nvec * VEC + tid; c < n; c += NTHREADS) {
    dst[c] = src[c];
  }
}

// v1: one block = one row, BLK threads cooperate on both segments.
template <typename T, uint32_t VEC, uint32_t BLK>
__launch_bounds__(BLK) __global__
void pack_kv_latent_kernel(const PackKvLatentParams<T> __grid_constant__ p) {
  const uint32_t row = blockIdx.x;
  if (row >= p.num_rows) return;

  const T* nope_row = p.k_nope + static_cast<int64_t>(row) * p.nope_stride;
  const T* pe_row = p.k_pe + static_cast<int64_t>(row) * p.pe_stride;
  T* out_row = p.out + static_cast<int64_t>(row) * p.out_stride;

  copy_segment<T, VEC, BLK>(nope_row, out_row, p.dim_nope, threadIdx.x);
  copy_segment<T, VEC, BLK>(pe_row, out_row + p.dim_nope, p.dim_rope, threadIdx.x);
}

// v2: one WARP = one row, kRowsPerBlock warps (= kRowsPerBlock*32 threads)
// per block. No cross-row synchronization needed (independent rows).
template <typename T, uint32_t VEC, uint32_t kRowsPerBlock>
__launch_bounds__(kRowsPerBlock * 32) __global__
void pack_kv_latent_kernel_v2(const PackKvLatentParams<T> __grid_constant__ p) {
  constexpr uint32_t kThreadsPerRow = 32;
  const uint32_t tid = threadIdx.x;
  const uint32_t row_in_block = tid / kThreadsPerRow;
  const uint32_t lane = tid % kThreadsPerRow;
  const uint32_t row = blockIdx.x * kRowsPerBlock + row_in_block;
  if (row >= p.num_rows) return;

  const T* nope_row = p.k_nope + static_cast<int64_t>(row) * p.nope_stride;
  const T* pe_row = p.k_pe + static_cast<int64_t>(row) * p.pe_stride;
  T* out_row = p.out + static_cast<int64_t>(row) * p.out_stride;

  copy_segment<T, VEC, kThreadsPerRow>(nope_row, out_row, p.dim_nope, lane);
  copy_segment<T, VEC, kThreadsPerRow>(pe_row, out_row + p.dim_nope, p.dim_rope, lane);
}

// v3: persistent variant of v2 -- grid-stride over row-groups, grid capped to
// SMs * blocksPerSM so the kernel launches at most one "full wave" (no
// partial tail wave).
template <typename T, uint32_t VEC, uint32_t kRowsPerBlock>
__launch_bounds__(kRowsPerBlock * 32) __global__
void pack_kv_latent_kernel_v3_persist(const PackKvLatentParams<T> __grid_constant__ p) {
  constexpr uint32_t kThreadsPerRow = 32;
  const uint32_t tid = threadIdx.x;
  const uint32_t row_in_block = tid / kThreadsPerRow;
  const uint32_t lane = tid % kThreadsPerRow;
  const uint32_t rows_per_grid_step = gridDim.x * kRowsPerBlock;

  for (uint32_t row = blockIdx.x * kRowsPerBlock + row_in_block; row < p.num_rows;
       row += rows_per_grid_step) {
    const T* nope_row = p.k_nope + static_cast<int64_t>(row) * p.nope_stride;
    const T* pe_row = p.k_pe + static_cast<int64_t>(row) * p.pe_stride;
    T* out_row = p.out + static_cast<int64_t>(row) * p.out_stride;

    copy_segment<T, VEC, kThreadsPerRow>(nope_row, out_row, p.dim_nope, lane);
    copy_segment<T, VEC, kThreadsPerRow>(pe_row, out_row + p.dim_nope, p.dim_rope, lane);
  }
}

// ---------------------------------------------------------------------------
// Host wrapper. Template args: <dtype, vec_size, block_threads>.
// ---------------------------------------------------------------------------
template <typename T, int64_t kVecSize, int64_t kBlockThreads>
struct PackKvLatentKernel {
  static constexpr uint32_t kVEC = static_cast<uint32_t>(kVecSize);
  static constexpr uint32_t kBLK = static_cast<uint32_t>(kBlockThreads);
  static constexpr auto kernel = pack_kv_latent_kernel<T, kVEC, kBLK>;

  static_assert(kVEC == 8 || kVEC == 4 || kVEC == 2 || kVEC == 1, "unsupported VEC");
  static_assert(kBLK % 32 == 0 && kBLK <= 1024, "block_threads must be a multiple of 32, <= 1024");

  static void run(
      const tvm::ffi::TensorView k_nope,
      const tvm::ffi::TensorView k_pe,
      const tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    auto N = SymbolicSize{"num_rows"};
    auto Dn = SymbolicSize{"dim_nope"};
    auto Dr = SymbolicSize{"dim_rope"};
    auto D = SymbolicSize{"out_dim"};
    auto SN = SymbolicSize{"nope_stride"};
    auto SR = SymbolicSize{"pe_stride"};
    auto SO = SymbolicSize{"out_stride"};

    TensorMatcher({N, Dn}).with_strides({SN, 1}).with_dtype<T>().with_device(device).verify(k_nope);
    TensorMatcher({N, Dr}).with_strides({SR, 1}).with_dtype<T>().with_device(device).verify(k_pe);
    TensorMatcher({N, D}).with_strides({SO, 1}).with_dtype<T>().with_device(device).verify(out);

    const auto n = static_cast<uint32_t>(N.unwrap());
    const auto dim_nope = static_cast<uint32_t>(Dn.unwrap());
    const auto dim_rope = static_cast<uint32_t>(Dr.unwrap());
    const auto out_dim = static_cast<uint32_t>(D.unwrap());

    RuntimeCheck(out_dim == dim_nope + dim_rope, "out last dim must equal dim_nope + dim_rope");
    RuntimeCheck(
        kVEC == 1 || (dim_nope % kVEC == 0 && dim_rope % kVEC == 0),
        "dim_nope and dim_rope must both be divisible by vec_size");
    if (n == 0) return;

    const auto params = PackKvLatentParams<T>{
        .k_nope = static_cast<const T*>(k_nope.data_ptr()),
        .k_pe = static_cast<const T*>(k_pe.data_ptr()),
        .out = static_cast<T*>(out.data_ptr()),
        .nope_stride = SN.unwrap(),
        .pe_stride = SR.unwrap(),
        .out_stride = SO.unwrap(),
        .dim_nope = dim_nope,
        .dim_rope = dim_rope,
        .num_rows = n,
    };
    LaunchKernel(dim3(n), dim3(kBLK), device.unwrap())(kernel, params);
  }
};

// ---------------------------------------------------------------------------
// v2 host wrapper: warp-per-row, kRowsPerBlock rows/block (32*kRowsPerBlock
// threads/block). Template args: <dtype, vec_size, rows_per_block>.
// ---------------------------------------------------------------------------
template <typename T, int64_t kVecSize, int64_t kRowsPerBlockArg>
struct PackKvLatentKernelV2 {
  static constexpr uint32_t kVEC = static_cast<uint32_t>(kVecSize);
  static constexpr uint32_t kRPB = static_cast<uint32_t>(kRowsPerBlockArg);
  static constexpr auto kernel = pack_kv_latent_kernel_v2<T, kVEC, kRPB>;

  static_assert(kVEC == 8 || kVEC == 4 || kVEC == 2 || kVEC == 1, "unsupported VEC");
  static_assert(kRPB >= 1 && kRPB <= 32, "rows_per_block must be in [1, 32]");

  static void run(
      const tvm::ffi::TensorView k_nope,
      const tvm::ffi::TensorView k_pe,
      const tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    auto N = SymbolicSize{"num_rows"};
    auto Dn = SymbolicSize{"dim_nope"};
    auto Dr = SymbolicSize{"dim_rope"};
    auto D = SymbolicSize{"out_dim"};
    auto SN = SymbolicSize{"nope_stride"};
    auto SR = SymbolicSize{"pe_stride"};
    auto SO = SymbolicSize{"out_stride"};

    TensorMatcher({N, Dn}).with_strides({SN, 1}).with_dtype<T>().with_device(device).verify(k_nope);
    TensorMatcher({N, Dr}).with_strides({SR, 1}).with_dtype<T>().with_device(device).verify(k_pe);
    TensorMatcher({N, D}).with_strides({SO, 1}).with_dtype<T>().with_device(device).verify(out);

    const auto n = static_cast<uint32_t>(N.unwrap());
    const auto dim_nope = static_cast<uint32_t>(Dn.unwrap());
    const auto dim_rope = static_cast<uint32_t>(Dr.unwrap());
    const auto out_dim = static_cast<uint32_t>(D.unwrap());

    RuntimeCheck(out_dim == dim_nope + dim_rope, "out last dim must equal dim_nope + dim_rope");
    RuntimeCheck(
        kVEC == 1 || (dim_nope % kVEC == 0 && dim_rope % kVEC == 0),
        "dim_nope and dim_rope must both be divisible by vec_size");
    if (n == 0) return;

    const auto params = PackKvLatentParams<T>{
        .k_nope = static_cast<const T*>(k_nope.data_ptr()),
        .k_pe = static_cast<const T*>(k_pe.data_ptr()),
        .out = static_cast<T*>(out.data_ptr()),
        .nope_stride = SN.unwrap(),
        .pe_stride = SR.unwrap(),
        .out_stride = SO.unwrap(),
        .dim_nope = dim_nope,
        .dim_rope = dim_rope,
        .num_rows = n,
    };
    const auto grid = host::div_ceil(n, kRPB);
    LaunchKernel(dim3(grid), dim3(kRPB * 32), device.unwrap())(kernel, params);
  }
};

// ---------------------------------------------------------------------------
// v3 host wrapper: persistent grid-stride variant of v2. Grid is capped to
// SMs * blocksPerSM (at most one full wave, no partial tail wave).
// Template args: <dtype, vec_size, rows_per_block>.
// ---------------------------------------------------------------------------
template <typename T, int64_t kVecSize, int64_t kRowsPerBlockArg>
struct PackKvLatentKernelV3Persist {
  static constexpr uint32_t kVEC = static_cast<uint32_t>(kVecSize);
  static constexpr uint32_t kRPB = static_cast<uint32_t>(kRowsPerBlockArg);
  static constexpr auto kernel = pack_kv_latent_kernel_v3_persist<T, kVEC, kRPB>;

  static_assert(kVEC == 8 || kVEC == 4 || kVEC == 2 || kVEC == 1, "unsupported VEC");
  static_assert(kRPB >= 1 && kRPB <= 32, "rows_per_block must be in [1, 32]");

  static void run(
      const tvm::ffi::TensorView k_nope,
      const tvm::ffi::TensorView k_pe,
      const tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    auto N = SymbolicSize{"num_rows"};
    auto Dn = SymbolicSize{"dim_nope"};
    auto Dr = SymbolicSize{"dim_rope"};
    auto D = SymbolicSize{"out_dim"};
    auto SN = SymbolicSize{"nope_stride"};
    auto SR = SymbolicSize{"pe_stride"};
    auto SO = SymbolicSize{"out_stride"};

    TensorMatcher({N, Dn}).with_strides({SN, 1}).with_dtype<T>().with_device(device).verify(k_nope);
    TensorMatcher({N, Dr}).with_strides({SR, 1}).with_dtype<T>().with_device(device).verify(k_pe);
    TensorMatcher({N, D}).with_strides({SO, 1}).with_dtype<T>().with_device(device).verify(out);

    const auto n = static_cast<uint32_t>(N.unwrap());
    const auto dim_nope = static_cast<uint32_t>(Dn.unwrap());
    const auto dim_rope = static_cast<uint32_t>(Dr.unwrap());
    const auto out_dim = static_cast<uint32_t>(D.unwrap());

    RuntimeCheck(out_dim == dim_nope + dim_rope, "out last dim must equal dim_nope + dim_rope");
    RuntimeCheck(
        kVEC == 1 || (dim_nope % kVEC == 0 && dim_rope % kVEC == 0),
        "dim_nope and dim_rope must both be divisible by vec_size");
    if (n == 0) return;

    const auto params = PackKvLatentParams<T>{
        .k_nope = static_cast<const T*>(k_nope.data_ptr()),
        .k_pe = static_cast<const T*>(k_pe.data_ptr()),
        .out = static_cast<T*>(out.data_ptr()),
        .nope_stride = SN.unwrap(),
        .pe_stride = SR.unwrap(),
        .out_stride = SO.unwrap(),
        .dim_nope = dim_nope,
        .dim_rope = dim_rope,
        .num_rows = n,
    };
    const auto dev = device.unwrap();
    const uint32_t sm = runtime::get_sm_count(dev.device_id);
    const uint32_t bpsm = runtime::get_blocks_per_sm(kernel, static_cast<int32_t>(kRPB * 32));
    const uint32_t row_groups = div_ceil(n, kRPB);
    uint32_t grid = sm * (bpsm == 0u ? 1u : bpsm);
    if (grid > row_groups) grid = row_groups;
    if (grid == 0u) grid = 1u;
    LaunchKernel(dim3(grid), dim3(kRPB * 32), dev)(kernel, params);
  }
};

}  // namespace
