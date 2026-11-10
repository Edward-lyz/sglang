// K3 DCP MLA decode: peer-memory replacement of the a2a staging pack + NCCL
// all_to_all_single in dcp_a2a_lse_reduce.
//
// Every rank writes its per-destination chunk straight into the destination's
// symmetric receive buffer, in exactly the ``[N src][payload | fp32 lse]``
// layout that dcp_fused_pack + all_to_all_single produce, so the unchanged
// Triton LSE combine that follows reads bit-identical bytes.
//
// Protocol (single receive buffer, per-block epochs):
//   * block (dst, s) on rank r first tells dst "r has entered epoch e" (ack),
//     which by stream order means r has finished reading its receive buffer
//     for epoch e - 1;
//   * it waits until dst has entered epoch e too (dst's buffer is free), copies
//     token slice s of its chunk for dst, then release-publishes flag
//     (src=r, slice=s) = e on dst;
//   * finally it waits for the incoming flag (src=dst, slice=s) = e on r, so
//     when the grid exits every incoming slice of every source has landed.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <cstdint>

namespace {

constexpr uint32_t kMaxRanks = 8;

struct DcpPushParams {
  const uint8_t* out;              // [B, N * Hpr, D] payload dtype, contiguous
  const uint8_t* lse;              // [B, N * Hpr] fp32, contiguous
  uint8_t* recv[kMaxRanks];        // per-rank receive buffer: [N][chunk_bytes]
  uint32_t* flags[kMaxRanks];      // per-rank control: ack[N] then data[N src][S]
  uint32_t* counter;               // local, [N * S] epochs
  int64_t chunk_bytes;             // bytes of one source's chunk
  int64_t lse_offset_bytes;        // offset of the fp32 lse block in a chunk
  uint32_t run_vecs;               // 16B vectors per (token, dst) payload run
  uint32_t lse_vecs;               // 16B vectors per (token, dst) lse run
  uint32_t num_tokens;
  uint32_t num_ranks;
  uint32_t rank;
  uint32_t num_slices;
};

SGL_DEVICE uint32_t ld_acquire_sys(const uint32_t* ptr) {
  uint32_t val;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(val) : "l"(ptr) : "memory");
  return val;
}

SGL_DEVICE void st_release_sys(uint32_t* ptr, uint32_t val) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(ptr), "r"(val) : "memory");
}

__global__ void dcp_push_kernel(const __grid_constant__ DcpPushParams p) {
  const uint32_t dst = blockIdx.x / p.num_slices;
  const uint32_t s = blockIdx.x % p.num_slices;
  const uint32_t N = p.num_ranks;
  const uint32_t epoch = p.counter[blockIdx.x] + 1;

  uint32_t* const dst_ack = p.flags[dst];                // ack[N]
  uint32_t* const own_ack = p.flags[p.rank];
  uint32_t* const dst_data = p.flags[dst] + N;           // data[N][S]
  uint32_t* const own_data = p.flags[p.rank] + N;

  if (threadIdx.x == 0) {
    st_release_sys(dst_ack + p.rank, epoch);
    while (static_cast<int32_t>(ld_acquire_sys(own_ack + dst) - epoch) < 0) {
    }
  }
  __syncthreads();

  // Token slice s of this rank's chunk for dst.
  const uint32_t per = (p.num_tokens + p.num_slices - 1) / p.num_slices;
  const uint32_t t0 = s * per;
  const uint32_t t1 = min(p.num_tokens, t0 + per);
  if (t0 < t1) {
    const uint32_t row_vecs = N * p.run_vecs;  // 16B vectors per source token row
    const uint32_t lse_row_vecs = N * p.lse_vecs;
    const auto src_out = reinterpret_cast<const uint4*>(p.out);
    const auto src_lse = reinterpret_cast<const uint4*>(p.lse);
    uint8_t* const chunk = p.recv[dst] + static_cast<int64_t>(p.rank) * p.chunk_bytes;
    const auto dst_out = reinterpret_cast<uint4*>(chunk);
    const auto dst_lse = reinterpret_cast<uint4*>(chunk + p.lse_offset_bytes);

    // kUnroll loads in flight per thread before the (remote) stores.
    constexpr uint32_t kUnroll = 4;
    const uint32_t n_out = (t1 - t0) * p.run_vecs;
    for (uint32_t base = threadIdx.x; base < n_out; base += blockDim.x * kUnroll) {
      uint4 v[kUnroll];
      int64_t d[kUnroll];
#pragma unroll
      for (uint32_t u = 0; u < kUnroll; ++u) {
        const uint32_t i = base + u * blockDim.x;
        if (i < n_out) {
          const uint32_t t = t0 + i / p.run_vecs;
          const uint32_t c = i % p.run_vecs;
          v[u] = src_out[static_cast<int64_t>(t) * row_vecs + dst * p.run_vecs + c];
          d[u] = static_cast<int64_t>(t) * p.run_vecs + c;
        }
      }
#pragma unroll
      for (uint32_t u = 0; u < kUnroll; ++u) {
        if (base + u * blockDim.x < n_out) dst_out[d[u]] = v[u];
      }
    }
    const uint32_t n_lse = (t1 - t0) * p.lse_vecs;
    for (uint32_t i = threadIdx.x; i < n_lse; i += blockDim.x) {
      const uint32_t t = t0 + i / p.lse_vecs;
      const uint32_t v = i % p.lse_vecs;
      dst_lse[static_cast<int64_t>(t) * p.lse_vecs + v] =
          src_lse[static_cast<int64_t>(t) * lse_row_vecs + dst * p.lse_vecs + v];
    }
  }

  // Publish: the barrier orders the block's stores before thread 0, whose
  // system-scope fence + release flag makes them visible to dst.
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
    st_release_sys(dst_data + p.rank * p.num_slices + s, epoch);
    // Incoming slice s from source dst.
    while (ld_acquire_sys(own_data + dst * p.num_slices + s) != epoch) {
    }
    p.counter[blockIdx.x] = epoch;
  }
}

struct DcpPushKernel {
  // ptr_table: CPU int64 [2 * N] = recv bases, then control bases.
  static void
  run(const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView lse,
      const tvm::ffi::TensorView ptr_table,
      const tvm::ffi::TensorView counter,
      const int64_t chunk_bytes,
      const int64_t lse_offset_bytes,
      const int64_t rank,
      const int64_t num_slices,
      const int64_t block_size) {
    using namespace host;

    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto B = SymbolicSize{"num_tokens"};
    auto H = SymbolicSize{"num_heads"};
    auto D = SymbolicSize{"head_dim"};
    auto R = SymbolicSize{"ptr_table"};
    auto C = SymbolicSize{"num_counters"};
    TensorMatcher({B, H, D}).with_device(device).verify(out);
    TensorMatcher({B, H}).with_dtype<float>().with_device(device).verify(lse);
    TensorMatcher({R}).with_dtype<int64_t>().with_device<kDLCPU>().verify(ptr_table);
    TensorMatcher({C}).with_dtype<int32_t>().with_device(device).verify(counter);

    const auto num_ranks = static_cast<uint32_t>(R.unwrap() / 2);
    RuntimeCheck(num_ranks >= 1 && num_ranks <= kMaxRanks && R.unwrap() == 2 * num_ranks);
    RuntimeCheck(H.unwrap() % num_ranks == 0, "heads must split evenly across DCP ranks");
    RuntimeCheck(C.unwrap() == static_cast<int64_t>(num_ranks) * num_slices);
    const int64_t elem = out.dtype().bits / 8;
    const int64_t hpr = H.unwrap() / num_ranks;
    const int64_t run_bytes = hpr * D.unwrap() * elem;
    const int64_t lse_bytes = hpr * 4;
    RuntimeCheck(run_bytes % 16 == 0 && lse_bytes % 16 == 0, "runs must be 16B multiples");
    RuntimeCheck(chunk_bytes % 16 == 0 && lse_offset_bytes % 16 == 0);
    RuntimeCheck(lse_offset_bytes == B.unwrap() * run_bytes);
    RuntimeCheck(lse_offset_bytes + B.unwrap() * lse_bytes <= chunk_bytes);

    const auto* table = static_cast<const int64_t*>(ptr_table.data_ptr());
    DcpPushParams params{};
    params.out = static_cast<const uint8_t*>(out.data_ptr());
    params.lse = static_cast<const uint8_t*>(lse.data_ptr());
    for (uint32_t i = 0; i < num_ranks; ++i) {
      params.recv[i] = reinterpret_cast<uint8_t*>(table[i]);
      params.flags[i] = reinterpret_cast<uint32_t*>(table[num_ranks + i]);
    }
    params.counter = static_cast<uint32_t*>(counter.data_ptr());
    params.chunk_bytes = chunk_bytes;
    params.lse_offset_bytes = lse_offset_bytes;
    params.run_vecs = static_cast<uint32_t>(run_bytes / 16);
    params.lse_vecs = static_cast<uint32_t>(lse_bytes / 16);
    params.num_tokens = static_cast<uint32_t>(B.unwrap());
    params.num_ranks = num_ranks;
    params.rank = static_cast<uint32_t>(rank);
    params.num_slices = static_cast<uint32_t>(num_slices);

    LaunchKernel(num_ranks * num_slices, block_size, device.unwrap())(dcp_push_kernel, params);
  }
};

}  // namespace
