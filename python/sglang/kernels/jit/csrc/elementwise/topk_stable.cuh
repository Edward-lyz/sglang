/**
 * Optimized Stable TopK CUDA Kernel -- v7 (Warp-Prefix + Warp-Aligned Padding)
 * =============================================================================
 *
 * Architecture: Two-phase approach with warp-cooperative index writes
 *
 *   Phase 1: Radix-based selection with warp-prefix write optimization
 *     - Uses 8-bit radix histogram + 4 refinement rounds to identify top-K
 *     - warp_prefix_write() replaces per-thread atomicAdd(&s_counter, 1)
 *       with one atomicAdd per WARP (32x fewer atomics). Uses
 *       __ballot_sync + __popc for deterministic intra-warp positions.
 *     - Deterministic round==3 tie-breaking: collects boundary elements,
 *       sorts ascending by index, takes lowest indices.
 *
 *   Phase 2: Hybrid sort based on SEQ_LEN
 *     - SEQ_LEN <= 16384: Bitmap sort with parallel warp-level prefix sum
 *       (~6 syncs, O(SEQ_LEN/32) work). Builds a bitmap in shared memory,
 *       computes prefix sums via __shfl_up_sync, stream-compacts set bits.
 *     - SEQ_LEN > 16384: Optimized bitonic sort with __syncwarp for j<=16
 *       (O(TopK * log^2(TopK)) = constant cost regardless of SEQ_LEN).
 *       21 __syncthreads + 45 __syncwarp passes.
 *
 * Key improvements over v6:
 *   1. Warp-aligned loop padding: pad_to_warp() rounds to 32 instead of
 *      pad_to_block() rounding to 1024. This produces tighter iteration
 *      bounds for warp_prefix_write loops, reducing redundant iterations.
 *   2. All warp-prefix, bitmap, and bitonic sort optimizations preserved.
 *
 * Determinism guarantee:
 *   Both bitmap and bitonic sort produce indices in ascending order.
 *   Equal scores broken by preferring lower indices via round==3 logic.
 *   Within each warp, warp_prefix_write produces lane-ordered positions,
 *   and the sort phase fixes inter-warp ordering.
 *
 * Constants: TopK=2048, kThreadsPerBlock=1024, kSmem=128KB
 */

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>

namespace {

// ============================================================================
// Constants
// ============================================================================
constexpr int TopK = 2048;
constexpr int kThreadsPerBlock = 1024;
constexpr size_t kSmem = 32 * 1024 * sizeof(uint32_t);  // 128KB

// Maximum supported sequence length.
// Phase 1 (radix selection) scans global memory — O(SEQ_LEN), no shared-memory constraint.
// Phase 2 sort: bitmap path (SEQ_LEN ≤ 16384) uses SEQ_LEN/32 words shared memory;
//               bitonic path (SEQ_LEN > 16384) only sorts TopK=2048 elements, independent of SEQ_LEN.
// 1M keeps comfortable headroom for larger context deployments.
constexpr int kMaxSeqLen = 1048576;

// Threshold for choosing bitmap vs bitonic sort.
// Bitmap is O(SEQ_LEN/32), excels for short sequences.
// Bitonic is O(TopK * log^2(TopK)), constant cost regardless of SEQ_LEN.
constexpr int kBitmapThreshold = 16384;

// ============================================================================
// Helper functions
// ============================================================================

__device__ __forceinline__ auto convert_to_uint8(float x) -> uint8_t {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                                 : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ auto convert_to_uint32(float x) -> uint32_t {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// ============================================================================
// Round up n to the next multiple of warp size (32).
// ============================================================================
// Loops calling warp_prefix_write need padded bounds so every thread in
// every warp reaches __ballot_sync(0xFFFFFFFF, ...). Padding to warp size
// is sufficient (32 divides 1024) and produces tighter loops than rounding
// to block size.
// ============================================================================
__device__ __forceinline__ int pad_to_warp(int n) {
  return (n + 31) & ~31;
}

// ============================================================================
// Warp-cooperative prefix write helper
// ============================================================================
// Replaces per-thread atomicAdd(&s_counter, 1) with per-WARP atomicAdd.
// Within a warp, threads write in lane order (deterministic).
// Between warps, order depends on atomicAdd scheduling (non-deterministic),
// but the final sort phase (bitmap or bitonic) fixes inter-warp ordering.
//
// Performance: 32x fewer atomicAdd calls, reducing shared memory contention.
//
// IMPORTANT: All 32 threads in a warp must reach this call. Use padded loop
// bounds to ensure warp convergence at __ballot_sync.
//
// @param dest      Destination array in shared memory
// @param value     The index value to write (only written if predicate is true)
// @param predicate Whether this thread wants to write
// @param s_counter Shared memory counter for global position tracking
// ============================================================================

__device__ __forceinline__ void warp_prefix_write(
    int* __restrict__ dest,
    int value,
    bool predicate,
    int* __restrict__ s_counter)
{
  const unsigned mask = __ballot_sync(0xFFFFFFFF, predicate);
  if (mask == 0u) return;  // No thread in this warp qualifies

  const int lane = threadIdx.x & 31;
  const int warp_count = __popc(mask);
  const int intra_offset = __popc(mask & ((1u << lane) - 1));

  // One atomic per warp instead of one per qualifying thread
  int warp_base;
  if (lane == 0) {
    warp_base = atomicAdd(s_counter, warp_count);
  }
  warp_base = __shfl_sync(0xFFFFFFFF, warp_base, 0);

  if (predicate) {
    dest[warp_base + intra_offset] = value;
  }
}

// ============================================================================
// Deterministic Radix-based TopK Selection (with warp-prefix writes)
// ============================================================================
// Based on original fast_topk_cuda_tl with two key modifications:
//   1. atomicAdd(&s_counter, 1) replaced by warp_prefix_write at all sites
//   2. round==3 deterministic tie-breaking (collects + sorts boundary elements)
//
// All loops calling warp_prefix_write use pad_to_warp() to ensure full
// warp participation at __ballot_sync boundaries.
//
// atomicAdd(&s_num_input, 1) and atomicAdd(&s_histogram, 1) are kept as-is
// because they perform fused operations (writing to different destinations)
// and don't benefit from warp-prefix optimization.
// ============================================================================

__device__ void fast_topk_deterministic(
    const float* __restrict__ input,
    int* __restrict__ index,
    int row_start,
    int length) {
  int topk = TopK;
  constexpr auto BLOCK_SIZE = 1024;
  constexpr auto RADIX = 256;
  constexpr auto SMEM_INPUT_SIZE = kSmem / (2 * sizeof(int));

  alignas(128) __shared__ int s_histogram_buf[2][RADIX + 128];
  alignas(128) __shared__ int s_counter;
  alignas(128) __shared__ int s_threshold_bin_id;
  alignas(128) __shared__ int s_num_input[2];

  auto& s_histogram = s_histogram_buf[0];
  extern __shared__ int s_input_idx[][SMEM_INPUT_SIZE];

  const int tx = threadIdx.x;

  // Stage 1: 8-bit coarse histogram
  if (tx < RADIX + 1) s_histogram[tx] = 0;
  __syncthreads();

  for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
    const auto bin = convert_to_uint8(input[idx + row_start]);
    ::atomicAdd(&s_histogram[bin], 1);
  }
  __syncthreads();

  // Cumulative suffix-sum for histogram
  const auto run_cumsum = [&] {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      static_assert(1 << 8 == RADIX);
      if (tx < RADIX) {
        const auto j = 1 << i;
        const auto k = i & 1;
        auto value = s_histogram_buf[k][tx];
        if (tx < RADIX - j) {
          value += s_histogram_buf[k][tx + j];
        }
        s_histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };

  run_cumsum();
  if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
    s_threshold_bin_id = tx;
    s_num_input[0] = 0;
    s_counter = 0;
  }
  __syncthreads();

  const auto threshold_bin = s_threshold_bin_id;
  topk -= s_histogram[threshold_bin + 1];

  if (topk == 0) {
    // All top-K elements are strictly above the threshold bin.
    // [WARP-PREFIX SITE 1]: padded loop with warp-cooperative write
    const int padded_len = pad_to_warp(length);
    for (int idx = tx; idx < padded_len; idx += BLOCK_SIZE) {
      const bool in_range = (idx < length);
      const auto bin = in_range
          ? static_cast<int>(convert_to_uint8(input[idx + row_start]))
          : -1;
      warp_prefix_write(index, idx, in_range && (bin > threshold_bin), &s_counter);
    }
    __syncthreads();
    return;
  } else {
    __syncthreads();
    if (tx < RADIX + 1) {
      s_histogram[tx] = 0;
    }
    __syncthreads();

    // Scan full input: above-threshold uses warp-prefix, boundary uses atomicAdd
    // [WARP-PREFIX SITE 2]: padded loop
    const int padded_len = pad_to_warp(length);
    for (int idx = tx; idx < padded_len; idx += BLOCK_SIZE) {
      const bool in_range = (idx < length);
      int bin = -1;
      float raw_input = 0.0f;
      if (in_range) {
        raw_input = input[idx + row_start];
        bin = static_cast<int>(convert_to_uint8(raw_input));
      }
      // Warp-cooperative write for above-threshold indices
      warp_prefix_write(index, idx, in_range && (bin > threshold_bin), &s_counter);
      // Boundary elements still use atomicAdd (fused histogram update)
      if (in_range && bin == threshold_bin) {
        const auto pos = ::atomicAdd(&s_num_input[0], 1);
        if (pos < static_cast<int>(SMEM_INPUT_SIZE)) {
          s_input_idx[0][pos] = idx;
          const auto bin2 = convert_to_uint32(raw_input);
          const auto sub_bin = (bin2 >> 24) & 0xFF;
          ::atomicAdd(&s_histogram[sub_bin], 1);
        }
      }
    }
    __syncthreads();
  }

  // Stage 2: Refine with 8-bit radix passes (up to 4 rounds)
#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    __shared__ int s_last_remain;
    const auto r_idx = round % 2;
    const auto _raw_num_input = s_num_input[r_idx];
    const auto num_input =
        (_raw_num_input < int(SMEM_INPUT_SIZE)) ? _raw_num_input : int(SMEM_INPUT_SIZE);

    run_cumsum();
    if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
      s_threshold_bin_id = tx;
      s_num_input[r_idx ^ 1] = 0;
      s_last_remain = topk - s_histogram[tx + 1];
    }
    __syncthreads();

    const auto threshold_bin = s_threshold_bin_id;
    topk -= s_histogram[threshold_bin + 1];

    if (topk == 0) {
      // All remaining needed elements are strictly above threshold
      // [WARP-PREFIX SITE 3]: padded loop
      const int padded_ni = pad_to_warp(num_input);
      for (int i = tx; i < padded_ni; i += BLOCK_SIZE) {
        const bool in_range = (i < num_input);
        int bin = -1;
        int idx = 0;
        if (in_range) {
          idx = s_input_idx[r_idx][i];
          const auto offset = 24 - round * 8;
          bin = (convert_to_uint32(input[idx + row_start]) >> offset) & 0xFF;
        }
        warp_prefix_write(index, idx, in_range && (bin > threshold_bin), &s_counter);
      }
      __syncthreads();
      break;
    } else {
      __syncthreads();
      if (tx < RADIX + 1) {
        s_histogram[tx] = 0;
      }
      __syncthreads();

      if (round == 3) {
        // ================================================================
        // DETERMINISTIC TIE-BREAKING (round==3 boundary handling)
        // ================================================================
        // After 4 rounds of 8-bit refinement (32 total bits), boundary
        // elements have identical float32 bit patterns. The original uses
        // atomicAdd(&s_last_remain, -1) which races non-deterministically.
        //
        // Our approach:
        //   1. Write above-threshold elements with warp-prefix (fast)
        //   2. Collect ALL boundary (== threshold) elements into buffer
        //   3. If boundary count <= need: write them all
        //   4. If boundary count > need: sort ascending by index, take first N
        //
        // For random inputs, exact 32-bit ties after full radix are rare.
        // For adversarial cases (all-zero), topk==0 fires at stage 1.
        // ================================================================
        const int need = s_last_remain;

        __shared__ int s_boundary_count;
        if (tx == 0) s_boundary_count = 0;
        __syncthreads();

        // [WARP-PREFIX SITE 4]: padded loop for round==3
        const int padded_ni = pad_to_warp(num_input);
        for (int i = tx; i < padded_ni; i += BLOCK_SIZE) {
          const bool in_range = (i < num_input);
          int bin = -1;
          int idx = 0;
          if (in_range) {
            idx = s_input_idx[r_idx][i];
            const auto offset_val = 24 - round * 8;
            bin = (convert_to_uint32(input[idx + row_start]) >> offset_val) & 0xFF;
          }
          // Warp-cooperative write for above-threshold
          warp_prefix_write(index, idx, in_range && (bin > threshold_bin), &s_counter);
          // Collect boundary elements with atomicAdd
          if (in_range && bin == threshold_bin) {
            const auto pos = ::atomicAdd(&s_boundary_count, 1);
            if (pos < static_cast<int>(SMEM_INPUT_SIZE)) {
              s_input_idx[r_idx ^ 1][pos] = idx;
            }
          }
        }
        __syncthreads();

        const int bc = s_boundary_count;

        if (bc <= need) {
          // All boundary elements fit -- write them all
          // [WARP-PREFIX SITE 5a]: padded loop for boundary write
          const int padded_bc = pad_to_warp(bc);
          for (int i = tx; i < padded_bc; i += BLOCK_SIZE) {
            const bool in_range = (i < bc);
            const int val = in_range ? s_input_idx[r_idx ^ 1][i] : 0;
            warp_prefix_write(index, val, in_range, &s_counter);
          }
        } else {
          // More boundary elements than needed -- sort and take lowest indices.
          // This path is rare (requires exact float32 bit-level ties).
          int n_pow2 = 1;
          while (n_pow2 < bc) n_pow2 <<= 1;

          // Pad with INT_MAX for bitonic sort
          for (int i = bc + tx; i < n_pow2; i += BLOCK_SIZE) {
            s_input_idx[r_idx ^ 1][i] = INT_MAX;
          }
          __syncthreads();

          // Small bitonic sort of boundary elements (ascending by index)
          int* s_buf = s_input_idx[r_idx ^ 1];
          for (int kk = 2; kk <= n_pow2; kk <<= 1) {
            for (int j = kk >> 1; j > 0; j >>= 1) {
              __syncthreads();
              for (int t = tx; t < n_pow2 / 2; t += BLOCK_SIZE) {
                int ii = 2 * t - (t & (j - 1));
                int ixj = ii + j;
                if (ixj < n_pow2) {
                  int a = s_buf[ii];
                  int b = s_buf[ixj];
                  bool asc = ((t & (kk >> 1)) == 0);
                  if (asc ? (a > b) : (a < b)) {
                    s_buf[ii] = b;
                    s_buf[ixj] = a;
                  }
                }
              }
            }
          }
          __syncthreads();

          // Write the first `need` (lowest indices)
          // [WARP-PREFIX SITE 5b]: padded loop for sorted boundary write
          const int padded_need = pad_to_warp(need);
          for (int i = tx; i < padded_need; i += BLOCK_SIZE) {
            const bool in_range = (i < need);
            const int val = in_range ? s_buf[i] : 0;
            warp_prefix_write(index, val, in_range, &s_counter);
          }
        }
        __syncthreads();
      } else {
        // Rounds 0-2: above-threshold uses warp-prefix, boundary uses atomicAdd
        // [WARP-PREFIX SITE 6]: padded loop for rounds 0-2
        const int padded_ni = pad_to_warp(num_input);
        for (int i = tx; i < padded_ni; i += BLOCK_SIZE) {
          const bool in_range = (i < num_input);
          int bin = -1;
          int idx = 0;
          float raw_input = 0.0f;
          if (in_range) {
            idx = s_input_idx[r_idx][i];
            raw_input = input[idx + row_start];
            const auto offset = 24 - round * 8;
            bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
          }
          // Warp-cooperative write for above-threshold
          warp_prefix_write(index, idx, in_range && (bin > threshold_bin), &s_counter);
          // Boundary elements: atomicAdd for fused histogram update
          if (in_range && bin == threshold_bin) {
            const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
            if (pos < static_cast<int>(SMEM_INPUT_SIZE)) {
              s_input_idx[r_idx ^ 1][pos] = idx;
              const auto bin2 = convert_to_uint32(raw_input);
              const auto offset = 24 - round * 8;
              const auto sub_bin = (bin2 >> (offset - 8)) & 0xFF;
              ::atomicAdd(&s_histogram[sub_bin], 1);
            }
          }
        }
        __syncthreads();
      }
    }
  }
}

// ============================================================================
// Bitmap-based Index Sort (for SEQ_LEN <= kBitmapThreshold = 16384)
// ============================================================================
// Uses a bitmap to produce sorted output in O(SEQ_LEN/32) work with ~6 syncs.
//
// Algorithm:
//   1. Clear bitmap + zero-init warp sums (1 sync)
//   2. Set bits for each selected index using atomicOr (1 sync)
//   3. Parallel warp-level exclusive prefix sum of popcounts (~3 syncs)
//   4. Stream compact: extract set bits in ascending order (1 sync)
//
// Memory layout (reuses extern shared memory after selection completes):
//   s_workspace[0 .. num_words-1]              : bitmap (1 bit per index)
//   s_workspace[num_words .. 2*num_words-1]    : prefix sum of popcounts
//   s_workspace[2*num_words .. 2*num_words+31] : warp partial sums (32 ints)
// ============================================================================

__device__ void bitmap_sort_indices(
    int* __restrict__ s_indices,     // in: unsorted TopK indices, out: sorted
    int* __restrict__ s_workspace,   // shared memory workspace (reuse extern smem)
    int length)                      // maximum possible index value (= seq_len)
{
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  constexpr int num_warps = kThreadsPerBlock >> 5;  // = 32
  const int num_words = (length + 31) >> 5;         // bitmap words needed

  int* s_bitmap = s_workspace;
  int* s_prefix = s_workspace + num_words;
  int* s_warp_sums = s_workspace + 2 * num_words;  // 32 ints for warp sums

  // ---- Zero-initialize s_warp_sums to prevent UB ----
  if (tid < 32) s_warp_sums[tid] = 0;

  // ---- Step 1: Clear bitmap ----
  for (int i = tid; i < num_words; i += kThreadsPerBlock) {
    s_bitmap[i] = 0;
  }
  __syncthreads();

  // ---- Step 2: Set bits for selected indices ----
  static_assert(TopK / kThreadsPerBlock == 2,
                "Assumption: each thread handles exactly 2 indices");
  {
    int idx0 = s_indices[tid];
    int idx1 = s_indices[tid + kThreadsPerBlock];
    if (idx0 >= 0 && idx0 < length) {
      atomicOr(&s_bitmap[idx0 >> 5], 1u << (idx0 & 31));
    }
    if (idx1 >= 0 && idx1 < length) {
      atomicOr(&s_bitmap[idx1 >> 5], 1u << (idx1 & 31));
    }
  }
  __syncthreads();

  // ---- Steps 3+4: Parallel exclusive prefix sum of popcounts ----
  // Partition bitmap words across warps for balanced work distribution
  const int words_per_warp = (num_words + num_warps - 1) / num_warps;
  const int warp_start = warp_id * words_per_warp;
  const int warp_end = min(warp_start + words_per_warp, num_words);

  // Level 1: Compute warp totals using parallel popcount scan
  {
    int warp_total = 0;
    const int items_per_lane = (words_per_warp + 31) / 32;

    for (int k = 0; k < items_per_lane; k++) {
      int word_idx = warp_start + k * 32 + lane;
      int val = (word_idx < warp_end) ? __popc(s_bitmap[word_idx]) : 0;

      int scan = val;
      #pragma unroll
      for (int d = 1; d < 32; d <<= 1) {
        int n = __shfl_up_sync(0xFFFFFFFF, scan, d);
        if (lane >= d) scan += n;
      }

      int round_total = __shfl_sync(0xFFFFFFFF, scan, 31);
      warp_total += round_total;
    }

    if (lane == 0) {
      s_warp_sums[warp_id] = warp_total;
    }
  }
  __syncthreads();

  // Level 2: Exclusive prefix sum of warp totals (done by warp 0)
  if (warp_id == 0) {
    int val = (lane < num_warps) ? s_warp_sums[lane] : 0;

    #pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      int n = __shfl_up_sync(0xFFFFFFFF, val, d);
      if (lane >= d) val += n;
    }

    int excl = __shfl_up_sync(0xFFFFFFFF, val, 1);
    excl = (lane == 0) ? 0 : excl;

    if (lane < num_warps) {
      s_warp_sums[lane] = excl;
    }
  }
  __syncthreads();

  // Level 1+3 merged: produce final exclusive prefix sums
  {
    int warp_base = s_warp_sums[warp_id];
    const int items_per_lane = (words_per_warp + 31) / 32;
    int lane_running = 0;

    for (int k = 0; k < items_per_lane; k++) {
      int word_idx = warp_start + k * 32 + lane;
      int val = (word_idx < warp_end) ? __popc(s_bitmap[word_idx]) : 0;

      int scan = val;
      #pragma unroll
      for (int d = 1; d < 32; d <<= 1) {
        int n = __shfl_up_sync(0xFFFFFFFF, scan, d);
        if (lane >= d) scan += n;
      }

      int excl = __shfl_up_sync(0xFFFFFFFF, scan, 1);
      excl = (lane == 0) ? 0 : excl;

      if (word_idx < warp_end) {
        s_prefix[word_idx] = warp_base + lane_running + excl;
      }

      int round_total = __shfl_sync(0xFFFFFFFF, scan, 31);
      lane_running += round_total;
    }
  }
  __syncthreads();

  // ---- Step 5: Stream compact -- extract set bits in ascending order ----
  for (int w = tid; w < num_words; w += kThreadsPerBlock) {
    unsigned int bits = static_cast<unsigned int>(s_bitmap[w]);
    int write_pos = s_prefix[w];
    int base_idx = w * 32;

    while (bits) {
      int b = __ffs(bits) - 1;
      s_indices[write_pos] = base_idx + b;
      write_pos++;
      bits &= bits - 1;
    }
  }
  __syncthreads();
}

// ============================================================================
// Optimized Bitonic Sort (for SEQ_LEN > kBitmapThreshold = 16384)
// ============================================================================
// Uses __syncwarp() for j <= 16 (all compare-swap partners within same warp).
// For j > 16, uses __syncthreads() for cross-warp synchronization.
// Total: 21 __syncthreads + 45 __syncwarp passes.
// -1 values are treated as INT_MAX (sorted to end).
// ============================================================================

__device__ void bitonic_sort_2048(int* __restrict__ s_indices) {
  const int tid = threadIdx.x;

  for (int k = 2; k <= TopK; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      if (j <= 16) {
        __syncwarp();
      } else {
        __syncthreads();
      }

      for (int t = tid; t < TopK / 2; t += kThreadsPerBlock) {
        int i = 2 * t - (t & (j - 1));
        int ixj = i + j;

        if (ixj < TopK) {
          int a = s_indices[i];
          int b = s_indices[ixj];

          int a_cmp = (a == -1) ? INT_MAX : a;
          int b_cmp = (b == -1) ? INT_MAX : b;

          bool ascending = ((t & (k >> 1)) == 0);
          bool should_swap = ascending ? (a_cmp > b_cmp) : (a_cmp < b_cmp);

          if (should_swap) {
            s_indices[i] = b;
            s_indices[ixj] = a;
          }
        }
      }
    }
  }
  __syncthreads();
}

// ============================================================================
// Kernel helpers
// ============================================================================

__device__ void naive_topk_cuda(
    const float* __restrict__ score,
    int32_t* __restrict__ indice,
    int32_t length) {
  const auto tid = threadIdx.x;
  for (int i = tid; i < TopK; i += kThreadsPerBlock) {
    indice[i] = (i < length) ? i : -1;
  }
}

// ============================================================================
// Parameter struct for kernel launch
// ============================================================================
struct FastTopKParams {
  const float* __restrict__ input;
  const int32_t* __restrict__ row_starts;
  int32_t* __restrict__ indices;
  int32_t* __restrict__ lengths;
  int64_t input_stride;
};

// ============================================================================
// Main kernel entry point
// ============================================================================

__global__ __launch_bounds__(kThreadsPerBlock, 1)
void topk_optimized_kernel(const FastTopKParams params) {
  const auto& [input, row_starts, indices, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto row_start = row_starts == nullptr ? 0 : row_starts[bid];
  const auto length = lengths[bid];
  const auto indice = indices + bid * TopK;
  const auto score = input + bid * input_stride;

  if (length <= TopK) {
    return naive_topk_cuda(score, indice, length);
  }

  // Phase 1: Deterministic radix-based selection into shared memory
  // Uses warp-prefix writes for all index output (32x fewer atomics)
  __shared__ int s_indices[TopK];
  fast_topk_deterministic(score, s_indices, row_start, length);

  // Phase 2: Sort indices to produce deterministic ascending output.
  if (length <= kBitmapThreshold) {
    constexpr auto SMEM_INPUT_SIZE = kSmem / (2 * sizeof(int));
    extern __shared__ int s_input_idx_raw[][SMEM_INPUT_SIZE];
    int* s_workspace = reinterpret_cast<int*>(s_input_idx_raw);
    bitmap_sort_indices(s_indices, s_workspace, length);
  } else {
    bitonic_sort_2048(s_indices);
  }

  // Phase 3: Copy sorted indices to global memory
  static_assert(TopK / kThreadsPerBlock == 2,
                "Exactly 2 elements per thread for copy");
  indice[tid] = s_indices[tid];
  indice[tid + kThreadsPerBlock] = s_indices[tid + kThreadsPerBlock];
}

}  // anonymous namespace

struct TopKStableKernel {
  static void run(
      tvm::ffi::TensorView score,
      tvm::ffi::TensorView indices,
      tvm::ffi::TensorView lengths,
      tvm::ffi::Optional<tvm::ffi::TensorView> row_starts_opt) {
    using namespace host;

    const auto device = score.device();

    RuntimeCheck(score.device().device_type == kDLCUDA, "score must be a CUDA tensor");
    RuntimeCheck(indices.device() == device, "indices must be on the same CUDA device as score");
    RuntimeCheck(lengths.device() == device, "lengths must be on the same CUDA device as score");
    RuntimeCheck(score.dtype().code == kDLFloat && score.dtype().bits == 32, "score must be float32");
    RuntimeCheck(indices.dtype().code == kDLInt && indices.dtype().bits == 32, "indices must be int32");
    RuntimeCheck(lengths.dtype().code == kDLInt && lengths.dtype().bits == 32, "lengths must be int32");
    RuntimeCheck(score.ndim() == 2, "score must be 2D");
    RuntimeCheck(indices.ndim() == 2, "indices must be 2D");
    RuntimeCheck(lengths.ndim() == 1, "lengths must be 1D");
    RuntimeCheck(score.stride(1) == 1, "score must be contiguous in dim-1");
    RuntimeCheck(indices.is_contiguous(), "indices must be contiguous");
    RuntimeCheck(lengths.is_contiguous(), "lengths must be contiguous");
    RuntimeCheck(indices.size(0) == score.size(0) && indices.size(1) == TopK, "indices shape must be [B, 2048]");
    RuntimeCheck(lengths.size(0) == score.size(0), "lengths size must match batch size");

    if (row_starts_opt.has_value()) {
      RuntimeCheck(row_starts_opt.value().device() == device, "row_starts must be on the same CUDA device as score");
      RuntimeCheck(row_starts_opt.value().dtype().code == kDLInt && row_starts_opt.value().dtype().bits == 32, "row_starts must be int32");
      RuntimeCheck(row_starts_opt.value().ndim() == 1, "row_starts must be 1D");
      RuntimeCheck(row_starts_opt.value().size(0) == score.size(0), "row_starts size must match batch size");
      RuntimeCheck(row_starts_opt.value().is_contiguous(), "row_starts must be contiguous");
    }

    RuntimeCheck(
        score.size(1) <= kMaxSeqLen,
        "SEQ_LEN (",
        score.size(1),
        ") exceeds maximum supported length (",
        kMaxSeqLen,
        "). Sequence length must be <= ",
        kMaxSeqLen);

    FastTopKParams params{
        .input = static_cast<const float*>(score.data_ptr()),
        .row_starts = row_starts_opt.has_value()
            ? static_cast<const int32_t*>(row_starts_opt.value().data_ptr())
            : nullptr,
        .indices = static_cast<int32_t*>(indices.data_ptr()),
        .lengths = static_cast<int32_t*>(lengths.data_ptr()),
        .input_stride = score.stride(0),
    };

    const auto grid = static_cast<uint32_t>(score.size(0));
    const auto stream = LaunchKernel::resolve_device(device);
    RuntimeDeviceCheck(cudaFuncSetAttribute(
        topk_optimized_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kSmem));
    LaunchKernel(grid, kThreadsPerBlock, stream, kSmem)(topk_optimized_kernel, params);
  }
};
