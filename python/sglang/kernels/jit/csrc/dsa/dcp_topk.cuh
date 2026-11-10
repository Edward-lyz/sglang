// SGLANG_K31_INDEXER_DCP: one-kernel EXACT global DSA top-k over a DCP-striped
// index-K (see kernels/ops/attention/dsa/dcp_topk_fused.py).
//
// Every rank holds its local top-k candidates (raw local positions + their
// local KV rows) of each row, and the DCP group must agree on the exact global
// k-th score. A radix select over order-preserving uint32 keys runs in lockstep
// across the group: round r histograms the next 8 key bits of the candidates
// still tied with the running prefix, and the group sums the histograms.
// As soon as the group-wide boundary bucket holds <= kGatherTotal candidates the
// ranks stop narrowing and exchange the bucket's KEYS instead (one "gather"
// exchange, variable length): every rank then ranks the <= kGatherTotal keys
// itself -- equal keys
// ordered by (DCP rank, candidate order) -- and keeps the ones of its own that
// fall inside the remaining need. Only a bucket that is still larger after all
// 4 rounds (> kGatherTotal exactly equal keys) takes the count-only tie split by
// rank order. The selected set equals a single-rank top-k over the gathered
// scores, ties broken by rank, deterministically.
//
// Rows are owned by CTA (row % grid), the same on every rank (identical row
// count and grid), so a CTA only ever talks to the same-index CTA on its
// peers: no grid-wide barrier. Every exchange pushes all of the CTA's rows in
// LL form -- each 8 B word carries 32 data bits plus the epoch as its own flag,
// stored with one 8 B volatile store -- and the receivers poll the data itself
// (no system fence, no separate flag). Receive-side polling is spread over the
// whole CTA (all rows x peers x words in parallel). A CTA stops as soon as all
// its rows are resolved; every decision derives from group-summed counts of
// its rows, so the peer CTAs agree.
//
// Buffers are double-buffered by a per-CTA epoch parity; the epoch advances
// only on calls where the CTA exchanged (again a group-identical decision).
// Writing epoch e+2's slots needs the peer's epoch e+1 round 0 first, i.e. the
// peer has finished reading epoch e's slots of the same parity.
//
// Buffer capacity: the receive regions hold kMaxRowsPerCta rows per CTA, so a
// call takes at most min(num_ctas, max_ctas) * kMaxRowsPerCta rows.
//
// History (B300, 192 rows x 40K ctx, standalone): per-row exchange with 4 B
// stores 142 us/layer; batched exchange behind 3 grid barriers per round 85 us;
// per-CTA exchange with data + fence + release flag 72 us; LL words 54 us
// (4 rounds, per-row serial receive polling).

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <algorithm>
#include <cstdint>

namespace {

constexpr uint32_t kMaxRanks = 8;
constexpr uint32_t kTopK = 2048;
constexpr uint32_t kThreads = 1024;
constexpr uint32_t kItems = kTopK / kThreads;  // candidates per thread per row
constexpr uint32_t kBins = 256;
constexpr uint32_t kRadixRounds = 4;
constexpr uint32_t kMaxRowsPerCta = 4;
constexpr uint32_t kWarps = kThreads / 32;
constexpr uint32_t kWordsPerRow = kBins / 2;  // radix slot: two int16 counts per LL word
constexpr uint32_t kGatherTotal = 1024;       // max group-wide boundary keys gathered
constexpr uint32_t kGatherWords = 1 + kGatherTotal;  // gather slot: count, then keys
static_assert(kMaxRowsPerCta <= kWarps);

enum : uint8_t { kModeTake = 0, kModeActive = 1, kModeGather = 2, kModeTie = 3 };

struct DcpTopkParams {
  const float* logits;
  const int32_t* raw;         // [rows, kTopK] local candidate positions, -1 pad
  const int32_t* local_rows;  // [rows, kTopK] their local KV rows
  const int32_t* lengths;     // [rows] global row lengths
  int32_t* out;               // [rows, kTopK] widened slots, -1 tail
  int32_t* out_counts;        // [rows]
  // LL words: radix slots [2][kRadixRounds][dcp][max_ctas][kMaxRowsPerCta][kWordsPerRow],
  // then the gather slot [2][dcp][max_ctas][kMaxRowsPerCta][kGatherWords].
  uint2* recv[kMaxRanks];
  int32_t* epochs;            // [max_ctas] local
  int64_t logits_stride;
  uint32_t rows;
  uint32_t max_ctas;
  uint32_t rank;
  uint32_t dcp;
  uint32_t grid;
};

// LL word: {32 data bits, epoch}. An aligned 8 B store lands atomically, so a
// word whose second half equals the expected epoch carries that epoch's data.
SGL_DEVICE void st_ll(uint2* ptr, uint32_t data, uint32_t epoch) {
  asm volatile("st.volatile.global.v2.u32 [%0], {%1, %2};" ::"l"(ptr), "r"(data), "r"(epoch) : "memory");
}

SGL_DEVICE uint32_t ld_ll(const uint2* ptr, uint32_t epoch) {
  uint32_t data, flag;
  do {
    asm volatile("ld.volatile.global.v2.u32 {%0, %1}, [%2];" : "=r"(data), "=r"(flag) : "l"(ptr) : "memory");
  } while (flag != epoch);
  return data;
}

SGL_DEVICE uint32_t order_key(float v) {
  const uint32_t b = __float_as_uint(v);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

struct Shared {
  alignas(16) uint32_t words[kMaxRowsPerCta][kGatherWords];  // push source
  int32_t ghist[kMaxRowsPerCta][kBins];                      // group-summed counts
  uint32_t dkey[kGatherTotal];                               // dense gathered keys of one row
  uint16_t dtag[kGatherTotal];                               // (src << 10) | index
  int32_t gcnt[kMaxRowsPerCta][kMaxRanks];
  int32_t goff[kMaxRanks + 1];
  uint8_t mine_sel[kMaxRowsPerCta][kGatherTotal];  // own bucket candidate i is in
  int32_t warp_sums[kWarps];
  int32_t bin[kMaxRowsPerCta];
  int32_t above[kMaxRowsPerCta];
  int32_t bucket[kMaxRowsPerCta];
  int32_t epoch;
};

SGL_DEVICE int32_t block_exclusive_scan(int32_t v, Shared& sh, int32_t* total) {
  const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int32_t x = v;
#pragma unroll
  for (uint32_t d = 1; d < 32; d <<= 1) {
    const int32_t y = __shfl_up_sync(0xffffffffu, x, d);
    if (lane >= d) x += y;
  }
  if (lane == 31) sh.warp_sums[warp] = x;
  __syncthreads();
  if (warp == 0) {
    int32_t w = lane < kWarps ? sh.warp_sums[lane] : 0;
#pragma unroll
    for (uint32_t d = 1; d < 32; d <<= 1) {
      const int32_t y = __shfl_up_sync(0xffffffffu, w, d);
      if (lane >= d) w += y;
    }
    if (lane < kWarps) sh.warp_sums[lane] = w;
  }
  __syncthreads();
  const int32_t base = warp == 0 ? 0 : sh.warp_sums[warp - 1];
  *total = sh.warp_sums[kWarps - 1];
  __syncthreads();
  return base + x - v;
}

SGL_DEVICE int64_t recv_off(const DcpTopkParams& p, uint32_t parity, uint32_t slot, uint32_t src, uint32_t s) {
  if (slot < kRadixRounds) {
    return (((static_cast<int64_t>(parity) * kRadixRounds + slot) * p.dcp + src) * p.max_ctas + blockIdx.x) *
               (kMaxRowsPerCta * kWordsPerRow) +
           s * kWordsPerRow;
  }
  const int64_t radix_words = 2ll * kRadixRounds * p.dcp * p.max_ctas * kMaxRowsPerCta * kWordsPerRow;
  return radix_words +
         ((static_cast<int64_t>(parity) * p.dcp + src) * p.max_ctas + blockIdx.x) * (kMaxRowsPerCta * kGatherWords) +
         s * kGatherWords;
}

SGL_DEVICE const uint2* recv_word(const DcpTopkParams& p, const Shared& sh, uint32_t slot, uint32_t src, uint32_t s, uint32_t w) {
  return p.recv[p.rank] + recv_off(p, static_cast<uint32_t>(sh.epoch) & 1u, slot, src, s) + w;
}

SGL_DEVICE uint32_t row_of(const DcpTopkParams& p, uint32_t s) {
  return blockIdx.x + s * p.grid;
}

// Push LL words [0, nwords[s]) of every local row's sh.words to every peer.
SGL_DEVICE void push(
    const DcpTopkParams& p, const Shared& sh, uint32_t slot, uint32_t num_local, uint32_t stride, const uint32_t* nwords) {
  const uint32_t parity = static_cast<uint32_t>(sh.epoch) & 1u;
  const uint32_t total = p.dcp * num_local * stride;
  for (uint32_t i = threadIdx.x; i < total; i += kThreads) {
    const uint32_t w = i % stride;
    const uint32_t s = (i / stride) % num_local;
    if (w >= nwords[s]) continue;
    const uint32_t dst = i / (stride * num_local);
    st_ll(p.recv[dst] + recv_off(p, parity, slot, p.rank, s) + w, sh.words[s][w], static_cast<uint32_t>(sh.epoch));
  }
}

// Group-sum the histograms of the rows in `mask` (every row x peer x word
// polled in parallel), then warp s locates row s's bin holding the need-th
// largest key: above(b) < need <= above(b) + h(b).
SGL_DEVICE void reduce_and_pick(
    const DcpTopkParams& p, Shared& sh, uint32_t slot, uint32_t num_local, uint32_t mask, const int32_t* need) {
  for (uint32_t i = threadIdx.x; i < num_local * kBins; i += kThreads) sh.ghist[i / kBins][i % kBins] = 0;
  __syncthreads();
  const uint32_t per_row = p.dcp * kWordsPerRow;
  for (uint32_t i = threadIdx.x; i < num_local * per_row; i += kThreads) {
    const uint32_t s = i / per_row;
    if (!((mask >> s) & 1u)) continue;
    const uint32_t src = (i % per_row) / kWordsPerRow, w = i % kWordsPerRow;
    const uint32_t data = ld_ll(recv_word(p, sh, slot, src, s, w), sh.epoch);
    atomicAdd(&sh.ghist[s][2 * w], static_cast<int32_t>(data & 0xffffu));
    atomicAdd(&sh.ghist[s][2 * w + 1], static_cast<int32_t>(data >> 16));
  }
  __syncthreads();
  const uint32_t s = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (s < num_local && ((mask >> s) & 1u)) {
    int32_t h[8], sum = 0;  // lane owns bins [8 * lane, 8 * lane + 8)
#pragma unroll
    for (uint32_t j = 0; j < 8; ++j) {
      h[j] = sh.ghist[s][lane * 8 + j];
      sum += h[j];
    }
    // acc = counts in the bins above this lane's (inclusive suffix scan - own).
    int32_t suffix = sum;
#pragma unroll
    for (uint32_t d = 1; d < 32; d <<= 1) {
      const int32_t y = __shfl_down_sync(0xffffffffu, suffix, d);
      if (lane + d < 32) suffix += y;
    }
    int32_t acc = suffix - sum;
    const int32_t nd = need[s];
#pragma unroll
    for (int j = 7; j >= 0; --j) {
      if (acc < nd && acc + h[j] >= nd) {
        sh.bin[s] = static_cast<int32_t>(lane * 8 + j);
        sh.above[s] = acc;
        sh.bucket[s] = h[j];
      }
      acc += h[j];
    }
  }
  __syncthreads();
}

__global__ __launch_bounds__(kThreads, 1) void dcp_topk_kernel(const __grid_constant__ DcpTopkParams p) {
  __shared__ Shared sh;
  const uint32_t num_local = (p.rows - blockIdx.x + p.grid - 1) / p.grid;

  uint32_t key[kMaxRowsPerCta][kItems];
  int32_t loc[kMaxRowsPerCta][kItems];
  bool valid[kMaxRowsPerCta][kItems];
  uint32_t prefix[kMaxRowsPerCta];
  int32_t need[kMaxRowsPerCta];
  uint32_t depth[kMaxRowsPerCta];  // resolved key bytes
  uint8_t mode[kMaxRowsPerCta];
  int32_t quota[kMaxRowsPerCta];   // tie mode: this rank's share of the bucket

  bool exchanged = false;
  if (threadIdx.x == 0) sh.epoch = p.epochs[blockIdx.x] + 1;
#pragma unroll
  for (uint32_t s = 0; s < kMaxRowsPerCta; ++s) {
    prefix[s] = 0;
    need[s] = kTopK;
    depth[s] = 0;
    mode[s] = kModeTake;
    quota[s] = 0;
    if (s < num_local) {
      const uint32_t row = row_of(p, s);
      const int64_t base = static_cast<int64_t>(row) * kTopK;
#pragma unroll
      for (uint32_t j = 0; j < kItems; ++j) {
        const uint32_t i = j * kThreads + threadIdx.x;
        const int32_t pos = p.raw[base + i];
        valid[s][j] = pos >= 0;
        loc[s][j] = p.local_rows[base + i];
        key[s][j] = valid[s][j] ? order_key(p.logits[row * p.logits_stride + pos]) : 0u;
      }
      // len <= kTopK: every candidate is in the global top-k.
      if (p.lengths[row] > static_cast<int32_t>(kTopK)) mode[s] = kModeActive;
    }
  }
  __syncthreads();

  for (uint32_t round = 0; round < kRadixRounds; ++round) {
    uint32_t mask = 0;
#pragma unroll
    for (uint32_t s = 0; s < kMaxRowsPerCta; ++s) mask |= (s < num_local && mode[s] == kModeActive) ? (1u << s) : 0u;
    if (!mask) break;  // uniform within the CTA and across the peer CTAs
    const uint32_t shift = 24 - 8 * round;
    for (uint32_t i = threadIdx.x; i < num_local * kWordsPerRow; i += kThreads) {
      sh.words[i / kWordsPerRow][i % kWordsPerRow] = 0;
    }
    __syncthreads();
#pragma unroll
    for (uint32_t s = 0; s < kMaxRowsPerCta; ++s) {
      if (!((mask >> s) & 1u)) continue;
#pragma unroll
      for (uint32_t j = 0; j < kItems; ++j) {
        if (valid[s][j] && (round == 0 || (key[s][j] >> (shift + 8)) == prefix[s])) {
          const uint32_t b = (key[s][j] >> shift) & 0xffu;
          atomicAdd(&sh.words[s][b >> 1], (b & 1u) ? (1u << 16) : 1u);
        }
      }
    }
    __syncthreads();
    const uint32_t full[kMaxRowsPerCta] = {kWordsPerRow, kWordsPerRow, kWordsPerRow, kWordsPerRow};
    push(p, sh, round, num_local, kWordsPerRow, full);
    exchanged = true;
    reduce_and_pick(p, sh, round, num_local, mask, need);
#pragma unroll
    for (uint32_t s = 0; s < kMaxRowsPerCta; ++s) {
      if (!((mask >> s) & 1u)) continue;
      prefix[s] = (prefix[s] << 8) | static_cast<uint32_t>(sh.bin[s]);
      need[s] -= sh.above[s];
      depth[s] = round + 1;
      const int32_t bucket = sh.bucket[s];
      if (need[s] == bucket) {
        mode[s] = kModeTake;
      } else if (bucket <= static_cast<int32_t>(kGatherTotal)) {
        mode[s] = kModeGather;
      } else if (round + 1 == kRadixRounds) {
        mode[s] = kModeTie;  // > kGatherTotal exactly equal keys
      }
    }
    __syncthreads();
  }

  // Gather / tie exchange: gather rows send their boundary-bucket keys (word 0
  // = count), tie rows their bucket count; other rows send a 0 count.
  bool any_gather = false, any_tie = false;
#pragma unroll
  for (uint32_t s = 0; s < kMaxRowsPerCta; ++s) {
    any_gather |= s < num_local && mode[s] == kModeGather;
    any_tie |= s < num_local && mode[s] == kModeTie;
  }
  if (any_gather || any_tie) {
    uint32_t nwords[kMaxRowsPerCta];
#pragma unroll
    for (uint32_t s = 0; s < kMaxRowsPerCta; ++s) {
      nwords[s] = 1;
      if (s >= num_local) continue;
      const uint32_t drop = 32 - 8 * depth[s];
      int32_t mine = 0;
      bool in_b[kItems];
#pragma unroll
      for (uint32_t j = 0; j < kItems; ++j) {
        in_b[j] = (mode[s] == kModeGather || mode[s] == kModeTie) && valid[s][j] && (key[s][j] >> drop) == prefix[s];
        mine += in_b[j] ? 1 : 0;
      }
      int32_t total;
      int32_t idx = block_exclusive_scan(mine, sh, &total);
      if (mode[s] == kModeGather) {
#pragma unroll
        for (uint32_t j = 0; j < kItems; ++j) {
          if (in_b[j]) sh.words[s][1 + idx++] = key[s][j];
        }
        nwords[s] = 1 + static_cast<uint32_t>(total);
      }
      if (threadIdx.x == 0) sh.words[s][0] = static_cast<uint32_t>(total);
      quota[s] = total;
    }
    __syncthreads();
    push(p, sh, kRadixRounds, num_local, any_gather ? kGatherWords : 1, nwords);

    for (uint32_t i = threadIdx.x; i < num_local * p.dcp; i += kThreads) {
      const uint32_t s = i / p.dcp, src = i % p.dcp;
      sh.gcnt[s][src] = static_cast<int32_t>(ld_ll(recv_word(p, sh, kRadixRounds, src, s, 0), sh.epoch));
    }
    __syncthreads();

#pragma unroll
    for (uint32_t s = 0; s < kMaxRowsPerCta; ++s) {
      if (s >= num_local) break;
      if (mode[s] == kModeTie) {
        int32_t before = 0;
        for (uint32_t src = 0; src < p.rank; ++src) before += sh.gcnt[s][src];
        quota[s] = min(max(need[s] - before, 0), quota[s]);
      } else if (mode[s] == kModeGather) {
        if (threadIdx.x == 0) {
          int32_t off = 0;
          for (uint32_t src = 0; src < p.dcp; ++src) {
            sh.goff[src] = off;
            off += sh.gcnt[s][src];
          }
          sh.goff[p.dcp] = off;
        }
        __syncthreads();
        // Poll every peer's keys of this row straight into the dense array,
        // tagged (src, index) for the tie order.
        const int32_t n = sh.goff[p.dcp];
        for (int32_t t = threadIdx.x; t < n; t += kThreads) {
          uint32_t src = 0;
          while (t >= sh.goff[src + 1]) ++src;
          const uint32_t k = static_cast<uint32_t>(t - sh.goff[src]);
          sh.dkey[t] = ld_ll(recv_word(p, sh, kRadixRounds, src, s, 1 + k), sh.epoch);
          sh.dtag[t] = static_cast<uint16_t>((src << 10) | k);
        }
        __syncthreads();
        // Own candidate k is in iff fewer than `need` gathered keys precede it
        // (greater key, or equal key with a lower (src, index) tag). 8 threads
        // per own candidate, striding over the dense keys.
        const int32_t mine_n = sh.gcnt[s][p.rank];
        const int32_t mine_off = sh.goff[p.rank];
        const uint32_t part = threadIdx.x & 7;
        for (int32_t k0 = 0; k0 < mine_n; k0 += kThreads / 8) {
          const int32_t k = k0 + static_cast<int32_t>(threadIdx.x >> 3);
          int32_t before = 0;
          if (k < mine_n) {
            const uint32_t mykey = sh.dkey[mine_off + k];
            const uint16_t mytag = sh.dtag[mine_off + k];
            for (int32_t t = part; t < n; t += 8) {
              const uint32_t other = sh.dkey[t];
              before += (other > mykey || (other == mykey && sh.dtag[t] < mytag)) ? 1 : 0;
            }
          }
#pragma unroll
          for (uint32_t d = 1; d < 8; d <<= 1) before += __shfl_xor_sync(0xffffffffu, before, d);
          if (part == 0 && k < mine_n) sh.mine_sel[s][k] = before < need[s] ? 1 : 0;
        }
        __syncthreads();
      }
    }
  }

  // Select keys above the boundary prefix, plus the boundary bucket by mode:
  // whole (take), ranked against the gathered keys (gather), or this rank's
  // quota in candidate order (tie). Compact into widened slots.
#pragma unroll
  for (uint32_t s = 0; s < kMaxRowsPerCta; ++s) {
    if (s >= num_local) break;
    const uint32_t row = row_of(p, s);
    const uint32_t drop = 32 - 8 * depth[s];
    bool high[kItems], bucket[kItems];
    int32_t in_bucket = 0;
#pragma unroll
    for (uint32_t j = 0; j < kItems; ++j) {
      const uint32_t top = depth[s] == 0 ? 0u : (key[s][j] >> drop);
      high[j] = valid[s][j] && depth[s] > 0 && top > prefix[s];
      bucket[j] = valid[s][j] && (depth[s] == 0 || top == prefix[s]);
      in_bucket += bucket[j] ? 1 : 0;
    }
    int32_t unused;
    int32_t brank = block_exclusive_scan(in_bucket, sh, &unused);
    bool sel[kItems];
    int32_t count = 0;
#pragma unroll
    for (uint32_t j = 0; j < kItems; ++j) {
      bool in = false;
      if (bucket[j]) {
        if (mode[s] == kModeTake) {
          in = true;
        } else if (mode[s] == kModeGather) {
          in = sh.mine_sel[s][brank] != 0;
        } else {
          in = brank < quota[s];
        }
      }
      sel[j] = high[j] || in;
      brank += bucket[j] ? 1 : 0;
      count += sel[j] ? 1 : 0;
    }
    int32_t total;
    int32_t pos = block_exclusive_scan(count, sh, &total);
    int32_t* out_row = p.out + static_cast<int64_t>(row) * kTopK;
#pragma unroll
    for (uint32_t j = 0; j < kItems; ++j) {
      if (sel[j]) out_row[pos++] = loc[s][j] * static_cast<int32_t>(p.dcp) + static_cast<int32_t>(p.rank);
    }
    for (uint32_t i = total + threadIdx.x; i < kTopK; i += kThreads) out_row[i] = -1;
    if (threadIdx.x == 0) p.out_counts[row] = total;
  }

  if (threadIdx.x == 0 && exchanged) p.epochs[blockIdx.x] = sh.epoch;
}

struct DcpTopkKernel {
  // ptr_table: CPU int64 [dcp] = the peers' LL receive bases.
  static void
  run(const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView raw,
      const tvm::ffi::TensorView local_rows,
      const tvm::ffi::TensorView lengths,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView out_counts,
      const tvm::ffi::TensorView ptr_table,
      const tvm::ffi::TensorView epochs,
      const int64_t rank,
      const int64_t num_ctas) {
    using namespace host;

    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto R = SymbolicSize{"rows"};
    auto W = SymbolicSize{"width"};
    auto S = SymbolicSize{"logits_stride"};
    auto T = SymbolicSize{"ptr_table"};
    auto C = SymbolicSize{"max_ctas"};
    const auto K = static_cast<int64_t>(kTopK);
    TensorMatcher({R, W}).with_strides({S, 1}).with_dtype<float>().with_device(device).verify(logits);
    TensorMatcher({R, K}).with_dtype<int32_t>().with_device(device).verify(raw);
    TensorMatcher({R, K}).with_dtype<int32_t>().with_device(device).verify(local_rows);
    TensorMatcher({R}).with_dtype<int32_t>().with_device(device).verify(lengths);
    TensorMatcher({R, K}).with_dtype<int32_t>().with_device(device).verify(out);
    TensorMatcher({R}).with_dtype<int32_t>().with_device(device).verify(out_counts);
    TensorMatcher({T}).with_dtype<int64_t>().with_device<kDLCPU>().verify(ptr_table);
    TensorMatcher({C}).with_dtype<int32_t>().with_device(device).verify(epochs);

    const auto dcp = static_cast<uint32_t>(T.unwrap());
    RuntimeCheck(dcp >= 1 && dcp <= kMaxRanks);
    RuntimeCheck(rank >= 0 && rank < static_cast<int64_t>(dcp));
    const auto rows = static_cast<uint32_t>(R.unwrap());
    if (rows == 0) return;
    const auto grid = static_cast<uint32_t>(std::min<int64_t>({R.unwrap(), num_ctas, C.unwrap()}));
    RuntimeCheck(grid >= 1 && rows <= grid * kMaxRowsPerCta, "too many rows for the grid");

    const auto* table = static_cast<const int64_t*>(ptr_table.data_ptr());
    DcpTopkParams params{};
    params.logits = static_cast<const float*>(logits.data_ptr());
    params.raw = static_cast<const int32_t*>(raw.data_ptr());
    params.local_rows = static_cast<const int32_t*>(local_rows.data_ptr());
    params.lengths = static_cast<const int32_t*>(lengths.data_ptr());
    params.out = static_cast<int32_t*>(out.data_ptr());
    params.out_counts = static_cast<int32_t*>(out_counts.data_ptr());
    for (uint32_t i = 0; i < dcp; ++i) params.recv[i] = reinterpret_cast<uint2*>(table[i]);
    params.epochs = static_cast<int32_t*>(epochs.data_ptr());
    params.logits_stride = S.unwrap();
    params.rows = rows;
    params.max_ctas = static_cast<uint32_t>(C.unwrap());
    params.rank = static_cast<uint32_t>(rank);
    params.dcp = dcp;
    params.grid = grid;

    LaunchKernel(grid, kThreads, device.unwrap())(dcp_topk_kernel, params);
  }
};

}  // namespace
