// Kimi-K3 KDA short_conv: fused q/k/v width-4 depthwise causal conv1d + SiLU
// with an in-place conv-state (sliding-window) update, replacing the three
// serial `causal_conv1d_fn` (Triton) calls in
// srt/layers/attention/linear/kda_backend.py::KDAAttnBackend.forward_extend.
//
// Ported verbatim (arithmetic and memory-access structure unchanged) from the
// verified standalone implementation `r02c1_pipeline_v004`; only the host shim
// was rewritten onto this tree's tvm-ffi TensorView / TensorMatcher /
// LaunchKernel conventions. Measured 1.787x aggregate over 5 shapes vs the
// project Triton path on sm_100 (B200), bit-exact (torch.equal).
//
// Design: software-pipelined token loop. A compile-time-indexed circular
// register buffer holds `PF` *raw, unconverted* bf16 pair-words, issued PF
// tokens ahead of their unpack, so at steady state each thread keeps PF LDGs
// in flight instead of 1 and the L1TEX return latency is covered by four
// complete compute bodies. bf16->fp32 is a pure bit shift, so deferring the
// unpack cannot change a single output bit.
//
// THE ARITHMETIC IS LOAD-BEARING. It reproduces, instruction for
// instruction, the PTX that Triton emits for
// sglang.kernels.ops.mamba.causal_conv1d_triton._causal_conv1d_fwd_kernel on
// triton 3.5.1 / torch 2.9.1+cu130 / sm_100:
//   * taps 0..3 accumulated in fp32, tap order 0,1,2,3, acc seeded by
//     fma.rn.f32(w0, x0, +0.0f);  no reassociation;
//   * EVEN channels: 4x fma.rn.f32.
//     ODD  channels: fma.rn(w0,x0,0), fma.rn(w1,x1,acc), then mul.f32(w2,x2),
//                    mul.f32(w3,x3), add.f32, add.f32 -- i.e. two extra
//                    roundings, because ptxas vectorises the odd lane with
//                    Blackwell packed-fp32 (mul.f32x2). Using fma on all four
//                    taps for both parities diverges on ~1 in 88,000
//                    elements. MODE 1 (the asymmetry) is the shipped mode;
//                    MODE 0 exists only as a diagnostic.
//   * SiLU as inline asm: sub.f32 / mul.f32 x log2(e) / ex2.approx.f32 /
//     add.f32 1.0 / div.full.f32.  expf / __expf / __fdividef are NOT
//     equivalent here.
//   * pack with cvt.rn.bf16x2.f32.
// Bit-exactness is therefore bound to the toolchain: after a toolchain bump
// the FMA mode must be re-derived from a fresh Triton PTX dump.
//
// Two structural details that must not be "simplified":
//   1. The ragged tail is a SEPARATE template instantiation (FULL=false) with
//      a fully unrolled loop and an `i >= nt` early exit, so `buf[i % PF]`
//      stays a compile-time constant index. A dynamic loop spills the ring
//      buffer to local memory and the whole optimization evaporates.
//   2. Out-of-range prefetches are predicated off, not clamped.
// CH=4 channels/thread, NT=20 tokens/thread, block (32,4), PF=5 are a
// measured local optimum (CH=2 and CH=8 are both worse; the register budget
// matters: 56 registers, 0 spills). Do not retune without re-running the
// bit-exactness gate.
//
// No PDL: the state-update kernel writes the conv-state pool that the main
// conv kernel reads (has_initial_state sequences), so the two launches must
// stay stream-ordered.

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck, div_ceil

#include <sgl_kernel/type.cuh>   // For bf16_t, fp32_t
#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

#define KDA_SHORT_CONV_LOG2E 1.4426950408889634f

// ---- Triton-PTX-identical scalar primitives (see header comment) ----------
SGL_DEVICE float kda_ex2_approx(float x) {
  float r;
  asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
SGL_DEVICE float kda_div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
SGL_DEVICE float kda_silu_triton(float acc) {
  float t = __fsub_rn(0.0f, acc);
  float e = kda_ex2_approx(__fmul_rn(t, KDA_SHORT_CONV_LOG2E));
  return kda_div_full(acc, __fadd_rn(e, 1.0f));
}
SGL_DEVICE unsigned kda_pack_bf16x2(float lo, float hi) {
  unsigned r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
SGL_DEVICE void kda_unpack_bf16x2(unsigned v, float& lo, float& hi) {
  lo = __uint_as_float(v << 16);
  hi = __uint_as_float(v & 0xffff0000u);
}

struct KdaShortConvParams {
  const bf16_t* __restrict__ x[3];  // (dim, T) view, stride (1, x_row_stride)
  const fp32_t* __restrict__ w[3];  // (dim, 4) row-major
  bf16_t* __restrict__ st[3];       // conv-state pool, (line, 3, dim) slice
  bf16_t* __restrict__ out[3];      // (dim, T), stride (1, out_tok)
  const int32_t* __restrict__ qsl;
  const int32_t* __restrict__ ci;
  const bool* __restrict__ his;
  // x_row_stride is the EXPLICIT element stride between two consecutive tokens
  // of `x`, taken from mixed_qkv.stride(0). It is NOT total_dim: in production
  // mixed_qkv is a torch.split view of the fused q/k/v/g projection, so a row
  // is 4 * dim elements wide while only the first 3 * dim are ours
  // (contiguous mixed_qkv: 36864; fused-qkvg view at attn_tp=1: 49152).
  // Never re-derive the token step from total_dim.
  int64_t x_row_stride;
  int64_t out_tok;  // element stride between tokens in out      (12288)
  int64_t st_line;  // element stride between cache lines        (110592)
  int64_t st_tok;   // element stride between state slots        (36864)
  int32_t dim;      // 12288, per projection
  // total_dim is the LOGICAL channel count (3 * dim): grid extent and the
  // q/k/v segment split only. It never participates in x addressing.
  int32_t total_dim;
  int32_t pad_slot_id;
};

// ---- vectorised bf16 group load/store, CH channels (CH*2 bytes) ----------
template <int CH>
SGL_DEVICE void kda_load_words(const bf16_t* p, unsigned* w);
template <>
SGL_DEVICE void kda_load_words<2>(const bf16_t* p, unsigned* w) {
  w[0] = *reinterpret_cast<const unsigned*>(p);
}
template <>
SGL_DEVICE void kda_load_words<4>(const bf16_t* p, unsigned* w) {
  uint2 v = *reinterpret_cast<const uint2*>(p);
  w[0] = v.x;
  w[1] = v.y;
}
template <>
SGL_DEVICE void kda_load_words<8>(const bf16_t* p, unsigned* w) {
  uint4 v = *reinterpret_cast<const uint4*>(p);
  w[0] = v.x;
  w[1] = v.y;
  w[2] = v.z;
  w[3] = v.w;
}
template <int CH>
SGL_DEVICE void kda_store_words(bf16_t* p, const unsigned* w);
template <>
SGL_DEVICE void kda_store_words<2>(bf16_t* p, const unsigned* w) {
  *reinterpret_cast<unsigned*>(p) = w[0];
}
template <>
SGL_DEVICE void kda_store_words<4>(bf16_t* p, const unsigned* w) {
  *reinterpret_cast<uint2*>(p) = make_uint2(w[0], w[1]);
}
template <>
SGL_DEVICE void kda_store_words<8>(bf16_t* p, const unsigned* w) {
  *reinterpret_cast<uint4*>(p) = make_uint4(w[0], w[1], w[2], w[3]);
}

template <int CH>
SGL_DEVICE void kda_load_words_cs(const bf16_t* p, unsigned* w) {
  if constexpr (CH == 2) {
    w[0] = __ldcs(reinterpret_cast<const unsigned*>(p));
  } else if constexpr (CH == 4) {
    uint2 v = __ldcs(reinterpret_cast<const uint2*>(p));
    w[0] = v.x;
    w[1] = v.y;
  } else {
    uint4 v = __ldcs(reinterpret_cast<const uint4*>(p));
    w[0] = v.x;
    w[1] = v.y;
    w[2] = v.z;
    w[3] = v.w;
  }
}
template <int CH>
SGL_DEVICE void kda_store_words_cs(bf16_t* p, const unsigned* w) {
  if constexpr (CH == 2) {
    __stcs(reinterpret_cast<unsigned*>(p), w[0]);
  } else if constexpr (CH == 4) {
    __stcs(reinterpret_cast<uint2*>(p), make_uint2(w[0], w[1]));
  } else {
    __stcs(reinterpret_cast<uint4*>(p), make_uint4(w[0], w[1], w[2], w[3]));
  }
}

// HINT bit0 = .cs on the x load, bit1 = .cs on the out store. The shipped
// configuration is HINT=0 (default cache policy); the .cs paths lost the
// cache-hint sweep and are kept only so the template surface matches the
// verified reference.
template <int CH, int HINT>
SGL_DEVICE void kda_load_words_h(const bf16_t* p, unsigned* w) {
  if constexpr (HINT & 1)
    kda_load_words_cs<CH>(p, w);
  else
    kda_load_words<CH>(p, w);
}
template <int CH, int HINT>
SGL_DEVICE void kda_store_channels_h(bf16_t* p, const unsigned* w) {
  if constexpr (HINT & 2)
    kda_store_words_cs<CH>(p, w);
  else
    kda_store_words<CH>(p, w);
}

// MODE 0: fma for all four taps (both parities) -- diagnostic only.
// MODE 1: replicate Triton's even/odd asymmetry <-- required for bit-exactness
template <int MODE>
SGL_DEVICE float kda_accumulate4(const float* w, float x0, float x1, float x2, float x3, bool odd) {
  float acc = __fmaf_rn(w[0], x0, 0.0f);
  acc = __fmaf_rn(w[1], x1, acc);
  if (MODE == 0 || !odd) {
    acc = __fmaf_rn(w[2], x2, acc);
    acc = __fmaf_rn(w[3], x3, acc);
  } else {
    float p2 = __fmul_rn(w[2], x2);
    float p3 = __fmul_rn(w[3], x3);
    acc = __fadd_rn(acc, p2);
    acc = __fadd_rn(acc, p3);
  }
  return acc;
}

// --------------------------------------------------------------------------
// software-pipelined token loop
//
// FULL == true  -> nt is statically NT (steady-state tile), every guard folds
// FULL == false -> ragged tail; guards stay dynamic but the buffer indices
//                  remain compile-time constants so the ring stays in
//                  registers (see header note 1).
// --------------------------------------------------------------------------
template <int CH, int NT, int MODE, int HINT, int PF, bool FULL>
SGL_DEVICE void kda_pipe_token_loop(
    const KdaShortConvParams& p,
    const float w[][4],
    float* h0,
    float* h1,
    float* h2,
    const bf16_t* xp,
    bf16_t* op,
    int nt) {
  constexpr int WRD = CH / 2;
  unsigned buf[PF][WRD];
  const bf16_t* pp = xp;

  // ---- prologue: fill the pipeline, PF LDGs in flight before any unpack ----
#pragma unroll
  for (int s = 0; s < PF; ++s) {
    if (FULL ? (s < NT) : (s < nt)) kda_load_words_h<CH, HINT>(pp, buf[s]);
    pp += p.x_row_stride;
  }

  // ---- steady state ----
#pragma unroll
  for (int i = 0; i < NT; ++i) {
    if (!FULL && i >= nt) break;
    // 1. unpack token i out of the register buffer (data landed PF iters ago)
    float xv[CH];
#pragma unroll
    for (int j = 0; j < WRD; ++j) kda_unpack_bf16x2(buf[i % PF][j], xv[2 * j], xv[2 * j + 1]);
    // 2. immediately re-issue that slot for token i+PF -> keeps PF LDGs in
    //    flight across the whole (long, div.full-dominated) compute below
    if (FULL ? (i + PF < NT) : (i + PF < nt)) {
      kda_load_words_h<CH, HINT>(pp, buf[i % PF]);
    }
    pp += p.x_row_stride;
    // 3. compute + store
    unsigned outw[CH / 2];
#pragma unroll
    for (int j = 0; j < CH / 2; ++j) {
      float a = kda_silu_triton(
          kda_accumulate4<MODE>(w[2 * j], h0[2 * j], h1[2 * j], h2[2 * j], xv[2 * j], false));
      float b = kda_silu_triton(kda_accumulate4<MODE>(
          w[2 * j + 1], h0[2 * j + 1], h1[2 * j + 1], h2[2 * j + 1], xv[2 * j + 1], true));
      outw[j] = kda_pack_bf16x2(a, b);
    }
    kda_store_channels_h<CH, HINT>(op, outw);
    op += p.out_tok;
#pragma unroll
    for (int j = 0; j < CH; ++j) {
      h0[j] = h1[j];
      h1[j] = h2[j];
      h2[j] = xv[j];
    }
  }
}

// grid  = (num_seq, ceil(max_seqlen / (NT*TG)), total_dim / (TX*CH))
// block = (TX, TG)
template <int CH, int NT, int TX, int TG, int MODE, int HINT, int PF>
__launch_bounds__(TX* TG) __global__ void kda_short_conv_kernel(KdaShortConvParams p) {
  const int s = blockIdx.x;
  const int line = p.ci[s];
  if (line == p.pad_slot_id) return;  // padded slot: no output, no state touch
  const int start = p.qsl[s];
  const int L = p.qsl[s + 1] - start;
  const int tile = blockIdx.y * (NT * TG);
  if (tile >= L) return;
  const int t0 = tile + (int)threadIdx.y * NT;
  if (t0 >= L) return;

  const int g = blockIdx.z * (TX * CH) + (int)threadIdx.x * CH;
  const int seg = g / p.dim;
  const int c = g - seg * p.dim;

  const bf16_t* xb = p.x[seg] + c + (int64_t)start * p.x_row_stride;
  bf16_t* ob = p.out[seg] + c + (int64_t)(start + t0) * p.out_tok;

  // weights: (dim, 4) row-major -> one 128-bit float4 per channel
  float w[CH][4];
  {
    const float* wb = p.w[seg] + (int64_t)c * 4;
#pragma unroll
    for (int j = 0; j < CH; ++j) {
      float4 t = *reinterpret_cast<const float4*>(wb + j * 4);
      w[j][0] = t.x;
      w[j][1] = t.y;
      w[j][2] = t.z;
      w[j][3] = t.w;
    }
  }

  // 3-tap history, unpacked to fp32 exactly once. All three word loads are
  // issued back-to-back before any unpack (same trick, applied to the prologue).
  float h0[CH], h1[CH], h2[CH];
  if (t0 == 0) {
    if (p.his[s]) {
      const bf16_t* stb = p.st[seg] + (int64_t)line * p.st_line + c;
      unsigned s0[CH / 2], s1[CH / 2], s2[CH / 2];
      kda_load_words<CH>(stb + 0 * p.st_tok, s0);
      kda_load_words<CH>(stb + 1 * p.st_tok, s1);
      kda_load_words<CH>(stb + 2 * p.st_tok, s2);
#pragma unroll
      for (int j = 0; j < CH / 2; ++j) {
        kda_unpack_bf16x2(s0[j], h0[2 * j], h0[2 * j + 1]);
        kda_unpack_bf16x2(s1[j], h1[2 * j], h1[2 * j + 1]);
        kda_unpack_bf16x2(s2[j], h2[2 * j], h2[2 * j + 1]);
      }
    } else {
#pragma unroll
      for (int j = 0; j < CH; ++j) {
        h0[j] = 0.0f;
        h1[j] = 0.0f;
        h2[j] = 0.0f;
      }
    }
  } else {
    const bf16_t* hp = xb + (int64_t)(t0 - 3) * p.x_row_stride;
    unsigned s0[CH / 2], s1[CH / 2], s2[CH / 2];
    kda_load_words<CH>(hp, s0);
    kda_load_words<CH>(hp + p.x_row_stride, s1);
    kda_load_words<CH>(hp + 2 * p.x_row_stride, s2);
#pragma unroll
    for (int j = 0; j < CH / 2; ++j) {
      kda_unpack_bf16x2(s0[j], h0[2 * j], h0[2 * j + 1]);
      kda_unpack_bf16x2(s1[j], h1[2 * j], h1[2 * j + 1]);
      kda_unpack_bf16x2(s2[j], h2[2 * j], h2[2 * j + 1]);
    }
  }

  const bf16_t* xp = xb + (int64_t)t0 * p.x_row_stride;
  const int nt = min(NT, L - t0);
  if (nt == NT) {
    kda_pipe_token_loop<CH, NT, MODE, HINT, PF, true>(p, w, h0, h1, h2, xp, ob, NT);
  } else {
    kda_pipe_token_loop<CH, NT, MODE, HINT, PF, false>(p, w, h0, h1, h2, xp, ob, nt);
  }
}

// conv-state shift-left update; stores the last 3 PRE-ACTIVATION input
// columns, so the next decode step reads the same state the Triton path would
// have written. Sequences shorter than 3 keep the tail of the old state.
// grid = (num_seq, ceil(total_dim / BT)), block = BT
template <int BT>
__global__ void kda_short_conv_state_update_kernel(KdaShortConvParams p) {
  const int s = blockIdx.x;
  const int line = p.ci[s];
  if (line == p.pad_slot_id) return;
  const int start = p.qsl[s];
  const int L = p.qsl[s + 1] - start;
  if (L <= 0) return;
  const int g = blockIdx.y * BT + (int)threadIdx.x;
  if (g >= p.total_dim) return;
  const int seg = g / p.dim;
  const int c = g - seg * p.dim;
  const bf16_t* xb = p.x[seg] + c;
  bf16_t* stb = p.st[seg] + (int64_t)line * p.st_line + c;

  bf16_t nv[3];
  if (L >= 3) {
#pragma unroll
    for (int i = 0; i < 3; ++i) nv[i] = xb[(int64_t)(start + L - 3 + i) * p.x_row_stride];
  } else {
    const int keep = 3 - L;
    const bool hi = p.his[s];
    for (int i = 0; i < keep; ++i)
      nv[i] = hi ? stb[(int64_t)(i + L) * p.st_tok] : __float2bfloat16(0.0f);
    for (int i = keep; i < 3; ++i) nv[i] = xb[(int64_t)(start + i - keep) * p.x_row_stride];
  }
#pragma unroll
  for (int i = 0; i < 3; ++i) stb[(int64_t)i * p.st_tok] = nv[i];
}

// --------------------------------------------------------------------------
// host launcher
//
// Takes the production tensors as they exist at the kda_backend call site:
//   mixed_qkv        (T, 3*dim) bf16, stride (row_stride, 1) with
//                    row_stride >= 3*dim -- row-major with optional padding,
//                    NOT necessarily contiguous. In production it is the first
//                    of `torch.split(fused_qkvg_states, [3*dim, dim], dim=-1)`,
//                    so row_stride == 4*dim. The three projections are channel
//                    slices of it, so the (dim, T) stride-(1, row_stride) views
//                    the Triton path builds are just base + seg*dim here.
//                    Never materialized / never made contiguous.
//   conv_weights     (3*dim, 4) fp32 contiguous
//   conv_states_pool (num_lines, 3, 3*dim) bf16 contiguous -- the pool as
//                    stored; the Triton path passes its transpose, whose last
//                    dim is not contiguous. st_line/st_tok below reproduce
//                    exactly that addressing.
//   out_q/k/v        (dim, T) bf16 with compact channel-last strides
//                    (1, dim) (what torch.empty_like gives for the input
//                    views); the caller transposes them back to (T, dim).
// The conv-state update is an in-place output: conv_states_pool is mutated.
// --------------------------------------------------------------------------
template <int CH, int NT, int TX, int TG, int MODE, int HINT, int PF, int kStateThreads>
struct KdaShortConvKernel {
  static constexpr auto kernel = kda_short_conv_kernel<CH, NT, TX, TG, MODE, HINT, PF>;
  static constexpr auto state_kernel = kda_short_conv_state_update_kernel<kStateThreads>;

  static void run(
      const tvm::ffi::TensorView mixed_qkv,
      const tvm::ffi::TensorView conv_weights,
      const tvm::ffi::TensorView conv_states_pool,
      const tvm::ffi::TensorView out_q,
      const tvm::ffi::TensorView out_k,
      const tvm::ffi::TensorView out_v,
      const tvm::ffi::TensorView query_start_loc,
      const tvm::ffi::TensorView cache_indices,
      const tvm::ffi::TensorView has_initial_state,
      const int64_t max_seqlen,
      const int64_t pad_slot_id) {
    using namespace host;

    auto T_ = SymbolicSize{"num_tokens"};
    auto TD_ = SymbolicSize{"total_dim"};
    auto XRS_ = SymbolicSize{"x_row_stride"};
    auto D_ = SymbolicSize{"dim"};
    auto NL_ = SymbolicSize{"num_cache_lines"};
    auto NS_ = SymbolicSize{"num_seqs"};
    auto NS1_ = SymbolicSize{"num_seqs_plus_one"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    // mixed_qkv may be ROW-STRIDED: in production it is
    // `torch.split(fused_qkvg_states, [3*dim, dim], dim=-1)[0]`, i.e. shape
    // (T, 3*dim) with stride (4*dim, 1) and is_contiguous() == False. The
    // channel axis must still be unit-stride; the row pitch is captured
    // symbolically (an unbound SymbolicSize in `with_strides` binds to the
    // observed stride) and checked below, so no explicit stride argument has
    // to be threaded through the FFI signature.
    TensorMatcher({T_, TD_})
        .with_strides({XRS_, 1})
        .with_dtype<bf16_t>()
        .with_device<kDLCUDA>(device)
        .verify(mixed_qkv);
    TensorMatcher({TD_, 4}).with_dtype<fp32_t>().with_device<kDLCUDA>(device).verify(conv_weights);
    TensorMatcher({NL_, 3, TD_})
        .with_dtype<bf16_t>()
        .with_device<kDLCUDA>(device)
        .verify(conv_states_pool);

    const int64_t total_dim = TD_.unwrap();
    RuntimeCheck(total_dim % 3 == 0, "total_dim must be 3 * dim, got ", total_dim);
    D_.set_value(total_dim / 3);

    // Row-major-with-padding contract for x. `with_strides({XRS_, 1})` already
    // pinned stride(1) == 1; the pitch only has to be wide enough that the
    // 3 * dim channels of one token cannot alias the next token's row.
    const int64_t x_row_stride = XRS_.unwrap();
    RuntimeCheck(
        x_row_stride >= total_dim,
        "mixed_qkv row stride (",
        x_row_stride,
        ") must be at least total_dim (",
        total_dim,
        "); rows would overlap");
    // x is read CH channels at a time with one CH*2-byte vector load at
    // base + c + t * x_row_stride, and c is always a multiple of CH, so the
    // pitch has to be a multiple of CH or the load goes misaligned from the
    // second token on. 3*dim and 4*dim both satisfy this (dim % (TX*CH) == 0).
    RuntimeCheck(
        x_row_stride % CH == 0,
        "mixed_qkv row stride (",
        x_row_stride,
        ") must be a multiple of the ",
        CH,
        "-channel vector load width");

    // (dim, T) with compact channel-last strides (1, dim)
    TensorMatcher({D_, T_})
        .with_strides({1, D_})
        .with_dtype<bf16_t>()
        .with_device<kDLCUDA>(device)
        .verify(out_q)
        .verify(out_k)
        .verify(out_v);

    TensorMatcher({NS1_}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(query_start_loc);
    TensorMatcher({NS_}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(cache_indices);
    // torch.bool maps to DLPack kDLBool, which has no C++ trait here, so the
    // dtype is checked by width instead of through the matcher.
    TensorMatcher({NS_}).with_device<kDLCUDA>(device).verify(has_initial_state);
    RuntimeCheck(
        has_initial_state.dtype().bits == 8 && has_initial_state.dtype().lanes == 1,
        "has_initial_state must be an 8-bit boolean tensor");

    const int64_t dim = D_.unwrap();
    const int64_t num_seq = NS_.unwrap();
    RuntimeCheck(
        NS1_.unwrap() == num_seq + 1,
        "query_start_loc must have num_seqs + 1 entries, got ",
        NS1_.unwrap(),
        " for ",
        num_seq,
        " sequences");

    constexpr int64_t kChannelsPerBlock = static_cast<int64_t>(TX) * CH;
    RuntimeCheck(
        total_dim % kChannelsPerBlock == 0 && dim % kChannelsPerBlock == 0,
        "the channel tile (",
        kChannelsPerBlock,
        ") must divide both dim (",
        dim,
        ") and total_dim (",
        total_dim,
        ")");

    if (num_seq == 0 || max_seqlen <= 0) return;

    auto p = KdaShortConvParams{};
    p.dim = static_cast<int32_t>(dim);
    p.total_dim = static_cast<int32_t>(total_dim);
    p.pad_slot_id = static_cast<int32_t>(pad_slot_id);
    p.x_row_stride = x_row_stride;  // 36864 contiguous / 49152 fused-qkvg view
    p.out_tok = out_q.stride(1);              // 12288
    p.st_line = conv_states_pool.stride(0);   // 110592
    p.st_tok = conv_states_pool.stride(1);    // 36864
    const auto* x_base = static_cast<const bf16_t*>(mixed_qkv.data_ptr());
    const auto* w_base = static_cast<const fp32_t*>(conv_weights.data_ptr());
    auto* st_base = static_cast<bf16_t*>(conv_states_pool.data_ptr());
    bf16_t* out_base[3] = {
        static_cast<bf16_t*>(out_q.data_ptr()),
        static_cast<bf16_t*>(out_k.data_ptr()),
        static_cast<bf16_t*>(out_v.data_ptr()),
    };
    for (int i = 0; i < 3; ++i) {
      p.x[i] = x_base + i * dim;
      p.w[i] = w_base + i * dim * 4;
      p.st[i] = st_base + i * dim;
      p.out[i] = out_base[i];
    }
    p.qsl = static_cast<const int32_t*>(query_start_loc.data_ptr());
    p.ci = static_cast<const int32_t*>(cache_indices.data_ptr());
    p.his = static_cast<const bool*>(has_initial_state.data_ptr());

    const auto dev = device.unwrap();
    const dim3 grid(
        static_cast<unsigned>(num_seq),
        static_cast<unsigned>(div_ceil<int64_t>(max_seqlen, static_cast<int64_t>(NT) * TG)),
        static_cast<unsigned>(total_dim / kChannelsPerBlock));
    LaunchKernel(grid, dim3(TX, TG), dev)(kernel, p);

    const dim3 state_grid(
        static_cast<unsigned>(num_seq),
        static_cast<unsigned>(div_ceil<int64_t>(total_dim, kStateThreads)));
    LaunchKernel(state_grid, kStateThreads, dev)(state_kernel, p);
  }
};

}  // namespace
