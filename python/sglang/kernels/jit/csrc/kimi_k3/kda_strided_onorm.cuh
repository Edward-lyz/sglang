// Fused strided gated RMSNorm for the KDA extend/prefill output tail.
//
// Replaces two kernels in the KDA extend path: an ATen layout copy (forced
// by `x.reshape(-1, D)` on a non-contiguous tensor) followed by a Triton
// gated RMSNorm. This kernel reads `x` through its real strides and writes a
// fresh contiguous [T, H*D] output in one pass, so the copy disappears.
//
// Semantics (fp32 reductions, no mean subtraction):
//   xf   = x[0, t, h, :].float()
//   rstd = 1 / sqrt(mean(xf * xf) + eps)
//   y    = xf * rstd * weight.float()
//   gf   = g[0, t, h, :].float()
//   act  = gf * sigmoid(gf)   if activation == swish/silu
//          sigmoid(gf)        if activation == sigmoid
//   out[t, h*D : (h+1)*D] = (y * act).to(x.dtype)
//
// Two dispatch paths, both templated and selected at runtime from the input
// strides (never hardcoded):
//   - Path A (SMEM transpose): x's token axis has stride 1 (the KDA extend
//     case, where x aliases the transposed conv-output buffer). A
//     [TILE_T][D+1] shared-memory tile is staged with vectorized loads
//     walking the stride-1 token axis, then each warp-group reduces one
//     (token, head) row out of the tile.
//   - Path B (flat): x, g and out are all row-major [T*H, D] with matching
//     row stride D (x contiguous). No shared memory, no (t, h) decomposition.
//
// This intentionally ports only the two paths relevant to the KDA extend
// call site (strided-token SMEM path + contiguous flat path); the
// reference's rowwise and fully-generic scalar fallbacks are not needed here
// because the JIT wrapper's covered() predicate only accepts the two
// stride patterns handled below.

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck, div_ceil

#include <sgl_kernel/type.cuh>   // For bf16_t, fp16_t, fp32_t, device::cast
#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE, PDL helpers
#include <sgl_kernel/vec.cuh>    // For AlignedVector

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

// activation ids, kept in sync with the Python wrapper
constexpr int64_t kActSwish = 0;
constexpr int64_t kActSigmoid = 1;

SGL_DEVICE float sigmoid_f32(float v) {
  return 1.0f / (1.0f + expf(-v));
}

template <int kAct>
SGL_DEVICE float gate_activation(float gf) {
  const float s = sigmoid_f32(gf);
  if constexpr (kAct == kActSwish) {
    return gf * s;
  } else {
    return s;
  }
}

// reduce sum-of-squares across LPR contiguous lanes (LPR a power of two <= 32)
template <int LPR>
SGL_DEVICE float group_sum(float v) {
#pragma unroll
  for (int off = LPR >> 1; off > 0; off >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, off);
  }
  return v;
}

// ---------------------------------------------------------------------------
// Path A: SMEM-tiled transpose fused into the norm (x token axis stride 1).
// ---------------------------------------------------------------------------
template <typename T, int D, int kAct, int TILE_T, int NWARP, int VT, int EPL>
struct KdaStridedOnormSmemParams {
  const T* __restrict__ x;       // strided [1, T, H, D]; only x_t, x_h, x_d given
  const T* __restrict__ g;       // contiguous-per-row [1, T, H, D]
  const T* __restrict__ weight;  // [D] contiguous
  T* __restrict__ out;           // fresh contiguous [T, H*D]
  int32_t Tlen;
  float eps;
  int64_t sx_t;
  int64_t sx_h;
  int64_t sx_d;
  int64_t sg_t;
  int64_t sg_h;
  int64_t so_t;
  int64_t so_h;
};

template <typename T, int D, int kAct, int TILE_T, int NWARP, int VT, int EPL, bool kUsePDL>
__global__ __launch_bounds__(NWARP * 32) void kda_strided_onorm_smem_kernel(
    const __grid_constant__ KdaStridedOnormSmemParams<T, D, kAct, TILE_T, NWARP, VT, EPL> params) {
  using namespace device;

  constexpr int LD = D + 1;        // odd padding: 2-way instead of 32-way SMEM banking
  constexpr int TC = TILE_T / VT;  // token chunks per head_dim column
  constexpr int NT = NWARP * 32;
  constexpr int LPR = D / EPL;  // lanes cooperating on one (token, head) row
  constexpr int RPW = 32 / LPR;  // rows handled per warp per pass
  static_assert(TILE_T % VT == 0, "TILE_T must be a multiple of VT");
  static_assert(D % EPL == 0 && LPR <= 32 && 32 % LPR == 0, "D/EPL must divide the warp evenly");

  __shared__ T tile[TILE_T * LD];

  // grid is (H, ceil(T/TILE_T)): blockIdx.x is the head so concurrently
  // scheduled blocks cover every head of the same token tile, and their g/out
  // traffic forms one contiguous H*D element region per token.
  const int32_t t0 = blockIdx.y * TILE_T;
  const int32_t h = static_cast<int32_t>(blockIdx.x);
  const T* xb = params.x + static_cast<int64_t>(h) * params.sx_h;

  using vec_vt_t = AlignedVector<T, VT>;

  // stage tile: coalesced vectorized reads along the stride-1 token axis
  for (int32_t idx = threadIdx.x; idx < D * TC; idx += NT) {
    const int32_t tc = idx % TC;
    const int32_t d = idx / TC;
    const int32_t t = t0 + tc * VT;
    const T* src = xb + static_cast<int64_t>(t) * params.sx_t + static_cast<int64_t>(d) * params.sx_d;
    if (t + VT <= params.Tlen) {
      vec_vt_t v;
      v.load(src);
#pragma unroll
      for (int i = 0; i < VT; ++i) {
        tile[(tc * VT + i) * LD + d] = v[i];
      }
    } else {
#pragma unroll
      for (int i = 0; i < VT; ++i) {
        if (t + i < params.Tlen) tile[(tc * VT + i) * LD + d] = src[i];
      }
    }
  }
  __syncthreads();

  const int32_t warp = static_cast<int32_t>(threadIdx.x >> 5);
  const int32_t lane = static_cast<int32_t>(threadIdx.x & 31);
  const int32_t sub = lane / LPR;  // row within this warp's row group
  const int32_t li = lane % LPR;
  const int32_t dbase = li * EPL;

  using vec_epl_t = AlignedVector<T, EPL>;

  float wv[EPL];
  {
    vec_epl_t wvec;
    wvec.load(params.weight + dbase);
#pragma unroll
    for (int k = 0; k < EPL; ++k) wv[k] = cast<fp32_t>(wvec[k]);
  }

  PDLWaitPrimary<kUsePDL>();

  for (int32_t r = warp * RPW + sub; r < TILE_T; r += NWARP * RPW) {
    const int32_t t = t0 + r;
    if (t >= params.Tlen) break;

    // scalar SMEM reads: LD is odd, so no 4/8/16B alignment can be assumed
    const T* srow = &tile[r * LD + dbase];
    float xf[EPL];
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < EPL; ++k) {
      xf[k] = cast<fp32_t>(srow[k]);
      ss += xf[k] * xf[k];
    }
    ss = group_sum<LPR>(ss);
    const float rstd = 1.0f / sqrtf(ss / static_cast<float>(D) + params.eps);

    vec_epl_t gv;
    gv.load(params.g + static_cast<int64_t>(t) * params.sg_t + static_cast<int64_t>(h) * params.sg_h + dbase);
    vec_epl_t ov;
#pragma unroll
    for (int k = 0; k < EPL; ++k) {
      const float gf = cast<fp32_t>(gv[k]);
      ov[k] = cast<T>(xf[k] * rstd * wv[k] * gate_activation<kAct>(gf));
    }
    ov.store(params.out + static_cast<int64_t>(t) * params.so_t + static_cast<int64_t>(h) * params.so_h + dbase);
  }

  PDLTriggerSecondary<kUsePDL>();
}

// ---------------------------------------------------------------------------
// Path B: flat contiguous x/g/out, all row-major [T*H, D] with row stride D.
// ---------------------------------------------------------------------------
template <typename T, int D, int kAct, int EPL>
struct KdaStridedOnormFlatParams {
  const T* __restrict__ x;
  const T* __restrict__ g;
  const T* __restrict__ weight;
  T* __restrict__ out;
  int64_t nrows;
  int32_t H;
  float eps;
  // g's (token, head) strides -- read generically because g may be a column
  // slice of a wider fused tensor (e.g. TP-packed g_proj), whose token-axis
  // stride is then a multiple of H*D rather than exactly H*D. x is always
  // literally row-major here (that is the flat path's dispatch condition),
  // so only g needs runtime strides.
  int64_t sg_t;
  int64_t sg_h;
};

template <typename T, int D, int kAct, int NWARP, int EPL, bool kUsePDL>
__global__ __launch_bounds__(NWARP * 32) void kda_strided_onorm_flat_kernel(
    const __grid_constant__ KdaStridedOnormFlatParams<T, D, kAct, EPL> params) {
  using namespace device;

  constexpr int LPR = D / EPL;
  constexpr int RPW = 32 / LPR;
  constexpr int RPB = NWARP * RPW;
  static_assert(D % EPL == 0 && LPR <= 32 && 32 % LPR == 0, "D/EPL must divide the warp evenly");

  using vec_epl_t = AlignedVector<T, EPL>;

  const int32_t lane = static_cast<int32_t>(threadIdx.x & 31);
  const int32_t dbase = (lane % LPR) * EPL;

  float wv[EPL];
  {
    vec_epl_t wvec;
    wvec.load(params.weight + dbase);
#pragma unroll
    for (int k = 0; k < EPL; ++k) wv[k] = cast<fp32_t>(wvec[k]);
  }

  PDLWaitPrimary<kUsePDL>();

  const int64_t stride = static_cast<int64_t>(gridDim.x) * RPB;
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * RPB + (threadIdx.x >> 5) * RPW + (lane / LPR);
       row < params.nrows;
       row += stride) {
    const int64_t off = row * D + dbase;
    vec_epl_t xv;
    xv.load(params.x + off);
    float xf[EPL];
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < EPL; ++k) {
      xf[k] = cast<fp32_t>(xv[k]);
      ss += xf[k] * xf[k];
    }
    ss = group_sum<LPR>(ss);
    const float rstd = 1.0f / sqrtf(ss / static_cast<float>(D) + params.eps);

    // row decomposes into (t, h) to address g through its own (possibly
    // non-row-major) strides -- x/out stay flat since the dispatch
    // condition guarantees those are literally [T*H, D] row-major.
    const int32_t h = static_cast<int32_t>(row % params.H);
    const int64_t t = row / params.H;
    const T* grow = params.g + t * params.sg_t + static_cast<int64_t>(h) * params.sg_h;
    vec_epl_t gv;
    gv.load(grow + dbase);
    vec_epl_t ov;
#pragma unroll
    for (int k = 0; k < EPL; ++k) {
      const float gf = cast<fp32_t>(gv[k]);
      ov[k] = cast<T>(xf[k] * rstd * wv[k] * gate_activation<kAct>(gf));
    }
    ov.store(params.out + off);
  }

  PDLTriggerSecondary<kUsePDL>();
}

// ---------------------------------------------------------------------------
// Host dispatch
// ---------------------------------------------------------------------------

template <typename T, int D, bool kUsePDL>
struct KdaStridedOnormKernel {
  static constexpr int kTileT = 128;
  static constexpr int kNWarp = 8;
  static constexpr int kVT = 8;
  static constexpr int kEpl = 8;
  static constexpr int kThreads = kNWarp * 32;

  static void run(
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView g,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView out,
      int64_t act,
      double eps) {
    using namespace host;

    RuntimeCheck(act == kActSwish || act == kActSigmoid, "act must be 0 (swish) or 1 (sigmoid)");

    auto Tlen_ = SymbolicSize{"Tlen"};
    auto H_ = SymbolicSize{"H"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    // x/g: [1, T, H, D]. Strides are read and dispatched on manually below
    // (SMEM path wants x's token axis stride 1; flat path wants full
    // contiguity), so TensorMatcher is told to accept ANY stride here via
    // -1 placeholders -- the real stride legality checks are the
    // RuntimeCheck calls further down, after Tlen/H are known.
    TensorMatcher({1, Tlen_, H_, D})  //
        .with_strides({-1, -1, -1, -1})
        .with_dtype<T>()
        .with_device(device_)
        .verify(x);
    TensorMatcher({1, Tlen_, H_, D})  //
        .with_strides({-1, -1, -1, -1})
        .with_dtype<T>()
        .with_device(device_)
        .verify(g);
    TensorMatcher({D})  //
        .with_dtype<T>()
        .with_device(device_)
        .verify(weight);

    const int32_t Tlen = static_cast<int32_t>(Tlen_.unwrap());
    const int32_t H = static_cast<int32_t>(H_.unwrap());
    const auto device = device_.unwrap();
    if (Tlen == 0 || H == 0) return;

    // out is [T, H*D]; H*D is only known after unwrapping H above, so verify
    // its shape with literal ints rather than folding it into the symbolic
    // matcher used for x/g (SymbolicSize has no operator*).
    TensorMatcher({static_cast<int64_t>(Tlen), static_cast<int64_t>(H) * D})  //
        .with_dtype<T>()
        .with_device(device_)
        .verify(out);

    const int64_t sx_t = x.stride(1);
    const int64_t sx_h = x.stride(2);
    const int64_t sx_d = x.stride(3);
    const int64_t sg_t = g.stride(1);
    const int64_t sg_h = g.stride(2);
    const int64_t sg_d = g.stride(3);
    RuntimeCheck(sg_d == 1, "g's head_dim axis must be contiguous");

    const int64_t so_t = static_cast<int64_t>(H) * D;
    const int64_t so_h = D;

    const bool x_token_stride1 =
        (sx_t == 1) && (sx_d % kVT == 0) && (sx_h % kVT == 0) && (Tlen % kVT == 0);

    if (x_token_stride1) {
      const auto params = KdaStridedOnormSmemParams<T, D, kActSwish, kTileT, kNWarp, kVT, kEpl>{
          .x = static_cast<const T*>(x.data_ptr()),
          .g = static_cast<const T*>(g.data_ptr()),
          .weight = static_cast<const T*>(weight.data_ptr()),
          .out = static_cast<T*>(out.data_ptr()),
          .Tlen = Tlen,
          .eps = static_cast<float>(eps),
          .sx_t = sx_t,
          .sx_h = sx_h,
          .sx_d = sx_d,
          .sg_t = sg_t,
          .sg_h = sg_h,
          .so_t = so_t,
          .so_h = so_h,
      };
      const dim3 grid(H, (Tlen + kTileT - 1) / kTileT);
      if (act == kActSwish) {
        LaunchKernel(grid, kThreads, device).enable_pdl(kUsePDL)(
            kda_strided_onorm_smem_kernel<T, D, kActSwish, kTileT, kNWarp, kVT, kEpl, kUsePDL>, params);
      } else {
        // reinterpret params with the sigmoid activation id (layout identical)
        const auto params_sig = KdaStridedOnormSmemParams<T, D, kActSigmoid, kTileT, kNWarp, kVT, kEpl>{
            .x = params.x,
            .g = params.g,
            .weight = params.weight,
            .out = params.out,
            .Tlen = params.Tlen,
            .eps = params.eps,
            .sx_t = params.sx_t,
            .sx_h = params.sx_h,
            .sx_d = params.sx_d,
            .sg_t = params.sg_t,
            .sg_h = params.sg_h,
            .so_t = params.so_t,
            .so_h = params.so_h,
        };
        LaunchKernel(grid, kThreads, device).enable_pdl(kUsePDL)(
            kda_strided_onorm_smem_kernel<T, D, kActSigmoid, kTileT, kNWarp, kVT, kEpl, kUsePDL>, params_sig);
      }
      return;
    }

    // Path B: flat -- x row-major [T*H, D] contiguous; g only needs its
    // head_dim axis contiguous (sg_d == 1, already RuntimeCheck'd above),
    // its token/head axes are read through runtime strides so a TP-sliced
    // or otherwise non-row-major g (sg_h > D, sg_t a multiple of H*sg_h
    // rather than exactly H*D) is still served without a contiguity copy.
    RuntimeCheck(sx_d == 1, "flat path requires x's head_dim axis contiguous");
    RuntimeCheck(sx_h == D, "flat path requires x row-major [T*H, D]");
    RuntimeCheck(sx_t == static_cast<int64_t>(H) * D, "flat path requires x row-major [T*H, D]");

    const int64_t nrows = static_cast<int64_t>(Tlen) * H;
    constexpr int kRpb = kNWarp * (32 / (D / kEpl));
    const uint32_t n_blocks = static_cast<uint32_t>((nrows + kRpb - 1) / kRpb);

    if (act == kActSwish) {
      const auto params = KdaStridedOnormFlatParams<T, D, kActSwish, kEpl>{
          .x = static_cast<const T*>(x.data_ptr()),
          .g = static_cast<const T*>(g.data_ptr()),
          .weight = static_cast<const T*>(weight.data_ptr()),
          .out = static_cast<T*>(out.data_ptr()),
          .nrows = nrows,
          .H = H,
          .eps = static_cast<float>(eps),
          .sg_t = sg_t,
          .sg_h = sg_h,
      };
      LaunchKernel(n_blocks, kThreads, device).enable_pdl(kUsePDL)(
          kda_strided_onorm_flat_kernel<T, D, kActSwish, kNWarp, kEpl, kUsePDL>, params);
    } else {
      const auto params = KdaStridedOnormFlatParams<T, D, kActSigmoid, kEpl>{
          .x = static_cast<const T*>(x.data_ptr()),
          .g = static_cast<const T*>(g.data_ptr()),
          .weight = static_cast<const T*>(weight.data_ptr()),
          .out = static_cast<T*>(out.data_ptr()),
          .nrows = nrows,
          .H = H,
          .eps = static_cast<float>(eps),
          .sg_t = sg_t,
          .sg_h = sg_h,
      };
      LaunchKernel(n_blocks, kThreads, device).enable_pdl(kUsePDL)(
          kda_strided_onorm_flat_kernel<T, D, kActSigmoid, kNWarp, kEpl, kUsePDL>, params);
    }
  }
};

}  // namespace
