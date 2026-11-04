// Round-1 candidate C3: KDA extend conv-state tracking + has_initial_state flag
// in a single launch. Replaces ops #7/#8/#9 (compare_scalar + vectorized_gather
// + index_put) with one kernel.
//
// Block (0, 0) additionally writes has_initial_state[i] = extend_prefix_lens[i] > 0;
// the flag is independent of the copy, so folding it in costs a handful of
// threads and removes a whole launch.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

constexpr int kTrackFlagThreads = 256;
constexpr int kTrackFlagElementsPerVector = 8;

template <typename TrackIndexType, typename DestinationIndexType, typename PrefixLenType>
struct KdaTrackConvStateFlagParams {
  const bf16_t* __restrict__ mixed_qkv;
  bf16_t* __restrict__ conv_state;
  const TrackIndexType* __restrict__ track_indices;
  const DestinationIndexType* __restrict__ destination_indices;
  const PrefixLenType* __restrict__ prefix_lens;
  uint8_t* __restrict__ has_initial_state;
  int64_t mixed_qkv_stride_token;
  int64_t conv_state_stride_slot;
  int64_t conv_state_stride_window;
  int64_t track_stride_batch;
  int64_t track_stride_window;
  int64_t destination_stride;
  int64_t prefix_lens_stride;
  uint32_t state_dim;
  uint32_t num_tracks;
  uint32_t num_seqs;
};

template <int WindowSize, typename TrackIndexType, typename DestinationIndexType, typename PrefixLenType>
__global__ void kda_track_conv_state_flag_kernel(
    const KdaTrackConvStateFlagParams<TrackIndexType, DestinationIndexType, PrefixLenType>
        __grid_constant__ params) {
  // Folded op #7: extend_prefix_lens > 0.
  if (blockIdx.x == 0 && blockIdx.y == 0 && params.has_initial_state != nullptr) {
    for (uint32_t i = threadIdx.x; i < params.num_seqs; i += blockDim.x) {
      params.has_initial_state[i] = params.prefix_lens[i * params.prefix_lens_stride] > 0 ? 1 : 0;
    }
  }

  const int64_t batch_index = blockIdx.y;
  if (batch_index >= params.num_tracks) return;
  const uint32_t channel = (blockIdx.x * blockDim.x + threadIdx.x) * kTrackFlagElementsPerVector;
  if (channel >= params.state_dim) return;

  const int64_t destination_slot =
      static_cast<int64_t>(params.destination_indices[batch_index * params.destination_stride]);
  const int64_t destination_base = destination_slot * params.conv_state_stride_slot + channel;
  const int64_t track_base = batch_index * params.track_stride_batch;

#pragma unroll
  for (int window_index = 0; window_index < WindowSize; ++window_index) {
    const int64_t source_token =
        static_cast<int64_t>(params.track_indices[track_base + window_index * params.track_stride_window]);
    const uint4 value =
        *reinterpret_cast<const uint4*>(params.mixed_qkv + source_token * params.mixed_qkv_stride_token + channel);
    *reinterpret_cast<uint4*>(params.conv_state + destination_base + window_index * params.conv_state_stride_window) =
        value;
  }
}

template <int WindowSize, typename TrackIndexType, typename DestinationIndexType, typename PrefixLenType>
struct KdaTrackConvStateFlagKernel {
  static void
  run(const tvm::ffi::TensorView mixed_qkv,
      const tvm::ffi::TensorView conv_state,
      const tvm::ffi::TensorView track_indices,
      const tvm::ffi::TensorView destination_indices,
      const tvm::ffi::TensorView prefix_lens,
      const tvm::ffi::TensorView has_initial_state) {
    using namespace host;

    auto num_tokens = SymbolicSize{"num_tokens"};
    auto num_tracks = SymbolicSize{"num_tracks"};
    auto num_seqs = SymbolicSize{"num_seqs"};
    auto state_dim = SymbolicSize{"state_dim"};
    auto window_size = SymbolicSize{"window_size"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    window_size.set_value(WindowSize);

    TensorMatcher({num_tokens, state_dim})
        .with_strides({-1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(mixed_qkv);
    // Only the channel axis must be unit-stride: the pool storage may be a
    // padded slice, which is what can_fuse_track_conv_state_flag allows.
    TensorMatcher({-1, window_size, state_dim})
        .with_strides({-1, -1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(conv_state);
    TensorMatcher({num_tracks, window_size}).with_dtype<TrackIndexType>().with_device(device).verify(track_indices);
    TensorMatcher({num_tracks}).with_dtype<DestinationIndexType>().with_device(device).verify(destination_indices);
    TensorMatcher({num_seqs}).with_dtype<PrefixLenType>().with_device(device).verify(prefix_lens);
    TensorMatcher({num_seqs}).with_dtype<uint8_t>().with_device(device).verify(has_initial_state);

    const int64_t tracks = num_tracks.unwrap();
    const int64_t seqs = num_seqs.unwrap();
    const int64_t dim = state_dim.unwrap();
    RuntimeCheck(conv_state.stride(2) == 1, "KDA conv-state channel dimension must be contiguous");
    RuntimeCheck(has_initial_state.stride(0) == 1, "KDA has_initial_state must be contiguous");
    RuntimeCheck(dim % kTrackFlagElementsPerVector == 0, "KDA state dimension must be divisible by 8");
    RuntimeCheck(
        mixed_qkv.stride(0) % kTrackFlagElementsPerVector == 0, "KDA mixed-QKV row stride must stay 16B aligned");
    RuntimeCheck(
        conv_state.stride(0) % kTrackFlagElementsPerVector == 0 &&
            conv_state.stride(1) % kTrackFlagElementsPerVector == 0,
        "KDA conv-state strides must stay 16B aligned");
    RuntimeCheck(
        reinterpret_cast<uintptr_t>(mixed_qkv.data_ptr()) % 16 == 0, "KDA mixed-QKV pointer must be 16-byte aligned");
    RuntimeCheck(
        reinterpret_cast<uintptr_t>(conv_state.data_ptr()) % 16 == 0, "KDA conv-state pointer must be 16-byte aligned");

    if (dim == 0 || (tracks == 0 && seqs == 0)) return;

    const auto params = KdaTrackConvStateFlagParams<TrackIndexType, DestinationIndexType, PrefixLenType>{
        .mixed_qkv = static_cast<const bf16_t*>(mixed_qkv.data_ptr()),
        .conv_state = static_cast<bf16_t*>(conv_state.data_ptr()),
        .track_indices = static_cast<const TrackIndexType*>(track_indices.data_ptr()),
        .destination_indices = static_cast<const DestinationIndexType*>(destination_indices.data_ptr()),
        .prefix_lens = static_cast<const PrefixLenType*>(prefix_lens.data_ptr()),
        .has_initial_state = static_cast<uint8_t*>(has_initial_state.data_ptr()),
        .mixed_qkv_stride_token = mixed_qkv.stride(0),
        .conv_state_stride_slot = conv_state.stride(0),
        .conv_state_stride_window = conv_state.stride(1),
        .track_stride_batch = track_indices.stride(0),
        .track_stride_window = track_indices.stride(1),
        .destination_stride = destination_indices.stride(0),
        .prefix_lens_stride = prefix_lens.stride(0),
        .state_dim = static_cast<uint32_t>(dim),
        .num_tracks = static_cast<uint32_t>(tracks),
        .num_seqs = static_cast<uint32_t>(seqs),
    };

    const uint32_t vectors = static_cast<uint32_t>(dim / kTrackFlagElementsPerVector);
    const dim3 grid{
        div_ceil(vectors, static_cast<uint32_t>(kTrackFlagThreads)),
        static_cast<uint32_t>(tracks > 0 ? tracks : 1),
    };
    LaunchKernel(grid, dim3{kTrackFlagThreads}, device.unwrap())(
        kda_track_conv_state_flag_kernel<WindowSize, TrackIndexType, DestinationIndexType, PrefixLenType>, params);
  }
};

}  // namespace
