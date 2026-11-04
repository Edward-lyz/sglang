// Copy selected mixed-QKV rows directly into persistent convolution states.
// This replaces advanced-index gather + index_put with one bit-exact kernel.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

constexpr int kTrackThreads = 256;
constexpr int kTrackElementsPerVector = 8;

template <typename TrackIndexType, typename DestinationIndexType>
struct KdaTrackConvStateParams {
  const bf16_t* __restrict__ mixed_qkv;
  bf16_t* __restrict__ conv_state;
  const TrackIndexType* __restrict__ track_indices;
  const DestinationIndexType* __restrict__ destination_indices;
  int64_t mixed_qkv_stride_token;
  int64_t conv_state_stride_slot;
  int64_t conv_state_stride_window;
  int64_t track_stride_batch;
  int64_t track_stride_window;
  int64_t destination_stride;
  uint32_t state_dim;
};

template <int WindowSize, typename TrackIndexType, typename DestinationIndexType>
__global__ void kda_track_conv_state_kernel(
    const KdaTrackConvStateParams<TrackIndexType, DestinationIndexType> __grid_constant__ params) {
  const int64_t batch_index = blockIdx.y;
  const uint32_t channel = (blockIdx.x * blockDim.x + threadIdx.x) * kTrackElementsPerVector;
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

template <int WindowSize, typename TrackIndexType, typename DestinationIndexType>
struct KdaTrackConvStateKernel {
  static void
  run(const tvm::ffi::TensorView mixed_qkv,
      const tvm::ffi::TensorView conv_state,
      const tvm::ffi::TensorView track_indices,
      const tvm::ffi::TensorView destination_indices) {
    using namespace host;

    auto num_tokens = SymbolicSize{"num_tokens"};
    auto num_tracks = SymbolicSize{"num_tracks"};
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
    // padded slice, which is exactly what can_fuse_track_conv_state allows.
    TensorMatcher({-1, window_size, state_dim})
        .with_strides({-1, -1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(conv_state);
    TensorMatcher({num_tracks, window_size}).with_dtype<TrackIndexType>().with_device(device).verify(track_indices);
    TensorMatcher({num_tracks}).with_dtype<DestinationIndexType>().with_device(device).verify(destination_indices);

    const int64_t tracks = num_tracks.unwrap();
    const int64_t dim = state_dim.unwrap();
    RuntimeCheck(conv_state.stride(2) == 1, "KDA conv-state channel dimension must be contiguous");
    RuntimeCheck(dim % kTrackElementsPerVector == 0, "KDA state dimension must be divisible by 8");
    RuntimeCheck(
        mixed_qkv.stride(0) % kTrackElementsPerVector == 0, "KDA mixed-QKV row stride must preserve 16-byte alignment");
    RuntimeCheck(
        conv_state.stride(0) % kTrackElementsPerVector == 0 && conv_state.stride(1) % kTrackElementsPerVector == 0,
        "KDA conv-state strides must preserve 16-byte alignment");
    RuntimeCheck(
        reinterpret_cast<uintptr_t>(mixed_qkv.data_ptr()) % 16 == 0, "KDA mixed-QKV pointer must be 16-byte aligned");
    RuntimeCheck(
        reinterpret_cast<uintptr_t>(conv_state.data_ptr()) % 16 == 0, "KDA conv-state pointer must be 16-byte aligned");

    if (tracks == 0 || dim == 0) return;

    const auto params = KdaTrackConvStateParams<TrackIndexType, DestinationIndexType>{
        .mixed_qkv = static_cast<const bf16_t*>(mixed_qkv.data_ptr()),
        .conv_state = static_cast<bf16_t*>(conv_state.data_ptr()),
        .track_indices = static_cast<const TrackIndexType*>(track_indices.data_ptr()),
        .destination_indices = static_cast<const DestinationIndexType*>(destination_indices.data_ptr()),
        .mixed_qkv_stride_token = mixed_qkv.stride(0),
        .conv_state_stride_slot = conv_state.stride(0),
        .conv_state_stride_window = conv_state.stride(1),
        .track_stride_batch = track_indices.stride(0),
        .track_stride_window = track_indices.stride(1),
        .destination_stride = destination_indices.stride(0),
        .state_dim = static_cast<uint32_t>(dim),
    };

    const uint32_t vectors = static_cast<uint32_t>(dim / kTrackElementsPerVector);
    const dim3 grid{
        div_ceil(vectors, static_cast<uint32_t>(kTrackThreads)),
        static_cast<uint32_t>(tracks),
    };
    const dim3 block{kTrackThreads};
    LaunchKernel(grid, block, device.unwrap())(
        kda_track_conv_state_kernel<WindowSize, TrackIndexType, DestinationIndexType>, params);
  }
};

}  // namespace
