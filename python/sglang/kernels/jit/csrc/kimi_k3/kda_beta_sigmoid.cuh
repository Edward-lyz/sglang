// Convert a strided BF16 beta slice to contiguous FP32 and apply sigmoid.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

constexpr int kBetaThreads = 256;
constexpr int kBetaVectorSize = 4;

struct KdaBetaSigmoidParams {
  const bf16_t* __restrict__ beta;
  fp32_t* __restrict__ output;
  int64_t beta_stride_token;
  uint32_t vectors_per_token;
  uint32_t num_vectors;
};

__global__ void kda_beta_sigmoid_kernel(const KdaBetaSigmoidParams __grid_constant__ params) {
  using namespace device;

  using input_vector_t = AlignedVector<bf16_t, kBetaVectorSize>;
  using output_vector_t = AlignedVector<fp32_t, kBetaVectorSize>;

  const uint32_t vector_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (vector_index >= params.num_vectors) return;

  const uint32_t token_index = vector_index / params.vectors_per_token;
  const uint32_t vector_in_token = vector_index - token_index * params.vectors_per_token;

  input_vector_t input;
  output_vector_t output;
  input.load(params.beta + token_index * params.beta_stride_token, vector_in_token);

#pragma unroll
  for (int i = 0; i < kBetaVectorSize; ++i) {
    const float value = cast<fp32_t>(input[i]);
    output[i] = 1.0f / (1.0f + expf(-value));
  }
  output.store(params.output, vector_index);
}

struct KdaBetaSigmoidKernel {
  static void run(const tvm::ffi::TensorView beta, const tvm::ffi::TensorView output) {
    using namespace host;

    auto num_tokens = SymbolicSize{"num_tokens"};
    auto num_heads = SymbolicSize{"num_heads"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({num_tokens, num_heads}).with_strides({-1, 1}).with_dtype<bf16_t>().with_device(device).verify(beta);
    TensorMatcher({num_tokens, num_heads}).with_dtype<fp32_t>().with_device(device).verify(output);

    const int64_t tokens = num_tokens.unwrap();
    const int64_t heads = num_heads.unwrap();
    RuntimeCheck(output.stride(1) == 1 && output.stride(0) == heads, "KDA beta output must be contiguous");
    RuntimeCheck(heads % kBetaVectorSize == 0, "KDA beta head dimension must be divisible by 4");
    RuntimeCheck(beta.stride(0) % kBetaVectorSize == 0, "KDA beta token stride must preserve vector alignment");
    RuntimeCheck(reinterpret_cast<uintptr_t>(beta.data_ptr()) % 8 == 0, "KDA beta pointer must be 8-byte aligned");
    RuntimeCheck(
        reinterpret_cast<uintptr_t>(output.data_ptr()) % 16 == 0, "KDA beta output pointer must be 16-byte aligned");

    if (tokens == 0 || heads == 0) return;

    const uint32_t vectors_per_token = static_cast<uint32_t>(heads / kBetaVectorSize);
    const uint32_t num_vectors = static_cast<uint32_t>(tokens * vectors_per_token);
    const auto params = KdaBetaSigmoidParams{
        .beta = static_cast<const bf16_t*>(beta.data_ptr()),
        .output = static_cast<fp32_t*>(output.data_ptr()),
        .beta_stride_token = beta.stride(0),
        .vectors_per_token = vectors_per_token,
        .num_vectors = num_vectors,
    };

    const uint32_t blocks = div_ceil(num_vectors, static_cast<uint32_t>(kBetaThreads));
    LaunchKernel(blocks, kBetaThreads, device.unwrap())(kda_beta_sigmoid_kernel, params);
  }
};

}  // namespace
