import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.quantization.mla_qkv_fp8_cast import quantize_mla_qkv_fp8_triton
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=6, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

FP8 = torch.float8_e4m3fn
QK_HEAD_DIM = 192
NOPE_DIM = 128
KV_LAST_DIM = 256


def torch_casts(q, k, v):
    return q.to(FP8), k.to(FP8), v.to(FP8)


def triton_gated(q, k, v):
    return quantize_mla_qkv_fp8_triton(q, k, v)


def triton_forced(q, k, v):
    # Bypass the size gate so the small shapes show the kernel's own cost rather
    # than the torch fallback the gate would pick.
    return quantize_mla_qkv_fp8_triton(q, k, v, force_kernel=True)


FN_MAP = {
    "torch_casts": torch_casts,
    "triton_gated": triton_gated,
    "triton_forced": triton_forced,
}


@marker.parametrize("num_tokens", [4097, 8192, 16384, 32768], [4097, 32768])
@marker.parametrize("heads", [12, 96])
@marker.benchmark("impl", ["torch_casts", "triton_gated", "triton_forced"])
def benchmark(num_tokens: int, heads: int, impl: str):
    q = torch.randn(
        num_tokens, heads, QK_HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    k = torch.randn(
        num_tokens, heads, QK_HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    # v must be the trailing slice of a contiguous kv tensor, which is what makes
    # the torch cast fall back to the unvectorized elementwise kernel.
    kv = torch.randn(
        num_tokens, heads, KV_LAST_DIM, dtype=torch.bfloat16, device="cuda"
    )
    v = kv[..., NOPE_DIM:]
    return marker.do_bench(FN_MAP[impl], input_args=(q, k, v))


if __name__ == "__main__":
    benchmark.run()
