"""Fused MLA FP8 decode prepare vs the concat + three-cast aten chain it replaces."""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.attention.fused_mla_prepare_fp8 import fused_mla_prepare_fp8
from sglang.kernels.ops.attention.utils import concat_mla_absorb_q_general
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=6, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

NOPE, ROPE = 512, 64
FP8 = torch.float8_e4m3fn


def _aten_chain(q_nope, q_rope, k_nope, k_rope):
    q_fp8 = concat_mla_absorb_q_general(q_nope, q_rope).to(FP8)
    return q_fp8, k_nope.to(FP8), k_rope.to(FP8)


FN_MAP = {"fused": fused_mla_prepare_fp8, "aten_chain": _aten_chain}


@marker.parametrize("tokens", [4, 64, 256, 800], [4, 800])
@marker.parametrize("heads", [12], [12])
@marker.benchmark("impl", ["fused", "aten_chain"])
def benchmark(tokens: int, heads: int, impl: str):
    gen = torch.Generator(device="cuda").manual_seed(tokens)

    def rand(*shape):
        return (
            torch.randn(*shape, generator=gen, device="cuda", dtype=torch.float32)
            .mul(0.1)
            .to(torch.bfloat16)
        )

    # q_nope mirrors production: transpose view of the absorb BMM output.
    q_nope_base = rand(heads, tokens, NOPE)
    q_nope = q_nope_base.transpose(0, 1)
    q_rope = rand(tokens, heads, ROPE)
    k_nope = rand(tokens, NOPE)
    k_rope = rand(tokens, ROPE)
    args = (q_nope, q_rope, k_nope, k_rope)
    return marker.do_bench(
        FN_MAP[impl],
        input_args=args,
        graph_clone_args=(0, 1, 2, 3),
        memory_args=(q_nope_base, q_rope, k_nope, k_rope),
    )


if __name__ == "__main__":
    benchmark.run()
