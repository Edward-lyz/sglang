"""Benchmark: fused strided gated RMSNorm (kda_strided_onorm) vs. the
current KDA extend path (ATen layout copy via x.reshape + Triton
rms_norm_gated), at the production shape and adjacent variants.
"""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.attention.fla.fused_norm_gate import rms_norm_gated
from sglang.kernels.ops.kimi_k3.kda_strided_onorm import kda_strided_onorm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=20, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

_DEVICE = "cuda"


def _make_strided_x(T: int, H: int, D: int, dtype: torch.dtype) -> torch.Tensor:
    base = torch.randn(1, H * D, T, dtype=dtype, device=_DEVICE)
    return base.transpose(1, 2).unflatten(-1, (H, D))


def _baseline(x, g, weight, activation, eps):
    # what the KDA extend path does today: rms_norm_gated forces the ATen
    # layout copy internally (x.reshape(-1, D) on non-contiguous x), then the
    # Triton gated-norm kernel runs; finally squeeze+flatten to [T, H*D].
    out = rms_norm_gated(x, g, weight, bias=None, activation=activation, eps=eps)
    return out.squeeze(0).flatten(-2)


def _fused(x, g, weight, activation, eps):
    return kda_strided_onorm(x, g, weight, activation, eps)


FN_MAP = {
    "baseline": _baseline,
    "fused": _fused,
}


@marker.parametrize(
    "T,H,D,activation",
    [
        (4096, 96, 128, "sigmoid"),  # production shape
        (2048, 96, 128, "swish"),
        (16384, 96, 128, "swish"),
    ],
    ci_vals=[(512, 8, 128, "sigmoid")],
)
@marker.benchmark("impl", ["baseline", "fused"])
def benchmark(T: int, H: int, D: int, activation: str, impl: str):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    eps = 1e-5
    x = _make_strided_x(T, H, D, dtype)
    g = torch.randn(1, T, H, D, dtype=dtype, device=_DEVICE)
    weight = torch.randn(D, dtype=dtype, device=_DEVICE)
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(x, g, weight, activation, eps),
        graph_clone_args=(0, 1, 2),
    )


if __name__ == "__main__":
    benchmark.run()
