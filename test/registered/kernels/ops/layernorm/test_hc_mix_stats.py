import sys

import pytest
import torch

from sglang.kernels.ops.layernorm import mhc as _mhc
from sglang.kernels.ops.layernorm.mhc import hc_mix_stats
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.cuda is None,
    reason="hc_mix_stats requires CUDA",
)

EPS = 1e-6


def _reference(x_flat: torch.Tensor, hc_fn: torch.Tensor) -> torch.Tensor:
    x = x_flat.double()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)
    return (x @ hc_fn.double().T) * rsqrt


@pytest.mark.parametrize("m", [1, 7, 64, 300, 1024, 4096])
@pytest.mark.parametrize("k", [4096, 20480])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_hc_mix_stats_matches_reference(m: int, k: int, dtype: torch.dtype):
    torch.manual_seed(0)
    x_flat = torch.randn(m, k, device="cuda", dtype=dtype)
    hc_fn = torch.randn(24, k, device="cuda", dtype=torch.float32) * 0.02

    got = hc_mix_stats(x_flat, hc_fn, EPS)
    ref = _reference(x_flat, hc_fn)

    assert got.shape == (m, 24) and got.dtype == torch.float32
    scale = ref.abs().max().clamp(min=1e-6)
    err_kernel = (got.double() - ref).abs().max() / scale
    # The Triton split-K producer accumulates with tf32x3 (~fp32); the optional
    # deep_gemm.tf32_hc_prenorm_gemm producer accumulates with single-pass tf32,
    # matching the accuracy class of DeepSeek's own mega_mhc kernel. Bound the
    # error against the independent FP64 result per active producer.
    deepgemm_active = _mhc._mhc_deepgemm_eligible(x_flat, hc_fn)
    tol = 3e-4 if deepgemm_active else 1e-4
    assert err_kernel < tol



if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
