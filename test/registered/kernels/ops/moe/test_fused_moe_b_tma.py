"""B-only TMA in the triton fused-MoE kernel (port of vllm#63820).

Loading the expert weight through a TensorDescriptor must not change numerics:
the TMA path is compared bitwise against the pointer path.
"""

import sys

import pytest
import torch

import sglang.kernels.ops.moe.fused_moe_triton_kernels as kernels
from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.triton_utils import override_config
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
    fused_experts_impl,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils import is_sm90_supported, is_sm100_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and (is_sm90_supported() or is_sm100_supported())),
    reason="B-only TMA for block FP8 fused MoE is gated to SM90/SM100",
)

BLOCK = [128, 128]


def _make_inputs(M, E, hidden, inter, topk):
    torch.manual_seed(123)
    dev = "cuda"
    x = torch.randn(M, hidden, device=dev, dtype=torch.bfloat16) / 10
    w1 = (torch.randn(E, 2 * inter, hidden, device=dev) / 10).to(torch.float8_e4m3fn)
    w2 = (torch.randn(E, hidden, inter, device=dev) / 10).to(torch.float8_e4m3fn)
    w1_s = torch.rand(E, 2 * inter // 128, hidden // 128, device=dev) * 0.01
    w2_s = torch.rand(E, hidden // 128, inter // 128, device=dev) * 0.01
    score = torch.softmax(torch.randn(M, E, device=dev), dim=-1)
    topk_w, topk_ids = torch.topk(score, topk, dim=-1)
    return x, w1, w2, w1_s, w2_s, topk_w.contiguous(), topk_ids.to(torch.int32)


def _run(config, x, w1, w2, w1_s, w2_s, topk_w, topk_ids):
    # Pinning the config also keeps tuned down configs (A/C TMA) out of the way.
    with override_config(dict(config)):
        return fused_experts_impl(
            x.clone(),
            w1,
            w2,
            topk_w,
            topk_ids,
            inplace=True,
            use_fp8_w8a8=True,
            w1_scale=w1_s,
            w2_scale=w2_s,
            block_shape=BLOCK,
        )


# BLOCK_SIZE_M=16 takes the swap_ab branch on SM90.
@pytest.mark.parametrize("block_m", [16, 64])
@pytest.mark.parametrize("M", [1, 24, 96, 1024])
@pytest.mark.parametrize(
    "E,hidden,inter,topk,block_n,expect_up,expect_down",
    [
        (256, 1024, 256, 8, 128, True, True),
        # down N=hidden=1152 is not a multiple of BLOCK_SIZE_N=256: down falls
        # back to pointers while up still takes TMA.
        (64, 1152, 256, 6, 256, True, False),
    ],
)
def test_b_tma_matches_pointer(
    monkeypatch, block_m, M, E, hidden, inter, topk, block_n, expect_up, expect_down
):
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    inputs = _make_inputs(M, E, hidden, inter, topk)
    config = {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 3,
    }

    decisions = []
    gate = kernels._use_b_only_tma

    def spy(B, *args, **kwargs):
        decisions.append(gate(B, *args, **kwargs))
        return decisions[-1]

    monkeypatch.setattr(kernels, "_use_b_only_tma", spy)

    with envs.SGLANG_OPT_MOE_TRITON_B_TMA.override(False):
        pointer_out = _run(config, *inputs)
    assert decisions == [False, False]

    decisions.clear()
    with envs.SGLANG_OPT_MOE_TRITON_B_TMA.override(True):
        tma_out = _run(config, *inputs)
    assert decisions == [expect_up, expect_down]

    torch.testing.assert_close(tma_out, pointer_out, atol=0, rtol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
