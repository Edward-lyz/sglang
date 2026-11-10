"""K3 MoE front overlap on an unmerged (mixed-dtype) EP front.

`SGLANG_K3_FRONT_OVERLAP` makes `KimiK3MoE._ep_front_overlap` run the router
gate + top-k on the side stream while the latent down-proj runs on the main
stream, for EP a2a layers whose gate and down-proj cannot be merged (FP8
down-proj next to a bf16 gate). The ops are the same ones the serial front
runs, so for identical input the top-k weights/ids and the routed input must
be bit-identical to the serial order -- eagerly and under CUDA-graph replay.
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")

if not torch.cuda.is_available():
    pytest.skip("needs CUDA streams", allow_module_level=True)

from sglang.srt.models.kimi_k3 import KimiK3MoE  # noqa: E402

DEV = "cuda"
HIDDEN, EXPERTS, LATENT, TOPK = 7168, 896, 3584, 16


class _DownProj(torch.nn.Linear):
    """RowParallel-style call contract: returns (output, bias)."""

    def forward(self, x):
        return super().forward(x), None


def _layer(seed):
    torch.manual_seed(seed)
    gate = torch.nn.Linear(HIDDEN, EXPERTS, bias=False).to(DEV, torch.bfloat16)
    down = _DownProj(HIDDEN, LATENT, bias=False).to(DEV, torch.bfloat16)
    bias = torch.randn(EXPERTS, device=DEV, dtype=torch.float32) * 0.01

    def topk(hidden_states, router_logits):
        scores = torch.sigmoid(router_logits.float())
        _, ids = torch.topk(scores + bias, TOPK, dim=-1)
        weights = scores.gather(-1, ids)
        weights = weights / weights.sum(-1, keepdim=True)
        return weights, ids.to(torch.int32), router_logits

    layer = SimpleNamespace(
        alt_stream=torch.cuda.Stream(),
        _ep_front_eligible=False,
        _ep_front_overlap_unmerged=True,
        gate=gate,
        topk=topk,
        routed_expert_down_proj=down,
    )
    return layer


def _serial(layer, x):
    logits = layer.gate(x)
    topk_output = layer.topk(x, logits)
    routed, _ = layer.routed_expert_down_proj(x)
    return topk_output, routed


def _assert_same(got, want):
    (gw, gi, gl), gr = got
    (ww, wi, wl), wr = want
    assert torch.equal(gi, wi), "top-k ids differ"
    assert torch.equal(gw, ww), "top-k weights differ"
    assert torch.equal(gl, wl), "router logits differ"
    assert torch.equal(gr, wr), "routed input differs"


@pytest.mark.parametrize("tokens", [1, 16, 80])
def test_overlap_matches_serial_eager(tokens):
    layer = _layer(tokens)
    x = torch.randn(tokens, HIDDEN, device=DEV, dtype=torch.bfloat16)
    want = _serial(layer, x)
    got = KimiK3MoE._ep_front_overlap(layer, x)
    torch.cuda.synchronize()
    assert got is not None, "overlap path not taken"
    _assert_same(got, want)


def test_overlap_disabled_returns_none():
    layer = _layer(0)
    layer._ep_front_overlap_unmerged = False
    x = torch.randn(8, HIDDEN, device=DEV, dtype=torch.bfloat16)
    assert KimiK3MoE._ep_front_overlap(layer, x) is None


@pytest.mark.parametrize("tokens", [16, 80])
def test_overlap_matches_serial_cuda_graph(tokens):
    layer = _layer(100 + tokens)
    x = torch.randn(tokens, HIDDEN, device=DEV, dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        KimiK3MoE._ep_front_overlap(layer, x)  # warmup
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = KimiK3MoE._ep_front_overlap(layer, x)
    for seed in range(3):
        x.copy_(torch.randn(tokens, HIDDEN, device=DEV, dtype=torch.bfloat16))
        graph.replay()
        torch.cuda.synchronize()
        _assert_same(captured, _serial(layer, x))
