"""Correctness tests for the fused MLA FP8 decode prepare kernel.

``fused_mla_prepare_fp8`` replaces ``mla_quantize_without_rope_for_fp8``'s
``concat_mla_absorb_q -> .to(float8_e4m3fn) x3`` chain, so it must be BIT-equal to
that chain, not merely close: the conversion is a round-to-nearest bf16 -> e4m3
cast with torch's overflow-to-NaN behaviour (``|x| > 464``, inf and nan all become
0x7f with the sign kept), NOT the hardware clamp to 448.

Covers the production layout (``q_nope`` as the transpose view of the absorb BMM
output, so head stride is ``tokens * 512``), extreme values, the env-gated
dispatch through ``mla_quantize_without_rope_for_fp8``, the unsupported-input
fallback and CUDA graph capture/replay.
"""

import pytest
import torch

from sglang.kernels.ops.attention.fused_mla_prepare_fp8 import (
    can_use_fused_mla_prepare_fp8,
    fused_mla_prepare_fp8,
)
from sglang.kernels.ops.attention.utils import (
    concat_mla_absorb_q_general,
    mla_quantize_without_rope_for_fp8,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

NOPE, ROPE = 512, 64
FP8 = torch.float8_e4m3fn


def _reference(q_nope, q_rope, k_nope, k_rope):
    q = concat_mla_absorb_q_general(q_nope, q_rope).to(FP8)
    return q, k_nope.to(FP8), k_rope.to(FP8)


def _assert_bit_equal(got, want):
    for name, g, w in zip(("q", "k_nope", "k_rope"), got, want):
        assert g.dtype == w.dtype and g.shape == w.shape, name
        gb = g.contiguous().view(torch.uint8)
        wb = w.contiguous().view(torch.uint8)
        mismatched = int((gb != wb).sum())
        assert mismatched == 0, f"{name}: {mismatched} mismatched bytes"


def _make_inputs(tokens, heads, seed, extreme=False):
    """``q_nope`` mirrors production: a transpose view of ``[heads, tokens, 512]``."""
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def rand(*shape, scale=1.0):
        return (
            torch.randn(*shape, generator=gen, device="cuda", dtype=torch.float32)
            .mul(scale)
            .to(torch.bfloat16)
        )

    q_nope = rand(heads, tokens, NOPE, scale=100.0).transpose(0, 1)
    q_rope = rand(tokens, heads, ROPE, scale=100.0)
    k_nope = rand(tokens, NOPE, scale=100.0)
    k_rope = rand(tokens, ROPE, scale=100.0)
    if extreme:
        # Exercise every encoding torch and the hardware disagree about.
        specials = torch.tensor(
            [465.0, -465.0, 1e4, -1e4, float("inf"), -float("inf"), float("nan")],
            device="cuda",
            dtype=torch.bfloat16,
        )
        specials = torch.cat([specials, -specials[-1:]])  # -nan
        n = specials.numel()
        q_nope[0, 0, :n] = specials
        q_rope[0, 0, :n] = specials
        k_nope[0, :n] = specials
        k_rope[0, :n] = specials
    return q_nope, q_rope, k_nope, k_rope


@pytest.mark.parametrize("tokens,heads", [(4, 12), (256, 12), (800, 12), (1, 1)])
@pytest.mark.parametrize("extreme", [False, True])
def test_bit_exact_vs_reference(tokens, heads, extreme):
    inputs = _make_inputs(tokens, heads, seed=1234 + tokens, extreme=extreme)
    assert can_use_fused_mla_prepare_fp8(*inputs)
    _assert_bit_equal(fused_mla_prepare_fp8(*inputs), _reference(*inputs))


def test_env_gate_dispatches_and_matches():
    inputs = _make_inputs(256, 12, seed=7, extreme=True)
    want = _reference(*inputs)
    with envs.SGLANG_OPT_USE_FUSED_MLA_PREPARE_FP8.override(False):
        _assert_bit_equal(mla_quantize_without_rope_for_fp8(*inputs), want)
    with envs.SGLANG_OPT_USE_FUSED_MLA_PREPARE_FP8.override(True):
        _assert_bit_equal(mla_quantize_without_rope_for_fp8(*inputs), want)


def test_unsupported_inputs_fall_back():
    q_nope, q_rope, k_nope, k_rope = _make_inputs(8, 12, seed=11)

    # fp16 instead of bf16.
    assert not can_use_fused_mla_prepare_fp8(
        q_nope.to(torch.float16),
        q_rope.to(torch.float16),
        k_nope.to(torch.float16),
        k_rope.to(torch.float16),
    )
    # Non-unit inner stride.
    wide = torch.empty(8, 12, NOPE * 2, device="cuda", dtype=torch.bfloat16)
    assert not can_use_fused_mla_prepare_fp8(wide[..., ::2], q_rope, k_nope, k_rope)
    # Unexpected head dim.
    assert not can_use_fused_mla_prepare_fp8(
        q_nope[..., : NOPE // 2], q_rope, k_nope[..., : NOPE // 2], k_rope
    )
    # The gated entry point must still produce the reference result for these.
    with envs.SGLANG_OPT_USE_FUSED_MLA_PREPARE_FP8.override(True):
        half = (q_nope[..., : NOPE // 2], q_rope, k_nope[..., : NOPE // 2], k_rope)
        _assert_bit_equal(mla_quantize_without_rope_for_fp8(*half), _reference(*half))


def test_cuda_graph_capture_replay():
    inputs = _make_inputs(800, 12, seed=99, extreme=True)
    fused_mla_prepare_fp8(*inputs)  # warm up JIT before capture
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = fused_mla_prepare_fp8(*inputs)
    graph.replay()
    torch.cuda.synchronize()
    _assert_bit_equal(outputs, _reference(*inputs))

    # Refill the inputs in place and replay: the captured kernel must recompute.
    fresh = _make_inputs(800, 12, seed=100, extreme=True)
    for dst, src in zip(inputs, fresh):
        dst.copy_(src)
    graph.replay()
    torch.cuda.synchronize()
    _assert_bit_equal(outputs, _reference(*inputs))
