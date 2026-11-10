"""Unit tests for the per-PP-stage chunked-prefill cost model.

ChunkCostModel decides how many new tokens a prefill chunk may contain so that
no pipeline stage does more work than it would on one fresh chunk of the
configured base size. The properties asserted here are what the scheduler
actually relies on:

- fixed point: with no prefix hit the model reproduces the static token budget
  exactly, so enabling it cannot change behaviour on fresh traffic;
- feasibility: the chosen chunk never exceeds any stage's remaining budget --
  a stage that overshoots is a straggler, i.e. the pipeline bubble this model
  exists to remove;
- tightness: some stage is saturated, so the chunk is the largest feasible one
  (checked as a property, not by re-deriving it with a second search);
- monotonicity and forced progress, so chunked requests cannot stall.

Coefficients are asserted through the public API only. Numbers are
DeepSeek-V3-class MLA shapes on a K3-style uneven PP16 partition; they are
illustrative, the assertions are scale-free.

CPU-only: numpy arithmetic on ChunkCostModel, no engine, model or GPU.
"""

import numpy as np
import pytest
import torch

from sglang.srt.managers.chunk_cost_model import ChunkCostModel
from sglang.srt.managers.chunk_flops_estimator import (
    aggregate_stage_flops,
    estimate_stage_flops_per_token,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-b-test-cpu")

# Per-layer constants for a DeepSeek-V3-class MLA layer:
#   attention core  2 * n_heads * ((qk_nope + qk_rope) + v_head)  per (q, kv) pair
#   re-expansion    2 * n_heads * kv_lora * (qk_nope + v_head)    per token
ATTN_PER_PAIR_PER_LAYER = 2.0 * 128 * ((128 + 64) + 128)
REEXPAND_PER_LAYER = 2.0 * 128 * 512 * (128 + 128)
# Per-token weight-GEMM FLOPs of one layer (MoE with topk/num_experts applied).
GEMM_PER_LAYER = 1.08e9

# K3 PP16: SGLANG_PP_LAYER_PARTITION="7,6,6,6,6,6,6,5,6,6,6,6,5,6,5,5" and a
# hybrid MLA/KDA stack, so total layers and MLA layers vary independently.
K3_LAYERS = (7, 6, 6, 6, 6, 6, 6, 5, 6, 6, 6, 6, 5, 6, 5, 5)
K3_MLA_LAYERS = (2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 1)

BASE_CHUNK = 24 * 1024
PAGE_SIZE = 64


def _model(layers=K3_LAYERS, mla_layers=K3_MLA_LAYERS, base_chunk=BASE_CHUNK):
    model = ChunkCostModel()
    model.init_analytic(
        flops=[n * GEMM_PER_LAYER for n in layers],
        mla_layers=list(mla_layers),
        attn_flops_per_pair_per_layer=ATTN_PER_PAIR_PER_LAYER,
        reexpand_flops_per_layer=REEXPAND_PER_LAYER,
    )
    model.set_target_cost(base_chunk)
    return model


def _tol(rem):
    return 1e-9 * np.maximum(np.abs(rem), 1.0)


def _chunk_limit(model, history_len, rem, page_size=PAGE_SIZE, cap=BASE_CHUNK):
    """Mirror of PrefillAdder._cost_chunk_limit (float-error nudge, page
    alignment, forced progress, per-request cap) so the sequential-loop
    invariants are tested against what the scheduler really applies."""
    x = int(model.solve_x(history_len, rem) * (1.0 + 1e-9) + 1e-6)
    x = min(x, cap)
    return max(x // page_size * page_size, page_size)


def _configs(count=64, stages=16, seed=20260824):
    """Deterministic pseudo-random per-stage configurations (LCG, so failures
    reproduce without a hypothesis dependency)."""
    state = seed
    for _ in range(count):
        layers, mla = [], []
        for _ in range(stages):
            state = (1103515245 * state + 12345) % (1 << 31)
            layers.append(5 + state % 3)
            state = (1103515245 * state + 12345) % (1 << 31)
            mla.append(state % 4)
        if not any(mla):
            mla[0] = 1
        yield tuple(layers), tuple(mla)


HISTORY_GRID = (0, 1024, 8192, 24576, 65536, 131072, 200000)


def test_zero_history_reproduces_the_static_token_budget():
    """The whole point of the fixed point: turning the cost model on must not
    change how a fresh request is chunked."""
    model = _model()
    for base_chunk in (2048, 8192, BASE_CHUNK, 64 * 1024):
        model.set_target_cost(base_chunk)
        x = model.solve_x(0, model.target_cost)
        assert x == pytest.approx(base_chunk, rel=1e-9)
        # Also survives the scheduler's int()/page-align/cap pipeline.
        assert _chunk_limit(model, 0, model.target_cost, cap=base_chunk) == base_chunk


def test_budget_is_the_pipeline_period_not_each_stage_s_own_fresh_cost():
    """A stage that is cheap on a fresh chunk has real slack, because the
    microbatch takes as long as the *slowest* stage. Budgeting it at its own
    cost_s(0, base) is a tighter constraint with nothing behind it, and it shrank
    long-prefix chunks by binding on a stage that was never the bottleneck."""
    for layers, mla in _configs(count=16):
        model = _model(layers, mla)
        per_stage = model.req_cost(0, BASE_CHUNK)
        period = per_stage.max()
        assert np.all(model.target_cost == period)
        for history_len in HISTORY_GRID:
            x = model.solve_x(history_len, model.target_cost)
            if x <= 0.0:
                continue
            cost = model.req_cost(history_len, x)
            # no stage exceeds the period ...
            assert np.all(cost <= period + _tol(model.target_cost))
            # ... and the chunk is not capped by any cheaper stage's own figure
            assert x >= model.solve_x(history_len, per_stage) - 1e-9


def test_chunk_is_feasible_and_tight_on_every_stage():
    for layers, mla in _configs():
        model = _model(layers, mla)
        for history_len in HISTORY_GRID:
            # Fresh batch, and a batch that already spent part of its budget.
            for spent in (0.0, 0.3, 0.7):
                rem = model.target_cost * (1.0 - spent)
                x = model.solve_x(history_len, rem)
                if x <= 0.0:
                    # Only legitimate when some stage cannot even afford the
                    # prefix re-expansion alone.
                    assert np.any(model.k_e * history_len >= rem)
                    continue
                cost = model.req_cost(history_len, x)
                assert np.all(cost <= rem + _tol(rem)), (layers, mla, history_len)
                assert np.any(cost >= rem - _tol(rem)), (layers, mla, history_len)


def test_chunk_shrinks_as_the_prefix_grows():
    for layers, mla in _configs(count=16):
        model = _model(layers, mla)
        previous = None
        for history_len in HISTORY_GRID:
            x = model.solve_x(history_len, model.target_cost)
            if previous is not None:
                assert x <= previous + 1e-9, (layers, mla, history_len)
            previous = x


def test_stage_without_full_attention_layers_uses_the_linear_branch():
    """An all-KDA stage has k_e = k_a = 0, so its quadratic degenerates. It must
    still be priced (its GEMMs are real) and must not divide by zero."""
    model = _model(mla_layers=(0,) * 8 + (2,) * 8)
    assert model.k_a[0] == 0.0 and model.k_e[0] == 0.0
    assert model.k_g[0] > 0.0
    x = model.solve_x(0, model.target_cost)
    assert x == pytest.approx(BASE_CHUNK, rel=1e-9)
    # The linear stage alone would allow a much larger chunk at long prefixes;
    # the MLA-heavy stages are what bind there.
    long_x = model.solve_x(131072, model.target_cost)
    assert 0.0 < long_x < BASE_CHUNK


def test_unaffordable_prefix_yields_zero_but_callers_still_progress():
    model = _model()
    # Budget below what one stage pays just to re-expand the prefix.
    history_len = 200000
    rem = model.k_e * history_len * 0.5
    assert model.solve_x(history_len, rem) == 0.0
    # The scheduler's floor keeps a chunked request moving regardless.
    assert _chunk_limit(model, history_len, rem) == PAGE_SIZE


def test_sequential_chunking_covers_a_long_sequence():
    """End-to-end invariant the single-point assertions cannot see: page
    alignment combined with min-over-stages must still terminate."""
    model = _model()
    total = 400 * 1024
    history_len = 0
    sizes = []
    while history_len < total:
        rem = model.target_cost  # one request per batch
        x = min(_chunk_limit(model, history_len, rem), total - history_len)
        assert x >= PAGE_SIZE or x == total - history_len
        cost = model.req_cost(history_len, x)
        assert np.all(cost <= rem * 1.05 + _tol(rem)), (history_len, x)
        sizes.append(x)
        history_len += x
    assert history_len == total
    assert len(sizes) < 200, "chunking should not degenerate into slivers"
    # Ignoring the final remainder, chunks never grow.
    body = sizes[:-1]
    assert all(b >= a for a, b in zip(body[1:], body[:-1]))


def test_splitting_a_long_prefix_request_repays_the_re_expansion():
    """Why the admission rule prefers deferring over carving a sliver: each
    extra chunk re-expands the whole latent prefix again."""
    model = _model()
    history_len = 124 * 1024
    extend = 4096
    whole = model.req_cost(history_len, extend).max()
    split = (
        model.req_cost(history_len, extend // 2).max()
        + model.req_cost(history_len + extend // 2, extend // 2).max()
    )
    assert split > whole
    assert model.reexpand_cost(history_len) > 0.0


def test_aggregate_stage_flops_keeps_every_stage():
    buckets = [
        {"flops": 7.0, "mla_layers": 2.0},
        {"flops": 5.0, "mla_layers": 4.0},
    ]
    stages = aggregate_stage_flops(buckets)
    # No collapsing: a per-term max would invent a (7 flops, 4 MLA) stage that
    # does not exist, and picking one stage would mispredict the other regime.
    assert list(stages["flops"]) == [7.0, 5.0]
    assert list(stages["mla_layers"]) == [2.0, 4.0]
    with pytest.raises(ValueError):
        aggregate_stage_flops([])


def test_init_analytic_rejects_unusable_constants():
    model = ChunkCostModel()
    with pytest.raises(ValueError):  # a stage with no GEMM work
        model.init_analytic([1.0, 0.0], [1.0, 1.0], 1.0, 1.0)
    with pytest.raises(ValueError):  # no full-attention layer anywhere
        model.init_analytic([1.0, 1.0], [0.0, 0.0], 1.0, 1.0)
    with pytest.raises(ValueError):  # mismatched per-stage arrays
        model.init_analytic([1.0, 1.0], [1.0], 1.0, 1.0)


class _FakeParam:
    """Minimal stand-in for a parameter: the walk only reads dim/numel/dtype."""

    def __init__(self, shape, dtype=None):
        self._shape = shape
        self.dtype = dtype if dtype is not None else torch.bfloat16

    def dim(self):
        return len(self._shape)

    def numel(self):
        n = 1
        for d in self._shape:
            n *= d
        return n


class _FakeModel:
    def __init__(self, params):
        self._params = params

    def named_parameters(self):
        return iter(self._params)


class _FakeConfig:
    num_experts_per_token = 8
    n_routed_experts = 256


def test_mla_layer_count_is_naming_agnostic():
    """MLA layers are counted by kv_b_proj-owning prefix, so a quantized naming
    neither misses a layer nor counts one twice. Miscounting silently rescales
    k_a/k_e, which is the coefficient the whole model turns on."""
    for suffixes in (
        (".weight",),
        (".qweight", ".weight_scale"),
        (".weight_packed", ".weight_scale", ".weight_shape"),
    ):
        params = []
        for layer in range(3):
            for suffix in suffixes:
                params.append(
                    (f"model.layers.{layer}.self_attn.kv_b_proj{suffix}",
                     _FakeParam((512, 32768)))
                )
        buckets = estimate_stage_flops_per_token(_FakeModel(params), _FakeConfig())
        assert buckets["mla_layers"] == 3.0, suffixes
        # kv_b_proj is priced by k_e, so it must not enter the GEMM bucket.
        assert buckets["flops"] == 0.0, suffixes


def test_weight_walk_flops_accounting():
    params = [
        ("model.layers.0.self_attn.o_proj.weight", _FakeParam((7168, 16384))),
        # Packed 4-bit: two elements per stored byte.
        ("model.layers.0.mlp.gate_proj.weight", _FakeParam((1024, 1024), torch.uint8)),
        # Stacked routed experts: only topk / num_experts are activated.
        ("model.layers.0.mlp.experts.w13_weight", _FakeParam((32, 2048, 7168))),
        # Not per-token GEMM work.
        ("model.embed_tokens.weight", _FakeParam((129280, 7168))),
        ("lm_head.weight", _FakeParam((129280, 7168))),
        ("model.layers.0.input_layernorm.weight", _FakeParam((7168,))),
    ]
    buckets = estimate_stage_flops_per_token(_FakeModel(params), _FakeConfig())
    expected = 2.0 * (
        7168 * 16384
        + 1024 * 1024 * 2
        + 32 * 2048 * 7168 * 8 / 256
    )
    assert buckets["flops"] == pytest.approx(expected)
    assert buckets["mla_layers"] == 0.0
