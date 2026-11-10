"""Uneven PP layer partition and per-stage correction for the cost model.

K3 PP16 runs SGLANG_PP_LAYER_PARTITION="7,5,7,5,7,5,7,5,7,5,7,5,7,5,6,3", so
stages differ in layer count and in how many full-attention (MLA) layers they
hold. The scheduler relies on:

- the binding stage being found per microbatch, not assumed: 7-layer stages
  bind fresh chunks, the MLA-heaviest stage binds long prefixes;
- every stage staying within the period, cheap stages keeping their slack;
- per-stage scales (for work outside the timed forward, e.g. DSpark on the
  last stage) moving the binding stage and shrinking chunks accordingly;
- the runtime observer turning event timings into the scale to apply.

CPU-only: numpy arithmetic, fake events, no engine or GPU.
"""

import numpy as np
import pytest

from sglang.srt.managers.chunk_cost_model import ChunkCostModel
from sglang.srt.managers.chunk_stage_observer import (
    StageCostObserver,
    parse_stage_scales,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-b-test-cpu")

PARTITION = (7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 6, 3)
BASE_CHUNK = 16384
# Illustrative per-layer ms costs: (g per token, r per token, a per pair).
G_LAYER = 4.0e-4
R_MLA = 5.0e-7
A_MLA = 1.5e-9
E_STAGE = 1.0


def _mla_per_stage(every=4):
    # Hybrid stack: one MLA layer every `every` layers, the rest KDA.
    out, start = [], 0
    for n in PARTITION:
        out.append(sum(1 for i in range(start, start + n) if i % every == every - 1))
        start += n
    return out


def _model(scales=None):
    mla = _mla_per_stage()
    coefs = [
        [E_STAGE, G_LAYER * n, R_MLA * m, A_MLA * m] for n, m in zip(PARTITION, mla)
    ]
    model = ChunkCostModel()
    model.init_measured(coefs)
    if scales is not None:
        model.scale_stages(scales)
    model.set_target_cost(BASE_CHUNK)
    return model, mla


def test_binding_stage_follows_the_partition():
    model, mla = _model()
    fresh = model.binding_stage([(0, BASE_CHUNK)])
    assert PARTITION[fresh] == max(PARTITION)
    # Long enough prefixes are priced by MLA work, so the binding stage moves
    # to one with the most MLA layers -- which is not a 7-layer stage here.
    long_bind = model.binding_stage([(1048576, 4096)])
    assert mla[long_bind] == max(mla)
    assert long_bind != fresh


def test_every_stage_fits_and_cheap_stages_keep_slack():
    model, _ = _model()
    envelope = float((model.target_cost + model.k_overhead).max())
    for L in (0, 8192, 65536, 131072, 262144, 524288):
        x = model.solve_x(L, model.target_cost)
        if x <= 0.0:
            continue
        per_stage = model.batch_period([(L, x)])
        assert np.all(per_stage <= envelope * (1 + 1e-9)), L
        assert per_stage.max() == pytest.approx(envelope, rel=1e-9), L
    # The 3-layer last stage idles most of a fresh-chunk period.
    fresh = model.batch_period([(0, BASE_CHUNK)])
    assert fresh[-1] < 0.6 * fresh.max()


def test_stage_scale_moves_the_binding_stage_and_shrinks_chunks():
    base, _ = _model()
    # Pretend the observer found the last stage 3x slower than its forward
    # alone (DSpark draft-KV injection + sampling live there).
    scales = [1.0] * len(PARTITION)
    scales[-1] = 3.0
    scaled, _ = _model(scales)
    assert scaled.binding_stage([(0, BASE_CHUNK)]) == len(PARTITION) - 1
    for L in (0, 65536, 262144):
        assert scaled.solve_x(L, scaled.target_cost) <= base.solve_x(
            L, base.target_cost
        ) * (1 + 1e-9)
    with pytest.raises(ValueError):
        base.scale_stages([1.0] * 3)
    with pytest.raises(ValueError):
        base.scale_stages([1.0] * (len(PARTITION) - 1) + [0.0])


def test_parse_stage_scales():
    assert parse_stage_scales("", 16) is None
    assert parse_stage_scales("1, 2", 2) == [1.0, 2.0]
    with pytest.raises(ValueError):
        parse_stage_scales("1,2,3", 2)


class _Event:
    def __init__(self, t, done=True):
        self.t, self.done = t, done

    def query(self):
        return self.done

    def elapsed_time(self, end):
        return end.t - self.t


def test_observer_reports_actual_over_predicted_per_stage():
    model, _ = _model()
    stage = len(PARTITION) - 1
    obs = StageCostObserver(model, stage, interval_s=1e9)
    items = [(0, BASE_CHUNK)]
    predicted = float(model.batch_period(items)[stage])
    for i in range(10):
        obs.submit(items, _Event(0.0), _Event(2.0 * predicted))
    pending = _Event(1.0, done=False)
    obs.submit(items, _Event(0.0), pending)
    obs.poll(now=0.0)
    s = obs.summary()
    assert s["n"] == 10  # the unfinished microbatch is not read yet
    assert s["ratio_p50"] == pytest.approx(2.0)
    assert s["binding_frac"] == 0.0  # a 3-layer stage never binds fresh chunks
    pending.done = True
    obs.poll(now=0.0)
    assert obs.summary()["n"] == 11
    # Bounded backlog when events are never polled.
    obs2 = StageCostObserver(model, 0, max_pending=4)
    for _ in range(10):
        obs2.submit(items, _Event(0.0, False), _Event(1.0, False))
    assert len(obs2._pending) == 4


def test_fast_paths_match_the_reference_formulas():
    """req_cost (matvec) and solve_x (scalar loop, stable root) must agree
    with the plain formulas, including an all-KDA stage where the quadratic
    degenerates to linear."""
    rng = np.random.RandomState(7)
    coefs = rng.uniform(0.1, 2.0, size=(16, 4)) * np.array([1.0, 1e-3, 1e-6, 1e-9])
    coefs[3, 2:] = 0.0  # all-KDA stage
    model = ChunkCostModel()
    model.init_measured(coefs)
    model.set_target_cost(BASE_CHUNK)
    e, g, r, a = coefs.T
    for _ in range(200):
        L = float(rng.randint(0, 1 << 20))
        x = float(rng.randint(1, 1 << 15))
        ref = g * x + r * (L + x) + a * x * (L + 0.5 * x)
        np.testing.assert_allclose(model.req_cost(L, x), ref, rtol=1e-12)
        rem = model.target_cost * rng.uniform(0.05, 1.0)
        got = model.solve_x(L, rem)
        rr = rem - r * L
        if np.any(rr <= 0):
            assert got == 0.0
            continue
        b = g + r + a * L
        with np.errstate(divide="ignore", invalid="ignore"):
            quad = (-b + np.sqrt(b * b + 2 * a * rr)) / a
        ref_x = np.where(a > 0, quad, rr / b).min()
        assert got == pytest.approx(ref_x, rel=1e-6)
        assert np.all(model.req_cost(L, got) <= rem * (1 + 1e-9))
