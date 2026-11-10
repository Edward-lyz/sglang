"""Unit tests for the measured (ms) mode of the chunked-prefill cost model.

Measured mode keeps ChunkCostModel's function and solver but takes per-stage
coefficients fitted from timed forwards, plus a per-microbatch constant that is
charged once per batch. What the scheduler relies on:

- fitting recovers the coefficients that generated the timings, including the
  constant, from the same grid shape the startup calibration uses;
- the fixed point survives the constant: with no prefix hit and the derived
  period, the binding stage still gets exactly --chunked-prefill-size;
- an explicit period moves every chunk the expected way and never lets a
  stage exceed it once its constant is paid;
- coefficients that cannot price a chunk are rejected, so startup falls back.

CPU-only: numpy arithmetic, no engine, model or GPU.
"""

import numpy as np
import pytest

from sglang.srt.managers.chunk_cost_model import (
    ChunkCostModel,
    fit_stage_coefficients,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-b-test-cpu")

BASE_CHUNK = 16384
PAGE_SIZE = 64

# Illustrative K3-PP16-like ms coefficients: (e, g, r, a) per stage. Stages
# with more MLA layers carry larger r/a; one all-KDA stage has r = a = 0.
_E = 1.5
_G = 2.4e-3
_R = 1.0e-6
_A = 3.0e-9
MLA_PER_STAGE = (2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 0)
LAYERS_PER_STAGE = (7, 6, 6, 6, 6, 6, 6, 5, 6, 6, 6, 6, 5, 6, 5, 5)


def _coefs():
    return [
        [_E, _G * n / 6.0, _R * m, _A * m]
        for n, m in zip(LAYERS_PER_STAGE, MLA_PER_STAGE)
    ]


def _model(period=None):
    model = ChunkCostModel()
    model.init_measured(_coefs())
    model.set_target_cost(BASE_CHUNK, period=period)
    return model


def _grid():
    xs = (1024, 2048, 4096, 8192, 16384)
    grid = [[(L, x)] for L in (0, 8192, 32768, 65536, 131072, 262144) for x in xs]
    grid.append([(0, 4096)] * 4)
    grid.append([(0, 2048)] * 8)
    grid.append([(32768, 4096), (0, 4096)])
    return grid


def _time(coef, items):
    e, g, r, a = coef
    t = e
    for L, x in items:
        t += g * x + r * (L + x) + a * x * (L + 0.5 * x)
    return t


def test_fit_recovers_generating_coefficients():
    rng = np.random.RandomState(0)
    for coef in _coefs():
        samples = [
            (items, _time(coef, items) * (1.0 + 0.01 * rng.randn()))
            for items in _grid()
        ]
        fitted, err = fit_stage_coefficients(samples)
        assert err < 0.05
        e, g, r, a = fitted
        assert e == pytest.approx(coef[0], rel=0.2)
        assert g == pytest.approx(coef[1], rel=0.05)
        if coef[3] > 0:
            assert a == pytest.approx(coef[3], rel=0.1)
        assert np.all(fitted >= 0.0)


def test_fit_reports_large_error_for_a_model_that_does_not_fit():
    # Times that grow with L^2 cannot be expressed by the model; the LOO error
    # must say so instead of silently returning a curve.
    samples = [
        (items, 1.0 + sum(1e-9 * (L + x) ** 2 for L, x in items)) for items in _grid()
    ]
    _, err = fit_stage_coefficients(samples)
    assert err > 0.10


def test_zero_history_reproduces_the_static_token_budget_with_overhead():
    model = _model()
    x = model.solve_x(0, model.target_cost)
    assert x == pytest.approx(BASE_CHUNK, rel=1e-9)
    # The constant is out of the per-request budget, not out of the period.
    period = model.batch_period([(0, BASE_CHUNK)]).max()
    assert np.all(model.target_cost + model.k_overhead == pytest.approx(period))


def test_microbatch_never_exceeds_the_period():
    for period in (None, 30.0, 80.0):
        model = _model(period)
        envelope = float((model.target_cost + model.k_overhead).max())
        for L in (0, 8192, 65536, 131072, 262144):
            x = model.solve_x(L, model.target_cost)
            if x <= 0.0:
                continue
            per_stage = model.batch_period([(L, x)])
            assert np.all(per_stage <= envelope * (1 + 1e-9)), (period, L)
            assert np.any(per_stage >= envelope * (1 - 1e-9)), (period, L)


def test_explicit_period_scales_chunks_and_prefix_shrinks_them():
    derived = _model()
    small = _model(period=0.5 * float((derived.target_cost + derived.k_overhead).max()))
    for L in (0, 32768, 131072):
        assert small.solve_x(L, small.target_cost) < derived.solve_x(L, derived.target_cost)
    previous = None
    for L in (0, 8192, 32768, 131072, 262144):
        x = derived.solve_x(L, derived.target_cost)
        if previous is not None:
            assert x <= previous + 1e-9
        previous = x


def test_period_below_the_constant_is_rejected():
    model = ChunkCostModel()
    model.init_measured(_coefs())
    with pytest.raises(ValueError):
        model.set_target_cost(BASE_CHUNK, period=_E * 0.5)


def test_init_measured_rejects_unusable_coefficients():
    model = ChunkCostModel()
    with pytest.raises(ValueError):  # wrong shape
        model.init_measured([[1.0, 1.0, 0.0]])
    with pytest.raises(ValueError):  # a stage with no per-token price
        model.init_measured([[1.0, 0.0, 0.0, 0.0]])
    with pytest.raises(ValueError):  # negative
        model.init_measured([[1.0, 1.0, -1.0, 0.0]])
    with pytest.raises(ValueError):  # not finite
        model.init_measured([[1.0, float("nan"), 0.0, 0.0]])


def test_flops_mode_is_unchanged_by_the_constant():
    """init_analytic has no constant, so its target stays the plain envelope."""
    model = ChunkCostModel()
    model.init_analytic([2.0, 1.0], [1.0, 2.0], 1e-3, 1e-2)
    model.set_target_cost(BASE_CHUNK)
    assert np.all(model.k_overhead == 0.0)
    assert np.all(model.target_cost == model.req_cost(0, BASE_CHUNK).max())
