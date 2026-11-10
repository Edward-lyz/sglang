import unittest
from types import SimpleNamespace

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

from sglang.srt.managers.scheduler_components.prefill_cost_model import (
    CostScheduleConfig,
    PrefillCostModel,
    PrefillCostScheduler,
    StepSample,
    step_features,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _req(rid, *, prompt, cached=0, as_hint=0):
    return SimpleNamespace(
        rid=rid,
        origin_input_ids=[0] * prompt,
        output_ids=[],
        prefix_indices=[0] * cached,
        as_max_hit_token_len=[as_hint] if as_hint else [],
    )


def _scheduler(**overrides):
    config = dict(
        target_step_seconds=0.25,
        aging_seconds=1.0,
        min_long_chunk=8192,
        short_seconds=0.06,
    )
    config.update(overrides)
    return PrefillCostScheduler(
        config=CostScheduleConfig(**config), model=PrefillCostModel()
    )


class TestPrefillCostModel(CustomTestCase):
    def test_fit_recovers_coefficients_and_drops_negative_terms(self):
        """The refit must recover the generating coefficients and clamp a term at 0.

        A plain least-squares fit of noisy data can return a negative
        coefficient, which would make long chunks look cheaper than short ones.
        """
        truth = (0.18, 9.0e-7, 0.0, 1.5e-3)  # no context term in the data
        model = PrefillCostModel()
        rng = np.random.default_rng(0)
        for _ in range(64):
            extend = rng.integers(64, 81920, size=rng.integers(1, 8)).tolist()
            prefix = rng.integers(0, 400_000, size=len(extend)).tolist()
            new_tokens, work, n = step_features(extend_lens=extend, prefix_lens=prefix)
            seconds = truth[0] + truth[1] * new_tokens + truth[3] * n
            seconds += rng.normal(0, 1e-3)
            model.observe(
                StepSample(new_tokens=new_tokens, attn_work=work, num_reqs=n, seconds=seconds)
            )
        self.assertTrue(model.fitted)
        c0, b, d, e = model.coeffs
        self.assertTrue(all(c >= 0 for c in model.coeffs))
        self.assertAlmostEqual(c0, truth[0], delta=0.01)
        self.assertAlmostEqual(b, truth[1], delta=0.1e-7 * 9)
        self.assertLess(d, 1e-13)

    def test_max_chunk_respects_budget_alignment_and_floor(self):
        model = PrefillCostModel()
        for p, budget in ((0, 0.03), (300_000, 0.03), (0, 10.0), (0, 0.0)):
            q = model.max_chunk(p=p, budget_seconds=budget, align=1024, lo=8192, hi=81920)
            self.assertEqual(q % 1024, 0)
            self.assertGreaterEqual(q, 8192)  # the floor wins over the budget
            self.assertLessEqual(q, 81920)
            if q > 8192:
                self.assertLessEqual(model.req_cost(q=q, p=p), budget)
            if q < 81920:
                self.assertGreater(model.req_cost(q=q + 1024, p=p), budget)
        # More context means a smaller chunk for the same budget.
        self.assertLess(
            model.max_chunk(p=600_000, budget_seconds=0.05, align=1024, lo=1024, hi=81920),
            model.max_chunk(p=0, budget_seconds=0.05, align=1024, lo=1024, hi=81920),
        )


class TestPrefillCostScheduler(CustomTestCase):
    def test_orders_by_uncached_work_not_prompt_length(self):
        """A long prompt that is mostly cached is cheap and must go first."""
        sched = _scheduler()
        sched.set_clock(10.0)
        cold_short = _req("cold", prompt=40_000)
        warm_long = _req("warm", prompt=600_000, cached=598_000)
        hinted = _req("hint", prompt=300_000, as_hint=299_000)  # AS hit not loaded yet
        queue = [cold_short, warm_long, hinted]
        sched.note_ready(queue)
        sched.order(queue)
        self.assertEqual({r.rid for r in queue[:2]}, {"hint", "warm"})
        self.assertEqual(queue[-1].rid, "cold")

    def test_aged_requests_are_served_fcfs_first(self):
        sched = _scheduler(aging_seconds=1.0)
        sched.set_clock(0.0)
        old_expensive = _req("old", prompt=500_000)
        sched.note_ready([old_expensive])
        sched.set_clock(5.0)
        new_cheap = _req("new", prompt=100)
        sched.note_ready([new_cheap])
        queue = [new_cheap, old_expensive]
        sched.order(queue)
        self.assertEqual([r.rid for r in queue], ["old", "new"])

    def test_long_chunk_is_throttled_only_while_short_work_exists(self):
        sched = _scheduler()
        sched.set_clock(0.0)
        hi = 81920
        self.assertIsNone(sched.chunk_cap(context_len=0, waiting_queue=[], hi=hi))
        short = _req("short", prompt=2000)
        cap = sched.chunk_cap(context_len=0, waiting_queue=[short], hi=hi)
        self.assertIsNotNone(cap)
        self.assertLess(cap, hi)
        # Still throttled right after the short request was admitted ...
        sched.set_clock(0.1)
        self.assertIsNotNone(sched.chunk_cap(context_len=0, waiting_queue=[], hi=hi))
        # ... but not once no short work was seen for a while.
        sched.set_clock(10.0)
        self.assertIsNone(sched.chunk_cap(context_len=0, waiting_queue=[], hi=hi))

    def test_zero_target_never_throttles_after_small_steps(self):
        """With no explicit target, small observed steps must not shrink chunks.

        A target taken from observed steps locked one fake-dp replica into
        8k chunks after its 1-token warm-up steps: each throttled step kept the
        median low, and that replica ran at 1/4 the throughput of the other.
        """
        sched = _scheduler(target_step_seconds=0.0)
        for _ in range(32):
            sched.model.observe(
                StepSample(new_tokens=8192, attn_work=8192.0**2, num_reqs=1, seconds=0.105)
            )
        sched.set_clock(0.0)
        short = _req("short", prompt=2000)
        self.assertIsNone(sched.chunk_cap(context_len=0, waiting_queue=[short], hi=81920))


if __name__ == "__main__":
    unittest.main()
