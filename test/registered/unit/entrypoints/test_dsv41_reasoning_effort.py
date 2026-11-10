"""The V4.1 effort budgets must match the published encoder definition.

DeepSeek publishes the V4.1 encoder as ``encoding/encoding.py`` in the
DeepSeek-V4.1-Flash repository. The port in this file used to carry its own
budget table (low=25, high=50, xhigh=75), so a request asking for the
documented "high" rendered 50 where the published encoder renders 75, and the
prompt token ids stopped matching.

These tests pin the published definition:
    REASONING_EFFORT_MAPPINGS = {"low": 50, "high": 75, "max": 100}
    DEFAULT_REASONING_EFFORT = "high"
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

import unittest

from sglang.srt.entrypoints.openai import encoding_dsv41
from sglang.test.test_utils import CustomTestCase

# Copied from the published encoder: encoding/encoding.py in
# deepseek-ai/DeepSeek-V4.1-Flash.
PUBLISHED_MAPPINGS = {"low": 50, "high": 75, "max": 100}
PUBLISHED_DEFAULT = "high"
# "xhigh" is not a published tier. It is the effort level above "high" that the
# Anthropic adapter accepts, so the port renders it at the highest budget.
PORT_ONLY_TIERS = {"xhigh": 100}


class TestDsV41ReasoningEffort(CustomTestCase):
    def test_budgets_match_the_published_encoder(self):
        for tier, budget in PUBLISHED_MAPPINGS.items():
            with self.subTest(tier=tier):
                self.assertEqual(
                    encoding_dsv41.REASONING_EFFORT_MAPPINGS.get(tier),
                    budget,
                    f"tier {tier!r} must use the published budget",
                )

    def test_the_port_only_adds_the_anthropic_tier(self):
        extra = {
            tier: budget
            for tier, budget in encoding_dsv41.REASONING_EFFORT_MAPPINGS.items()
            if tier not in PUBLISHED_MAPPINGS
        }
        self.assertEqual(extra, PORT_ONLY_TIERS)

    def test_default_tier_matches_the_published_encoder(self):
        self.assertEqual(encoding_dsv41.DEFAULT_REASONING_EFFORT, PUBLISHED_DEFAULT)

    def test_rendered_prompt_carries_the_published_budget(self):
        cases = [("low", 50), ("high", 75), ("max", 100), ("xhigh", 100)]
        for tier, budget in cases:
            with self.subTest(tier=tier):
                prompt = encoding_dsv41.render_reasoning_effort(0, "thinking", tier)
                self.assertIn(f"Reasoning Effort: {budget} ", prompt)

    def test_every_budget_stays_inside_the_documented_range(self):
        for tier, budget in encoding_dsv41.REASONING_EFFORT_MAPPINGS.items():
            with self.subTest(tier=tier):
                self.assertTrue(1 <= budget <= 100)

    def test_an_integer_budget_passes_through_unchanged(self):
        prompt = encoding_dsv41.render_reasoning_effort(0, "thinking", 60)
        self.assertIn("Reasoning Effort: 60 ", prompt)

    def test_reasoning_prefix_is_not_rendered_when_thinking_is_off(self):
        self.assertEqual(
            encoding_dsv41.render_reasoning_effort(0, "non_thinking", "high"), ""
        )


if __name__ == "__main__":
    unittest.main()
