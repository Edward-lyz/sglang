"""Unit tests for request-local scheduler thinking-budget state."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams


class _Tokenizer:
    think_end_id = 99


class _TokenizerWithoutParser:
    def encode(self, text, add_special_tokens=False):
        assert text == "</think>"
        assert add_special_tokens is False
        return [199]


class _TokenizerWithMultiTokenEnd:
    def encode(self, text, add_special_tokens=False):
        assert text == "</think>"
        assert add_special_tokens is False
        return [199, 200]


def _make_req(budget=2, prompt_ids=None):
    req = Req.__new__(Req)
    req.sampling_params = SamplingParams(
        thinking_budget=budget,
        thinking_prompt_token_ids=prompt_ids,
    )
    req.tokenizer = _Tokenizer()
    return req


class TestThinkingBudget(CustomTestCase):
    def test_budget_counts_tokens_and_injects_transition(self):
        req = _make_req(2, [7, 8])
        req.init_thinking_budget([99])

        req.update_thinking_state([1])
        self.assertEqual(req.thinking_budget, 1)
        req.update_thinking_state([2])
        self.assertEqual(req.thinking_budget, 0)
        self.assertEqual(req.maybe_inject_thinking_token(123), 7)
        self.assertEqual(req.maybe_inject_thinking_token(123), 8)
        self.assertEqual(req.maybe_inject_thinking_token(123), 99)
        self.assertEqual(req.thinking_budget, -1)

    def test_speculative_injection_rewrites_window(self):
        req = _make_req(0, [7, 8])
        req.init_thinking_budget([99])
        tokens = torch.tensor([1, 2, 3, 4], dtype=torch.int32)

        count = req.inject_thinking_prompt_tokens(tokens, 0, 3)

        self.assertEqual(count, 3)
        self.assertEqual(tokens.tolist(), [7, 8, 99, 4])
        self.assertEqual(req.thinking_budget, -1)

    def test_think_end_id_falls_back_to_tokenizer_encoding(self):
        req = _make_req(0, [7])
        req.tokenizer = _TokenizerWithoutParser()
        req.init_thinking_budget()

        self.assertEqual(req.thinking_prompt_token_ids, [7, 199])
        self.assertEqual(req.maybe_inject_thinking_token(123), 7)
        self.assertEqual(req.maybe_inject_thinking_token(123), 199)
        self.assertEqual(req.thinking_budget, -1)

    def test_think_end_id_fallback_accepts_multiple_tokens(self):
        req = _make_req(0, [7])
        req.tokenizer = _TokenizerWithMultiTokenEnd()
        req.init_thinking_budget()

        self.assertEqual(req.thinking_prompt_token_ids, [7, 199, 200])
        self.assertEqual(req.maybe_inject_thinking_token(123), 7)
        self.assertEqual(req.maybe_inject_thinking_token(123), 199)
        self.assertEqual(req.maybe_inject_thinking_token(123), 200)
        self.assertEqual(req.thinking_budget, -1)


if __name__ == "__main__":
    unittest.main()
