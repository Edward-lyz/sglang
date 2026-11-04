"""llguidance runtime matcher errors must stay per-request.

Incident: with llguidance as the grammar backend, the scheduler died on
``ValueError: lexer error: too many states: 250000 >= 250000`` raised from a
mask fill inside the decode forward, taking every TP rank (and the pod) down.

Two independent defects, one test class each:

* ``GuidanceGrammar.copy()`` used ``LLMatcher.deep_copy()``, which shares a
  single lexer DFA between the cached template and every request copied from
  it. DFA states only grow, so a hot grammar accumulated all its requests'
  states until it hit ``max_lexer_states``; the error is sticky, so from then on
  every request on that grammar failed on its first fill.
* A runtime matcher error raised out of ``fill_vocab_mask`` / ``rollback`` and
  out of the speculative tree traversal, none of which have a per-request
  handler, so one request's grammar error crashed the scheduler process.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from llguidance import LLParserLimits, LLTokenizer, grammar_from

from sglang.srt.constrained.llguidance_backend import GuidanceGrammar
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.speculative.spec_utils import generate_token_bitmask
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_EOS = 256
_LETTERS = "abcdefghijklmnopqrstuvwxyz"
# 26 counted lexemes picked by the first byte: every position of each one is a
# distinct lexer DFA state, so each request builds a few hundred fresh states
# and requests starting with different letters do not reuse each other's.
_GRAMMAR = grammar_from(
    "lark", "start: v\nv: " + " | ".join(f"/{c}[a-z]{{0,300}}/" for c in _LETTERS)
)


def _tokenizer() -> LLTokenizer:
    return LLTokenizer.from_tiktoken(
        encoder={bytes([b]): b for b in range(256)},
        special_tokens={"<eos>": _EOS},
        pattern=r".",
        eos_token=_EOS,
        n_vocab=_EOS + 1,
    )


def _grammar(max_lexer_states: int) -> GuidanceGrammar:
    return GuidanceGrammar(
        llguidance_tokenizer=_tokenizer(),
        serialized_grammar=_GRAMMAR,
        limits=LLParserLimits(max_lexer_states=max_lexer_states),
    )


def _drive(grammar: GuidanceGrammar, first_letter: str, num_tokens: int) -> None:
    """Decode loop: fill the mask, then commit one token, ``num_tokens`` times."""
    mask = grammar.allocate_vocab_mask(
        vocab_size=_EOS + 1, batch_size=1, device="cpu"
    )
    tokens = [ord(first_letter)] + [ord("q")] * (num_tokens - 1)
    for token in tokens:
        grammar.fill_vocab_mask(mask, 0)
        grammar.accept_token(token)


def _allowed_tokens(mask_row: torch.Tensor) -> list:
    return [t for t in range(_EOS + 1) if (int(mask_row[t // 32]) >> (t % 32)) & 1]


class TestCopiesDoNotShareLexerBudget(unittest.TestCase):
    def test_hot_cached_template_survives_many_requests(self):
        # ~300 states per request against a 3000 budget: a shared lexer runs out
        # around the 11th request and every later copy fails on its first fill;
        # independent per-request matchers never come close.
        template = _grammar(max_lexer_states=3000)
        for letter in _LETTERS:
            _drive(template.copy(), letter, num_tokens=290)
        _drive(template.copy(), "a", num_tokens=5)


class TestRuntimeErrorStaysWithTheRequest(unittest.TestCase):
    def _errored_grammar(self):
        # A budget of 100 states is exhausted by a single request ~34 tokens in.
        grammar = _grammar(max_lexer_states=100)
        mask = grammar.allocate_vocab_mask(
            vocab_size=_EOS + 1, batch_size=1, device="cpu"
        )
        grammar.fill_vocab_mask(mask, 0)
        grammar.accept_token(ord("a"))
        for _ in range(100):
            grammar.fill_vocab_mask(mask, 0)
            if grammar.ll_matcher.is_error():
                return grammar, mask
            grammar.accept_token(ord("q"))
        self.fail("lexer state budget was never exhausted")

    def test_fill_and_rollback_do_not_raise_accept_does(self):
        grammar, mask = self._errored_grammar()
        # Fail closed: the errored row only admits EOS ...
        self.assertEqual(_allowed_tokens(mask[0]), [_EOS])
        grammar.fill_vocab_mask(mask, 0)
        grammar.rollback(1)
        # ... and committing anything, EOS included, raises so the result
        # processor aborts the request instead of reporting a clean stop.
        with self.assertRaisesRegex(ValueError, "too many states"):
            grammar.accept_token(_EOS)

    def test_spec_tree_traversal_aborts_only_the_failing_request(self):
        # Two requests in one verify batch; the first one's grammar raises while
        # the draft tree is walked. Tree per request: 0 -- 1 (one draft token).
        failing = MagicMock()
        failing.is_terminated.return_value = False
        failing.accept_token.side_effect = ValueError("lexer error: too many states")
        # 2 reqs x 2 nodes rows; one int32 word covers the 32-token vocab.
        failing.allocate_vocab_mask.return_value = torch.full(
            (4, 1), -1, dtype=torch.int32
        )
        healthy = MagicMock()
        healthy.is_terminated.return_value = False
        reqs = [
            SimpleNamespace(rid="bad", grammar=failing, to_finish=None),
            SimpleNamespace(rid="good", grammar=healthy, to_finish=None),
        ]

        generate_token_bitmask(
            reqs,
            SimpleNamespace(grammar=None),
            torch.tensor([[1, -1], [1, -1]]),
            torch.tensor([[-1, -1], [-1, -1]]),
            torch.tensor([[0, 7], [0, 7]]),
            32,
        )

        self.assertIsInstance(reqs[0].to_finish, FINISH_ABORT)
        self.assertIsNone(reqs[1].to_finish)
        self.assertEqual(healthy.accept_token.call_count, 1)


if __name__ == "__main__":
    unittest.main()
