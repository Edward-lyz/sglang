"""XGrammarGrammar.rollback keeps its debug token log in step with the matcher.

Spec verify rolls back every draft node each decode step, so rollback must stay
O(k) in the number of rolled-back tokens rather than O(len(accepted_tokens)).
"""

import unittest
from unittest.mock import MagicMock

from sglang.srt.constrained.xgrammar_backend import XGrammarGrammar
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestXGrammarRollback(unittest.TestCase):
    def _grammar(self):
        matcher = MagicMock()
        matcher.is_terminated.return_value = False
        matcher.accept_token.return_value = True
        return XGrammarGrammar(matcher, 128, ctx=None, override_stop_tokens=None)

    def test_rollback_trims_tail_and_forwards_to_matcher(self):
        g = self._grammar()
        for t in [1, 2, 3, 4, 5]:
            g.accept_token(t)
        log = g.accepted_tokens
        g.rollback(2)
        self.assertEqual(g.accepted_tokens, [1, 2, 3])
        self.assertIs(g.accepted_tokens, log)  # trimmed in place, not copied
        g.matcher.rollback.assert_called_with(2)
        g.rollback(1)
        g.accept_token(9)
        self.assertEqual(g.accepted_tokens, [1, 2, 9])

    def test_rollback_zero_keeps_log(self):
        g = self._grammar()
        for t in [7, 8]:
            g.accept_token(t)
        g.rollback(0)
        self.assertEqual(g.accepted_tokens, [7, 8])

    def test_rollback_past_start_empties_log(self):
        g = self._grammar()
        g.accept_token(3)
        g.rollback(4)
        self.assertEqual(g.accepted_tokens, [])


if __name__ == "__main__":
    unittest.main()
