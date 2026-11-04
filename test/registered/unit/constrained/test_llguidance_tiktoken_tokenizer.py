"""``GuidanceBackend`` accepts K3's tiktoken-backed tokenizers.

llguidance's ``from_tokenizer`` only accepts a fast HF
``PreTrainedTokenizerFast``. K3 has *two* unrelated tiktoken wrappers, neither of
which is fast:

  * sglang's ``TiktokenTokenizer`` (loaded from an xtok ``.json``), and
  * Kimi's HF ``TikTokenTokenizer`` (``trust_remote_code``, module
    ``tokenization_kimi``, a slow ``PreTrainedTokenizer``) -- the class a real K3
    deployment loads from a model directory.

Feeding the Kimi HF class straight to ``from_tokenizer`` raised
``ValueError: Only fast tokenizers are supported`` and crashed the scheduler
(SIGQUIT). Both wrappers hold a genuine ``tiktoken.Encoding``
(``.tokenizer`` / ``.model`` respectively), so the backend must route both
through ``LLTokenizer.from_tiktoken`` instead.
"""

import unittest

import tiktoken
from llguidance import LLTokenizer

from sglang.srt.constrained.llguidance_backend import (
    GuidanceBackend,
    _tiktoken_encoding_for_llguidance,
)
from sglang.srt.tokenizer.tiktoken_tokenizer import PAT_STR_B
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_SPECIALS = {"<|open|>": 256, "<|sep|>": 257, "<|close|>": 258, "<|eos|>": 259}


def _encoding():
    return tiktoken.Encoding(
        name="k3-syn",
        pat_str=PAT_STR_B,
        mergeable_ranks={bytes([b]): b for b in range(256)},
        special_tokens=_SPECIALS,
    )


class _KimiHFTokenizer:
    """Stand-in for Kimi's HF ``TikTokenTokenizer`` (slow ``PreTrainedTokenizer``).

    Only the attributes the backend reads are modelled: the inner tiktoken
    ``Encoding`` on ``.model`` and ``eos_token_id``. The class *name* and
    ``__module__`` matter -- the backend detects the Kimi tokenizer by them
    (mirroring ``patch_tokenizer._is_kimi_tiktoken_tokenizer``) rather than
    importing the trust_remote_code module.
    """

    def __init__(self, encoding):
        self.model = encoding
        self.eos_token_id = _SPECIALS["<|eos|>"]


# Match the real class name and a ``tokenization_kimi`` module so the detector
# recognizes it exactly as it would the trust_remote_code class.
_KimiHFTokenizer.__name__ = "TikTokenTokenizer"
_KimiHFTokenizer.__qualname__ = "TikTokenTokenizer"
_KimiHFTokenizer.__module__ = "transformers_modules.kimi.tokenization_kimi"


class TestLLGuidanceTiktokenTokenizer(unittest.TestCase):
    def test_kimi_hf_tiktoken_is_recognized(self):
        # The detector must find the inner Encoding on ``.model`` for the Kimi HF
        # class; returning None here is what sent it to ``from_tokenizer``.
        enc = _encoding()
        tok = _KimiHFTokenizer(enc)
        self.assertIs(_tiktoken_encoding_for_llguidance(tok), enc)

    def test_plain_tokenizer_is_not_recognized(self):
        # A tokenizer that is neither wrapper falls back to ``from_tokenizer``.
        class _Other:
            pass

        self.assertIsNone(_tiktoken_encoding_for_llguidance(_Other()))

    def test_backend_builds_from_kimi_hf_tiktoken(self):
        # Regression: constructing the backend with the Kimi HF tokenizer used to
        # raise ``ValueError: Only fast tokenizers are supported`` from
        # ``from_tokenizer``; it must now build an LLTokenizer via the tiktoken
        # bridge instead.
        tok = _KimiHFTokenizer(_encoding())
        backend = GuidanceBackend(
            tokenizer=tok,
            n_vocab=260,
            eos_token_ids=_SPECIALS["<|eos|>"],
        )
        self.assertIsInstance(backend.llguidance_tokenizer, LLTokenizer)


if __name__ == "__main__":
    unittest.main()
