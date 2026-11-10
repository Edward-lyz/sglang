"""Unit tests for routing Kimi K3 rendering through the official encoding_k3."""

import sys
import types
import unittest

from sglang.srt.entrypoints.openai import encoding_k3, encoding_kimi_k3
from sglang.srt.entrypoints.openai.encoding_kimi_k3 import (
    use_official_kimi_k3_encoding,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _install_fake_tokenizer(suffix):
    """Mimic tokenization_kimi: apply_chat_template calls a module-level
    build_chat_segments imported from the model directory's encoding_k3."""
    module = types.ModuleType(f"fake_tokenization_kimi_{suffix}")
    module.EncodeSegment = encoding_k3.EncodeSegment
    exec(
        "def build_chat_segments(messages, tools=None, **kwargs):\n"
        "    return [EncodeSegment('shipped')]\n"
        "class TikTokenTokenizer:\n"
        "    def apply_chat_template(self, messages, tools=None, **kwargs):\n"
        "        return ''.join(s.text for s in build_chat_segments(\n"
        "            messages, tools, **kwargs))\n",
        module.__dict__,
    )
    module.TikTokenTokenizer.__module__ = module.__name__
    sys.modules[module.__name__] = module
    return module, module.TikTokenTokenizer()


def _assistant_turn(arguments, reasoning=None):
    message = {
        "role": "assistant",
        "content": "ok",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "f", "arguments": arguments},
            }
        ],
    }
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return [
        {"role": "user", "content": "hi"},
        message,
        {"role": "tool", "tool_call_id": "c1", "content": "r"},
    ]


class TestUseOfficialKimiK3Encoding(unittest.TestCase):
    def test_routes_rendering_through_official_encoder_idempotently(self):
        module, tokenizer = _install_fake_tokenizer("route")
        self.assertEqual(tokenizer.apply_chat_template([]), "shipped")

        use_official_kimi_k3_encoding(tokenizer)
        use_official_kimi_k3_encoding(tokenizer)

        self.assertIs(module.build_chat_segments, encoding_k3.build_chat_segments)
        prompt = tokenizer.apply_chat_template(
            _assistant_turn('{"a": [1,2,  3], "b": 1e3, "s": "x"}', reasoning=" \n"),
            add_generation_prompt=False,
        )
        # Non-string argument values keep the client's source text.
        self.assertIn('key="a" type="array"<|sep|>[1,2,  3]<|close|>', prompt)
        self.assertIn('key="b" type="number"<|sep|>1e3<|close|>', prompt)
        self.assertIn('key="s" type="string"<|sep|>x<|close|>', prompt)
        # Whitespace-only reasoning is rendered once, verbatim.
        self.assertIn("<|open|>think<|sep|> \n<|close|>think<|sep|>", prompt)

    def test_skips_tokenizer_without_build_chat_segments(self):
        tokenizer = type("Other", (), {"__module__": __name__})()
        with self.assertLogs(encoding_kimi_k3.logger, level="WARNING"):
            use_official_kimi_k3_encoding(tokenizer)


class TestOfficialToolArguments(unittest.TestCase):
    """Raw tool-call argument strings, as serving_chat now forwards them."""

    def _tools_block(self, arguments):
        text = "".join(
            s.text
            for s in encoding_k3.build_chat_segments(
                _assistant_turn(arguments), add_generation_prompt=False
            )
        )
        return text[text.index("<|open|>call") : text.index("<|close|>call")]

    def test_malformed_or_non_object_falls_back_to_json_block(self):
        for arguments in (" \n", '["a"]', "{bad"):
            with self.subTest(arguments=arguments):
                self.assertIn(
                    f'<|open|>json type="object"<|sep|>{arguments}<|close|>json',
                    self._tools_block(arguments),
                )

    def test_empty_arguments_render_no_argument(self):
        self.assertNotIn("argument", self._tools_block(""))


if __name__ == "__main__":
    unittest.main()
