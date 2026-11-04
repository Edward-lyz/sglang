"""Unit tests for the DeepSeek-V4.1 chat encoder -- no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.entrypoints.openai.encoding_dsv41 import (
    encode_messages,
)
from sglang.test.test_utils import CustomTestCase


def _tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a value",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
            },
        },
    }


class TestDsmlTags(CustomTestCase):
    def test_multiturn_prompt_preserves_reasoning_and_tool_results(self):
        messages = [
            {"role": "system", "content": "system", "tools": [_tool()]},
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "reasoning_content": "reason",
                "content": "summary",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "lookup", "arguments": {"query": query}},
                    }
                    for call_id, query in (("a", "first"), ("b", "second"))
                ],
            },
            {"role": "tool", "tool_call_id": "b", "content": "second result"},
            {"role": "tool", "tool_call_id": "a", "content": "first result"},
        ]
        prompt = encode_messages(messages, thinking_mode="thinking")
        expected = (
            "<｜User｜>question<｜Assistant｜><think>reason</think>summary\n\n"
            "<｜DSML｜ calls>\n"
            '<｜DSML｜ invoke name="lookup">\n'
            '<｜DSML｜ parameter name="query" string="true">first</｜DSML｜ parameter>\n'
            "</｜DSML｜ invoke>\n"
            '<｜DSML｜ invoke name="lookup">\n'
            '<｜DSML｜ parameter name="query" string="true">second</｜DSML｜ parameter>\n'
            "</｜DSML｜ invoke>\n"
            "</｜DSML｜ calls><｜end▁of▁sentence｜>"
            "<｜User｜><tool_result>first result</tool_result>\n\n"
            "<tool_result>second result</tool_result><｜Assistant｜><think>"
        )
        self.assertEqual(prompt[prompt.index("<｜User｜>") :], expected)

    def test_malformed_history_arguments_render_verbatim(self):
        """A history call whose arguments are malformed JSON must encode, not 400."""
        malformed = '{"edits": [{{\\"old\\": 1}]}'
        messages = [
            {"role": "system", "content": "", "tools": [_tool()]},
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "a",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": malformed},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": "invalid arguments"},
        ]
        prompt = encode_messages(messages, thinking_mode="chat")
        self.assertIn(
            f'<｜DSML｜ invoke name="lookup">\n{malformed}\n</｜DSML｜ invoke>', prompt
        )

    def test_non_object_history_arguments_still_rejected(self):
        messages = [
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "a",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '"Beijing"'},
                    }
                ],
            },
        ]
        with self.assertRaisesRegex(ValueError, "must be a JSON object"):
            encode_messages(messages, thinking_mode="chat")


if __name__ == "__main__":
    unittest.main()
