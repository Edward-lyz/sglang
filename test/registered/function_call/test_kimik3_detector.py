import json
import sys
import unittest

import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.kimik3_detector import (
    KimiK3Detector as KimiK3FuncDetector,
)
from sglang.srt.function_call.kimik3_format import (
    MESSAGE_CLOSE,
    RESPONSE_CLOSE,
    RESPONSE_OPEN,
    THINK_CLOSE,
    THINK_OPEN,
    TOOLS_CLOSE,
    TOOLS_OPEN,
)
from sglang.srt.parser.reasoning_parser import KimiK3Detector as KimiK3ReasoningDetector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_tool(name):
    return Tool(
        type="function",
        function=Function(
            name=name,
            description=f"{name} tool",
            parameters={
                "type": "object",
                "properties": {"code": {"type": "string"}},
            },
        ),
    )


def _call_block(tool, index, args):
    parts = [f'<|open|>call tool="{tool}" index="{index}"<|sep|>']
    for key, (arg_type, value) in args.items():
        parts.append(
            f'<|open|>argument key="{key}" type="{arg_type}"<|sep|>'
            f"{value}<|close|>argument<|sep|>"
        )
    parts.append("<|close|>call<|sep|>")
    return "".join(parts)


def _stream_chunks(text, size):
    return [text[i : i + size] for i in range(0, len(text), size)]


class TestKimiK3FuncDetector(unittest.TestCase):
    def setUp(self):
        self.detector = KimiK3FuncDetector()
        self.tools = [_make_tool("python")]

    def _stream(self, chunks):
        text = ""
        calls = []
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            text += result.normal_text
            calls.extend(result.calls)
        return text, calls

    @staticmethod
    def _assemble(events):
        """Rebuild tool calls the way an OpenAI streaming client does: keep the
        first non-null name per index and concatenate every argument delta."""
        calls = {}
        for item in events:
            slot = calls.setdefault(item.tool_index, ["", ""])
            if item.name:
                slot[0] = item.name
            slot[1] += item.parameters
        return {index: tuple(slot) for index, slot in calls.items()}

    def _assert_stream_matches_one_shot(self, text, size):
        """Stream `text` in `size`-char chunks and return the emitted events.

        Asserts the two contracts the serving layer depends on: the assembled
        result equals the one-shot parse, and every prefix a client has
        accumulated so far stays a prefix of the final arguments (a delta that
        breaks this cannot be taken back once sent).
        """
        expected = {
            call.tool_index: (call.name, call.parameters)
            for call in KimiK3FuncDetector().detect_and_parse(text, self.tools).calls
        }
        self.detector = KimiK3FuncDetector()
        _, events = self._stream(_stream_chunks(text, size))

        running = {}
        for item in events:
            running[item.tool_index] = (
                running.get(item.tool_index, "") + item.parameters
            )
            self.assertTrue(
                expected[item.tool_index][1].startswith(running[item.tool_index]),
                f"chunk={size} tool_index={item.tool_index}: streamed "
                f"{running[item.tool_index]!r} is not a prefix of "
                f"{expected[item.tool_index][1]!r}",
            )
        self.assertEqual(self._assemble(events), expected, f"chunk={size}")
        return events

    def test_streaming_split_markers(self):
        chunks = [
            "<|open|>",
            "response",
            "<|sep|>Hel",
            "lo!",
            "<|close|>respo",
            "nse<|sep|>",
            "<|open|>tools",
            "<|sep|>",
            '<|open|>call tool="py',
            'thon" index="1"<|sep|>',
            '<|open|>argument key="code" type="string"<|sep|>',
            "print(2)",
            "<|close|>argument<|sep|>",
            "<|close|>call<|sep|>",
            "<|close|>tools<|sep|>",
        ]
        text, events = self._stream(chunks)
        self.assertEqual(text, "Hello!")
        self.assertEqual(
            self._assemble(events), {0: ("python", '{"code": "print(2)"}')}
        )

    def test_streaming_two_calls_small_chunks(self):
        text = (
            f"{TOOLS_OPEN}"
            + _call_block("python", 1, {"code": ("string", "a")})
            + _call_block("python", 2, {"code": ("string", "b")})
            + f"{TOOLS_CLOSE}"
        )
        _, events = self._stream(_stream_chunks(text, 7))
        self.assertEqual(
            self._assemble(events),
            {0: ("python", '{"code": "a"}'), 1: ("python", '{"code": "b"}')},
        )

    def test_streaming_emits_argument_deltas(self):
        """Arguments stream incrementally instead of landing in one chunk.

        Regression guard: the detector used to withhold a call until its closing
        marker arrived, so a long argument produced a single chunk identical to
        the non-streaming response.
        """
        text = (
            f"{TOOLS_OPEN}"
            + _call_block("python", 1, {"code": ("string", "print(1)\nprint(2)")})
            + f"{TOOLS_CLOSE}"
        )
        events = self._assert_stream_matches_one_shot(text, 1)

        # One name-only event, then several argument deltas.
        self.assertEqual([item.name for item in events].count("python"), 1)
        self.assertEqual(events[0].parameters, "")
        self.assertGreater(len(events), 4)
        self.assertTrue(all(item.name is None for item in events[1:]))

    def test_streaming_prefix_invariant_across_chunk_boundaries(self):
        """No chunk boundary may produce a delta that isn't a prefix of the final
        arguments, for any argument shape.

        Covers the shapes whose partial serialization is unstable: a value holding
        a marker fragment, one needing JSON escaping, a non-string value that is
        unparseable for several chunks before it becomes an object (a
        pretty-printed value whose leading indent is not JSON on its own — folding
        that indent in as a quoted string would flip shape later and corrupt the
        arguments a client already accumulated), and one that never parses.
        """
        text = (
            f"{TOOLS_OPEN}"
            + _call_block(
                "python",
                1,
                {
                    "marker": ("string", "a<|close|>x"),
                    "escaped": ("string", 'tab\t and "quote" and \\'),
                    "flag": ("boolean", "true"),
                    "opts": ("object", '\n  {"a": 1, "b": [1, 2]}'),
                    "broken": ("object", "{not json"),
                },
            )
            + _call_block("python", 2, {"n": ("number", "-3.5")})
            + f"{TOOLS_CLOSE}"
        )
        for size in (1, 2, 3, 5, 11, 37, len(text)):
            self._assert_stream_matches_one_shot(text, size)

    def test_streaming_legacy_mode_emits_one_item_per_call(self):
        """`SGLANG_ENABLE_KIMI_K3_TOOL_ARGUMENT_STREAMING=0` restores the
        pre-delta wire shape for clients that cannot consume partial arguments:
        nothing is emitted until a call's closing marker arrives, and the call
        then lands as a single item carrying both the name and the complete
        arguments.
        """
        text = (
            f"{TOOLS_OPEN}"
            + _call_block("python", 1, {"code": ("string", "print(1)\nprint(2)")})
            + _call_block("python", 2, {"code": ("string", "b")})
            + f"{TOOLS_CLOSE}"
        )
        with envs.SGLANG_ENABLE_KIMI_K3_TOOL_ARGUMENT_STREAMING.override(False):
            self.detector = KimiK3FuncDetector()
            _, events = self._stream(_stream_chunks(text, 3))
        self.assertEqual(
            [(item.tool_index, item.name, item.parameters) for item in events],
            [
                (0, "python", '{"code": "print(1)\\nprint(2)"}'),
                (1, "python", '{"code": "b"}'),
            ],
        )

    def test_streaming_plain_text_only(self):
        text, calls = self._stream(["just a ", "plain ", "reply"])
        self.assertEqual(text, "just a plain reply")
        self.assertEqual(calls, [])

    def test_streaming_bookkeeping_for_serving_layer(self):
        text = (
            f"{TOOLS_OPEN}"
            + _call_block("python", 1, {"code": ("string", "a")})
            + f"{TOOLS_CLOSE}"
        )
        self._stream(_stream_chunks(text, 9))
        self.assertEqual(self.detector.current_tool_id, 0)
        self.assertEqual(
            self.detector.prev_tool_call_arr[0],
            {"name": "python", "arguments": {"code": "a"}},
        )
        self.assertEqual(
            json.loads(self.detector.streamed_args_for_tool[0]), {"code": "a"}
        )

    def test_streaming_truncated_call_can_be_closed_by_serving_layer(self):
        """A stream cut mid-argument must leave the serving layer able to finish
        the JSON: `prev_tool_call_arr` holds a dict whose serialization extends
        what was already streamed (see `_check_for_unstreamed_tool_args`)."""
        text = (
            f"{TOOLS_OPEN}"
            '<|open|>call tool="python" index="1"<|sep|>'
            '<|open|>argument key="code" type="string"<|sep|>print(1'
        )
        _, events = self._stream(list(text))
        streamed = self._assemble(events)[0][1]

        expected_args = self.detector.prev_tool_call_arr[0]["arguments"]
        self.assertIsInstance(expected_args, dict)
        expected_call = json.dumps(expected_args, ensure_ascii=False)
        self.assertTrue(expected_call.startswith(streamed))
        self.assertEqual(expected_call, '{"code": "print(1"}')

    def test_detect_and_parse_single_call(self):
        text = (
            f"{RESPONSE_OPEN}Let me run it.{RESPONSE_CLOSE}"
            f"{TOOLS_OPEN}"
            + _call_block(
                "python",
                1,
                {"code": ("string", "print(1)"), "opts": ("object", '{"a": 1}')},
            )
            + f"{TOOLS_CLOSE}"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, "Let me run it.")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "python")
        self.assertEqual(
            json.loads(result.calls[0].parameters),
            {"code": "print(1)", "opts": {"a": 1}},
        )

    def test_detect_and_parse_no_tools_channel(self):
        text = f"{RESPONSE_OPEN}hi there{RESPONSE_CLOSE}{MESSAGE_CLOSE}"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, "hi there")
        self.assertEqual(result.calls, [])

    def test_detect_and_parse_multiple_calls(self):
        text = (
            f"{TOOLS_OPEN}"
            + _call_block("python", 1, {"code": ("string", "a")})
            + _call_block("python", 2, {"code": ("string", "b")})
            + f"{TOOLS_CLOSE}"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].tool_index, 0)
        self.assertEqual(result.calls[1].tool_index, 1)
        self.assertEqual(json.loads(result.calls[1].parameters), {"code": "b"})

    def test_detect_and_parse_unclosed_tools_section(self):
        text = f"{TOOLS_OPEN}" + _call_block("python", 1, {"code": ("string", "x")})
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(json.loads(result.calls[0].parameters), {"code": "x"})

    def test_attr_unescaping_and_raw_string_args(self):
        text = (
            f"{TOOLS_OPEN}"
            '<|open|>call tool="a&amp;b" index="1"<|sep|>'
            '<|open|>argument key="q" type="string"<|sep|>'
            "say &quot;hi&quot;<|close|>argument<|sep|>"
            "<|close|>call<|sep|>"
            f"{TOOLS_CLOSE}"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.calls[0].name, "a&b")
        self.assertEqual(
            json.loads(result.calls[0].parameters), {"q": "say &quot;hi&quot;"}
        )

    def test_non_string_arg_json_decoding(self):
        text = (
            f"{TOOLS_OPEN}"
            + _call_block(
                "python",
                1,
                {
                    "n": ("number", "42"),
                    "flag": ("boolean", "true"),
                    "bad": ("object", "{not json"),
                },
            )
            + f"{TOOLS_CLOSE}"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(
            json.loads(result.calls[0].parameters),
            {"n": 42, "flag": True, "bad": "{not json"},
        )

    def test_detect_and_parse_unclosed_call(self):
        """The one-shot path must not surface a call whose closing marker never
        arrived: the streaming regex tolerates the missing terminator, so only the
        `end` group gate keeps half-read arguments out of a non-streamed response."""
        text = (
            f"{TOOLS_OPEN}"
            '<|open|>call tool="python" index="1"<|sep|>'
            '<|open|>argument key="code" type="string"<|sep|>print(1'
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.calls, [])

    def test_has_tool_call(self):
        self.assertTrue(self.detector.has_tool_call(f"x{TOOLS_OPEN}y"))
        self.assertFalse(self.detector.has_tool_call(f"{RESPONSE_OPEN}x"))

    def test_constraint_capabilities(self):
        self.assertTrue(self.detector.supports_structural_tag())
        self.assertFalse(self.detector.parses_required_natively())


class TestKimiK3ReasoningDetector(unittest.TestCase):
    def _stream(self, detector, chunks):
        reasoning = ""
        content = ""
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk)
            reasoning += result.reasoning_text or ""
            content += result.normal_text or ""
        return reasoning, content

    def test_non_stream_full_markers(self):
        detector = KimiK3ReasoningDetector(force_reasoning=True)
        result = detector.detect_and_parse(
            f"{THINK_OPEN}deep thought{THINK_CLOSE}"
            f"{RESPONSE_OPEN}the answer{RESPONSE_CLOSE}{MESSAGE_CLOSE}"
        )
        self.assertEqual(result.reasoning_text, "deep thought")
        self.assertEqual(result.normal_text, "the answer")

    def test_non_stream_consumed_prefix(self):
        detector = KimiK3ReasoningDetector(force_reasoning=True)
        result = detector.detect_and_parse(
            f"thinking...{THINK_CLOSE}{RESPONSE_OPEN}done{RESPONSE_CLOSE}"
        )
        self.assertEqual(result.reasoning_text, "thinking...")
        self.assertEqual(result.normal_text, "done")

    def test_non_stream_thinking_disabled(self):
        detector = KimiK3ReasoningDetector(force_reasoning=False)
        result = detector.detect_and_parse(
            f"{RESPONSE_OPEN}plain reply{RESPONSE_CLOSE}{MESSAGE_CLOSE}"
        )
        self.assertFalse(result.reasoning_text)
        self.assertEqual(result.normal_text, "plain reply")

    def test_non_stream_truncated_reasoning(self):
        detector = KimiK3ReasoningDetector(force_reasoning=True)
        result = detector.detect_and_parse("still going")
        self.assertEqual(result.reasoning_text, "still going")
        self.assertFalse(result.normal_text)

    def test_non_stream_tools_channel_passthrough(self):
        detector = KimiK3ReasoningDetector(force_reasoning=True)
        tools_channel = (
            f"{TOOLS_OPEN}"
            + _call_block("python", 1, {"code": ("string", "1")})
            + f"{TOOLS_CLOSE}"
        )
        result = detector.detect_and_parse(
            f"t{THINK_CLOSE}{RESPONSE_OPEN}r{RESPONSE_CLOSE}{tools_channel}"
        )
        self.assertEqual(result.reasoning_text, "t")
        self.assertEqual(result.normal_text, f"r{tools_channel}")

    def test_streaming_split_markers(self):
        detector = KimiK3ReasoningDetector(force_reasoning=True)
        chunks = [
            "I am thinking<|close|>thi",
            "nk<|sep|>",
            "<|open|>respon",
            "se<|sep|>Ans",
            "wer here",
            "<|close|>response<|sep|>",
            "<|close|>message<|sep|>",
        ]
        reasoning, content = self._stream(detector, chunks)
        self.assertEqual(reasoning, "I am thinking")
        self.assertEqual(content, "Answer here")

    def test_streaming_char_by_char(self):
        detector = KimiK3ReasoningDetector(force_reasoning=True)
        full = (
            f"{THINK_OPEN}abc{THINK_CLOSE}"
            f"{RESPONSE_OPEN}xyz{RESPONSE_CLOSE}{MESSAGE_CLOSE}"
        )
        reasoning, content = self._stream(detector, list(full))
        self.assertEqual(reasoning, "abc")
        self.assertEqual(content, "xyz")

    def test_streaming_chains_into_tool_detector(self):
        detector = KimiK3ReasoningDetector(force_reasoning=True)
        full = (
            f"hmm{THINK_CLOSE}{RESPONSE_OPEN}ok{RESPONSE_CLOSE}"
            f"{TOOLS_OPEN}"
            + _call_block("python", 1, {"code": ("string", "1")})
            + f"{TOOLS_CLOSE}"
        )
        reasoning, content = self._stream(detector, _stream_chunks(full, 5))
        self.assertEqual(reasoning, "hmm")
        self.assertTrue(content.startswith(f"ok{TOOLS_OPEN}"))

        func_detector = KimiK3FuncDetector()
        result = func_detector.parse_streaming_increment(
            content, [_make_tool("python")]
        )
        self.assertEqual(result.normal_text, "ok")
        self.assertEqual([item.name for item in result.calls], ["python", None])
        self.assertEqual(
            "".join(item.parameters for item in result.calls), '{"code": "1"}'
        )

    def test_streaming_thinking_disabled(self):
        detector = KimiK3ReasoningDetector(force_reasoning=False)
        full = f"{RESPONSE_OPEN}plain{RESPONSE_CLOSE}{MESSAGE_CLOSE}"
        reasoning, content = self._stream(detector, _stream_chunks(full, 6))
        self.assertEqual(reasoning, "")
        self.assertEqual(content, "plain")

    def test_streaming_in_band_open_marker(self):
        detector = KimiK3ReasoningDetector(force_reasoning=False)
        full = (
            f"{THINK_OPEN}abc{THINK_CLOSE}"
            f"{RESPONSE_OPEN}xyz{RESPONSE_CLOSE}{MESSAGE_CLOSE}"
        )
        reasoning, content = self._stream(detector, _stream_chunks(full, 4))
        self.assertEqual(reasoning, "abc")
        self.assertEqual(content, "xyz")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
