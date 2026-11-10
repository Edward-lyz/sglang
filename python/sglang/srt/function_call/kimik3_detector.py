import json
import logging
import re
from typing import Iterator, List, Literal, Optional, Union

from partial_json_parser.core.options import Allow
from xgrammar import StructuralTag

from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
from sglang.srt.environ import envs
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.kimik3_format import (
    ARGUMENT_CLOSE,
    CALL_CLOSE,
    MESSAGE_CLOSE,
    RESPONSE_CLOSE,
    RESPONSE_OPEN,
    TOOLS_CLOSE,
    TOOLS_OPEN,
)
from sglang.srt.function_call.kimik3_structural_tag import (
    get_kimik3_auto_tool_call_structural_tag,
    get_kimik3_structural_tag,
)
from sglang.srt.function_call.utils import _find_common_prefix, _partial_json_loads

logger = logging.getLogger(__name__)

# ``end`` is empty when the closing marker hasn't streamed in yet, which is how
# the streaming path recognizes a call that is still being generated.
_CALL_RE = re.compile(
    r"<\|open\|>call\s+(?P<attrs>(?:(?!<\|sep\|>).)*?)<\|sep\|>"
    r"(?P<body>.*?)(?P<end><\|close\|>call<\|sep\|>|\Z)",
    re.DOTALL,
)
_ARG_RE = re.compile(
    r"<\|open\|>argument\s+(?P<attrs>(?:(?!<\|sep\|>).)*?)<\|sep\|>"
    r"(?P<val>.*?)<\|close\|>argument<\|sep\|>",
    re.DOTALL,
)
# Trailing argument whose closing marker hasn't arrived; matched against the call
# body left over after every complete ``_ARG_RE`` match.
_PARTIAL_ARG_RE = re.compile(
    r"<\|open\|>argument\s+(?P<attrs>(?:(?!<\|sep\|>).)*?)<\|sep\|>(?P<val>.*)\Z",
    re.DOTALL,
)
_ATTR_RE = re.compile(r'(?P<k>\w+)="(?P<v>[^"]*)"')


def _unescape_attr(value: str) -> str:
    return value.replace("&quot;", '"').replace("&amp;", "&")


def _parse_attrs(attrs: str) -> dict:
    return {m["k"]: _unescape_attr(m["v"]) for m in _ATTR_RE.finditer(attrs)}


def _partial_suffix_len(text: str, markers: List[str]) -> int:
    best = 0
    for marker in markers:
        for n in range(min(len(marker) - 1, len(text)), best, -1):
            if text.endswith(marker[:n]):
                best = n
                break
    return best


def _strip_response_wrappers(text: str) -> str:
    open_idx = text.find(RESPONSE_OPEN)
    if open_idx != -1:
        close_idx = text.find(RESPONSE_CLOSE, open_idx + len(RESPONSE_OPEN))
        if close_idx != -1:
            text = text[open_idx + len(RESPONSE_OPEN) : close_idx]
        else:
            text = text[open_idx + len(RESPONSE_OPEN) :]
    else:
        text = text.replace(RESPONSE_CLOSE, "")
    return text.replace(MESSAGE_CLOSE, "")


def _decode_argument(attrs: str, raw_value: str) -> tuple[str, object]:
    arg_attrs = _parse_attrs(attrs)
    key = arg_attrs.get("key", "")
    if arg_attrs.get("type", "string") == "string":
        return key, raw_value
    try:
        return key, json.loads(raw_value)
    except json.JSONDecodeError:
        return key, raw_value


def _decode_partial_argument(remainder: str, arguments: dict) -> None:
    """Fold the still-open trailing argument of a call into ``arguments``.

    ``remainder`` is whatever follows the last closed argument, so it may end in a
    fragment of the closing marker; that fragment is held back rather than leaked
    into the value.
    """
    holdback = _partial_suffix_len(remainder, [ARGUMENT_CLOSE, CALL_CLOSE])
    if holdback:
        remainder = remainder[: len(remainder) - holdback]
    match = _PARTIAL_ARG_RE.search(remainder)
    if match is None or not match["val"]:
        return
    arg_attrs = _parse_attrs(match["attrs"])
    key = arg_attrs.get("key", "")
    if arg_attrs.get("type", "string") == "string":
        arguments[key] = match["val"]
        return
    # A non-string value serializes as a quoted string while its text is not yet
    # valid JSON, then flips to its parsed shape once it is. Emitting the quoted
    # form would break the prefix invariant the streamed deltas rely on, so the
    # key stays out of the partial view until it parses.
    try:
        arguments[key] = _partial_json_loads(match["val"], Allow.ALL)[0]
    except (ValueError, IndexError):
        # partial_json_parser signals "not JSON (yet)" with several ValueError
        # subclasses of its own alongside the stdlib one, and can surface an
        # IndexError from its fast path.
        return


class KimiK3Detector(BaseFormatDetector):
    """Detector for the Kimi K3 XTML tool-call format.

    K3 emits tool calls in a ``tools`` channel built from dedicated special
    tokens; the plain reply lives in a preceding ``response`` channel:

    ```
    <|open|>response<|sep|>text<|close|>response<|sep|>
    <|open|>tools<|sep|>
      <|open|>call tool="name" index="1"<|sep|>
        <|open|>argument key="k" type="string"<|sep|>raw text<|close|>argument<|sep|>
      <|close|>call<|sep|>
    <|close|>tools<|sep|>
    ```

    ``type="string"`` argument values are raw text; other types are
    JSON-decoded. Attribute values reverse the template's ``&amp;``/``&quot;``
    escaping.

    While streaming, arguments are emitted as deltas as the model writes them.
    ``SGLANG_ENABLE_KIMI_K3_TOOL_ARGUMENT_STREAMING=0`` restores the legacy
    behavior for clients that cannot consume partial arguments: a call is
    surfaced only once its closing marker arrives, as one item carrying the name
    and the complete arguments.
    """

    def __init__(
        self,
        tools=None,
        tool_choice="auto",
        parallel_tool_calls=True,
    ):
        super().__init__(tools, tool_choice, parallel_tool_calls)
        self.bot_token = TOOLS_OPEN
        self.eot_token = TOOLS_CLOSE
        self._sent_normal_idx = 0
        self._stream_arguments = (
            envs.SGLANG_ENABLE_KIMI_K3_TOOL_ARGUMENT_STREAMING.get()
        )

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def supports_structural_tag(self) -> bool:
        return True

    def parses_required_natively(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError(
            "Kimi K3 uses its model-native structural tag implementation"
        )

    def get_auto_tool_call_structural_tag(
        self,
        tools: Union[List[Tool], None] = None,
        thinking_mode: bool = False,
        parallel_tool_calls: bool = True,
    ) -> Optional[StructuralTag]:
        return get_kimik3_auto_tool_call_structural_tag(
            tools or [],
            thinking_mode=thinking_mode,
            parallel_tool_calls=parallel_tool_calls,
        )

    def get_structural_tag(
        self,
        tools: Union[List[Tool], None] = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required"]] = "auto",
        thinking_mode: bool = False,
        parallel_tool_calls: bool = True,
    ) -> StructuralTag:
        return get_kimik3_structural_tag(
            tools=tools or [],
            tool_choice=tool_choice,
            thinking_mode=thinking_mode,
            parallel_tool_calls=parallel_tool_calls,
        )

    def _decode_call(
        self, attrs: str, body: str, allow_partial: bool = False
    ) -> dict | None:
        call_attrs = _parse_attrs(attrs)
        tool_name = call_attrs.get("tool", "")
        if not tool_name:
            return None
        arguments = {}
        consumed = 0
        for arg in _ARG_RE.finditer(body):
            key, value = _decode_argument(arg["attrs"], arg["val"])
            arguments[key] = value
            consumed = arg.end()
        if allow_partial:
            _decode_partial_argument(body[consumed:], arguments)
        return {
            "name": tool_name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        }

    def _parse_calls(self, section: str) -> List[dict]:
        """Fully closed calls only — the one-shot path must not surface half-read
        arguments as if the model had finished writing them."""
        return [
            call
            for m in _CALL_RE.finditer(section)
            if m["end"] and (call := self._decode_call(m["attrs"], m["body"])) is not None
        ]

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        open_idx = text.find(self.bot_token)
        if open_idx == -1:
            return StreamingParseResult(normal_text=_strip_response_wrappers(text))
        try:
            before = _strip_response_wrappers(text[:open_idx])
            section_start = open_idx + len(self.bot_token)
            close_idx = text.find(self.eot_token, section_start)
            section = (
                text[section_start:]
                if close_idx == -1
                else text[section_start:close_idx]
            )
            calls = [
                ToolCallItem(
                    tool_index=i,
                    name=call["name"],
                    parameters=call["arguments"],
                )
                for i, call in enumerate(self._parse_calls(section))
            ]
            return StreamingParseResult(normal_text=before, calls=calls)
        except Exception as e:
            logger.error("Error in Kimi K3 detect_and_parse: %s", e, exc_info=True)
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        self._buffer += new_text
        try:
            open_idx = self._buffer.find(self.bot_token)
            if open_idx == -1:
                return StreamingParseResult(normal_text=self._emit_normal_text())

            normal_text = self._emit_normal_text(limit=open_idx)
            section = self._buffer[open_idx + len(self.bot_token) :]
            calls = []
            for index, (decoded, is_complete) in enumerate(
                self._iter_streaming_calls(section)
            ):
                if index >= self.current_tool_id:
                    calls.extend(
                        self._stream_call(index, decoded, is_complete=is_complete)
                    )
            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(
                "Error in Kimi K3 parse_streaming_increment: %s", e, exc_info=True
            )
            self._buffer = ""
            return StreamingParseResult()

    def _iter_streaming_calls(self, section: str) -> Iterator[tuple[dict, bool]]:
        """Yield ``(decoded_call, is_complete)`` for every call in emission order.

        Undecodable headers are dropped so the yielded position matches the
        ``tool_index`` the one-shot path would assign. Only the last call of a
        section can be incomplete, so legacy mode simply stops there.
        """
        for m in _CALL_RE.finditer(section):
            is_complete = bool(m["end"])
            if not is_complete and not self._stream_arguments:
                return
            decoded = self._decode_call(
                m["attrs"], m["body"], allow_partial=not is_complete
            )
            if decoded is not None:
                yield decoded, is_complete

    def _stream_call(
        self, index: int, decoded: dict, *, is_complete: bool
    ) -> List[ToolCallItem]:
        calls = []
        if index > self.current_tool_id:
            self.current_tool_id = index
            while len(self.prev_tool_call_arr) <= index:
                self.prev_tool_call_arr.append({})
            while len(self.streamed_args_for_tool) <= index:
                self.streamed_args_for_tool.append("")
            if self._stream_arguments:
                calls.append(
                    ToolCallItem(tool_index=index, name=decoded["name"], parameters="")
                )

        arguments = decoded["arguments"]
        delta = self._argument_delta(index, arguments, is_complete=is_complete)
        self.prev_tool_call_arr[index] = {
            "name": decoded["name"],
            "arguments": json.loads(arguments),
        }
        if delta:
            self.streamed_args_for_tool[index] += delta
            calls.append(
                ToolCallItem(
                    tool_index=index,
                    # Legacy mode emits a single item per call, so the name rides
                    # along with the complete arguments.
                    name=None if self._stream_arguments else decoded["name"],
                    parameters=delta,
                )
            )
        return calls

    def _argument_delta(self, index: int, current: str, *, is_complete: bool) -> str:
        """Next slice of the arguments JSON that is safe to hand to the client.

        Clients concatenate these slices, so everything emitted has to stay a
        prefix of the final arguments JSON. While the call is open only the prefix
        that two consecutive parses agree on is trusted — the unstable tail (the
        value's closing quote, the closing brace) always differs and is withheld.
        """
        streamed = self.streamed_args_for_tool[index]
        if is_complete:
            target = current
        else:
            previous = self.prev_tool_call_arr[index].get("arguments")
            if previous is None:
                return ""
            previous_json = json.dumps(previous, ensure_ascii=False)
            if current == previous_json:
                # Two identical parses agree on the closing brace too, which is
                # exactly the character that must not go out early.
                return ""
            target = _find_common_prefix(current, previous_json)
        if not target.startswith(streamed):
            if is_complete:
                logger.warning(
                    "Kimi K3 streamed arguments diverged from the parsed call "
                    "(tool_index=%d): streamed=%r parsed=%r",
                    index,
                    streamed,
                    current,
                )
            return ""
        return target[len(streamed) :]

    def _emit_normal_text(self, limit: int | None = None) -> str:
        if limit is None:
            holdback = _partial_suffix_len(
                self._buffer,
                [self.bot_token, RESPONSE_OPEN, RESPONSE_CLOSE, MESSAGE_CLOSE],
            )
            limit = len(self._buffer) - holdback
        if limit <= self._sent_normal_idx:
            return ""
        pending = self._buffer[self._sent_normal_idx : limit]
        for marker in (RESPONSE_OPEN, RESPONSE_CLOSE, MESSAGE_CLOSE):
            if marker in pending:
                pending = pending.replace(marker, "")
        self._sent_normal_idx = limit
        return pending
