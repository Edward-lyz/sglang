"""Parser for JSON array tool calls with JSON schema constraints.

This module provides JsonArrayParser for parsing JSON array tool calls
when JSON schema constraints are active, typically used when tool_choice
is set to "required" or a specific tool is named.
"""

import logging
from typing import List, Optional

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import StreamingParseResult
from sglang.srt.function_call.utils import (
    StreamingInputBuffer,
    collect_streaming_markers,
)

logger = logging.getLogger(__name__)


class JsonArrayParser(BaseFormatDetector):
    """
    Parser for JSON array tool calls when JSON schema constraints are active.

    This parser is used when tool_choice="required" or a specific tool is named,
    bypassing model-specific parsers in favor of direct JSON array parsing.
    """

    def __init__(
        self,
        tools: Optional[List[Tool]] = None,
        *,
        canonical_streaming: bool = False,
        tokenizer=None,
    ):
        super().__init__(tools=tools or [])
        # Configure for JSON array parsing
        self.bot_token = "["
        self.eot_token = "]"
        self.tool_call_separator = ","
        self._stream_input = (
            StreamingInputBuffer(collect_streaming_markers(self, tokenizer))
            if canonical_streaming
            else None
        )

    def has_tool_call(self, text: str) -> bool:
        """
        Check if the given text contains a JSON tool call (array or single object).
        """
        return "[" in text or "{" in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        Parse JSON tool calls using the base class implementation.
        """
        raise NotImplementedError(
            "Detect and parse not supported for JSON schema constraints."
        )

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool], *, final: bool = False
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing with tool validation.
        """
        if self._stream_input is None:
            return super().parse_streaming_increment(new_text, tools)

        parts = self._stream_input.feed(new_text)
        if final:
            parts.extend(self._stream_input.finish())
        normal_text = []
        calls = []
        for part in parts:
            result = super().parse_streaming_increment(part, tools)
            normal_text.append(result.normal_text)
            calls.extend(result.calls)
        if final:
            while True:
                result = super().parse_streaming_increment("", tools)
                normal_text.append(result.normal_text)
                calls.extend(result.calls)
                if not result.normal_text and not result.calls:
                    break
        return StreamingParseResult(normal_text="".join(normal_text), calls=calls)

    def replay_continuation(self, continuation_text: str) -> bool:
        """Replay parser state and report whether it emitted tool calls."""
        if not continuation_text:
            return False
        if not self.tools:
            return False
        # First call with the full continuation text
        result = self.parse_streaming_increment(continuation_text, self.tools)
        saw_tool_calls = bool(result.calls)
        if self._stream_input is not None:
            return saw_tool_calls

        # Keep draining buffered content until parser reaches stable state
        max_iters = 100  # safety limit
        for _ in range(max_iters):
            result = self.parse_streaming_increment("", self.tools)
            saw_tool_calls = saw_tool_calls or bool(result.calls)
            if not result.calls and not result.normal_text:
                break
        else:
            logger.warning(
                "replay_continuation: hit max_iters=%d; "
                "parser state may be incomplete",
                max_iters,
            )
        return saw_tool_calls

    def structure_info(self) -> callable:
        """
        Return a function that creates StructureInfo for constrained generation.
        This is not used for JSON schema constraints as they are handled
        by the constraint backends directly.
        """
        raise NotImplementedError("structure_info not used for JSON schema constraints")
