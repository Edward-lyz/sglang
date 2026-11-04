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

logger = logging.getLogger(__name__)


class JsonArrayParser(BaseFormatDetector):
    """
    Parser for JSON array tool calls when JSON schema constraints are active.

    This parser is used when tool_choice="required" or a specific tool is named,
    bypassing model-specific parsers in favor of direct JSON array parsing.
    """

    def __init__(self, tools: Optional[List[Tool]] = None):
        super().__init__(tools=tools or [])
        # Configure for JSON array parsing
        self.bot_token = "["
        self.eot_token = "]"
        self.tool_call_separator = ","

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
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing with tool validation.
        """
        return super().parse_streaming_increment(new_text, tools)

    # Multi-char structural markers. Must be fed atomically during replay;
    # splitting them mid-marker triggers detector buffer-clear and drops state.
    # Subclasses with extra markers MUST extend this list.
    _REPLAY_MARKER_ATTRS = (
        "bot_token",
        "eot_token",
        "tool_call_separator",
    )

    def replay_continuation(self, continuation_text: str):
        """Replay continuation text to restore parser state.

        Char-by-char so partial-args state (prev_tool_call_arr,
        streamed_args_for_tool) tracks live streaming. Multi-char structural
        markers fed atomically — splitting them would let detector buffer-clear
        drop the marker.
        """
        if not continuation_text or not self.tools:
            return

        markers = {
            v
            for a in self._REPLAY_MARKER_ATTRS
            if isinstance(v := getattr(self, a, None), str) and len(v) > 1
        }
        if not markers:
            for ch in continuation_text:
                self.parse_streaming_increment(ch, self.tools)
            return

        import re

        pattern = "|".join(re.escape(m) for m in sorted(markers, key=len, reverse=True))
        for part in re.split(f"({pattern})", continuation_text):
            if not part:
                continue
            if part in markers:
                self.parse_streaming_increment(part, self.tools)
            else:
                for ch in part:
                    self.parse_streaming_increment(ch, self.tools)

    def structure_info(self) -> callable:
        """
        Return a function that creates StructureInfo for constrained generation.
        This is not used for JSON schema constraints as they are handled
        by the constraint backends directly.
        """
        raise NotImplementedError("structure_info not used for JSON schema constraints")
