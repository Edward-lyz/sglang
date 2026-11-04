import inspect
import logging
from typing import Dict, List, Literal, Optional, Set, Tuple, Type, Union

from sglang.srt.entrypoints.openai.protocol import (
    LegacyStructuralTagResponseFormat,
    StructuralTagResponseFormat,
    StructuresResponseFormat,
    Tool,
    ToolCallConstraint,
    ToolChoice,
)
from sglang.srt.environ import ToolStrictLevel, envs
from sglang.srt.function_call.apertus2509_detector import Apertus2509Detector
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.cohere_command4_detector import CohereCommand4Detector
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.deepseekv3_detector import DeepSeekV3Detector
from sglang.srt.function_call.deepseekv4_detector import DeepSeekV4Detector
from sglang.srt.function_call.deepseekv31_detector import DeepSeekV31Detector
from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector
from sglang.srt.function_call.gemma4_detector import Gemma4Detector
from sglang.srt.function_call.gigachat3_detector import GigaChat3Detector
from sglang.srt.function_call.glm4_moe_detector import Glm4MoeDetector
from sglang.srt.function_call.glm47_moe_detector import Glm47MoeDetector
from sglang.srt.function_call.gpt_oss_detector import GptOssDetector
from sglang.srt.function_call.hermes_detector import HermesDetector
from sglang.srt.function_call.hunyuan_detector import HunyuanDetector
from sglang.srt.function_call.inkling_detector import InklingDetector
from sglang.srt.function_call.internlm_detector import InternlmDetector
from sglang.srt.function_call.kimik2_detector import KimiK2Detector
from sglang.srt.function_call.kimik3_detector import KimiK3Detector
from sglang.srt.function_call.kimik3_structural_tag import promote_tools_to_strict
from sglang.srt.function_call.lfm2_detector import Lfm2Detector
from sglang.srt.function_call.llama32_detector import Llama32Detector
from sglang.srt.function_call.mimo_detector import MiMoDetector
from sglang.srt.function_call.minicpm5_detector import MiniCPM5Detector
from sglang.srt.function_call.minimax_m2 import MinimaxM2Detector
from sglang.srt.function_call.minimax_m3 import MinimaxM3Detector
from sglang.srt.function_call.mistral_detector import MistralDetector
from sglang.srt.function_call.poolside_v1_detector import PoolsideV1Detector
from sglang.srt.function_call.pythonic_detector import PythonicDetector
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from sglang.srt.function_call.qwen25_detector import Qwen25Detector
from sglang.srt.function_call.step3_detector import Step3Detector
from sglang.srt.function_call.trinity_detector import TrinityDetector
from sglang.srt.function_call.utils import (
    StreamingInputBuffer,
    _get_tool_schema_defs,
    collect_streaming_markers,
    get_json_schema_constraint,
)

logger = logging.getLogger(__name__)


class FunctionCallParser:
    """
    Parser for function/tool calls in model outputs.

    This class handles both streaming and non-streaming parsing of function calls using a detector.
    In streaming scenarios, each time new_text is received, it calls detector.parse_streaming_increment
    and returns the resulting normal_text and calls to the upper layer (or SSE).
    """

    ToolCallParserEnum: Dict[str, Type[BaseFormatDetector]] = {
        "apertus2509": Apertus2509Detector,
        "cohere_command4": CohereCommand4Detector,
        "deepseekv3": DeepSeekV3Detector,
        "deepseekv31": DeepSeekV31Detector,
        "deepseekv32": DeepSeekV32Detector,
        "deepseekv4": DeepSeekV4Detector,
        "glm": Glm4MoeDetector,
        "glm45": Glm4MoeDetector,
        "glm47": Glm47MoeDetector,
        "gpt-oss": GptOssDetector,
        "kimi_k2": KimiK2Detector,
        "kimi_k3": KimiK3Detector,
        "lfm2": Lfm2Detector,
        "llama3": Llama32Detector,
        "mimo": MiMoDetector,
        "minicpm5": MiniCPM5Detector,
        "mistral": MistralDetector,
        "poolside_v1": PoolsideV1Detector,
        "pythonic": PythonicDetector,
        "qwen": Qwen25Detector,
        "qwen25": Qwen25Detector,
        "qwen3_coder": Qwen3CoderDetector,
        "step3": Step3Detector,
        "step3p5": Qwen3CoderDetector,
        "minimax-m2": MinimaxM2Detector,
        "minimax-m3": MinimaxM3Detector,
        "trinity": TrinityDetector,
        "interns1": InternlmDetector,
        "hermes": HermesDetector,
        "hunyuan": HunyuanDetector,
        "gigachat3": GigaChat3Detector,
        "gemma4": Gemma4Detector,
        "inkling": InklingDetector,
    }

    def __init__(
        self,
        tool_call_parser: str,
        tools: List[Tool],
        tool_choice: Union[Literal["required", "auto", "none"], ToolChoice] = "auto",
        parallel_tool_calls: bool = True,
        thinking_mode: bool = False,
        tokenizer=None,
        canonical_streaming: bool = False,
    ):
        detector_class = self.ToolCallParserEnum.get(tool_call_parser)
        if detector_class:
            signature = inspect.signature(detector_class)
            accepts_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            kwargs = {
                "tools": tools,
                "tool_choice": tool_choice,
                "parallel_tool_calls": parallel_tool_calls,
            }
            if tokenizer is not None:
                kwargs["tokenizer"] = tokenizer
            if not accepts_kwargs:
                # Older detectors receive request options through their parse
                # methods and still expose a parameterless constructor.
                kwargs = {
                    name: value
                    for name, value in kwargs.items()
                    if name in signature.parameters
                }
            detector = detector_class(**kwargs)
        else:
            raise ValueError(f"Unsupported tool_call_parser: {tool_call_parser}")

        self.detector = detector
        self.tools = tools
        self.tool_choice = tool_choice
        self.tool_strict_level = envs.SGLANG_TOOL_STRICT_LEVEL.get()
        self.parallel_tool_calls = parallel_tool_calls
        self.thinking_mode = thinking_mode
        self._stream_input = (
            StreamingInputBuffer(collect_streaming_markers(detector, tokenizer))
            if canonical_streaming
            else None
        )

    def has_tool_call(self, text: str) -> bool:
        """
        Check if the given text contains a tool call in the format supported by this parser.
        This delegates to the detector's implementation.

        Args:
            text: The text to check for tool calls

        Returns:
            True if the text contains a tool call, False otherwise
        """
        if not self.tools:
            return False
        return self.detector.has_tool_call(text)

    def parse_non_stream(self, full_text: str) -> Tuple[str, list[ToolCallItem]]:
        """
        One-time parsing of the full text to extract tool calls.

        Args:
            full_text: The complete text to parse

        Returns:
            A tuple containing:
            - The remaining text after parsing that was not consumed by the detector (can be treated as normal text)
            - A list of tool calls parsed from the text
        """
        if not self.tools:
            return full_text, []
        has_tool_call = self.detector.has_tool_call(full_text)
        parsed_result = self.detector.detect_and_parse(full_text, self.tools)
        tool_call_list = parsed_result.calls
        if tool_call_list or has_tool_call:
            return parsed_result.normal_text, tool_call_list
        else:
            return full_text, []

    def parse_stream_chunk(
        self, chunk_text: str, *, final: bool = False
    ) -> Tuple[str, list[ToolCallItem]]:
        """
        Streaming incremental parsing of chunks of text as they arrive.

        Args:
            chunk_text: The new chunk of text to parse

        Returns:
            A tuple containing:
            - The normal text that should be displayed to the user
            - A list of tool calls parsed from the chunk
        """
        if not self.tools:
            return chunk_text, []
        final_normal_text = []
        final_calls = []

        if self._stream_input is None:
            parts = [chunk_text]
        else:
            parts = self._stream_input.feed(chunk_text)
            if final:
                parts.extend(self._stream_input.finish())

        for part in parts:
            result = self.detector.parse_streaming_increment(part, self.tools)
            final_normal_text.append(result.normal_text)
            final_calls.extend(result.calls)

        if final and self._stream_input is not None:
            # Some detectors emit a name, arguments and subsequent calls on
            # separate invocations. Drain only at a real stop, never at a
            # request's length boundary or the end of a replayed prefix.
            while True:
                result = self.detector.parse_streaming_increment("", self.tools)
                final_normal_text.append(result.normal_text)
                final_calls.extend(result.calls)
                if not result.normal_text and not result.calls:
                    break

        return "".join(final_normal_text), final_calls

    def replay_continuation(self, continuation_text: str) -> bool:
        """Replay parser state and report whether it emitted tool calls."""
        if not self.tools:
            return False
        _, calls = self.parse_stream_chunk(continuation_text)
        saw_tool_calls = bool(calls)
        if self._stream_input is not None:
            # Live input and replay retain exactly the same unfinished block
            # and detector state. Empty-input draining would change that state.
            return saw_tool_calls
        max_iters = 100
        for _ in range(max_iters):
            normal_text, calls = self.parse_stream_chunk("")
            saw_tool_calls = saw_tool_calls or bool(calls)
            if not calls and not normal_text:
                break
        else:
            logger.warning(
                "replay_continuation: hit max_iters=%d; "
                "parser state may be incomplete",
                max_iters,
            )
        return saw_tool_calls

    def get_legacy_structural_tag(
        self, at_least_one: bool = False
    ) -> StructuralTagResponseFormat:
        """
        Generate a structural tag response format for all available tools.

        This creates the necessary structural tags that guide the model's output format.

        Args:
            at_least_one: If True, the grammar forces at least one tool call
                (no free text allowed). Used for required/named tool_choice.

        Raises:
            ValueError: If tools have conflicting $defs schemas.
        """
        # Validate $defs consistency before building structural tags
        _get_tool_schema_defs(self.tools)

        tool_structures: List[StructuresResponseFormat] = list()
        tool_trigger_set: Set[str] = set()

        get_structure_info = self.detector.structure_info()
        for tool in self.tools:
            function = tool.function
            name = function.name
            assert name is not None
            info = get_structure_info(name)

            # accept all if not strict, otherwise only accept the schema
            is_strict = (
                function.strict or self.tool_strict_level >= ToolStrictLevel.PARAMETER
            )
            schema = function.parameters if is_strict else {}

            tool_structures.append(
                StructuresResponseFormat(
                    begin=info.begin,
                    schema=schema or {},  # type: ignore
                    end=info.end,
                )
            )
            tool_trigger_set.add(info.trigger)

        # TODO(dark): move this into new structural tag format
        # This requires all grammar backend support the new format
        return LegacyStructuralTagResponseFormat(
            type="structural_tag",
            structures=tool_structures,
            triggers=list(tool_trigger_set),
            at_least_one=at_least_one,
        )

    def _get_kimik3_structure_constraint(self) -> Optional[ToolCallConstraint]:
        """Route Kimi K3 to its model-native structural tag.

        ``auto`` with no strictness requested still gets the auto hook, which
        constrains a call once the model opens the tools channel without forcing
        one. Requesting strictness -- per-tool ``strict`` or
        ``SGLANG_TOOL_STRICT_LEVEL`` -- switches to the full tag, and PARAMETER
        level promotes every tool to strict.
        """
        wants_call = self.tool_choice == "required" or isinstance(
            self.tool_choice, ToolChoice
        )
        wants_strict = self.tool_strict_level >= ToolStrictLevel.FUNCTION or any(
            tool.function.strict for tool in self.tools
        )

        if not wants_call and not wants_strict:
            structural_tag = self.detector.get_auto_tool_call_structural_tag(
                tools=self.tools,
                thinking_mode=self.thinking_mode,
                parallel_tool_calls=self.parallel_tool_calls,
            )
            if structural_tag is None:
                return None
            return ("structural_tag", structural_tag)

        tools = self.tools
        if self.tool_strict_level >= ToolStrictLevel.PARAMETER:
            tools = promote_tools_to_strict(tools)
        return (
            "structural_tag",
            self.detector.get_structural_tag(
                tools=tools,
                tool_choice=self.tool_choice,
                thinking_mode=self.thinking_mode,
                parallel_tool_calls=self.parallel_tool_calls,
            ),
        )

    def get_structure_constraint(self) -> Optional[ToolCallConstraint]:
        """
        Returns the appropriate structure constraint for tool calls based on the tool_choice.
        The constraint is used to guide the model's output format.

        Args:
            tool_choice: The tool choice setting from the request

        Returns:
            A tuple of (constraint_type, constraint_value) to be added to sampling parameters,
            or None if no constraint applies.
        """
        # TODO(liyaolang): Should move to the newest community implementation later.
        if isinstance(self.detector, KimiK3Detector):
            return self._get_kimik3_structure_constraint()

        # NOTE: structural_tag only supports JSON-compatible content between the begin and end.
        # It cannot parse or validate function call Pythonic or XML-ish syntax.
        if (
            self.detector.supports_structural_tag()
            and self.tool_choice == "auto"
            and self.tool_strict_level >= ToolStrictLevel.FUNCTION
        ):
            # Use new structural tag to support parallel tool callings
            # TODO(liyaolang): Other models should also support in the future
            if isinstance(self.detector, KimiK2Detector):
                return self.detector.get_structure_constraint()
            tag = self._get_legacy_structural_tag()
            return ("structural_tag", tag)
        elif self.tool_choice == "required" or isinstance(self.tool_choice, ToolChoice):
            json_schema = get_json_schema_constraint(
                self.tools,
                self.tool_choice,
                parallel_tool_calls=self.parallel_tool_calls,
            )
            return ("json_schema", json_schema)
