from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import re
import time
import uuid
from collections import Counter, defaultdict, deque
from enum import Enum
from http import HTTPStatus
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)


class ThinkingMode(str, Enum):
    """Mode for message encoding - chat vs thinking/reasoning."""

    CHAT = "chat"
    THINKING = "thinking"


import msgspec
import orjson
from fastapi import Request
from fastapi.responses import ORJSONResponse, StreamingResponse
from jsonschema import Draft202012Validator, SchemaError

from sglang.srt.entrypoints.openai import encoding_dsv4, encoding_dsv32
from sglang.srt.entrypoints.openai.encoding_kimi_k3 import (
    use_official_kimi_k3_encoding,
)
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionMessageGenericParam,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatCompletionTokenLogprob,
    ChatMessage,
    ChoiceLogprobs,
    DeltaMessage,
    ErrorResponse,
    FunctionResponse,
    MessageProcessingResult,
    PromptTokensDetails,
    ResponseParserProtocol,
    SglExt,
    Tool,
    ToolCall,
    ToolCallProcessingResult,
    ToolChoice,
    TopLogprob,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.entrypoints.openai.sse_utils import build_sse_content
from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor
from sglang.srt.entrypoints.openai.utils import (
    cached_tokens_details_from_dict,
    process_cached_tokens_details_from_ret,
    process_hidden_states_from_ret,
    process_routed_experts_from_ret,
    should_include_usage,
)
from sglang.srt.environ import envs
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.json_array_parser import JsonArrayParser
from sglang.srt.function_call.utils import (
    expand_json_schema_refs,
    get_json_schema_constraint,
    normalize_json_schema_ids,
    normalize_json_schema_types,
)
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.parser.conversation import generate_chat_conv
from sglang.srt.parser.jinja_template_utils import process_content_for_template_format
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.utils import ImageData

_SSE_DATA_B = b"data: "
_SSE_NL_B = b"\n\n"


class _StreamDelta(msgspec.Struct, omit_defaults=True):
    # OpenAI Python SDK's ChoiceDelta does not declare reasoning_content; it is
    # surfaced via pydantic `extra`. With omit_defaults=True, defaulting to
    # None would drop the key entirely from the SSE payload, making
    # `data.reasoning_content` raise AttributeError on the client. Keep it
    # required (no default) so it is always serialized as null or a string.
    # `content` is also kept required for the same reason: downstream consumers
    # access `delta.content` directly and would break if the key is omitted
    # (e.g. on the final stop chunk where only reasoning_content is set).
    # Both are required (no default) so omit_defaults can never drop them; the
    # callers coerce None -> "" so they always serialize as a string.
    content: Optional[str]
    reasoning_content: Optional[str]
    role: Optional[str] = None
    tokens: Optional[int] = None
    # Moonshot/Kimi streaming spec; omitted unless
    # stream_options.include_internal_content is requested.
    internal_content: Optional[dict] = None


class _StreamChoice(msgspec.Struct):
    index: int
    delta: _StreamDelta
    logprobs: Optional[dict] = None
    finish_reason: Optional[str] = None
    matched_stop: Union[None, int, str] = None
    output_token_ids: Union[List[int], msgspec.UnsetType] = msgspec.UNSET


class _StreamChunk(msgspec.Struct, omit_defaults=True):
    id: str
    object: str
    created: int
    model: str
    choices: List[_StreamChoice]
    usage: Optional[dict] = None


_stream_encoder = msgspec.json.Encoder()

# Kimi-K3 tool_call ids: `<name>_<global_index>`.
_KIMI_K3_TOOL_CALL_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*_(\d+)$")


def _fast_sse_content(
    chunk_id: str,
    created: int,
    model: str,
    index: int,
    role: Optional[str] = None,
    content: Optional[str] = None,
    reasoning_content: Optional[str] = None,
    finish_reason: Optional[str] = None,
    logprobs: Optional[dict] = None,
    matched_stop: Union[None, int, str] = None,
    usage: Optional[dict] = None,
    tokens: Optional[int] = None,
    output_token_ids: Optional[List[int]] = None,
    internal_content: Optional[dict] = None,
) -> str:
    delta = _StreamDelta(
        role=role,
        content=content if content is not None else "",
        reasoning_content=reasoning_content if reasoning_content is not None else "",
        tokens=tokens,
        internal_content=internal_content,
    )
    choice = _StreamChoice(
        index=index,
        delta=delta,
        logprobs=logprobs,
        finish_reason=finish_reason,
        matched_stop=matched_stop,
        output_token_ids=(
            output_token_ids if output_token_ids is not None else msgspec.UNSET
        ),
    )
    chunk = _StreamChunk(
        id=chunk_id,
        object="chat.completion.chunk",
        created=created,
        model=model,
        choices=[choice],
        usage=usage,
    )
    return (_SSE_DATA_B + _stream_encoder.encode(chunk) + _SSE_NL_B).decode()


def _flush_pending_token_ids(
    pending: Dict[int, List[int]],
    index: int,
    *,
    emit_output_token_ids: bool,
    emit_internal_content: bool,
) -> Tuple[Optional[List[int]], Optional[Dict[str, Any]]]:
    """Consume the token ids buffered since the last emitted SSE chunk.

    The same buffer feeds two wire fields, so it must be popped exactly once
    per emitted chunk: `choices[].output_token_ids` (this fork's function-call
    continuation field) and `delta.internal_content` (Moonshot streaming spec).

    Returns `(output_token_ids, internal_content)`, each None when not requested
    or when nothing was buffered.
    """
    ids = pending.pop(index, None) or None
    if ids is None:
        return None, None
    return (
        ids if emit_output_token_ids else None,
        {"token_ids": ids} if emit_internal_content else None,
    )


def _flush_pending_logprobs(
    pending: Dict[int, List[Dict[str, Any]]], index: int
) -> Optional[Dict[str, Any]]:
    """Consume the logprob entries buffered since the last emitted SSE chunk.

    Logprobs arrive with the generator step, but the chunk carrying that step's
    text may be the reasoning, tool-call or finish frame, so they are buffered
    and attached to whichever chunk is emitted next instead of being dropped.
    """
    entries = pending.pop(index, None)
    return {"content": entries} if entries else None


def _coalesce_tool_stream_chunks(chunks: List[str]) -> ChatCompletionStreamResponse:
    """Commit all projections of one token batch in a single resumable SSE.

    A token prefix cannot identify a position between its reasoning, name and
    argument events. Keep them together so replay suppresses exactly the
    semantic output already received by the client.
    """
    merged = ChatCompletionStreamResponse.model_validate_json(chunks[0][6:])
    choice = merged.choices[0]
    calls = list(choice.delta.tool_calls or [])
    for raw in chunks[1:]:
        chunk = ChatCompletionStreamResponse.model_validate_json(raw[6:])
        other = chunk.choices[0]
        assert choice.index == other.index
        for field in ("content", "reasoning_content"):
            value = getattr(other.delta, field)
            if value:
                setattr(
                    choice.delta, field, (getattr(choice.delta, field) or "") + value
                )
        calls.extend(other.delta.tool_calls or [])
        if other.delta.tokens is not None:
            choice.delta.tokens = (choice.delta.tokens or 0) + other.delta.tokens
        if other.output_token_ids:
            choice.output_token_ids = (
                choice.output_token_ids or []
            ) + other.output_token_ids
        if other.delta.internal_content:
            current = choice.delta.internal_content or {"token_ids": []}
            current["token_ids"].extend(other.delta.internal_content["token_ids"])
            choice.delta.internal_content = current
        if other.logprobs:
            if choice.logprobs is None:
                choice.logprobs = other.logprobs
            else:
                choice.logprobs.content = (choice.logprobs.content or []) + (
                    other.logprobs.content or []
                )
        if chunk.usage is not None:
            merged.usage = chunk.usage

    by_index = {}
    for call in calls:
        if call.index not in by_index:
            by_index[call.index] = call
            continue
        previous = by_index[call.index]
        if call.function.name:
            previous.function.name = (previous.function.name or "") + call.function.name
        previous.function.arguments += call.function.arguments
        if call.id:
            previous.id = call.id
    choice.delta.tool_calls = list(by_index.values()) or None
    return merged


if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager
    from sglang.srt.parser.template_manager import TemplateManager

logger = logging.getLogger(__name__)


def normalize_tool_content(role: str, content):
    """Normalize tool message content from OpenAI array format to plain string.

    OpenAI clients may send tool content as a list of content parts
    (e.g. [{"type":"text","text":"..."}]) but most chat templates expect
    a plain string for tool messages. Only flatten when ALL items are
    pure OpenAI text parts; preserve lists containing non-text-type items
    that some templates intentionally iterate over.
    """
    if role != "tool" or not isinstance(content, list):
        return content
    parts = content
    is_openai_text_parts = all(
        (isinstance(p, dict) and p.get("type") == "text") or isinstance(p, str)
        for p in parts
    )
    if is_openai_text_parts:
        text_parts = [p.get("text", "") if isinstance(p, dict) else p for p in parts]
        return " ".join(text_parts)
    return content


def parse_tool_call_arguments(arguments: str) -> Dict[str, Any]:
    """Parse OpenAI tool call arguments for chat templates."""
    try:
        parsed_arguments = orjson.loads(arguments)
    except orjson.JSONDecodeError as exc:
        raise ValueError(
            "Assistant tool call function.arguments must be valid JSON."
        ) from exc

    if not isinstance(parsed_arguments, dict):
        raise ValueError(
            "Assistant tool call function.arguments must be a JSON object."
        )

    return parsed_arguments


def normalize_assistant_tool_call_arguments(message: Dict[str, Any]) -> None:
    """Normalize assistant history tool call arguments in-place."""
    if message.get("role") != "assistant" or not isinstance(
        message.get("tool_calls"), list
    ):
        return

    for item in message["tool_calls"]:
        function = item.get("function") if isinstance(item, dict) else None
        if not isinstance(function, dict):
            continue
        if "arguments" in function and isinstance(function["arguments"], str):
            function["arguments"] = parse_tool_call_arguments(function["arguments"])


def _extract_max_dynamic_patch(request: ChatCompletionRequest):
    img_vals = []
    vid_vals = []
    for msg in request.messages or []:
        content = getattr(msg, "content", None)
        if not isinstance(content, list):
            continue
        for part in content:
            # pydantic object or dict type
            if getattr(part, "type", None) == "image_url":
                iu = getattr(part, "image_url", None)
                mdp = getattr(iu, "max_dynamic_patch", None) if iu else None
                if mdp is not None:
                    img_vals.append(int(mdp))
            elif getattr(part, "type", None) == "video_url":
                vu = getattr(part, "video_url", None)
                mdp = getattr(vu, "max_dynamic_patch", None) if vu else None
                if mdp is not None:
                    vid_vals.append(int(mdp))

    # TODO(yuan-luo): per-item max_dynamic_patch for both image and video
    img_max_dynamic_patch = min(img_vals) if img_vals else None
    vid_max_dynamic_patch = min(vid_vals) if vid_vals else None
    return img_max_dynamic_patch, vid_max_dynamic_patch


def _compute_image_task_id(model_id: str, media_url: str) -> str:
    """md5(model_id + media_url)."""
    return hashlib.md5((model_id + media_url).encode("utf-8")).hexdigest()


def _compute_video_task_id(model_id: str, media_url: str, sample_fps: float) -> str:
    """md5(model_id + media_url + fps). sample_fps default: 2.0"""
    fps_str = f"{sample_fps:.2f}"
    return hashlib.md5((model_id + media_url + fps_str).encode("utf-8")).hexdigest()


def _extract_ordered_task_ids(
    request: ChatCompletionRequest, model_id: str
) -> Optional[List[str]]:
    """Compute the expected task_id for each image/video media item in content order."""
    sample_fps = request.sample_fps
    task_ids: List[str] = []
    for msg in request.messages:
        content = msg.content
        if not isinstance(content, list):
            continue
        for part in content:
            if part.type == "image_url":
                url = part.image_url.url
                if url:
                    task_ids.append(_compute_image_task_id(model_id, url))
            elif part.type == "video_url":
                url = part.video_url.url
                if url:
                    if sample_fps is None:
                        return None
                    task_ids.append(_compute_video_task_id(model_id, url, sample_fps))
    return task_ids


_MEDIA_CONTENT_PART_TYPES = frozenset(
    {"image_url", "input_image", "video_url", "audio_url"}
)


def _contains_media_content(request: ChatCompletionRequest) -> bool:
    """Whether the messages carry any image/video/audio part."""
    for msg in request.messages or []:
        content = getattr(msg, "content", None)
        if not isinstance(content, list):
            continue
        for part in content:
            part_type = (
                part.get("type")
                if isinstance(part, dict)
                else getattr(part, "type", None)
            )
            if part_type in _MEDIA_CONTENT_PART_TYPES:
                return True
    return False


def _reorder_task_results_by_content(
    request: ChatCompletionRequest,
    task_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Reorder task_results to align with the media order in request content."""
    model_ids = {
        task_result.get("model_id")
        or (task_result.get("encoder_addr") or {}).get("model_id")
        for task_result in task_results
    }
    model_ids.discard(None)
    model_ids.discard("")
    if len(model_ids) != 1:
        logger.warning(
            "Skip task_results reorder: expected one model_id, got %s", model_ids
        )
        return task_results

    model_id = next(iter(model_ids))
    ordered_task_ids = _extract_ordered_task_ids(request, model_id)
    if ordered_task_ids is None:
        logger.warning("Skip task_results reorder: video sample_fps is not resolved")
        return task_results

    result_task_ids = [task_result.get("task_id") for task_result in task_results]
    if Counter(ordered_task_ids) != Counter(result_task_ids):
        logger.warning("Skip task_results reorder: task_id multiset mismatch")
        return task_results

    task_id_to_positions = defaultdict(deque)
    for position, task_id in enumerate(ordered_task_ids):
        task_id_to_positions[task_id].append(position)

    ranked_results = [
        (task_id_to_positions[task_result["task_id"]].popleft(), task_result)
        for task_result in task_results
    ]
    ranked_results.sort(key=lambda item: item[0])
    return [task_result for _, task_result in ranked_results]


class OpenAIServingChat(OpenAIServingBase):
    """Handler for /v1/chat/completions requests"""

    _default_sampling_params_logged = False
    _KIMI_K3_GENERATION_STUB_TOKENS = 3

    def __init__(
        self,
        tokenizer_manager: TokenizerManager,
        template_manager: TemplateManager,
    ):
        super().__init__(tokenizer_manager)
        self.template_manager = template_manager
        self.tool_call_parser = self.tokenizer_manager.server_args.tool_call_parser
        self.reasoning_parser = self.tokenizer_manager.server_args.reasoning_parser
        self.default_chat_template_kwargs = (
            self.tokenizer_manager.server_args.default_chat_template_kwargs or {}
        )
        # Parse reasoning content in OpenAI serving when the upstream response
        # still contains raw think tags.
        self.mocked_disable_reasoning_parser = False
        self._reasoning_detector = None
        if self.reasoning_parser:
            try:
                rp = ReasoningParser(
                    model_type=self.reasoning_parser,
                    stream_reasoning=True,
                    tokenizer=self.tokenizer_manager.tokenizer,
                )
                self._reasoning_detector = rp.detector
            except ValueError as e:
                logger.warning(
                    "Failed to initialize reasoning detector for parser '%s': %s",
                    self.reasoning_parser,
                    e,
                )

        # Get default sampling parameters from model's generation config
        self.default_sampling_params = (
            self.tokenizer_manager.model_config.get_default_sampling_params()
        )
        if (
            self.default_sampling_params
            and not OpenAIServingChat._default_sampling_params_logged
        ):
            logger.info(
                f"Using default chat sampling params from model generation config: {self.default_sampling_params}",
            )
            OpenAIServingChat._default_sampling_params_logged = True

        # Check if the model is a GPT-OSS model
        self.is_gpt_oss = (
            hasattr(self.tokenizer_manager.model_config, "hf_config")
            and hasattr(self.tokenizer_manager.model_config.hf_config, "model_type")
            and self.tokenizer_manager.model_config.hf_config.model_type == "gpt_oss"
        )
        self.is_gemma4 = (
            hasattr(self.tokenizer_manager.model_config, "hf_config")
            and hasattr(self.tokenizer_manager.model_config.hf_config, "model_type")
            and self.tokenizer_manager.model_config.hf_config.model_type
            in ("gemma4", "gemma4_unified")
        )

        # Which Python-based chat encoder (if any) bypasses apply_chat_template.
        # Values: "dsv32", "dsv4", or custom values set by subclass. None for default.
        self.chat_encoding_spec = self._resolve_chat_encoding_spec()
        if (
            self.chat_encoding_spec == "kimi_k3"
            and self.tokenizer_manager.tokenizer is not None
        ):
            use_official_kimi_k3_encoding(self.tokenizer_manager.tokenizer)

        # Resolve the env-configured Inkling effort default once: the env var is
        # frozen for the server's lifetime, and a misconfigured value should
        # fail at boot, not 400 every request.
        self._inkling_default_reasoning_effort: Optional[float] = (
            self._get_inkling_default_reasoning_effort()
            if self.chat_encoding_spec == "inkling"
            else None
        )

        # Per-request response parser for custom decoding (set by _encode_messages)
        self._response_parser: Optional[ResponseParserProtocol] = None

        # Probe whether ``encode("")`` returns specials. If it does, we must
        # keep ``add_special_tokens=False`` at the chat-template encode site
        # to avoid double BOS; otherwise the kwarg is a no-op and dropping it
        # lets slow tokenizers (e.g. Kimi's TikTokenTokenizer) stay on the
        # fast internal path.
        try:
            self._tokenizer_auto_adds_specials = (
                len(self.tokenizer_manager.tokenizer.encode("")) > 0
            )
        except Exception:
            self._tokenizer_auto_adds_specials = True
        self.tool_call_empty_return_interval = float(
            os.getenv("SGLANG_TOOL_CALL_EMPTY_RETURN_INTERVAL", "5")
        )


    def _handle_last_assistant_message(
        self,
        messages: List[Dict[str, Any]],
        request: ChatCompletionRequest,
    ) -> tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Handle continue_final_message feature: separate final assistant message.

        If continue_final_message is enabled and the last message is from assistant,
        extract its content and remove it from the message list.
        If continue_final_message is False and the last message is from assistant,
        convert it to a user message to ensure the last message is always from user.

        Only processes text-based content (strings), ignoring multimodal content (lists).

        Args:
            messages: List of message dictionaries
            request: ChatCompletionRequest with continue_final_message flag

        Returns:
            Tuple of (processed_messages, assistant_prefix)
            - processed_messages: Messages with last assistant message handled appropriately
            - assistant_prefix: Content of the last assistant message (string only), or None
        """
        assistant_prefix = None
        if messages and messages[-1].get("role") == "assistant":
            last_content = messages[-1].get("content")
            # Only process string content, ignore multimodal content (lists)
            if isinstance(last_content, str):
                if request.continue_final_message:
                    # Extract content and remove the assistant message
                    assistant_prefix = last_content
                    messages = messages[:-1]
                else:
                    # Convert the last assistant message to user message
                    messages[-1] = {"role": "user", "content": last_content}
        return messages, assistant_prefix

    def _append_assistant_prefix_to_prompt_ids(
        self, prompt_ids: List[int], assistant_prefix: str
    ) -> List[int]:
        """
        Append assistant prefix to prompt_ids.

        Args:
            prompt_ids: Current prompt token IDs
            assistant_prefix: Assistant message content to append

        Returns:
            Updated prompt_ids with assistant prefix appended
        """
        encoded = self.tokenizer_manager.tokenizer.encode(assistant_prefix)
        if encoded and encoded[0] == self.tokenizer_manager.tokenizer.bos_token_id:
            encoded = encoded[1:]
        return prompt_ids + encoded

    def _resolve_chat_encoding_spec(self) -> Optional[str]:
        """Determine which chat encoding spec to use.

        Override in subclass to add custom encoding specs.
        """
        from sglang.srt.entrypoints.openai.chat_encoding import (
            resolve_chat_encoding_spec,
        )

        return resolve_chat_encoding_spec(
            hf_config=self.tokenizer_manager.model_config.hf_config,
            tokenizer=self.tokenizer_manager.tokenizer,
            tool_call_parser=self.tool_call_parser,
        )

    def _flatten_kimi_k3_content(
        self,
        msg: Dict[str, Any],
        image_data: list,
        video_data: list,
        audio_data: list,
        media_kinds: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Flatten Kimi-K3 content while collecting ordered media metadata."""
        if media_kinds is None:
            media_kinds = []
        parts = []
        for chunk in msg["content"]:
            if not isinstance(chunk, dict):
                continue
            chunk_type = chunk.get("type")
            if chunk_type in ("text", "input_text"):
                text = chunk["text"]
                if "<|kimi_image_placeholder|>" in text:
                    raise ValueError(
                        "<|kimi_image_placeholder|> is reserved for Kimi-K3 image input"
                    )
                parts.append({"type": "text", "text": text})
            elif chunk_type in ("image_url", "input_image"):
                image_obj = chunk.get("image_url") or {}
                if isinstance(image_obj, str):
                    image_obj = {"url": image_obj, "detail": chunk.get("detail")}
                image_data.append(
                    ImageData(
                        url=image_obj["url"],
                        detail=image_obj.get("detail") or "auto",
                        max_dynamic_patch=image_obj.get("max_dynamic_patch"),
                    )
                )
                parts.append({"type": "image_url", "image_url": image_obj})
                media_kinds.append("image")
            elif chunk_type == "video_url":
                video_obj = chunk["video_url"]
                video_data.append(video_obj["url"])
                # Typed as an image part on purpose: K3's chat encoder consumes
                # one ordered media-prompt list, its only allow_special channel.
                parts.append({"type": "image_url", "image_url": video_obj})
                media_kinds.append("video")
            elif chunk_type == "audio_url":
                audio_data.append(chunk["audio_url"]["url"])
        new_msg = {k: v for k, v in msg.items() if v is not None and k != "content"}
        new_msg["content"] = parts
        return new_msg

    @staticmethod
    def _build_kimi_k3_media_prompts(
        media_kinds: List[str],
        task_results: Optional[List[Dict[str, Any]]],
    ) -> List[str]:
        """One rendered media prompt per media item, in request content order.

        An image gets a bare placeholder, which the multimodal processor later
        frames with the uploaded resolution. A video gets the per-chunk prompts
        the encode server already rendered, since only it knows the chunk count.
        Substituting here rather than after tokenization keeps K3's structural
        markers on their special-token ids.
        """
        num_videos = media_kinds.count("video")
        if num_videos == 0:
            return ["<|media_pad|>"] * len(media_kinds)

        video_prompts = []
        for task_result in task_results or []:
            encoder_addr = task_result.get("encoder_addr") or {}
            chunk_timestamps = encoder_addr.get("chunk_timestamps")
            if chunk_timestamps:
                video_prompts.append("".join(chunk_timestamps))

        if len(video_prompts) != num_videos:
            raise ValueError(
                "Kimi-K3 video input requires encoder chunk prompts for every "
                f"video: expected {num_videos}, got {len(video_prompts)}"
            )

        media_prompts = []
        video_index = 0
        for kind in media_kinds:
            if kind == "video":
                media_prompts.append(video_prompts[video_index])
                video_index += 1
            else:
                media_prompts.append("<|media_pad|>")
        return media_prompts

    def _request_id_prefix(self) -> str:
        return "chatcmpl-"

    def _encode_messages(
        self,
        messages: List[Dict[str, Any]],
        request: ChatCompletionRequest,
        thinking_mode: ThinkingMode,
        tools: Optional[List[Dict]] = None,
    ) -> Optional[List[int]]:
        """Encode messages for custom chat_encoding_spec values.

        Returns prompt_ids if handled, None to use default encoding.
        """
        if self.chat_encoding_spec == "inkling":
            # Inkling: render messages -> input_ids with framing tokens + ONE placeholder per
            # media (encoding/expansion happens later in InklingMultimodalProcessor). The
            # server's tokenizer is the base tiktoken backend; wrap it so encode_special
            # supplies the framing-token overlay.
            from sglang.srt.parser.inkling_renderer import render_inkling_messages
            from sglang.srt.parser.inkling_tokenizer import (
                CONTENT_TEXT,
                MESSAGE_MODEL,
                InklingTokenizer,
            )

            inkling_tokenizer = InklingTokenizer(
                tokenizer=self.tokenizer_manager.tokenizer
            )
            reasoning_effort = self._parse_inkling_reasoning_effort(
                request.reasoning_effort
            )
            if reasoning_effort is None:
                reasoning_effort = self._inkling_default_reasoning_effort
            assistant_prefix = self._pop_inkling_assistant_prefix(messages, request)
            prompt_ids = render_inkling_messages(
                messages,
                inkling_tokenizer,
                add_generation_prompt=False,
                tools=tools,
                reasoning_effort=reasoning_effort,
            )
            if assistant_prefix is not None:
                # Continue the final assistant message inside an OPEN model text
                # block: header + payload, no <|end_message|> and no
                # <|content_model_end_sampling|>, so the model resumes the turn.
                prompt_ids += [
                    inkling_tokenizer.encode_special(MESSAGE_MODEL),
                    inkling_tokenizer.encode_special(CONTENT_TEXT),
                    *inkling_tokenizer.encode_text(assistant_prefix),
                ]
            return prompt_ids
        return None

    @staticmethod
    def _pop_inkling_assistant_prefix(
        messages: List[Dict[str, Any]],
        request: ChatCompletionRequest,
    ) -> Optional[str]:
        """Extract the trailing assistant text for ``continue_final_message``.

        Only a plain-string assistant message with no tool calls and no
        reasoning content can be continued; anything else renders as a closed
        historical turn. Mutates ``messages`` in place (callers pass a copy).
        """
        if not request.continue_final_message or not messages:
            return None
        last = messages[-1]
        if (
            last.get("role") != "assistant"
            or not isinstance(last.get("content"), str)
            or last.get("tool_calls")
            or last.get("reasoning_content")
        ):
            return None
        messages.pop()
        return last["content"]

    @staticmethod
    def _parse_inkling_reasoning_effort(
        value: Optional[Union[str, float]],
    ) -> Optional[float]:
        """Convert an OpenAI-style reasoning_effort to an Inkling float."""
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("Inkling reasoning_effort must not be a boolean")
        if isinstance(value, (int, float)):
            parsed = float(value)
            if not math.isfinite(parsed) or not 0.0 <= parsed <= 0.99:
                raise ValueError("Inkling reasoning_effort must be in [0.0, 0.99]")
            return parsed
        _EFFORT_MAP = {
            "none": 0.0,
            "minimal": 0.1,
            "low": 0.2,
            "medium": 0.7,
            "high": 0.9,
            "xhigh": 0.99,
            "max": 0.99,
        }
        if value in _EFFORT_MAP:
            return _EFFORT_MAP[value]
        try:
            parsed = float(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid Inkling reasoning_effort: {value!r}") from exc
        if not math.isfinite(parsed) or not 0.0 <= parsed <= 0.99:
            raise ValueError("Inkling reasoning_effort must be in [0.0, 0.99]")
        return parsed

    @staticmethod
    def _get_inkling_default_reasoning_effort() -> float:
        """Read the default Inkling reasoning effort from the environment."""
        from sglang.srt.environ import envs

        val = envs.SGLANG_INKLING_DEFAULT_REASONING_EFFORT.get()
        if not val:
            return 0.9
        try:
            parsed = float(val)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "SGLANG_INKLING_DEFAULT_REASONING_EFFORT must be numeric"
            ) from exc
        if not math.isfinite(parsed) or not 0.0 <= parsed <= 0.99:
            raise ValueError(
                "SGLANG_INKLING_DEFAULT_REASONING_EFFORT must be in [0.0, 0.99]"
            )
        return parsed

    def _decode_response(self, ret_item: Dict[str, Any]) -> Union[str, ErrorResponse]:
        """Extract text from response."""
        return ret_item["text"]

    def _get_parsed_response_fields(
        self,
        reasoning_text: Optional[str],
        tool_calls: Optional[List[Dict]],
    ) -> tuple[Optional[str], Optional[List[Dict]]]:
        """Post-process reasoning and tool_calls before building response."""
        return reasoning_text, tool_calls

    def _continuous_usage_cached_details(
        self, content: Dict[str, Any]
    ) -> Optional[PromptTokensDetails]:
        if not self.tokenizer_manager.server_args.enable_cache_report:
            return None
        return PromptTokensDetails(
            cached_tokens=content["meta_info"].get("cached_tokens", 0)
        )

    def _reported_prompt_tokens(self, meta_info: Dict[str, Any]) -> int:
        prompt_tokens = meta_info.get("prompt_tokens", 0)
        if self.chat_encoding_spec == "kimi_k3":
            # K3's three-token assistant generation stub is model input, but the
            # reference API excludes it from billed/reported prompt tokens.
            prompt_tokens = max(0, prompt_tokens - self._KIMI_K3_GENERATION_STUB_TOKENS)
        return prompt_tokens

    async def _generate_stream_content(
        self,
        content: Dict[str, Any],
        index: int,
        request: ChatCompletionRequest,
        stream_offsets: Dict[int, int],
        reasoning_parser_dict: Dict,
        parser_dict: Dict,
        has_tool_calls: Dict[int, bool],
        choice_logprobs: Optional[Dict],
        finish_reason_type: Optional[str],
        continuous_usage_stats: bool,
        prompt_tokens: Dict[int, int],
        reasoning_tokens: Dict[int, int],
        completion_tokens: Dict[int, int],
    ) -> AsyncGenerator[str, None]:
        """Generate SSE chunks for streaming content."""
        offset = stream_offsets.get(index, 0)
        if self.tokenizer_manager.server_args.incremental_streaming_output:
            delta = content["text"]
        else:
            delta = content["text"][offset:]
            stream_offsets[index] = len(content["text"])

        # Attach logprobs to the first chunk emitted this step (reasoning,
        # tool-call, or content) so they aren't dropped when a parser is active
        # nor duplicated across chunks; flush any leftover at the end.
        remaining_logprobs = choice_logprobs

        # Handle reasoning content
        if self.reasoning_parser and request.separate_reasoning:
            reasoning_text, delta = self._process_reasoning_stream(
                index,
                delta,
                reasoning_parser_dict,
                content,
                request,
                finish_reason_type,
            )
            if reasoning_text:
                usage = None
                if continuous_usage_stats:
                    usage = UsageProcessor.calculate_token_usage(
                        prompt_tokens=prompt_tokens.get(index, 0),
                        reasoning_tokens=reasoning_tokens.get(index, 0),
                        completion_tokens=completion_tokens.get(index, 0),
                        cached_tokens=self._continuous_usage_cached_details(content),
                    ).model_dump()

                yield build_sse_content(
                    chunk_id=content["meta_info"]["id"],
                    created=int(time.time()),
                    model=request.model,
                    index=index,
                    reasoning_content=reasoning_text,
                    logprobs=remaining_logprobs,
                    usage=usage,
                )
                remaining_logprobs = None

        # Handle tool calls
        if (
            request.tool_choice != "none"
            and self.tool_call_parser
            and self._collect_tools(request)
        ):
            async for chunk in self._process_tool_call_stream(
                index,
                delta,
                parser_dict,
                content,
                request,
                has_tool_calls,
                continuous_usage_stats,
                finish_reason_type=finish_reason_type,
            ):
                if chunk:
                    yield chunk

            # Send any remaining tool call arguments when generation finishes
            if finish_reason_type == "stop" and index in parser_dict:
                parser = parser_dict[index]
                remaining_chunk = self._check_for_unstreamed_tool_args(
                    parser, content, request, index
                )
                if remaining_chunk:
                    yield remaining_chunk

        else:
            # Regular content
            if delta:
                usage = None
                if continuous_usage_stats:
                    usage = UsageProcessor.calculate_token_usage(
                        prompt_tokens=prompt_tokens.get(index, 0),
                        reasoning_tokens=reasoning_tokens.get(index, 0),
                        completion_tokens=completion_tokens.get(index, 0),
                        cached_tokens=self._continuous_usage_cached_details(content),
                    ).model_dump()

                yield build_sse_content(
                    chunk_id=content["meta_info"]["id"],
                    created=int(time.time()),
                    model=request.model,
                    index=index,
                    content=delta,
                    logprobs=remaining_logprobs,
                    usage=usage,
                )
                remaining_logprobs = None

        # Flush logprobs still unattached this step — only when a parser is
        # active, since _process_tool_call_stream may consume the delta and emit
        # no content chunk. On the plain path an empty-delta step has no chunk
        # to attach to either way, and a standalone empty-delta logprobs chunk
        # is not a shape clients expect.
        if remaining_logprobs is not None and (
            self.reasoning_parser or self.tool_call_parser
        ):
            usage = None
            if continuous_usage_stats:
                usage = UsageProcessor.calculate_token_usage(
                    prompt_tokens=prompt_tokens.get(index, 0),
                    reasoning_tokens=reasoning_tokens.get(index, 0),
                    completion_tokens=completion_tokens.get(index, 0),
                    cached_tokens=self._continuous_usage_cached_details(content),
                ).model_dump()

            yield build_sse_content(
                chunk_id=content["meta_info"]["id"],
                created=int(time.time()),
                model=request.model,
                index=index,
                logprobs=remaining_logprobs,
                usage=usage,
            )

    def _validate_request(self, request: ChatCompletionRequest) -> Optional[str]:
        """Validate that the input is valid."""
        if not request.messages:
            return "Messages cannot be empty."

        dynamic_tool_error = self._validate_dynamic_tools(request)
        if dynamic_tool_error:
            return dynamic_tool_error

        all_tools = self._collect_tools(request) or []

        if (
            isinstance(request.tool_choice, str)
            and request.tool_choice.lower() == "required"
            and not all_tools
        ):
            return "Tools cannot be empty if tool choice is set to required."

        if request.tool_choice is not None and not isinstance(request.tool_choice, str):
            if not all_tools:
                return "Tools cannot be empty if tool choice is set to a specific tool."
            tool_name = request.tool_choice.function.name
            tool_exists = any(tool.function.name == tool_name for tool in all_tools)
            if not tool_exists:
                return f"Tool '{tool_name}' not found in tools list."

        # Validate tool definitions
        for i, tool in enumerate(all_tools):
            if tool.function.parameters is None:
                continue
            try:
                # Optionally inline local $ref pointers against the schema's
                # $defs/definitions so downstream prompt rendering and
                # constrained decoding get a self-contained tree.
                if envs.SGLANG_ENABLE_TOOL_SCHEMA_REF_EXPANSION.get():
                    tool.function.parameters = expand_json_schema_refs(
                        tool.function.parameters
                    )
                # Rewrite DB/ORM-style aliases (e.g. "varchar", "enum", "int")
                # to standard JSON Schema types before validation. RecursionError
                # guards against hand-crafted cyclic schemas so the request gets
                # a 400 instead of crashing into a 500.
                normalize_json_schema_types(tool.function.parameters)
                # Draft-07 `$id` fragments ("#user", "…#/definitions/name") are a
                # schema error under 2020-12, which would reject the whole request
                # before it ever reaches grammar compilation.
                if envs.SGLANG_ENABLE_TOOL_SCHEMA_ID_NORMALIZATION.get():
                    normalize_json_schema_ids(tool.function.parameters)
                Draft202012Validator.check_schema(tool.function.parameters)
            except SchemaError as e:
                return f"Tool {i} function has invalid 'parameters' schema: {str(e)}"
            except RecursionError:
                return (
                    f"Tool {i} function 'parameters' schema is too deeply nested "
                    "or contains a cycle."
                )

        max_output_tokens = request.max_completion_tokens or request.max_tokens
        server_context_length = self.tokenizer_manager.server_args.context_length
        if (
            max_output_tokens
            and server_context_length
            and max_output_tokens > server_context_length
        ) and not self.tokenizer_manager.server_args.allow_auto_truncate:
            return (
                f"max_completion_tokens is too large: {max_output_tokens}."
                f"This model supports at most {server_context_length} completion tokens."
            )

        if request.response_format and request.response_format.type == "json_schema":
            schema = getattr(request.response_format.json_schema, "schema_", None)
            if schema is None:
                return "schema_ is required for json_schema response format request."

        # Without task_results an EPD server would fall back to local encoding
        # and only fail much later as a KV-transfer timeout.
        if (
            not request.task_results
            and self.tokenizer_manager.server_args.epd_requires_task_results()
            and _contains_media_content(request)
        ):
            return (
                "Multi-modal request requires task_results parameter from IC: "
                "this server offloads media encoding to the encode service and "
                "cannot process image/video/audio content locally."
            )

        return None

    def _convert_to_internal_request(
        self,
        request: ChatCompletionRequest,
        raw_request: Request = None,
    ) -> tuple[GenerateReqInput, ChatCompletionRequest]:
        reasoning_effort = (
            request.chat_template_kwargs.pop("reasoning_effort", None)
            if request.chat_template_kwargs
            else None
        )
        if self.is_gpt_oss and reasoning_effort == "none":
            raise ValueError(
                f"Harmony does not support reasoning effort {reasoning_effort}"
            )

        if reasoning_effort is not None:
            request.reasoning_effort = reasoning_effort

        if request.stream:
            if request.return_prompt_token_ids:
                raise ValueError(
                    "return_prompt_token_ids is not supported with streaming. "
                    "Please set stream=false when using return_prompt_token_ids=true."
                )
            if request.return_token_ids:
                raise ValueError(
                    "return_token_ids is not supported with streaming on "
                    "/v1/chat/completions. Please set stream=false when using "
                    "return_token_ids=true."
                )
            if request.return_meta_info:
                raise ValueError(
                    "return_meta_info is not supported with streaming. "
                    "Please set stream=false when using return_meta_info=true."
                )

        is_multimodal = self.tokenizer_manager.model_config.is_multimodal

        # Reordered before rendering: K3 splices the encoder's video prompts into
        # the prompt, so it needs them in request content order.
        ordered_task_results = request.task_results
        if ordered_task_results:
            ordered_task_results = _reorder_task_results_by_content(
                request, ordered_task_results
            )

        # Process messages and apply chat template
        processed_messages = self._process_messages(
            request, is_multimodal, task_results=ordered_task_results
        )
        # Build sampling parameters
        sampling_params = request.to_sampling_params(
            stop=processed_messages.stop,
            model_generation_config=self.default_sampling_params,
            tool_call_constraint=processed_messages.tool_call_constraint,
        )

        # Handle single vs multiple requests
        if request.input_ids is not None:
            prompt_kwargs = {"input_ids": processed_messages.prompt_ids}
        elif is_multimodal and self.chat_encoding_spec == "kimi_k3":
            prompt_kwargs = {"input_ids": processed_messages.prompt_ids}
        elif is_multimodal:
            # Standard VLMs render a text prompt (with placeholder strings) for the MM
            # processor to tokenize. Inkling's custom encoder instead produces pre-rendered
            # input_ids with single placeholders; pass those through so the MM processor
            # expands them rather than re-tokenizing an empty prompt. Gated on the Inkling
            # encoding spec so every other model keeps the standard text path.
            if (
                self.chat_encoding_spec == "inkling"
                and isinstance(processed_messages.prompt_ids, list)
                and processed_messages.prompt_ids
            ):
                prompt_kwargs = {"input_ids": processed_messages.prompt_ids}
            else:
                prompt_kwargs = {"text": processed_messages.prompt}
        else:
            if isinstance(processed_messages.prompt_ids, str):
                prompt_kwargs = {"text": processed_messages.prompt_ids}
            else:
                prompt_kwargs = {"input_ids": processed_messages.prompt_ids}

        # Extract custom labels from raw request headers
        custom_labels = self.extract_custom_labels(raw_request)

        # Extract routed_dp_rank from header (has higher priority than body)
        effective_routed_dp_rank = self.extract_routed_dp_rank_from_header(
            raw_request, request.routed_dp_rank
        )

        # Resolve LoRA adapter from model parameter or explicit lora_path
        lora_path = self._resolve_lora_path(request.model, request.lora_path)
        img_max_dynamic_patch, vid_max_dynamic_patch = _extract_max_dynamic_patch(
            request
        )
        require_reasoning = self._get_reasoning_from_request(request)
        adapted_request = GenerateReqInput(
            **prompt_kwargs,
            image_data=processed_messages.image_data,
            video_data=processed_messages.video_data,
            audio_data=processed_messages.audio_data,
            sampling_params=sampling_params,
            return_logprob=request.logprobs,
            logprob_start_len=-1,
            top_logprobs_num=request.top_logprobs or 0,
            stream=request.stream,
            return_text_in_logprobs=True,
            modalities=processed_messages.modalities,
            lora_path=lora_path,
            bootstrap_host=request.bootstrap_host,
            bootstrap_port=request.bootstrap_port,
            bootstrap_room=request.bootstrap_room,
            task_results=ordered_task_results,
            routed_dp_rank=effective_routed_dp_rank,
            disagg_prefill_dp_rank=request.disagg_prefill_dp_rank,
            return_hidden_states=request.return_hidden_states,
            return_routed_experts=request.return_routed_experts,
            routed_experts_start_len=request.routed_experts_start_len,
            rid=request.rid,
            session_id=request.session_id,
            extra_key=self._compute_extra_key(request),
            require_reasoning=require_reasoning,
            priority=request.priority,
            routing_key=self.extract_routing_key(raw_request),
            custom_labels=custom_labels,
            custom_logit_processor=request.custom_logit_processor,
            images_config=getattr(request, "images_config", None),
            # AttentionStore parameters (populated below from kv_transfer_params)
            as_kv_cache_params=None,
            as_max_hit_host_ip=None,
            as_max_hit_token_len=None,
            image_max_dynamic_patch=img_max_dynamic_patch,
            video_max_dynamic_patch=vid_max_dynamic_patch,
            max_dynamic_patch=getattr(request, "max_dynamic_patch", None),
            use_audio_in_video=getattr(request, "use_audio_in_video", False),
            return_prompt_token_ids=request.return_prompt_token_ids
            or request.return_token_ids,
        )

        if request.continuation_token_ids:
            logger.info(
                "FC continuation request (rid=%s): %d continuation tokens",
                adapted_request.rid,
                len(request.continuation_token_ids),
            )

        # Convert kv_cache_params and kv_transfer_params to as_* flat fields
        if request.kv_transfer_params is not None:
            kv_cache_dict = (
                request.kv_cache_params.dict() if request.kv_cache_params else {}
            )
            adapted_request.as_kv_cache_params = {
                key: str(value) if value is not None else ""
                for key, value in kv_cache_dict.items()
            }
            adapted_request.as_max_hit_host_ip = (
                request.kv_transfer_params.max_hit_host_ip
            )
            adapted_request.as_max_hit_token_len = (
                request.kv_transfer_params.max_hit_token_len
            )

        return adapted_request, request

    def _collect_tools(self, request: ChatCompletionRequest) -> Optional[List[Tool]]:
        tools = list(request.tools) if request.tools else []
        for msg in request.messages:
            if (
                isinstance(msg, ChatCompletionMessageGenericParam)
                and msg.role in ("system", "developer")
                and msg.tools
            ):
                tools.extend(msg.tools)
        return tools or None

    # Kimi K3 dynamic-tool name rule: 1-256 chars, must start with a letter or
    # underscore and contain only letters, digits, underscores and hyphens.
    _K3_TOOL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,255}$")

    def _validate_dynamic_tools(self, request: ChatCompletionRequest) -> Optional[str]:
        """Validate message-level (dynamic) and global tool declarations for K3.

        Returns an error string (-> 400) or None when valid. Only enforced for
        the kimi_k3 chat encoding; other models keep their prior behavior.
        """
        if self.chat_encoding_spec != "kimi_k3":
            return None

        seen_names: set[str] = set()

        def _check_tool(tool: Tool) -> Optional[str]:
            # ``type`` defaults to "function" in the model, so an omitted or
            # unsupported type is caught via the raw fields_set / value.
            if "type" not in tool.model_fields_set:
                return "Tool definition is missing required field 'type'."
            if tool.type != "function":
                return f"Unsupported tool type {tool.type!r}; only 'function' is supported."
            name = tool.function.name
            if not self._K3_TOOL_NAME_RE.match(name):
                return (
                    f"Invalid tool name {name!r}: must be 1-256 characters, start "
                    "with a letter or underscore, and contain only letters, "
                    "digits, underscores and hyphens."
                )
            if name in seen_names:
                return f"Duplicate tool name {name!r} in tools list."
            seen_names.add(name)
            return None

        # Global (request-level) tools first, then dynamic tools per message.
        for tool in request.tools or []:
            err = _check_tool(tool)
            if err:
                return err

        for msg in request.messages:
            if not getattr(msg, "tools", None):
                continue
            if not (
                isinstance(msg, ChatCompletionMessageGenericParam)
                and msg.role in ("system", "developer")
            ):
                return "Dynamic tools may only be declared in a system or developer message."
            if isinstance(msg.content, str) and msg.content != "":
                return (
                    "A system or developer message declaring dynamic tools must "
                    "have empty content."
                )
            for tool in msg.tools:
                err = _check_tool(tool)
                if err:
                    return err

        return None

    def _process_messages(
        self,
        request: ChatCompletionRequest,
        is_multimodal: bool,
        task_results: Optional[List[Dict[str, Any]]] = None,
    ) -> MessageProcessingResult:
        """Process chat messages and apply chat template"""
        if self.default_chat_template_kwargs:
            ctk = dict(request.chat_template_kwargs or {})
            for k, v in self.default_chat_template_kwargs.items():
                ctk.setdefault(k, v)
            request.chat_template_kwargs = ctk
            effort = ctk.get("reasoning_effort")
            if effort is not None and request.reasoning_effort is None:
                request.reasoning_effort = effort

        # GptOss model needs to keep special tokens for harmony parsing
        if self.is_gpt_oss or self.is_gemma4:
            request.skip_special_tokens = False

        self._patch_reasoning_skip_special_tokens(request)

        thinking_mode = self._get_reasoning_from_request(request)
        # SGLang's ReasonerGrammarBackend owns the reasoning prefix
        # when --reasoning-parser is configured, so builtin xgrammar
        # tags must describe only the post-reasoning tool-call suffix.
        xgrammar_reasoning = thinking_mode and (
            self.tokenizer_manager.server_args.reasoning_parser is None
        )
        tool_call_constraint = None

        # Apply chat template and its stop strings
        tools = None
        required_parsed_natively = False
        all_tools = self._collect_tools(request) or []
        if all_tools and request.tool_choice != "none":
            request.skip_special_tokens = False
            # Use all_tools (global + message-level dynamic tools) so the
            # tool-call constraint covers tools declared inside system messages,
            # not just request-level ones.
            if not isinstance(request.tool_choice, str):
                tools = [
                    item.model_dump()
                    for item in all_tools
                    if item.function.name == request.tool_choice.function.name
                ] or None
            else:
                tools = [item.model_dump() for item in all_tools]
            if self.tool_call_parser:
                parser = FunctionCallParser(
                    tool_call_parser=self.tool_call_parser,
                    tools=all_tools,
                    tool_choice=request.tool_choice,
                    parallel_tool_calls=request.parallel_tool_calls,
                    thinking_mode=xgrammar_reasoning,
                    tokenizer=self.tokenizer_manager.tokenizer,
                )
                required_parsed_natively = parser.detector.parses_required_natively()
                tool_call_constraint = parser.get_structure_constraint()

            # Fallback: use generic JSON schema for required/named tool choice
            # only when no parser-specific constraint was set
            if (
                tool_call_constraint is None
                and not required_parsed_natively
                and (
                    request.tool_choice == "required"
                    or isinstance(request.tool_choice, ToolChoice)
                )
            ):
                json_schema = get_json_schema_constraint(
                    all_tools,
                    request.tool_choice,
                    parallel_tool_calls=request.parallel_tool_calls,
                )
                tool_call_constraint = ("json_schema", json_schema)

        # When input_ids are provided, skip template tokenization entirely;
        # only stop tokens and tool_call_constraint are needed.
        if request.input_ids is not None:
            result = MessageProcessingResult(
                prompt="",
                prompt_ids=request.input_ids,
                image_data=None,
                audio_data=None,
                video_data=None,
                modalities=[],
                stop=request.stop or [],
            )
        elif self.template_manager.chat_template_name is None:
            result = self._apply_jinja_template(
                request, tools, is_multimodal, task_results=task_results
            )
        else:
            result = self._apply_conversation_template(request, is_multimodal)

        result.tool_call_constraint = tool_call_constraint
        return result

    def _apply_jinja_template(
        self,
        request: ChatCompletionRequest,
        tools: Optional[List[Dict]],
        is_multimodal: bool,
        task_results: Optional[List[Dict[str, Any]]] = None,
    ) -> MessageProcessingResult:
        """Apply Jinja chat template"""
        prompt = ""
        prompt_ids = []
        openai_compatible_messages = []
        image_data = []
        video_data = []
        audio_data = []
        modalities = []

        template_content_format = self.template_manager.jinja_template_content_format

        # Try custom encoding first (override in subclass for custom renderers)
        thinking_requested = (request.chat_template_kwargs or {}).get(
            "thinking", envs.SGLANG_DEFAULT_THINKING.get()
        )
        thinking_mode = (
            ThinkingMode.THINKING if thinking_requested else ThinkingMode.CHAT
        )
        messages = [msg.model_dump() for msg in request.messages]
        # K3 renders history tool-call arguments from their raw string
        # (see encoding_k3); parsing them here would lose the source text.
        if self.chat_encoding_spec != "kimi_k3":
            for message in messages:
                normalize_assistant_tool_call_arguments(message)

        prompt_ids = self._encode_messages(
            copy.deepcopy(messages),
            request,
            thinking_mode,
            tools=tools,
        )

        if prompt_ids is not None:
            # Custom encoding produced prompt_ids. Text-only encoders (dsv4/dsv32) need
            # nothing more; Inkling is the only multimodal custom encoder and still needs the
            # image/audio media harvested from the messages for the MM processor.
            if self.chat_encoding_spec == "inkling":
                for message in request.messages:
                    msg_dict = message.model_dump()
                    if msg_dict.get("content") is None:
                        msg_dict["content"] = ""
                    process_content_for_template_format(
                        msg_dict,
                        "openai",
                        image_data,
                        video_data,
                        audio_data,
                        modalities,
                    )
        elif self.chat_encoding_spec == "kimi_k3":
            # K3 ships no Jinja chat template; its tokenizer renders messages
            # through the Python XTML encoding. Tokenize eagerly so structural
            # markers keep their special-token ids while user- and
            # tool-supplied text stays ordinary (re-encoding a rendered string
            # would lose that distinction).
            messages = copy.deepcopy(messages)
            media_kinds: List[str] = []
            for i, msg in enumerate(messages):
                if isinstance(msg.get("content"), list):
                    messages[i] = self._flatten_kimi_k3_content(
                        msg, image_data, video_data, audio_data, media_kinds
                    )
                elif msg.get("content") is None:
                    msg["content"] = ""

            # Re-serialize message-level (system/developer) tools slim, dropping
            # optional/default fields (strict, defer_loading, ...) that
            # model_dump() would otherwise emit and inflate the rendered prompt.
            for msg, source in zip(messages, request.messages):
                if (
                    isinstance(source, ChatCompletionMessageGenericParam)
                    and source.role in ("system", "developer")
                    and source.tools
                ):
                    msg["tools"] = [
                        tool.model_dump(exclude_unset=True, by_alias=True)
                        for tool in source.tools
                    ]

            for msg in messages:
                if msg.get("role") == "developer":
                    msg["role"] = "system"

            # Only split off a trailing assistant message for
            # continue_final_message. Otherwise leave it in place: K3 renders it
            # as a closed historical assistant turn, and rewriting it to a user
            # message (the default _handle_last_assistant_message behavior) would
            # drop the assistant framing tokens and undercount the prompt.
            assistant_prefix = None
            if request.continue_final_message:
                messages, assistant_prefix = self._handle_last_assistant_message(
                    messages, request
                )

            template_kwargs = dict(request.chat_template_kwargs or {})
            template_kwargs.pop("tokenize", None)
            template_kwargs.pop("return_dict", None)
            template_kwargs["image_prompts"] = self._build_kimi_k3_media_prompts(
                media_kinds, task_results
            )
            # encoding_k3 accepts thinking_effort in {low, high, max} and
            # asserts on anything else; "none" is handled at the protocol
            # level by disabling thinking.
            if (
                request.reasoning_effort is not None
                and "thinking_effort" not in template_kwargs
            ):
                if request.reasoning_effort in ("low", "high", "max"):
                    template_kwargs["thinking_effort"] = request.reasoning_effort
                elif request.reasoning_effort != "none":
                    logger.warning(
                        "Kimi K3 supports thinking_effort low/high/max; ignoring "
                        "reasoning_effort=%r.",
                        request.reasoning_effort,
                    )

            # Only forward tool_choice to the renderer for required/none when
            # tools are actually present, or for a client-sent "none". The
            # protocol defaults tool_choice to "none" for tool-less requests;
            # forwarding that default would make the renderer emit the
            # tool-choice block for a plain chat, while an explicit "none" is
            # rendered by the reference API even without tools.
            if isinstance(request.tool_choice, str) and (
                (
                    self._collect_tools(request)
                    and request.tool_choice in ("required", "none")
                )
                or (request.tool_choice == "none" and request.tool_choice_explicit)
            ):
                template_kwargs.setdefault("tool_choice", request.tool_choice)
            if request.response_format is not None:
                template_kwargs.setdefault(
                    "response_format",
                    request.response_format.model_dump(
                        exclude_unset=True, by_alias=True
                    ),
                )

            # Top-level tools: render even when tool_choice == "none" (the
            # reference API still declares them), and dump slim so default
            # fields don't inflate the prompt.
            request_tools = (
                [
                    tool.model_dump(exclude_unset=True, by_alias=True)
                    for tool in request.tools
                ]
                if request.tools
                else None
            )

            prompt_ids = self.tokenizer_manager.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                tools=request_tools,
                return_dict=False,
                **template_kwargs,
            )

            if assistant_prefix:
                prompt_ids = self._append_assistant_prefix_to_prompt_ids(
                    prompt_ids, assistant_prefix
                )

            # The multimodal processor consumes a text prompt and re-encodes
            # around the media placeholders, so hand it the decoded prompt.
            if is_multimodal:
                prompt = self.tokenizer_manager.tokenizer.decode(prompt_ids)
        elif self.chat_encoding_spec is not None:
            # dsv4/dsv32 encoding path
            messages = copy.deepcopy(messages)

            # dsv4/dsv32 are text-only and consume string content; flatten
            # OpenAI parts-list content here so the encoder sees a plain string.
            for i, msg in enumerate(messages):
                if isinstance(msg.get("content"), list):
                    messages[i] = process_content_for_template_format(
                        msg, "string", [], [], [], []
                    )

            for msg in messages:
                if msg.get("content") is None:
                    msg["content"] = ""
                processed_msg = process_content_for_template_format(
                    msg,
                    template_content_format,
                    image_data,
                    video_data,
                    audio_data,
                    modalities,
                    use_dpsk_v32_encoding=self.chat_encoding_spec == "dsv32",
                )
                msg.update(processed_msg)

            # Handle continue_final_message: separate final assistant message
            messages, assistant_prefix = self._handle_last_assistant_message(
                messages, request
            )

            if messages[0]["role"] != "system":
                # insert an empty system prompt to help render tool system prompt
                messages.insert(0, {"role": "system", "content": ""})
            if request.tools:
                messages[0]["tools"] = [tool.model_dump() for tool in request.tools]

            # Default encoding (dsv4/dsv32)
            if self.chat_encoding_spec == "dsv4":
                # V4 encoder only accepts "max" / "high" / None.
                # OpenAI protocol defaults to "medium" which V4 rejects; drop it.
                # Fallback: if request didn't set it, try env SGLANG_DSV4_REASONING_EFFORT.
                effort_source = request.reasoning_effort
                if effort_source is None:
                    env_val = envs.SGLANG_DSV4_REASONING_EFFORT.get()
                    if env_val:
                        effort_source = env_val
                v4_reasoning_effort = (
                    effort_source if effort_source in ("max", "high") else None
                )
                if request.task is not None:
                    encoding_dsv4.attach_task_to_last_user_message(
                        messages, request.task
                    )
                real_input = encoding_dsv4.encode_messages(
                    messages,
                    thinking_mode=thinking_mode,
                    reasoning_effort=v4_reasoning_effort,
                )
                prompt_ids = self.tokenizer_manager.tokenizer.encode(real_input)
            else:
                real_input = encoding_dsv32.encode_messages(
                    messages, thinking_mode=thinking_mode
                )
                prompt_ids = self.tokenizer_manager.tokenizer.encode(real_input)

            # Append assistant prefix if continue_final_message is enabled
            if assistant_prefix:
                prompt_ids = self._append_assistant_prefix_to_prompt_ids(
                    prompt_ids, assistant_prefix
                )
        else:
            for msg_dict in copy.deepcopy(messages):
                if msg_dict.get("content") is None:
                    msg_dict["content"] = ""

                # Process content based on detected template format
                processed_msg = process_content_for_template_format(
                    msg_dict,
                    template_content_format,
                    image_data,
                    video_data,
                    audio_data,
                    modalities,
                )

                processed_msg["content"] = normalize_tool_content(
                    processed_msg["role"], processed_msg.get("content")
                )

                # per the Transformers docs & maintainers, tool call arguments in
                # assistant-role messages with tool_calls need to be dicts not JSON str -
                # this is how tool-use chat templates will expect them moving forwards
                # so, for messages that have tool_calls, parse the string (which we get
                # from openAI format) to dict
                if (
                    processed_msg["role"] == "assistant"
                    and "tool_calls" in processed_msg
                    and isinstance(processed_msg["tool_calls"], list)
                ):
                    valid_tool_calls = []
                    for item in processed_msg["tool_calls"]:
                        if "arguments" in item["function"] and isinstance(
                            item["function"]["arguments"], str
                        ):
                            try:
                                item["function"]["arguments"] = orjson.loads(
                                    item["function"]["arguments"]
                                )
                                if not item["function"]["name"]:
                                    error_msg = (
                                        "Model chat template render failed due to invalid message format "
                                        "(missing 'name' in tool_calls is null)"
                                    )
                                    raise ValueError(error_msg)
                                valid_tool_calls.append(item)
                            except orjson.JSONDecodeError:
                                pass
                        else:
                            valid_tool_calls.append(item)
                    if valid_tool_calls:
                        processed_msg["tool_calls"] = valid_tool_calls
                    else:
                        # Be permissive: drop tool_calls but keep assistant content.
                        processed_msg.pop("tool_calls", None)

                openai_compatible_messages.append(processed_msg)

            # Handle continue_final_message: separate final assistant message
            openai_compatible_messages, assistant_prefix = (
                self._handle_last_assistant_message(openai_compatible_messages, request)
            )

            extra_template_kwargs = {}
            if request.reasoning_effort is not None:
                extra_template_kwargs["reasoning_effort"] = request.reasoning_effort
            if request.chat_template_kwargs:
                extra_template_kwargs.update(request.chat_template_kwargs)

            rc = self.template_manager.reasoning_config
            if rc is not None and rc.effort_kwarg is not None:
                if request.reasoning_effort == "low":
                    extra_template_kwargs.setdefault(rc.effort_kwarg, True)
                elif request.reasoning_effort in ("medium", "high", "max"):
                    logger.warning(
                        "Model '%s' supports only 'low' reasoning effort; "
                        "requested '%s' treated as default thinking",
                        self.tokenizer_manager.server_args.served_model_name,
                        request.reasoning_effort,
                    )

            # Split apply_chat_template(tokenize=True) into render + encode so we
            # can skip add_special_tokens=False on tokenizers that don't auto-add
            # specials (Kimi-like, OpenAI-chat analogue of #25265). Chat
            # templates already include role/special tokens, so the encode must
            # avoid double BOS on tokenizers that would add it.
            encode_kwargs = (
                {"add_special_tokens": False}
                if self._tokenizer_auto_adds_specials
                else {}
            )
            try:
                rendered_prompt = self.tokenizer_manager.tokenizer.apply_chat_template(
                    openai_compatible_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    tools=tools,
                    return_dict=False,
                    **extra_template_kwargs,
                )
                prompt_ids = self.tokenizer_manager.tokenizer.encode(
                    rendered_prompt, **encode_kwargs
                )
            except Exception:
                # If the first attempt fails, try with flat function-only format.
                # Some templates (e.g. Mistral) expect tools without the OpenAI wrapper.
                tools = (
                    [t["function"] if "function" in t else t for t in tools]
                    if tools
                    else None
                )
                try:
                    rendered_prompt = (
                        self.tokenizer_manager.tokenizer.apply_chat_template(
                            openai_compatible_messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            tools=tools,
                            return_dict=False,
                            **extra_template_kwargs,
                        )
                    )
                    prompt_ids = self.tokenizer_manager.tokenizer.encode(
                        rendered_prompt, **encode_kwargs
                    )
                except Exception as template_error:
                    # 原本这里只捕获 jinja2.TemplateError，但是实际线上大量非法请求因为 chat template 编写不完善没
                    # 抛出 jinja2.TemplateError，所以改成捕获所有异常
                    # Template errors (e.g., from raise_exception in Jinja templates)
                    # and TypeError (e.g., tojson filter on Jinja2 Undefined variables)
                    # should be treated as client errors (400 BadRequest)
                    err_msg = f"failed to apply chat template, err: {template_error}"
                    raise ValueError(err_msg) from template_error

            # Append assistant prefix if continue_final_message is enabled
            if assistant_prefix:
                prompt_ids = self._append_assistant_prefix_to_prompt_ids(
                    prompt_ids, assistant_prefix
                )

            if is_multimodal:
                prompt = self.tokenizer_manager.tokenizer.decode(prompt_ids)

        stop = request.stop
        image_data = image_data if image_data else None
        audio_data = audio_data if audio_data else None
        video_data = video_data if video_data else None
        modalities = modalities if modalities else []
        return MessageProcessingResult(
            prompt=prompt,
            prompt_ids=prompt_ids,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            modalities=modalities,
            stop=stop,
        )

    def _apply_conversation_template(
        self,
        request: ChatCompletionRequest,
        is_multimodal: bool,
    ) -> MessageProcessingResult:
        """Apply conversation template"""
        prompt = ""
        prompt_ids = []
        conv = generate_chat_conv(request, self.template_manager.chat_template_name)

        # If we should continue the final assistant message, adjust the conversation.
        if (
            request.continue_final_message
            and request.messages
            and request.messages[-1].role == "assistant"
        ):
            # Remove the auto-added blank assistant turn, if present.
            if conv.messages and conv.messages[-1][1] is None:
                conv.messages.pop()
            # Rebuild the prompt from the conversation.
            prompt = conv.get_prompt()
            # Strip trailing stop tokens or separators that indicate end-of-assistant.
            if isinstance(conv.stop_str, list):
                for stop_token in conv.stop_str:
                    if prompt.endswith(stop_token):
                        prompt = prompt[: -len(stop_token)]
            elif isinstance(conv.stop_str, str) and prompt.endswith(conv.stop_str):
                prompt = prompt[: -len(conv.stop_str)]
            if conv.sep and prompt.endswith(conv.sep):
                prompt = prompt[: -len(conv.sep)]
            if getattr(conv, "sep2", None) and prompt.endswith(conv.sep2):
                prompt = prompt[: -len(conv.sep2)]
        else:
            prompt = conv.get_prompt()
            if self._get_reasoning_from_request(request) and (
                self._reasoning_detector is None
                or not self._reasoning_detector.thinks_internally
            ):
                # Models with thinks_internally=True think without a leading <think> token
                prompt += "<think>"  # Note(Xinyuan): hard code thinking token

        image_data = conv.image_data if conv.image_data else None
        video_data = conv.video_data if conv.video_data else None
        audio_data = conv.audio_data if conv.audio_data else None
        modalities = conv.modalities if conv.modalities else []
        stop = copy.copy(conv.stop_str or [] if not request.ignore_eos else [])

        if request.stop:
            if isinstance(request.stop, str):
                stop.append(request.stop)
            else:
                stop.extend(request.stop)

        if not is_multimodal:
            prompt_ids = self.tokenizer_manager.tokenizer.encode(prompt)

        return MessageProcessingResult(
            prompt=prompt,
            prompt_ids=prompt_ids,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            modalities=modalities,
            stop=stop,
        )

    async def _handle_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: ChatCompletionRequest,
        raw_request: Request,
    ) -> Union[StreamingResponse, ErrorResponse]:
        """Handle streaming chat completion request"""
        generator = self._generate_chat_stream(adapted_request, request, raw_request)

        # Kick-start the generator to trigger validation before HTTP 200 is sent.
        # If validation fails (e.g., context length exceeded), we can still return
        # a proper HTTP 400 error response instead of streaming it as SSE payload.
        try:
            first_chunk = await generator.__anext__()
        except ValueError as e:
            return self.create_error_response(str(e))

        async def prepend_first_chunk():
            yield first_chunk
            async for chunk in generator:
                yield chunk

        return StreamingResponse(
            prepend_first_chunk(),
            media_type="text/event-stream",
            background=self.tokenizer_manager.create_abort_task(adapted_request),
        )

    async def _generate_chat_stream(
        self,
        adapted_request: GenerateReqInput,
        request: ChatCompletionRequest,
        raw_request: Request,
        tokenizer_rev_request_time: Optional[float] = None,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming chat completion response"""
        # Parsers for tool calls and reasoning
        parser_dict = {}
        reasoning_parser_dict = {}

        # State tracking for streaming
        is_firsts = {}
        stream_offsets = {}
        n_prev_tokens = {}
        has_tool_calls = {}
        finish_reasons = {}

        # Usage tracking
        prompt_tokens = {}
        reasoning_tokens = {}
        completion_tokens = {}
        cached_tokens = {}
        hidden_states = {}
        routed_experts = {}
        cached_tokens_details = {}
        image_tokens = {}
        audio_tokens = {}
        video_tokens = {}
        tool_call_last_return_time = time.time()

        last_return_tokens = {}
        output_ids_offset = {}  # Cumulative-to-incremental offset (stream_output=False)
        pending_output_token_ids = {}  # Accumulate token IDs, flush on SSE chunk emit
        pending_logprobs = {}  # Accumulate logprob entries, flush on SSE chunk emit

        stream_started = False
        try:
            include_usage, continuous_usage_stats = should_include_usage(
                request.stream_options,
                self.tokenizer_manager.server_args.stream_response_default_include_usage,
            )
            # Moonshot streaming spec: `delta.internal_content.token_ids` on every
            # increment frame. Same source as the function-call `output_token_ids`
            # field, so one buffer feeds both.
            emit_internal_content = bool(
                request.stream_options
                and request.stream_options.include_internal_content
            )
            # Continuation applies to every tool-call request, including K3/DS
            # dynamic tools declared on system/developer messages rather than
            # only in the top-level request.tools field.
            emit_output_token_ids = request.tool_choice != "none" and bool(
                self._collect_tools(request)
            )

            async for content in self.tokenizer_manager.generate_request(
                adapted_request, raw_request, tokenizer_rev_request_time
            ):
                index = content.get("index", 0)

                prompt_tokens[index] = self._reported_prompt_tokens(
                    content["meta_info"]
                )
                completion_tokens[index] = content["meta_info"].get(
                    "completion_tokens", 0
                )
                reasoning_tokens[index] = content["meta_info"].get(
                    "reasoning_tokens", 0
                )
                cached_tokens[index] = content["meta_info"].get("cached_tokens", 0)
                hidden_states[index] = content["meta_info"].get("hidden_states", None)
                routed_experts[index] = content["meta_info"].get("routed_experts", None)
                cached_tokens_details[index] = content["meta_info"].get(
                    "cached_tokens_details", None
                )
                image_tokens[index] = content["meta_info"].get("image_tokens", 0)
                audio_tokens[index] = content["meta_info"].get("audio_tokens", 0)
                video_tokens[index] = content["meta_info"].get("video_tokens", 0)

                # Extract incremental output_token_ids for function call
                # continuation and/or the streaming-spec internal_content.
                raw_output_ids = (
                    content.get("output_ids")
                    if (emit_output_token_ids or emit_internal_content)
                    else None
                )
                if raw_output_ids:
                    if self.tokenizer_manager.server_args.incremental_streaming_output:
                        # tokenizer_manager already returns incremental
                        incremental_ids = raw_output_ids
                    else:
                        # tokenizer_manager returns cumulative, compute incremental
                        prev_len = output_ids_offset.get(index, 0)
                        incremental_ids = raw_output_ids[prev_len:]
                        output_ids_offset[index] = len(raw_output_ids)
                    # Accumulate into pending buffer; will be flushed when
                    # an SSE chunk is actually emitted to the client.
                    if index not in pending_output_token_ids:
                        pending_output_token_ids[index] = []
                    pending_output_token_ids[index].extend(incremental_ids)

                # Handle logprobs
                if request.logprobs:
                    n_prev_token = n_prev_tokens.get(index, 0)
                    total_output_logprobs = content["meta_info"][
                        "output_token_logprobs_length"
                    ]
                    if n_prev_token < total_output_logprobs:
                        pending_logprobs.setdefault(index, []).extend(
                            self._process_streaming_logprobs(
                                content, n_prev_token, total_output_logprobs
                            ).model_dump()["content"]
                        )
                    n_prev_tokens[index] = total_output_logprobs

                finish_reason = content["meta_info"].get("finish_reason", None)
                finish_reason_type = finish_reason["type"] if finish_reason else None

                # Track finish_reason for each index
                if finish_reason_type:
                    # Abort with an explicit error status_code is a system error
                    # (timeout, OOM, validation). If no chunk has been sent yet
                    # the HTTP status is still ours to choose, so raise and let
                    # the kick-start in _handle_streaming_request turn it into a
                    # real error response; otherwise HTTP 200 is already
                    # committed and the only option is an in-band error chunk.
                    # A graceful abort (no status_code, e.g. user-initiated via
                    # /abort_request or session lifecycle cleanup) falls through
                    # to the normal chunk path, matching the non-stream behavior
                    # in tokenizer_manager._handle_abort_finish_reason.
                    if finish_reason_type == "abort" and isinstance(
                        finish_reason.get("status_code"), HTTPStatus
                    ):
                        if not stream_started:
                            self.raise_abort_before_stream(finish_reason)
                        code = finish_reason["status_code"]
                        error = self.create_streaming_error_response(
                            finish_reason.get("message", "Generation aborted."),
                            code.name,
                            code.value,
                        )
                        yield f"data: {error}\n\n"
                        break
                    finish_reasons[index] = finish_reason

                # First chunk with role
                if is_firsts.get(index, True):
                    is_firsts[index] = False
                    # Set before the yield: once the chunk leaves this generator
                    # HTTP 200 is committed, and the flag is what guards against
                    # raising after that point.
                    stream_started = True
                    yield build_sse_content(
                        chunk_id=content["meta_info"]["id"],
                        created=int(time.time()),
                        model=request.model,
                        index=index,
                        role="assistant",
                        content="",
                    )

                offset = stream_offsets.get(index, 0)
                if self.tokenizer_manager.server_args.incremental_streaming_output:
                    # content["text"] is already the incremental delta
                    delta = content["text"]
                else:
                    delta = content["text"][offset:]
                stream_offsets[index] = len(content["text"])

                needs_tool_processing = (
                    request.tool_choice != "none"
                    and self._collect_tools(request)
                    and (
                        self.tool_call_parser
                        or request.tool_choice == "required"
                        or isinstance(request.tool_choice, ToolChoice)
                    )
                )
                step_chunks = []

                # Handle reasoning content
                # TODO(liyaolang): Reasoning content is parsed by upstream currently,
                #                  and should moved here in the future.
                if (
                    self.reasoning_parser
                    and request.separate_reasoning
                    and not self.mocked_disable_reasoning_parser
                ):
                    reasoning_text, delta = self._process_reasoning_stream(
                        index,
                        delta,
                        reasoning_parser_dict,
                        content,
                        request,
                        finish_reason_type if needs_tool_processing else None,
                    )
                    if reasoning_text:
                        # On a reasoning/content boundary both projections share
                        # the same tokens. Tool streams coalesce these internal
                        # frames before exposing the token prefix to the client.
                        flushed_ids, internal_content = _flush_pending_token_ids(
                            pending_output_token_ids,
                            index,
                            emit_output_token_ids=emit_output_token_ids,
                            emit_internal_content=emit_internal_content,
                        )
                        usage = None
                        if continuous_usage_stats:
                            usage = UsageProcessor.calculate_token_usage(
                                prompt_tokens=prompt_tokens.get(index, 0),
                                reasoning_tokens=reasoning_tokens.get(index, 0),
                                completion_tokens=completion_tokens.get(index, 0),
                            ).model_dump()

                        reasoning_chunk = _fast_sse_content(
                            chunk_id=content["meta_info"]["id"],
                            created=int(time.time()),
                            model=request.model,
                            index=index,
                            reasoning_content=reasoning_text,
                            usage=usage,
                            tokens=completion_tokens.get(index, 0)
                            - last_return_tokens.get(index, 0),
                            output_token_ids=flushed_ids,
                            internal_content=internal_content,
                            logprobs=_flush_pending_logprobs(pending_logprobs, index),
                        )
                        if needs_tool_processing:
                            step_chunks.append(reasoning_chunk)
                        else:
                            yield reasoning_chunk
                        last_return_tokens[index] = completion_tokens.get(index, 0)

                # Handle tool calls
                if needs_tool_processing:
                    async for chunk in self._process_tool_call_stream(
                        index,
                        delta,
                        parser_dict,
                        content,
                        request,
                        has_tool_calls,
                        continuous_usage_stats,
                        tool_call_last_return_time,
                        finish_reason_type=finish_reason_type,
                    ):
                        if chunk:
                            chunk = ChatCompletionStreamResponse.model_validate_json(
                                chunk[6:-2]
                            )
                            chunk.choices[0].delta.tokens = completion_tokens.get(
                                index, 0
                            ) - last_return_tokens.get(index, 0)
                            last_return_tokens[index] = completion_tokens.get(index, 0)
                            # Flush all accumulated output_token_ids into this chunk
                            flushed_ids, internal_content = _flush_pending_token_ids(
                                pending_output_token_ids,
                                index,
                                emit_output_token_ids=emit_output_token_ids,
                                emit_internal_content=emit_internal_content,
                            )
                            chunk.choices[0].output_token_ids = flushed_ids
                            chunk.choices[0].delta.internal_content = internal_content
                            flushed_logprobs = _flush_pending_logprobs(
                                pending_logprobs, index
                            )
                            if flushed_logprobs is not None:
                                chunk.choices[0].logprobs = (
                                    ChoiceLogprobs.model_validate(flushed_logprobs)
                                )
                            step_chunks.append(f"data: {chunk.model_dump_json()}\n\n")
                            tool_call_last_return_time = time.time()

                    # Send any remaining tool call arguments when generation finishes
                    if finish_reason_type == "stop" and index in parser_dict:
                        parser = parser_dict[index]
                        remaining_chunk = self._check_for_unstreamed_tool_args(
                            parser, content, request, index
                        )
                        if remaining_chunk:
                            flushed_ids, internal_content = _flush_pending_token_ids(
                                pending_output_token_ids,
                                index,
                                emit_output_token_ids=emit_output_token_ids,
                                emit_internal_content=emit_internal_content,
                            )
                            if flushed_ids is not None or internal_content is not None:
                                rc = ChatCompletionStreamResponse.model_validate_json(
                                    remaining_chunk[6:-2]
                                )
                                rc.choices[0].output_token_ids = flushed_ids
                                rc.choices[0].delta.internal_content = internal_content
                                remaining_chunk = f"data: {rc.model_dump_json()}\n\n"
                            step_chunks.append(remaining_chunk)

                    if step_chunks:
                        atomic_chunk = _coalesce_tool_stream_chunks(step_chunks)
                        choice = atomic_chunk.choices[0]
                        choice.delta.tokens = (
                            (choice.delta.tokens or 0)
                            + completion_tokens.get(index, 0)
                            - last_return_tokens.get(index, 0)
                        )
                        last_return_tokens[index] = completion_tokens.get(index, 0)
                        # The final JSON supplement can be the only output of
                        # this step; include its token accounting and logprobs.
                        remaining_logprobs = _flush_pending_logprobs(
                            pending_logprobs, index
                        )
                        if remaining_logprobs:
                            tail = ChoiceLogprobs.model_validate(remaining_logprobs)
                            if choice.logprobs is None:
                                choice.logprobs = tail
                            else:
                                choice.logprobs.content = (
                                    choice.logprobs.content or []
                                ) + (tail.content or [])
                        if finish_reason_type == "stop":
                            # EOF flushing is not part of prefix replay. Publish
                            # its output together with the terminal marker.
                            choice.finish_reason = (
                                "tool_calls" if has_tool_calls.get(index) else "stop"
                            )
                            choice.matched_stop = finish_reason.get("matched")
                            finish_reasons.pop(index, None)
                        yield f"data: {atomic_chunk.model_dump_json()}\n\n"

                else:
                    # Regular content
                    if delta:
                        # Flush accumulated output_token_ids for regular content
                        flushed_ids, internal_content = _flush_pending_token_ids(
                            pending_output_token_ids,
                            index,
                            emit_output_token_ids=emit_output_token_ids,
                            emit_internal_content=emit_internal_content,
                        )

                        usage = None
                        # Add usage stats if continuous_usage_stats is enabled
                        if continuous_usage_stats:
                            usage = UsageProcessor.calculate_token_usage(
                                prompt_tokens=prompt_tokens.get(index, 0),
                                reasoning_tokens=reasoning_tokens.get(index, 0),
                                completion_tokens=completion_tokens.get(index, 0),
                            ).model_dump()

                        yield _fast_sse_content(
                            chunk_id=content["meta_info"]["id"],
                            created=int(time.time()),
                            model=request.model,
                            index=index,
                            content=delta,
                            logprobs=_flush_pending_logprobs(pending_logprobs, index),
                            usage=usage,
                            tokens=completion_tokens.get(index, 0)
                            - last_return_tokens.get(index, 0),
                            output_token_ids=flushed_ids,
                            internal_content=internal_content,
                        )
                        last_return_tokens[index] = completion_tokens.get(index, 0)

            # Send finish_reason chunks for each index that completed
            for idx, finish_reason_data in finish_reasons.items():
                finish_reason_type = finish_reason_data["type"]

                # Change finish_reason to "tool_calls" if we had tool calls and stopped naturally
                final_finish_reason = finish_reason_type
                if has_tool_calls.get(idx, False) and finish_reason_type == "stop":
                    final_finish_reason = "tool_calls"

                matched_stop = finish_reason_data.get("matched")
                final_output_ids, final_internal_content = _flush_pending_token_ids(
                    pending_output_token_ids,
                    idx,
                    emit_output_token_ids=emit_output_token_ids,
                    emit_internal_content=emit_internal_content,
                )
                yield _fast_sse_content(
                    chunk_id=content["meta_info"]["id"],
                    created=int(time.time()),
                    model=request.model,
                    index=idx,
                    finish_reason=final_finish_reason,
                    matched_stop=matched_stop,
                    tokens=completion_tokens.get(idx, 0)
                    - last_return_tokens.get(idx, 0),
                    output_token_ids=final_output_ids,
                    internal_content=final_internal_content,
                    logprobs=_flush_pending_logprobs(pending_logprobs, idx),
                )
                last_return_tokens[idx] = completion_tokens.get(idx, 0)

            # Send hidden states if requested
            if request.return_hidden_states and hidden_states:
                for index, choice_hidden_states in hidden_states.items():
                    if choice_hidden_states:
                        last_token_hidden_states = (
                            choice_hidden_states[-1]
                            if len(choice_hidden_states) > 1
                            else []
                        )
                        hidden_states_chunk = ChatCompletionStreamResponse(
                            id=content["meta_info"]["id"],
                            created=int(time.time()),
                            choices=[
                                ChatCompletionResponseStreamChoice(
                                    index=index,
                                    delta=DeltaMessage(
                                        hidden_states=last_token_hidden_states
                                    ),
                                    finish_reason=None,  # Hidden states don't need finish_reason
                                )
                            ],
                            model=request.model,
                        )
                        yield f"data: {hidden_states_chunk.model_dump_json()}\n\n"

            sglext_routed = None
            if request.return_routed_experts and routed_experts:
                sglext_routed = next(
                    (v for v in routed_experts.values() if v is not None), None
                )

            sglext_details = None
            if request.return_cached_tokens_details and cached_tokens_details:
                first_details = next(
                    (v for v in cached_tokens_details.values() if v is not None), None
                )
                if first_details is not None:
                    sglext_details = cached_tokens_details_from_dict(first_details)

            if sglext_routed is not None or sglext_details is not None:
                sglext_chunk = ChatCompletionStreamResponse(
                    id=content["meta_info"]["id"],
                    created=int(time.time()),
                    choices=[],  # sglext is at response level
                    model=request.model,
                    sglext=SglExt(
                        routed_experts=sglext_routed,
                        cached_tokens_details=sglext_details,
                    ),
                )
                yield f"data: {sglext_chunk.model_dump_json()}\n\n"

            # Additional usage chunk
            if include_usage:
                # Multimodal tokens are per-prompt (input side), so aggregate
                # once per prompt (first choice), matching prompt/cached semantics.
                total_image_tokens = sum(
                    tok for idx, tok in image_tokens.items() if idx % request.n == 0
                )
                total_audio_tokens = sum(
                    tok for idx, tok in audio_tokens.items() if idx % request.n == 0
                )
                total_video_tokens = sum(
                    tok for idx, tok in video_tokens.items() if idx % request.n == 0
                )
                usage = UsageProcessor.calculate_streaming_usage(
                    prompt_tokens,
                    reasoning_tokens,
                    completion_tokens,
                    cached_tokens=cached_tokens,
                    n_choices=request.n,
                    enable_cache_report=self.tokenizer_manager.server_args.enable_cache_report,
                    image_tokens=total_image_tokens,
                    audio_tokens=total_audio_tokens,
                    video_tokens=total_video_tokens,
                )
                usage_chunk = ChatCompletionStreamResponse(
                    id=content["meta_info"]["id"],
                    created=int(time.time()),
                    choices=[],  # Empty choices array as per OpenAI spec
                    model=request.model,
                    usage=usage,
                    trace_info=self._build_trace_info(request, content["meta_info"]),
                )
                yield f"data: {usage_chunk.model_dump_json()}\n\n"

        except ValueError as e:
            if not stream_started:
                raise
            error = self.create_streaming_error_response(str(e))
            yield f"data: {error}\n\n"

        yield "data: [DONE]\n\n"

    async def _handle_non_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: ChatCompletionRequest,
        raw_request: Request,
        tokenizer_rev_request_time: Optional[float] = None,
    ) -> Union[ChatCompletionResponse, ErrorResponse, ORJSONResponse]:
        """Handle non-streaming chat completion request"""
        try:
            ret = await self.tokenizer_manager.generate_request(
                adapted_request, raw_request, tokenizer_rev_request_time
            ).__anext__()
        except ValueError as e:
            return self.create_error_response(str(e))

        if not isinstance(ret, list):
            ret = [ret]

        response = self._build_chat_response(
            request,
            ret,
            int(time.time()),
        )

        return response

    def _build_chat_response(
        self,
        request: ChatCompletionRequest,
        ret: List[Dict[str, Any]],
        created: int,
    ) -> Union[ChatCompletionResponse, ORJSONResponse]:
        """Build chat completion response from generation results"""
        if self.chat_encoding_spec == "kimi_k3":
            ret = [
                {
                    **item,
                    "meta_info": {
                        **item["meta_info"],
                        "prompt_tokens": self._reported_prompt_tokens(
                            item["meta_info"]
                        ),
                    },
                }
                for item in ret
            ]

        choices = []

        # Build sglext at response level (from first ret_item, as these are per-request)
        first_ret = ret[0]
        routed_experts = process_routed_experts_from_ret(first_ret, request)
        cached_tokens_details = process_cached_tokens_details_from_ret(
            first_ret, request
        )
        response_sglext = None
        if routed_experts or cached_tokens_details:
            response_sglext = SglExt(
                routed_experts=routed_experts,
                cached_tokens_details=cached_tokens_details,
            )

        for idx, ret_item in enumerate(ret):
            # Process logprobs
            choice_logprobs = None
            if request.logprobs:
                choice_logprobs = self._process_response_logprobs(ret_item)

            # Handle hidden states
            hidden_states = process_hidden_states_from_ret(ret_item, request)

            finish_reason = ret_item["meta_info"]["finish_reason"]
            text = ret_item["text"]
            output_token_ids = ret_item.get("output_ids") if request.tools else None

            # Handle reasoning content
            reasoning_text = None
            reasoning_parser = self.reasoning_parser
            # TODO(liyaolang): Reasoning content is parsed by upstream currently,
            #                  and should moved here in the future.
            if (
                reasoning_parser
                and request.separate_reasoning
                and not self.mocked_disable_reasoning_parser
            ):
                is_force_reasoning = (
                    self.template_manager.force_reasoning
                    or self._get_reasoning_from_request(request)
                )
                try:
                    parser = ReasoningParser(
                        model_type=self.reasoning_parser,
                        stream_reasoning=False,
                        force_reasoning=is_force_reasoning,
                        request=request,
                        tokenizer=self.tokenizer_manager.tokenizer,
                    )
                    reasoning_text, text = parser.parse_non_stream(text)
                except Exception as e:
                    logger.error(f"Reasoning parsing error: {e}")
                    return self.create_error_response(
                        "Failed to parse reasoning content",
                        err_type="InternalServerError",
                        status_code=500,
                    )

            all_tools = self._collect_tools(request) or []
            # For required/named tool_choice, JSON is parsed directly unless
            # the active constraint emits a model-native format such as K3 XTML.
            needs_tool_processing = (
                request.tool_choice != "none"
                and all_tools
                and (
                    self.tool_call_parser
                    or request.tool_choice == "required"
                    or isinstance(request.tool_choice, ToolChoice)
                )
            )
            if needs_tool_processing:
                # Restore detector state for continuation requests
                original_text = text
                if request.continuation_token_ids:
                    # Token-level concat avoids decode inconsistencies at boundaries.
                    new_output_ids = ret_item.get("output_ids") or []
                    full_ids = list(request.continuation_token_ids) + list(
                        new_output_ids
                    )
                    is_required = request.tool_choice == "required" or isinstance(
                        request.tool_choice, ToolChoice
                    )
                    native_required_parser = (
                        self._build_native_required_tool_parser(all_tools, request)
                        if is_required
                        else None
                    )
                    # Generic required/named output is JSON. Native K3 output
                    # must retain its XTML special-token markers for parsing.
                    strip_special = is_required and native_required_parser is None
                    text = self.tokenizer_manager.tokenizer.decode(
                        full_ids, skip_special_tokens=strip_special
                    )
                    # Re-strip reasoning after token-level concat.
                    if (
                        reasoning_parser
                        and request.separate_reasoning
                        and not self.mocked_disable_reasoning_parser
                    ):
                        try:
                            rp = ReasoningParser(
                                model_type=reasoning_parser,
                                stream_reasoning=False,
                                force_reasoning=is_force_reasoning,
                                request=request,
                            )
                            _, text = rp.parse_non_stream(text)
                        except Exception:
                            pass

            # Handle tool calls
            tool_calls = None
            if request.tool_choice != "none" and all_tools and self.tool_call_parser:
                history_tool_calls_cnt = self._get_history_tool_calls_cnt(request)
                tool_calls, text, finish_reason = self._process_tool_calls(
                    text,
                    all_tools,
                    finish_reason,
                    request.tool_choice,
                    request.parallel_tool_calls,
                    history_tool_calls_cnt,
                )
                # when no tool call detected, restore original text to avoid
                # leaking continuation_text into response content.
                if request.continuation_token_ids and not tool_calls:
                    text = original_text

            # Extract prompt_token_ids if requested
            choice_prompt_token_ids = (
                ret_item.get("prompt_token_ids")
                if request.return_prompt_token_ids or request.return_token_ids
                else None
            )
            choice_token_ids = (
                ret_item["output_ids"] if request.return_token_ids else None
            )

            choice_meta_info = (
                ret_item["meta_info"] if request.return_meta_info else None
            )
            # NOTE: content should not be None but empty string to make sure retokenize consistency.
            reasoning_text, tool_calls = self._get_parsed_response_fields(
                reasoning_text, tool_calls
            )

            choice_data = ChatCompletionResponseChoice(
                index=idx,
                message=ChatMessage(
                    role="assistant",
                    content=text if text else "",
                    tool_calls=tool_calls,
                    reasoning_content=reasoning_text if reasoning_text else None,
                ),
                logprobs=choice_logprobs,
                finish_reason=finish_reason["type"] if finish_reason else None,
                matched_stop=(
                    finish_reason["matched"]
                    if finish_reason and "matched" in finish_reason
                    else None
                ),
                hidden_states=hidden_states,
                prompt_token_ids=choice_prompt_token_ids,
                token_ids=choice_token_ids,
                meta_info=choice_meta_info,
                output_token_ids=output_token_ids,
            )
            choices.append(choice_data)

        # Calculate usage. Multimodal tokens are per-prompt (input side), so
        # aggregate once per prompt (stride by n), matching prompt/cached semantics.
        image_tokens = sum(
            ret[i]["meta_info"].get("image_tokens", 0)
            for i in range(0, len(ret), request.n)
        )
        audio_tokens = sum(
            ret[i]["meta_info"].get("audio_tokens", 0)
            for i in range(0, len(ret), request.n)
        )
        video_tokens = sum(
            ret[i]["meta_info"].get("video_tokens", 0)
            for i in range(0, len(ret), request.n)
        )
        usage = UsageProcessor.calculate_response_usage(
            ret,
            n_choices=request.n,
            enable_cache_report=self.tokenizer_manager.server_args.enable_cache_report,
            image_tokens=image_tokens,
            audio_tokens=audio_tokens,
            video_tokens=video_tokens,
        )

        return ChatCompletionResponse(
            id=ret[0]["meta_info"]["id"],
            created=created,
            model=request.model,
            choices=choices,
            usage=usage,
            metadata={"weight_version": ret[0]["meta_info"]["weight_version"]},
            sglext=response_sglext,
            trace_info=self._build_trace_info(request, ret[0]["meta_info"]),
        )

    def _process_logprobs_tokens(
        self,
        token_logprobs: List[Any],
        top_logprobs: Optional[List[Any]] = None,
    ) -> List[ChatCompletionTokenLogprob]:
        """Build OpenAI-style chat logprob entries from raw meta_info triples.

        Each `token_logprobs` entry is `(logprob, token_id, token_text)`; each
        `top_logprobs` entry is a list of the same triples for one position (or
        None when that position has no candidates). The two lists are
        positionally aligned, so candidates are read with the token's own index.

        Deliberately not routed through `to_openai_style_logprobs`: that helper
        collapses a position's candidates into a `{token_text: logprob}` dict,
        which silently drops candidates that detokenize to the same text and
        makes `len(top_logprobs)` smaller than the requested `top_logprobs`.
        """
        entries = []
        for token_idx, (logprob, _token_id, token) in enumerate(token_logprobs):
            candidates = top_logprobs[token_idx] if top_logprobs else None
            entries.append(
                ChatCompletionTokenLogprob(
                    token=token,
                    bytes=list(token.encode("utf-8")),
                    logprob=logprob,
                    top_logprobs=[
                        TopLogprob(
                            token=cand_token,
                            bytes=list(cand_token.encode("utf-8")),
                            logprob=cand_logprob,
                        )
                        for cand_logprob, _cand_id, cand_token in (candidates or [])
                    ],
                )
            )
        return entries

    def _process_response_logprobs(self, ret_item: Dict[str, Any]) -> ChoiceLogprobs:
        """Process logprobs for non-streaming response"""
        meta_info = ret_item["meta_info"]
        return ChoiceLogprobs(
            content=self._process_logprobs_tokens(
                token_logprobs=meta_info["output_token_logprobs"],
                top_logprobs=meta_info.get("output_top_logprobs"),
            )
        )

    def _process_tool_call_id(
        self,
        call_item: ToolCallItem,
        history_tool_calls_cnt: int,
    ) -> str:
        """Process for generating a new and unique `tool_call_id`"""
        if self.tool_call_parser == "kimi_k3":
            # Kimi-K3 ids must match `^[A-Za-z_][A-Za-z0-9_]*_\d+$`, i.e.
            # {name}_{zero_based_ordinal} — underscore separated, not the `:`
            # used by Kimi-K2. The ordinal spans the whole conversation, so
            # offset the local position by the number of history tool calls.
            return f"{call_item.name}_{history_tool_calls_cnt + call_item.tool_index}"
        elif self.tool_call_parser != "kimi_k2":
            # A simple uuid is sufficient for all models except for Kimi-K2/K3.
            tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
            return tool_call_id
        else:
            # Align with Kimi-K2 format: functions.{name}:{index}
            # Kimi-K2 allows multiple tool_calls in one message; SGLang sets call_item.tool_index to the *local* position inside that message.
            # Therefore, the index must be corrected by using `history_tool_calls_cnt + call_item.tool_index` to ensure globally unique and properly ordered.
            tool_call_id = f"functions.{call_item.name}:{history_tool_calls_cnt+call_item.tool_index}"
            logger.debug(
                f"Process tool call idx, parser: {self.tool_call_parser}, tool_call_id: {tool_call_id}, history_cnt: {history_tool_calls_cnt}"
            )
            return tool_call_id

    def _process_tool_calls(
        self,
        text: str,
        tools: List[Any],
        finish_reason: Dict[str, Any],
        tool_choice: Optional[Union[str, ToolChoice]] = None,
        parallel_tool_calls: bool = True,
        history_tool_calls_cnt: int = 0,
    ) -> ToolCallProcessingResult:
        """Process tool calls in the response"""

        is_required = tool_choice == "required" or isinstance(tool_choice, ToolChoice)

        # Try model-specific parser when output is in native format.
        # For required/named: only use parser when structural_tag was used
        # as constraint (mirrors the streaming path). For auto: always try.
        if self.tool_call_parser:
            parser = FunctionCallParser(
                tool_call_parser=self.tool_call_parser,
                tools=tools,
                tokenizer=self.tokenizer_manager.tokenizer,
            )
            should_try_parser = (
                not is_required
                or parser.detector.supports_structural_tag()
                or parser.detector.parses_required_natively()
            )
            if should_try_parser and parser.has_tool_call(text):
                try:
                    text, call_info_list = parser.parse_non_stream(text)
                    if not call_info_list:
                        return ToolCallProcessingResult(None, text, finish_reason)

                    tool_calls = []
                    for call_info in call_info_list:
                        tool_id = self._process_tool_call_id(
                            call_info, history_tool_calls_cnt
                        )
                        tool_calls.append(
                            ToolCall(
                                id=tool_id,
                                index=getattr(call_info, "tool_index", None),
                                function=FunctionResponse(
                                    name=call_info.name,
                                    arguments=call_info.parameters,
                                ),
                            )
                        )
                    if finish_reason["type"] == "stop":
                        finish_reason["type"] = "tool_calls"
                        finish_reason["matched"] = None
                    return ToolCallProcessingResult(tool_calls, text, finish_reason)
                except Exception as e:
                    logger.error(f"Tool call parsing error: {e}")
                    return ToolCallProcessingResult(None, text, finish_reason)

        # json_schema constraint → JSON array output for required/named
        if is_required:
            original_finish_type = finish_reason["type"]
            if finish_reason["type"] == "stop":
                finish_reason["type"] = "tool_calls"
                finish_reason["matched"] = None
            try:
                tool_call_data = orjson.loads(text)
                tool_calls = []
                for i, tool in enumerate(tool_call_data):
                    call_info = ToolCallItem(
                        tool_index=i,
                        name=tool["name"],
                        parameters=json.dumps(tool["parameters"], ensure_ascii=False),
                    )
                    tool_id = self._process_tool_call_id(
                        call_info, history_tool_calls_cnt
                    )
                    tool_calls.append(
                        ToolCall(
                            id=tool_id,
                            index=i,
                            function=FunctionResponse(
                                name=call_info.name,
                                arguments=call_info.parameters,
                            ),
                        )
                    )
                # Only set finish_reason after successful parsing
                if finish_reason["type"] == "stop":
                    finish_reason["type"] = "tool_calls"
                    finish_reason["matched"] = None
                return ToolCallProcessingResult(tool_calls, "", finish_reason)
            # orjson.JSONDecodeError inherits json.JSONDecodeError
            except (json.JSONDecodeError, TypeError, KeyError, AttributeError) as e:
                logger.error(f"Tool call parsing error: {e}")
                return ToolCallProcessingResult(None, text, finish_reason)

        # Use parser since output is not constrained by JSON schema
        parser = FunctionCallParser(
            self.tool_call_parser, tools, tool_choice, parallel_tool_calls
        )
        if parser.has_tool_call(text):
            if finish_reason["type"] == "stop":
                finish_reason["type"] = "tool_calls"
                finish_reason["matched"] = None
            try:
                text, call_info_list = parser.parse_non_stream(text)
                tool_calls = []
                for call_info in call_info_list:
                    tool_id = self._process_tool_call_id(
                        call_info, history_tool_calls_cnt
                    )
                    tool_calls.append(
                        ToolCall(
                            id=tool_id,
                            index=getattr(call_info, "tool_index", None),
                            function=FunctionResponse(
                                name=call_info.name, arguments=call_info.parameters
                            ),
                        )
                    )
                return ToolCallProcessingResult(tool_calls, text, finish_reason)
            except Exception as e:
                logger.error(f"Tool call parsing error: {e}")
                finish_reason["type"] = original_finish_type
                return ToolCallProcessingResult(None, text, finish_reason)

        return ToolCallProcessingResult(None, text, finish_reason)

    def _process_streaming_logprobs(
        self,
        content: Dict[str, Any],
        n_prev_token: int,
        total_output_logprobs: int,
    ) -> ChoiceLogprobs:
        """Process logprobs for streaming response"""
        output_token_logprobs = content["meta_info"]["output_token_logprobs"]
        output_top_logprobs = content["meta_info"].get("output_top_logprobs", [])
        if not self.tokenizer_manager.server_args.incremental_streaming_output:
            output_token_logprobs = output_token_logprobs[
                n_prev_token:total_output_logprobs
            ]
            output_top_logprobs = output_top_logprobs[
                n_prev_token:total_output_logprobs
            ]
        return ChoiceLogprobs(
            content=self._process_logprobs_tokens(
                token_logprobs=output_token_logprobs,
                top_logprobs=output_top_logprobs,
            )
        )

    def _new_streaming_reasoning_parser(
        self, request: ChatCompletionRequest
    ) -> ReasoningParser:
        return ReasoningParser(
            self.reasoning_parser,
            request.stream_reasoning,
            self.template_manager.force_reasoning
            or self._get_reasoning_from_request(request),
            request,
            tokenizer=self.tokenizer_manager.tokenizer,
            canonical_streaming=(
                request.tool_choice != "none"
                and bool(self._collect_tools(request))
                and bool(
                    self.tool_call_parser
                    or request.tool_choice == "required"
                    or isinstance(request.tool_choice, ToolChoice)
                )
            ),
        )

    def _process_reasoning_stream(
        self,
        index: int,
        delta: str,
        reasoning_parser_dict: Dict[int, ReasoningParser],
        content: Dict[str, Any],
        request: ChatCompletionRequest,
        finish_reason_type: Optional[str] = None,
    ) -> tuple[Optional[str], str]:
        """Process reasoning content in streaming response"""
        if index not in reasoning_parser_dict:
            reasoning_parser_dict[index] = self._new_streaming_reasoning_parser(request)
            if request.continuation_token_ids:
                replay_text = self.tokenizer_manager.tokenizer.decode(
                    request.continuation_token_ids,
                    skip_special_tokens=False,
                    spaces_between_special_tokens=(
                        request.chat_template_kwargs or {}
                    ).get("spaces_between_special_tokens", True),
                )
                if replay_text:
                    # Restore the reasoning/content phase without re-emitting output.
                    reasoning_parser_dict[index].replay_continuation(replay_text)
        reasoning_parser = reasoning_parser_dict[index]
        reasoning_text, normal_text = reasoning_parser.parse_stream_chunk(delta)
        if (
            finish_reason_type is not None
            and finish_reason_type != "abort"
            and (finish_reason_type == "stop" or reasoning_parser._stream_input is None)
        ):
            end_reasoning_text, end_normal_text = reasoning_parser.parse_stream_end()
            if end_reasoning_text:
                reasoning_text = (reasoning_text or "") + end_reasoning_text
            if end_normal_text:
                normal_text = (normal_text or "") + end_normal_text
        return reasoning_text, normal_text

    def _get_history_tool_calls_cnt(self, request: ChatCompletionRequest) -> int:
        """Counts the number of tool calls in the request's message history.

        NOTE: This method is only useful for models that include self-increasing
        history tool call idx in tool calls id, such as kimi-k2. For kimi-k3 the
        result is the next global index: ``max(count, max_history_suffix + 1)``.

        Args:
            request: The chat completion request object.

        Returns:
            The total number of tool calls in the history, or 0 if not applicable.
        """
        messages = getattr(request, "messages", [])
        idx = 0
        max_suffix = -1
        for msg in messages:
            if msg.role == "assistant":
                tool_calls = getattr(msg, "tool_calls", None)
                tool_calls = list(tool_calls) if tool_calls is not None else []
                idx += len(tool_calls)  # noqa
                if self.tool_call_parser == "kimi_k3":
                    for tc in tool_calls:
                        m = _KIMI_K3_TOOL_CALL_ID_RE.match(tc.id or "")
                        if m:
                            max_suffix = max(max_suffix, int(m.group(1)))
        # Kimi-K3: truncated / compacted histories may keep sparse ids such as
        # `get_weather_2`/`get_weather_1`, so counting history calls can collide
        # with an existing id. Continue from the max suffix across all names.
        return max(idx, max_suffix + 1)

    def _patch_reasoning_skip_special_tokens(
        self, request: ChatCompletionRequest
    ) -> None:
        """Keep parser-specific reasoning markers in the decoded text.

        Some reasoning parsers rely on special-token delimiters that would be
        removed during detokenization when ``skip_special_tokens=True``.
        """
        if self.reasoning_parser == "apertus2509":
            request.skip_special_tokens = False

        if self.reasoning_parser == "kimi_k3" or self.chat_encoding_spec == "kimi_k3":
            request.skip_special_tokens = False

        if (
            self.reasoning_parser in ["mistral"]
            and request.reasoning_effort is not None
            and request.reasoning_effort != "none"
        ):
            request.skip_special_tokens = False
        elif self.reasoning_parser == "inkling":
            request.skip_special_tokens = False

    def wrap_reasoning_history(self, reasoning_text: str) -> str:
        """Wrap prior-turn reasoning in the detector's own start/end tokens.

        Pulling the delimiters from the detector keeps adapters in lockstep
        with any future parser that ships non-``<think>`` markers — Mistral's
        ``[THINK]``, Gemma4's ``think_start_self_label = "thought\\n"``, etc.
        Falling back to a plain string is unsafe: it would let prior
        thinking text reach a non-reasoning model as ordinary assistant
        content, so the caller must surface this state, not paper over it.
        """
        if self._reasoning_detector is None:
            raise ValueError(
                "Cannot rewrap thinking history: no reasoning detector is "
                "configured for this model"
            )
        d = self._reasoning_detector
        return (
            f"{d.think_start_token}{d.think_start_self_label}"
            f"{reasoning_text}\n{d.think_end_token}"
        )

    def _reasoning_default_mode(self) -> Optional[str]:
        if self._reasoning_detector is None:
            return None
        return self._reasoning_detector.reasoning_default

    def _get_reasoning_toggle_param(self) -> Optional[str]:
        """Resolve the chat-template kwarg that toggles reasoning, if any."""
        config = self.template_manager.reasoning_config
        if config is not None:
            return config.toggle_param

        mode = self._reasoning_default_mode()
        if mode in ("thinking", "enable_thinking"):
            return mode
        if mode in ("explicit_thinking", "explicit_enable_thinking"):
            return mode.replace("explicit_", "")
        return None

    def apply_reasoning_enabled(
        self, request: ChatCompletionRequest, enabled: bool
    ) -> None:
        """Force the request into the requested reasoning-on/off mode.

        Mirrors the read-side logic in ``_get_reasoning_from_request``;
        the two must stay in sync. Always-on models cannot be disabled,
        so explicit ``enabled=False`` raises rather than silently leaving
        reasoning on.
        """
        if not self.reasoning_parser:
            if enabled:
                raise ValueError(
                    "Anthropic thinking is not supported for models without "
                    "a reasoning parser"
                )
            return

        if self.reasoning_parser == "hunyuan":
            request.reasoning_effort = "medium" if enabled else "no_think"
            return

        config = self.template_manager.reasoning_config
        is_mistral = (config is not None and config.special_case == "mistral") or (
            config is None and self._reasoning_default_mode() == "mistral"
        )
        if is_mistral:
            request.reasoning_effort = "medium" if enabled else "none"
            return

        is_always_on = (config is not None and config.special_case == "always") or (
            config is None and self._reasoning_default_mode() == "always"
        )
        if is_always_on:
            if not enabled:
                raise ValueError(
                    f"Reasoning parser '{self.reasoning_parser}' is always-on "
                    f"and cannot be disabled via Anthropic thinking"
                )
            return

        toggle_param = self._get_reasoning_toggle_param()
        # The read side (``_get_reasoning_from_request``) returns False
        # whenever ``config.toggle_param is None`` OR
        # ``config.default_enabled is None``. The write side must mirror
        # both conditions: if ``default_enabled`` is unset we cannot
        # actually honor an ``enabled=True`` request even when the toggle
        # name itself is resolvable, so writing the kwarg would set up the
        # template to emit reasoning tokens while the parser ignores them
        # (literal ``<think>`` markers leak into the assistant text).
        config = self.template_manager.reasoning_config
        read_side_supported = toggle_param is not None and (
            config is None or config.default_enabled is not None
        )
        if not read_side_supported:
            if not enabled:
                return
            raise ValueError(
                f"Anthropic thinking is not supported for reasoning parser "
                f"'{self.reasoning_parser}'"
            )

        chat_template_kwargs = dict(request.chat_template_kwargs or {})
        chat_template_kwargs[toggle_param] = enabled
        request.chat_template_kwargs = chat_template_kwargs

    def _get_reasoning_from_request(self, request: ChatCompletionRequest) -> bool:
        """Determine whether reasoning mode should be enabled for this request.

        NOTE: This is predefined based on model's chat template
        """
        if not self.reasoning_parser:
            return False

        if self.reasoning_parser == "minimax-m3":
            # M3 template prefills <mm:think> for thinking_mode=enabled, so it never
            # appears in output and reasoning must be forced. Mirrors reasoning_parser.py.
            return (request.chat_template_kwargs or {}).get(
                "thinking_mode"
            ) == "enabled"

        if self.reasoning_parser == "hunyuan":
            # Hy3-preview template emits no <think> when reasoning_effort is
            # "no_think" / "none" / unset; forcing reasoning would route all
            # output into reasoning_content.
            return request.reasoning_effort not in (None, "none", "no_think")

        config = self.template_manager.reasoning_config
        if config is None:
            # Fallback to parser-level defaults when template toggle config
            # cannot be inferred (e.g., parser-only <think> templates).
            mode = (
                self._reasoning_detector.reasoning_default
                if self._reasoning_detector is not None
                else None
            )
            if mode is None:
                return False
            if mode == "always":
                return True
            if mode == "mistral":
                return (
                    request.reasoning_effort is not None
                    and request.reasoning_effort != "none"
                )
            if mode in ("thinking", "enable_thinking"):
                return (
                    not request.chat_template_kwargs
                    or request.chat_template_kwargs.get(mode) is not False
                )
            if mode in ("explicit_thinking", "explicit_enable_thinking"):
                toggle = mode.replace("explicit_", "")
                return (
                    request.chat_template_kwargs is not None
                    and request.chat_template_kwargs.get(toggle) is True
                )
            logger.warning(
                "Unknown reasoning_default mode '%s', defaulting to reasoning disabled",
                mode,
            )
            return False

        if config.special_case == "always":
            return True

        if config.special_case == "mistral":
            return (
                request.reasoning_effort is not None
                and request.reasoning_effort != "none"
            )

        if config.toggle_param is None or config.default_enabled is None:
            return False

        if config.default_enabled:
            return (
                not request.chat_template_kwargs
                or request.chat_template_kwargs.get(config.toggle_param) is not False
            )
        return (
            request.chat_template_kwargs is not None
            and request.chat_template_kwargs.get(config.toggle_param) is True
        )

    def _build_native_required_tool_parser(
        self,
        tools: List[Tool],
        request: ChatCompletionRequest,
        *,
        canonical_streaming: bool = False,
    ) -> Optional[FunctionCallParser]:
        """Build the parser when required/named output uses model-native syntax."""
        if not self.tool_call_parser:
            return None

        parser = FunctionCallParser(
            tools=tools,
            tool_call_parser=self.tool_call_parser,
            tool_choice=request.tool_choice,
            parallel_tool_calls=request.parallel_tool_calls,
            tokenizer=self.tokenizer_manager.tokenizer,
            canonical_streaming=canonical_streaming,
        )
        if parser.detector.parses_required_natively():
            return parser

        # K3 emits native XTML for required/named calls only when its strict
        # structural-tag constraint is active. Merely advertising structural
        # tag support is insufficient: with strictness off the constraint and
        # output remain generic JSON.
        constraint = parser.get_structure_constraint()
        if (
            self.tool_call_parser == "kimi_k3"
            and constraint is not None
            and constraint[0] == "structural_tag"
        ):
            return parser
        return None

    async def _process_tool_call_stream(
        self,
        index: int,
        delta: str,
        parser_dict: Dict[int, Union[FunctionCallParser, JsonArrayParser]],
        content: Dict[str, Any],
        request: ChatCompletionRequest,
        has_tool_calls: Dict[int, bool],
        continuous_usage_stats: bool = False,
        tool_call_last_return_time: float = 0.0,
        finish_reason_type: Optional[str] = None,
    ):
        """Process tool calls in streaming response"""
        all_tools = self._collect_tools(request) or []
        if index not in parser_dict:
            is_required = request.tool_choice == "required" or isinstance(
                request.tool_choice, ToolChoice
            )
            # For required/named tool choice: use JsonArrayParser when the
            # constrained output is plain JSON (detector doesn't support
            # structural_tag or no parser configured). Use FunctionCallParser
            # only when the detector supports structural_tag and will produce
            # native format output.
            if is_required:
                native_parser = self._build_native_required_tool_parser(
                    all_tools, request, canonical_streaming=True
                )
                if native_parser is not None:
                    parser_dict[index] = native_parser
                else:
                    # Tools are needed so replay_continuation can restore parser
                    # state; without them new deltas fall through as plain text.
                    parser_dict[index] = JsonArrayParser(
                        tools=all_tools,
                        canonical_streaming=True,
                        tokenizer=self.tokenizer_manager.tokenizer,
                    )
            else:
                parser_dict[index] = FunctionCallParser(
                    tool_call_parser=self.tool_call_parser,
                    tools=all_tools,
                    tool_choice=request.tool_choice,
                    parallel_tool_calls=request.parallel_tool_calls,
                    tokenizer=self.tokenizer_manager.tokenizer,
                    canonical_streaming=True,
                )

            # Restore detector state for continuation requests.
            if request.continuation_token_ids:
                use_reasoning = (
                    self.reasoning_parser
                    and request.separate_reasoning
                    and not self.mocked_disable_reasoning_parser
                )
                # Match live detokenization: _process_messages retains special
                # tokens for both native and generic tool-call output.
                replay_text = self.tokenizer_manager.tokenizer.decode(
                    request.continuation_token_ids,
                    skip_special_tokens=False,
                    spaces_between_special_tokens=(
                        request.chat_template_kwargs or {}
                    ).get("spaces_between_special_tokens", True),
                )
                # Replay the same streaming reasoning -> tool pipeline used by
                # live output. Non-stream reasoning parsing flushes/strips a
                # different prefix, especially when a marker crosses requests.
                if use_reasoning:
                    rp = self._new_streaming_reasoning_parser(request)
                    _, replay_text = rp.parse_stream_chunk(replay_text)
                if replay_text:
                    import asyncio

                    if await asyncio.to_thread(
                        parser_dict[index].replay_continuation, replay_text
                    ):
                        # Replayed calls still determine the finish reason.
                        has_tool_calls[index] = True

        parser = parser_dict[index]
        if finish_reason_type is None:
            finish_reason = content["meta_info"].get("finish_reason") or {}
            finish_reason_type = finish_reason.get("type")
        final = finish_reason_type == "stop"

        # Handle both FunctionCallParser and JsonArrayParser
        if isinstance(parser, JsonArrayParser):
            result = parser.parse_streaming_increment(delta, all_tools, final=final)
            normal_text, calls = result.normal_text, result.calls
        else:
            normal_text, calls = parser.parse_stream_chunk(delta, final=final)

        has_yielded = False
        # Yield normal text
        if normal_text:
            has_yielded = True
            choice_data = ChatCompletionResponseStreamChoice(
                index=index,
                delta=DeltaMessage(content=normal_text),
                finish_reason=None,
            )
            chunk = ChatCompletionStreamResponse(
                id=content["meta_info"]["id"],
                created=int(time.time()),
                choices=[choice_data],
                model=request.model,
            )

            # Add usage stats if continuous_usage_stats is enabled
            if continuous_usage_stats:
                prompt_tokens = self._reported_prompt_tokens(content["meta_info"])
                completion_tokens = content["meta_info"].get("completion_tokens", 0)
                reasoning_tokens = content["meta_info"].get("reasoning_tokens", 0)
                chunk.usage = UsageProcessor.calculate_token_usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    reasoning_tokens=reasoning_tokens,
                    cached_tokens=self._continuous_usage_cached_details(content),
                )

            yield f"data: {chunk.model_dump_json()}\n\n"

        # Yield tool calls
        history_tool_calls_cnt = self._get_history_tool_calls_cnt(request)
        for call_item in calls:
            has_yielded = True
            # Mark that this choice has tool calls
            has_tool_calls[index] = True

            # Tool call ID should be generated only once per tool call
            if call_item.name:
                # First chunk: include ID and function name
                tool_call_id = self._process_tool_call_id(
                    call_item, history_tool_calls_cnt
                )
                function_name = call_item.name
            else:
                # Subsequent chunks: null ID and name for argument deltas
                tool_call_id = None
                function_name = None

            tool_call = ToolCall(
                id=tool_call_id,
                index=call_item.tool_index,
                function=FunctionResponse(
                    name=function_name,
                    arguments=call_item.parameters,
                ),
            )

            choice_data = ChatCompletionResponseStreamChoice(
                index=index,
                delta=DeltaMessage(tool_calls=[tool_call]),
                finish_reason=None,
            )
            chunk = ChatCompletionStreamResponse(
                id=content["meta_info"]["id"],
                created=int(time.time()),
                choices=[choice_data],
                model=request.model,
            )

            # Add usage stats if continuous_usage_stats is enabled
            if continuous_usage_stats:
                prompt_tokens = self._reported_prompt_tokens(content["meta_info"])
                completion_tokens = content["meta_info"].get("completion_tokens", 0)
                reasoning_tokens = content["meta_info"].get("reasoning_tokens", 0)
                chunk.usage = UsageProcessor.calculate_token_usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    reasoning_tokens=reasoning_tokens,
                    cached_tokens=self._continuous_usage_cached_details(content),
                )

            yield f"data: {chunk.model_dump_json()}\n\n"

        # --- 核心修改：保活逻辑 ---
        # 如果模型输出了 delta (非空)，但解析器没吐出任何有效内容
        # 我们强制发一个空包防止 tpot 超时
        if (
            not has_yielded
            and delta
            and time.time() - tool_call_last_return_time
            >= self.tool_call_empty_return_interval
        ):
            logger.info(f"req: {request.rid} needs keep alive, yield an empty chunk")
            choice_data = ChatCompletionResponseStreamChoice(
                index=index,
                delta=DeltaMessage(content="", tool_calls=[]),
                finish_reason=None,
            )
            chunk = ChatCompletionStreamResponse(
                id=content["meta_info"]["id"],
                created=int(time.time()),
                choices=[choice_data],
                model=request.model,
            )

            if request.stream_options and request.stream_options.continuous_usage_stats:
                chunk.usage = UsageProcessor.calculate_token_usage(
                    prompt_tokens=content["meta_info"].get("prompt_tokens", 0),
                    completion_tokens=content["meta_info"].get("completion_tokens", 0),
                )

            yield f"data: {chunk.model_dump_json()}\n\n"

    def _check_for_unstreamed_tool_args(
        self,
        parser: Union[FunctionCallParser, JsonArrayParser],
        content: Dict[str, Any],
        request: ChatCompletionRequest,
        index: int,
    ) -> Optional[str]:
        """
        Check for any remaining tool call arguments that need to be streamed
        when generation finishes. This ensures tool calls are properly completed
        even if the model generates the final arguments in the last chunk.
        """
        finish_reason = content["meta_info"].get("finish_reason") or {}
        if finish_reason.get("type") in ("length", "abort"):
            # These are request boundaries, not the end of the logical tool
            # call. Synthesized quotes/braces cannot be undone by continuation.
            return None

        # Get the detector - either from FunctionCallParser or directly if json detector
        detector = parser.detector if hasattr(parser, "detector") else parser

        # Only check if we have tool calls and the detector has tracked data
        if (
            not hasattr(detector, "prev_tool_call_arr")
            or not detector.prev_tool_call_arr
        ):
            return None

        if (
            not hasattr(detector, "streamed_args_for_tool")
            or not detector.streamed_args_for_tool
        ):
            return None

        # Get the last tool call that was being processed
        tool_index = len(detector.prev_tool_call_arr) - 1
        if tool_index < 0 or tool_index >= len(detector.streamed_args_for_tool):
            return None

        # Get expected vs actual arguments
        expected_args = detector.prev_tool_call_arr[tool_index].get("arguments", {})
        actual_call = detector.streamed_args_for_tool[tool_index]

        # Some detectors retain the complete argument object as a JSON string.
        # Accept that legacy representation only when it decodes to an object.
        if isinstance(expected_args, str):
            try:
                expected_args = json.loads(expected_args)
            except (json.JSONDecodeError, ValueError):
                return None
        if not isinstance(expected_args, dict):
            return None

        # Parser hasn't started args; emitting "{}" here would corrupt continuation.
        if not actual_call:
            return None

        # Valid JSON already: any drift vs expected_args is an upstream parser bug;
        # supplementing would duplicate or corrupt the client stream.
        try:
            json.loads(actual_call)
            return None
        except (json.JSONDecodeError, ValueError):
            pass

        # Partial stream: streaming accumulates via json.dumps default separators,
        # match against those. Any other divergence is a parser bug — don't fabricate.
        remaining_call = ""
        expected_call = json.dumps(expected_args, ensure_ascii=False)
        if expected_call.startswith(actual_call):
            remaining_call = expected_call[len(actual_call) :]
        else:
            logger.warning(
                "FC unstreamed args skipped (rid=%s, tool_index=%d): "
                "streamed=%r expected=%r",
                request.rid,
                tool_index,
                actual_call,
                expected_args,
            )

        if remaining_call:
            # Create tool call chunk with remaining arguments
            tool_call = ToolCall(
                id=None,  # No ID for argument deltas
                index=tool_index,
                function=FunctionResponse(
                    name=None,  # No name for argument deltas
                    arguments=remaining_call,
                ),
            )

            choice_data = ChatCompletionResponseStreamChoice(
                index=index,
                delta=DeltaMessage(tool_calls=[tool_call]),
                finish_reason=None,  # Don't send finish_reason with this chunk
            )

            chunk = ChatCompletionStreamResponse(
                id=content["meta_info"]["id"],
                created=int(time.time()),
                choices=[choice_data],
                model=request.model,
            )

            return f"data: {chunk.model_dump_json()}\n\n"

        return None
