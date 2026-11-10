"""SSE chunk building utilities for OpenAI chat completions streaming."""

from __future__ import annotations

from typing import List, Optional, Union

import msgspec

_SSE_DATA_B = b"data: "
_SSE_NL_B = b"\n\n"


class StreamDelta(msgspec.Struct, omit_defaults=True):
    """Delta content for streaming responses.

    OpenAI Python SDK's ChoiceDelta does not declare reasoning_content; it is
    surfaced via pydantic `extra`. With omit_defaults=True, defaulting to
    None would drop the key entirely from the SSE payload, making
    `data.reasoning_content` raise AttributeError on the client. Keep it
    required (no default) so it is always serialized as null or a string.
    """

    reasoning_content: Optional[str]
    # Number of generated tokens carried by this chunk. Restores the pre-0.5.17
    # contract used by the gateway to count speculative (MTP) steps, where one
    # chunk may cover several tokens. Required (no default, must stay before
    # the optional fields) so the key is always present, matching the old
    # pydantic payload where it was null when unknown.
    tokens: Optional[int]
    role: Optional[str] = None
    content: Optional[str] = None


class StreamChoice(msgspec.Struct):
    """A single choice in a streaming response."""

    index: int
    delta: StreamDelta
    logprobs: Optional[dict] = None
    finish_reason: Optional[str] = None
    matched_stop: Union[None, int, str] = None
    # incremental; excludes continuation_token_ids prefix
    output_token_ids: Optional[List[int]] = None


class StreamChunk(msgspec.Struct, omit_defaults=True):
    """A complete streaming chunk."""

    id: str
    object: str
    created: int
    model: str
    choices: List[StreamChoice]
    usage: Optional[dict] = None


_stream_encoder = msgspec.json.Encoder()


def build_sse_content(
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
    output_token_ids: Optional[List[int]] = None,
    tokens: Optional[int] = None,
) -> str:
    """Build an SSE chunk string for content/reasoning updates.

    Args:
        chunk_id: Request ID for this chunk
        created: Unix timestamp
        model: Model name
        index: Choice index
        role: Message role (usually "assistant")
        content: Text content delta
        reasoning_content: Reasoning/thinking content delta
        finish_reason: Finish reason if done
        logprobs: Log probabilities if requested
        matched_stop: Stop token/string that was matched
        usage: Token usage statistics
        output_token_ids: Incremental output token ids (continuation)
        tokens: Number of generated tokens carried by this delta

    Returns:
        SSE-formatted string "data: {...}\\n\\n"
    """
    # The aiak gateway's SSE parser drops the connection when a delta omits
    # `content`, which omit_defaults does whenever it is None. Keep the key
    # present on every chunk, the same way reasoning_content always is:
    # reasoning-only deltas, the logprobs/usage-only chunk and the trailing
    # finish_reason chunk all carried no content before.
    if content is None:
        content = ""
    delta = StreamDelta(
        role=role,
        content=content,
        reasoning_content=reasoning_content,
        tokens=tokens,
    )
    choice = StreamChoice(
        index=index,
        delta=delta,
        logprobs=logprobs,
        finish_reason=finish_reason,
        matched_stop=matched_stop,
        output_token_ids=output_token_ids,
    )
    chunk = StreamChunk(
        id=chunk_id,
        object="chat.completion.chunk",
        created=created,
        model=model,
        choices=[choice],
        usage=usage,
    )
    return (_SSE_DATA_B + _stream_encoder.encode(chunk) + _SSE_NL_B).decode()
