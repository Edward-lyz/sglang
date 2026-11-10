"""Content detection recorder module for tracking request chunks and context."""

from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Union
import logging
logger = logging.getLogger(__name__)


@dataclass
class ChunkInfo:
    """Information about a single output chunk."""

    text: str
    logprob: Optional[List[float]] = field(default_factory=list)
    token_ids: list[int] | None = None
    top_logprobs: Optional[List[Dict[str, float]]] = field(default_factory=list)
    timestamp: Optional[float] = None


@dataclass
class RequestRecorder:
    """Records chunks for a single request with a sliding window."""

    max_window_size: int = 4096
    chunk_list: List[ChunkInfo] = field(default_factory=list)

    def append(self, chunk: ChunkInfo) -> None:
        """Append a chunk and trim if exceeding max window size."""
        self.chunk_list.append(chunk)
        if len(self.chunk_list) > self.max_window_size:
            self.chunk_list = self.chunk_list[-self.max_window_size :]

    def get_recent_chunks(self, window: int) -> list[ChunkInfo]:
        """Return the most recent chunks up to the given window size."""
        return self.chunk_list[-window:]

    def get_recent_text(self, window: int = 512) -> str:
        """Return concatenated text from recent chunks."""
        chunk_list = self.get_recent_chunks(window)
        return "".join(chunk.text for chunk in chunk_list)


@dataclass
class RequestContext:
    """Context for a single request, managing recording, abort, and trace."""

    rid: Optional[Union[str, List[str]]]
    step: int = 0
    recorder: RequestRecorder = field(default_factory=RequestRecorder)
    metadata: dict[str, Any] = field(default_factory=dict)
    is_abort: bool = field(default=False)
    is_trace: bool = field(default=False)
    detection_task: Optional[asyncio.Task] = None
    tokenizer_manager: Optional[Any] = None
    trace_config: Optional[Dict[str, Any]] = None

    def record(self, chunk: ChunkInfo) -> None:
        """Record a chunk and increment step counter."""
        if self.is_abort:
            return
        if self.recorder is None:
            return
        self.recorder.append(chunk)
        self.step += 1

    def get_text(self, window: int) -> str:
        """Get recent text within the given window."""
        return self.recorder.get_recent_text(window)

    def get_chunks(self, window: int) -> list[ChunkInfo]:
        """Get recent chunks within the given window."""
        return self.recorder.get_recent_chunks(window)

    def abort_request(self, rid: Optional[Union[str, List[str]]] = None) -> None:
        """Abort the request via tokenizer_manager."""
        if self.is_abort:
            return
        self.is_abort = True
        target_rid = rid if rid is not None else self.rid
        logger.info(f"Abort request by content detection for rid: {target_rid}")
        if self.tokenizer_manager is None:
            return
        try:
            self.tokenizer_manager.abort_request(rid=target_rid)
        except Exception:
            logger.exception(f"abort request failed for rid={target_rid}")

    def trace_request(self, rid: Optional[Union[str, List[str]]] = None) -> None:
        """Trigger trace request via tokenizer_manager.

        Only sends the rid signal. Scheduler will parse trace_config from Req
        and assemble a full TraceReq with trace_type/dump_dir/trigger/payload.
        """
        if self.is_trace:
            return

        target_rid = rid if rid is not None else self.rid
        if self.tokenizer_manager is None:
            return
        self.is_trace = True
        try:
            self.tokenizer_manager.trace_request(rid=target_rid)
        except Exception:
            logger.exception(f"trace request failed for rid={target_rid}")
