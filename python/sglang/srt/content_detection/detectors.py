"""Content detection detectors for identifying problematic output patterns."""

from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, List, Dict, Iterable
from abc import ABC, abstractmethod
from .recorder import RequestContext, ChunkInfo
import logging

logger = logging.getLogger(__name__)


class DetectionAction(Enum):
    """Actions that can be taken when content is detected."""

    WARN = "warn"
    TRACE = "trace"
    ABORT = "abort"


@dataclass
class DetectionResult:
    """Result of a detection check."""

    detector_name: str
    triggered: bool
    message: str
    action: DetectionAction
    rid: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "rid": self.rid,
            "detector_name": self.detector_name,
            "triggered": self.triggered,
            "message": self.message,
            "action": self.action.value,
            "metadata": self.metadata,
        }


class BaseDetector(ABC):
    """Abstract base class for content detectors."""

    name = "base"
    priority = 0

    @abstractmethod
    def detect(
        self,
        ctx: RequestContext,
        window_chunks: list[ChunkInfo],
    ) -> DetectionResult | None:
        """Detect problematic content in the given chunks."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset detector state."""
        pass


class TokenRepeatDetector(BaseDetector):
    """Detector for repeated text patterns in output."""

    name = "token_repeat"
    priority = 100

    def __init__(
        self,
        min_pattern_len: int = 4,
        max_pattern_len: int = 64,
        pattern_repeat_times: int = 3,
    ):
        self.min_pattern_len = min_pattern_len
        self.max_pattern_len = max_pattern_len
        self.pattern_repeat_times = pattern_repeat_times

    def detect(
        self,
        ctx: RequestContext,
        window_chunks: list[ChunkInfo],
    ) -> DetectionResult | None:
        """Detect repeated text patterns in chunks."""
        text = "".join(chunk.text for chunk in window_chunks)
        if not text:
            return None

        pattern_result = self._detect_repeated_text_pattern(text)
        if pattern_result:
            pattern_result.rid = ctx.rid
            return pattern_result

        return None

    def _detect_repeated_text_pattern(
        self, text: str,
    ) -> DetectionResult | None:
        compact_text = "".join(text.split())
        if len(compact_text) < self.min_pattern_len * self.pattern_repeat_times:
            return None

        max_pattern_len = min(
            self.max_pattern_len, len(compact_text) // self.pattern_repeat_times
        )
        for pattern_len in range(self.min_pattern_len, max_pattern_len + 1):
            repeat_count, pattern = self._suffix_repeat(compact_text, pattern_len)
            if repeat_count >= self.pattern_repeat_times:
                return DetectionResult(
                    detector_name=self.name,
                    triggered=True,
                    message=f"text pattern {pattern!r} repeated {repeat_count} times",
                    action=DetectionAction.WARN,
                    metadata={"repeat_count": repeat_count, "pattern": pattern},
                )
        return None

    def _suffix_repeat(self, text: str, pattern_len: int) -> tuple[int, str]:
        pattern = text[-pattern_len:]
        if len(set(pattern)) == 1:
            return 0, pattern

        repeat_count = 0
        cursor = len(text)
        while cursor >= pattern_len and text[cursor - pattern_len : cursor] == pattern:
            repeat_count += 1
            cursor -= pattern_len
        return repeat_count, pattern


class FixedPatternDetector(BaseDetector):
    """Detector for fixed string patterns in output."""

    name = "fixed_pattern"
    priority = 200

    def __init__(
        self,
        patterns: Iterable[str] = None,
        action: DetectionAction = DetectionAction.WARN,
    ):
        self.patterns = [pattern for pattern in patterns if pattern] if patterns else []
        self.action = action

    def detect(
        self,
        ctx: RequestContext,
        window_chunks: list[ChunkInfo],
    ) -> DetectionResult | None:
        """Detect fixed patterns in chunk text."""

        text = "".join(chunk.text for chunk in window_chunks)
        trace_config = ctx.trace_config or {}
        patterns = trace_config.get("fixed_patterns", None) or self.patterns

        if not text or not patterns:
            return None

        for pattern in patterns:
            if pattern in text:
                return DetectionResult(
                    detector_name=self.name,
                    triggered=True,
                    message=f"fixed pattern {pattern!r} matched",
                    action=self.action,
                    rid=ctx.rid,
                    metadata={"pattern": pattern},
                )
        return None


class ContentDetector:
    """Orchestrates multiple detectors to check content."""

    def __init__(self, detectors: list[BaseDetector]):
        self.detectors = sorted(
            detectors, key=lambda detector: detector.priority, reverse=True
        )

    async def detect(
        self,
        ctx: RequestContext,
        window_chunks: list[ChunkInfo],
    ) -> list[DetectionResult]:
        """Run all enabled detectors concurrently and return triggered results."""
        trace_config = ctx.trace_config or {}
        allowed_detectors = trace_config.get("detectors", None)
        detectors = self.detectors
        if allowed_detectors is not None:
            allowed_set = set(allowed_detectors)
            detectors = [d for d in detectors if d.name in allowed_set]

        detector_tasks = [
            asyncio.create_task(asyncio.to_thread(detector.detect, ctx, window_chunks))
            for detector in detectors
        ]
        if not detector_tasks:
            return []

        task_results = await asyncio.gather(*detector_tasks, return_exceptions=True)
        results: list[DetectionResult] = []
        for detector, result in zip(detectors, task_results):
            if isinstance(result, Exception):
                logger.exception(
                    f"detector {detector.__class__.__name__} detection failed",
                    exc_info=result,
                )
                continue
            if result is not None:
                logger.info(
                    f"detector {detector.__class__.__name__} detection result "
                    f"{result.to_dict() if result else None}"
                )
            if result and result.triggered:
                results.append(result)
        return results
