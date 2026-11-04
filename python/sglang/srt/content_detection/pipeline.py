"""Content detection pipeline that orchestrates scheduling, detection, and action execution."""

from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Dict
from .config import DETECTION_CONFIG
from .recorder import RequestContext, ChunkInfo
from sglang.srt.server_args import (
    ServerArgs,
)
from .detectors import (
    DetectionResult,
    DetectionAction,
    ContentDetector,
    FixedPatternDetector,
    TokenRepeatDetector,
)
import logging

logger = logging.getLogger(__name__)


@dataclass
class DetectionScheduler:
    """Scheduler that determines when and how to trigger detection."""

    interval: int = 128
    window: int = 128

    def should_detect(self, ctx: RequestContext, chunk: ChunkInfo | None = None) -> bool:
        """Determine whether detection should run at the current step."""
        if chunk is None:
            return False
        step = ctx.step
        if step <= 0:
            return False
        return step % self.interval == 0

    def detection_window(self, ctx: RequestContext) -> int:
        """Return the detection window size."""
        return self.window


class DetectionActionExecutor:
    """Executes actions based on detection results."""

    def _resolve_action(
        self,
        ctx: RequestContext,
        action: DetectionAction,
    ) -> DetectionAction:
        trace_config = ctx.trace_config or {}
        config_action = trace_config.get("action", None)
        if config_action is None:
            return action

        if isinstance(config_action, DetectionAction):
            return config_action

        if isinstance(config_action, list):
            if action.value in config_action:
                return DetectionAction.TRACE
            return action

        if config_action == DetectionAction.TRACE.value:
            return DetectionAction.TRACE
        if config_action == DetectionAction.ABORT.value:
            return DetectionAction.ABORT
        if config_action == DetectionAction.WARN.value:
            return DetectionAction.WARN
        return action

    async def execute(
        self,
        ctx: RequestContext,
        result: DetectionResult,
    ):
        """Execute the resolved action for a detection result."""
        ctx.metadata.setdefault("detections", []).append(result.to_dict())
        action = self._resolve_action(ctx, result.action)

        if action == DetectionAction.WARN:
            logger.warning(f"req {ctx.rid} detected by {result.detector_name} return message {result.message}")

        if action == DetectionAction.TRACE:
            ctx.trace_request(ctx.rid)

        if action == DetectionAction.ABORT:
            ctx.abort_request(ctx.rid)


class DetectionPipeline:
    """Main pipeline that processes chunks through detection and action execution."""

    def __init__(
        self,
        scheduler: DetectionScheduler,
        detector: ContentDetector,
        executor: DetectionActionExecutor,
    ):
        self.scheduler = scheduler
        self.detector = detector
        self.executor = executor

    async def process(self, ctx: RequestContext, chunk: ChunkInfo):
        """Process a chunk through the detection pipeline.

        Trace is checked before detection so explicit trace requests still work without
        recording chunks when detection is disabled.
        """
        if ctx.is_abort:
            return

        ctx.record(chunk)
        if not self.scheduler.should_detect(ctx, chunk):
            return

        if ctx.detection_task is not None and not ctx.detection_task.done():
            return

        window_chunks = ctx.get_chunks(
            self.scheduler.detection_window(ctx),
        )
        ctx.detection_task = asyncio.create_task(
            self._run_detection(ctx, window_chunks)
        )

    async def _run_detection(
        self,
        ctx: RequestContext,
        window_chunks: list[ChunkInfo],
    ) -> None:
        try:
            results = await self.detector.detect(ctx, window_chunks)
            for result in results:
                await self.executor.execute(ctx, result)
        except Exception:
            logger.exception(f"req {ctx.rid} detection pipeline failed")


def build_detection_pipeline(
    server_args: ServerArgs,
) -> DetectionPipeline:
    """Build the default detection pipeline from server arguments and config."""
    enabled_detectors = set(
        [
            FixedPatternDetector.name,
            TokenRepeatDetector.name,
        ]
    )
    detectors = []
    if FixedPatternDetector.name in enabled_detectors:
        detectors.append(
            FixedPatternDetector()
        )
    if TokenRepeatDetector.name in enabled_detectors:
        detectors.append(TokenRepeatDetector(
            min_pattern_len=DETECTION_CONFIG.tr_min_pattern_len,
            max_pattern_len=DETECTION_CONFIG.tr_max_pattern_len,
            pattern_repeat_times=DETECTION_CONFIG.tr_repeat_times,
        ))

    return DetectionPipeline(
        scheduler=DetectionScheduler(
            interval=DETECTION_CONFIG.detection_interval,
            window=DETECTION_CONFIG.detection_window,
        ),
        detector=ContentDetector(detectors=detectors),
        executor=DetectionActionExecutor(),
    )
