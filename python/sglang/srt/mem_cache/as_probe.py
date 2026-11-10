"""Lightweight AS instrumentation.

Set ``AS_PROBE=1`` to emit host-side timing and counters. The default path is
disabled and does not synchronize CUDA or change transfer ordering.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict

ENABLED = os.getenv("AS_PROBE") == "1"

logger = logging.getLogger(__name__)
_lock = threading.Lock()
_counters = defaultdict(float)


def now() -> float:
    """返回主机侧单调时钟，供AS阶段计时。"""
    return time.perf_counter()


def ms(start: float, end: float | None = None) -> float:
    """Return elapsed host time in milliseconds."""
    return ((time.perf_counter() if end is None else end) - start) * 1000.0


def add(key: str, value: float = 1.0) -> None:
    """采集开启时累加计数，默认关闭时直接返回。"""
    if not ENABLED:
        return
    with _lock:
        _counters[key] += value


def take() -> dict:
    """Return and clear all accumulated counters."""
    if not ENABLED:
        return {}
    with _lock:
        snapshot = dict(_counters)
        _counters.clear()
    return snapshot


def _format(value) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_format(item) for item in value) + "]"
    return str(value)


def emit(tag: str, **fields) -> None:
    """采集开启时记录阶段字段及当前线程，不触发CUDA同步。"""
    if not ENABLED:
        return
    body = " ".join(f"{key}={_format(value)}" for key, value in fields.items())
    logger.info(
        "[ASPROBE] %s thread=%s %s",
        tag,
        threading.current_thread().name,
        body,
    )
