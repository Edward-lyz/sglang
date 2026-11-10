"""Configuration module for content detection, reading parameters from environment variables."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name, "").lower()
    if v in ("1", "true", "yes"):
        return True
    if v in ("0", "false", "no"):
        return False
    return default


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


class ContentDetectionConfig:
    """Singleton that reads all content-detection parameters from env vars."""

    def __init__(self) -> None:
        self.reload()

    def reload(self) -> None:
        """Re-read every parameter from environment variables."""

        # --- Feature switch ---
        self.enable_detection: bool = _env_bool("CD_ENABLE_DETECTION", False)

        # --- DetectionScheduler ---
        self.detection_interval: int = _env_int("CD_DETECTION_INTERVAL", 64)
        self.detection_window: int = _env_int("CD_DETECTION_WINDOW", 128)

        # --- TokenRepeatDetector ---
        self.tr_min_pattern_len: int = _env_int("CD_TR_MIN_PATTERN_LEN", 4)
        self.tr_max_pattern_len: int = _env_int("CD_TR_MAX_PATTERN_LEN", 64)
        self.tr_repeat_times: int = _env_int("CD_TR_REPEAT_TIMES", 3)

        # --- Trace config ---
        self.enable_trace: bool = _env_bool("CD_ENABLE_TRACE", False)
        _dump_dir = _env_str("CD_TRACE_DUMP_DIR", "/workspace/sgl-trace")
        self.trace_dump_dir: Optional[str] = _dump_dir if _dump_dir else None

    def build_trace_config(self) -> Dict[str, Any]:
        """Build trace configuration dictionary from current settings."""
        cfg: Dict[str, Any] = {
            "enable_trace": self.enable_trace,
            "trace_type": None,
            "trigger": "on_finish",
            "action": "warn",
            "fixed_patterns": [],
            "detectors": None,
        }
        if self.enable_detection and self.enable_trace:
            cfg["action"] = "trace"
        if self.trace_dump_dir is not None:
            cfg["dump_dir"] = self.trace_dump_dir
        return cfg

    def __repr__(self) -> str:
        return (
            f"ContentDetectionConfig("
            f"detection={self.enable_detection}, "
            f"interval={self.detection_interval}, window={self.detection_window}, "
            f"tr=[min={self.tr_min_pattern_len}, max={self.tr_max_pattern_len}, "
            f"repeat={self.tr_repeat_times}], "
            f"trace={self.enable_trace}, "
            f"dump_dir={self.trace_dump_dir}"
        )


DETECTION_CONFIG = ContentDetectionConfig()
