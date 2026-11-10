"""
Encoder health check and probe utilities.

This leaf module consolidates all health-state tracking, probe-image
generation, and idle-probe helpers that were previously scattered across
encode_server.py and encode_fetcher.py.  It has no imports from either
of those modules, so it can be safely imported by both without creating
circular dependencies.
"""

import asyncio
import base64
import io
import logging
import threading
import time

from PIL import Image

logger = logging.getLogger(__name__)


# =============================================================================
# Probe image & constants
# =============================================================================

def _generate_probe_image_base64() -> str:
    """Generate a 14x14 probe image as a base64-encoded data URI."""
    img = Image.new("RGB", (14, 14), color=(128, 128, 128))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{base64_data}"


PROBE_IMAGE_BASE64: str = _generate_probe_image_base64()
PROBE_REQ_ID_PREFIX: str = "ENCODER_PROBE_"
PROBE_TRIGGER_THRESHOLD: float = 60.0  # seconds without output before probing
HEALTH_CHECK_TIMEOUT: float = 30.0     # max wait for a probe response


def generate_probe_req_id() -> str:
    """Generate a unique probe request ID."""
    return f"{PROBE_REQ_ID_PREFIX}{time.time()}"


# =============================================================================
# Encoder health state
# =============================================================================

class EncoderHealthState:
    """Track the last time the encoder produced output."""

    def __init__(self):
        self.last_encoder_output_tstamp: float = 0.0

    def update_encoder_output(self):
        """Update the timestamp of last encoder output."""
        self.last_encoder_output_tstamp = time.time()

    def has_output_since(self, tic: float) -> bool:
        """Check if encoder has produced output since the given time."""
        return self.last_encoder_output_tstamp > tic


encoder_health_state: EncoderHealthState = EncoderHealthState()


def update_encoder_output_timestamp():
    """Update the timestamp of last encoder output."""
    encoder_health_state.update_encoder_output()


def get_encoder_output_timestamp() -> float:
    """Get the timestamp of last encoder output."""
    return encoder_health_state.last_encoder_output_tstamp


# =============================================================================
# Transfer health state
# =============================================================================

TRANSFER_FAILURE_THRESHOLD: float = 120.0  # seconds of continuous failure → unhealthy


class _SessionTransferState:
    """Per-session transfer health tracker."""

    def __init__(self):
        self.last_attempt: float = 0.0
        self.last_success: float = 0.0

    def record_attempt(self):
        """Record a transfer attempt."""
        self.last_attempt = time.time()

    def record_success(self):
        """Record a successful transfer."""
        now = time.time()
        self.last_attempt = now
        self.last_success = now

    def is_healthy(self) -> bool:
        """Check if this session's transfer is healthy."""
        if self.last_attempt == 0.0:
            return True  # idle
        if self.last_success >= self.last_attempt:
            return True  # last attempt succeeded
        # No new attempt for longer than threshold → treat as idle / healthy
        if (time.time() - self.last_attempt) >= TRANSFER_FAILURE_THRESHOLD:
            return True
        # Failing — check how long since last success
        return (time.time() - self.last_success) < TRANSFER_FAILURE_THRESHOLD


class TransferHealthState:
    """Per-session transfer health state.

    Each session_id gets its own _SessionTransferState bucket.  Global health
    is True as long as at least one session is healthy (or no sessions exist).
    This prevents a single bad endpoint from masking others, and prevents a
    single bad endpoint from causing the whole encoder to report unhealthy.

    Thread-safe: all mutations are protected by _lock.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: dict[str, _SessionTransferState] = {}

    def _get_or_create(self, session_id: str) -> _SessionTransferState:
        s = self._sessions.get(session_id)
        if s is None:
            s = _SessionTransferState()
            self._sessions[session_id] = s
        return s

    def record_attempt(self, session_id: str = ""):
        """Record a transfer attempt for the given session."""
        with self._lock:
            self._get_or_create(session_id).record_attempt()

    def record_success(self, session_id: str = ""):
        """Record a successful transfer for the given session."""
        with self._lock:
            self._get_or_create(session_id).record_success()

    def is_healthy(self) -> bool:
        """Return True if the encoder-level transfer is healthy.

        Healthy when:
          - No sessions have been seen yet (idle)
          - At least one active session is still healthy

        A single failing session does NOT make the encoder unhealthy as long
        as other sessions continue to work.  All sessions failing → unhealthy.
        """
        with self._lock:
            if not self._sessions:
                return True  # idle
            return any(s.is_healthy() for s in self._sessions.values())

    def unhealthy_sessions(self) -> list[str]:
        """Return list of session_ids that are currently unhealthy."""
        with self._lock:
            return [sid for sid, s in self._sessions.items() if not s.is_healthy()]

    def snapshot(self) -> dict[str, dict[str, float | bool]]:
        """Return a debug snapshot of all per-session transfer states."""
        with self._lock:
            return {
                sid: {
                    "last_attempt": s.last_attempt,
                    "last_success": s.last_success,
                    "healthy": s.is_healthy(),
                }
                for sid, s in self._sessions.items()
            }

    # Legacy no-arg compatibility (used when session_id is not known at call site)
    @property
    def last_transfer_attempt(self) -> float:
        """Return the latest transfer attempt timestamp across all sessions."""
        with self._lock:
            if not self._sessions:
                return 0.0
            return max(s.last_attempt for s in self._sessions.values())

    @property
    def last_transfer_success(self) -> float:
        """Return the latest successful transfer timestamp across all sessions."""
        with self._lock:
            if not self._sessions:
                return 0.0
            return max(s.last_success for s in self._sessions.values())


transfer_health_state: TransferHealthState = TransferHealthState()


def record_transfer_attempt(session_id: str = ""):
    """Record a transfer attempt for the given session."""
    transfer_health_state.record_attempt(session_id)


def record_transfer_success(session_id: str = ""):
    """Record a successful transfer for the given session."""
    transfer_health_state.record_success(session_id)


# =============================================================================
# Probe request
# =============================================================================

async def send_probe_request(encoder) -> bool:
    """Send a minimal probe image through the full encode pipeline.

    Args:
        encoder: MMEncoder instance to probe.

    Returns:
        True if the probe completed successfully, False otherwise.
    """
    if encoder is None:
        return False

    req_id = generate_probe_req_id()
    try:
        logger.debug(f"Sending encoder probe request: req_id={req_id}")
        await asyncio.wait_for(
            encoder.encode(
                mm_items=[PROBE_IMAGE_BASE64],
                req_id=req_id,
                num_parts=1,
                part_idx=0,
                task_id=req_id,
                skip_cache=True,
            ),
            timeout=HEALTH_CHECK_TIMEOUT,
        )
        update_encoder_output_timestamp()
        logger.info("Encoder probe: 200 OK")
        return True
    except asyncio.TimeoutError:
        logger.error(f"Encoder probe timeout: waited {HEALTH_CHECK_TIMEOUT}s")
        return False
    except Exception as e:
        logger.error(f"Encoder probe failed: {e}")
        return False
