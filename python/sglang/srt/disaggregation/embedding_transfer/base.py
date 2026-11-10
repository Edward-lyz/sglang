"""
Shared types for the EPD embedding transfer stack.

Leaf module of the ``embedding_transfer`` package: dtype helpers, the request
key alias, the pull/poll data structures, the status enum (the EP analogue of
``KVPoll``) and the abstract transport interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import torch

# (request_id, task_id) — uniquely identifies a pull request across instances
RequestKey = Tuple[str, str]

# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------

_TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def parse_torch_dtype(dtype_str: str) -> torch.dtype:
    """Parse a dtype string to torch.dtype.

    Args:
        dtype_str: String representation of the dtype (e.g., "float16", "bfloat16").

    Returns:
        Corresponding torch.dtype, defaults to torch.float16 if not found.
    """
    return _TORCH_DTYPE_MAP.get(dtype_str, torch.float16)


# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------


class WaitingImageRequestStatus(IntEnum):
    """Status for WaitingImageRequest."""

    FAIL = -1
    PENDING = 0
    SUCCESS = 1
    TIMEOUT = -2


@dataclass
class PullTarget:
    """Target for WaitingImageRequest to pull embedding data."""

    task_id: str
    request_id: str
    transfer_type: str
    endpoint: str
    # Mooncake specific fields
    embedding_meta: Optional[Dict[str, Any]] = None
    image_url: Optional[str] = None
    modality: str = "image"
    chunk_timestamps: Optional[List[str]] = None
    sent: bool = False


@dataclass
class TransportResult:
    """Unified result returned by EmbeddingTransport.poll()."""

    task_id: str
    embedding: Optional[torch.Tensor] = None
    grid_thw: Optional[torch.Tensor] = None
    error_msg: Optional[str] = None
    error_code: Optional[int] = None
    buf_addr: Optional[int] = None

    @property
    def success(self) -> bool:
        """Check if the transport result was successful.

        Returns:
            True if error_msg is None, indicating success.
        """
        return self.error_msg is None


# ---------------------------------------------------------------------------
# Endpoint helper
# ---------------------------------------------------------------------------


def make_endpoint(encoder_addr: dict, scheme: str = "tcp") -> str:
    """Build endpoint URL from encoder_addr dict.

    Trusted internal call — no defensive validation on ip/port types.
    """
    return f"{scheme}://{encoder_addr['ip']}:{encoder_addr['http_port']}"


# ---------------------------------------------------------------------------
# Abstract transport interface
# ---------------------------------------------------------------------------


class EmbeddingTransport(ABC):
    """Abstract interface for pulling embedding data from encoder.

    Lifecycle:  send() -> poll() [repeated] -> cleanup()
    All methods must be non-blocking (safe to call from scheduler main loop).
    """

    @abstractmethod
    def send(self, pull_target: PullTarget) -> bool:
        """Send a pull request to encoder. Must be non-blocking.

        Returns True if the transfer was submitted, False if unable to
        submit now (e.g. buffer pool full). The caller is responsible
        for retrying later.
        """

    @abstractmethod
    def poll(self) -> List[TransportResult]:
        """Non-blocking poll for completed transfers. May return empty list."""

    @abstractmethod
    def cleanup(self) -> None:
        """Release all transport resources."""
