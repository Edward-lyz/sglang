"""State shared by the two-phase prefill admission path.

The normal scheduler seals an admission immediately.  The threaded prefill
path keeps the admission open while the previous forward is running so
requests that become ready during that interval can join the same batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Set


@dataclass
class PrefillAdmission:
    """A PrefillAdder and the state needed to seal it later."""

    running_batch: Any
    prefill_delayer_single_pass: Any = None
    adder: Any = None
    initial_candidates: List[Any] = field(default_factory=list)
    running_loras: Optional[Set] = None
    mamba_allocator: Any = None
    closed: bool = False
    offered: Set[str] = field(default_factory=set)
    as_admission_reclaimed_tokens: int = 0
