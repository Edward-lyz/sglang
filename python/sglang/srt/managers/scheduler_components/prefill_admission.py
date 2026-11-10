"""State of one prefill admission that may stay open across scheduler polls."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Optional, Set

import msgspec


class PrefillAdmission(msgspec.Struct, kw_only=True):
    """One prefill batch under construction.

    ``adder`` is None when nothing could be admitted at open time. Continuous
    admission keeps offering newly ready requests until the batch is sealed;
    ``closed`` marks that the budget or a slot limit already stopped it.
    """

    running_batch: Any
    prefill_delayer_single_pass: Any = None
    adder: Any = None
    initial_candidates: Any = ()
    running_loras: Optional[Set] = None
    mamba_allocator: Any = None
    cost_scheduler: Any = None
    closed: bool = False
    # rids already offered to this admission.
    offered: Set[str] = msgspec.field(default_factory=set)
    # rids re-offered once after their load-back committed (bounds retries).
    reoffered: Set[str] = msgspec.field(default_factory=set)


@contextmanager
def clamped_chunk_budget(adder, cap: Optional[int]) -> Iterator[bool]:
    """Temporarily cap the adder's chunk budget; the withheld part is returned after.

    Yields whether the budget was actually clamped.
    """
    if cap is None or adder.rem_chunk_tokens is None or cap >= adder.rem_chunk_tokens:
        yield False
        return
    withheld = adder.rem_chunk_tokens - cap
    adder.rem_chunk_tokens = cap
    try:
        yield True
    finally:
        adder.rem_chunk_tokens += withheld
