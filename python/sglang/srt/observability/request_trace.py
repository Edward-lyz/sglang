"""Per-request stage trace, gated by ``SGLANG_LOG_REQUEST_TRACE``.

Each trace line names one request and one stage boundary, so a rid can be
followed across the api server, tokenizer, dp controller, scheduler and
detokenizer processes when a request stalls or disappears. The trace is off by
default: it costs one log record per request per stage. Health-check probes
never reach the log, since their synthetic rids would bury real traffic.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.environ import envs
from sglang.srt.utils.msgspec_utils import msgspec_to_builtins

# Grammar payloads carry a whole schema, which is too large for a trace line.
_GRAMMAR_SAMPLING_FIELDS = frozenset({"structural_tag", "ebnf", "regex", "json_schema"})


def _is_traceable(rid: Optional[str]) -> bool:
    return isinstance(rid, str) and not rid.startswith(HEALTH_CHECK_RID_PREFIX)


def traced_rid(rid: Optional[str]) -> Optional[str]:
    """The rid to name in a trace line, or None when it must not be traced."""
    if not envs.SGLANG_LOG_REQUEST_TRACE.get():
        return None
    return rid if _is_traceable(rid) else None


def traced_rids(rids: Iterable[Optional[str]]) -> List[str]:
    """The rids of one batch worth tracing; empty when the trace is off."""
    if not envs.SGLANG_LOG_REQUEST_TRACE.get():
        return []
    return [rid for rid in rids if _is_traceable(rid)]


def traced_sampling_params(sampling_params: Any) -> Dict[str, Any]:
    """Sampling params as plain values, without the grammar payloads."""
    return {
        key: value
        for key, value in msgspec_to_builtins(sampling_params).items()
        if key not in _GRAMMAR_SAMPLING_FIELDS
    }


def trace(logger: logging.Logger, message: str, *args: Any) -> None:
    """Emit one trace line. Callers gate on traced_rid / traced_rids first."""
    logger.info("[trace] " + message, *args)


def log_input_ids(logger: logging.Logger, rid: str, stage: str, input_ids) -> None:
    """Dump the ids one stage hands on, when SGLANG_LOG_TOKEN_IDS is set.

    The tokenizer's ids still carry the raw media placeholder token, one per
    grid position. The scheduler's ids carry each media item's pad value in
    those positions, which is what the model input and the prefix-cache key are
    built from, so a replay against a platform dump compares the scheduler ids.
    """
    if input_ids is None or not envs.SGLANG_LOG_TOKEN_IDS.get():
        return
    logger.info("rid=%s %s input_ids=%s", rid, stage, list(input_ids))
