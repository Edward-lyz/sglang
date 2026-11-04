"""Bounded opt-in diagnostics for scheduler admission and writer ownership."""

import atexit
from collections import Counter, deque
from functools import wraps
import json
import os
from pathlib import Path
import time

from sglang.srt.environ import envs


class AdmissionDiagnostic:
    """Collect bounded, opt-in scheduler admission and write diagnostics."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.control = self.directory / "active"
        self.destination = None
        self.records = deque(maxlen=32768)
        self.dropped = 0
        self.active = False
        self.next_control_check = 0
        self.loop_id = 0
        self.attempt_id = 0
        self.scheduler = None
        self.adder = None
        self.counts = Counter()
        self.last_reason = None
        self.submitted_aid = None
        atexit.register(self.dump)

    def poll(self):
        """Read the control file and start or stop bounded event collection.

        Returns:
            None.

        Raises:
            OSError: A control file or output path cannot be accessed.
        """
        now = time.monotonic()
        if now < self.next_control_check:
            return
        self.next_control_check = now + 0.01
        try:
            destination = self.control.read_text().strip()
        except FileNotFoundError:
            destination = ""
        if destination and destination != self.destination:
            self.dump()
            self.records.clear()
            self.dropped = 0
            self.destination = destination
            self.active = True
        elif not destination and self.active:
            self.active = False
            self.dump()
            self.destination = None

    def dump(self):
        """Write the current event ring and drop count to the selected
        directory.

        Returns:
            None.

        Raises:
            OSError: The output directory or report cannot be written.
        """
        if not self.destination or not self.records:
            return
        directory = Path(self.destination)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"admission_{os.getpid()}.json"
        path.write_text(json.dumps({
            "pid": os.getpid(),
            "dropped": self.dropped,
            "clock": "perf_counter_ns",
            "events": list(self.records),
        }) + "\n")

    def emit(self, event, **fields):
        """Append an event only while diagnostics are active.

        Args:
            event: Diagnostic event name.
            fields: Additional event fields copied into the diagnostic record.

        Returns:
            None.
        """
        if not self.active:
            return
        if len(self.records) == self.records.maxlen:
            self.dropped += 1
        self.records.append({
            "t_ns": time.perf_counter_ns(),
            "loop": self.loop_id,
            "aid": self.attempt_id,
            "event": event,
            **fields,
        })

    def pool(self, scheduler):
        """Read scheduler queue sizes and available allocator capacities.

        Args:
            scheduler: Scheduler whose queues and pool capacities are sampled.

        Returns:
            dict: Current queue, cache and pool measurements.
        """
        allocator = scheduler.token_to_kv_pool_allocator
        cache = scheduler.tree_cache
        observer = scheduler.pool_stats_observer
        values = {
            "gpu": scheduler.ps.gpu_id,
            "full_capacity": observer.full_tokens_per_layer,
            "swa_capacity": observer.swa_tokens_per_layer,
            "req_available": scheduler.req_to_token_pool.available_size(),
            "queue": len(scheduler.waiting_queue),
            "bootstrap": len(scheduler.disagg_prefill_bootstrap_queue.queue),
            "inflight": len(scheduler.disagg_prefill_inflight_queue),
            "chunk_rid": getattr(scheduler.chunked_req, "rid", None),
        }
        for name, owner, method in (
            ("full_available", allocator, "full_available_size"),
            ("swa_available", allocator, "swa_available_size"),
            ("full_evictable", cache, "full_evictable_size"),
            ("swa_evictable", cache, "swa_evictable_size"),
        ):
            function = getattr(owner, method, None)
            if function is not None:
                values[name] = function()
        coordinator = getattr(cache, "attention_store_coordinator", None)
        if coordinator is not None:
            values["pending_writes"] = len(coordinator._pending_writes)
            values["pending_load_backs"] = len(coordinator._pending_load_backs)
        return values

    def budget(self, adder):
        """Read the admission budget without changing the adder.

        Args:
            adder: Admission adder holding the current token budgets.

        Returns:
            dict: Token budgets and accepted-request counters.
        """
        values = {
            "rem_total": adder.rem_total_tokens,
            "cur_rem": adder.cur_rem_tokens,
            "rem_input": adder.rem_input_tokens,
            "rem_chunk": adder.rem_chunk_tokens,
            "total_offset": adder.rem_total_token_offset,
            "cur_offset": adder.cur_rem_token_offset,
            "accepted": len(adder.can_run_list),
            "accepted_new": adder.log_input_tokens,
            "accepted_hit": adder.log_hit_tokens,
            "page_size": adder.page_size,
        }
        if adder.is_hybrid_swa:
            values["rem_swa"] = adder.rem_swa_tokens
            values["swa_offset"] = adder.rem_swa_token_offset
        return values

    def request(self, req):
        """Describe one request and its outstanding load-back state.

        Args:
            req: Request whose diagnostic state is sampled, when supplied.

        Returns:
            dict: Request identity, lengths and load-back indicators.
        """
        return {
            "rid": req.rid,
            "fill_len": len(req.full_untruncated_fill_ids),
            "prefix_len": len(req.prefix_indices),
            "host_hit": req.host_hit_length,
            "host_loaded": req.host_loaded_length,
            "swa_host_hit": req.swa_host_hit_length,
            "pending_load_back": req._as_pending_load_back is not None,
            "middle_chunks": req.inflight_middle_chunks,
        }

    def phase(self, scheduler, stage, batch=None):
        """Record a scheduler phase with its current pool measurements.

        Args:
            scheduler: Scheduler whose queues and pool capacities are sampled.
            stage: Name of the scheduler phase being recorded.
            batch: Optional scheduled batch associated with the phase.

        Returns:
            None.
        """
        if not self.active:
            return
        self.emit(
            stage,
            batch_aid=getattr(batch, "admission_attempt_id", None),
            submitted_aid=self.submitted_aid,
            **self.pool(scheduler),
        )

    def reason(self, reason, adder=None, req=None, **fields):
        """Record an admission decision and optional request or budget details.

        Args:
            reason: Diagnostic decision or skip reason.
            adder: Admission adder holding the current token budgets.
            req: Request whose diagnostic state is sampled, when supplied.
            fields: Additional event fields copied into the diagnostic record.

        Returns:
            None.
        """
        if not self.active:
            return
        self.last_reason = reason
        self.emit(
            "reason", reason=reason,
            budget=self.budget(adder) if adder is not None else None,
            request=self.request(req) if req is not None else None,
            pool=self.pool(self.scheduler) if self.scheduler is not None else None,
            **fields,
        )

    def skip(self, reason):
        """Increment the named skip counter while diagnostics are active.

        Args:
            reason: Diagnostic decision or skip reason.

        Returns:
            None.
        """
        if self.active:
            self.counts["skip_" + reason] += 1

    def attach(self, adder):
        """Attach the active adder and record its initial budget.

        Args:
            adder: Admission adder holding the current token budgets.

        Returns:
            None.
        """
        if self.active:
            self.adder = adder
            self.emit("adder_start", budget=self.budget(adder))

    def candidate(self, req, adder, result=None, previous_count=None):
        """Record a candidate before or after ordinary admission.

        Args:
            req: Request whose diagnostic state is sampled, when supplied.
            adder: Admission adder holding the current token budgets.
            result: Existing operation result to record or finalize.
            previous_count: Accepted-request count before evaluating the
                candidate.

        Returns:
            None.
        """
        if not self.active:
            return
        if result is None:
            self.counts["scanned"] += 1
        else:
            self.counts["accepted_waiting"] += len(adder.can_run_list) - previous_count
        self.emit(
            "candidate_before" if result is None else "candidate_after",
            request=self.request(req), budget=self.budget(adder),
            result=getattr(result, "name", None),
        )

    def write_event(self, event, pending_write=None, **fields):
        """Record a writer event with optional logical task identity.

        Args:
            event: Diagnostic event name.
            pending_write: Optional pending write whose task identity is
                recorded.
            fields: Additional event fields copied into the diagnostic record.

        Returns:
            None.
        """
        if not self.active:
            return
        if pending_write is not None:
            fields.update(
                write_id=pending_write.logical_id.hex(),
                rid=getattr(pending_write.task, "rid", None),
                requested_blocks=pending_write.task.requested_blocks,
            )
        self.emit(event, **fields)


ADMISSION_DIAG = (
    AdmissionDiagnostic(envs.SGLANG_ADMISSION_DIAG_DIR.get())
    if envs.SGLANG_ADMISSION_DIAG_DIR.get()
    else None
)


def admission_result(adder, result, reason, req=None, **fields):
    """Record an admission reason and return the original result unchanged.

    Args:
        adder: Admission adder holding the current token budgets.
        result: Existing operation result to record or finalize.
        reason: Diagnostic decision or skip reason.
        req: Request whose diagnostic state is sampled, when supplied.
        fields: Additional event fields copied into the diagnostic record.

    Returns:
        Any: The original admission result.
    """
    if ADMISSION_DIAG is not None:
        ADMISSION_DIAG.reason(reason, adder, req, **fields)
    return result


def admission_stage(stage):
    """Create a stage decorator that is inert when diagnostics are disabled.

    Args:
        stage: Name of the scheduler phase being recorded.

    Returns:
        Callable: A decorator preserving the wrapped method result.
    """
    def decorate(function):
        if ADMISSION_DIAG is None:
            return function

        @wraps(function)
        def wrapped(scheduler, *args, **kwargs):
            diagnostic = ADMISSION_DIAG
            if stage == "retire":
                diagnostic.poll()
                diagnostic.loop_id += 1
                diagnostic.scheduler = scheduler
            if not diagnostic.active:
                return function(scheduler, *args, **kwargs)
            batch = args[0] if args and hasattr(args[0], "reqs") else None
            if stage == "submit":
                diagnostic.submitted_aid = getattr(batch, "admission_attempt_id", None)
            diagnostic.phase(scheduler, stage + "_before", batch)
            try:
                return function(scheduler, *args, **kwargs)
            finally:
                diagnostic.phase(scheduler, stage + "_after", batch)

        return wrapped

    return decorate


def trace_admission(function):
    """Wrap scheduler admission with bounded diagnostic events when enabled.

    Args:
        function: Callable whose original return value must be preserved.

    Returns:
        Callable: The original function or its diagnostic wrapper.
    """
    if ADMISSION_DIAG is None:
        return function

    @wraps(function)
    def wrapped(scheduler, *args, **kwargs):
        diagnostic = ADMISSION_DIAG
        if not diagnostic.active:
            return function(scheduler, *args, **kwargs)
        from sglang.srt.utils.nvtx_utils import operations_nvtx_range

        diagnostic.scheduler = scheduler
        diagnostic.attempt_id += 1
        diagnostic.counts.clear()
        diagnostic.last_reason = None
        diagnostic.adder = None
        diagnostic.phase(scheduler, "admission_before")
        with operations_nvtx_range(f"sched.admission.a{diagnostic.attempt_id}"):
            result = function(scheduler, *args, **kwargs)
        batch = result[0]
        if batch is not None:
            batch.admission_attempt_id = diagnostic.attempt_id
        diagnostic.emit(
            "attempt_end",
            reason=diagnostic.last_reason or "queue_exhausted",
            counts=dict(diagnostic.counts),
            new=getattr(batch, "extend_num_tokens", 0) or 0,
            reqs=len(batch.reqs) if batch is not None else 0,
            budget=diagnostic.budget(diagnostic.adder) if diagnostic.adder is not None else None,
            pool=diagnostic.pool(scheduler),
        )
        return result

    return wrapped
