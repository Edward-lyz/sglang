"""Rank-consistent control state of the continuous prefill admission loop.

Every decision that changes what a rank admits (stop polling, the clock used
for aging, the step times the cost model learns from) is taken on the leader
rank (attn-tp 0, attn-cp 0) and distributed with one small MAX all-reduce over
the attention groups, so all ranks of a waiting queue stay in lockstep.
"""

from __future__ import annotations

import logging
import queue
import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

from sglang.srt.managers.scheduler_components.prefill_cost_model import (
    CostScheduleConfig,
    PrefillCostModel,
    PrefillCostScheduler,
    StepSample,
    step_features,
)

logger = logging.getLogger(__name__)

# Contribution of non-leader ranks to a MAX reduce: any leader value wins.
_NEG = -1.0e300


class _StepRecord:
    __slots__ = ("features", "events", "late_admitted", "polls", "wait_s", "seal_s")

    def __init__(self, features, events, late_admitted, polls, wait_s, seal_s):
        self.features = features
        self.events = events
        self.late_admitted = late_admitted
        self.polls = polls
        self.wait_s = wait_s
        self.seal_s = seal_s


class PrefillSchedStats:
    """Interval summary of prefill steps, logged by the leader rank."""

    def __init__(self, *, interval_s: float):
        self.interval_s = interval_s
        self._last_log = time.perf_counter()
        self._rows: Dict[str, List[float]] = {}
        self._prev_end_event = None

    def add(self, **values: float) -> None:
        """Append one value per named stat row."""
        for key, value in values.items():
            self._rows.setdefault(key, []).append(float(value))

    def gpu_times(self, events) -> Tuple[Optional[float], Optional[float]]:
        """(forward ms, GPU gap ms since the previous forward) of finished events."""
        start, end = events
        end.synchronize()
        gpu_ms = start.elapsed_time(end)
        gap_ms = None
        if self._prev_end_event is not None:
            gap_ms = max(0.0, self._prev_end_event.elapsed_time(start))
        self._prev_end_event = end
        return gpu_ms, gap_ms

    def maybe_log(self) -> None:
        """Log interval percentiles and reset once the interval has elapsed."""
        now = time.perf_counter()
        if now - self._last_log < self.interval_s or not self._rows:
            return
        steps = len(self._rows.get("gpu_ms", ()))
        parts = [f"steps={steps}"]
        for key in sorted(self._rows):
            values = np.asarray(self._rows[key])
            parts.append(
                f"{key}=p50:{np.percentile(values, 50):.2f}/p90:{np.percentile(values, 90):.2f}"
                f"/sum:{values.sum():.1f}"
            )
        logger.info("Prefill sched stats (%.0fs): %s", now - self._last_log, " ".join(parts))
        self._rows = {}
        self._last_log = now


class ContinuousPrefillController:
    """Leader-driven polling, clock and cost-model state of one waiting queue."""

    def __init__(
        self,
        *,
        is_leader: bool,
        attn_tp_cpu_group,
        attn_cp_cpu_group,
        poll_s: float,
        stats_interval_s: float,
        cost_config: Optional[CostScheduleConfig],
        continuous: bool,
    ):
        self.is_leader = is_leader
        self._groups = [
            group
            for group in (attn_tp_cpu_group, attn_cp_cpu_group)
            if group is not None and dist.get_world_size(group=group) > 1
        ]
        self.poll_s = poll_s
        self.continuous = continuous
        self.stats = PrefillSchedStats(interval_s=stats_interval_s) if stats_interval_s > 0 else None
        self.cost_model = PrefillCostModel() if cost_config is not None else None
        self.cost_scheduler = (
            PrefillCostScheduler(config=cost_config, model=self.cost_model)
            if cost_config is not None
            else None
        )
        # GPU events are recorded on the leader only: it alone measures time.
        self.needs_events = is_leader and (self.stats is not None or self.cost_model is not None)
        self._step_id = 0
        self._records: Dict[int, _StepRecord] = {}
        self._measured: Deque[Tuple[int, float]] = deque()
        self._last_observed = 0
        self._step_start_wall = 0.0
        self._cur_polls = 0
        self._cur_late = 0
        self._cur_wait_s = 0.0
        self._cur_seal_s = 0.0
        self.now = 0.0

    # ---- collectives (call in the same order on every rank) -------------

    def _reduce_max(self, values: List[float]) -> List[float]:
        tensor = torch.tensor(values if self.is_leader else [_NEG] * len(values), dtype=torch.float64)
        for group in self._groups:
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=group)
        return tensor.tolist()

    def sync_clock(self) -> None:
        """Distribute the leader clock and its oldest unshared step time."""
        if self.cost_scheduler is None:
            return
        step_id, seconds = self._measured.popleft() if self._measured else (0, 0.0)
        now, step_id, seconds = self._reduce_max([time.perf_counter(), float(step_id), seconds])
        self._set_now(now)
        step_id = int(step_id)
        if step_id > self._last_observed:
            self._last_observed = step_id
            record = self._records.pop(step_id, None)
            if record is not None and seconds > 0:
                new_tokens, attn_work, num_reqs = record.features
                self.cost_model.observe(
                    StepSample(new_tokens=new_tokens, attn_work=attn_work, num_reqs=num_reqs, seconds=seconds)
                )

    def poll_forward_done(self, try_get: Callable[[float], bool]) -> bool:
        """One poll period: True once the leader's in-flight forward has returned.

        ``try_get(timeout)`` blocks the leader up to ``timeout`` (GIL released)
        and returns whether a result arrived; other ranks wait in the reduce.
        """
        done = try_get(self.poll_s) if self.is_leader else False
        flag, now = self._reduce_max([1.0 if done else 0.0, time.perf_counter()])
        self._set_now(now)
        self._cur_polls += 1
        return flag > 0

    def _set_now(self, now: float) -> None:
        self.now = now
        if self.cost_scheduler is not None:
            self.cost_scheduler.set_clock(now)

    # ---- bookkeeping (local) --------------------------------------------

    def note_ready(self, reqs) -> None:
        """Stamp newly ready requests for cost-scheduler aging."""
        if self.cost_scheduler is not None:
            self.cost_scheduler.note_ready(reqs)

    def note_late_admitted(self, count: int, seconds: float) -> None:
        """Accumulate requests admitted while the forward ran."""
        self._cur_late += count
        self._cur_wait_s += seconds

    def note_seal(self, seconds: float) -> None:
        """Record the seal duration of the next submitted step."""
        self._cur_seal_s = seconds

    def on_submit(self, batch, events) -> int:
        """Record a submitted batch; returns its step id."""
        self._step_id += 1
        features = step_features(extend_lens=batch.extend_lens, prefix_lens=batch.prefix_lens)
        self._records[self._step_id] = _StepRecord(
            features, events, self._cur_late, self._cur_polls, self._cur_wait_s, self._cur_seal_s
        )
        self._cur_polls = 0
        self._cur_late = 0
        self._cur_wait_s = 0.0
        self._cur_seal_s = 0.0
        return self._step_id

    def attach_events(self, step_id: int, events) -> None:
        """Attach the forward's timing events to its step record."""
        record = self._records.get(step_id)
        if record is not None:
            record.events = events

    def on_result_processed(self, step_id: int) -> None:
        """After the step's result was processed (its outputs are synchronized)."""
        record = self._records.get(step_id)
        if record is None:
            return
        if self.cost_scheduler is None:
            self._records.pop(step_id, None)
        if not self.is_leader or record.events is None:
            return
        gpu_ms, gap_ms = (
            self.stats.gpu_times(record.events) if self.stats is not None else _elapsed(record.events)
        )
        if self.cost_model is not None:
            self._measured.append((step_id, gpu_ms / 1000.0))
        if self.stats is not None:
            new_tokens, _, num_reqs = record.features
            row = dict(
                gpu_ms=gpu_ms,
                new_tokens=new_tokens,
                reqs=num_reqs,
                late_admitted=record.late_admitted,
                polls=record.polls,
                admit_wait_ms=record.wait_s * 1000.0,
                seal_ms=record.seal_s * 1000.0,
            )
            if gap_ms is not None:
                row["gpu_gap_ms"] = gap_ms
            self.stats.add(**row)
            self.stats.maybe_log()
        record.events = None


def _elapsed(events) -> Tuple[float, None]:
    start, end = events
    end.synchronize()
    return start.elapsed_time(end), None


def make_try_get(out_q: "queue.Queue", stash: Callable[[object], None]) -> Callable[[float], bool]:
    """Leader-side non-busy wait on the forward thread's result queue."""

    def try_get(timeout: float) -> bool:
        try:
            item = out_q.get(timeout=timeout)
        except queue.Empty:
            return False
        stash(item)
        return True

    return try_get
