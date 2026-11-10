"""Online step-time model of a prefill forward and the decisions built on it.

A prefill step over requests ``i`` (``q_i`` new tokens on top of ``p_i`` tokens
already in the KV cache) is modeled as

    T = c0 + b * sum(q_i) + d * sum(q_i * (p_i + q_i)) + e * num_reqs

``c0`` is the per-forward floor (eager launch, collectives), ``b`` the per-token
dense work, ``d`` the context-proportional indexer / attention work and ``e``
the per-request overhead. The coefficients are refit from measured steps.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import msgspec
import numpy as np

# Prior in seconds, from the DSv4.1 CP4 prefill trace (2026-10-29): ~0.2s floor,
# ~0.26s for a cold 81920-token chunk. Only used until enough steps are observed.
_PRIOR = (0.2, 7.0e-7, 1.5e-12, 2.0e-3)
_REFIT_EVERY = 16


class StepSample(msgspec.Struct, frozen=True):
    """One measured prefill step: model features and its GPU seconds."""
    new_tokens: int
    attn_work: float  # sum(q_i * (p_i + q_i))
    num_reqs: int
    seconds: float


def step_features(*, extend_lens: Sequence[int], prefix_lens: Sequence[int]) -> Tuple[int, float, int]:
    """(sum q, sum q*(p+q), n) of one batch."""
    new_tokens = 0
    attn_work = 0.0
    for q, p in zip(extend_lens, prefix_lens):
        new_tokens += q
        attn_work += float(q) * float(p + q)
    return new_tokens, attn_work, len(extend_lens)


class PrefillCostModel:
    """Non-negative least-squares fit of the step-time model over recent steps."""

    def __init__(self, *, window: int = 256, min_samples: int = 16):
        self._samples: Deque[StepSample] = deque(maxlen=window)
        self._min_samples = min_samples
        self._since_fit = 0
        self.coeffs: Tuple[float, float, float, float] = _PRIOR
        self.fitted = False

    def observe(self, sample: StepSample) -> None:
        """Add a measured step; refits periodically once min_samples are held."""
        if sample.seconds <= 0 or sample.num_reqs <= 0:
            return
        self._samples.append(sample)
        self._since_fit += 1
        if len(self._samples) >= self._min_samples and self._since_fit >= _REFIT_EVERY:
            self._since_fit = 0
            self._refit()

    def predict_step(self, *, new_tokens: int, attn_work: float, num_reqs: int) -> float:
        """Predicted seconds of a step with the given features."""
        c0, b, d, e = self.coeffs
        return c0 + b * new_tokens + d * attn_work + e * num_reqs

    def req_cost(self, *, q: int, p: int) -> float:
        """Marginal seconds one request with ``q`` new tokens on ``p`` adds to a step."""
        _, b, d, e = self.coeffs
        return b * q + d * float(q) * float(p + q) + e

    @property
    def floor_seconds(self) -> float:
        """Per-forward floor c0 in seconds."""
        return self.coeffs[0]

    def median_step_seconds(self) -> Optional[float]:
        """Median measured step seconds, or None before any sample."""
        if not self._samples:
            return None
        return float(np.median([s.seconds for s in self._samples]))

    def max_chunk(self, *, p: int, budget_seconds: float, align: int, lo: int, hi: int) -> int:
        """Largest ``q`` in ``[lo, hi]``, a multiple of ``align``, with req_cost <= budget.

        ``lo`` wins over the budget so a throttled prompt still progresses.
        """
        lo = max(align, lo // align * align)
        hi = max(lo, hi // align * align)
        if self.req_cost(q=hi, p=p) <= budget_seconds:
            return hi
        if self.req_cost(q=lo, p=p) > budget_seconds:
            return lo
        # req_cost is increasing in q (non-negative coefficients): bisect on units.
        left, right = lo // align, hi // align
        while left < right:
            mid = (left + right + 1) // 2
            if self.req_cost(q=mid * align, p=p) <= budget_seconds:
                left = mid
            else:
                right = mid - 1
        return left * align

    def _refit(self) -> None:
        rows = np.array(
            [[1.0, s.new_tokens, s.attn_work, s.num_reqs] for s in self._samples],
            dtype=np.float64,
        )
        target = np.array([s.seconds for s in self._samples], dtype=np.float64)
        coeffs = _nonnegative_lstsq(rows, target)
        if coeffs is not None:
            self.coeffs = tuple(float(c) for c in coeffs)
            self.fitted = True


def _nonnegative_lstsq(rows: np.ndarray, target: np.ndarray) -> Optional[np.ndarray]:
    """Active-set least squares with every coefficient clamped at >= 0."""
    scale = np.abs(rows).max(axis=0)
    scale[scale == 0] = 1.0
    normalized = rows / scale
    active = np.ones(rows.shape[1], dtype=bool)
    for _ in range(rows.shape[1]):
        if not active.any():
            return None
        solution, *_ = np.linalg.lstsq(normalized[:, active], target, rcond=None)
        full = np.zeros(rows.shape[1])
        full[active] = solution
        negative = full < 0
        if not negative.any():
            return full / scale
        active &= ~negative
    return None


class CostScheduleConfig(msgspec.Struct, frozen=True):
    """Static knobs of PrefillCostScheduler, read from env at startup."""
    target_step_seconds: float  # <= 0 disables long-chunk throttling
    aging_seconds: float
    min_long_chunk: int
    short_seconds: float
    chunk_align: int = 1024


class PrefillCostScheduler:
    """Orders the waiting queue by predicted cost and throttles long chunks.

    Requests are costed on their *uncached* work: a prompt whose prefix is
    mostly cached is cheap even when it is long. Requests that have waited past
    the aging bound are served FCFS ahead of everything else.

    Every input must be identical on all ranks of the attention group, since
    each rank orders its own waiting queue: the clock is the rank-0 timestamp
    the caller distributes, entry stamps are taken on that clock, and the model
    only observes samples measured on rank 0.
    """

    def __init__(self, *, config: CostScheduleConfig, model: PrefillCostModel):
        self.config = config
        self.model = model
        self.now = 0.0
        self._entry: Dict[str, float] = {}
        self._last_short_seen = float("-inf")

    def set_clock(self, now: float) -> None:
        """Set the rank-consistent clock used for aging."""
        self.now = now

    def note_ready(self, reqs: Iterable) -> None:
        """Stamp requests that just entered the waiting queue."""
        for req in reqs:
            self._entry.setdefault(req.rid, self.now)

    def forget(self, reqs: Iterable) -> None:
        """Drop the entry stamps of these requests."""
        for req in reqs:
            self._entry.pop(req.rid, None)

    def order(self, waiting_queue: List, *, prune: bool = True) -> None:
        """Stable in-place sort: aged FCFS first, then ascending predicted cost.

        ``prune`` drops entry stamps of requests no longer in ``waiting_queue``;
        pass False when sorting a subset.
        """
        aging = self.config.aging_seconds
        live = set()
        keyed = []
        for index, req in enumerate(waiting_queue):
            live.add(req.rid)
            entry = self._entry.setdefault(req.rid, self.now)
            q, p = estimate_work(req)
            cost = self.model.req_cost(q=q, p=p)
            if cost <= self.config.short_seconds:
                self._last_short_seen = self.now
            aged = self.now - entry >= aging
            keyed.append(((0, entry, index) if aged else (1, cost, index), req))
        keyed.sort(key=lambda item: item[0])
        waiting_queue[:] = [req for _, req in keyed]
        if prune:
            for rid in [rid for rid in self._entry if rid not in live]:
                del self._entry[rid]

    def chunk_cap(
        self, *, context_len: int, waiting_queue: Sequence, hi: int, used_seconds: float = 0.0
    ) -> Optional[int]:
        """Cap for a long prompt's chunk this step, or None to leave it unthrottled.

        Throttles only while cheap work is waiting or was seen within the last
        two steps: with nothing short around the full chunk keeps throughput.
        A non-positive target disables throttling.
        """
        target = self.config.target_step_seconds
        if target <= 0 or not self._short_work_expected(waiting_queue, horizon=target):
            return None
        budget = max(0.0, target - self.model.floor_seconds - used_seconds)
        return self.model.max_chunk(
            p=context_len,
            budget_seconds=budget,
            align=self.config.chunk_align,
            lo=min(self.config.min_long_chunk, hi),
            hi=hi,
        )

    def _short_work_expected(self, waiting_queue: Sequence, *, horizon: float) -> bool:
        if self.now - self._last_short_seen <= 2 * horizon:
            return True
        for req in waiting_queue:
            q, p = estimate_work(req)
            if self.model.req_cost(q=q, p=p) <= self.config.short_seconds:
                self._last_short_seen = self.now
                return True
        return False


def estimate_work(req) -> Tuple[int, int]:
    """(uncached tokens, cached tokens) of a waiting request, best known estimate.

    Uses the device prefix of the last admission attempt and the tokenizer's
    AttentionStore hit hint; before admission nothing is matched yet.
    """
    total = len(req.origin_input_ids) + len(req.output_ids)
    hint = max(req.as_max_hit_token_len) if req.as_max_hit_token_len else 0
    cached = min(total - 1, max(len(req.prefix_indices), int(hint)))
    cached = max(0, cached)
    return total - cached, cached
