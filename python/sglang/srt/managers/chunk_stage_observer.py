"""Runtime per-stage check of the dynamic-chunking cost model.

The cost model prices each PP stage separately, so an uneven layer partition
(K3 PP16 "7,5,7,5,...,6,3") is already modelled: the microbatch period is the
max over stages and the chunk solve is the min over stages. What the
coefficients cannot see is per-stage work outside the timed model forward --
DSpark's context projection and draft-KV injection, the eager-bank PP hop,
the last stage's sampling path -- which lands unevenly on some stages.

Every rank times its own prefill microbatches (CUDA events around run_batch,
read without blocking) and compares them with what the model predicts for its
stage. The log line gives the per-stage correction to feed back through
SGLANG_DYNAMIC_CHUNKING_STAGE_SCALE. Purely local: no collective, nothing
decided from it at runtime.
"""

import logging
import time
from collections import deque
from typing import Deque, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class StageCostObserver:
    """Compares this stage's measured microbatch times with the cost model."""

    def __init__(
        self,
        cost_model,
        stage: int,
        interval_s: float = 60.0,
        window: int = 512,
        max_pending: int = 64,
    ):
        self.cost_model = cost_model
        self.stage = stage
        self.interval_s = interval_s
        self.max_pending = max_pending
        self._pending: Deque[Tuple[List[Tuple[int, int]], object, object]] = deque()
        self._ratios: Deque[float] = deque(maxlen=window)
        self._actual: Deque[float] = deque(maxlen=window)
        self._binding: Deque[int] = deque(maxlen=window)
        self._next_log: Optional[float] = None

    def submit(self, items: Sequence[Tuple[int, int]], start_event, end_event) -> None:
        """Queue one microbatch's (extend, prefix) items and its CUDA events."""
        if not items:
            return
        if len(self._pending) >= self.max_pending:
            # The poll fell behind (it only runs on later launches); drop the
            # oldest rather than grow without bound.
            self._pending.popleft()
        self._pending.append((list(items), start_event, end_event))

    def poll(self, now: Optional[float] = None) -> None:
        """Collect finished microbatches without blocking; log every interval."""
        while self._pending and self._pending[0][2].query():
            items, start, end = self._pending.popleft()
            actual = float(start.elapsed_time(end))
            period = self.cost_model.batch_period(items)
            predicted = float(period[self.stage])
            if predicted > 0.0 and actual > 0.0:
                self._ratios.append(actual / predicted)
                self._actual.append(actual)
                self._binding.append(int(np.argmax(period)))
        now = time.monotonic() if now is None else now
        if self._next_log is None:
            self._next_log = now + self.interval_s
        elif now >= self._next_log and self._ratios:
            self._next_log = now + self.interval_s
            self.log()

    def summary(self) -> Optional[dict]:
        """Actual/predicted statistics over the window, or None if empty."""
        if not self._ratios:
            return None
        ratios = np.asarray(self._ratios)
        binding = np.asarray(self._binding)
        return {
            "stage": self.stage,
            "n": int(ratios.size),
            "ratio_p50": float(np.percentile(ratios, 50)),
            "ratio_p90": float(np.percentile(ratios, 90)),
            "actual_p50": float(np.percentile(np.asarray(self._actual), 50)),
            # How often this stage is the one the model says sets the period.
            "binding_frac": float(np.mean(binding == self.stage)),
        }

    def log(self) -> None:
        """Log the summary with the suggested per-stage scale."""
        s = self.summary()
        if s is None:
            return
        logger.info(
            "[ChunkCost-Obs] stage=%d n=%d actual/predicted p50=%.3f p90=%.3f "
            "actual_p50=%.2f %s binding_frac=%.2f -> suggested "
            "SGLANG_DYNAMIC_CHUNKING_STAGE_SCALE[%d]=%.3f (x current)",
            s["stage"],
            s["n"],
            s["ratio_p50"],
            s["ratio_p90"],
            s["actual_p50"],
            "ms" if self.cost_model.unit == "ms" else "ms (model unit differs)",
            s["binding_frac"],
            s["stage"],
            s["ratio_p50"],
        )


def parse_stage_scales(spec: str, pp_size: int) -> Optional[List[float]]:
    """``"1,1,1.2,..."`` -> one float per stage; empty -> None."""
    spec = spec.strip()
    if not spec:
        return None
    scales = [float(v) for v in spec.split(",") if v.strip()]
    if len(scales) != pp_size:
        raise ValueError(f"need {pp_size} stage scales, got {len(scales)}: {spec!r}")
    return scales
