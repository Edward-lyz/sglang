"""Per-stage compute-cost model for dynamic chunked prefill.

Prices a chunk of x new tokens on top of L cached history tokens, separately
for every PP stage:

    cost_s(L, x) = k_g[s] * x                  # GEMM / MoE, linear in new tokens
                 + k_e[s] * (L + x)            # MLA kv_b_proj latent re-expansion
                 + k_a[s] * x * (L + 0.5 * x)  # causal attention (bilinear)

A pipeline microbatch takes as long as its slowest stage, so the batch
constraint is ``max_s cost_s(L, x) <= max_s cost_s(0, base_chunk)``: no
microbatch may cost more than a fresh chunk of the configured size does. Both
sides are a max over stages -- the budget is one absolute envelope, not a
per-stage one (see set_target_cost). The argmax moves with the operating point
-- GEMM-heavy stages dominate fresh chunks, MLA-heavy stages dominate long
prefixes -- so all stages are kept and evaluated. Collapsing
to one "bottleneck stage" mispredicts on the wrong side of the crossover, and
collapsing to a per-term max invents a stage that is heavier than any real one
(K3 partitions layers 7/6/5 and spreads MLA layers unevenly, so the per-term
argmaxes are different stages).

Coefficients are pure FLOPs counts divided by ``max_s k_g[s]``, i.e. the unit
is "GEMM-equivalent tokens": ``cost(0, B) ~ B``, directly comparable to
--chunked-prefill-size in logs. Any uniform scale factor cancels in admission
because the budget is derived from the same coefficients, which is why no
peak-TFLOPS table, MFU target or FLOPs->ms conversion appears here.

The k_e term is what an ``A + B*T`` per-token formula cannot express: the KV
pool stores only the MLA latent, so every chunked forward re-expands the whole
(L + x) prefix. It is also why splitting a long-prefix request into more chunks
strictly increases total work -- each chunk repays it.

Measured mode (``init_measured``) keeps the same function but takes per-stage
coefficients fitted from timed forwards, in milliseconds, plus a per-microbatch
constant ``e[s]`` that the FLOPs walk cannot see (launches, PP send/recv setup,
MoE small-batch inefficiency folded into the fit). The constant is charged once
per microbatch, so it is taken out of the budget rather than added to each
request: ``target[s] = period - e[s]``.

This module is intentionally dependency-free (numpy only) so it can be unit
tested without importing the rest of sglang.
"""

import logging
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ChunkCostModel:
    """Compute budget for one prefill microbatch, priced per PP stage.

    Built once at startup by ``init_analytic`` from each stage's walked
    per-token FLOPs, then queried by the scheduler on every admission:
    ``req_cost`` for what a candidate would add, ``solve_x`` for the largest
    chunk that still fits, ``reexpand_cost`` for the price of splitting a
    long-prefix request once more.

    Coefficients and budgets are per-stage arrays (length ``pp_size``); a
    microbatch is bounded by ``max_s cost_s <= target_s``, so callers pass and
    decrement whole ``rem_cost`` vectors rather than a scalar. See the module
    docstring for the cost function and why no stage is collapsed away.
    """

    def __init__(self):
        # Per-stage coefficient arrays, in GEMM-equivalent tokens.
        self.k_g: Optional[np.ndarray] = None
        self.k_e: Optional[np.ndarray] = None
        self.k_a: Optional[np.ndarray] = None
        # Per-microbatch constant, charged once per batch (zero in FLOPs mode).
        self.k_overhead: Optional[np.ndarray] = None
        # Per-stage budget available to requests: period - k_overhead.
        self.target_cost: Optional[np.ndarray] = None
        self.unit = "gemm-equivalent tokens"
        self.is_ready = False

    def init_analytic(
        self,
        flops: Sequence[float],
        mla_layers: Sequence[float],
        attn_flops_per_pair_per_layer: float,
        reexpand_flops_per_layer: float,
    ):
        """Build the coefficients from per-stage walked constants.

        ``flops`` is each stage's per-new-token weight-GEMM FLOPs (from the
        weight walk, so already PP/TP- and quantization-aware) and
        ``mla_layers`` its full-attention layer count. The per-layer attention
        and re-expansion constants come from the model config and are the same
        on every stage, so the stage spread is carried entirely by these two
        arrays.
        """
        k_g = np.asarray(flops, dtype=np.float64)
        n_mla = np.asarray(mla_layers, dtype=np.float64)
        if k_g.ndim != 1 or k_g.shape != n_mla.shape:
            raise ValueError(
                f"flops {k_g.shape} and mla_layers {n_mla.shape} must be "
                "matching 1-D per-stage arrays"
            )
        if not np.all(k_g > 0):
            raise ValueError(f"every stage needs per-new-token FLOPs > 0: {k_g}")
        if attn_flops_per_pair_per_layer <= 0:
            raise ValueError("attention FLOPs per (q, kv) pair per layer must be > 0")
        if reexpand_flops_per_layer < 0:
            raise ValueError("re-expansion FLOPs per layer must be >= 0")
        if not np.any(n_mla > 0):
            raise ValueError(
                "no full-attention layers found on any stage; the weight walk "
                "did not see any kv_b_proj"
            )
        g_ref = float(k_g.max())
        self.k_g = k_g / g_ref
        self.k_e = (reexpand_flops_per_layer * n_mla) / g_ref
        self.k_a = (attn_flops_per_pair_per_layer * n_mla) / g_ref
        self.k_overhead = np.zeros_like(self.k_g)
        self._finish_init()
        logger.info(
            "[ChunkCostModel] %d stages, GEMM-equivalent-token coefficients "
            "(1 unit = one new token through the heaviest stage's GEMMs): "
            "k_g=%s, k_e=%s, k_a=%s",
            len(k_g),
            np.array2string(self.k_g, precision=3),
            np.array2string(self.k_e, precision=4),
            np.array2string(self.k_a, precision=3, formatter={"float": "{:.3e}".format}),
        )

    def init_measured(self, coefs: Sequence[Sequence[float]]):
        """Build the model from per-stage fitted coefficients.

        ``coefs`` is ``[pp_size, 4]``: ``(e, g, r, a)`` per stage in ms, i.e.
        ``t_s = e + sum_req[g*x + r*(L+x) + a*x*(L+0.5x)]``, as produced by
        ``fit_stage_coefficients``. ``r``/``a`` may be zero (a stage without MLA
        layers); ``g`` must be positive so every chunk has a price.
        """
        c = np.asarray(coefs, dtype=np.float64)
        if c.ndim != 2 or c.shape[1] != 4 or c.shape[0] == 0:
            raise ValueError(f"measured coefficients must be [pp_size, 4]: {c.shape}")
        if not np.all(np.isfinite(c)) or np.any(c < 0.0):
            raise ValueError(f"measured coefficients must be finite and >= 0: {c}")
        if not np.all(c[:, 1] > 0.0):
            raise ValueError(f"every stage needs a positive per-token cost: {c[:, 1]}")
        self.k_overhead = c[:, 0].copy()
        self.k_g = c[:, 1].copy()
        self.k_e = c[:, 2].copy()
        self.k_a = c[:, 3].copy()
        self.unit = "ms"
        self._finish_init()
        logger.info(
            "[ChunkCostModel] %d stages, measured coefficients (ms): e=%s, "
            "g=%s, r=%s, a=%s",
            len(self.k_g),
            np.array2string(self.k_overhead, precision=3),
            np.array2string(self.k_g, formatter={"float": "{:.3e}".format}),
            np.array2string(self.k_e, formatter={"float": "{:.3e}".format}),
            np.array2string(self.k_a, formatter={"float": "{:.3e}".format}),
        )

    def _finish_init(self):
        # solve_x runs per admitted request per scheduling iteration, so keep
        # its hot path to one allocation-free vectorised expression: hoist the
        # constant sum and note when no stage degenerates to linear (only an
        # all-KDA stage does, which is rare).
        self._k_ge = self.k_g + self.k_e
        self._all_quadratic = bool(np.all(self.k_a > 0.0))
        # req_cost as one matvec: k_g*x + k_e*(L+x) + k_a*x*(L+x/2)
        #   = [x, L, x*(L+x/2)] @ [k_g+k_e; k_e; k_a]
        self._cost_rows = np.stack([self._k_ge, self.k_e, self.k_a])
        # solve_x walks plain floats (see there).
        self._k_ge_l = self._k_ge.tolist()
        self._k_e_l = self.k_e.tolist()
        self._k_a_l = self.k_a.tolist()
        self._k_e_max = float(self.k_e.max())

    def scale_stages(self, scales: Sequence[float]):
        """Multiply every coefficient of stage s by ``scales[s]``.

        Corrects per-stage work the coefficients do not see -- on K3 PP16 that
        is the DSpark context projection / draft-KV injection, the lm_head on
        the last stage and the cross-node PP hop -- using the ratios the
        runtime stage observer logs. Must be followed by set_target_cost.
        """
        s = np.asarray(scales, dtype=np.float64)
        if s.shape != self.k_g.shape:
            raise ValueError(
                f"need one scale per stage ({self.k_g.shape[0]}), got {s.shape}"
            )
        if not np.all(np.isfinite(s)) or np.any(s <= 0.0):
            raise ValueError(f"stage scales must be finite and > 0: {s}")
        self.k_overhead = self.k_overhead * s
        self.k_g = self.k_g * s
        self.k_e = self.k_e * s
        self.k_a = self.k_a * s
        self._finish_init()
        logger.info(
            "[ChunkCostModel] Applied per-stage scales %s",
            np.array2string(s, precision=3),
        )

    def binding_stage(self, items: Sequence[Tuple[int, int]]) -> int:
        """The stage that sets this microbatch's period (argmax of
        batch_period). With an uneven layer partition it moves with the
        batch: GEMM-heavy stages bind fresh chunks, MLA-heavy ones long
        prefixes."""
        return int(np.argmax(self.batch_period(items)))

    def batch_period(self, items: Sequence[Tuple[int, int]]) -> np.ndarray:
        """Per-stage predicted time of one microbatch of ``(history_len,
        chunk_len)`` items, including the per-microbatch constant."""
        total = self.k_overhead.copy()
        for history_len, chunk_len in items:
            total += self.req_cost(history_len, chunk_len)
        return total

    def req_cost(self, history_len: int, chunk_len: int) -> np.ndarray:
        """Per-stage cost of prefilling chunk_len tokens on history_len cached
        tokens."""
        if chunk_len <= 0:
            return np.zeros_like(self.k_g)
        L = float(history_len)
        x = float(chunk_len)
        return np.dot((x, L, x * (L + 0.5 * x)), self._cost_rows)

    def reexpand_cost(self, history_len: int) -> float:
        """Cost of re-expanding the latent prefix once, on the stage that pays
        the most for it. This is repaid by every additional chunk, so it is the
        price of splitting a long-prefix request one more time."""
        return self._k_e_max * float(history_len)

    def set_target_cost(self, base_chunk_size: int, period: Optional[float] = None):
        """Budget for one microbatch: the *pipeline period* of a fresh chunk of
        the configured base size, i.e. max_s cost_s(0, base_chunk_size).

        Deliberately not each stage's own cost_s(0, base). The period is set by
        the slowest stage, so a stage that is cheap on a fresh chunk has real
        slack; capping it at its own low figure is a tighter constraint than
        "no slower than a fresh base chunk" with nothing behind it. On K3 that
        cost 2.9% of the new tokens per batch: cost_s(0, 24K) spans 25% across
        the 16 stages, and long-prefix chunks ended up bound by a cheap-GEMM /
        MLA-heavy stage that was never the bottleneck.

        ``period`` overrides the derived envelope with an explicit value in
        the model's unit (ms in measured mode). Each stage's per-microbatch
        constant is taken out of the period, so what remains is what requests
        may spend.

        With no prefix hit and no explicit period this still reproduces the
        static token budget exactly, on the stage that sets the period (see
        solve_x).
        """
        per_stage = self.k_overhead + self.req_cost(0, base_chunk_size)
        if not np.all(per_stage > 0):
            raise ValueError(
                f"cost-model target must be positive per stage: {per_stage}"
            )
        envelope = float(per_stage.max())
        if period is not None and period > 0:
            envelope = float(period)
        target = envelope - self.k_overhead
        if not np.all(target > 0):
            raise ValueError(
                f"microbatch period {envelope} does not cover the per-microbatch "
                f"constant of every stage: {self.k_overhead}"
            )
        self.target_cost = target
        logger.info(
            "[ChunkCostModel] Target period per microbatch (base_chunk_size=%d): "
            "%.3f %s (explicit=%s), from per-stage %s (max/min=%.2f, i.e. "
            "the layer partition is that imbalanced on a fresh chunk)",
            base_chunk_size,
            envelope,
            self.unit,
            period is not None and period > 0,
            np.array2string(per_stage, precision=3),
            per_stage.max() / per_stage.min(),
        )

    def solve_x(self, history_len: int, rem_cost: np.ndarray) -> float:
        """Largest chunk x that fits every stage's remaining budget.

        Per stage this solves 0.5*k_a*x^2 + (k_g + k_e + k_a*L)*x + k_e*L = rem.
        Each stage's feasible set is [0, x_s], so the intersection is
        [0, min_s x_s] and the minimum is exact, not an upper bound.

        Returns 0.0 when some stage cannot afford even x=0 (its share of the
        prefix re-expansion alone exceeds the budget); callers enforce their own
        minimum-progress floor on top.
        """
        L = float(history_len)
        # Hot path (once per resumed / chunked admission): with ~16 stages the
        # cost is numpy's per-call overhead, not arithmetic, so a scalar loop
        # over cached lists beats ~10 tiny array ops by 2-3x. The root is taken
        # in the cancellation-free form 2r / (b + sqrt(b^2 + 2*k_a*r)), which
        # is algebraically the quadratic root and degenerates to r / b on an
        # all-KDA stage (k_a = 0) without a separate branch.
        best = math.inf
        rems = rem_cost.tolist() if isinstance(rem_cost, np.ndarray) else rem_cost
        for b0, ke, ka, rem in zip(self._k_ge_l, self._k_e_l, self._k_a_l, rems):
            r = rem - ke * L
            if r <= 0.0:
                return 0.0
            b = b0 + ka * L
            x = 2.0 * r / (b + math.sqrt(b * b + 2.0 * ka * r))
            if x < best:
                best = x
        return float(best)


# One calibration sample: the (history_len, chunk_len) items of one timed
# microbatch and its measured forward time in ms.
CalibrationSample = Tuple[Sequence[Tuple[int, int]], float]


def _design_row(items: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Regressors of t = e + g*sum(x) + r*sum(L+x) + a*sum(x*(L+0.5x))."""
    row = np.zeros(4, dtype=np.float64)
    row[0] = 1.0
    for history_len, chunk_len in items:
        L = float(history_len)
        x = float(chunk_len)
        row[1] += x
        row[2] += L + x
        row[3] += x * (L + 0.5 * x)
    return row


def _nnls(A: np.ndarray, b: np.ndarray, max_iter: Optional[int] = None) -> np.ndarray:
    """Non-negative least squares (Lawson-Hanson). Small dense problems only."""
    m, n = A.shape
    max_iter = max_iter or 30 * n
    x = np.zeros(n)
    passive = np.zeros(n, dtype=bool)
    w = A.T @ (b - A @ x)
    tol = 1e-10 * max(1.0, float(np.abs(A).max()) * float(np.abs(b).max()))
    for _ in range(max_iter):
        if passive.all() or float(np.max(np.where(passive, -np.inf, w))) <= tol:
            break
        passive[int(np.argmax(np.where(passive, -np.inf, w)))] = True
        for _ in range(max_iter):
            z = np.zeros(n)
            z[passive] = np.linalg.lstsq(A[:, passive], b, rcond=None)[0]
            if np.all(z[passive] > 0.0):
                x = z
                break
            neg = passive & (z <= 0.0)
            alpha = float(np.min(x[neg] / (x[neg] - z[neg])))
            x = x + alpha * (z - x)
            passive &= x > 1e-15
            x[~passive] = 0.0
            if not passive.any():
                break
        w = A.T @ (b - A @ x)
    return x


def _fit_rows(rows: np.ndarray, t: np.ndarray) -> np.ndarray:
    # Relative-error objective: weight every row by 1/t so long samples do
    # not drown out the short ones that pin the constant term.
    A = rows / t[:, None]
    b = np.ones_like(t)
    # The columns span ~10 orders of magnitude (1 vs x*L); scale them to unit
    # norm or the solver loses the small terms to rounding.
    scale = np.linalg.norm(A, axis=0)
    scale[scale == 0.0] = 1.0
    return _nnls(A / scale, b) / scale


def _fit_floor(rows: np.ndarray, t: np.ndarray) -> Tuple[float, np.ndarray]:
    """Fit ``t = max(f, rows @ c)``: a per-microbatch host floor ``f`` under
    the linear model. Deep stages (24-48 layers) launch for longer than small
    microbatches run on the GPU, so their times sit flat at the launch cost
    and only then grow; one linear fit through both regimes is off by ~15%
    at either end. ``f`` is searched; ``c`` is fit on the samples above it."""
    best = None
    t_min = float(t.min())
    for f in np.linspace(0.9 * t_min, 1.6 * t_min, 36):
        above = t > 1.02 * f
        if above.sum() < 6:
            continue
        c = _fit_rows(rows[above], t[above])
        err = float(np.percentile(np.abs(np.maximum(f, rows @ c) - t) / t, 90))
        if best is None or err < best[0]:
            best = (err, float(f), c)
    if best is None:
        return 0.0, _fit_rows(rows, t)
    return best[1], best[2]


def _loo_p90(rows: np.ndarray, t: np.ndarray, use_floor: bool) -> float:
    errs: List[float] = []
    for i in range(len(t)):
        keep = np.arange(len(t)) != i
        if use_floor:
            f, c = _fit_floor(rows[keep], t[keep])
        else:
            f, c = 0.0, _fit_rows(rows[keep], t[keep])
        errs.append(abs(max(f, float(rows[i] @ c)) - t[i]) / t[i])
    return float(np.percentile(errs, 90))


def fit_stage_coefficients(
    samples: Sequence[CalibrationSample],
) -> Tuple[np.ndarray, float]:
    """Fit one stage's ``(e, g, r, a)`` in ms from timed microbatches.

    Returns the coefficients and the p90 leave-one-out relative error, which
    the caller compares against its acceptance threshold. Two fits are tried
    -- plain linear and linear over a host-launch floor (see ``_fit_floor``)
    -- and the one with the lower LOO error wins. The returned coefficients
    are the linear part either way, which is what budgets (far above the
    floor) are sized with.
    """
    if len(samples) < 6:
        raise ValueError(f"need at least 6 calibration samples, got {len(samples)}")
    rows = np.stack([_design_row(items) for items, _ in samples])
    t = np.asarray([float(ms) for _, ms in samples], dtype=np.float64)
    if not np.all(t > 0.0) or not np.all(np.isfinite(t)):
        raise ValueError(f"calibration times must be finite and > 0: {t}")
    err_lin = _loo_p90(rows, t, use_floor=False)
    err_floor = _loo_p90(rows, t, use_floor=True)
    if err_floor < err_lin:
        return _fit_floor(rows, t)[1], err_floor
    return _fit_rows(rows, t), err_lin
