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

This module is intentionally dependency-free (numpy only) so it can be unit
tested without importing the rest of sglang.
"""

import logging
from typing import Optional, Sequence

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
        # Budget for one microbatch, the same envelope on every stage.
        self.target_cost: Optional[np.ndarray] = None
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
        # solve_x runs per admitted request per scheduling iteration, so keep
        # its hot path to one allocation-free vectorised expression: hoist the
        # constant sum and note when no stage degenerates to linear (only an
        # all-KDA stage does, which is rare).
        self._k_ge = self.k_g + self.k_e
        self._all_quadratic = bool(np.all(self.k_a > 0.0))
        logger.info(
            "[ChunkCostModel] %d stages, GEMM-equivalent-token coefficients "
            "(1 unit = one new token through the heaviest stage's GEMMs): "
            "k_g=%s, k_e=%s, k_a=%s",
            len(k_g),
            np.array2string(self.k_g, precision=3),
            np.array2string(self.k_e, precision=4),
            np.array2string(self.k_a, precision=3, formatter={"float": "{:.3e}".format}),
        )

    def req_cost(self, history_len: int, chunk_len: int) -> np.ndarray:
        """Per-stage cost of prefilling chunk_len tokens on history_len cached
        tokens."""
        if chunk_len <= 0:
            return np.zeros_like(self.k_g)
        L = float(history_len)
        x = float(chunk_len)
        return self.k_g * x + self.k_e * (L + x) + self.k_a * x * (L + 0.5 * x)

    def reexpand_cost(self, history_len: int) -> float:
        """Cost of re-expanding the latent prefix once, on the stage that pays
        the most for it. This is repaid by every additional chunk, so it is the
        price of splitting a long-prefix request one more time."""
        return float(self.k_e.max()) * float(history_len)

    def set_target_cost(self, base_chunk_size: int):
        """Budget for one microbatch: the *pipeline period* of a fresh chunk of
        the configured base size, i.e. max_s cost_s(0, base_chunk_size).

        Deliberately not each stage's own cost_s(0, base). The period is set by
        the slowest stage, so a stage that is cheap on a fresh chunk has real
        slack; capping it at its own low figure is a tighter constraint than
        "no slower than a fresh base chunk" with nothing behind it. On K3 that
        cost 2.9% of the new tokens per batch: cost_s(0, 24K) spans 25% across
        the 16 stages, and long-prefix chunks ended up bound by a cheap-GEMM /
        MLA-heavy stage that was never the bottleneck.

        With no prefix hit this still reproduces the static token budget
        exactly, on the stage that sets the period (see solve_x).
        """
        per_stage = self.req_cost(0, base_chunk_size)
        if not np.all(per_stage > 0):
            raise ValueError(
                f"cost-model target must be positive per stage: {per_stage}"
            )
        self.target_cost = np.full_like(per_stage, per_stage.max())
        logger.info(
            "[ChunkCostModel] Target cost per microbatch (base_chunk_size=%d): "
            "%.0f GEMM-equivalent tokens, from per-stage %s (max/min=%.2f, i.e. "
            "the layer partition is that imbalanced on a fresh chunk)",
            base_chunk_size,
            per_stage.max(),
            np.array2string(per_stage, precision=0),
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
        rem = np.asarray(rem_cost, dtype=np.float64)
        r = rem - self.k_e * L
        if np.any(r <= 0.0):
            return 0.0
        b = self._k_ge + self.k_a * L
        # a = 0.5 * k_a, so 4*a*r = 2*k_a*r and 2*a = k_a.
        if self._all_quadratic:
            return float(
                ((-b + np.sqrt(b * b + 2.0 * self.k_a * r)) / self.k_a).min()
            )
        a = 0.5 * self.k_a
        x = np.full_like(r, np.inf)
        quad = a > 0.0
        bq, aq, rq = b[quad], a[quad], r[quad]
        x[quad] = (-bq + np.sqrt(bq * bq + 4.0 * aq * rq)) / (2.0 * aq)
        lin = ~quad & (b > 0.0)
        if np.any(lin):
            x[lin] = r[lin] / b[lin]
        return float(x.min())
