"""Distribution-equivalence + no-exp_noise-read test for the flag-gated,
graph-safe DSpark inline-RNG sampling fusion (SGLANG_FUSE_DSPARK_INLINE_RNG).

What it proves
--------------
1. For identical logits, the ORIGINAL precomputed-noise path (exp_noise drawn
   by exponential_()) and the INLINE-RNG path (philox ``-log(u)`` produced
   inside the sampling kernel) sample draft tokens from the SAME categorical
   distribution softmax(logits): both empirical pmfs match the theory and their
   total-variation distance stays within sampling noise. Bit-exactness is NOT
   required (draft proposals; target verify guards correctness).
2. The inline path never reads exp_noise: a NaN-poisoned exp_noise passed
   alongside rng_counter is ignored (identical tokens, no NaN out), and the
   path also runs with exp_noise=None.
3. The graph-safe counter drives the draw: same counter -> identical noise
   (replay-stable), different counter -> fresh noise (per-decode-step redraw).

Runs on CPU (torch reference path, so it is safe anywhere and never contends
with a pod GPU). The ``check_*`` helpers take a device argument, so a developer
can still point them at ``torch.device("cuda")`` to exercise the real Triton
kernels manually.
"""

import math
import unittest

import torch

from sglang.kernels.ops.speculative.dspark.dspark_draft_model import SampleStepTokens
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

_P_THRESHOLD = 0.03  # max |empirical pmf - softmax(logits)| per token
_N_DEFAULT = 20000


def _make_logits(V: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(1234)
    # Moderately peaked, non-uniform logits so the pmf has real structure.
    return torch.randn(V, generator=g) * 1.5


def _collect(
    logits_vec,
    *,
    n,
    device,
    inline,
    counter_val=12345,
    rng_step=0,
    rng_num_steps=1,
    poison=False,
):
    """Draw `n` draft tokens for a single logit vector via one batched call.

    Each of the `n` rows is the same logits but gets independent noise, so the
    row axis doubles as the sample axis. `inline=False` is the original path
    (exp_noise via exponential_()); `inline=True` is the fused philox path.
    """
    V = logits_vec.shape[0]
    logits = logits_vec.to(device).unsqueeze(0).expand(n, V).contiguous()
    temps = torch.ones(n, dtype=torch.float32, device=device)
    gmask = torch.zeros(n, dtype=torch.bool, device=device)
    if inline:
        counter = torch.tensor([counter_val], dtype=torch.int64, device=device)
        exp_noise = (
            torch.full((n, V), float("nan"), dtype=torch.float32, device=device)
            if poison
            else None
        )
        return SampleStepTokens.execute(
            step_logits=logits,
            temperatures=temps,
            greedy_mask=gmask,
            exp_noise=exp_noise,
            rng_counter=counter,
            rng_step=rng_step,
            rng_num_steps=rng_num_steps,
        )
    exp_noise = torch.empty((n, V), dtype=torch.float32, device=device).exponential_(1)
    return SampleStepTokens.execute(
        step_logits=logits,
        temperatures=temps,
        greedy_mask=gmask,
        exp_noise=exp_noise,
    )


def _pmf(tokens, V):
    return torch.bincount(tokens.flatten().cpu(), minlength=V).float() / tokens.numel()


def _tv(a, b):
    return 0.5 * (a - b).abs().sum().item()


def check_distribution_equivalence(device, *, V=32, n=_N_DEFAULT):
    """Both paths sample from softmax(logits): each empirical pmf matches theory
    and their mutual TV distance stays within multi-sigma sampling noise."""
    logits = _make_logits(V)
    theory = torch.softmax(logits.float(), dim=-1)
    toks_pre = _collect(logits, n=n, device=device, inline=False)
    toks_in = _collect(logits, n=n, device=device, inline=True)
    pmf_pre = _pmf(toks_pre, V)
    pmf_in = _pmf(toks_in, V)
    tv = _tv(pmf_pre, pmf_in)
    err_pre = (pmf_pre - theory).abs().max().item()
    err_in = (pmf_in - theory).abs().max().item()
    # Expected TV between two independent n-sample empirical pmfs (Gaussian
    # approx of |phat - qhat| summed over tokens); allow a generous multiple so
    # the bound is robust to the support size without ever being vacuous.
    tv_exp = (theory * (1 - theory)).sqrt().sum().item() / math.sqrt(n * math.pi)
    tv_tol = max(0.02, 4.0 * tv_exp)
    print(
        f"[dist V={V} n={n} dev={device}] TV(pre,inline)={tv:.4f} "
        f"(tol={tv_tol:.4f}) maxerr_pre={err_pre:.4f} maxerr_inline={err_in:.4f}"
    )
    assert tv < tv_tol, f"pmf mismatch between paths: TV={tv:.4f} >= {tv_tol:.4f}"
    assert err_pre < _P_THRESHOLD, f"precomputed vs softmax off by {err_pre:.4f}"
    assert err_in < _P_THRESHOLD, f"inline vs softmax off by {err_in:.4f}"


def check_inline_ignores_exp_noise(device, *, V=48, n=1024):
    """Inline path must not read exp_noise_ptr: a NaN-poisoned buffer passed
    with rng_counter is ignored, so the result equals the exp_noise=None run
    and carries no NaN."""
    logits = _make_logits(V)
    kw = dict(
        n=n, device=device, inline=True, counter_val=777, rng_step=3, rng_num_steps=5
    )
    toks_none = _collect(logits, poison=False, **kw)
    toks_pois = _collect(logits, poison=True, **kw)
    assert torch.equal(toks_none, toks_pois), (
        "inline path consulted the NaN-poisoned exp_noise buffer"
    )
    assert not torch.isnan(toks_none.float()).any(), "inline draw produced NaN tokens"
    assert int(toks_none.min()) >= 0 and int(toks_none.max()) < V
    print(f"[no-exp-noise-read dev={device}] poisoned buffer ignored; tokens valid")


def check_counter_drives_redraw(device, *, V=48, n=4096):
    """Graph-safe counter semantics: same counter -> identical noise (replay is
    deterministic), different counter -> a materially different draw."""
    logits = _make_logits(V)
    theory = torch.softmax(logits.float(), dim=-1)
    a = _collect(logits, n=n, device=device, inline=True, counter_val=100)
    a2 = _collect(logits, n=n, device=device, inline=True, counter_val=100)
    b = _collect(logits, n=n, device=device, inline=True, counter_val=101)
    assert torch.equal(a, a2), "same counter must give identical noise (replay-stable)"
    frac_diff = (a != b).float().mean().item()
    # Expected fraction of rows whose argmax changes when the noise is redrawn.
    expected_change = 1.0 - (theory**2).sum().item()
    print(
        f"[counter dev={device}] same-counter identical; diff-counter changed "
        f"{frac_diff * 100:.1f}% of tokens (expect ~{expected_change * 100:.1f}%)"
    )
    assert frac_diff > 0.5 * expected_change, "different counter barely changed the draw"


def check_greedy_unaffected(device, *, V=64, n=256):
    """Greedy rows ignore the noise entirely and return argmax(logits), the
    same under both paths; this is the part that must stay exact."""
    logits = _make_logits(V).to(device)
    logits_b = logits.unsqueeze(0).expand(n, V).contiguous()
    temps = torch.ones(n, dtype=torch.float32, device=device)
    gmask = torch.ones(n, dtype=torch.bool, device=device)  # all greedy
    counter = torch.tensor([5], dtype=torch.int64, device=device)
    toks = SampleStepTokens.execute(
        step_logits=logits_b,
        temperatures=temps,
        greedy_mask=gmask,
        rng_counter=counter,
        rng_step=0,
        rng_num_steps=1,
    )
    expect = int(torch.argmax(logits))
    assert int(toks.min()) == expect and int(toks.max()) == expect, (
        "greedy rows must ignore noise and return argmax(logits)"
    )
    print(f"[greedy dev={device}] all rows -> argmax={expect} regardless of noise")


class TestDsparkInlineRng(unittest.TestCase):
    """CPU entry points (safe to run anywhere; no GPU required)."""

    device = torch.device("cpu")

    def test_env_flag_default_off(self):
        self.assertTrue(hasattr(envs, "SGLANG_FUSE_DSPARK_INLINE_RNG"))
        self.assertIs(envs.SGLANG_FUSE_DSPARK_INLINE_RNG.get(), False)

    def test_distribution_equivalence_cpu(self):
        check_distribution_equivalence(self.device)

    def test_inline_ignores_exp_noise_cpu(self):
        check_inline_ignores_exp_noise(self.device)

    def test_counter_drives_redraw_cpu(self):
        check_counter_drives_redraw(self.device)

    def test_greedy_unaffected_cpu(self):
        check_greedy_unaffected(self.device)


if __name__ == "__main__":
    unittest.main()
