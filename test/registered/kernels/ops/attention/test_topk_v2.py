"""Correctness tests for the DeepSeek-V4 (DSA indexer) JIT top-k transform v2.

The v2 kernel selects the per-row top-k of ``scores`` (ragged ``seq_lens``) and
writes the page-table transform of the selected raw indices into the output. We
validate against ``torch.topk`` with a small tolerance for boundary ties (the
fp16 coarse histogram can swap elements of equal score).

Coverage is organized around the kernel's dispatch so every template and its
boundaries are exercised:

  template      per-row seq            reached when
  --------      ----------             ------------
  trivial       seq <= k
  Register2     k < seq <= 8192        max_seq <= 8192          (level 0)
  Register4     8192 < seq <= 16384    max_seq <= 16384         (level 1)
  Streaming     16384 < seq <= floor   max_seq > 16384, non-cluster (level 2)
  Cluster       seq > floor(=65536)    max_seq > floor and batch <= 128

and two cluster dispatch shapes: the fused small-batch kernel (batch <= 30) and
the persistent-pool + main kernel (30 < batch <= 128). Boundary seq lengths
(8192/8193, 16384/16385, 65535/65536/65537) and batch sizes (30/31, 128/129) are
included explicitly, across k in {512,1024,2048} and identity/perm page tables.
"""

from __future__ import annotations

import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2, topk_transform_512_v2
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")

PAGE_SIZE = 64  # c4 page size = 256 // 4
PAGE_BITS = PAGE_SIZE.bit_length() - 1
PAGE_MASK = PAGE_SIZE - 1
MAX_PERMIT_ERROR = 5
FLOOR = 65536  # kClusterFloor

# (batch, seq) chosen to land on each template and each dispatch boundary.
FIXED_CONFIGS = [
    # --- trivial (seq <= k) ---
    (8, 256),  # trivial for every k
    (16, 1024),  # trivial for k>=1024
    # --- Register2 (level 0: max_seq <= 8192) ---
    (8, 4096),
    (8, 8192),  # reg2 upper boundary
    (128, 8192),
    (300, 8192),  # batch > 128, still level 0
    # --- Register4 (level 1: 8192 < max_seq <= 16384) ---
    (8, 8193),  # just over reg2
    (64, 16384),  # reg4 upper boundary
    (256, 16384),  # batch > 128
    # --- Streaming (level 2: max_seq > 16384, non-cluster) ---
    (8, 16385),  # just over reg4 (small batch, seq < floor => non-cluster)
    (4, 32768),
    (16, 65535),  # just under floor
    (4, 65536),  # at floor (seq == floor => non-cluster)
    (100, 65536),
    # --- Cluster, fused small-batch kernel (batch <= 30, max_seq > floor) ---
    (1, 65537),  # single row just over floor
    (2, 131072),
    (8, 98304),
    (30, 131072),  # batch == pool boundary
    # --- Cluster, persistent pool + main kernel (30 < batch <= 128) ---
    (31, 131072),  # just over small-batch
    (40, 262144),  # N > pool of 30 => round-robin
    (64, 196608),
    (128, 131072),  # cluster batch upper boundary
    # --- batch > 128 => non-cluster streaming even at long ctx ---
    (129, 131072),
    (200, 262144),
]


def _assert_topk_close(scores_cpu, ref_raw, our_raw, bs, seq_lens, k):
    """Set-compare our top-k raw indices vs torch's, tolerating equal-score ties."""
    bad = 0
    for i in range(bs):
        L = int(seq_lens[i])
        ref, our = set(ref_raw[i]), set(our_raw[i])
        more, less = our - ref, ref - our
        if more or less:
            mv = sorted(scores_cpu[i, list(more)].tolist())
            lv = sorted(scores_cpu[i, list(less)].tolist())
            if mv != lv:  # not merely a tie swap -> genuine error
                bad += len(more)
                print(
                    f"b={i} L={L} k={k}: more={list(more)[:4]} less={list(less)[:4]} mv={mv[:3]} lv={lv[:3]}"
                )
        assert len(our) == min(
            k, L
        ), f"b={i} L={L} k={k}: {len(our)} valid != {min(k, L)}"
    assert bad <= MAX_PERMIT_ERROR, f"{bad=} > {MAX_PERMIT_ERROR}"


def _make_page_table(batch, num_pages, mode, device, per_row=False):
    if mode == "identity":
        pt = torch.arange(num_pages, dtype=torch.int32, device=device)
        full = pt.unsqueeze(0).expand(batch, -1).contiguous()
        inv = pt.unsqueeze(0).expand(batch, -1).cpu()
        return full, inv
    # permutation (optionally a distinct permutation per row)
    rows = batch if per_row else 1
    full = torch.stack(
        [torch.randperm(num_pages, device=device) for _ in range(rows)]
    ).to(torch.int32)
    inv = torch.empty_like(full)
    ar = torch.arange(num_pages, dtype=torch.int32, device=device)
    for r in range(rows):
        inv[r, full[r].long()] = ar
    if not per_row:
        full = full.expand(batch, -1).contiguous()
        inv = inv.expand(batch, -1)
    return full, inv.cpu()


def _invert(out_row, inv_row):
    """Undo page_to_indices for one row's page indices (drop -1 padding)."""
    return [
        (int(inv_row[v >> PAGE_BITS]) << PAGE_BITS) | (v & PAGE_MASK)
        for v in out_row
        if v != -1
    ]


def _reference(scores, seq_lens, k):
    """torch.topk reference indices per row (trivial rows -> all positions)."""
    ref = []
    for i in range(scores.shape[0]):
        L = int(seq_lens[i])
        if L <= k:
            ref.append(list(range(L)))
        else:
            ref.append(
                torch.topk(scores[i, :L], k, sorted=False).indices.cpu().tolist()
            )
    return ref


def _run(scores, seq_lens, page_table, inv_cpu, k):
    batch = scores.shape[0]
    out = torch.full((batch, k), -1, dtype=torch.int32, device=scores.device)
    metadata = plan_topk_v2(seq_lens)
    topk_transform_512_v2(scores, seq_lens, page_table, out, PAGE_SIZE, metadata)
    torch.cuda.synchronize()
    out_cpu = out.cpu().tolist()
    return [_invert(out_cpu[i], inv_cpu[i]) for i in range(batch)]


def _run_raw(scores, seq_lens, page_table, k):
    """Run the kernel and return its optional raw (pre-transform) top-k index
    output per row, dropping -1 padding -- the selected positions themselves,
    NOT the page-table transform of them."""
    batch = scores.shape[0]
    out = torch.full((batch, k), -1, dtype=torch.int32, device=scores.device)
    raw = torch.full((batch, k), -1, dtype=torch.int32, device=scores.device)
    metadata = plan_topk_v2(seq_lens)
    topk_transform_512_v2(scores, seq_lens, page_table, out, PAGE_SIZE, metadata, raw)
    torch.cuda.synchronize()
    raw_cpu = raw.cpu().tolist()
    return [[v for v in raw_cpu[i] if v != -1] for i in range(batch)]


@pytest.mark.parametrize("page_mode", ["identity", "perm"])
@pytest.mark.parametrize("k", [512, 1024, 2048])
@pytest.mark.parametrize("batch,seq", FIXED_CONFIGS)
@torch.inference_mode()
def test_topk_v2(batch: int, seq: int, k: int, page_mode: str) -> None:
    torch.manual_seed(batch * 100003 + seq * 7 + k)
    device = "cuda"
    # Pad the row stride to a multiple of 4 (16-byte vectorized load) while keeping
    # the exact seq_len -- this also exercises the scalar-tail path for odd seq.
    width = (seq + 3) & ~3
    scores = torch.randn(batch, width, dtype=torch.float32, device=device)[:, :seq]
    seq_lens = torch.full((batch,), seq, dtype=torch.int32, device=device)
    num_pages = (seq + PAGE_SIZE - 1) // PAGE_SIZE
    page_table, inv_cpu = _make_page_table(batch, num_pages, page_mode, device)

    our_raw = _run(scores, seq_lens, page_table, inv_cpu, k)
    ref_raw = _reference(scores, seq_lens, k)
    _assert_topk_close(scores.cpu(), ref_raw, our_raw, batch, seq_lens.cpu(), k)


@pytest.mark.parametrize("k", [512, 1024, 2048])
@pytest.mark.parametrize(
    "batch,shape",
    [
        (20, "small_batch"),  # fused small-batch kernel (<= pool of 30)
        (64, "persistent"),  # persistent pool + main kernel
        (128, "persistent"),  # cluster batch boundary
    ],
)
@pytest.mark.parametrize("per_row_pt", [False, True])
@torch.inference_mode()
def test_topk_v2_ragged(batch: int, shape: str, k: int, per_row_pt: bool) -> None:
    """Ragged lengths spanning trivial..cluster in one launch, both dispatch shapes.

    ``per_row_pt`` gives each row a distinct page-table permutation, exercising
    the per-batch page_table indexing (batch_id stride) rather than a shared one.
    """
    torch.manual_seed(7777 + batch + k + int(per_row_pt))
    device = "cuda"
    seq = 262144
    scores = torch.randn(batch, seq, dtype=torch.float32, device=device)
    # span every path; guarantee at least one > floor row so cluster dispatch fires
    buckets = [max(1, k // 2), k, 4096, 12000, 40000, 65536, 98304, 262144]
    g = torch.Generator(device="cpu").manual_seed(batch + k)
    lengths = torch.tensor(
        [
            buckets[int(torch.randint(0, len(buckets), (1,), generator=g))]
            for _ in range(batch)
        ],
        dtype=torch.int32,
        device=device,
    )
    lengths[0] = max(1, k // 2)  # a trivial row
    lengths[1] = 262144  # a long (cluster) row
    num_pages = (seq + PAGE_SIZE - 1) // PAGE_SIZE
    page_table, inv_cpu = _make_page_table(
        batch, num_pages, "perm", device, per_row=per_row_pt
    )

    our_raw = _run(scores, lengths, page_table, inv_cpu, k)
    ref_raw = _reference(scores, lengths, k)
    _assert_topk_close(scores.cpu(), ref_raw, our_raw, batch, lengths.cpu(), k)


@pytest.mark.parametrize("page_mode", ["identity", "perm"])
@pytest.mark.parametrize(
    "batch,seq",
    [
        (8, 256),  # trivial
        (8, 4096),  # register
        (4, 131072),  # fused small-batch cluster
        (64, 131072),  # persistent cluster + main<3> epilogue
        (256, 131072),  # non-cluster streaming
    ],
)
@torch.inference_mode()
def test_topk_v2_raw_indices(batch: int, seq: int, page_mode: str) -> None:
    """The optional raw-index output must be the pre-transform position of each
    transformed output slot (out[j] == page_to_indices(raw[j])), and -1 aligns."""
    k = 512
    torch.manual_seed(batch * 131 + seq)
    device = "cuda"
    width = (seq + 3) & ~3
    scores = torch.randn(batch, width, dtype=torch.float32, device=device)[:, :seq]
    seq_lens = torch.full((batch,), seq, dtype=torch.int32, device=device)
    num_pages = (seq + PAGE_SIZE - 1) // PAGE_SIZE
    page_table, inv_cpu = _make_page_table(batch, num_pages, page_mode, device)
    out = torch.full((batch, k), -1, dtype=torch.int32, device=device)
    raw = torch.full((batch, k), -1, dtype=torch.int32, device=device)

    metadata = plan_topk_v2(seq_lens)
    topk_transform_512_v2(scores, seq_lens, page_table, out, PAGE_SIZE, metadata, raw)
    torch.cuda.synchronize()

    out_cpu, raw_cpu = out.cpu().tolist(), raw.cpu().tolist()
    for i in range(batch):
        for j in range(k):
            o, r = out_cpu[i][j], raw_cpu[i][j]
            if o == -1:
                assert r == -1, f"b={i} j={j}: out=-1 but raw={r}"
            else:
                inv = (int(inv_cpu[i][o >> PAGE_BITS]) << PAGE_BITS) | (o & PAGE_MASK)
                assert r == inv, f"b={i} j={j}: raw={r} != inverse(out)={inv}"


@pytest.mark.parametrize("k", [512, 1024, 2048])
@pytest.mark.parametrize("batch,seq", FIXED_CONFIGS)
@torch.inference_mode()
def test_topk_v2_output_indices(batch: int, seq: int, k: int) -> None:
    """Validate the raw (pre-transform) index output DIRECTLY against torch.topk.

    Unlike ``test_topk_v2`` -- which checks the page-transformed output and inverts
    it through the page table -- this exercises the selected indices themselves, so
    it isolates the top-k selection from the page-table transform. A permuted page
    table is used so raw != out, catching any bug that leaks transformed page
    indices into the raw buffer. Covers every dispatch template/boundary.
    """
    torch.manual_seed(batch * 100003 + seq * 7 + k + 1)
    device = "cuda"
    width = (seq + 3) & ~3
    scores = torch.randn(batch, width, dtype=torch.float32, device=device)[:, :seq]
    seq_lens = torch.full((batch,), seq, dtype=torch.int32, device=device)
    num_pages = (seq + PAGE_SIZE - 1) // PAGE_SIZE
    page_table, _ = _make_page_table(batch, num_pages, "perm", device)

    our_raw = _run_raw(scores, seq_lens, page_table, k)
    ref_raw = _reference(scores, seq_lens, k)
    _assert_topk_close(scores.cpu(), ref_raw, our_raw, batch, seq_lens.cpu(), k)


def _tight_cluster(batch: int, seq: int, device: str) -> torch.Tensor:
    """Scores that all land in ONE fp16 coarse bin (both the 12-bit register/
    streaming bins and the 10-bit cluster bins), in shuffled order.

    [1, 1 + 2^-7) sits inside a single fp16 bin of width 2^-6, so every element
    is a threshold-bin candidate. Positions are shuffled per row so the order
    in which threads claim tie-buffer slots does not follow the score order.
    """
    span = 2.0**-7
    base = 1.0 + torch.arange(seq, dtype=torch.float64, device=device) * (span / seq)
    rows = [base[torch.randperm(seq, device=device)] for _ in range(batch)]
    return torch.stack(rows).to(torch.float32)


def _assert_topk_values_exact(scores_cpu, our_raw, seq_lens, k):
    """Selected scores must equal torch.topk's as a multiset (index swaps among
    equal scores are fine; any lower-scoring pick is not)."""
    for i in range(scores_cpu.shape[0]):
        L = int(seq_lens[i])
        n = min(k, L)
        assert len(our_raw[i]) == n, f"b={i} L={L} k={k}: {len(our_raw[i])} != {n}"
        ref = torch.topk(scores_cpu[i, :L], n).values.sort().values
        ours = scores_cpu[i, our_raw[i]].sort().values
        # Picks strictly below the k-th largest score are genuine misses; equal
        # scores at the cutoff are interchangeable.
        below_cutoff = int((ours < ref[0]).sum())
        assert torch.equal(ours, ref), (
            f"b={i} L={L} k={k}: {below_cutoff}/{n} picks below the top-k "
            f"cutoff (lowest picked {ours[0].item():.9f} vs "
            f"cutoff {ref[0].item():.9f})"
        )


@pytest.mark.parametrize("k", [512, 2048])
@pytest.mark.parametrize(
    "batch,seq",
    [
        (4, 4096),  # Register2
        (4, 16384),  # Register4
        (4, 65536),  # Streaming
        (2, 131072),  # Cluster, fused small-batch kernel
        (40, 131072),  # Cluster, persistent pool + main kernel
    ],
)
@torch.inference_mode()
def test_topk_v2_threshold_bin_overflow(batch: int, seq: int, k: int) -> None:
    """More threshold-bin candidates than the kMaxNumTie (2048) tie buffer.

    Mirrors the overflow case sgl-project/sglang#37625 fixed for the legacy and
    v1 selectors: when the coarse threshold bin holds more candidates than the
    tie buffer, the extras must not be dropped before full scores are compared.
    """
    torch.manual_seed(batch * 1009 + seq + k)
    device = "cuda"
    scores = _tight_cluster(batch, seq, device)
    seq_lens = torch.full((batch,), seq, dtype=torch.int32, device=device)
    num_pages = (seq + PAGE_SIZE - 1) // PAGE_SIZE
    page_table, _ = _make_page_table(batch, num_pages, "perm", device)

    our_raw = _run_raw(scores, seq_lens, page_table, k)
    _assert_topk_values_exact(scores.cpu(), our_raw, seq_lens.cpu(), k)


def _overflow_scores(pattern: str, batch: int, seq: int, device: str) -> torch.Tensor:
    if pattern == "constant":
        # One fp32 key holds every candidate: the exact search cannot narrow
        # below kMaxNumTie, so any `remain` of them must be emitted.
        return torch.full((batch, seq), 0.5, dtype=torch.float32, device=device)
    if pattern == "around_zero":
        # The coarse bin around 0 spans ~2^31 fp32 keys (both signs, every
        # small exponent): several 12-bit levels, plus the +0/-0 bound case.
        vals = (torch.rand(batch, seq, device=device) - 0.5) * 1e-6
        vals[:, ::7] = 0.0
        vals[:, 3::7] = -0.0
        return vals
    # "few_distinct": each value repeated many times -- ties straddle the cutoff.
    levels = 1.0 + torch.arange(64, dtype=torch.float32, device=device) * 2.0**-14
    return levels[torch.randint(0, 64, (batch, seq), device=device)]


@pytest.mark.parametrize("pattern", ["constant", "around_zero", "few_distinct"])
@pytest.mark.parametrize("k", [512, 2048])
@pytest.mark.parametrize("batch,seq", [(4, 16384), (4, 65536), (2, 131072), (40, 131072)])
@torch.inference_mode()
def test_topk_v2_threshold_bin_overflow_patterns(batch: int, seq: int, k: int, pattern: str) -> None:
    """Overflow corner cases: a single key, multi-level narrowing around 0 and
    heavy exact ties at the cutoff, on the register, streaming and both
    cluster dispatch shapes."""
    torch.manual_seed(batch * 7919 + seq + k)
    device = "cuda"
    scores = _overflow_scores(pattern, batch, seq, device)
    seq_lens = torch.full((batch,), seq, dtype=torch.int32, device=device)
    num_pages = (seq + PAGE_SIZE - 1) // PAGE_SIZE
    page_table, _ = _make_page_table(batch, num_pages, "perm", device)

    our_raw = _run_raw(scores, seq_lens, page_table, k)
    for i in range(batch):
        assert len(set(our_raw[i])) == len(our_raw[i]), f"b={i}: duplicate indices"
    _assert_topk_values_exact(scores.cpu(), our_raw, seq_lens.cpu(), k)


def _midpoint_mismatch_row(seq: int, hist_bits: int):
    """A row whose fp32 tie set overflows while the fp16 histogram's does not.

    The collect pass classifies by fp32 bin boundaries, which are midpoints
    between adjacent fp16 values. For negative scores the lower boundary of a
    bin rounds (ties-to-even) DOWN into the bin below, so 3000 copies of it are
    fp32 "ties" yet sit outside the histogram's threshold bin.
    """
    shift = 16 - hist_bits
    # Negative fp16 near -1: ordered key = ~bits; pick a bin-start key.
    bits = torch.tensor([-1.0], dtype=torch.float16).view(torch.int16).item() & 0xFFFF
    key = ((~bits) & 0xFFFF) >> shift << shift
    to_f = lambda k: torch.tensor([(~k) & 0xFFFF], dtype=torch.int32).to(torch.int16).view(torch.float16).float().item()
    lower, start = to_f(key - 1), to_f(key)
    mid = (lower + start) / 2
    mid32 = torch.tensor([mid], dtype=torch.float32)
    assert mid32.half().float().item() == lower, "midpoint must round into the lower bin"
    interior = [to_f(key + j) for j in range(1, (1 << shift) - 1)]
    row = torch.full((seq,), -1000.0)
    above = 10.0 + torch.arange(400, dtype=torch.float32) * 1e-3
    row[:400] = above
    row[400:600] = torch.tensor(interior, dtype=torch.float32).repeat(200 // len(interior) + 1)[:200]
    row[600:3600] = mid32.item()
    return row[torch.randperm(seq)]


@pytest.mark.parametrize("batch,seq", [(4, 16384), (4, 32768), (2, 131072), (40, 131072)])
@torch.inference_mode()
def test_topk_v2_boundary_midpoint_mismatch(batch: int, seq: int) -> None:
    """The fp32 collect overflows the tie buffer although the histogram's
    threshold bin does not (covers the cluster path's rank-0 fallback)."""
    torch.manual_seed(batch + seq)
    k = 512
    hist_bits = 10 if seq > 65536 else 12
    scores = torch.stack([_midpoint_mismatch_row(seq, hist_bits) for _ in range(batch)]).cuda()
    seq_lens = torch.full((batch,), seq, dtype=torch.int32, device="cuda")
    num_pages = (seq + PAGE_SIZE - 1) // PAGE_SIZE
    page_table, _ = _make_page_table(batch, num_pages, "perm", "cuda")

    our_raw = _run_raw(scores, seq_lens, page_table, k)
    _assert_topk_values_exact(scores.cpu(), our_raw, seq_lens.cpu(), k)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
