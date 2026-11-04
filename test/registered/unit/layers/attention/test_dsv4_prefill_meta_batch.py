"""Equivalence tests for SGLANG_OPT_PREFILL_META_BATCH in the DSv4 attention metadata.

The batched paths (paired c1/c2 low-ratio metadata and sparse buffers, grouped CP
reindex gather) must produce the same values, dtypes, shapes and contiguity as the
per-ratio / per-field paths they replace.
"""

import types

import pytest

torch = pytest.importorskip("torch")

from sglang.srt.layers.attention import deepseek_v4_backend as be  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def _inputs(num_tokens=37, num_write=29, seed=0):
    """Random per-token causal lengths and write slots, int32 like the backend's."""
    g = torch.Generator().manual_seed(seed)
    seq = torch.randint(0, 300, (num_tokens,), generator=g, dtype=torch.int32)
    raw = torch.randint(0, 10_000, (num_write,), generator=g, dtype=torch.int32)
    return seq, raw


@pytest.mark.parametrize("num_write", [0, 1, 29, 37])
def test_low_ratio_metadata_pair_matches_single_ratio(num_write):
    """Row r - 1 of the paired result equals the single-ratio result for ratio r."""
    seq, raw = _inputs(num_write=num_write)
    out_loc, clamp1 = be._low_ratio_compression_metadata_pair(seq, raw)
    for row, ratio in enumerate((1, 2)):
        ref_loc, ref_clamp = be._low_ratio_compression_metadata(ratio, seq, raw)
        assert out_loc[row].dtype == ref_loc.dtype
        assert clamp1[row].dtype == ref_clamp.dtype
        assert torch.equal(out_loc[row], ref_loc)
        assert torch.equal(clamp1[row], ref_clamp)
        assert out_loc[row].is_contiguous() and clamp1[row].is_contiguous()


@pytest.mark.parametrize("is_prefill", [True, False])
@pytest.mark.parametrize("topk", [512, 1000])
def test_low_ratio_sparse_buffers_pair_matches_single(is_prefill, topk):
    """Lengths, -1 page buffers at the padded width and raw buffer shapes match."""
    seq, raw = _inputs()
    _, clamp1 = be._low_ratio_compression_metadata_pair(seq, raw)
    rows = (clamp1[0].clone(), clamp1[1].clone())
    lengths, pages, raws = be._low_ratio_sparse_buffers_pair(rows, topk, is_prefill)
    for i in range(2):
        ref_len, ref_pages, ref_raw = be._low_ratio_sparse_buffers(
            rows[i], topk, is_prefill
        )
        assert torch.equal(lengths[i], ref_len)
        assert torch.equal(pages[i], ref_pages)
        assert pages[i].is_contiguous()
        if is_prefill:
            assert raws[i].shape == ref_raw.shape and raws[i].dtype == ref_raw.dtype
            assert raws[i].is_contiguous()
        else:
            assert raws[i] is None and ref_raw is None


def _fake_metadata(num_tokens, seed=0):
    """Object carrying every CP-reindexed field the grouped gather touches."""
    g = torch.Generator().manual_seed(seed)

    def ints(*shape):
        """Random int32 tensor of ``shape``."""
        return torch.randint(0, 1000, shape, generator=g, dtype=torch.int32)

    md = types.SimpleNamespace(
        seq_lens_casual=ints(num_tokens),
        positions_casual=ints(num_tokens),
        swa_page_indices=ints(num_tokens, 128),
        swa_topk_lengths=ints(num_tokens),
        page_table=ints(num_tokens, 9),
        c4_topk_lengths_raw=ints(num_tokens),
        c4_topk_lengths_clamp1=ints(num_tokens),
        c128_page_indices=ints(num_tokens, 64),
        c128_topk_lengths_clamp1=ints(num_tokens),
        c1_topk_lengths_clamp1=ints(num_tokens),
        c2_topk_lengths_clamp1=ints(num_tokens),
        cut_swa_page_indices=None,
        cut_swa_topk_lengths=None,
    )
    md._CP_REINDEX_FIELDS = be.DSV4AttnMetadata._CP_REINDEX_FIELDS
    md._CP_REINDEX_OPTIONAL_FIELDS = be.DSV4AttnMetadata._CP_REINDEX_OPTIONAL_FIELDS
    return md


def test_grouped_cp_reindex_matches_per_field_gather():
    """One gather per (dtype, row shape) group equals a per-field ``val[idx]``."""
    num_tokens = 41
    local_index = torch.tensor([0, 4, 8, 13, 17, 22, 40], dtype=torch.int64)
    md = _fake_metadata(num_tokens)
    names = md._CP_REINDEX_FIELDS + md._CP_REINDEX_OPTIONAL_FIELDS
    ref = {
        n: getattr(md, n)[local_index].contiguous()
        for n in names
        if getattr(md, n) is not None
    }
    be.DSV4AttnMetadata._cp_reindex_grouped(md, local_index)
    for n in names:
        got = getattr(md, n)
        if n not in ref:
            assert got is None
            continue
        assert got.dtype == ref[n].dtype and got.shape == ref[n].shape
        assert got.is_contiguous()
        assert torch.equal(got, ref[n])
