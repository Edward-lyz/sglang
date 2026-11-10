"""Kimi-K3 KDA short_conv: fused q/k/v causal conv1d + SiLU (SM100+).

Gate: the kernel must be BIT-EXACT (torch.equal, not allclose) against the
project Triton `causal_conv1d_fn` path it replaces, because it deliberately
reproduces that path's per-channel fp32 rounding (even/odd fma asymmetry) and
its inline-asm SiLU. `assert_close` here would hide exactly the regressions
this test exists to catch.

Pinned failure modes:
- the interleaved input view: `x` is the (dim, T) stride-(1, mixed_qkv.stride(0))
  slice of mixed_qkv and must never be made contiguous;
- the PRODUCTION row-strided layout: `mixed_qkv` is
  `torch.split(fused_qkvg_states, [3*dim, dim], dim=-1)[0]`, i.e. (T, 3*dim) with
  stride (4*dim, 1) and `is_contiguous() is False` -- at attn_tp 1 as well as 8.
  A guard requiring contiguity made this kernel unreachable in production for a
  whole integration cycle, so `strided=True` (which asserts non-contiguity, so
  the case cannot silently regress to a contiguous one) is exercised for every
  case, every attn_tp dim and in `covered`;
- the in-place conv-state pool update, which is part of the output: after each
  sequence its slot holds that sequence's last 3 PRE-activation input columns,
  shifted left -- covered for L >= 3 and, separately, for L < 3 where part of
  the old state has to survive (and where has_initial_state decides between the
  old state and zeros);
- ragged tails: sequence lengths that are not a multiple of the token tile
  (NT * TG = 80) hit the separate FULL=false template instantiation and its
  divergent early exit; `short_odd` is the only case that does;
- pad slots: `cache_indices[s] == pad_slot_id` sequences must be skipped
  entirely (no output columns, state untouched), so their output columns are
  uninitialised in BOTH paths and are excluded from the comparison;
- has_initial_state on/off within one batch, which selects between the cached
  3-tap history and zeros.
"""

import unittest

import torch

import sglang.kernels.ops.kimi_k3.kda_short_conv as kda_short_conv_mod
from sglang.kernels.ops.kimi_k3.kda_short_conv import (
    CHANNEL_TILE,
    SUPPORTED_WIDTH,
    covered,
    kimi_k3_kda_short_conv,
    reject_reason,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import PAD_SLOT_ID, causal_conv1d_fn
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

# num_heads=96 * head_dim=128 at attn_tp_size=1. attn_tp 2/4/8 give
# 6144/3072/1536; test_attn_tp_dims_bit_exact covers those, because the grid's
# z extent is dim / CHANNEL_TILE and a launch-geometry change alone has been
# observed to break bit-exactness while leaving the conv-state pool correct.
_DIM = 12288
_TOTAL_DIM = 3 * _DIM
_WIDTH = SUPPORTED_WIDTH
_NUM_LINES = 8
# per-projection dim for attn_tp_size 1 / 2 / 4 / 8
_ATTN_TP_DIMS = (12288, 6144, 3072, 1536)


def _make_inputs(seq_lens, has_init, cache_idx, seed, dim=_DIM, strided=False, pad_cols=None):
    """`strided=True` builds the real production layout: a (T, 4*dim) fused
    q/k/v/g projection output, `torch.split(..., [3*dim, dim], dim=-1)`, and the
    FIRST view as mixed_qkv -- shape (T, 3*dim), stride (4*dim, 1), NOT
    contiguous. This is what `kimi_k3.py` hands to `KDAAttnBackend`.

    `pad_cols` overrides the trailing-column count (the g projection is `dim`
    wide, but the non-full-rank-gate variant of `kimi_k3.py` splits off
    `num_heads + 2 * head_dim` instead, so the pitch is not always 4 * dim)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    total_tokens = int(sum(seq_lens))
    total_dim = 3 * dim
    if strided or pad_cols:
        pad = dim if pad_cols is None else int(pad_cols)
        fused_states = (
            torch.randn(total_tokens, total_dim + pad, generator=gen, dtype=torch.float32)
            .cuda()
            .bfloat16()
        )
        mixed_qkv, _g_proj_states = torch.split(fused_states, [total_dim, pad], dim=-1)
        # The whole point of this variant: if torch ever starts materializing
        # this split, the test must fail rather than quietly re-test the
        # contiguous case.
        assert mixed_qkv.is_contiguous() is False, "expected a non-contiguous view"
        assert tuple(mixed_qkv.shape) == (total_tokens, total_dim)
        assert mixed_qkv.stride() == (total_dim + pad, 1)
    else:
        mixed_qkv = (
            torch.randn(total_tokens, total_dim, generator=gen, dtype=torch.float32)
            .cuda()
            .bfloat16()
        )
    conv_weights = (
        torch.randn(total_dim, _WIDTH, generator=gen, dtype=torch.float32) * 0.1
    ).cuda()
    pool = (
        torch.randn(_NUM_LINES, _WIDTH - 1, total_dim, generator=gen, dtype=torch.float32)
        .cuda()
        .bfloat16()
    )
    query_start_loc = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device="cuda")
    query_start_loc[1:] = torch.tensor(
        seq_lens, dtype=torch.int32, device="cuda"
    ).cumsum(0)
    cache_indices = torch.tensor(
        [PAD_SLOT_ID if c is None else c for c in cache_idx],
        dtype=torch.int32,
        device="cuda",
    )
    has_initial_state = torch.tensor(has_init, dtype=torch.bool, device="cuda")
    return (
        mixed_qkv,
        conv_weights,
        pool,
        query_start_loc,
        cache_indices,
        has_initial_state,
        list(seq_lens),
    )


def _triton_reference(args, dim=_DIM):
    """The three serial causal_conv1d_fn calls this kernel replaces."""
    mixed_qkv, conv_weights, pool, qsl, ci, his, seq_lens = args
    splits = [dim] * 3
    conv_states = pool.transpose(-1, -2)
    outs = [
        causal_conv1d_fn(
            x,
            w,
            None,
            activation="silu",
            conv_states=st,
            has_initial_state=his,
            cache_indices=ci,
            query_start_loc=qsl,
            seq_lens_cpu=seq_lens,
            pad_slot_id=PAD_SLOT_ID,
        )
        for x, w, st in zip(
            mixed_qkv.transpose(0, 1).split(splits, dim=0),
            conv_weights.split(splits, dim=0),
            conv_states.split(splits, dim=-2),
        )
    ]
    return outs, pool


def _non_pad_columns(seq_lens, cache_idx):
    """Pad-slot sequences are skipped by both paths, so their output columns
    are uninitialised in both and must be excluded."""
    keep, offset = [], 0
    for length, c in zip(seq_lens, cache_idx):
        if c is not None and c != PAD_SLOT_ID:
            keep.append(torch.arange(offset, offset + length, device="cuda"))
        offset += length
    return torch.cat(keep) if keep else None


class TestKdaShortConv(CustomTestCase):
    CASES = {
        # name: (seq_lens, has_initial_state, cache_indices, seed)
        "single_tile_multiple": ([4000], [False], [5], 4101),
        "single_with_state": ([4000], [True], [5], 4102),
        "varlen_mixed": ([2048, 1024, 777, 256], [True, False, True, False], [7, 3, 1, 0], 4103),
        # only case whose lengths are not multiples of the NT*TG=80 token tile
        "short_odd": ([1, 7, 33], [False, True, True], [2, 6, 4], 4104),
        # sequences shorter than width-1: part of the old state must survive
        "shorter_than_window": ([1, 2, 3, 4], [True, True, False, True], [1, 4, 2, 6], 4105),
        "padded_slots": ([1024, 512, 256, 128], [True, False, True, False], [6, None, 3, None], 4106),
    }

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required")
        if torch.cuda.get_device_capability()[0] < 10:
            raise unittest.SkipTest("kimi_k3_kda_short_conv requires SM100+")

    def _run_case(self, name, strided=False):
        seq_lens, has_init, cache_idx, seed = self.CASES[name]
        ref_args = _make_inputs(seq_lens, has_init, cache_idx, seed, strided=strided)
        got_args = _make_inputs(seq_lens, has_init, cache_idx, seed, strided=strided)
        # Both paths get the SAME (possibly strided) view; nothing is made
        # contiguous on either side.
        self.assertEqual(got_args[0].is_contiguous(), not strided)
        self.assertTrue(
            covered(got_args[0], got_args[1], got_args[2], None, [_DIM] * 3),
            f"{name}: covered() rejected a supported configuration "
            f"({reject_reason(got_args[0], got_args[1], got_args[2], None, [_DIM] * 3)})",
        )

        exp_outs, exp_pool = _triton_reference(ref_args)
        got_outs = kimi_k3_kda_short_conv(*got_args, pad_slot_id=PAD_SLOT_ID)

        name = f"{name}[{'strided' if strided else 'contiguous'}]"
        cols = _non_pad_columns(seq_lens, cache_idx)
        for label, got, exp in zip(("out_q", "out_k", "out_v"), got_outs, exp_outs):
            if cols is not None:
                got, exp = got.index_select(1, cols), exp.index_select(1, cols)
            self.assertEqual(got.shape, exp.shape, f"{name}/{label}: shape")
            self.assertEqual(got.dtype, exp.dtype, f"{name}/{label}: dtype")
            self.assertTrue(
                torch.equal(got, exp),
                f"{name}/{label}: not bit-exact vs Triton, "
                f"{int((got != exp).sum())}/{got.numel()} elements differ",
            )
        # the in-place conv-state pool update is part of the output
        self.assertTrue(
            torch.equal(got_args[2], exp_pool),
            f"{name}/conv_states_pool: not bit-exact vs Triton, "
            f"{int((got_args[2] != exp_pool).sum())}/{exp_pool.numel()} elements differ",
        )

    def test_single_tile_multiple(self):
        self._run_case("single_tile_multiple")

    def test_single_with_state(self):
        self._run_case("single_with_state")

    def test_varlen_mixed(self):
        self._run_case("varlen_mixed")

    def test_short_odd(self):
        self._run_case("short_odd")

    def test_shorter_than_window(self):
        self._run_case("shorter_than_window")

    def test_padded_slots(self):
        self._run_case("padded_slots")

    # --- same six cases on the real production (row-strided) layout ----------
    def test_single_tile_multiple_strided(self):
        self._run_case("single_tile_multiple", strided=True)

    def test_single_with_state_strided(self):
        self._run_case("single_with_state", strided=True)

    def test_varlen_mixed_strided(self):
        self._run_case("varlen_mixed", strided=True)

    def test_short_odd_strided(self):
        self._run_case("short_odd", strided=True)

    def test_shorter_than_window_strided(self):
        self._run_case("shorter_than_window", strided=True)

    def test_padded_slots_strided(self):
        self._run_case("padded_slots", strided=True)

    def test_output_layout(self):
        """The call site does `.transpose(0, 1)` on the results and expects a
        row-contiguous (tokens, dim) view, i.e. compact (1, dim) strides."""
        args = _make_inputs([256], [False], [1], 4107)
        outs = kimi_k3_kda_short_conv(*args, pad_slot_id=PAD_SLOT_ID)
        for out in outs:
            self.assertEqual(tuple(out.shape), (_DIM, 256))
            self.assertEqual(tuple(out.stride()), (1, _DIM))
            self.assertTrue(out.transpose(0, 1).is_contiguous())

    def test_output_layout_strided_input(self):
        """A row-strided input must still yield compact (1, dim) outputs: the
        call site transposes them and expects a row-contiguous view."""
        args = _make_inputs([256], [False], [1], 4107, strided=True)
        self.assertFalse(args[0].is_contiguous())
        outs = kimi_k3_kda_short_conv(*args, pad_slot_id=PAD_SLOT_ID)
        for out in outs:
            self.assertEqual(tuple(out.shape), (_DIM, 256))
            self.assertEqual(tuple(out.stride()), (1, _DIM))
            self.assertTrue(out.transpose(0, 1).is_contiguous())

    def test_arbitrary_row_pitch_bit_exact(self):
        """The contract is `stride(1) == 1 and stride(0) >= 3 * dim` (aligned to
        the CH-channel vector load), not `stride(0) in (3*dim, 4*dim)`: the
        non-full-rank-gate variant of `kimi_k3.py` splits off
        `num_heads + 2 * head_dim` columns instead of `dim`."""
        pad = 96 + 2 * 128  # num_heads + 2 * head_dim at attn_tp 1
        seq_lens, has_init, cache_idx = [333, 80, 7], [True, False, True], [5, 2, 7]
        ref_args = _make_inputs(seq_lens, has_init, cache_idx, 4300, pad_cols=pad)
        got_args = _make_inputs(seq_lens, has_init, cache_idx, 4300, pad_cols=pad)
        self.assertEqual(got_args[0].stride(), (_TOTAL_DIM + pad, 1))
        self.assertFalse(got_args[0].is_contiguous())
        self.assertIsNone(
            reject_reason(got_args[0], got_args[1], got_args[2], None, [_DIM] * 3)
        )
        exp_outs, exp_pool = _triton_reference(ref_args)
        got_outs = kimi_k3_kda_short_conv(*got_args, pad_slot_id=PAD_SLOT_ID)
        for label, got, exp in zip(("out_q", "out_k", "out_v"), got_outs, exp_outs):
            self.assertTrue(
                torch.equal(got, exp),
                f"pitch={_TOTAL_DIM + pad}/{label}: not bit-exact, "
                f"{int((got != exp).sum())}/{got.numel()} elements differ",
            )
        self.assertTrue(torch.equal(got_args[2], exp_pool))

    def test_attn_tp_dims_bit_exact(self):
        """Bit-exact at every attn_tp_size the model can run.

        `dim` is a runtime kernel argument, but it sets the grid's z extent
        (dim / CHANNEL_TILE), so each value is a different launch geometry --
        and a geometry change alone has been observed to break bit-exactness
        while leaving the conv-state pool correct. Hence a real per-dim check
        rather than trusting that `covered()` accepts them."""
        seq_lens, has_init, cache_idx = [1024, 333, 80, 7], [True, False, True, True], [5, 2, 7, 1]
        for dim in _ATTN_TP_DIMS:
            # Both layouts at every dim: production is row-strided at every
            # attn_tp size (do_fuse_qkvbfg holds at tp=1 and tp=8 alike), and
            # the row pitch changes with dim (4*dim), so the strided variant is
            # a different addressing case per dim, not a redundant one.
            for strided in (False, True):
                with self.subTest(dim=dim, attn_tp=12288 // dim, strided=strided):
                    self.assertEqual(dim % CHANNEL_TILE, 0)
                    ref_args = _make_inputs(
                        seq_lens, has_init, cache_idx, 4200, dim=dim, strided=strided
                    )
                    got_args = _make_inputs(
                        seq_lens, has_init, cache_idx, 4200, dim=dim, strided=strided
                    )
                    self.assertEqual(got_args[0].is_contiguous(), not strided)
                    self.assertTrue(
                        covered(got_args[0], got_args[1], got_args[2], None, [dim] * 3),
                        f"dim={dim} strided={strided}: covered() rejected a supported "
                        "attn_tp size: "
                        f"{reject_reason(got_args[0], got_args[1], got_args[2], None, [dim] * 3)}",
                    )
                    exp_outs, exp_pool = _triton_reference(ref_args, dim=dim)
                    got_outs = kimi_k3_kda_short_conv(*got_args, pad_slot_id=PAD_SLOT_ID)
                    for label, got, exp in zip(("out_q", "out_k", "out_v"), got_outs, exp_outs):
                        self.assertEqual(tuple(got.shape), (dim, int(sum(seq_lens))))
                        self.assertEqual(tuple(got.stride()), (1, dim))
                        self.assertTrue(
                            torch.equal(got, exp),
                            f"dim={dim} strided={strided}/{label}: not bit-exact, "
                            f"{int((got != exp).sum())}/{got.numel()} elements differ",
                        )
                    self.assertTrue(
                        torch.equal(got_args[2], exp_pool),
                        f"dim={dim} strided={strided}/conv_states_pool: not bit-exact, "
                        f"{int((got_args[2] != exp_pool).sum())} elements differ",
                    )

    def test_covered_rejects_unsupported(self):
        """Unsupported cases must be rejected so the caller keeps the Triton
        branch instead of hitting an assert inside the kernel."""
        mixed_qkv, conv_weights, pool, *_ = _make_inputs([64], [False], [1], 4108)
        splits = [_DIM] * 3
        self.assertTrue(covered(mixed_qkv, conv_weights, pool, None, splits))
        # a bias is not supported
        bias = torch.zeros(_TOTAL_DIM, dtype=torch.float32, device="cuda")
        self.assertFalse(covered(mixed_qkv, conv_weights, pool, bias, splits))
        # fp16 input is not supported
        self.assertFalse(covered(mixed_qkv.half(), conv_weights, pool, None, splits))
        # bf16 weights are not supported
        self.assertFalse(
            covered(mixed_qkv, conv_weights.bfloat16(), pool, None, splits)
        )
        # a different per-projection dim is not supported
        self.assertFalse(
            covered(mixed_qkv, conv_weights, pool, None, [_DIM, _DIM, _DIM // 2])
        )
        # a dim the channel tile does not divide is not supported, even though
        # the three splits are equal and mixed_qkv would match
        odd_dim = CHANNEL_TILE * 3 + 1
        odd = _make_inputs([64], [False], [1], 4109, dim=odd_dim)
        self.assertFalse(covered(odd[0], odd[1], odd[2], None, [odd_dim] * 3))
        # the PRODUCTION row-strided view IS supported (this is the whole point
        # of the row-stride contract; requiring contiguity made the kernel
        # unreachable in production)
        strided_qkv = _make_inputs([64], [False], [1], 4110, strided=True)[0]
        self.assertFalse(strided_qkv.is_contiguous())
        self.assertIsNone(reject_reason(strided_qkv, conv_weights, pool, None, splits))
        self.assertTrue(covered(strided_qkv, conv_weights, pool, None, splits))
        # a genuinely bad layout is not: channel axis must be unit-stride
        channel_strided = (
            torch.zeros(_TOTAL_DIM, 64, dtype=torch.bfloat16, device="cuda")
            .transpose(0, 1)
        )
        self.assertEqual(tuple(channel_strided.shape), (64, _TOTAL_DIM))
        self.assertNotEqual(channel_strided.stride(1), 1)
        self.assertFalse(covered(channel_strided, conv_weights, pool, None, splits))
        self.assertIn(
            "stride(1) == 1",
            reject_reason(channel_strided, conv_weights, pool, None, splits),
        )
        # a mixed_qkv whose rows overlap (pitch < 3 * dim) is not supported
        overlapping = torch.zeros(
            64 * _TOTAL_DIM, dtype=torch.bfloat16, device="cuda"
        ).as_strided((64, _TOTAL_DIM), (_DIM, 1))
        self.assertFalse(covered(overlapping, conv_weights, pool, None, splits))
        self.assertIn(
            "stride(0) >= 3 * dim",
            reject_reason(overlapping, conv_weights, pool, None, splits),
        )
        # a pitch that breaks the CH-channel vector load alignment is rejected
        misaligned = torch.zeros(
            64 * (_TOTAL_DIM + 2) + 8, dtype=torch.bfloat16, device="cuda"
        ).as_strided((64, _TOTAL_DIM), (_TOTAL_DIM + 2, 1))
        self.assertEqual(misaligned.stride(0) % 4, 2)
        self.assertFalse(covered(misaligned, conv_weights, pool, None, splits))
        self.assertIn(
            "stride(0) % 4 == 0",
            reject_reason(misaligned, conv_weights, pool, None, splits),
        )
        # a transposed mixed_qkv is still rejected
        self.assertFalse(
            covered(mixed_qkv.transpose(0, 1), conv_weights, pool, None, splits)
        )

    def test_warns_once_when_flag_on_but_rejected(self):
        """A silent fallback is how the row-stride mismatch stayed invisible for
        a whole integration cycle. `warn_if_rejected=True` (passed by the call
        site only when SGLANG_K3_KDA_SHORT_CONV is on) must log the failing
        predicate exactly once; the default must stay silent."""
        mixed_qkv, conv_weights, pool, *_ = _make_inputs([64], [False], [1], 4111)
        splits = [_DIM] * 3
        bad = mixed_qkv.half()  # fp16 x: rejected

        # flag off (default: warn_if_rejected=False) -> no log at all
        kda_short_conv_mod._warned_rejected = False
        with self.assertNoLogs(kda_short_conv_mod.__name__, level="WARNING"):
            self.assertFalse(covered(bad, conv_weights, pool, None, splits))
        self.assertFalse(kda_short_conv_mod._warned_rejected)

        # flag on -> exactly one warning naming the predicate, then silence
        with self.assertLogs(kda_short_conv_mod.__name__, level="WARNING") as cm:
            self.assertFalse(
                covered(bad, conv_weights, pool, None, splits, warn_if_rejected=True)
            )
        self.assertEqual(len(cm.output), 1, cm.output)
        self.assertIn("SGLANG_K3_KDA_SHORT_CONV", cm.output[0])
        self.assertIn("mixed_qkv.dtype == bfloat16", cm.output[0])
        with self.assertNoLogs(kda_short_conv_mod.__name__, level="WARNING"):
            for _ in range(3):
                self.assertFalse(
                    covered(bad, conv_weights, pool, None, splits, warn_if_rejected=True)
                )

        # a covered input never warns, even with the flag on
        kda_short_conv_mod._warned_rejected = False
        with self.assertNoLogs(kda_short_conv_mod.__name__, level="WARNING"):
            self.assertTrue(
                covered(
                    mixed_qkv, conv_weights, pool, None, splits, warn_if_rejected=True
                )
            )
        kda_short_conv_mod._warned_rejected = False


if __name__ == "__main__":
    unittest.main()
