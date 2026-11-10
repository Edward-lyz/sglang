import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.linear.kda_backend import (
    kda_extend_initial_state_and_track,
)
from sglang.srt.models.kimi_k3 import kda_activate_beta
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=30,
    stage="base-b-kernel-unit",
    runner_config="1-gpu-large",
)


class TestKdaPreprocessGating(CustomTestCase):
    """Flag-off / flag-on / parity for the two KDA extend-preprocess replacements."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")

    @staticmethod
    def _beta_case(tokens=257, bfa_width=224, heads=96):
        torch.manual_seed(0)
        bfa = torch.randn(tokens, bfa_width, dtype=torch.bfloat16, device="cuda")
        return bfa[:, bfa_width - heads :]

    @staticmethod
    def _track_case(num_tokens=128, pool=32, num_tracks=5, window=3, state_dim=4608, num_seqs=7):
        torch.manual_seed(0)
        return dict(
            mixed_qkv=torch.randn(
                num_tokens, state_dim, dtype=torch.bfloat16, device="cuda"
            ),
            conv_pool=torch.randn(
                pool, window, state_dim, dtype=torch.bfloat16, device="cuda"
            ),
            track_indices=torch.randint(
                0, num_tokens, (num_tracks, window), dtype=torch.int32, device="cuda"
            ),
            destination_indices=torch.tensor(
                [1, 4, 7, 11, 16], dtype=torch.int64, device="cuda"
            ),
            prefix_lens=torch.tensor(
                [0, 3, 0, 17, 0, 0, 9], dtype=torch.int32, device="cuda"
            ),
        )

    def test_beta_parity_flag_off_and_on(self):
        beta = self._beta_case()
        with envs.SGLANG_OPT_USE_K3_KDA_FUSED_BETA_SIGMOID.override(False):
            old = kda_activate_beta(beta)
        with envs.SGLANG_OPT_USE_K3_KDA_FUSED_BETA_SIGMOID.override(True):
            new = kda_activate_beta(beta)
        torch.cuda.synchronize()

        self.assertEqual(old.dtype, torch.float32)
        self.assertEqual(new.dtype, old.dtype)
        self.assertEqual(new.shape, old.shape)
        self.assertTrue(new.is_contiguous())
        self.assertTrue(torch.equal(new, old))

    def test_beta_flag_on_uncovered_layout_falls_back(self):
        # heads % 4 != 0 is outside can_fuse_beta_sigmoid; must not raise.
        beta = torch.randn(8, 6, dtype=torch.bfloat16, device="cuda")
        with envs.SGLANG_OPT_USE_K3_KDA_FUSED_BETA_SIGMOID.override(True):
            got = kda_activate_beta(beta)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(got, beta.float().sigmoid()))

    def test_track_parity_flag_off_and_on(self):
        case = self._track_case()

        with envs.SGLANG_OPT_USE_K3_KDA_FUSED_TRACK_CONV_STATE.override(False):
            old_pool = case["conv_pool"].clone()
            old_flag = kda_extend_initial_state_and_track(
                prefix_lens=case["prefix_lens"],
                mixed_qkv=case["mixed_qkv"],
                conv_pool=old_pool,
                track_indices=case["track_indices"],
                destination_indices=case["destination_indices"],
            )
        with envs.SGLANG_OPT_USE_K3_KDA_FUSED_TRACK_CONV_STATE.override(True):
            new_pool = case["conv_pool"].clone()
            new_flag = kda_extend_initial_state_and_track(
                prefix_lens=case["prefix_lens"],
                mixed_qkv=case["mixed_qkv"],
                conv_pool=new_pool,
                track_indices=case["track_indices"],
                destination_indices=case["destination_indices"],
            )
        torch.cuda.synchronize()

        self.assertEqual(new_flag.dtype, torch.bool)
        self.assertTrue(torch.equal(new_flag, old_flag))
        # in-place conv-pool write must be byte-identical
        self.assertTrue(
            torch.equal(
                new_pool.contiguous().view(torch.uint8),
                old_pool.contiguous().view(torch.uint8),
            )
        )

    def test_track_int64_prefix_lens(self):
        case = self._track_case()
        case["prefix_lens"] = case["prefix_lens"].to(torch.int64)
        with envs.SGLANG_OPT_USE_K3_KDA_FUSED_TRACK_CONV_STATE.override(True):
            flag = kda_extend_initial_state_and_track(
                prefix_lens=case["prefix_lens"],
                mixed_qkv=case["mixed_qkv"],
                conv_pool=case["conv_pool"].clone(),
                track_indices=case["track_indices"],
                destination_indices=case["destination_indices"],
            )
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(flag, case["prefix_lens"] > 0))

    def test_no_tracking_only_writes_flag(self):
        case = self._track_case()
        for flag_value in (False, True):
            with self.subTest(flag=flag_value):
                with envs.SGLANG_OPT_USE_K3_KDA_FUSED_TRACK_CONV_STATE.override(
                    flag_value
                ):
                    got = kda_extend_initial_state_and_track(
                        prefix_lens=case["prefix_lens"],
                        mixed_qkv=case["mixed_qkv"],
                    )
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(got, case["prefix_lens"] > 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
