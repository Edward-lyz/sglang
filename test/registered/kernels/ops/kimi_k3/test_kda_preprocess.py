import unittest

import torch

from sglang.kernels.ops.kimi_k3.kda_preprocess import (
    can_fuse_beta_sigmoid,
    can_fuse_track_conv_state,
    can_fuse_track_conv_state_flag,
    fused_beta_sigmoid,
    fused_track_conv_state,
    fused_track_conv_state_flag,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=30,
    stage="base-b-kernel-unit",
    runner_config="1-gpu-large",
)


class TestKdaPreprocess(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")

    def test_beta_sigmoid_matches_strided_reference(self):
        torch.manual_seed(0)
        bfa = torch.randn(
            257,
            224,
            dtype=torch.bfloat16,
            device="cuda",
        )
        beta = bfa[:, 128:224]
        output = torch.empty(beta.shape, dtype=torch.float32, device="cuda")

        self.assertFalse(beta.is_contiguous())
        self.assertTrue(can_fuse_beta_sigmoid(beta))
        fused_beta_sigmoid(beta, output)

        torch.testing.assert_close(
            output,
            beta.float().sigmoid(),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_track_conv_state_matches_advanced_indexing(self):
        torch.manual_seed(0)
        num_tokens = 128
        num_slots = 32
        num_tracks = 7
        window_size = 3
        state_dim = 4608

        mixed_qkv = torch.randn(
            num_tokens,
            state_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        conv_storage = torch.randn(
            num_slots,
            window_size,
            state_dim + 8,
            dtype=torch.bfloat16,
            device="cuda",
        )
        conv_state = conv_storage[..., :state_dim]
        destination_indices = torch.tensor(
            [1, 4, 7, 11, 16, 23, 29],
            dtype=torch.int64,
            device="cuda",
        )

        for track_dtype in (torch.int32, torch.int64):
            with self.subTest(track_dtype=track_dtype):
                track_indices = torch.randint(
                    0,
                    num_tokens,
                    (num_tracks, window_size),
                    dtype=track_dtype,
                    device="cuda",
                )
                test_storage = conv_storage.clone()
                test_conv_state = test_storage[..., :state_dim]
                reference = conv_state.clone()
                reference[destination_indices] = mixed_qkv[track_indices.long()]

                self.assertTrue(
                    can_fuse_track_conv_state(
                        mixed_qkv,
                        test_conv_state,
                        track_indices,
                        destination_indices,
                    )
                )
                fused_track_conv_state(
                    mixed_qkv,
                    test_conv_state,
                    track_indices,
                    destination_indices,
                )
                torch.cuda.synchronize()

                self.assertTrue(torch.equal(test_conv_state, reference))

    def test_empty_track_is_a_noop(self):
        mixed_qkv = torch.empty(
            1,
            8,
            dtype=torch.bfloat16,
            device="cuda",
        )
        conv_state = torch.zeros(
            2,
            3,
            8,
            dtype=torch.bfloat16,
            device="cuda",
        )
        track_indices = torch.empty(
            0,
            3,
            dtype=torch.int64,
            device="cuda",
        )
        destination_indices = torch.empty(
            0,
            dtype=torch.int64,
            device="cuda",
        )
        reference = conv_state.clone()

        self.assertTrue(
            can_fuse_track_conv_state(
                mixed_qkv,
                conv_state,
                track_indices,
                destination_indices,
            )
        )
        fused_track_conv_state(
            mixed_qkv,
            conv_state,
            track_indices,
            destination_indices,
        )
        self.assertTrue(torch.equal(conv_state, reference))


    def test_track_conv_state_flag_matches_indexing_and_compare(self):
        torch.manual_seed(0)
        num_tokens = 128
        num_slots = 32
        num_tracks = 5
        window_size = 3
        state_dim = 4608
        num_seqs = 7

        mixed_qkv = torch.randn(
            num_tokens, state_dim, dtype=torch.bfloat16, device="cuda"
        )
        conv_state = torch.randn(
            num_slots, window_size, state_dim, dtype=torch.bfloat16, device="cuda"
        )
        destination_indices = torch.tensor(
            [1, 4, 7, 11, 16], dtype=torch.int64, device="cuda"
        )
        track_indices = torch.randint(
            0, num_tokens, (num_tracks, window_size), dtype=torch.int32, device="cuda"
        )
        prefix_lens = torch.tensor(
            [0, 3, 0, 17, 0, 0, 9], dtype=torch.int32, device="cuda"
        )

        reference_state = conv_state.clone()
        reference_state[destination_indices] = mixed_qkv[track_indices.long()]
        reference_flag = prefix_lens > 0

        state = conv_state.clone()
        flag = torch.empty(num_seqs, dtype=torch.bool, device="cuda")
        self.assertTrue(
            can_fuse_track_conv_state_flag(
                mixed_qkv,
                state,
                track_indices,
                destination_indices,
                prefix_lens,
                flag,
            )
        )
        fused_track_conv_state_flag(
            mixed_qkv, state, track_indices, destination_indices, prefix_lens, flag
        )
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(state, reference_state))
        self.assertTrue(torch.equal(flag, reference_flag))

    def test_track_conv_state_flag_without_tracks_still_writes_flag(self):
        mixed_qkv = torch.empty(1, 8, dtype=torch.bfloat16, device="cuda")
        conv_state = torch.zeros(2, 3, 8, dtype=torch.bfloat16, device="cuda")
        track_indices = torch.empty(0, 3, dtype=torch.int64, device="cuda")
        destination_indices = torch.empty(0, dtype=torch.int64, device="cuda")
        prefix_lens = torch.tensor([0, 5, 2], dtype=torch.int32, device="cuda")
        flag = torch.empty(3, dtype=torch.bool, device="cuda")
        reference_state = conv_state.clone()

        fused_track_conv_state_flag(
            mixed_qkv, conv_state, track_indices, destination_indices, prefix_lens, flag
        )
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(conv_state, reference_state))
        self.assertTrue(torch.equal(flag, prefix_lens > 0))


if __name__ == "__main__":
    unittest.main()
