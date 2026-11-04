"""Tests for the fused DCP a2a send-side pack (SGLANG_OPT_USE_FUSED_DCP_A2A_PACK).

Covers:
1. The fused path produces the same wire bytes and a numerically equivalent
   combined output as the original eager path, for several (N, B, H, D) shapes.
2. The env gate: off -> original branch, on -> fused branch (asserted by
   patching the fused entry point, so branch selection is observable).
3. The host-side guard rejects unsupported inputs, and such inputs fall back to
   the original path even with the flag on.
4. The cuda_graph_buffers path is untouched by the flag.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")


def _loopback_group(world_size):
    """A group whose all_to_all_single is a local byte copy.

    With one process there is no DCP group, so the exchange is emulated: rank i
    receives what it sent. That is enough to exercise pack -> exchange -> unpack
    -> combine end to end, which is where the fused path differs.
    """
    group = MagicMock()
    group.world_size = world_size

    def all_to_all_single(output, input_):
        output.copy_(input_)

    group.all_to_all_single = all_to_all_single
    return group


class TestFusedDcpA2aPack(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required for Triton kernel tests")
        cls.device = "cuda"

    def _inputs(self, N, B, H, D, dtype=torch.bfloat16, seed=0):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        out = torch.randn(
            B, H, D, device=self.device, dtype=torch.float32, generator=generator
        ).to(dtype)
        lse = (
            torch.randn(
                B, H, device=self.device, dtype=torch.float32, generator=generator
            )
            * 5.0
        )
        return out, lse

    def _reduce(self, out, lse, group, fused):
        from sglang.srt.environ import envs
        from sglang.srt.layers.dcp.comm import dcp_a2a_lse_reduce

        with envs.SGLANG_OPT_USE_FUSED_DCP_A2A_PACK.override(fused):
            return dcp_a2a_lse_reduce(
                out.clone(),
                lse.clone(),
                group,
                is_lse_base_on_e=False,
            )

    def test_fused_matches_original(self):
        for N, B, H, D in ((8, 200, 96, 512), (4, 17, 32, 128), (2, 1, 8, 64)):
            with self.subTest(N=N, B=B, H=H, D=D):
                out, lse = self._inputs(N, B, H, D)
                group = _loopback_group(N)
                want = self._reduce(out, lse, group, fused=False)
                got = self._reduce(out, lse, group, fused=True)
                self.assertEqual(got.shape, want.shape)
                self.assertEqual(got.dtype, want.dtype)
                # Same bytes on the wire, but the combine kernel is re-specialised
                # on the new strides, so near-cancellation elements may differ by
                # more than one ULP. Compare against the partial magnitudes.
                torch.testing.assert_close(
                    got.float(), want.float(), rtol=2e-2, atol=2e-2
                )

    def test_wire_bytes_are_a_permutation_of_the_original(self):
        """The fused pack must move exactly the original payload and LSE values."""
        from sglang.kernels.ops.attention.dcp_fused_pack import (
            dcp_fused_pack,
            dcp_fused_pack_recv_views,
        )

        N, B, H, D = 4, 9, 32, 128
        H_per_rank = H // N
        out, lse = self._inputs(N, B, H, D, seed=7)
        staged = dcp_fused_pack(out, lse, N)
        recv_output, recv_lse = dcp_fused_pack_recv_views(
            staged.reshape(-1), N, B, H_per_rank, D
        )
        expected_out = out.view(B, N, H_per_rank, D).permute(1, 0, 2, 3)
        expected_lse = lse.view(B, N, H_per_rank).permute(1, 0, 2)
        self.assertTrue(torch.equal(recv_output, expected_out))
        self.assertTrue(torch.equal(recv_lse, expected_lse))

    def test_gate_selects_the_branch(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.dcp import comm as dcp_comm

        N, B, H, D = 4, 9, 32, 128
        out, lse = self._inputs(N, B, H, D, seed=11)
        group = _loopback_group(N)

        for fused in (False, True):
            with self.subTest(fused=fused):
                with patch.object(
                    dcp_comm,
                    "_dcp_a2a_lse_reduce_fused_pack",
                    wraps=dcp_comm._dcp_a2a_lse_reduce_fused_pack,
                ) as spy:
                    with envs.SGLANG_OPT_USE_FUSED_DCP_A2A_PACK.override(fused):
                        dcp_comm.dcp_a2a_lse_reduce(
                            out.clone(),
                            lse.clone(),
                            group,
                            is_lse_base_on_e=False,
                        )
                self.assertEqual(spy.called, fused)


    def test_guard_rejects_unsupported_inputs(self):
        from sglang.kernels.ops.attention.dcp_fused_pack import (
            dcp_fused_pack_supported,
        )

        N, B, H, D = 4, 8, 32, 128
        out, lse = self._inputs(N, B, H, D, seed=3)
        self.assertTrue(dcp_fused_pack_supported(out, lse, N))
        self.assertFalse(dcp_fused_pack_supported(out.float(), lse, N))
        self.assertFalse(dcp_fused_pack_supported(out, lse.half(), N))
        self.assertFalse(
            dcp_fused_pack_supported(out.transpose(0, 1), lse.transpose(0, 1), N)
        )
        self.assertFalse(dcp_fused_pack_supported(out, lse, 7))
        kv = torch.ones(2, dtype=torch.int32, device=self.device)
        cum = torch.tensor([0, 4, 8], dtype=torch.int32, device=self.device)
        self.assertTrue(dcp_fused_pack_supported(out, lse, N, kv, cum))
        self.assertFalse(dcp_fused_pack_supported(out, lse, N, kv.long(), cum))
        self.assertFalse(dcp_fused_pack_supported(out, lse, N, kv, cum[:-1]))
        # a partially specified fixup must be rejected, not silently ignored
        self.assertFalse(dcp_fused_pack_supported(out, lse, N, kv, None))

    def test_unsupported_input_falls_back_with_flag_on(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.dcp.comm import dcp_a2a_lse_reduce

        N, B, H, D = 4, 8, 32, 128
        out, lse = self._inputs(N, B, H, D, seed=5)
        group = _loopback_group(N)
        # fp32 payload is outside the declared support boundary
        out32 = out.float()
        with patch(
            "sglang.srt.layers.dcp.comm._dcp_a2a_lse_reduce_fused_pack"
        ) as never:
            with envs.SGLANG_OPT_USE_FUSED_DCP_A2A_PACK.override(True):
                result = dcp_a2a_lse_reduce(
                    out32, lse.clone(), group, is_lse_base_on_e=False
                )
        never.assert_not_called()
        self.assertEqual(result.dtype, torch.float32)

    def test_cuda_graph_buffer_path_ignores_the_flag(self):
        from sglang.kernels.ops.attention.dcp_kernels import _lse_pack_dim
        from sglang.srt.environ import envs
        from sglang.srt.layers.dcp.comm import dcp_a2a_lse_reduce

        N, B, H, D = 4, 8, 32, 128
        H_per_rank = H // N
        lpd = _lse_pack_dim(torch.bfloat16)
        buffers = {
            "send_combined": torch.zeros(
                N, B, H_per_rank, D + lpd, device=self.device, dtype=torch.bfloat16
            ),
            "recv_combined": torch.zeros(
                N, B, H_per_rank, D + lpd, device=self.device, dtype=torch.bfloat16
            ),
            "send_lse": torch.zeros(
                N, B, H_per_rank, device=self.device, dtype=torch.float32
            ),
            "recv_lse": torch.zeros(
                N, B, H_per_rank, device=self.device, dtype=torch.float32
            ),
        }
        out, lse = self._inputs(N, B, H, D, seed=13)
        group = _loopback_group(N)
        with patch(
            "sglang.srt.layers.dcp.comm._dcp_a2a_lse_reduce_fused_pack"
        ) as never:
            with envs.SGLANG_OPT_USE_FUSED_DCP_A2A_PACK.override(True):
                dcp_a2a_lse_reduce(
                    out,
                    lse,
                    group,
                    is_lse_base_on_e=False,
                    cuda_graph_buffers=buffers,
                )
        never.assert_not_called()


if __name__ == "__main__":
    unittest.main()
