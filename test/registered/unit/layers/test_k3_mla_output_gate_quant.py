from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

import unittest

import torch

from sglang.kernels.ops.kimi_k3 import mla_output_gate
from sglang.kernels.ops.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
)


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestK3MlaOutputGateQuant(unittest.TestCase):
    def test_bit_identical_to_gate_then_quant(self):
        torch.manual_seed(0)
        # K3 MLA o_proj input widths and decode token counts (odd tails too).
        for tokens in (1, 6, 37, 192, 1000):
            for hidden in (512, 1536, 2048, 4096):
                with self.subTest(tokens=tokens, hidden=hidden):
                    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) * 4
                    gate = torch.randn_like(x) * 3
                    self.assertTrue(mla_output_gate.covered(x, gate))
                    self.assertTrue(mla_output_gate.quant_covered(x))
                    out_ref = mla_output_gate.kimi_k3_mla_output_gate(x, gate)
                    q_ref, s_ref = sglang_per_token_group_quant_fp8(
                        out_ref,
                        128,
                        column_major_scales=True,
                        scale_tma_aligned=True,
                        scale_ue8m0=True,
                    )
                    out, q, s = mla_output_gate.kimi_k3_mla_output_gate_quant(x, gate)
                    self.assertTrue(torch.equal(out, out_ref))
                    self.assertEqual(s.shape, s_ref.shape)
                    self.assertEqual(s.stride(), s_ref.stride())
                    self.assertTrue(torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)))
                    self.assertTrue(torch.equal(s, s_ref))

    def test_quant_covered_rejects_unaligned_width(self):
        self.assertFalse(
            mla_output_gate.quant_covered(
                torch.empty(4, 384, device="cuda", dtype=torch.bfloat16)
            )
        )


if __name__ == "__main__":
    unittest.main()
