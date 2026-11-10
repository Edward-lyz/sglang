from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

import unittest

import torch

from sglang.kernels.ops.kimi_k3.activation import (
    situ_and_mul,
    situ_and_mul_quant,
    situ_and_mul_quant_supported,
)
from sglang.kernels.ops.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
)


def _is_sm100_plus():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


def _unfused(x, beta, linear_beta):
    act = situ_and_mul(x, None, beta, linear_beta)
    return sglang_per_token_group_quant_fp8(
        act,
        128,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )


@unittest.skipUnless(_is_sm100_plus(), "needs SM100+")
class TestK3SituAndMulQuant(unittest.TestCase):
    def _check(self, x, beta, linear_beta):
        self.assertTrue(situ_and_mul_quant_supported(x))
        q_ref, s_ref = _unfused(x, beta, linear_beta)
        q, s = situ_and_mul_quant(x, beta, linear_beta)
        self.assertEqual(q.shape, q_ref.shape)
        self.assertEqual(s.shape, s_ref.shape)
        self.assertEqual(s.stride(), s_ref.stride())
        self.assertTrue(torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)))
        self.assertTrue(torch.equal(s, s_ref))

    def test_bit_identical_to_situ_then_quant(self):
        torch.manual_seed(0)
        # K3 decode token counts (BS=32, verify width 6) and odd tails.
        for tokens in (1, 6, 37, 192, 384, 1000):
            for hidden in (512, 3072, 6144):
                for beta, linear_beta in ((4.0, 25.0), (4.0, None)):
                    x = torch.randn(
                        tokens, 2 * hidden, device="cuda", dtype=torch.bfloat16
                    ) * 3
                    with self.subTest(
                        tokens=tokens, hidden=hidden, linear_beta=linear_beta
                    ):
                        self._check(x, beta, linear_beta)

    def test_row_strided_input(self):
        torch.manual_seed(1)
        hidden = 3072
        wide = torch.randn(192, 2 * hidden + 256, device="cuda", dtype=torch.bfloat16)
        self._check(wide[:, : 2 * hidden], 4.0, 25.0)

    def test_extreme_values(self):
        hidden = 512
        x = torch.randn(8, 2 * hidden, device="cuda", dtype=torch.bfloat16) * 100
        x[0].zero_()
        x[1, :hidden] = 1e-30
        self._check(x, 4.0, 25.0)

    def test_unsupported_shapes_rejected(self):
        x = torch.randn(4, 2 * 384, device="cuda", dtype=torch.bfloat16)
        self.assertFalse(situ_and_mul_quant_supported(x))
        self.assertFalse(situ_and_mul_quant_supported(x.half()))


if __name__ == "__main__":
    unittest.main()
