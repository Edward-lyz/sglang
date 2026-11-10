from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-small")

import unittest

import torch

from sglang.kernels.ops.kimi_k3.attn_res import (
    attn_res_fused_tma,
    attn_res_fused_tma_quant,
)
from sglang.kernels.ops.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
)
from sglang.srt.layers.quantization import fp8_utils

H, NB, EPS = 7168, 9, 1e-6


def _is_sm100_plus():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


def _inputs(tokens, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    prefix = torch.randn(tokens, H, device="cuda", dtype=torch.bfloat16, generator=g)
    bank = torch.randn(tokens, NB, H, device="cuda", dtype=torch.bfloat16, generator=g)
    cw = torch.randn(H, device="cuda", dtype=torch.bfloat16, generator=g) * 0.05
    ow = torch.rand(H, device="cuda", dtype=torch.bfloat16, generator=g) + 0.5
    return prefix, bank, cw, ow


@unittest.skipUnless(_is_sm100_plus(), "needs SM100+")
class TestK3AttnResFp8Out(unittest.TestCase):
    def test_bit_identical_to_attn_res_then_quant(self):
        # K3 decode token counts (BS=32 x verify width 6 on 8 SP shards, ...).
        for tokens in (1, 6, 24, 37, 192, 300):
            for nvb in range(1, 9):
                for write_prefix in (False, True):
                    with self.subTest(tokens=tokens, nvb=nvb, write_prefix=write_prefix):
                        prefix, bank, cw, ow = _inputs(tokens, tokens * 31 + nvb)
                        bank_ref = bank.clone()
                        out_ref = torch.empty_like(prefix)
                        attn_res_fused_tma(
                            prefix, bank_ref, cw, ow, out_ref, nvb, EPS,
                            write_prefix=write_prefix,
                        )
                        q_ref, s_ref = sglang_per_token_group_quant_fp8(
                            out_ref,
                            128,
                            column_major_scales=True,
                            scale_tma_aligned=True,
                            scale_ue8m0=True,
                        )
                        out = torch.empty_like(prefix)
                        q, s = attn_res_fused_tma_quant(
                            prefix, bank, cw, ow, out, nvb, EPS,
                            write_prefix=write_prefix,
                        )
                        self.assertTrue(torch.equal(out, out_ref))
                        self.assertTrue(torch.equal(bank, bank_ref))
                        self.assertEqual(s.shape, s_ref.shape)
                        self.assertEqual(s.stride(), s_ref.stride())
                        self.assertTrue(
                            torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
                        )
                        self.assertTrue(torch.equal(s, s_ref))

    def test_prequantized_slot_matches_identity_only(self):
        x = torch.randn(4, H, device="cuda", dtype=torch.bfloat16)
        q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        s = torch.empty(4, 14, device="cuda", dtype=torch.int32)
        fp8_utils.set_prequantized_input(x, q, s)
        hit = fp8_utils._lookup_prequantized_input(x)
        self.assertIsNotNone(hit)
        self.assertIs(hit[0], q)
        self.assertIs(hit[1], s)
        # A contiguous reshape keeping the last dim (the MoE's 2D view, the
        # KDA [1, T, H, D] -> [T, H * D] flatten) still matches; a copy, a
        # slice or a different last dim does not.
        self.assertIsNotNone(fp8_utils._lookup_prequantized_input(x.view(-1, H)))
        self.assertIsNotNone(fp8_utils._lookup_prequantized_input(x.view(1, 4, H)))
        self.assertIsNone(fp8_utils._lookup_prequantized_input(x.clone()))
        self.assertIsNone(fp8_utils._lookup_prequantized_input(x[:2]))
        self.assertIsNone(fp8_utils._lookup_prequantized_input(x.view(8, H // 2)))
        y = torch.randn_like(x)
        fp8_utils.set_prequantized_input(y, q, s)
        self.assertIsNone(fp8_utils._lookup_prequantized_input(x))


if __name__ == "__main__":
    unittest.main()
