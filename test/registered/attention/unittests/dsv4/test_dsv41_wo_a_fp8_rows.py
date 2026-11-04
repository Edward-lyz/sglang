"""SGLANG_OPT_FP8_WO_A_GEMM on a 32x32-block ue8m0 checkpoint (per-row fp8 wo_a).

The V4.1 1017 checkpoint stores wo_a as e4m3 with 32x32 ue8m0 block scales. The loader
dequantizes it to bf16; MQALayer.convert_wo_a_fp8_rows re-quantizes that to per-row 1x32
fp8 (lossless) and the non-fused-o path runs per-32 act quant + DeepGEMM fp8_einsum. The
reference model (inference/model.py) does act_quant(o, 32, ue8m0) then a bf16 einsum with
the dequantized wo_a, so the fp8 path must match that up to fp32-accumulation order.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

G, R, D = 8, 1024, 4096  # o_groups, o_lora_rank, heads/group * head_dim


def _ckpt_wo_a():
    """bf16 wo_a as dequantized from a 32x32-block ue8m0 e4m3 checkpoint."""
    w = torch.randn(G * R, D, device="cuda") * 0.02
    blk = w.view(G * R // 32, 32, D // 32, 32)
    amax = blk.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-30)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    q = (blk / scale).to(torch.float8_e4m3fn)
    return (q.float() * scale).view(G * R, D).to(torch.bfloat16)


class _Holder(torch.nn.Module):
    """Stands in for wo_a (only .weight is read)."""

    def __init__(self, w):
        """Wrap the bf16 weight as a parameter."""
        super().__init__()
        self.weight = torch.nn.Parameter(w, requires_grad=False)


class TestWoAFp8Rows(CustomTestCase):
    """convert_wo_a_fp8_rows + _wo_a_fp8_rows_forward against the reference math."""

    def _layer(self, w, fused=False):
        """A bare object carrying the attributes the two MQALayer methods read."""
        from sglang.srt.models.deepseek_v4 import MQALayer

        layer = MQALayer.__new__(MQALayer)
        torch.nn.Module.__init__(layer)
        layer.wo_a = _Holder(w)
        layer.wo_a_fp8_rows = True
        layer.fused_o_attn = fused
        layer.n_local_groups, layer.o_lora_rank = G, R
        layer.n_local_heads, layer.head_dim = 64, 512
        layer._wo_a_rows = None
        layer._wo_a_rows_blk = None
        return layer

    def test_lossless_requant_and_matches_reference(self):
        """Per-row codes dequantize back to the bf16 weight; output == act_quant ref."""
        from sglang.srt.layers.attention.dsv4.fused_o_attn import quantize_rows_ue8m0

        torch.manual_seed(0)
        w = _ckpt_wo_a()
        q, s = quantize_rows_ue8m0(w)
        back = (q.float().view(G * R, D // 32, 32) * s.unsqueeze(-1)).view(G * R, D)
        self.assertTrue(torch.equal(back.to(torch.bfloat16), w))

        layer = self._layer(w.clone())
        self.assertTrue(layer.convert_wo_a_fp8_rows())
        self.assertIsNone(getattr(layer.wo_a, "weight", None))
        for t in (1, 64, 1000, 8192):
            o = torch.randn(t, G, D, device="cuda", dtype=torch.bfloat16)
            got = layer._wo_a_fp8_rows_forward(o).float()
            g = o.float().view(t, G, D // 32, 32)
            amax = g.abs().amax(-1, keepdim=True).clamp(min=1e-4)
            sc = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
            o_ref = ((g / sc).to(torch.float8_e4m3fn).float() * sc).view(t, G, D)
            wf = w.float().view(G, R, D)
            ref = torch.einsum("tgd,grd->tgr", o_ref, wf).flatten(1)
            err = (got - ref).abs().max().item() / ref.abs().max().item()
            self.assertLess(err, 2e-2, f"T={t} rel err {err}")

    def test_fused_layout_copy_matches_plain(self):
        """One fp8 copy in the fused-o layout + block-permuted o == the plain copy."""
        from sglang.srt.layers.attention.dsv4.fused_o_attn import fused_o_attn_available

        if not fused_o_attn_available():
            self.skipTest("flash_mla fused attention-o not available")
        torch.manual_seed(1)
        w = _ckpt_wo_a()
        plain, fused = self._layer(w.clone()), self._layer(w.clone(), fused=True)
        self.assertTrue(plain.convert_wo_a_fp8_rows())
        self.assertTrue(fused.convert_wo_a_fp8_rows())
        self.assertIs(fused._wo_a_rows[0], fused._wo_a_fused_w)
        for t in (1, 64, 4096):
            o = torch.randn(t, G, D, device="cuda", dtype=torch.bfloat16)
            a = plain._wo_a_fp8_rows_forward(o).float()
            b = fused._wo_a_fp8_rows_forward(o).float()
            err = (a - b).abs().max().item() / a.abs().max().item()
            self.assertLess(err, 1e-2, f"T={t} rel err {err}")


if __name__ == "__main__":
    unittest.main()
