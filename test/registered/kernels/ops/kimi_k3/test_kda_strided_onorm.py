"""Correctness test for kernels.ops.kimi_k3.kda_strided_onorm (fused strided
gated RMSNorm for the KDA extend/prefill output tail).

Correctness is checked against the real reference implementation
`rms_norm_gated` (sglang.kernels.ops.attention.fla.fused_norm_gate), which is
what `FusedRMSNormGated`/`self.o_norm` calls today. Inputs are built with the
same non-contiguous layout the KDA extend path actually produces: a
[1, H*D, T] contiguous "conv-output-shaped" tensor, transposed on the last two
axes and unflattened, giving x.stride() == (T, 1, T*D, T) for shape
[1, T, H, D].
"""

from __future__ import annotations

import unittest

import torch

from sglang.kernels.ops.attention.fla.fused_norm_gate import rms_norm_gated
from sglang.kernels.ops.kimi_k3.kda_strided_onorm import covered, kda_strided_onorm
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _make_strided_x(T: int, H: int, D: int, dtype: torch.dtype, device: str) -> torch.Tensor:
    """[1, H*D, T] contiguous -> transpose(1, 2) -> unflatten(-1, (H, D)),
    giving shape [1, T, H, D] with stride (T, 1, T*D, T): token axis stride 1,
    head_dim axis stride T. Matches the KDA extend path's real layout
    (chunk_gla_fwd_o_gk writes into v's storage, v being the transposed conv
    output)."""
    base = torch.randn(1, H * D, T, dtype=dtype, device=device)
    return base.transpose(1, 2).unflatten(-1, (H, D))


def _reference(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    activation: str,
    eps: float,
) -> torch.Tensor:
    out = rms_norm_gated(
        x, g, weight, bias=None, activation=activation, eps=eps
    )
    return out.squeeze(0).flatten(-2)


class TestKdaStridedOnorm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        cls.device = "cuda"

    def _run_case(
        self,
        T: int,
        H: int,
        D: int,
        dtype: torch.dtype,
        activation: str,
        strided: bool,
        eps: float = 1e-5,
    ):
        torch.manual_seed(0)
        if strided:
            x = _make_strided_x(T, H, D, dtype, self.device)
        else:
            x = torch.randn(1, T, H, D, dtype=dtype, device=self.device)
        g = torch.randn(1, T, H, D, dtype=dtype, device=self.device)
        weight = torch.randn(D, dtype=dtype, device=self.device)

        self.assertTrue(
            covered(x, g, weight, activation, eps),
            f"expected covered() True for T={T} H={H} D={D} dtype={dtype} "
            f"act={activation} strided={strided}",
        )

        out = kda_strided_onorm(x, g, weight, activation, eps)
        ref = _reference(x, g, weight, activation, eps)

        self.assertEqual(out.shape, ref.shape)
        self.assertEqual(out.dtype, x.dtype)

        rtol = atol = 8e-3 if dtype == torch.bfloat16 else 2e-3
        torch.testing.assert_close(out.float(), ref.float(), rtol=rtol, atol=atol)

        # x's storage must be untouched: the kernel always allocates a fresh
        # output and never writes through x's aliasing buffer.
        x_bytes_before = x.clone()
        kda_strided_onorm(x, g, weight, activation, eps)
        torch.testing.assert_close(x, x_bytes_before)

    def test_bf16_strided_sigmoid_d128(self):
        # production shape family: T=4096,H=96,D=128,bf16,strided,sigmoid
        self._run_case(T=512, H=8, D=128, dtype=torch.bfloat16, activation="sigmoid", strided=True)

    def test_bf16_strided_swish_d128(self):
        self._run_case(T=256, H=4, D=128, dtype=torch.bfloat16, activation="swish", strided=True)

    def test_bf16_contiguous_sigmoid_d128(self):
        # contiguous-x case (reference README case_008 finding): flat path
        # must not regress vs. the SMEM path.
        self._run_case(T=256, H=4, D=128, dtype=torch.bfloat16, activation="sigmoid", strided=False)

    def test_fp16_strided_sigmoid_d128(self):
        self._run_case(T=256, H=4, D=128, dtype=torch.float16, activation="sigmoid", strided=True)

    def test_fp16_strided_swish_d64(self):
        self._run_case(T=384, H=6, D=64, dtype=torch.float16, activation="swish", strided=True)

    def test_bf16_strided_sigmoid_d64(self):
        self._run_case(T=192, H=3, D=64, dtype=torch.bfloat16, activation="sigmoid", strided=True)

    def test_tail_t_not_multiple_of_tile(self):
        # TILE_T is 128 in the SMEM path; T=40 (also not a multiple of the
        # VT=8 vector width's natural 8*TILE_T groupings) exercises the
        # boundary-checked tail of both the staging loop and the row loop.
        self._run_case(T=40, H=4, D=128, dtype=torch.bfloat16, activation="sigmoid", strided=True)

    def test_tail_t_small_single_tile(self):
        # T smaller than one SMEM tile (TILE_T=128) but still a multiple of
        # VT=8, covering the "single partial tile" boundary.
        self._run_case(T=24, H=2, D=64, dtype=torch.bfloat16, activation="swish", strided=True)

    def test_covered_rejects_unsupported_head_dim(self):
        x = torch.randn(1, 32, 4, 96, dtype=torch.bfloat16, device=self.device)
        g = torch.randn_like(x)
        weight = torch.randn(96, dtype=torch.bfloat16, device=self.device)
        self.assertFalse(covered(x, g, weight, "sigmoid", 1e-5))

    def test_covered_rejects_fp32(self):
        x = torch.randn(1, 32, 4, 128, dtype=torch.float32, device=self.device)
        g = torch.randn_like(x)
        weight = torch.randn(128, dtype=torch.float32, device=self.device)
        self.assertFalse(covered(x, g, weight, "sigmoid", 1e-5))

    def test_covered_rejects_noncontig_g(self):
        x = _make_strided_x(64, 4, 128, torch.bfloat16, self.device)
        g_base = torch.randn(1, 4 * 128, 64, dtype=torch.bfloat16, device=self.device)
        g = g_base.transpose(1, 2).unflatten(-1, (4, 128))  # g's head_dim axis not stride-1
        weight = torch.randn(128, dtype=torch.bfloat16, device=self.device)
        self.assertFalse(covered(x, g, weight, "sigmoid", 1e-5))


class TestKdaStridedOnormEnvGating(unittest.TestCase):
    """Env off / env on / parity, per the env-gated-wiring skill reference.

    Exercises the real call-site branch added to KimiK3DeltaAttention.forward
    in kimi_k3.py: with the flag off (or covered()==False) the original
    self.o_norm(...) + squeeze/flatten path runs; with the flag on and the
    input covered, the new kernel path runs and must match bit-for-bit-close
    output (same non-determinism budget as the direct kernel test above).
    """

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        cls.device = "cuda"

    def _legacy_path(self, x, g, weight, activation, eps):
        out = rms_norm_gated(x, g, weight, bias=None, activation=activation, eps=eps)
        return out.squeeze(0).flatten(-2)

    def _call_site_equivalent(self, x, g, weight, activation, eps):
        """Mirrors the exact branch added in kimi_k3.py's forward(): checks
        the env flag, then covered(), falling back identically."""
        if envs.SGLANG_OPT_USE_KDA_STRIDED_ONORM.get() and covered(
            x, g, weight, activation, eps
        ):
            return kda_strided_onorm(x, g, weight, activation, eps)
        return self._legacy_path(x, g, weight, activation, eps)

    def test_env_off_uses_legacy_path(self):
        torch.manual_seed(0)
        x = _make_strided_x(128, 4, 128, torch.bfloat16, self.device)
        g = torch.randn(1, 128, 4, 128, dtype=torch.bfloat16, device=self.device)
        weight = torch.randn(128, dtype=torch.bfloat16, device=self.device)

        with envs.SGLANG_OPT_USE_KDA_STRIDED_ONORM.override(False):
            out = self._call_site_equivalent(x, g, weight, "sigmoid", 1e-5)
        ref = self._legacy_path(x, g, weight, "sigmoid", 1e-5)
        # env off must be byte-for-byte the legacy call, not just numerically
        # close: no kernel is invoked, so this should be exact.
        torch.testing.assert_close(out, ref, rtol=0, atol=0)

    def test_env_on_triggers_fused_kernel(self):
        torch.manual_seed(0)
        x = _make_strided_x(128, 4, 128, torch.bfloat16, self.device)
        g = torch.randn(1, 128, 4, 128, dtype=torch.bfloat16, device=self.device)
        weight = torch.randn(128, dtype=torch.bfloat16, device=self.device)

        # Actually trigger JIT compilation via the covered() + kernel path
        # (this input is strided and covered, so env on must take the fused
        # branch, not silently fall back).
        self.assertTrue(covered(x, g, weight, "sigmoid", 1e-5))
        with envs.SGLANG_OPT_USE_KDA_STRIDED_ONORM.override(True):
            out = self._call_site_equivalent(x, g, weight, "sigmoid", 1e-5)
        direct = kda_strided_onorm(x, g, weight, "sigmoid", 1e-5)
        torch.testing.assert_close(out, direct, rtol=0, atol=0)

    def test_parity_env_on_vs_env_off(self):
        torch.manual_seed(0)
        x = _make_strided_x(256, 6, 128, torch.bfloat16, self.device)
        g = torch.randn(1, 256, 6, 128, dtype=torch.bfloat16, device=self.device)
        weight = torch.randn(128, dtype=torch.bfloat16, device=self.device)

        with envs.SGLANG_OPT_USE_KDA_STRIDED_ONORM.override(False):
            out_off = self._call_site_equivalent(x, g, weight, "sigmoid", 1e-5)
        with envs.SGLANG_OPT_USE_KDA_STRIDED_ONORM.override(True):
            out_on = self._call_site_equivalent(x, g, weight, "sigmoid", 1e-5)

        torch.testing.assert_close(out_off.float(), out_on.float(), rtol=8e-3, atol=8e-3)

    def test_parity_env_on_uncovered_input_falls_back(self):
        # D=96 is not in {64, 128}: covered() must be False, so env on must
        # still take the legacy branch and match env off exactly.
        torch.manual_seed(0)
        x = torch.randn(1, 32, 4, 96, dtype=torch.bfloat16, device=self.device)
        g = torch.randn_like(x)
        weight = torch.randn(96, dtype=torch.bfloat16, device=self.device)

        self.assertFalse(covered(x, g, weight, "sigmoid", 1e-5))
        with envs.SGLANG_OPT_USE_KDA_STRIDED_ONORM.override(True):
            out_on = self._call_site_equivalent(x, g, weight, "sigmoid", 1e-5)
        with envs.SGLANG_OPT_USE_KDA_STRIDED_ONORM.override(False):
            out_off = self._call_site_equivalent(x, g, weight, "sigmoid", 1e-5)
        torch.testing.assert_close(out_on, out_off, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
