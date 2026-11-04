from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

import unittest

import torch

from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.deep_gemm_wrapper import entrypoint

# K3 decode DeepEP-LL shapes: 112 local experts, m_max = 60 tokens/rank * 8 ranks,
# GEMM1 gate_up (N=6144, K=3584) and GEMM2 down (N=3584, K=3072), FP4 weights.
E, M_MAX = 112, 480
SHAPES = ((6144, 3584), (3584, 3072))


def _is_sm100():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


def _masked_m(global_tokens, hot_experts=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    p = torch.ones(896)
    if hot_experts:
        p = torch.zeros(896)
        p[torch.randperm(896, generator=g)[:hot_experts]] = 1.0
    ids = torch.stack(
        [torch.multinomial(p, 16, replacement=False, generator=g) for _ in range(global_tokens)]
    )
    cnt = torch.bincount(ids.flatten(), minlength=896)[:E].clamp(max=M_MAX)
    return cnt.to(torch.int32).cuda()


def _inputs(n, k):
    import deep_gemm
    from deep_gemm.utils.math import per_token_cast_to_fp4, per_token_cast_to_fp8

    torch.manual_seed(0)
    w = torch.randn(E, n, k, device="cuda", dtype=torch.bfloat16)
    wq, wsf = zip(*[per_token_cast_to_fp4(w[i], use_ue8m0=True, gran_k=32) for i in range(E)])
    wsf = deep_gemm.transform_sf_into_required_layout(
        torch.stack(wsf).float(), mn=n, k=k, recipe=(1, 32), num_groups=E
    )
    x = torch.randn(E, M_MAX, k, device="cuda", dtype=torch.bfloat16)
    a8, s8 = zip(*[per_token_cast_to_fp8(x[i], use_ue8m0=True, gran_k=128) for i in range(E)])
    s8 = deep_gemm.transform_sf_into_required_layout(
        torch.stack(s8).float(), mn=M_MAX, k=k, recipe=(1, 128), num_groups=E
    )
    return (torch.stack(a8), s8), (torch.stack(wq), wsf)


@unittest.skipUnless(_is_sm100() and deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM, "needs SM100 DeepGEMM")
class TestDeepGemmMaskedBlockM(unittest.TestCase):
    def _run(self, block_m, lhs, rhs, masked_m, expected_m, n):
        out = torch.zeros(E, M_MAX, n, device="cuda", dtype=torch.bfloat16)
        old = entrypoint._MASKED_BLOCK_M
        entrypoint._MASKED_BLOCK_M = block_m
        try:
            deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
                lhs, rhs, out, masked_m, expected_m, recipe_a=(1, 128), recipe_b=(1, 32)
            )
        finally:
            entrypoint._MASKED_BLOCK_M = old
        torch.cuda.synchronize()
        return out

    def test_block_m_bitwise_equal_and_alignment_restored(self):
        import deep_gemm

        default_align = deep_gemm.get_mk_alignment_for_contiguous_layout()
        for n, k in SHAPES:
            lhs, rhs = _inputs(n, k)
            for tokens, hot in ((96, None), (96, 64), (48, None), (16, None)):
                masked_m = _masked_m(tokens, hot)
                expected_m = (tokens * 16 + 896) // 896
                ref = self._run(0, lhs, rhs, masked_m, expected_m, n)
                for block_m in (16, 32, 64):
                    out = self._run(block_m, lhs, rhs, masked_m, expected_m, n)
                    self.assertEqual(
                        deep_gemm.get_mk_alignment_for_contiguous_layout(), default_align
                    )
                    for e in range(E):
                        rows = int(masked_m[e])
                        if rows:
                            self.assertTrue(
                                torch.equal(out[e, :rows], ref[e, :rows]),
                                f"N={n} K={k} tokens={tokens} hot={hot} block_m={block_m} expert={e}",
                            )


if __name__ == "__main__":
    unittest.main()
