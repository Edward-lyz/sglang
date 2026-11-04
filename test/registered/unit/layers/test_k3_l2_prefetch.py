from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

import unittest

import torch

from sglang.kernels.ops.kimi_k3 import l2_prefetch


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9,
    "needs SM90+ (cp.async.bulk.prefetch.L2)",
)
class TestK3L2Prefetch(unittest.TestCase):
    def test_prefetch_leaves_data_untouched(self):
        torch.manual_seed(0)
        # Sizes around the chunk (16 KiB) boundary and an expert-weight slice.
        for numel in (16, 4096, 8192 + 16, 3 * 2**20 + 48):
            t = torch.randint(0, 255, (numel,), dtype=torch.uint8, device="cuda")
            ref = t.clone()
            self.assertTrue(l2_prefetch.covered(t))
            l2_prefetch.l2_prefetch(t)
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(t, ref))

    def test_prefetch_under_cuda_graph(self):
        w = torch.randn(9, 6144, 1792, device="cuda").to(torch.float8_e4m3fn)
        ref = w.view(torch.uint8).clone()
        side = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            cur = torch.cuda.current_stream()
            side.wait_stream(cur)
            with torch.cuda.stream(side):
                l2_prefetch.l2_prefetch(w)
            cur.wait_stream(side)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(w.view(torch.uint8), ref))

    def test_covered_rejects_unaligned(self):
        t = torch.empty(33, dtype=torch.uint8, device="cuda")
        self.assertFalse(l2_prefetch.covered(t[1:17]))
        self.assertFalse(l2_prefetch.covered(t[:17]))


if __name__ == "__main__":
    unittest.main()
