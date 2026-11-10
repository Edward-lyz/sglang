from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

import unittest

import torch

from sglang.kernels.ops.moe import moe_route_radix


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestRouteRadixInt64Ids(unittest.TestCase):
    def tearDown(self):
        moe_route_radix.request_int64_ids(False)

    def _route(self, scores, bias, sorted_):
        return moe_route_radix.route_radix(
            scores, bias, 16, True, 2.5, True, sorted=sorted_
        )

    def test_int64_copy_matches_and_int32_unchanged(self):
        torch.manual_seed(0)
        for tokens in (1, 6, 192, 1000):
            for dtype in (torch.float32, torch.bfloat16):
                for sorted_ in (False, True):
                    scores = torch.randn(tokens, 896, device="cuda", dtype=dtype)
                    bias = torch.randn(896, device="cuda", dtype=torch.float32) * 0.1
                    moe_route_radix.request_int64_ids(False)
                    w_ref, i_ref = self._route(scores, bias, sorted_)
                    moe_route_radix.request_int64_ids(True)
                    w, i = self._route(scores, bias, sorted_)
                    self.assertTrue(torch.equal(w, w_ref))
                    self.assertTrue(torch.equal(i, i_ref))
                    self.assertEqual(i.dtype, torch.int32)
                    i64 = moe_route_radix.int64_ids_for(i)
                    self.assertIsNotNone(i64)
                    self.assertEqual(i64.dtype, torch.int64)
                    self.assertTrue(torch.equal(i64, i.to(torch.int64)))

    def test_lookup_only_for_the_same_ids(self):
        scores = torch.randn(8, 896, device="cuda", dtype=torch.float32)
        bias = torch.zeros(896, device="cuda", dtype=torch.float32)
        moe_route_radix.request_int64_ids(True)
        _, i = self._route(scores, bias, False)
        self.assertIsNone(moe_route_radix.int64_ids_for(i.clone()))
        self.assertIsNone(moe_route_radix.int64_ids_for(i[:4]))
        _, i2 = self._route(scores, bias, False)
        self.assertIsNone(moe_route_radix.int64_ids_for(i))
        self.assertIsNotNone(moe_route_radix.int64_ids_for(i2))


if __name__ == "__main__":
    unittest.main()
