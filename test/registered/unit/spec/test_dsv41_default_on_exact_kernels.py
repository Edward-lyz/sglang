"""Exactness tests for the default-on DSV4.1 decode kernels (decode_opt_1017).

Each kernel replaces a short PyTorch / two-kernel sequence and must give the same bits:
  * topk_ids_cast.cast_mask_topk_ids   == topk_idx.to(int32) + padded-row mask (-1)
  * logits_cast.copy_bf16_to_fp32      == dst.copy_(bf16 src)
  * hc_mean.hc_mean                    == x.mean(dim=1)
  * build_step_sample_triton           == SampleStepTokens(BuildStepLocal(...)) tokens and
                                          the rounded block logits it also writes
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestDefaultOnExactKernels(unittest.TestCase):
    def test_cast_mask_topk_ids(self):
        from sglang.kernels.ops.moe.topk_ids_cast import cast_mask_topk_ids

        torch.manual_seed(0)
        for rows in (1, 7, 256, 1536):
            for dtype in (torch.int64, torch.int32):
                idx = torch.randint(0, 256, (rows, 6), device="cuda", dtype=dtype)
                for n in (None, 0, 1, rows // 2, rows):
                    nn = (
                        None
                        if n is None
                        else torch.tensor([n], device="cuda", dtype=torch.int32)
                    )
                    ref = idx.to(torch.int32)
                    if n is not None:
                        ref[torch.arange(rows, device="cuda") >= n] = -1
                    self.assertTrue(torch.equal(cast_mask_topk_ids(idx, nn), ref))

    def test_logits_cast(self):
        from sglang.kernels.ops.speculative.logits_cast import copy_bf16_to_fp32

        src = torch.randn(300, 129280, device="cuda").bfloat16()
        dst = torch.empty(300, 129280, device="cuda", dtype=torch.float32)
        copy_bf16_to_fp32(src, dst)
        self.assertTrue(torch.equal(dst, src.float()))

    def test_hc_mean(self):
        from sglang.kernels.ops.layernorm.hc_mean import hc_mean

        torch.manual_seed(0)
        for rows in (1, 5, 256, 1536):
            x = (torch.randn(rows, 4, 5120, device="cuda") * 3).bfloat16()
            self.assertTrue(torch.equal(hc_mean(x), x.mean(dim=1)))

    def test_build_step_sample(self):
        from sglang.kernels.ops.speculative.dspark.dspark_draft_model import (
            BuildStepLocal,
            SampleStepTokens,
            build_step_sample_triton,
        )

        torch.manual_seed(0)
        vocab = 129280
        for bs, padded in ((256, 0), (37, 64)):
            width = vocab + padded
            base = torch.randn(bs, 5, width, device="cuda").bfloat16()[:, 2, :]
            bias = torch.randn(bs, vocab, device="cuda") * 2
            temps = torch.rand(bs, device="cuda") * 0.9 + 0.1
            greedy = (torch.rand(bs, device="cuda") < 0.5).to(torch.int32)
            counter = torch.tensor([12345], device="cuda", dtype=torch.int64)
            for step in range(5):
                dst_ref = torch.zeros(bs, 5, vocab, device="cuda").bfloat16()[:, step, :]
                dst = torch.zeros_like(dst_ref)
                full = BuildStepLocal.execute(
                    bias=bias, base_local=base, corrected_dst=dst_ref
                )[..., :vocab]
                ref = SampleStepTokens.execute(
                    step_logits=full,
                    temperatures=temps,
                    greedy_mask=greedy,
                    rng_counter=counter,
                    rng_step=step,
                    rng_num_steps=5,
                )
                got = build_step_sample_triton(
                    bias=bias,
                    base_local=base,
                    corrected_dst=dst,
                    vocab=vocab,
                    temperatures=temps,
                    greedy_mask=greedy,
                    rng_counter=counter,
                    rng_step=step,
                    rng_num_steps=5,
                )
                self.assertTrue(torch.equal(ref, got))
                self.assertTrue(torch.equal(dst_ref, dst))


    def test_commit_swa_loc(self):
        from sglang.kernels.ops.speculative.dspark.inject_loc import commit_swa_loc

        torch.manual_seed(0)
        for bs, verify_len in ((1, 6), (18, 6), (256, 6), (37, 5)):
            n = bs * verify_len
            mapping = torch.randint(0, 1 << 20, (4096,), device="cuda", dtype=torch.int64)
            for loc_dtype in (torch.int64, torch.int32):
                cache_loc = torch.randint(0, 4096, (n,), device="cuda", dtype=loc_dtype)
                commit = torch.randint(0, verify_len + 1, (bs,), device="cuda", dtype=torch.int32)
                for row_valid in (None, torch.rand(n, device="cuda") < 0.7):
                    ref = mapping[cache_loc.long()].to(torch.int32)
                    col = torch.arange(verify_len, device="cuda").view(1, -1)
                    committed = (col < commit.long().view(-1, 1)).reshape(-1)
                    ref = torch.where(committed, ref, torch.full_like(ref, -1))
                    if row_valid is not None:
                        ref = torch.where(row_valid, ref, torch.full_like(ref, -1))
                    got = commit_swa_loc(mapping, cache_loc, commit, verify_len, row_valid)
                    self.assertTrue(torch.equal(ref, got))

    def test_softmax_split_matches_torch(self):
        from sglang.kernels.ops.speculative.dspark.dspark_accept import (
            softmax_temp,
            softmax_temp_split_triton,
        )

        torch.manual_seed(0)
        vocab = 129280
        for rows_per_request, bs, dtype in ((6, 18, torch.float32), (5, 18, torch.bfloat16)):
            n = rows_per_request * bs
            x = (torch.randn(n, 2, vocab, device="cuda") * 3).to(dtype)[:, 1, :]
            temps = torch.rand(bs, device="cuda") * 1.5 + 0.2
            ref = softmax_temp(logits=x, temperatures=temps, rows_per_request=rows_per_request)
            got = softmax_temp_split_triton(
                logits=x, temperatures=temps, rows_per_request=rows_per_request
            )
            # not bit-exact (reduction order); fp32-rounding level agreement
            self.assertLess(((got - ref).abs() / (ref + 1e-30)).max().item(), 5e-5)
            self.assertLess((got.sum(-1) - 1).abs().max().item(), 1e-5)
            self.assertTrue(torch.equal(got.argmax(-1), ref.argmax(-1)))

    def test_softmax_split_handles_neg_inf_masks(self):
        # Grammar / vocab masks leave most of the vocab at -inf, so whole 4096-wide tiles are
        # all -inf. The split kernel used to return NaN rows there, which the sampler turned
        # into token vocab-1 (function-call / xgrammar requests then failed to accept it).
        from sglang.kernels.ops.speculative.dspark.dspark_accept import (
            softmax_temp,
            softmax_temp_split_triton,
        )

        torch.manual_seed(0)
        vocab = 129280
        rows_per_request, bs = 6, 4
        n = rows_per_request * bs
        logits = torch.randn(n, vocab, device="cuda").to(torch.bfloat16)
        temps = torch.rand(bs, device="cuda") * 1.5 + 0.2
        allowed_sets = (
            [1018, 10699, 128822, 30],  # tiles 0 and 31 only
            list(range(0, 5000)),  # first two tiles, second one partial
            [129279],  # a single token in the last (partial) tile
        )
        for allowed in allowed_sets:
            masked = torch.full_like(logits, float("-inf"))
            idx = torch.tensor(allowed, device="cuda")
            masked[:, idx] = logits[:, idx]
            ref = softmax_temp(logits=masked, temperatures=temps, rows_per_request=rows_per_request)
            got = softmax_temp_split_triton(
                logits=masked, temperatures=temps, rows_per_request=rows_per_request
            )
            self.assertFalse(torch.isnan(got).any().item())
            self.assertLess((got - ref).abs().max().item(), 1e-5)
            self.assertLess((got.sum(-1) - 1).abs().max().item(), 1e-5)
            self.assertTrue(torch.equal(got.argmax(-1), ref.argmax(-1)))

    def test_stage_sampling_params(self):
        from sglang.kernels.ops.speculative.dspark.host_sched import (
            stage_sampling_params_fused,
        )

        torch.manual_seed(0)
        for bs in (1, 18, 256, 1500):
            for t_dtype, k_dtype in ((torch.float32, torch.int32), (torch.bfloat16, torch.int64)):
                temps = torch.rand(bs, device="cuda").to(t_dtype) * 2
                temps[::7] = 0
                top_k = torch.randint(0, 5, (bs,), device="cuda", dtype=k_dtype)
                out_t = torch.zeros(2048, device="cuda")
                out_g = torch.zeros(2048, device="cuda", dtype=torch.bool)
                counter = torch.tensor([41], device="cuda", dtype=torch.int64)
                stage_sampling_params_fused(temps, top_k, out_t, out_g, counter, bs)
                ref_t = torch.clamp(temps.to(torch.float32), min=1e-5)
                self.assertTrue(torch.equal(out_t[:bs], ref_t))
                self.assertTrue(torch.equal(out_g[:bs], top_k <= 1))
                self.assertEqual(int(counter.item()), 42)
                self.assertEqual(float(out_t[bs:].abs().sum()), 0.0)

    def test_local_num_token_non_padded(self):
        from sglang.kernels.ops.speculative.dspark.host_sched import (
            local_num_token_non_padded_into,
        )

        for g_dtype in (torch.int32, torch.int64):
            for g, off, chunk in ((0, 0, 8), (5, 0, 8), (13, 8, 8), (100, 24, 8), (3, 8, 8)):
                glob = torch.tensor([g], device="cuda", dtype=g_dtype)
                out = torch.zeros(1, device="cuda", dtype=torch.int32)
                local_num_token_non_padded_into(out, glob, off, chunk)
                self.assertEqual(int(out.item()), min(max(g - off, 0), chunk))


if __name__ == "__main__":
    unittest.main()
