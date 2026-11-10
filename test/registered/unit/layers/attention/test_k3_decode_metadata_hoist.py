from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

import types
import unittest

import torch

from sglang.srt.layers.attention.linear.kda_backend import (
    KDAAttnBackend,
    _dense_verify_cu_seqlens,
)
from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend

# K3 decode on B300 TP8/DCP8, DSPARK block 5 -> verify width 6.
DRAFT_TOKEN_NUM = 6


def _arange(n, step):
    return torch.arange(0, n + step, step, device="cuda", dtype=torch.int32)


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestK3DecodeMetadataHoist(unittest.TestCase):
    def test_kda_verify_cu_seqlens_matches_per_layer_arange(self):
        backend = types.SimpleNamespace()
        for bs in (1, 7, 32, 64):
            fb = object()
            seq_len = bs * DRAFT_TOKEN_NUM
            ref = _arange(seq_len, DRAFT_TOKEN_NUM)
            first = KDAAttnBackend._verify_cu_seqlens(
                backend, fb, seq_len, DRAFT_TOKEN_NUM, torch.device("cuda")
            )
            self.assertTrue(torch.equal(first, ref))
            self.assertTrue(
                torch.equal(_dense_verify_cu_seqlens(seq_len, DRAFT_TOKEN_NUM, "cuda"), ref)
            )
            # Every later layer of the same forward reuses the tensor.
            again = KDAAttnBackend._verify_cu_seqlens(
                backend, fb, seq_len, DRAFT_TOKEN_NUM, torch.device("cuda")
            )
            self.assertIs(again, first)
        # A new forward_batch rebuilds even with the same shape.
        fb2 = object()
        other = KDAAttnBackend._verify_cu_seqlens(
            backend, fb2, 64 * DRAFT_TOKEN_NUM, DRAFT_TOKEN_NUM, torch.device("cuda")
        )
        self.assertIsNot(other, first)
        self.assertTrue(torch.equal(other, first))

    def test_kda_verify_cu_seqlens_rebuilt_under_capture(self):
        backend = types.SimpleNamespace()
        fb = object()
        eager = KDAAttnBackend._verify_cu_seqlens(
            backend, fb, 32 * DRAFT_TOKEN_NUM, DRAFT_TOKEN_NUM, torch.device("cuda")
        )
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            captured = KDAAttnBackend._verify_cu_seqlens(
                backend, fb, 32 * DRAFT_TOKEN_NUM, DRAFT_TOKEN_NUM, torch.device("cuda")
            )
            captured_again = KDAAttnBackend._verify_cu_seqlens(
                backend, fb, 32 * DRAFT_TOKEN_NUM, DRAFT_TOKEN_NUM, torch.device("cuda")
            )
        self.assertIsNot(captured, eager)
        self.assertIs(captured_again, captured)
        captured.zero_()
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(captured, _arange(32 * DRAFT_TOKEN_NUM, DRAFT_TOKEN_NUM)))

    def test_mla_dense_verify_q_indptr_matches_per_layer_arange(self):
        backend = types.SimpleNamespace()
        for bs in (1, 7, 32, 64):
            fb = object()
            ref = torch.arange(
                0,
                (bs + 1) * DRAFT_TOKEN_NUM,
                DRAFT_TOKEN_NUM,
                dtype=torch.int32,
                device="cuda",
            )
            first = TRTLLMMLABackend._dense_verify_q_indptr(
                backend, fb, bs, DRAFT_TOKEN_NUM, torch.device("cuda")
            )
            self.assertTrue(torch.equal(first, ref))
            again = TRTLLMMLABackend._dense_verify_q_indptr(
                backend, fb, bs, DRAFT_TOKEN_NUM, torch.device("cuda")
            )
            self.assertIs(again, first)


if __name__ == "__main__":
    unittest.main()
