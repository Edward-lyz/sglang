"""Row-sharded engram gather (engram_peer): the multi-shard packed gather returns the
same bytes as the single-table gather, and unpacking them gives the engram_gather
dequant bit for bit. Shards live in separate allocations addressed by raw
pointers, like the VMM-mapped peer shards."""

import unittest

import torch

from sglang.kernels.ops.embeddings.engram_gather import (
    engram_gather,
    engram_gather_packed,
    engram_peer_gather_packed,
    engram_unpack,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")

DIM, BLK = 256, 32
WIDTH = DIM + DIM // BLK


class TestEngramPeerGather(CustomTestCase):
    """engram_peer_gather_packed vs the single-table packed gather."""

    def test_matches_single_table(self):
        """Multi-shard gather + unpack is bit-identical to engram_gather."""
        torch.manual_seed(5)
        n_rows, shards, layers, cols = 50021, 8, 2, 24
        weight = (torch.randn(n_rows, DIM) * 3).to(torch.float8_e4m3fn)
        scale = torch.randint(110, 140, (n_rows, DIM // BLK), dtype=torch.uint8)
        starts = [n_rows * i // shards for i in range(shards + 1)]
        dev = torch.device("cuda:0")
        bufs = []
        for s in range(shards):
            lo, hi = starts[s], starts[s + 1]
            raw = torch.cat(
                [weight[lo:hi].view(torch.uint8).reshape(-1), scale[lo:hi].reshape(-1)]
            )
            bufs.append(raw.to(dev))
        bases = torch.tensor(
            [b.data_ptr() for b in bufs], dtype=torch.int64, device=dev
        )
        bounds = torch.tensor(starts, dtype=torch.int64, device=dev)

        w_full = weight.to(dev)
        s_full = scale.view(torch.float8_e8m0fnu).to(dev)
        # Token counts: decode (1), odd prefill (777); ids include -1 and past-the-end.
        for tokens in (1, 777):
            ids = torch.randint(
                -1, n_rows + 3, (tokens, layers, cols), dtype=torch.int32, device=dev
            )
            got = torch.empty(
                tokens, layers, cols, WIDTH, dtype=torch.uint8, device=dev
            )
            ref = torch.empty_like(got)
            for layer in range(layers):
                engram_peer_gather_packed(bases, bounds, ids, got, layer, DIM)
                engram_gather_packed(w_full, s_full, ids, ref, layer, 0)
            self.assertTrue(torch.equal(got, ref))

            fill = torch.zeros(layers, WIDTH, dtype=torch.uint8, device=dev)
            values, _ = engram_unpack(
                got.unsqueeze(0), tokens, tokens, fill=fill, out_bf16=True
            )
            for layer in range(layers):
                want = torch.empty(tokens * cols, DIM, dtype=torch.bfloat16, device=dev)
                engram_gather(
                    w_full.data_ptr(),
                    s_full.data_ptr(),
                    ids[:, layer].reshape(-1).contiguous(),
                    want,
                    DIM,
                    BLK,
                    0,
                    n_rows,
                )
                self.assertTrue(torch.equal(values[layer].view(-1, DIM), want))


if __name__ == "__main__":
    unittest.main()
