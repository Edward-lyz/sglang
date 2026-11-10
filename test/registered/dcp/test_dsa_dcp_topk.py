"""dcp_global_topk (SGLANG_K31_INDEXER_DCP) against a global top-k.

The DCP group is emulated on one GPU: one thread per rank, and a fake group
whose all_reduce sums the ranks' tensors in lockstep (barrier).
"""

import os
import threading
import unittest

import torch

from sglang.kernels.ops.attention.dsa.dcp_topk import dcp_global_topk

TOPK = 2048
# TRITON_INTERPRET=1 runs the kernels on CPU tensors.
DEVICE = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"


class _LockstepGroup:
    def __init__(self, world_size: int):
        self.world_size = world_size
        self.barrier = threading.Barrier(world_size)
        self.slots = [None] * world_size
        self.result = None

    def rank_view(self, rank: int):
        group = self

        class _View:
            def all_reduce(self, tensor):
                group.slots[rank] = tensor
                if group.barrier.wait() == 0:
                    group.result = torch.stack(group.slots).sum(0)
                group.barrier.wait()
                out = group.result.clone()
                group.barrier.wait()
                return out

        return _View()


def _run(dcp, lengths, rounds, seed=0):
    """Global scores per (row, position); rank r owns positions t % dcp == r."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    rows = len(lengths)
    max_len = max(lengths)
    scores = torch.rand((rows, max_len), generator=gen, device=DEVICE) * 10
    lengths_t = torch.tensor(lengths, dtype=torch.int32, device=DEVICE)
    group = _LockstepGroup(dcp)
    results = [None] * dcp

    def rank_fn(rank):
        local_w = (max_len + dcp - 1) // dcp
        logits = torch.full((rows, local_w), -1.0, device=DEVICE)
        local = scores[:, rank::dcp]
        logits[:, : local.shape[1]] = local
        local_lens = [len(range(rank, n, dcp)) for n in lengths]
        raw = torch.full((rows, TOPK), -1, dtype=torch.int32, device=DEVICE)
        for i, n in enumerate(local_lens):
            k = min(TOPK, n)
            if k:
                raw[i, :k] = torch.topk(logits[i, :n], k).indices.to(torch.int32)
        local_rows = raw.clone()  # local KV row == local position here
        out, counts = dcp_global_topk(
            logits, raw, local_rows, lengths_t, TOPK, group.rank_view(rank),
            rank, dcp, rounds=rounds,
        )
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        results[rank] = (out.cpu(), counts.cpu())

    threads = [threading.Thread(target=rank_fn, args=(r,)) for r in range(dcp)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return scores.cpu(), results


class TestDsaDcpTopk(unittest.TestCase):
    def _check(self, dcp, lengths, rounds, exact):
        scores, results = _run(dcp, lengths, rounds)
        for i, n in enumerate(lengths):
            k = min(TOPK, n)
            positions = []
            for rank, (out, counts) in enumerate(results):
                c = int(counts[i])
                slots = out[i, :c]
                self.assertTrue(bool((out[i, c:] == -1).all()))
                self.assertTrue(bool((slots % dcp == rank).all()))
                # widened slot = local row * dcp + rank; local row = position // dcp
                positions += (slots // dcp * dcp + rank).tolist()
            self.assertEqual(len(positions), k)
            self.assertEqual(len(set(positions)), k)
            got = scores[i, positions]
            ref = torch.topk(scores[i, :n], k).values
            if exact:
                self.assertTrue(torch.equal(got.sort().values, ref.sort().values))
            else:
                self.assertGreaterEqual(
                    float(got.min()), float(ref.min()) * (1 - 2**-12)
                )

    def test_exact_four_rounds(self):
        self._check(8, [100, 2048, 2049, 30000, 0, 70000], rounds=4, exact=True)

    def test_three_rounds(self):
        self._check(8, [5, 4096, 50000, 120000], rounds=3, exact=False)

    def test_dcp4(self):
        self._check(4, [3000, 10, 64000], rounds=4, exact=True)


if __name__ == "__main__":
    unittest.main()
