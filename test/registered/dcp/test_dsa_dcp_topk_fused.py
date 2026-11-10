"""dcp_global_topk_fused (SGLANG_K31_INDEXER_DCP_FUSED_TOPK) on N GPUs.

Each rank owns positions t % N == rank of shared random rows, keeps its local
top-2048 and runs the fused select; the union of the ranks' picks must be the
exact global top-k (plus ties broken consistently). Needs >= 2 GPUs (SM90+).
"""

import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

TOPK = 2048
LENGTHS = [0, 7, 2048, 2049, 5000, 40000, 300000]


class _Group:
    def __init__(self, world_size):
        self.world_size = world_size
        self.unique_name = "dcp_topk_test"
        self.cpu_group = dist.new_group(backend="gloo")


def _worker(rank, world, port, q):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    from sglang.kernels.ops.attention.dsa.dcp_topk_fused import dcp_global_topk_fused

    group = _Group(world)
    torch.manual_seed(0)
    rows = len(LENGTHS) * 2
    lens = LENGTHS * 2
    width = max(LENGTHS)
    scores = torch.rand((rows, width)) * 10
    scores[len(LENGTHS) :] = scores[len(LENGTHS) :].mul(4).round()  # exact ties
    local_w = (width + world - 1) // world
    logits = torch.full((rows, local_w), float("-inf"))
    loc = scores[:, rank::world]
    logits[:, : loc.shape[1]] = loc
    logits = logits.cuda()
    raw = torch.full((rows, TOPK), -1, dtype=torch.int32)
    for i, n in enumerate(lens):
        m = len(range(rank, n, world))
        k = min(TOPK, m)
        if k:
            raw[i, :k] = torch.topk(logits[i, :m].cpu(), k).indices.to(torch.int32)
    raw = raw.cuda()
    lengths = torch.tensor(lens, dtype=torch.int32, device="cuda")
    ok = True
    for it in range(3):  # buffer / epoch reuse
        out, counts = dcp_global_topk_fused(
            logits, raw, raw.clone(), lengths, group, rank
        )
        torch.cuda.synchronize()
        gathered = [None] * world
        dist.all_gather_object(gathered, (out.cpu(), counts.cpu()))
        if rank == 0:
            for i, n in enumerate(lens):
                pos = []
                for r, (o, c) in enumerate(gathered):
                    sl = o[i, : int(c[i])]
                    ok &= bool((sl % world == r).all()) and bool((o[i, int(c[i]) :] == -1).all())
                    pos += (sl // world * world + r).tolist()
                k = min(TOPK, n)
                ok &= len(pos) == k == len(set(pos))
                if k:
                    got = scores[i, pos].sort().values
                    ref = torch.topk(scores[i, :n], k).values.sort().values
                    ok &= torch.equal(got, ref)
    if rank == 0:
        q.put(ok)
    dist.barrier()
    dist.destroy_process_group()


@unittest.skipIf(torch.cuda.device_count() < 2, "needs >= 2 GPUs")
class TestDsaDcpTopkFused(unittest.TestCase):
    def test_exact(self):
        world = min(8, torch.cuda.device_count())
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        mp.spawn(_worker, args=(world, 29571, q), nprocs=world, join=True)
        self.assertTrue(q.get())


if __name__ == "__main__":
    unittest.main()
