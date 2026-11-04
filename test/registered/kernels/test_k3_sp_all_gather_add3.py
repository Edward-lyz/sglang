"""K3 SP-MoE push all-gather with the MoE add3 tail fused into its producer.

Run on one node: torchrun --nproc-per-node=8 test_k3_sp_all_gather_add3.py

all_gather_add3(a, b, c) must equal all_gather(add3(a, b, c)) bit-for-bit,
eagerly and under CUDA-graph replay, and must keep the push protocol in step
with plain all_gather calls issued around it.
"""

import os
import sys

import torch
import torch.distributed as dist

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="nightly-8-gpu-b200", nightly=True)

HIDDEN = 7168


def _check(name, got, want, rank):
    if not torch.equal(got, want):
        diff = (got.float() - want.float()).abs()
        raise AssertionError(f"rank {rank} {name}: mismatch, max diff {diff.max().item()}")


def main() -> int:
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )
    from sglang.kernels.ops.elementwise import add3
    from sglang.kernels.ops.kimi_k3 import sp_collective
    from sglang.srt.distributed.parallel_state import init_distributed_environment

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    init_distributed_environment(
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["RANK"]),
        local_rank=local_rank,
        backend="nccl",
    )
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", local_rank)
    comm = CustomAllReduceV2(group=dist.new_group(backend="gloo"), device=device)
    assert not comm.disabled and comm.mc_base_ptr != 0, "needs CustomAllReduceV2 multicast"
    sp_collective.register_comm(comm.obj, pull_sem_mc_ptr=comm.pull_sem_mc_ptr)

    def ag(x):
        out = torch.empty(x.shape[0] * world, HIDDEN, dtype=x.dtype, device=device)
        return sp_collective.all_gather(world, x, out, ws_mc_base=comm.mc_base_ptr)

    def ag_add3(a, b, c):
        out = torch.empty(a.shape[0] * world, HIDDEN, dtype=a.dtype, device=device)
        return sp_collective.all_gather_add3(
            world, a, b, c, out, ws_mc_base=comm.mc_base_ptr
        )

    for local_tokens in (1, 4, 24, 48):
        g = torch.Generator(device=device).manual_seed(100 * local_tokens + rank)
        a, b, c = (
            torch.randn(local_tokens, HIDDEN, dtype=torch.bfloat16, device=device, generator=g)
            * s
            for s in (1.0, 0.3, 4.0)
        )
        a[0, :16] = 0  # zero pairs exercise the push empty-marker rewrite
        b[0, :16] = 0
        c[0, :16] = 0
        want = ag(add3.add3(a, b, c))
        got = ag_add3(a, b, c)
        torch.cuda.synchronize()
        _check(f"eager T={local_tokens}", got, want, rank)
        # Interleaved with plain gathers: the shared phase counters stay in step.
        _check(f"plain-after T={local_tokens}", ag(a), ag(a.clone()), rank)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            ag_add3(a, b, c)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = ag_add3(a, b, c)
        for step in range(3):
            a.copy_(torch.randn_like(a) * (step + 1))
            graph.replay()
            torch.cuda.synchronize()
            _check(f"graph T={local_tokens} step={step}", captured, ag(add3.add3(a, b, c)), rank)
        dist.barrier()
    if rank == 0:
        print("all_gather_add3: all checks passed", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
