"""K3 DCP MLA decode: peer-memory pack + exchange (SGLANG_K3_DCP_PUSH).

Run on one node: torchrun --nproc-per-node=8 test_k3_dcp_push.py

_dcp_push_lse_reduce (peer stores into the receive layout, unchanged Triton
combine) must equal _dcp_a2a_lse_reduce_fused_pack (Triton pack + NCCL
all_to_all_single + the same combine) bit-for-bit, eagerly, across many
back-to-back calls (the single receive buffer is reused every call) and under
CUDA-graph replay.
"""

import os
import sys

import torch
import torch.distributed as dist

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="nightly-8-gpu-b200", nightly=True)


def _check(name, got, want, rank):
    # Bitwise (NaN-safe): rows whose partials are empty on every rank are NaN
    # in both paths.
    if not torch.equal(got.view(torch.int16), want.view(torch.int16)):
        diff = (got.float() - want.float()).abs().nan_to_num(0.0)
        raise AssertionError(f"rank {rank} {name}: mismatch, max diff {diff.max().item()}")


def main() -> int:
    from sglang.srt.distributed.parallel_state import (
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.layers.dcp import comm

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    world = int(os.environ["WORLD_SIZE"])
    init_distributed_environment(
        world_size=world, rank=int(os.environ["RANK"]), local_rank=local_rank, backend="nccl"
    )
    initialize_model_parallel(tensor_model_parallel_size=world)
    group = get_tp_group()
    rank = group.rank_in_group
    device = torch.device("cuda", local_rank)

    def inputs(B, H, D, seed):
        g = torch.Generator(device=device).manual_seed(seed * 100 + rank)
        out = torch.randn(B, H, D, dtype=torch.bfloat16, device=device, generator=g)
        lse = torch.randn(B, H, dtype=torch.float32, device=device, generator=g) * 4
        lse[0, : 4 * (rank % 2)] = float("-inf")  # empty-KV rows on half the ranks
        out[0, : 4 * (rank % 2)] = 0
        return out, lse

    for B in (192, 64, 8):
        H, D = 96, 128
        for seed in range(6):  # back-to-back calls reuse the receive buffer
            out, lse = inputs(B, H, D, seed)
            want = comm._dcp_a2a_lse_reduce_fused_pack(out, lse, group, False)
            got = comm._dcp_push_lse_reduce(out, lse, group, False)
            assert got is not None, "push path fell back"
            torch.cuda.synchronize()
            _check(f"eager B={B} seed={seed}", got, want, rank)

        out, lse = inputs(B, H, D, 99)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            comm._dcp_push_lse_reduce(out, lse, group, False)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            a = comm._dcp_push_lse_reduce(out, lse, group, False)
            b = comm._dcp_push_lse_reduce(out * 2, lse - 1, group, False)
        for step in range(3):
            o2, l2 = inputs(B, H, D, 200 + step)
            out.copy_(o2)
            lse.copy_(l2)
            graph.replay()
            torch.cuda.synchronize()
            _check(f"graph B={B} step={step} a", a, comm._dcp_a2a_lse_reduce_fused_pack(out, lse, group, False), rank)
            _check(f"graph B={B} step={step} b", b, comm._dcp_a2a_lse_reduce_fused_pack(out * 2, lse - 1, group, False), rank)
        dist.barrier()
    if rank == 0:
        print("dcp_push: all checks passed", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
