"""K3 SP-MoE GEMM-RS: o_proj FP8 GEMM whose epilogue pushes the RS slots.

Run on one node: torchrun --nproc-per-node=8 test_k3_gemm_rs.py
(needs the DeepGEMM fork API ``fp8_gemm_nt_push_rs``).

fp8_gemm_nt_push_rs + reduce_scatter_poll must equal fp8_gemm_nt +
reduce_scatter_res bit-for-bit (with/without residual, local/full residual),
eagerly and under CUDA-graph replay, and keep the push protocol in step with
plain reduce_scatter_res calls issued around it.
"""

import os
import sys

import torch
import torch.distributed as dist

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, suite="nightly-8-gpu-b200", nightly=True)

HIDDEN = 7168


def _check(name, got, want, rank):
    if not torch.equal(got, want):
        diff = (got.float() - want.float()).abs()
        raise AssertionError(f"rank {rank} {name}: mismatch, max diff {diff.max().item()}")


def main() -> int:
    import deep_gemm

    from sglang.kernels.ops.kimi_k3 import sp_collective
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )
    from sglang.srt.distributed.parallel_state import init_distributed_environment
    from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8

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
    layout = sp_collective.push_layout(world)
    assert layout.rank == rank

    def quant(x):
        return sglang_per_token_group_quant_fp8(
            x, 128, column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=True
        )

    def tuning(t):
        d = sp_collective.get_dispatch("reduce_scatter", world, HIDDEN, t, device)
        return sp_collective.DEFAULT_TUNING if d is None else d.tuning

    def ref(a, w, residual):
        d = torch.empty(a[0].shape[0], HIDDEN, dtype=torch.bfloat16, device=device)
        deep_gemm.fp8_gemm_nt(a, w, d)
        out = torch.empty(d.shape[0] // world, HIDDEN, dtype=d.dtype, device=device)
        return sp_collective.reduce_scatter_res(world, d, out, residual, tuning=tuning(d.shape[0]))

    def fused(a, w, residual):
        d = torch.empty(a[0].shape[0], HIDDEN, dtype=torch.bfloat16, device=device)
        deep_gemm.fp8_gemm_nt_push_rs(
            a, w, d, layout.push_workspaces, layout.counter_ptr, layout.stride_bytes, layout.rank
        )
        out = torch.empty(d.shape[0] // world, HIDDEN, dtype=d.dtype, device=device)
        return sp_collective.reduce_scatter_poll(world, d, out, residual, tuning=tuning(d.shape[0]))

    for k in (1536, 2048):
        g = torch.Generator(device=device).manual_seed(7 + k)  # same weight on every rank shape
        w_bf = torch.randn(HIDDEN, k, dtype=torch.bfloat16, device=device, generator=g) * 0.05
        w_q = w_bf.to(torch.float8_e4m3fn)
        # UE8M0 recipe: block scales must be powers of two
        w_s = torch.exp2(
            torch.randint(-3, 4, (HIDDEN // 128, k // 128), device=device, generator=g).float()
        )
        for tokens in (192, 96, 48):
            g = torch.Generator(device=device).manual_seed(1000 * k + 10 * tokens + rank)
            x = torch.randn(tokens, k, dtype=torch.bfloat16, device=device, generator=g)
            x[1] = 0  # zero output rows exercise the push empty-marker rewrite
            a = quant(x)
            res_full = torch.randn(tokens, HIDDEN, dtype=torch.bfloat16, device=device, generator=g)
            res_local = res_full[rank * (tokens // world) : (rank + 1) * (tokens // world)].contiguous()
            for res_name, res in (("none", None), ("local", res_local), ("full", res_full)):
                tag = f"K={k} T={tokens} res={res_name}"
                want = ref(a, (w_q, w_s), res)
                got = fused(a, (w_q, w_s), res)
                torch.cuda.synchronize()
                _check(f"eager {tag}", got, want, rank)
                # Interleaved with plain RS: the shared phase counters stay in step.
                _check(f"plain-after {tag}", ref(a, (w_q, w_s), res), want, rank)

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                fused(a, (w_q, w_s), res_local)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = fused(a, (w_q, w_s), res_local)
            for step in range(3):
                x.copy_(torch.randn_like(x) * (step + 1))
                a[0].copy_(quant(x)[0])
                a[1].copy_(quant(x)[1])
                graph.replay()
                torch.cuda.synchronize()
                _check(f"graph K={k} T={tokens} step={step}", captured, ref(a, (w_q, w_s), res_local), rank)
            dist.barrier()
    if rank == 0:
        print("gemm_rs: all checks passed", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
