"""Benchmark the fused DCP a2a send-side pack against the original staging path.

The collective itself is not benchmarked: with one process it is replaced by a
local byte copy, identical for both paths, so the delta is exactly the staging +
unpack + combine work that ``SGLANG_OPT_USE_FUSED_DCP_A2A_PACK`` replaces.

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python3 \
        benchmark/kernels/bench_dcp_fused_pack.py
    ... --shapes 800x96x512x8 --graph
"""

import argparse
from unittest.mock import MagicMock

import torch
import triton

from sglang.srt.environ import envs
from sglang.srt.layers.dcp.comm import dcp_a2a_lse_reduce

DEFAULT_SHAPES = (
    # B x H x D x dcp_size; the first is Kimi-K3 decode at cuda-graph-bs 200
    # with 4 spec tokens per request.
    "800x96x512x8",
    "64x96x512x8",
    "416x96x512x8",
)


def loopback_group(world_size: int):
    """Mock a NCCL group for a loopback test."""
    group = MagicMock()
    group.world_size = world_size

    def all_to_all_single(output, input_):
        output.copy_(input_)

    group.all_to_all_single = all_to_all_single
    return group


def bench_one(B, H, D, N, dtype, use_graph):
    """Benchmark a single shape."""
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(0)
    out = torch.randn(
        B, H, D, device=device, dtype=torch.float32, generator=generator
    ).to(dtype)
    lse = (
        torch.randn(B, H, device=device, dtype=torch.float32, generator=generator) * 5.0
    )
    group = loopback_group(N)

    results = {}
    for label, fused in (("original", False), ("fused", True)):
        with envs.SGLANG_OPT_USE_FUSED_DCP_A2A_PACK.override(fused):

            def run():
                dcp_a2a_lse_reduce(out, lse, group, is_lse_base_on_e=False)

            for _ in range(10):
                run()
            torch.cuda.synchronize()
            if use_graph:
                # Replay is the production scope: decode runs inside a cuda graph.
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3):
                        run()
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(20):
                        run()
                torch.cuda.synchronize()
                ms = triton.testing.do_bench(graph.replay, warmup=25, rep=100)
                results[label] = ms * 1000.0 / 20
            else:
                ms = triton.testing.do_bench(run, warmup=25, rep=100)
                results[label] = ms * 1000.0
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", nargs="*", default=list(DEFAULT_SHAPES))
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    parser.add_argument(
        "--graph",
        action="store_true",
        help="measure cuda-graph replay instead of eager launches",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    dtype = getattr(torch, args.dtype)

    print(f"{'shape (BxHxDxN)':<24}{'original us':>14}{'fused us':>12}{'speedup':>10}")
    for shape in args.shapes:
        B, H, D, N = (int(value) for value in shape.split("x"))
        result = bench_one(B, H, D, N, dtype, args.graph)
        speedup = result["original"] / result["fused"]
        print(
            f"{shape:<24}{result['original']:>14.3f}{result['fused']:>12.3f}"
            f"{speedup:>9.3f}x"
        )


if __name__ == "__main__":
    main()
