"""Benchmark the fused Kimi-K3 KDA short_conv against the Triton path.

`triton` is the production baseline this kernel replaces: three serial
`causal_conv1d_fn` calls over the q/k/v slices of mixed_qkv. Both providers do
the same total work (3 * num_tokens * 12288 output elements plus the conv-state
update), so the ratio between the two lines is the end-to-end speedup of the
env-gated replacement (SGLANG_K3_KDA_SHORT_CONV).
"""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.kimi_k3.kda_short_conv import (
    SUPPORTED_WIDTH,
    kimi_k3_kda_short_conv,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import PAD_SLOT_ID, causal_conv1d_fn
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=8,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
)

# attn_tp_size=1 (num_heads=96 * head_dim=128); attn_tp 2/4/8 give
# 6144/3072/1536 and are exercised by the unit test, not here.
_DIM = 12288
_TOTAL_DIM = 3 * _DIM
_WIDTH = SUPPORTED_WIDTH
_NUM_LINES = 4


def _make_inputs(num_tokens: int, num_seqs: int, layout: str = "strided"):
    """`layout="strided"` is the PRODUCTION layout: mixed_qkv is a (T, 3*dim)
    view of the (T, 4*dim) fused q/k/v/g projection output, so its row pitch is
    4*dim and it is not contiguous. The buffer is passed whole (and the view
    re-derived inside the providers) so the benchmark's per-iteration clone
    rotation cannot silently turn it into a contiguous tensor.
    """
    lengths = [num_tokens // num_seqs] * num_seqs
    lengths[-1] += num_tokens - sum(lengths)
    row = _TOTAL_DIM + (_DIM if layout == "strided" else 0)
    x_buffer = torch.randn(num_tokens, row, device="cuda").bfloat16()
    conv_weights = torch.randn(_TOTAL_DIM, _WIDTH, device="cuda", dtype=torch.float32)
    pool = torch.randn(_NUM_LINES, _WIDTH - 1, _TOTAL_DIM, device="cuda").bfloat16()
    query_start_loc = torch.zeros(num_seqs + 1, dtype=torch.int32, device="cuda")
    query_start_loc[1:] = torch.tensor(
        lengths, dtype=torch.int32, device="cuda"
    ).cumsum(0)
    cache_indices = torch.arange(num_seqs, dtype=torch.int32, device="cuda") % _NUM_LINES
    has_initial_state = torch.ones(num_seqs, dtype=torch.bool, device="cuda")
    return (
        x_buffer,
        conv_weights,
        pool,
        query_start_loc,
        cache_indices,
        has_initial_state,
        lengths,
    )


def _run_jit(x_buffer, conv_weights, pool, qsl, ci, his, lengths):
    mixed_qkv = x_buffer[:, :_TOTAL_DIM]
    kimi_k3_kda_short_conv(
        mixed_qkv, conv_weights, pool, qsl, ci, his, lengths, pad_slot_id=PAD_SLOT_ID
    )


def _run_triton(x_buffer, conv_weights, pool, qsl, ci, his, lengths):
    mixed_qkv = x_buffer[:, :_TOTAL_DIM]
    splits = [_DIM] * 3
    conv_states = pool.transpose(-1, -2)
    for x, w, st in zip(
        mixed_qkv.transpose(0, 1).split(splits, dim=0),
        conv_weights.split(splits, dim=0),
        conv_states.split(splits, dim=-2),
    ):
        causal_conv1d_fn(
            x,
            w,
            None,
            activation="silu",
            conv_states=st,
            has_initial_state=his,
            cache_indices=ci,
            query_start_loc=qsl,
            seq_lens_cpu=lengths,
            pad_slot_id=PAD_SLOT_ID,
        )


@marker.parametrize("layout", ["strided", "contiguous"], ["strided"])
@marker.parametrize("num_seqs", [1, 4], [1])
@marker.parametrize("num_tokens", [1024, 4096, 8192, 16384], [4096])
@marker.benchmark("impl", ["jit", "triton"])
def benchmark(num_tokens: int, num_seqs: int, layout: str, impl: str):
    if impl == "jit" and torch.cuda.get_device_capability()[0] < 10:
        marker.skip("kimi_k3_kda_short_conv requires SM100+")
    if num_seqs > num_tokens:
        marker.skip("need at least one token per sequence")

    args = _make_inputs(num_tokens, num_seqs, layout)
    fn = _run_jit if impl == "jit" else _run_triton
    return marker.do_bench(
        fn,
        input_args=args,
        # only the x buffer is a pure read arg; the pool is updated in place
        graph_clone_args=(0,),
        # bandwidth is charged for the 3*dim columns actually read, not the
        # padding, so the two layouts stay comparable
        memory_args=(args[0][:, :_TOTAL_DIM], args[1]),
        memory_output=None,
    )


if __name__ == "__main__":
    benchmark.run()
