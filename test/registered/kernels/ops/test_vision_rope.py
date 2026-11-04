import pytest
import torch

from sglang.kernels.ops.attention.vision_rope import apply_fused_qk_complex_rope
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

# Interleaved-QKV view stride(0) = 3 * num_heads * head_dim. The kernel indexes
# q_base = token * q_stride_token (+ ...); computed in int32 it overflows once
# token * stride0 exceeds 2**31, producing a wild pointer -> CUDA illegal memory
# access (observed on long K3 videos whose single vision forward exceeds this).
_QKV_STRIDE0 = 3 * 12 * 128  # 4608
_INT32_MAX = 2**31 - 1
# Smallest token count whose max element offset (max_token_idx * stride0) exceeds
# int32: (tokens - 1) * 4608 > INT32_MAX  ->  tokens >= 466035.
_INT32_OVERFLOW_TOKENS = _INT32_MAX // _QKV_STRIDE0 + 2


def native_complex_rope(q, k, freqs_cis):
    freqs_cis = freqs_cis.unsqueeze(-2)
    q_complex = torch.view_as_complex(q.float().view(*q.shape[:-1], -1, 2))
    k_complex = torch.view_as_complex(k.float().view(*k.shape[:-1], -1, 2))
    q_out = torch.view_as_real(q_complex * freqs_cis).flatten(-2)
    k_out = torch.view_as_real(k_complex * freqs_cis).flatten(-2)
    return q_out.type_as(q), k_out.type_as(k)


@pytest.mark.parametrize("tokens", [1, 480, 5660, 8360])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_qk_complex_rope_matches_native(tokens, dtype):
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("The fused vision RoPE path targets Hopper or newer NVIDIA GPUs")

    qkv = torch.randn(tokens, 3, 12, 128, dtype=dtype, device="cuda")
    q, k, _ = torch.unbind(qkv, dim=1)
    angles = torch.randn(tokens, 64, dtype=torch.float32, device="cuda")
    freqs_cis = torch.polar(torch.ones_like(angles), angles)

    expected_q, expected_k = native_complex_rope(q, k, freqs_cis)
    actual_q, actual_k = apply_fused_qk_complex_rope(q, k, freqs_cis)

    assert torch.equal(actual_q, expected_q)
    assert torch.equal(actual_k, expected_k)
    assert actual_q.is_contiguous()
    assert actual_k.is_contiguous()


def test_fused_qk_complex_rope_cuda_graph_replay():
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("The fused vision RoPE path targets Hopper or newer NVIDIA GPUs")

    qkv = torch.randn(480, 3, 12, 128, dtype=torch.bfloat16, device="cuda")
    q, k, _ = torch.unbind(qkv, dim=1)
    angles = torch.randn(480, 64, dtype=torch.float32, device="cuda")
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    expected_q, expected_k = native_complex_rope(q, k, freqs_cis)

    apply_fused_qk_complex_rope(q, k, freqs_cis)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual_q, actual_k = apply_fused_qk_complex_rope(q, k, freqs_cis)
    graph.replay()

    assert torch.equal(actual_q, expected_q)
    assert torch.equal(actual_k, expected_k)


def test_fused_qk_complex_rope_int32_offset_overflow():
    """Regression: a single vision forward large enough that token*stride0 exceeds
    int32 must NOT trigger a CUDA illegal memory access. Before widening the kernel
    index arithmetic to int64 this crashed the whole encoder (long K3 videos)."""
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("The fused vision RoPE path targets Hopper or newer NVIDIA GPUs")
    # qkv(bf16) + contiguous q_out/k_out for this token count need ~8GB; skip on
    # GPUs too small to hold it.
    if torch.cuda.get_device_properties(0).total_memory < 24 * 1024**3:
        pytest.skip("needs >=24GB GPU memory for the int32-overflow token count")

    tokens = _INT32_OVERFLOW_TOKENS
    # Sanity: this token count really does overflow int32 addressing.
    assert (tokens - 1) * _QKV_STRIDE0 > _INT32_MAX

    qkv = torch.randn(tokens, 3, 12, 128, dtype=torch.bfloat16, device="cuda")
    q, k, _ = torch.unbind(qkv, dim=1)
    angles = torch.randn(tokens, 64, dtype=torch.float32, device="cuda")
    freqs_cis = torch.polar(torch.ones_like(angles), angles)

    # Would raise/abort on the pre-fix int32 kernel; must complete and stay correct.
    actual_q, actual_k = apply_fused_qk_complex_rope(q, k, freqs_cis)
    expected_q, expected_k = native_complex_rope(q, k, freqs_cis)

    # Correctness at this scale is checked with a tolerance, not torch.equal: the
    # fused kernel contracts `a*cos - b*sin` into an FMA (one rounding) while the
    # native complex multiply rounds twice, so a handful of elements differ by ~1
    # bf16 ULP. That is benign and unrelated to the fix. What an int32 addressing
    # bug WOULD produce is entirely wrong rows (huge errors), so a tight ULP-level
    # bound both proves the addressing is correct and tolerates FMA rounding.
    tol = 2.0**-5  # ~2 bf16 ULP around unit magnitude
    assert (actual_q.float() - expected_q.float()).abs().max().item() <= tol
    assert (actual_k.float() - expected_k.float()).abs().max().item() <= tol
    assert actual_q.is_contiguous()
    assert actual_k.is_contiguous()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
