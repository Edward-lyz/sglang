import pytest
import torch

from sglang.kernels.ops.quantization.mla_qkv_fp8_cast import (
    quantize_mla_qkv_fp8_profitable,
    quantize_mla_qkv_fp8_supported,
    quantize_mla_qkv_fp8_triton,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=12, stage="base-b-kernel-unit", runner_config="1-gpu-large")

# PyTorch's Tensor.to(float8_e4m3fn) is not the hardware saturating conversion:
# |x| > 464 and every inf/nan become 0x7f with the sign preserved, while
# cvt.rn.satfinite.e4m3 clamps to 448. These values keep that difference in
# every case, so a kernel that relied on the hardware instruction alone fails.
_EXTREMES = (
    0.0,
    -0.0,
    448.0,
    -448.0,
    464.0,
    -464.0,
    464.5,
    -464.5,
    480.0,
    -480.0,
    1.0e4,
    -1.0e4,
    float("inf"),
    float("-inf"),
    float("nan"),
    -float("nan"),
    2.0**-6,
    -(2.0**-6),
    2.0**-9,
    -(2.0**-9),
    2.0**-10,
    1.0,
    -1.0,
    0.5,
)

QK_HEAD_DIM = 192
NOPE_DIM = 128
KV_LAST_DIM = 256


def _inject_extremes(tensor: torch.Tensor, rows: int = 4) -> None:
    flat = tensor.view(tensor.shape[0], -1)
    rows = min(rows, flat.shape[0])
    pattern = torch.tensor(_EXTREMES, dtype=torch.float32, device=tensor.device).to(
        tensor.dtype
    )
    width = flat.shape[1]
    repeats = (width + pattern.numel() - 1) // pattern.numel()
    row = pattern.repeat(repeats)[:width]
    for index in range(rows):
        flat[index] = torch.roll(row, shifts=index)


def _make_inputs(tokens: int, heads: int, seed: int):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    q = torch.randn(
        (tokens, heads, QK_HEAD_DIM),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        (tokens, heads, QK_HEAD_DIM),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    # v must stay a trailing slice of a contiguous kv tensor: that non-unit head
    # stride is what the row kernel exists for.
    kv = torch.randn(
        (tokens, heads, KV_LAST_DIM),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    for tensor in (q, k, kv):
        _inject_extremes(tensor)
    return q, k, kv[..., NOPE_DIM:]


def _assert_bit_identical(actual, expected, label):
    assert actual.dtype == expected.dtype, label
    assert actual.shape == expected.shape, label
    mismatched = int((actual.view(torch.uint8) != expected.view(torch.uint8)).sum())
    assert mismatched == 0, f"{label}: {mismatched} bytes differ"


@pytest.mark.parametrize("tokens,heads", [(4097, 12), (1024, 12), (129, 96)])
@pytest.mark.parametrize("seed", [101, 102])
def test_bit_exact_against_torch(tokens, heads, seed):
    q, k, v = _make_inputs(tokens, heads, seed)
    assert quantize_mla_qkv_fp8_supported(q, k, v)

    expected = tuple(t.to(torch.float8_e4m3fn) for t in (q, k, v))
    # force_kernel bypasses the size gate so the Triton path is exercised even
    # on the small shapes CI can afford.
    for force_kernel in (False, True):
        actual = quantize_mla_qkv_fp8_triton(q, k, v, force_kernel=force_kernel)
        for name, got, want in zip("qkv", actual, expected):
            _assert_bit_identical(got, want, f"{name} force_kernel={force_kernel}")


def test_size_gate_routes_small_shapes_to_torch():
    q, k, v = _make_inputs(1024, 12, 101)
    assert not quantize_mla_qkv_fp8_profitable(q, k, v)
    # Above the threshold the kernel path must be selected; 32768 x 12 heads is
    # the profiled prefill shape.
    big_q, big_k, big_v = _make_inputs(32768, 12, 101)
    assert quantize_mla_qkv_fp8_profitable(big_q, big_k, big_v)


def test_unsupported_inputs_are_rejected():
    q, k, v = _make_inputs(1024, 12, 101)

    assert not quantize_mla_qkv_fp8_supported(q.float(), k, v)
    assert not quantize_mla_qkv_fp8_supported(q.cpu(), k.cpu(), v.cpu())
    assert not quantize_mla_qkv_fp8_supported(q[:, :, ::2], k, v)
    # A non-unit stride on v's last dimension has no coalesced row to load.
    assert not quantize_mla_qkv_fp8_supported(q, k, v[:, :, ::2])
    # An inner dim that is not a power of two cannot bound tl.arange.
    assert not quantize_mla_qkv_fp8_supported(q, k, v[:, :, :120])
    assert not quantize_mla_qkv_fp8_supported(q[:0], k[:0], v[:0])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
