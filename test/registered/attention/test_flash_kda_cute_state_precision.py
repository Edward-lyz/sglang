"""CuTe KDA exact-state consistency across packed and ragged sequences."""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10):
    pytest.skip("CuTe KDA requires SM100+", allow_module_level=True)

pytest.importorskip("cutlass")

from sglang.kernels.ops.attention.linear.flash_kda_cute.impl import forward as cf


def _call(sequences, initial, a_log, bias):
    lengths = [sequence["q"].shape[1] for sequence in sequences]
    packed = {
        key: torch.cat([sequence[key] for sequence in sequences], 1).contiguous()
        for key in ["q", "k", "v", "g", "beta"]
    }
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    cu = torch.tensor(offsets, dtype=torch.int64, device="cuda")
    output, final = torch.empty_like(packed["v"]), torch.empty_like(initial)
    cf.fwd(
        packed["q"],
        packed["k"],
        packed["v"],
        packed["g"],
        packed["beta"],
        128**-0.5,
        output,
        a_log,
        bias,
        -5.0,
        initial_state=initial.clone(),
        final_state=final,
        cu_seqlens=cu,
        allow_approximate_split=False,
    )
    plan = cf._plan(cu, 96, sum(lengths), output.device, False, True)
    if plan[11]:
        assert plan[8].dtype == torch.float32
    return output, final


@pytest.mark.parametrize(
    "lengths,splits",
    [([512, 513], [256, 256]), ([64, 128, 257], [32, 64, 128])],
)
@torch.inference_mode()
def test_fp32_state_is_consistent_across_packing_and_splitting(lengths, splits):
    torch.manual_seed(sum(lengths))
    count = len(lengths)
    sequences = [
        {
            key: torch.randn(
                (1, length, 96) if key == "beta" else (1, length, 96, 128),
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 0.25
            for key in ["q", "k", "v", "g", "beta"]
        }
        for length in lengths
    ]
    initial = torch.randn(count, 96, 128, 128, device="cuda") * 0.05
    a_log = torch.randn(96, device="cuda") * 0.2 - 2
    bias = torch.randn(96, 128, device="cuda") * 0.1
    individual = [
        _call([sequence], initial[index : index + 1], a_log, bias)
        for index, sequence in enumerate(sequences)
    ]
    expected_output = torch.cat([pair[0] for pair in individual], 1)
    expected_final = torch.cat([pair[1] for pair in individual], 0)
    output, final = _call(sequences, initial, a_log, bias)
    prefix = [
        {key: value[:, :split] for key, value in sequence.items()}
        for sequence, split in zip(sequences, splits)
    ]
    suffix = [
        {key: value[:, split:] for key, value in sequence.items()}
        for sequence, split in zip(sequences, splits)
    ]
    prefix_output, checkpoint = _call(prefix, initial, a_log, bias)
    suffix_output, split_final = _call(suffix, checkpoint, a_log, bias)
    prefix_pieces = prefix_output.split(splits, 1)
    suffix_pieces = suffix_output.split(
        [length - split for length, split in zip(lengths, splits)],
        1,
    )
    split_output = torch.cat(
        [
            torch.cat([left, right], 1)
            for left, right in zip(prefix_pieces, suffix_pieces)
        ],
        1,
    )
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(final, expected_final, rtol=0, atol=0)
    torch.testing.assert_close(output, split_output, rtol=0, atol=0)
    torch.testing.assert_close(final, split_final, rtol=0, atol=0)
