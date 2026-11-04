"""CuTe snapshots must preserve exact CPU owner/position instructions."""

import pytest
import torch

from sglang.srt.layers.attention.linear.kernels.kda_cute_checkpoint import (
    CuTeCheckpointAdapter,
    build_cute_checkpoint_plan,
)
from sglang.srt.layers.attention.linear.kernels.kda_native_checkpoint import (
    KDANativePlanSink,
    fill_native_plan_from_existing_cpu_data,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_exact_plan_preserves_inactive_and_final_only_owners():
    sink = KDANativePlanSink(build_cute_plan=True)
    fill_native_plan_from_existing_cpu_data(
        sink,
        exact_checkpoint_tracking=True,
        prefixes=torch.tensor([0, 100, 0, 32]),
        extends=torch.tensor([65, 49, 64, 97]),
        targets=torch.tensor([32, -1, 64, 96]),
        track_mask=torch.tensor([True, False, True, True]),
        is_final_masked=torch.tensor([False, True, False]),
        h_dst=torch.tensor([10, 11]),
        device="cpu",
    )
    plan = sink.cute_plan
    assert plan.lengths == (65, 49, 64, 97)
    assert plan.offsets == (32, 0, 0, 64)
    assert plan.active_rows.tolist() == sink.owner_rows.tolist() == [0, 3]
    assert plan.prefix_cu.tolist() == [0, 32, 81, 145, 209]
    assert plan.suffix_cu.tolist() == [0, 33, 66]
    assert sorted(plan.prefix_tokens.tolist() + plan.suffix_tokens.tolist()) == list(
        range(275)
    )
    assert plan.suffix_tokens.tolist() == list(range(32, 65)) + list(range(242, 275))


@pytest.mark.parametrize("offset", [16, -32, 96])
def test_unsupported_offset_never_becomes_a_cute_launch(offset):
    assert (
        build_cute_checkpoint_plan(
            torch.tensor([96]), torch.tensor([offset]), device="cpu"
        )
        is None
    )


def test_c16_exact_plan_keeps_external_metadata_and_has_no_cute_plan():
    sink = KDANativePlanSink(build_cute_plan=True)
    fill_native_plan_from_existing_cpu_data(
        sink,
        exact_checkpoint_tracking=True,
        prefixes=torch.tensor([0]),
        extends=torch.tensor([65]),
        targets=torch.tensor([16]),
        track_mask=torch.tensor([True]),
        is_final_masked=torch.tensor([False]),
        h_dst=torch.tensor([9]),
        device="cpu",
    )
    assert sink.offsets.tolist() == [16]
    assert sink.owner_rows.tolist() == [0]
    assert sink.cute_plan is None


def test_adapter_rejects_bad_state_before_any_forward_or_write():
    calls = []
    adapter = CuTeCheckpointAdapter(lambda *a, **kw: calls.append(True))
    plan = build_cute_checkpoint_plan(
        torch.tensor([65]), torch.tensor([32]), device="cpu"
    )
    q = torch.zeros(1, 65, 96, 128, dtype=torch.bfloat16)
    output = torch.full_like(q, 7)
    state = torch.ones(1, 96, 128, 128)
    final, checkpoint = state.clone(), state.clone()
    with pytest.raises(ValueError, match="dense FP32"):
        adapter.invoke(
            q=q,
            k=q,
            v=q,
            g=q,
            beta=torch.zeros(1, 65, 96, dtype=torch.bfloat16),
            scale=128**-0.5,
            out=output,
            A_log=torch.zeros(96),
            dt_bias=torch.zeros(96, 128),
            lower_bound=-5.0,
            initial_state=state.bfloat16(),
            final_state=final,
            checkpoint_state=checkpoint,
            plan=plan,
        )
    assert calls == []
    assert torch.all(output == 7) and torch.equal(final, state)
    assert torch.equal(checkpoint, state)
