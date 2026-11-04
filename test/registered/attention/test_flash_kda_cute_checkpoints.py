"""Packed CuTe checkpoints through production dispatch and physical scatter."""

# ruff: noqa: E402 - GPU/dependency gates precede kernel imports.

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10):
    pytest.skip("CuTe KDA requires SM100+", allow_module_level=True)
pytest.importorskip("cutlass")

from sglang.kernels.ops.attention.linear import flash_kda_cute
from sglang.srt.environ import envs
from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend
from sglang.srt.layers.attention.linear.kernels import kda_flashkda
from sglang.srt.layers.attention.linear.kernels.kda_native_checkpoint import (
    KDANativeCheckpointResult,
    KDANativePlanSink,
    fill_native_plan_from_existing_cpu_data,
)


@pytest.mark.parametrize(
    "lengths,targets",
    [
        ([512, 513], [256, 256]),
        ([64, 128, 257], [32, -1, 128]),
        ([65, 64, 129], [32, 64, -1]),
    ],
)
@torch.inference_mode()
def test_packed_checkpoint_dispatch_and_physical_scatter(lengths, targets):
    torch.manual_seed(sum(lengths))
    n, total = len(lengths), sum(lengths)
    inputs = {
        key: torch.randn(1, total, 96, 128, dtype=torch.bfloat16, device="cuda") * 0.25
        for key in ("q", "k", "v", "g")
    }
    beta = torch.rand(1, total, 96, device="cuda").mul(0.8).add(0.1).bfloat16()
    a_log = torch.randn(96, device="cuda") * 0.2 - 2
    bias = torch.randn(96, 128, device="cuda") * 0.1
    pool = torch.randn(20, 96, 128, 128, device="cuda") * 0.05
    before = pool.clone()
    indices = torch.tensor([5, 1, 7][:n], device="cuda", dtype=torch.int32)
    cu_cpu = torch.tensor([0] + lengths).cumsum(0)
    cu = cu_cpu.to(device="cuda", dtype=torch.int32)
    lens_cpu, targets_cpu = torch.tensor(lengths), torch.tensor(targets)
    mask = targets_cpu > 0
    final_rows = torch.where(targets_cpu == lens_cpu)[0]
    interior_rows = torch.where(mask & (targets_cpu < lens_cpu))[0]
    h_dst = torch.arange(10, 10 + len(interior_rows), device="cuda")
    final_dst = torch.arange(15, 15 + len(final_rows), device="cuda")
    sink = KDANativePlanSink(build_cute_plan=True)
    fill_native_plan_from_existing_cpu_data(
        sink,
        exact_checkpoint_tracking=True,
        prefixes=torch.zeros(n, dtype=torch.int64),
        extends=lens_cpu,
        targets=targets_cpu,
        track_mask=mask,
        is_final_masked=(targets_cpu == lens_cpu)[mask],
        h_dst=h_dst.cpu(),
        device="cuda",
    )
    common = dict(
        **inputs,
        beta=beta,
        cache_indices=indices,
        query_start_loc=cu,
        A_log=a_log,
        dt_bias=bias,
        lower_bound=-5.0,
        extend_seq_lens_cpu=lengths,
    )
    with (
        envs.SGLANG_OPT_FLASH_KDA_CUTE.override(True),
        patch.object(
            kda_flashkda,
            "probe_pinned_style_native",
            side_effect=AssertionError("CuTe must not probe an external ABI"),
        ),
    ):
        kernel = kda_flashkda.FlashKDAKernel()
        assert kernel.native_checkpoint_capability == "cute_prefix_suffix_fp32_c32"
        reference_pool = before.clone()
        continuous = kernel.extend(**common, ssm_states=reference_pool)
        result = kernel.extend(
            **common,
            ssm_states=pool,
            return_intermediate_states=True,
            track_ssm_h_src=torch.arange(len(interior_rows), device="cuda"),
            native_checkpoint_offsets=sink.offsets,
            native_checkpoint_rows=sink.owner_rows,
            cute_checkpoint_plan=sink.cute_plan,
        )
    assert isinstance(result, KDANativeCheckpointResult)
    torch.testing.assert_close(result.output, continuous, rtol=0, atol=0)
    torch.testing.assert_close(pool[indices], reference_pool[indices], rtol=0, atol=0)
    logits = torch.logit(beta.float().clamp(1e-7, 1 - 1e-7)).bfloat16()
    for row in interior_rows.tolist():
        lo, hi = int(cu_cpu[row]), int(cu_cpu[row]) + targets[row]
        output = torch.empty_like(inputs["v"][:, lo:hi])
        state = torch.empty_like(before[indices[row : row + 1]])
        flash_kda_cute.fwd(
            *(inputs[key][:, lo:hi].contiguous() for key in ("q", "k", "v", "g")),
            logits[:, lo:hi].contiguous(),
            128**-0.5,
            output,
            a_log,
            bias,
            -5.0,
            initial_state=before[indices[row : row + 1]],
            final_state=state,
            cu_seqlens=torch.tensor([0, hi - lo], device="cuda", dtype=torch.int64),
        )
        torch.testing.assert_close(
            result.checkpoint_states[row : row + 1], state, rtol=0, atol=0
        )
    metadata = SimpleNamespace(
        kda_native_checkpoint_rows=sink.owner_rows,
        track_ssm_h_dst=h_dst,
        track_ssm_final_src=indices[final_rows.to("cuda")],
        track_ssm_final_dst=final_dst,
    )
    returned = KDAAttnBackend._scatter_native_checkpoint_result(
        result,
        metadata=metadata,
        ssm_states=pool,
        cache_indices=indices,
    )
    assert returned is result.output
    expected = before.clone()
    expected[indices] = reference_pool[indices]
    expected[h_dst] = result.checkpoint_states[sink.owner_rows]
    expected[final_dst] = reference_pool[metadata.track_ssm_final_src]
    torch.testing.assert_close(pool, expected, rtol=0, atol=0)


@pytest.mark.parametrize("offset,is_spec_decode", [(16, False), (32, True)])
@torch.inference_mode()
def test_unsupported_plan_or_speculation_falls_back_before_kda_writes(
    offset, is_spec_decode
):
    q = torch.zeros(1, 65, 96, 128, dtype=torch.bfloat16, device="cuda")
    pool = torch.ones(1, 96, 128, 128, device="cuda")
    sink = KDANativePlanSink(build_cute_plan=True)
    fill_native_plan_from_existing_cpu_data(
        sink,
        exact_checkpoint_tracking=True,
        prefixes=torch.tensor([0]),
        extends=torch.tensor([65]),
        targets=torch.tensor([offset]),
        track_mask=torch.tensor([True]),
        is_final_masked=torch.tensor([False]),
        h_dst=torch.tensor([9]),
        device="cuda",
    )
    sentinel = object()
    with envs.SGLANG_OPT_FLASH_KDA_CUTE.override(True):
        kernel = kda_flashkda.FlashKDAKernel()
        with (
            patch.object(
                kda_flashkda, "_triton_fallback", return_value=sentinel
            ) as fallback,
            patch.object(
                kernel, "_flashkda_extend", side_effect=AssertionError("invalid launch")
            ),
        ):
            result = kernel.extend(
                q,
                q,
                q,
                q,
                torch.zeros(1, 65, 96, device="cuda"),
                ssm_states=pool,
                cache_indices=torch.tensor([0], dtype=torch.int32, device="cuda"),
                query_start_loc=torch.tensor([0, 65], dtype=torch.int32, device="cuda"),
                A_log=torch.zeros(96, device="cuda"),
                dt_bias=torch.zeros(96, 128, device="cuda"),
                lower_bound=-5.0,
                extend_seq_lens_cpu=[65],
                return_intermediate_states=True,
                track_ssm_h_src=torch.tensor([0], device="cuda"),
                native_checkpoint_offsets=sink.offsets,
                native_checkpoint_rows=sink.owner_rows,
                cute_checkpoint_plan=sink.cute_plan,
                is_spec_decode=is_spec_decode,
            )
    assert result is sentinel and fallback.call_count == 1
    assert torch.all(pool == 1)
