# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Mega-MoE forward path and expert-weight prep shared by Deepseek V2/V4."""

from __future__ import annotations

import functools
import logging
import os
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.attention.dsv4 import mega_moe_pre_dispatch
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp
from sglang.srt.layers.dp_attention import get_dp_global_num_tokens
from sglang.srt.layers.moe.mega_moe_sm90 import (
    is_sm90_fp8_mega_moe_available,
    run_sm90_mega_routed,
)
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.models.deepseek_common.utils import _device_sm
from sglang.srt.runtime_context import get_exec, get_parallel

if TYPE_CHECKING:
    from deep_gemm import SymmBuffer

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE


logger = logging.getLogger(__name__)

_MEGA_MOE_SYMM_BUFFER: dict = {}
_MEGA_GATE_RUNTIME_DISABLED = False
_MEGA_GATE_LOGGED: set = set()
_MEGA_GATE_CAPTURED_LOGGED = False
_MEGA_GATE_CAPTURE_FALLBACK_WARNED = False

# DeepGEMM raises this host-side (before any stream work) when it would have to
# lazily initialise its per-stream workspace while the stream is capturing.
_CAPTURE_ASSERT_MARK = "CaptureStatus"


def _mega_moe_mma_type() -> str:
    return "mxf4xmxf4" if get_exec().moe.enable_w4a4_mxfp4_megamoe else "fp8xfp4"


@functools.lru_cache(maxsize=1)
def _mega_moe_max_num_sms() -> Optional[int]:
    if _device_sm < 100:
        # The SM90 MegaMoE implementation does not use the whole-grid clustered
        # launch that needs a residency margin.
        return None

    # Physical count, not deep_gemm.get_num_sms(): two-batch overlap and the DSA
    # indexer reconfigure that process-wide, so reserving on top would compound.
    num_sms = torch.cuda.get_device_properties(device="cuda").multi_processor_count
    reserved_num_sms = max(envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_RESERVED_SMS.get(), 0)
    return max(2, num_sms - reserved_num_sms)


@contextmanager
def _configure_mega_moe_deep_gemm_num_sms(deep_gemm):
    max_num_sms = _mega_moe_max_num_sms()
    if max_num_sms is None:
        yield
        return

    current_num_sms = deep_gemm.get_num_sms()
    # Stay under an outer context's budget instead of claiming SMs back from it.
    target_num_sms = min(max_num_sms, current_num_sms)
    # Round down: the clustered launch needs an even CTA count.
    target_num_sms -= target_num_sms % 2
    if target_num_sms == current_num_sms:
        yield
        return

    deep_gemm.set_num_sms(target_num_sms)
    try:
        yield
    finally:
        deep_gemm.set_num_sms(current_num_sms)


def _get_mega_moe_symm_buffer(
    group,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    num_shared_experts: int = 0,
) -> SymmBuffer:
    import deep_gemm

    mma_type = _mega_moe_mma_type()
    with _configure_mega_moe_deep_gemm_num_sms(deep_gemm):
        key = (
            id(group),
            num_max_tokens_per_rank,
            num_experts,
            num_topk,
            hidden,
            intermediate_hidden,
            mma_type,
            num_shared_experts,
            deep_gemm.get_num_sms(),
        )
        buf = _MEGA_MOE_SYMM_BUFFER.get(key)
        if buf is None:
            from sglang.srt.utils.community_opt import log_community_opt_once

            log_community_opt_once(
                "#39223 mega_moe_sm_budget_buffer",
                "MegaMoE symm buffer sized under the kernel's SM budget (num_sms=%d)",
                deep_gemm.get_num_sms(),
            )
            buf = deep_gemm.get_symm_buffer_for_mega_moe(
                group,
                num_experts,
                num_max_tokens_per_rank,
                num_topk,
                hidden,
                intermediate_hidden,
                num_shared_experts=num_shared_experts,
                mma_type=mma_type,
                activation="swiglu",
            )
            _MEGA_MOE_SYMM_BUFFER[key] = buf
    return buf


def should_use_mega_moe(moe: DeepseekV2MoE, hidden_states: torch.Tensor) -> bool:
    if not get_moe_a2a_backend().is_megamoe():
        return False
    if not getattr(moe.experts, "_mega_moe_weights_built", False):
        return False
    if _device_sm == 90:
        if not is_sm90_fp8_mega_moe_available(moe.experts):
            return False
    if get_is_capture_mode():
        return True

    global_num_tokens = get_dp_global_num_tokens()
    if global_num_tokens and not is_dsa_enable_prefill_cp():
        max_tokens_per_rank = max(global_num_tokens)
    else:
        max_tokens_per_rank = hidden_states.shape[0]
    cap = envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    return max_tokens_per_rank <= cap


def should_fuse_mega_moe_shared_experts(moe: DeepseekV2MoE) -> bool:
    """Fusing shared experts into DeepGEMM MegaMoE."""
    # DeepGEMM's in-kernel shared expert takes FP8 weights on SM100 fp8xfp4 only.
    # Upstream (#39313) admits [128, 128] blocks; [32, 32] ue8m0 blocks transform
    # to the same per-row 1x32 scale layout (see _transform_mega_moe_shared_sf).
    shared = getattr(moe, "shared_experts", None)
    return (
        envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_FUSE_SHARED_EXPERTS.get()
        and shared is not None
        and moe.num_fused_shared_experts == 0
        # The kernel adds the whole shared expert per token: it must be replicated.
        and getattr(moe, "_shared_expert_tp1", False)
        and get_moe_a2a_backend().is_megamoe()
        and getattr(moe.experts, "_mega_moe_weights_built", False)
        and _device_sm == 100
        and _mega_moe_mma_type() == "fp8xfp4"
        and moe.shared_experts_is_fp8
        and moe.shared_experts_weight_block_size in ([128, 128], [32, 32])
        and not getattr(shared.gate_up_proj, "aiter_bpreshuffled", False)
        and not getattr(shared.down_proj, "aiter_bpreshuffled", False)
    )


def forward_mega_moe(
    moe: DeepseekV2MoE,
    hidden_states: torch.Tensor,
    forward_batch: Optional[ForwardBatch] = None,
    input_ids_global: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]
    fused_shared = moe.mega_shared_l1_weights is not None
    if fused_shared:
        from sglang.srt.utils.community_opt import log_community_opt_once

        log_community_opt_once(
            "#39313 mega_moe_fused_shared",
            "shared expert fused into DeepGEMM MegaMoE (block=%s, tokens=%d)",
            moe.shared_experts_weight_block_size,
            num_tokens,
        )
        if (
            envs.SGLANG_COMMUNITY_SELFCHECK_MEGA_SHARED.get()
            and not get_is_capture_mode()
            and num_tokens >= 256
            and not getattr(forward_mega_moe, "_selfcheck_done", False)
        ):
            forward_mega_moe._selfcheck_done = True
            _selfcheck_fused_shared(
                moe, hidden_states, forward_batch, input_ids_global, num_tokens
            )

    sbo_overlap_flag = (
        moe.alt_stream is not None
        and (fused_shared or moe.num_fused_shared_experts == 0)
        and num_tokens > 0
        and get_is_capture_mode()
    )

    if sbo_overlap_flag:
        current_stream = torch.cuda.current_stream()
        moe.alt_stream.wait_stream(current_stream)
        mega_stream_ctx = torch.cuda.stream(moe.alt_stream)
    else:
        mega_stream_ctx = nullcontext()
    # The fused kernel adds the shared expert itself.
    shared_output = None if fused_shared else moe._forward_shared_experts(hidden_states)

    # Optionally fuse the routed scaling into the shared-expert residual add:
    # one torch.add(shared, y, alpha=scale) instead of y.mul_(scale) then
    # y.add_(shared). Only on the sm100 path (sm90 self-scales inside
    # run_sm90_mega_routed) and only when the scaling is not already folded into
    # topk and a shared output exists. Default OFF (SGLANG_FUSE_SHARED_SCALE_ADD).
    fuse_scale_add = (
        envs.SGLANG_FUSE_SHARED_SCALE_ADD.get()
        and shared_output is not None
        and _device_sm != 90
        and not moe.experts.should_fuse_routed_scaling_factor_in_topk
    )

    with mega_stream_ctx:
        y = _run_mega_routed(
            moe,
            hidden_states,
            forward_batch,
            input_ids_global,
            num_tokens,
            apply_routed_scaling=not fuse_scale_add,
            fuse_shared=fused_shared,
        )

    if sbo_overlap_flag:
        current_stream.wait_stream(moe.alt_stream)

    if fuse_scale_add and (
        y.shape == shared_output.shape
        and y.dtype == shared_output.dtype
        and y.is_contiguous()
        and shared_output.is_contiguous()
    ):
        # Single fused elementwise kernel: y = shared_output + scale * y.
        torch.add(shared_output, y, alpha=moe.routed_scaling_factor, out=y)
    elif shared_output is not None:
        if fuse_scale_add:
            # Flag on but the fused guard rejected these tensors: _run_mega_routed
            # skipped the scaling, so restore it before the plain add (fallback).
            y.mul_(moe.routed_scaling_factor)
        y.add_(shared_output)
    return y


_IMAGE_MASK_CACHE: Optional[tuple] = None


def _image_token_mask(
    input_ids: torch.Tensor,
    image_token_id: int,
    layer_id: int,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """``input_ids == image_token_id``. With SGLANG_OPT_MEGA_GATE_IMAGE_MASK_ONCE the
    compare runs once per forward instead of once per MoE layer: a new forward starts
    when the layer id stops increasing (or the id tensor / hidden buffer changes), which
    also keeps the result from outliving a CUDA-graph capture's tensors."""
    global _IMAGE_MASK_CACHE
    if not envs.SGLANG_OPT_MEGA_GATE_IMAGE_MASK_ONCE.get():
        return input_ids == image_token_id
    c = _IMAGE_MASK_CACHE
    if (
        c is None
        or c[0] is not input_ids
        or c[1] != input_ids.data_ptr()
        or c[2] != image_token_id
        or c[3] >= layer_id
        or c[5] != hidden_states.shape[0]
    ):
        mask = input_ids == image_token_id
    else:
        mask = c[4]
    _IMAGE_MASK_CACHE = (
        input_ids,
        input_ids.data_ptr(),
        image_token_id,
        layer_id,
        mask,
        hidden_states.shape[0],
    )
    return mask


def _try_mega_gate(
    moe: DeepseekV2MoE,
    hidden_states: torch.Tensor,
    num_token_non_padded: Optional[torch.Tensor],
    input_ids_global: Optional[torch.Tensor] = None,
    raw_topk_ids: bool = False,
):
    """Router GEMM + biased top-k in one `deep_gemm.bf16_mega_gate` launch.

    Replaces `moe.gate(hidden_states)` followed by `moe.topk(...)` (or
    `vision_topk`). The kernel scores with sigmoid/sqrtsoftplus, selects top-k
    on `score + correction_bias` and renormalizes the unbiased scores -- exactly
    the `moe_fused_gate` semantics, including     the V4.1 VL variant where
    image-token rows use `e_score_correction_bias_vl` (`image_bias` +
    `image_token_mask`). Anything outside that routing shape returns `None`.
    Gated by SGLANG_OPT_DEEPGEMM_MEGA_GATE.

    raw_topk_ids: return the kernel's ids as-is (no int32 cast, no padded-row
    mask); the caller folds both into mega_moe_pre_dispatch (vllm#57604).

    Runs under CUDA-graph capture too: DeepGEMM only refuses a capturing stream
    on its lazy-init path, and sglang's graph runner warms up on the capture
    stream first (see sglang/kernels/ops/layernorm/mega_mhc.py for the measured
    behaviour). A refusal throws before any stream work, so falling back inside
    the capture is safe.
    """
    global _MEGA_GATE_RUNTIME_DISABLED, _MEGA_GATE_CAPTURED_LOGGED
    global _MEGA_GATE_CAPTURE_FALLBACK_WARNED

    if _MEGA_GATE_RUNTIME_DISABLED or not envs.SGLANG_OPT_DEEPGEMM_MEGA_GATE.get():
        return None
    if _device_sm < 100 or getattr(moe, "is_hash", False):
        return None
    if moe.num_fused_shared_experts:
        return None


    gate = moe.gate
    bias = getattr(gate, "e_score_correction_bias", None)
    if bias is None or bias.dtype != torch.float32 or bias.dim() != 1:
        return None
    if gate.weight.dtype != torch.bfloat16 or hidden_states.dtype != torch.bfloat16:
        return None

    cfg = getattr(moe.topk, "topk_config", None)
    if cfg is None or getattr(moe.topk, "enable_waterfill", False):
        return None
    if (
        cfg.use_grouped_topk
        or not cfg.renormalize
        or cfg.top_k <= 1
        or cfg.num_fused_shared_experts
        or cfg.custom_routing_function is not None
        or cfg.scoring_func not in ("sigmoid", "sqrtsoftplus")
    ):
        return None
    # V4.1 VL: image-token rows route with a second correction bias. The kernel
    # expresses it with image_bias + a per-token bool image_token_mask, which
    # needs the token ids; without them `vision_topk` is not reproducible here.
    image_bias = getattr(gate, "e_score_correction_bias_vl", None)
    image_token_mask = None
    is_vl = isinstance(image_bias, torch.Tensor)
    if is_vl:
        image_token_id = getattr(moe.config, "image_token_id", None)
        if (
            image_token_id is None
            or input_ids_global is None
            or input_ids_global.shape[0] != hidden_states.shape[0]
            or image_bias.dtype != torch.float32
            or image_bias.shape != bias.shape
        ):
            return None
        image_token_mask = _image_token_mask(
            input_ids_global, image_token_id, moe.layer_id, hidden_states
        )
    else:
        image_bias = None

    # EPLB logical->physical remap (and the `fake` logit transform) can't be
    # reproduced by the fused kernel. Only the NON-VL fallback (`moe.topk`)
    # consumes expert_location_dispatch; the VL fallback (`vision_topk`) ignores
    # it, so for a VL gate no remap is applied on this path either -- declining
    # on a present dispatch info would just forfeit the fusion for nothing (e.g.
    # under `--ep-dispatch-algorithm fake`). Keep declining only for non-VL.
    if (
        not is_vl
        and ExpertLocationDispatchInfo.init_new(layer_id=moe.layer_id) is not None
    ):
        return None

    import deep_gemm

    # Routed scaling lives at two independent levels; this is the GATE level and
    # it mirrors `moe_fused_gate` (what the fallback `vision_topk` calls) exactly:
    #   * gate level: the kernel folds the factor into topk_weights, but only
    #     when `cfg.apply_routed_scaling_factor_on_output` says the TopK output
    #     is where the factor belongs -- otherwise we pass 1.0 and the weights
    #     stay unscaled. `moe_fused_gate` takes the factor and that same flag and
    #     applies the identical conditional internally, so the topk_weights
    #     returned here carry the same semantics as the unfused path's.
    #   * output level: `_run_mega_routed` scales the MoE output `y` further down
    #     (see `apply_routed_scaling` there), outside this branch, so it covers
    #     the fused and unfused gates alike. Nothing is double-scaled or dropped.
    routed_scaling = (
        cfg.routed_scaling_factor if cfg.apply_routed_scaling_factor_on_output else 1.0
    )
    capturing = torch.cuda.is_current_stream_capturing()
    try:
        topk_idx, topk_weights = deep_gemm.bf16_mega_gate(
            hidden_states.contiguous(),
            gate.weight,
            cfg.top_k,
            False,  # use_shared_as_routed
            0,  # num_shared_experts
            float(routed_scaling if routed_scaling is not None else 1.0),
            get_parallel().moe_ep_rank,
            cfg.scoring_func,
            None,  # mask
            bias,
            image_bias,
            image_token_mask,
        )
    except Exception as err:  # noqa: BLE001
        if _CAPTURE_ASSERT_MARK in str(err):
            # Lazy init refused under capture; nothing was issued on the stream,
            # so the graph stays valid and this capture records the TopK path.
            if not _MEGA_GATE_CAPTURE_FALLBACK_WARNED:
                logger.warning(
                    "deep_gemm.bf16_mega_gate declined a capturing stream that was "
                    "not warmed up; this graph records the router GEMM + TopK path"
                )
                _MEGA_GATE_CAPTURE_FALLBACK_WARNED = True
            return None
        logger.warning(
            "deep_gemm.bf16_mega_gate failed, falling back to the TopK path: %s", err
        )
        _MEGA_GATE_RUNTIME_DISABLED = True
        return None

    from sglang.srt.layers.moe.topk import _mask_topk_ids_padded_region

    log_key = (cfg.top_k, image_bias is not None)
    if log_key not in _MEGA_GATE_LOGGED:
        logger.info(
            "deep_gemm.bf16_mega_gate is serving the MoE router "
            "(top_k=%d, scoring_func=%s, routed_scaling=%s, vl_bias=%s)",
            cfg.top_k,
            cfg.scoring_func,
            routed_scaling,
            image_bias is not None,
        )
        _MEGA_GATE_LOGGED.add(log_key)
    if capturing and not _MEGA_GATE_CAPTURED_LOGGED:
        logger.info(
            "deep_gemm.bf16_mega_gate was captured into a CUDA graph (tokens=%d)",
            hidden_states.shape[0],
        )
        _MEGA_GATE_CAPTURED_LOGGED = True

    if raw_topk_ids:
        return topk_idx, topk_weights
    if envs.SGLANG_OPT_MEGA_GATE_CAST_MASK_FUSE.get():
        from sglang.kernels.ops.moe.topk_ids_cast import (
            can_cast_mask_topk_ids,
            cast_mask_topk_ids,
        )

        if can_cast_mask_topk_ids(topk_idx, num_token_non_padded):
            return cast_mask_topk_ids(topk_idx, num_token_non_padded), topk_weights
    topk_ids = topk_idx.to(torch.int32)
    _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
    return topk_ids, topk_weights


def _run_mega_routed(
    moe: DeepseekV2MoE,
    hidden_states: torch.Tensor,
    forward_batch: Optional[ForwardBatch],
    input_ids_global: Optional[torch.Tensor],
    num_tokens: int,
    apply_routed_scaling: bool = True,
    fuse_shared: bool = False,
    zero_routed: bool = False,
    routed_scale_in_weights: bool = False,
) -> torch.Tensor:
    # zero_routed / routed_scale_in_weights only serve _selfcheck_fused_shared:
    # zero every routed weight (ids stay valid, so the dispatch is unchanged and
    # a fused call returns the in-kernel shared expert alone), and move the
    # routed scaling onto the weights on the unfused path as the fused path does.
    import deep_gemm

    from sglang.srt.distributed.parallel_state import get_moe_ep_group

    hidden_size = moe.config.hidden_size
    # vllm#57604: the sm100 fp8xfp4 staging kernel does the int32 cast, the
    # padded-row mask and the routed-scaling multiply itself.
    fold_staging = (
        envs.SGLANG_OPT_MEGA_MOE_STAGING_FOLD.get()
        and _device_sm != 90
        and _mega_moe_mma_type() == "fp8xfp4"
    )
    staging_num_token_non_padded = None

    if num_tokens > 0:
        num_token_non_padded = (
            forward_batch.num_token_non_padded if forward_batch is not None else None
        )
        mega_gate_output = _try_mega_gate(
            moe,
            hidden_states,
            num_token_non_padded,
            input_ids_global,
            raw_topk_ids=fold_staging,
        )
        if mega_gate_output is not None:
            topk_ids, topk_weights = mega_gate_output
            if fold_staging:
                staging_num_token_non_padded = num_token_non_padded
        else:
            router_logits = moe.gate(hidden_states, forward_batch=forward_batch)
            if isinstance(
                getattr(moe.gate, "e_score_correction_bias_vl", None), torch.Tensor
            ):
                # V4.1 VL: image-token rows route with a separate correction bias.
                # MegaMoE consumes the same routed ids/weights as the normal TopK
                # path; mirror deepseek_v2.forward_normal's vision routing.
                from sglang.srt.multimodal.dsv41.vl_routing import vision_topk

                topk_output = vision_topk(
                    moe,
                    router_logits,
                    input_ids_global,
                    num_token_non_padded=num_token_non_padded,
                )
            else:
                topk_kwargs = {"input_ids": input_ids_global} if moe.is_hash else {}
                topk_output = moe.topk(
                    hidden_states,
                    router_logits,
                    num_token_non_padded=num_token_non_padded,
                    expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                        layer_id=moe.layer_id,
                    ),
                    **topk_kwargs,
                )
            topk_ids = topk_output.topk_ids
            topk_weights = topk_output.topk_weights
    else:
        topk_ids = None
        topk_weights = None

    ep_group = get_moe_ep_group().device_group
    num_experts = moe.experts.num_experts
    top_k = moe.config.num_experts_per_tok + moe.num_fused_shared_experts
    intermediate_size = moe.config.moe_intermediate_size
    num_max_tokens_per_rank = (
        envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    )
    assert num_tokens <= num_max_tokens_per_rank, (
        f"mega MoE: num_tokens={num_tokens} exceeds cap "
        f"SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK="
        f"{num_max_tokens_per_rank}; raise the env var or lower "
        f"--cuda-graph-max-bs-decode / --chunked-prefill-size accordingly"
    )

    buf = _get_mega_moe_symm_buffer(
        ep_group,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_topk=top_k,
        hidden=hidden_size,
        intermediate_hidden=intermediate_size,
        num_shared_experts=moe.n_shared_experts if fuse_shared else 0,
    )

    if num_tokens > 0:
        if fold_staging and topk_ids.dtype in (torch.int32, torch.int64):
            topk_ids_in = topk_ids.contiguous()
        else:
            topk_ids_in = topk_ids.to(torch.int32)
        topk_weights_in = topk_weights.to(torch.float32)
    else:
        topk_ids_in = hidden_states.new_empty((0, top_k), dtype=torch.int32)
        topk_weights_in = hidden_states.new_empty((0, top_k), dtype=torch.float32)
    if zero_routed:
        topk_weights_in = torch.zeros_like(topk_weights_in)

    if _device_sm == 90:
        return run_sm90_mega_routed(
            moe,
            hidden_states,
            topk_ids_in,
            topk_weights_in,
            buf,
            num_tokens,
        )

    mma_type = _mega_moe_mma_type()
    if mma_type == "mxf4xmxf4":
        # FP4 path goes through DeepGEMM's mega_moe_pre_dispatch which
        # handles the E2M1 packing variant. The jit implementation
        # only emits FP8.
        deep_gemm.mega_moe_pre_dispatch(
            hidden_states,
            topk_ids_in,
            topk_weights_in,
            buf.x,
            buf.x_sf,
            buf.topk_idx,
            buf.topk_weights,
            num_tokens=num_tokens,
            group_size=32,
            mma_type=mma_type,
        )
    else:
        # The in-kernel shared expert output is unscaled, so the routed scaling
        # moves onto the routed weights instead of y.
        scale_in_weights = (
            (fuse_shared or routed_scale_in_weights)
            and apply_routed_scaling
            and not moe.experts.should_fuse_routed_scaling_factor_in_topk
            and float(moe.routed_scaling_factor) != 1.0
        )
        topk_weight_scale = 1.0
        if scale_in_weights:
            if fold_staging:
                topk_weight_scale = float(moe.routed_scaling_factor)
            else:
                topk_weights_in = topk_weights_in * float(moe.routed_scaling_factor)
        shared_block_m = 0
        if fuse_shared:
            shared_block_m = deep_gemm.get_block_m_for_mega_moe(
                num_ranks=buf.group.size(),
                num_experts=buf.num_experts,
                # DeepGEMM rounds the requested capacity up for its buffer.
                num_max_tokens_per_rank=buf.num_max_tokens_per_rank,
                # Match the non-null output allocation passed to DeepGEMM.
                num_tokens=max(num_tokens, 1),
                num_topk=buf.num_topk,
                mma_type=buf.mma_type,
            )
        if fold_staging:
            from sglang.srt.utils.community_opt import log_community_opt_once

            log_community_opt_once(
                "vllm#57604 mega_moe_staging_fold",
                "mega_moe_pre_dispatch takes topk_idx=%s as-is, masks padded rows "
                "in-kernel (%s), scales weights by %s, writes only [:max(n,1)] "
                "(tokens=%d)",
                str(topk_ids_in.dtype).removeprefix("torch."),
                "on" if staging_num_token_non_padded is not None else "off",
                topk_weight_scale,
                num_tokens,
            )
            mega_moe_pre_dispatch(
                hidden_states,
                topk_ids_in,
                topk_weights_in,
                buf.x,
                buf.x_sf,
                buf.topk_idx,
                buf.topk_weights,
                quant_group_size=32,
                shared_x_sf=buf.shared_l1_acts_sf if fuse_shared else None,
                shared_block_m=shared_block_m,
                num_token_non_padded=staging_num_token_non_padded,
                topk_weight_scale=topk_weight_scale,
                # DeepGEMM reads topk rows [0, max(num_tokens, 1)) only.
                fill_padded_tail=False,
            )
        else:
            mega_moe_pre_dispatch(
                hidden_states,
                topk_ids_in,
                topk_weights_in,
                buf.x,
                buf.x_sf,
                buf.topk_idx,
                buf.topk_weights,
                quant_group_size=32,
                shared_x_sf=buf.shared_l1_acts_sf if fuse_shared else None,
                shared_block_m=shared_block_m,
            )

    # Allocate at least one row so y has a non-null CUDA data_ptr;
    # the DeepGEMM tvm-ffi binding rejects nullptr in convert_to_torch_tensor().
    y = torch.empty(
        (max(num_tokens, 1), hidden_size),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    swiglu_limit = getattr(moe.config, "swiglu_limit", None)
    with _configure_mega_moe_deep_gemm_num_sms(deep_gemm):
        deep_gemm.fp8_fp4_mega_moe(
            y,
            moe.experts.mega_l1_weights,
            moe.experts.mega_l2_weights,
            buf,
            shared_l1_weights=moe.mega_shared_l1_weights if fuse_shared else None,
            shared_l2_weights=moe.mega_shared_l2_weights if fuse_shared else None,
            recipe=(1, 1, 32),
            activation="swiglu",
            activation_clamp=swiglu_limit,
            fast_math=True,
        )
    y = y[:num_tokens]

    if fuse_shared or (routed_scale_in_weights and mma_type == "fp8xfp4"):
        return y
    if apply_routed_scaling and not moe.experts.should_fuse_routed_scaling_factor_in_topk:
        y.mul_(moe.routed_scaling_factor)
    return y


def _selfcheck_fused_shared(
    moe, hidden_states, forward_batch, input_ids_global, num_tokens
) -> None:
    """Verification aid (SGLANG_COMMUNITY_SELFCHECK_MEGA_SHARED): compare the fused
    output and the production path (routed + separate shared expert) against an
    fp32-weight shared-expert reference on the same input. Every EP rank reaches
    this at the same layer, so the extra MegaMoE launches stay collective.

    `fused` measures ||y_fused - (y_routed + shared_fp32)|| / ||shared_fp32||, so
    it also carries the bf16 roundings of the routed sum, which scale with
    ||routed||, not ||shared||. The extra terms separate them (report_D 2.3):
      fused_shared_only  the fused kernel with every routed weight zeroed (ids
                         kept valid) -- just the in-kernel shared expert, the
                         same measure as `production`;
      fused_vs_routed_w  `fused` against y_routed_w + shared_fp32, where
                         y_routed_w folds the routed scaling into the weights
                         like the fused kernel (one bf16 store of the routed sum);
      routed_scale_noise ||y_routed_w - y_routed|| / ||shared_fp32||: the extra
                         rounding of the y.mul_(scale) on the unfused path;
      store_floor        ||bf16(y_routed + shared_fp32) - (y_routed + shared_fp32)||
                         / ||shared_fp32||: what even an exact fused kernel shows
                         here, from the single bf16 store of routed + shared;
      routed/shared      ||y_routed|| / ||shared_fp32||."""
    from sglang.kernels.ops.attention.dsv4.moe import silu_and_mul_clamp

    y_fused = _run_mega_routed(
        moe, hidden_states, forward_batch, input_ids_global, num_tokens,
        fuse_shared=True,
    ).float()
    y_routed = _run_mega_routed(
        moe, hidden_states, forward_batch, input_ids_global, num_tokens,
        fuse_shared=False,
    ).float()
    y_shared_only = _run_mega_routed(
        moe, hidden_states, forward_batch, input_ids_global, num_tokens,
        fuse_shared=True, zero_routed=True,
    ).float()
    y_routed_w = _run_mega_routed(
        moe, hidden_states, forward_batch, input_ids_global, num_tokens,
        fuse_shared=False, routed_scale_in_weights=True,
    ).float()
    shared_prod = moe._forward_shared_experts(hidden_states).float()

    def dequant(linear):
        w = linear.weight.data.float()
        sf = linear.weight_scale_inv.data.float()
        bn, bk = linear.quant_method.quant_config.weight_block_size
        return w * sf.repeat_interleave(bn, 0)[: w.shape[0]].repeat_interleave(bk, 1)[:, : w.shape[1]]

    shared = moe.shared_experts
    gate_up = (hidden_states.float() @ dequant(shared.gate_up_proj).t()).to(torch.bfloat16)
    act = gate_up.new_empty((gate_up.shape[0], gate_up.shape[1] // 2))
    if shared.swiglu_limit is not None:
        silu_and_mul_clamp(gate_up, act, float(shared.swiglu_limit))
    else:
        act = shared.act_fn(gate_up)
    shared_fp32 = act.float() @ dequant(shared.down_proj).t()
    target = y_routed + shared_fp32
    denom = shared_fp32.norm().clamp_min(1e-12)
    err_prod = (shared_prod - shared_fp32).norm() / denom
    err_fused = (y_fused - target).norm() / denom
    err_fused_shared = (y_shared_only - shared_fp32).norm() / denom
    err_fused_routed_w = (y_fused - (y_routed_w + shared_fp32)).norm() / denom
    routed_scale_noise = (y_routed_w - y_routed).norm() / denom
    store_floor = (target.to(torch.bfloat16).float() - target).norm() / denom
    routed_over_shared = y_routed.norm() / denom
    logger.info(
        "[community-opt] #39313 selfcheck: layer=%s tokens=%d "
        "rel_err_vs_fp32_shared: production=%.4e fused=%.4e "
        "(fused vs production rel=%.4e) "
        "fused_shared_only=%.4e fused_vs_routed_w=%.4e routed_scale_noise=%.4e "
        "store_floor=%.4e routed/shared=%.3f",
        getattr(moe, "layer_id", None),
        num_tokens,
        err_prod.item(),
        err_fused.item(),
        ((y_fused - (y_routed + shared_prod)).norm() / denom).item(),
        err_fused_shared.item(),
        err_fused_routed_w.item(),
        routed_scale_noise.item(),
        store_floor.item(),
        routed_over_shared.item(),
    )


def _deepgemm_cache_entries() -> Optional[int]:
    """Kernels in DeepGEMM's on-disk JIT cache (None when the dir is unknown)."""
    cache_dir = os.environ.get("DG_JIT_CACHE_DIR")
    if not cache_dir:
        return None
    try:
        return len(os.listdir(os.path.join(cache_dir, "cache")))
    except OSError:
        return 0


def _mega_moe_warmup_plan(
    moe: DeepseekV2MoE, buf: SymmBuffer, max_tokens: int, use_gate: bool
) -> dict:
    """vllm#60468: {(bf16_mega_gate config, fp8_fp4_mega_moe BLOCK_M): largest
    token count in [1, max_tokens] selecting it}, from DeepGEMM's own config
    getters. Both JIT caches are keyed by those configs, not by the token count
    (fp8_fp4_mega_moe takes num_tokens at runtime), so running one count per
    pair compiles every variant a request in that range can select."""
    import deep_gemm

    if use_gate:
        num_routed, hidden = moe.gate.weight.shape
        top_k = moe.topk.topk_config.top_k
    plan = {}
    for n in range(1, max_tokens + 1):
        gate_cfg = (
            tuple(
                sorted(
                    deep_gemm.get_bf16_mega_gate_config(
                        n, hidden, num_routed, top_k
                    ).items()
                )
            )
            if use_gate
            else None
        )
        block_m = deep_gemm.get_block_m_for_mega_moe(
            buf.group.size(),
            buf.num_experts,
            buf.num_max_tokens_per_rank,
            n,
            buf.num_topk,
            buf.mma_type,
        )
        plan[(gate_cfg, block_m)] = n
    return plan


def warmup_mega_moe_kernels(moes, max_tokens: int) -> dict:
    """vllm#60468: run every bf16_mega_gate x fp8_fp4_mega_moe variant reachable
    with up to `max_tokens` tokens per rank once, before serving.

    One representative layer per kernel identity (experts, top-k, fused shared
    expert, VL gate bias, hash routing): the JIT caches do not depend on the
    layer. fp8_fp4_mega_moe is an EP collective, so every rank derives the same
    token-count list from the same config and runs it in the same order; the
    rank-local bf16_mega_gate variants are compiled first so the collective
    phase only waits on the MegaMoE compiles themselves."""
    from sglang.srt.distributed.parallel_state import get_moe_ep_group

    stats = {"layers": 0, "token_counts": 0, "gate_cfgs": 0, "block_ms": 0}
    if (
        _device_sm != 100
        or _mega_moe_mma_type() != "fp8xfp4"
        or not get_moe_a2a_backend().is_megamoe()
        or max_tokens <= 0
    ):
        return stats
    ep = get_moe_ep_group()
    num_max_tokens_per_rank = (
        envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    )
    max_tokens = min(max_tokens, num_max_tokens_per_rank)
    seen = set()
    # Gated layers first: a hash layer only adds a MegaMoE variant when no gated
    # layer shares its kernel.
    for moe in sorted(moes, key=lambda m: bool(getattr(m, "is_hash", False))):
        if not getattr(moe.experts, "_mega_moe_weights_built", False):
            continue
        fuse_shared = moe.mega_shared_l1_weights is not None
        is_hash = bool(getattr(moe, "is_hash", False))
        is_vl = isinstance(
            getattr(moe.gate, "e_score_correction_bias_vl", None), torch.Tensor
        )
        top_k = moe.config.num_experts_per_tok + moe.num_fused_shared_experts
        moe_kernel = (
            moe.experts.num_experts,
            top_k,
            fuse_shared,
            moe.config.hidden_size,
            moe.config.moe_intermediate_size,
        )
        layer_key = moe_kernel + (is_vl, is_hash)
        if layer_key in seen or (is_hash and moe_kernel in seen):
            continue
        seen.update((layer_key, moe_kernel))

        device = moe.gate.weight.device
        gen = torch.Generator(device=device).manual_seed(0)
        hidden_states = torch.randn(
            (max_tokens, moe.config.hidden_size),
            generator=gen,
            dtype=torch.bfloat16,
            device=device,
        )
        input_ids = (
            torch.zeros((max_tokens,), dtype=torch.int64, device=device)
            if (is_vl or is_hash)
            else None
        )

        def ids(n):
            return None if input_ids is None else input_ids[:n]

        # Whether the gate serves decides the token-count list of the collective
        # below, so the EP group agrees on it first.
        use_gate = _try_mega_gate(moe, hidden_states[:1], None, ids(1)) is not None
        flag = torch.tensor([int(use_gate)], dtype=torch.int32)
        torch.distributed.all_reduce(
            flag, op=torch.distributed.ReduceOp.MIN, group=ep.cpu_group
        )
        use_gate = bool(flag.item())

        buf = _get_mega_moe_symm_buffer(
            ep.device_group,
            num_experts=moe.experts.num_experts,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            num_topk=top_k,
            hidden=moe.config.hidden_size,
            intermediate_hidden=moe.config.moe_intermediate_size,
            num_shared_experts=moe.n_shared_experts if fuse_shared else 0,
        )
        plan = _mega_moe_warmup_plan(moe, buf, max_tokens, use_gate)
        counts = sorted(plan.values())
        if use_gate:
            for n in counts:
                _try_mega_gate(moe, hidden_states[:n], None, ids(n))
            torch.cuda.synchronize()
        ep.barrier()
        for n in counts:
            _run_mega_routed(
                moe, hidden_states[:n], None, ids(n), n, fuse_shared=fuse_shared
            )
        torch.cuda.synchronize()

        stats["layers"] += 1
        stats["token_counts"] += len(counts)
        stats["gate_cfgs"] += len({g for g, _ in plan} - {None})
        stats["block_ms"] += len({b for _, b in plan})
        del hidden_states, input_ids
    return stats


def _interleave_mega_moe_gate_up(t: torch.Tensor, gran: int = 8) -> torch.Tensor:
    # Match DeepGEMM's L1 gate/up layouts. FP8 activations use contiguous
    # gran-8 chunks; packed MXFP4 activations use even/odd gran-16 chunks.
    num_groups, n, *rest = t.shape
    half = n // 2
    gate = t[:, :half].reshape(num_groups, half // gran, gran, *rest)
    up = t[:, half:].reshape(num_groups, half // gran, gran, *rest)
    if gran == 16:
        result = torch.cat(
            [gate[:, :, 0::2], up[:, :, 0::2], gate[:, :, 1::2], up[:, :, 1::2]],
            dim=2,
        ).reshape(num_groups, n, *rest)
    else:
        result = torch.stack([gate, up], dim=2).reshape(num_groups, n, *rest)
    return torch.empty_like(t).copy_(result)


def _interleave_mega_moe_l1_weights(
    l1_weights: tuple[torch.Tensor, torch.Tensor],
    mma_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    gran = 16 if mma_type == "mxf4xmxf4" else 8
    return (
        _interleave_mega_moe_gate_up(l1_weights[0], gran=gran),
        _interleave_mega_moe_gate_up(l1_weights[1], gran=gran),
    )


def _transpose_mega_moe_sf_for_utccp(sf: torch.Tensor) -> torch.Tensor:
    num_groups, mn, packed_sf_k = sf.shape
    assert sf.dtype == torch.int and mn % 128 == 0
    result = (
        sf.reshape(num_groups, -1, 4, 32, packed_sf_k)
        .transpose(2, 3)
        .reshape(num_groups, mn, packed_sf_k)
    )
    return torch.empty_like(sf).copy_(result)


def build_mega_moe_experts_weights(experts) -> None:
    from deep_gemm import (
        transform_sf_into_required_layout,
    )

    if getattr(experts, "_mega_moe_weights_built", False):
        return

    mma_type = _mega_moe_mma_type()
    w13 = experts.w13_weight.data
    w13_sf_fp32 = experts.w13_weight_scale_inv.data
    w2 = experts.w2_weight.data
    w2_sf_fp32 = experts.w2_weight_scale_inv.data

    num_groups, n1, half_k1 = w13.shape
    k1 = half_k1 * 2
    _, n2, half_k2 = w2.shape
    k2 = half_k2 * 2

    w13_sf = transform_sf_into_required_layout(
        w13_sf_fp32,
        mn=n1,
        k=k1,
        recipe=(1, 32),
        num_groups=num_groups,
        disable_ue8m0_cast=False,
    )
    w2_sf = transform_sf_into_required_layout(
        w2_sf_fp32,
        mn=n2,
        k=k2,
        recipe=(1, 32),
        num_groups=num_groups,
        disable_ue8m0_cast=False,
    )

    # Build the interleaved L1 weight + scale once; share the weight buffer
    # between `w13_weight.data` (normal deep-ep path) and `mega_l1_weights[0]`
    # (mega moe path). Mega moe additionally needs a UTCCP-transposed scale;
    # the deep-ep path consumes the non-transposed interleaved scale and a
    # swizzle-aware activation kernel. L2 weight is untouched by the mega
    # transform, so the existing `w2_weight.data` is shared directly.
    w13_interleaved, w13_sf_interleaved = _interleave_mega_moe_l1_weights(
        (w13, w13_sf), mma_type
    )
    w13_sf_utccp = _transpose_mega_moe_sf_for_utccp(w13_sf_interleaved)
    w2_sf_utccp = _transpose_mega_moe_sf_for_utccp(w2_sf)

    experts.w13_weight.data = w13_interleaved
    experts.w13_weight_scale_inv.data = w13_sf_interleaved
    experts.w2_weight_scale_inv.data = w2_sf
    experts.w13_weight_scale_inv.format_ue8m0 = True
    experts.w2_weight_scale_inv.format_ue8m0 = True

    experts.mega_l1_weights = (experts.w13_weight.data, w13_sf_utccp)
    experts.mega_l2_weights = (experts.w2_weight.data, w2_sf_utccp)

    experts._mega_moe_weights_built = True


def build_mega_moe_shared_weights(shared_experts):
    """Interleaved FP8 L1 / L2 shared-expert weights for DeepGEMM's MegaMoE. The L1
    is a copy; the originals stay for the forward_normal fallback above the cap."""
    from deep_gemm import transform_weights_for_mega_moe

    gate_up_proj, down_proj = shared_experts.gate_up_proj, shared_experts.down_proj
    return transform_weights_for_mega_moe(
        (gate_up_proj.weight.data, _transform_mega_moe_shared_sf(gate_up_proj)),
        (down_proj.weight.data, _transform_mega_moe_shared_sf(down_proj)),
    )


def _transform_mega_moe_shared_sf(linear) -> torch.Tensor:
    from deep_gemm import transform_sf_into_required_layout

    from sglang.srt.layers.quantization.fp8_utils import inverse_transform_scale_ue8m0

    n, k = linear.weight.shape
    sf = linear.weight_scale_inv.data
    block_n, block_k = linear.quant_method.quant_config.weight_block_size
    if sf.dtype == torch.int32:
        # Requantized for the DeepGEMM linear runner (128-wide K groups).
        sf = inverse_transform_scale_ue8m0(sf, mn=n)[:, : k // 128]
        block_n, block_k = 128, 128
    sf = sf.float()
    if block_k == 128:
        sf, recipe = sf.repeat_interleave(4, dim=-1), (block_n, 32)
    else:
        # [32, 32] ue8m0 blocks: DeepGEMM broadcasts the 32-row granularity to the
        # same per-row 1x32 layout (checked equal to an explicit (1, 32) expansion).
        recipe = (block_n, block_k)
    return transform_sf_into_required_layout(
        sf, mn=n, k=k, recipe=recipe, disable_ue8m0_cast=False
    )
