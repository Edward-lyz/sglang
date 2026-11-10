# Copyright 2023-2026 SGLang Team
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
"""No-cuda-graph phase runner; the eager dual of BaseCudaGraphRunner."""

from __future__ import annotations

import contextlib
import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Tuple, Union

import torch

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.environ import envs
from sglang.srt.layers.cp.utils import (
    cp_gather_after_forward,
    cp_gather_rank_major,
    cp_shard_model_inputs,
    get_cp_strategy,
    is_cp_active,
    prepare_cp_forward,
)
from sglang.srt.layers.pooler import EmbeddingPoolerOutput
from sglang.srt.model_executor.cuda_graph_buffer_registry import (
    build_eager_registry,
)
from sglang.srt.model_executor.forward_batch_deepseek_mha_mixin import (
    create_chunked_prefix_cache_kv_indices,
)
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    get_req_to_token_pool,
    get_token_to_kv_pool,
)
from sglang.srt.model_executor.runner.base_runner import BaseRunner
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    enable_tc_piecewise_cuda_graph,
    set_tc_piecewise_forward_context,
)
from sglang.srt.model_executor.runner_utils import (
    maybe_publish_prefill_shared_read_done,
)
from sglang.srt.runtime_context import (
    get_exec,
    get_parallel,
    get_spec,
    max_prefill_buffer_tokens,
    max_speculative_num_draft_tokens,
)
from sglang.srt.utils import is_hip, is_npu
from sglang.srt.utils.common import (
    ceil_align,
    get_eager_max_batch_size,
    require_mlp_sync,
)
from sglang.srt.utils.device_timer import device_timer_ctx

logger = logging.getLogger(__name__)

_is_hip = is_hip()

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.model_runner import ModelRunner


def _cp_last_token_rows(hidden_states, forward_batch):
    """[fusion-opt] SGLANG_FUSE_CP_LAST_TOKEN_GATHER: when the logits only read each
    request's last prompt token (prefill, no input logprobs, no FULL hidden capture),
    CP-gather just those bs rows instead of the whole [T, hidden] (and the interleave
    reorder). Interleave CP keeps global token t on rank t % cp, local row t // cp; each
    rank contributes its owned rows, the gather is a plain copy -> bitwise the rows the
    stock gather + last-token prune yields. Returns rows in batch order (same tuple
    shape as cp_gather_after_forward), or None when the batch does not qualify."""
    from sglang.srt.environ import envs

    forward_batch.cp_last_rows_pruned = False
    if not envs.SGLANG_FUSE_CP_LAST_TOKEN_GATHER.get():
        return None
    from sglang.srt.layers.cp.base import get_cp_strategy
    from sglang.srt.layers.cp.interleave import InterleaveCPStrategy
    from sglang.srt.layers.dp_attention import attn_cp_all_gather_into_tensor
    from sglang.srt.runtime_context import get_parallel

    meta = getattr(forward_batch, "attn_cp_metadata", None)
    mode = forward_batch.forward_mode
    lens_cpu = forward_batch.extend_seq_lens_cpu
    if (
        meta is None
        or meta.gather_index is not None
        or not isinstance(get_cp_strategy(), InterleaveCPStrategy)
        or not mode.is_extend()
        or mode.is_target_verify()
        or mode.is_draft_extend_v2()
        or (
            # FULL capture only reads the full rows when the DSpark aux does not take
            # the F2b tail path (which replaces output.hidden_states with the aux)
            forward_batch.capture_hidden_mode.is_full()
            and getattr(forward_batch, "dspark_aux_tail_global_idx", None) is None
        )
        or forward_batch.extend_seq_lens is None
        or not lens_cpu
        or min(lens_cpu) <= 0
        or sum(lens_cpu) != int(meta.total_seq_lens)
    ):
        return None
    if forward_batch.return_logprob and any(
        e - st > 0
        for e, st in zip(lens_cpu, forward_batch.extend_logprob_start_lens_cpu)
    ):
        return None  # input logprobs need every row

    parallel = get_parallel()
    cp, rank = parallel.attn_cp_size, parallel.attn_cp_rank
    last = torch.cumsum(forward_batch.extend_seq_lens.to(torch.int64), dim=0) - 1
    owner = last % cp
    bs = last.shape[0]

    def pick(x):
        """CP-gather the last-token row of each request from its owner rank."""
        if x is None:
            return None
        local = x[(last // cp).clamp(max=max(x.shape[0] - 1, 0))]
        local = torch.where((owner == rank).view(-1, *[1] * (x.dim() - 1)), local, 0)
        gathered = local.new_empty((cp * bs, *x.shape[1:]))
        attn_cp_all_gather_into_tensor(gathered, local.contiguous())
        return gathered[owner * bs + torch.arange(bs, device=x.device)]

    from sglang.srt.utils.community_opt import log_community_opt_once

    log_community_opt_once(
        "fusion-opt cp_last_token_gather",
        "CP-gather only the last-token rows (bs=%d, total=%d, cp=%d)",
        bs,
        int(meta.total_seq_lens),
        cp,
    )
    forward_batch.cp_last_rows_pruned = True
    if isinstance(hidden_states, tuple):
        out = tuple(pick(item) for item in hidden_states)
        if len(out) == 2 and out[1] is None:
            return out[0]
        return out
    return pick(hidden_states)


def _cp_embed_reduce_scatter(embed, input_ids, forward_batch):
    """[fusion-opt] SGLANG_FUSE_CP_EMBED_REDUCE_SCATTER: this rank's interleave-CP shard of
    the vocab-parallel embedding via one reduce-scatter of rank-major ids, instead of an
    all-reduce over all T rows followed by the shard copy. Every row has exactly one
    non-zero contributor (the vocab shard owning its id), so the sum is exact and the
    result is bitwise the stock one. None when the layout does not qualify."""
    from sglang.srt.environ import envs

    if not envs.SGLANG_FUSE_CP_EMBED_REDUCE_SCATTER.get():
        return None
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.layers.cp.base import get_cp_strategy
    from sglang.srt.layers.cp.interleave import InterleaveCPStrategy
    from sglang.srt.runtime_context import get_parallel

    meta = getattr(forward_batch, "attn_cp_metadata", None)
    parallel = get_parallel()
    cp, tp = parallel.attn_cp_size, parallel.tp_size
    from sglang.srt.utils.community_opt import log_community_opt_once

    log_community_opt_once(
        "fusion-opt cp_embed_reduce_scatter gate",
        "embed RS gate: meta=%s strategy=%s use_attn_tp=%s enable_tp=%s emb_tp=%s tp=%s "
        "cp=%s tp_rank=%s cp_rank=%s gather_index=%s logical=%s actual=%s",
        meta is not None,
        type(get_cp_strategy()).__name__,
        getattr(embed, "use_attn_tp_group", None),
        getattr(embed, "enable_tp", None),
        getattr(embed, "tp_size", None),
        tp,
        cp,
        parallel.tp_rank,
        parallel.attn_cp_rank,
        None if meta is None else meta.gather_index is not None,
        None if meta is None else meta.per_rank_logical_token,
        None if meta is None else meta.per_rank_actual_token,
    )
    if (
        meta is None
        or not isinstance(get_cp_strategy(), InterleaveCPStrategy)
        or getattr(embed, "use_attn_tp_group", False)
        or not getattr(embed, "enable_tp", False)
        or getattr(embed, "tp_size", 1) != tp
        or tp != cp
        or parallel.tp_rank != parallel.attn_cp_rank
        or meta.gather_index is not None
        or len(set(meta.per_rank_actual_token)) != 1
    ):
        return None
    total = int(meta.total_seq_lens)
    rank_len = max(meta.per_rank_actual_token)
    if total == 0 or rank_len == 0:
        return None
    m = torch.arange(cp * rank_len, device=input_ids.device)
    t = (m % rank_len) * cp + m // rank_len
    ids_rm = torch.where(
        t < total, input_ids[t.clamp(max=total - 1)], torch.zeros_like(input_ids[:1])
    )
    from sglang.srt.utils.community_opt import log_community_opt_once

    log_community_opt_once(
        "fusion-opt cp_embed_reduce_scatter",
        "CP embedding via reduce-scatter (total=%d rank_len=%d cp=%d)",
        total,
        rank_len,
        cp,
    )
    partial = embed._embed_local_shard(ids_rm)
    out = partial.new_empty((rank_len, partial.shape[1]))
    get_tp_group().reduce_scatter_tensor(out, partial)
    # rows past this rank's logical count are the CP padding pad_local_rows zero-fills
    logical = meta.per_rank_logical_token or meta.per_rank_actual_token
    local_len = logical[parallel.attn_cp_rank]
    if local_len < rank_len:
        out[local_len:].zero_()
    if meta.per_rank_logical_token is None or meta.per_rank_logical_token == meta.per_rank_actual_token:
        return out[:local_len]
    return out


class EagerRunner(BaseRunner):
    def __init__(self, model_runner: ModelRunner) -> None:
        super().__init__(model_runner)
        mr = model_runner
        sa = mr.server_args
        # Built first so the cg runners coalesce onto its buffers via the shared
        # input pool; size to the largest tokens/req across modes the worker hits.
        num_tokens_per_req = 1
        if mr.spec_algorithm.is_speculative():
            # speculative_adaptive can grow draft tokens at runtime; size to the max.
            num_draft_tokens = max_speculative_num_draft_tokens() or 1
            if mr.is_draft_worker:
                num_tokens_per_req = max(
                    get_spec().speculative_eagle_topk or 1,
                    num_draft_tokens,
                    (
                        2 * (get_spec().speculative_num_steps or 0)
                        if get_spec().enable_multi_layer_eagle
                        else 0
                    ),
                )
            else:
                num_tokens_per_req = mr.decode_num_tokens_per_req(
                    num_draft_tokens=num_draft_tokens
                )
        else:
            dllm_config = DllmConfig.from_server_args(sa)
            if dllm_config is not None:
                # dLLM runs block_size tokens/request (DLLM_EXTEND).
                num_tokens_per_req = dllm_config.block_size
        max_bs = mr.max_running_requests
        if (
            mr.is_draft_worker
            and mr.spec_algorithm.is_frozen_kv_mtp()
            and get_spec().speculative_eagle_topk > 1
        ):
            # Frozen-KV MTP expands the draft batch by topk on the bs axis
            # (expand_for_topk_draft) before the eager fallback.
            max_bs *= get_spec().speculative_eagle_topk
        # Mirror prepare_mlp_sync_batch padding so the registry holds what load_batch copies.
        max_bs = get_eager_max_batch_size(max_bs)
        prefill_ceiling = max(mr.max_total_num_tokens, max_prefill_buffer_tokens())
        max_num_token = max(prefill_ceiling, max_bs * num_tokens_per_req)
        if require_mlp_sync():
            from sglang.srt.layers.cp.padding import get_cp_padding_align_size

            max_num_token = ceil_align(max_num_token, self.attn_tp_size)
            max_num_token = ceil_align(max_num_token, get_cp_padding_align_size())
        self._eager_max_bs = max_bs
        self._eager_num_tokens_per_req = num_tokens_per_req
        is_encoder_decoder = mr.model_config.is_encoder_decoder
        self._eager_registry = build_eager_registry(
            device=mr.device,
            max_bs=max_bs,
            max_num_token=max_num_token,
            cache_loc_dtype=torch.int64,
            enable_mamba_track=(
                get_exec().mamba.enable_mamba_extra_buffer
                and mr.spec_algorithm.is_none()
            ),
            is_encoder_decoder=is_encoder_decoder,
            encoder_len_fill_value=(
                getattr(mr.model_config.hf_config, "max_source_positions", 0)
                if is_encoder_decoder
                else 0
            ),
            encoder_lens_dtype=(
                torch.int64 if torch.device(mr.device).type == "cpu" else torch.int32
            ),
            dp_size=get_parallel().dp_size,
        )
        # Eager has no capture step, so warm up here (run-once via mr._kernel_warmed_up).
        self.warmup()

    def _autotune_buffers(self) -> Tuple[Any, int]:
        """Decode-shaped dummy buffers (bs * num_tokens_per_req) for the warmup
        flashinfer-autotune forward.

        flashinfer's MoE autotuner times candidate tactics against the buffer it
        is given, so it must match the live decode shape for the cached tactic to
        be optimal at decode. The eager input registry spans the prefill token
        ceiling; the dummy run only needs the decode-sized slice.
        """
        mr = self.model_runner
        num_tokens_per_req = 1
        if mr.spec_algorithm.is_speculative():
            num_tokens_per_req = mr.decode_num_tokens_per_req()
        return (
            self._alloc_dummy_decode_buffers(
                self._eager_max_bs, num_tokens_per_req=num_tokens_per_req
            ),
            self._eager_max_bs,
        )

    def can_run_graph(self, forward_batch: ForwardBatch) -> bool:
        # Eager never runs a cuda graph; callers dispatch on isinstance(...,
        # EagerRunner) and must not route an eager batch into a replay branch.
        return False

    def load_batch(
        self, forward_batch: ForwardBatch, pp_proxy_tensors=None, **kwargs
    ) -> ForwardBatch:
        """Copy the live batch into the fixed-max eager static buffers (sliced to
        this batch's shape) — the eager counterpart of the cuda-graph runners'
        load_batch."""
        if envs.SGLANG_EAGER_INPUT_NO_COPY.get():
            return replace(forward_batch)
        raw_bs = forward_batch.batch_size
        if forward_batch.input_ids is not None:
            raw_num_tokens = forward_batch.input_ids.shape[0]
        elif forward_batch.input_embeds is not None:
            raw_num_tokens = forward_batch.input_embeds.shape[0]
        else:
            raw_num_tokens = 0
        registry = self._eager_registry
        registry.fill_from(
            forward_batch,
            raw_bs=raw_bs,
            padded_bs=raw_bs,
            raw_num_tokens=raw_num_tokens,
            padded_num_tokens=raw_num_tokens,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        return registry.extract_buffer(
            padded_bs=raw_bs,
            padded_num_tokens=raw_num_tokens,
            forward_batch_template=forward_batch,
        )

    def execute(
        self, forward_batch: ForwardBatch, pp_proxy_tensors=None, **kwargs
    ) -> Any:
        mode = forward_batch.forward_mode
        if mode.is_mixed() and not is_npu() and get_cp_strategy() is None:
            # A mixed batch is extend-shaped (decode tails are 1-token
            # extends); run it as EXTEND. NPU keeps MIXED for its dedicated
            # kernel; CP keeps it to skip the zigzag split.
            forward_batch.forward_mode = ForwardMode.EXTEND
            mode = ForwardMode.EXTEND
        if mode.is_decode():
            return self._execute_decode(forward_batch, pp_proxy_tensors)
        if mode.is_idle():
            return self._execute_idle(forward_batch, pp_proxy_tensors)
        if mode.is_extend(include_draft_extend_v2=True):
            return self._execute_extend(forward_batch, pp_proxy_tensors)
        raise ValueError(f"Invalid forward mode for eager runner: {mode}")

    def _resolve_decode_pdmux(
        self,
    ) -> Tuple[Any, contextlib.AbstractContextManager]:
        """Resolve the (attn_backend, forward_context) the eager decode forward
        runs under. PDmux selects a per-stream backend and publishes it via an
        active ForwardContext; non-pdmux uses attn_backend + the ambient ctx."""
        model_runner = self.model_runner
        if self.enable_pdmux:
            return model_runner.decode_attn_backend, forward_context(
                ForwardContext(attn_backend=model_runner.decode_attn_backend)
            )
        return model_runner.attn_backend, contextlib.nullcontext()

    def _execute_decode(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors=None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        model_runner = self.model_runner
        enable_pdmux = self.enable_pdmux
        attn_backend, pdmux_ctx = self._resolve_decode_pdmux()
        if not enable_pdmux:
            forward_batch = self.load_batch(forward_batch, pp_proxy_tensors)
        if forward_batch.needs_forward_metadata_init():
            if hasattr(model_runner.model, "prepare_forward_batch"):
                # Prepare model-specific attention metadata before planning,
                # e.g. Moss-VL's prefill cross-attention custom mask.
                model_runner.model.prepare_forward_batch(forward_batch)
            attn_backend.init_forward_metadata(forward_batch)
        # FIXME: add pp_proxy_tensors arg to all models
        kwargs = model_runner._pp_kwargs(pp_proxy_tensors)

        ctx = device_timer_ctx(model_runner.device_timer, "decode")

        with ctx, pdmux_ctx:
            return model_runner.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            )

    def _execute_extend(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors=None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors, EmbeddingPoolerOutput]:
        model_runner = self.model_runner
        kwargs = model_runner._extend_forward_kwargs(forward_batch, pp_proxy_tensors)

        if not self.enable_pdmux:
            forward_batch = self.load_batch(forward_batch, pp_proxy_tensors)

        cp_active = is_cp_active(forward_batch)
        if cp_active:
            prepare_cp_forward(forward_batch)

        # Target verify can arrive with ``forward_metadata_ready`` set by an
        # upstream/speculative planning step.  That mark does not initialize
        # the final target hybrid backend, and unlike a graph replay eager has
        # no static metadata load to fill the gap.  Re-plan target verify from
        # the final batch every time; eager metadata is intentionally derived
        # directly from the live ``spec_info`` tensors.
        if (
            forward_batch.needs_forward_metadata_init()
            or cp_active
            or forward_batch.forward_mode.is_target_verify()
        ):
            if model_runner.ps.attn_dcp_size > 1 and hasattr(
                model_runner.model, "prepare_context_parallel_metadata_for_dcp"
            ):
                # prepare kv cache buffer for dcp to gather kv cache
                forward_batch.attn_dcp_metadata = (
                    model_runner.model.prepare_context_parallel_metadata_for_dcp(
                        forward_batch.seq_lens,
                        forward_batch.extend_prefix_lens,
                        forward_batch.extend_prefix_lens_cpu,
                        forward_batch.extend_seq_lens,
                        forward_batch.req_pool_indices,
                        get_req_to_token_pool().req_to_token,
                        forward_batch.seq_lens_sum,
                        get_token_to_kv_pool().get_kv_buffer_shape()[0],
                        model_runner.kv_cache_dtype,
                        model_runner.device,
                        create_chunked_prefix_cache_kv_indices,
                    )
                )
            if hasattr(model_runner.model, "prepare_forward_batch"):
                # Prepare model-specific attention metadata before planning,
                # e.g. Moss-VL's prefill cross-attention custom mask.
                model_runner.model.prepare_forward_batch(forward_batch)
            model_runner.attn_backend.init_forward_metadata(forward_batch)
            model_runner.attn_backend.prepare_prefill_shared_read_snapshot(
                forward_batch,
                num_qo_tokens=len(forward_batch.input_ids),
            )
            maybe_publish_prefill_shared_read_done(
                model_runner,
                forward_batch,
                torch.get_device_module(model_runner.device),
            )

        if not cp_active:
            forward_batch.attn_cp_metadata = None

        category = (
            "target_verify"
            if forward_batch.forward_mode.is_target_verify()
            else "extend"
        )
        with device_timer_ctx(model_runner.device_timer, category):
            pcg_runner = model_runner.prefill_cuda_graph_runner
            if (
                _is_hip
                and pcg_runner is not None
                and not isinstance(pcg_runner, EagerRunner)
                and not cp_active
            ):
                # HIP PCG eager fallback: enter the PCG context so Dynamo guards
                # and PCG-specific MoE/attention paths stay consistent.
                with (
                    enable_tc_piecewise_cuda_graph(),
                    set_tc_piecewise_forward_context(
                        forward_batch,
                        model_runner.attention_layers,
                        getattr(model_runner.model, "quant_config", None),
                        model_runner.moe_layers,
                        model_runner.moe_fusions,
                        dsa_indexers=model_runner.dsa_indexers,
                        mha_companion_layers=model_runner.mha_companion_layers,
                    ),
                ):
                    ret = model_runner.model.forward(
                        forward_batch.input_ids,
                        forward_batch.positions,
                        forward_batch,
                        **kwargs,
                    )
            elif cp_active:
                ret = self._execute_extend_cp(forward_batch, kwargs)
            else:
                ret = model_runner.model.forward(
                    forward_batch.input_ids,
                    forward_batch.positions,
                    forward_batch,
                    **kwargs,
                )
        return ret

    def _execute_extend_cp(
        self, forward_batch: ForwardBatch, kwargs: dict
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        """CP extend: shard inputs at the model boundary, run the body on the
        rank-local slice, then gather hidden states before the logits step.
        """
        model = self.model_runner.model

        input_embeds = kwargs.get("input_embeds")
        complete_input_ids = forward_batch.input_ids
        local_embeds = None
        if input_embeds is None:
            # CP vision fix: merge image features into the embeddings BEFORE the CP
            # shard, mirroring DeepseekV4ForCausalLM.forward. The CP path calls the
            # inner model body (model.model) directly and bypasses the causal-LM
            # forward's multimodal merge, so without this the body only sees the
            # text embedding of the image placeholder tokens and the model cannot
            # see the image (garbage multimodal output under CP).
            if (
                getattr(model, "vision", None) is not None
                and not forward_batch.forward_mode.is_decode()
                and getattr(forward_batch, "mm_inputs", None) is not None
                and any(x is not None for x in forward_batch.mm_inputs)
                and hasattr(model, "_prepare_mm_embeddings")
            ):
                input_embeds = model._prepare_mm_embeddings(
                    complete_input_ids, forward_batch
                )
                from sglang.srt.managers.schedule_batch import MM_PAD_SHIFT_VALUE

                # Remap image-hash pad IDs -> image_token_id for the model body
                # (input_ids_global gather / routing), matching the non-CP path.
                complete_input_ids = complete_input_ids.masked_fill(
                    complete_input_ids >= MM_PAD_SHIFT_VALUE,
                    model.config.image_token_id,
                )
            else:
                local_embeds = _cp_embed_reduce_scatter(
                    model.get_input_embeddings(), complete_input_ids, forward_batch
                )
                if local_embeds is not None:
                    # [fusion-opt] already CP-local; hand the shard helper a 1-column
                    # stand-in so the positions / ids / metadata sharding stays as is.
                    input_embeds = complete_input_ids.new_zeros(
                        (complete_input_ids.shape[0], 1)
                    )
                else:
                    input_embeds = model.get_input_embeddings()(complete_input_ids)
        with cp_shard_model_inputs(
            input_embeds,
            forward_batch.positions,
            forward_batch,
            complete_input_ids,
        ) as (sharded_input_embeds, sharded_positions, model_input_ids):
            if local_embeds is not None:
                assert local_embeds.shape[0] == sharded_input_embeds.shape[0], (
                    local_embeds.shape,
                    sharded_input_embeds.shape,
                )
                sharded_input_embeds = local_embeds
                local_embeds = None
            # [MEM-R3] The full-order embedding is [T, hidden] (640 MiB at a 65532-token
            # chunk) and exists only to produce the CP-local shard above; the body runs
            # on `sharded_input_embeds`. Holding the reference here would pin it for the
            # whole model body, where it stacks on the ratio-1/2 indexer transients.
            # (If the caller passed input_embeds in `kwargs`, that reference still owns
            # the tensor, so this only drops our own.)
            input_embeds = None
            model_kwargs = {"input_embeds": sharded_input_embeds}
            if (pp_proxy_tensors := kwargs.get("pp_proxy_tensors")) is not None:
                model_kwargs["pp_proxy_tensors"] = pp_proxy_tensors
            hidden_states = model.model(
                model_input_ids,
                sharded_positions,
                forward_batch,
                **model_kwargs,
            )
        capture_aux_hidden_states = getattr(model, "capture_aux_hidden_states", False)
        aux_hidden_states = None
        if capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if not model.pp_group.is_last_rank:
            return (
                (hidden_states, aux_hidden_states)
                if capture_aux_hidden_states
                else hidden_states
            )

        stream = torch.cuda.current_stream()
        # [MEM-P1] DSV4 returns (hidden_states, pre_hc_head). When DSpark aux
        # capture is on, pre_hc_head is dropped at the logits boundary below
        # (`if aux_hidden_states is None`), so CP-gathering it is pure waste:
        # pre_hc_head is [T, hc_mult*H] and the interleave gather materializes it
        # twice (all-gather dst + out-of-place index_select reorder).
        # Dropping item 1 before the gather removes ~2x[T, hc_mult*H] from the
        # prefill activation peak. Passing None keeps cp_gather_after_forward's
        # existing `(x, None) -> x` contract, so nothing downstream changes.
        if (
            aux_hidden_states is not None
            and isinstance(hidden_states, tuple)
            and len(hidden_states) == 2
        ):
            hidden_states = (hidden_states[0], None)
        pruned = _cp_last_token_rows(hidden_states, forward_batch)
        if pruned is not None:
            hidden_states = pruned
        else:
            hidden_states = cp_gather_after_forward(hidden_states, forward_batch, stream)
        # DSpark aux tensors ride the same CP token split; gather them the same way.
        aux_row_to_token = None
        dspark_commit_mask = None
        aux_tail_gathered = None
        if aux_hidden_states is not None:
            if isinstance(aux_hidden_states, torch.Tensor):
                aux_hidden_states = cp_gather_after_forward(
                    aux_hidden_states, forward_batch, stream
                )
            else:
                # [MEM-P3] Pack the K captures locally ([T_local, K*hidden], 240 MiB
                # here) and CP-gather ONCE in rank-major order, instead of gathering
                # K full [T, hidden] tensors and letting the logits processor cat
                # them into a [T, K*hidden] copy. Dropping the local list before the
                # gather keeps the K local captures out of the live set.
                # The local pack ALONE was a regression (measured 5.70 -> 6.41 GiB):
                # the ordinary gather keeps both the all-gather destination and the
                # out-of-place index_select reorder alive, so packing only moved the
                # cost. Rank-major drops the reorder, so the two only pay off
                # together. The aux consumer (DSpark write_target_hidden_kv) scatters
                # per row, so it does not need the global token order -- it needs
                # cache_loc/positions permuted the same way, which
                # hidden_states_row_token_index carries. See SOLUTION.md §4/§10.
                local_packed = torch.cat(aux_hidden_states, dim=-1)
                del aux_hidden_states[:]
                tail_global_idx = getattr(
                    forward_batch, "dspark_aux_tail_global_idx", None
                )
                if tail_global_idx is not None:
                    # [fusion-opt] SGLANG_FUSE_DSPARK_AUX_TAIL_GATHER: every rank holds
                    # its (equally padded) tail rows; gather them rank-major together
                    # with their global token index. The logits processor gets no aux
                    # (it would index it by token); the commit reads it below.
                    from sglang.srt.layers.dp_attention import (
                        attn_cp_all_gather_into_tensor,
                    )
                    from sglang.srt.runtime_context import get_parallel

                    cp_size = get_parallel().attn_cp_size
                    rows = local_packed.shape[0]
                    gathered = local_packed.new_empty(
                        (cp_size * rows, local_packed.shape[1])
                    )
                    attn_cp_all_gather_into_tensor(gathered, local_packed.contiguous())
                    gathered_idx = tail_global_idx.new_empty((cp_size * rows,))
                    attn_cp_all_gather_into_tensor(
                        gathered_idx, tail_global_idx.contiguous()
                    )
                    aux_tail_gathered = (gathered, gathered_idx.to(torch.int64))
                    aux_hidden_states = None
                elif forward_batch.capture_hidden_mode.is_full():
                    aux_hidden_states, aux_row_to_token = cp_gather_rank_major(
                        local_packed, forward_batch, stream
                    )
                else:
                    # LAST capture prunes the aux by global token index inside the
                    # logits processor, so those rows must be in global order. The
                    # pruned tensor is [bs, K*hidden] -- no memory concern.
                    aux_hidden_states = cp_gather_after_forward(
                        local_packed, forward_batch, stream
                    )
                del local_packed
                if aux_hidden_states is not None and getattr(
                    forward_batch, "dspark_commit_mask_appended", False
                ):
                    # [fusion-opt] SGLANG_FUSE_DSPARK_COMMIT_TAIL_ONLY: the model
                    # appended a 1-column commit mask; split it back off.
                    dspark_commit_mask = aux_hidden_states[:, -1] > 0.5
                    aux_hidden_states = aux_hidden_states[:, :-1]
        logits_kwargs = {}
        # DSV4 returns (hidden_states, hidden_states_before_norm) from its model body.
        if isinstance(hidden_states, tuple):
            hidden_states, hidden_states_before_norm = hidden_states
            # Mirror DeepseekV4ForCausalLM.forward: drop pre_hc_head when
            # DSpark aux capture is on, else it overrides the packed aux.
            if aux_hidden_states is None:
                logits_kwargs["hidden_states_before_norm"] = hidden_states_before_norm
        output = model.logits_processor(
            forward_batch.input_ids,
            hidden_states,
            model.lm_head,
            forward_batch,
            aux_hidden_states,
            **logits_kwargs,
        )
        if aux_row_to_token is not None:
            # [MEM-P3] Captured aux rows are rank-major, not in global token order.
            # Move them off `hidden_states` (post_forward_mlp_sync_batch prefix-
            # slices that field, which would drop live rank-major rows) and publish
            # the row -> global token map so the DSpark KV injector permutes
            # cache_loc/positions instead of the activation itself.
            output.hidden_states_rank_major = output.hidden_states
            output.hidden_states_row_token_index = aux_row_to_token
            output.hidden_states = None
        if aux_tail_gathered is not None:
            output.hidden_states_rank_major = aux_tail_gathered[0]
            output.hidden_states_row_token_index = aux_tail_gathered[1]
            output.hidden_states = None
        if dspark_commit_mask is not None:
            # [fusion-opt] rows of the (gathered) aux that carry a computed aux, and
            # an upper bound on their count (CPU-known, avoids a nonzero sync)
            output.dspark_commit_mask = dspark_commit_mask
            output.dspark_commit_rows_ub = forward_batch.dspark_commit_rows_ub
        return output

    def _execute_idle(
        self, forward_batch: ForwardBatch, pp_proxy_tensors=None
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        model_runner = self.model_runner
        # Padded idle (DP-attn MLP sync) needs metadata reinit; unpadded must
        # drop stale forward_metadata to avoid an SWA use-after-free on req_pool.
        if forward_batch.batch_size > 0:
            if not self.enable_pdmux:
                forward_batch = self.load_batch(forward_batch, pp_proxy_tensors)
            model_runner.attn_backend.init_forward_metadata(forward_batch)
        else:
            model_runner.attn_backend.forward_metadata = None

        kwargs = model_runner._pp_kwargs(pp_proxy_tensors)
        with device_timer_ctx(model_runner.device_timer, "idle"):
            return model_runner.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            )
