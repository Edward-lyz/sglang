import logging
from contextlib import nullcontext
from typing import Optional

import torch

from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    compute_position,
)
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.draft_swa_pool import (
    DraftWindowSlotMap,
    compact_draft_pool_multiplier,
    compute_compact_draft_pool_slots,
)
from sglang.srt.speculative.draft_worker_common import (
    build_block_pos_offsets,
    build_draft_tp_worker,
    make_draft_block_spec_info,
    make_draft_sampler_capture_hook,
)
from sglang.srt.speculative.dspark_components.dspark_config import (
    DSV4_DRAFT_ATTENTION_BACKEND,
    draft_is_deepseek_v4,
    resolve_runtime_config,
)
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftBlockProposer,
    make_next_draft_input,
    maybe_build_draft_sampler,
)
from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
    TargetHiddenKvInjector,
)
from sglang.srt.speculative.dspark_components.dspark_observability import (
    DsparkStepObservers,
    InfoSegment,
)
from sglang.srt.speculative.dspark_components.dspark_planner import (
    DSparkVerifyPlanner,
    alloc_verify_window,
    dp_global_verify_tier_num_tokens,
    idle_ragged_layout,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    CommitInjectCtx,
    DsparkVerifyEpilogue,
    TargetVerifyExecutor,
    verify_logits_adjustments_are_noop,
)
from sglang.srt.speculative.spec_utils import (
    GrammarTree,
    build_grammar_vocab_mask,
    draft_tp_context,
    prepare_mamba_track_for_verify,
)
from sglang.srt.utils import get_available_gpu_memory, is_cuda

logger = logging.getLogger(__name__)


class DSparkWorkerV2(BaseSpecWorker):

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        ps: ParallelState,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.ps = ps
        self.nccl_port = nccl_port
        self._target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.page_size = server_args.page_size
        self.device = target_worker.device

        self._draft_is_moe = draft_is_deepseek_v4(server_args=server_args)
        # Draft-side SWA: a private compact req_to_token over the target's global
        # KV index space, so draft attention only sees the recent window. None =
        # full context (no windowing). Normalized in speculative_hook.
        self.draft_window_size: Optional[int] = (
            server_args.speculative_draft_window_size
        )
        self.use_compact_draft_cache = self.draft_window_size is not None
        # Window-bounded draft KV: assigned in alloc_memory_pool, once the pool
        # geometry is known. Only meaningful with windowing enabled.
        self._draft_slot_map: Optional[DraftWindowSlotMap] = None
        self._draft_dp_context_enabled = (
            server_args.enable_dp_attention and not self._draft_is_moe
        )
        self._is_pd_prefill = server_args.disaggregation_mode == "prefill"
        self._decode_graph_allowed = (
            not server_args.disable_cuda_graph and not self._is_pd_prefill
        )
        if (
            server_args.enable_dp_attention
            and self._draft_is_moe
            and ps.attn_tp_size > 1
        ):
            raise ValueError(
                "DSpark + dp attention with a DeepSeek-V4 (MoE) draft requires "
                "attn_tp == 1 (set --dp-size == --tp). attn_tp > 1 corrupts the "
                "MoE-under-DP all-reduce."
            )

        with self._draft_context():
            bundle = build_draft_tp_worker(
                server_args=server_args,
                gpu_id=gpu_id,
                ps=ps,
                nccl_port=nccl_port,
                target_model_config=target_worker.model_runner.model_config,
                algo_label="DSPARK",
                attention_backend_override=(
                    DSV4_DRAFT_ATTENTION_BACKEND if self._draft_is_moe else None
                ),
            )
        self._draft_worker = bundle.draft_worker
        self.draft_model_runner = bundle.draft_model_runner
        self.draft_model = bundle.draft_model
        self._draft_sampler = None

        runtime_config = resolve_runtime_config(
            draft_hf_config=self.draft_model_runner.model_config.hf_config,
            speculative_num_draft_tokens=server_args.speculative_num_draft_tokens,
            target_vocab_size=int(
                self.target_worker.model_runner.model_config.vocab_size
            ),
        )
        self.gamma = runtime_config.gamma
        self.verify_num_draft_tokens = runtime_config.verify_num_draft_tokens
        self.speculative_num_draft_tokens = self.verify_num_draft_tokens
        self._mask_token_id = runtime_config.mask_token_id
        self._pp_context_feature_indices: Optional[list[int]] = None

        if self.ps.tp_rank == 0:
            logger.info(
                "Initialized DSpark draft runner. attention_backend=%s, model=%s, "
                "gamma=%s, verify_num_draft_tokens=%s, mask_token_id=%s, "
                "markov_head=%s",
                bundle.resolved_attention_backend,
                self.draft_model.__class__.__name__,
                self.gamma,
                self.verify_num_draft_tokens,
                self._mask_token_id,
                type(self.draft_model.markov_head).__name__,
            )

        self._block_pos_offsets = build_block_pos_offsets(
            length=self.verify_num_draft_tokens, device=self.device
        )
        self._draft_block_spec_info = make_draft_block_spec_info(
            draft_token_num=int(self.gamma), device=self.device
        )

        target_model = self.target_worker.model_runner.model
        lm_head = getattr(target_model, "lm_head", None)
        needs_lm_head = not (
            self._is_pd_prefill
            and self.ps.pp_size > 1
            and self.ps.pp_rank + 1 < self.ps.pp_size
            and not self._draft_is_moe
        )
        if needs_lm_head and (lm_head is None or not hasattr(lm_head, "weight")):
            raise RuntimeError(
                "DSpark requires the target model to expose `lm_head` with `weight`."
            )
        self.draft_model.attach_shared_modules(
            embed_tokens=self._resolve_target_embed_tokens(target_model),
            lm_head=lm_head,
        )
        # Set by attach_shared_modules: a shared target embed / lm_head that
        # shards its vocab over the global TP group (dp attention without
        # --enable-dp-lm-head, or SGLANG_ENABLE_EMBED_DP_SHARD) makes the draft
        # gather rows across attention-DP ranks, so every rank must join.
        self._dp_vocab_sync = bool(
            getattr(self.draft_model, "lm_head_dp_gather", False)
            or getattr(self.draft_model, "embed_dp_gather", False)
        )
        # Largest captured draft row count: the capacity every DP-gathered vocab
        # collective pads to when the batch fits, so graph replays (which bake
        # it) and eager / idle joins always agree on the shape.
        self._dp_graph_rows = (
            max(server_args.cuda_graph_config.decode.bs) * int(self.gamma)
            if self._dp_vocab_sync and self._decode_graph_allowed
            else None
        )
        self.draft_model.dp_embed_rows_capacity = self._dp_graph_rows

        self._verify_planner = DSparkVerifyPlanner(
            draft_model=self.draft_model,
            gamma=self.gamma,
            model_runner=self.model_runner,
            device=self.device,
            tp_rank=self.ps.tp_rank,
            server_args=self.server_args,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
        )
        if (
            server_args.enable_dp_attention
            and not self._draft_is_moe
            and self._verify_planner.is_compact_mode
            and self._decode_graph_allowed
        ):
            raise ValueError(
                "DSpark dense-draft compact verify under --enable-dp-attention does not "
                "yet support cuda graph (idle DP groups cannot join the token-keyed "
                "compact graph). Re-run with --disable-cuda-graph (eager is lossless), "
                "or use SGLANG_RAGGED_VERIFY_MODE=static. The dsv4 (MoE) draft supports "
                "cuda graph under DP."
            )
        self._kv_injector = TargetHiddenKvInjector(
            draft_model=self.draft_model,
            draft_model_runner=self.draft_model_runner,
            model_runner=self.model_runner,
            device=self.device,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            block_pos_offsets=self._block_pos_offsets,
            # Resolved lazily: the slot map is built in alloc_memory_pool, after
            # the injector. None until then, and None forever when windowing is
            # off -- in which case writes stay in the global index space.
            slot_translator=self._translate_draft_slots,
        )
        self._proposer = DraftBlockProposer(
            draft_model=self.draft_model,
            draft_model_runner=self.draft_model_runner,
            gamma=self.gamma,
            mask_token_id=self._mask_token_id,
            draft_block_spec_info=self._draft_block_spec_info,
            dp_moe_sync=self._draft_is_moe and server_args.enable_dp_attention,
            dp_vocab_sync=self._dp_vocab_sync,
            dp_graph_rows=self._dp_graph_rows,
            draft_window_size=self.draft_window_size,
            page_size=self.page_size,
            resolve_target_req_to_token=lambda: (
                self.model_runner.req_to_token_pool.req_to_token
            ),
            # Resolved per call: the slot map does not exist yet (it is built in
            # alloc_memory_pool), so passing the value here would capture None.
            resolve_slot_translator=lambda: self._translate_draft_slots_or_none,
        )
        self._verify_epilogue = None
        if (
            self._verify_planner.is_compact_mode
            and self._decode_graph_allowed
            and is_cuda()
        ):
            self._verify_epilogue = DsparkVerifyEpilogue(
                max_bs=max(server_args.cuda_graph_config.decode.bs),
                verify_num_draft_tokens=self.verify_num_draft_tokens,
                device=self.device,
                commit_ctx=CommitInjectCtx(
                    draft_model=self.draft_model,
                    block_pos_offsets=self._block_pos_offsets,
                    resolve_pool=lambda: self.draft_model_runner.token_to_kv_pool,
                    resolve_req_to_token=lambda: (
                        self.model_runner.req_to_token_pool.req_to_token
                    ),
                ),
            )
            self.model_runner.capture_tail_hooks.append(
                self._verify_epilogue.capture_hook
            )

        self._simulate_acc_len = float(envs.SGLANG_SIMULATE_ACC_LEN.get())
        if (
            self._simulate_acc_len > 0
            and self._simulate_acc_len != 1.0
            and not self._verify_planner.is_verify_all
        ):
            raise ValueError(
                "SGLANG_SIMULATE_ACC_LEN>1.0 with DSpark requires a verify-all "
                "schedule (SGLANG_RAGGED_VERIFY_MODE=static, or =compact with the "
                "uninitialized/flat SPS table): a constant simulated correct_len>0 "
                "can exceed a trimmed request's verify budget (cap-accept, or "
                "compact with a profiled SPS table) and break the cutoff/cap "
                "accounting. SGLANG_SIMULATE_ACC_LEN=1.0 yields correct_len=0 "
                "(commit is the bonus token only), which stays within every verify "
                "budget and is safe in any mode. Got mode="
                f"{self._verify_planner.mode_value!r}, simulate_acc_len="
                f"{self._simulate_acc_len}."
            )

        self._verify_executor = TargetVerifyExecutor(
            target_worker=self.target_worker,
            gamma=self.gamma,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            model_runner=self.model_runner,
            kv_injector=self._kv_injector,
            verify_epilogue=self._verify_epilogue,
            simulate_acc_len=self._simulate_acc_len,
        )

        self._forced_budget_frac: Optional[float] = None
        self._need_mamba_verify_commit = False

        self._observers = DsparkStepObservers(
            planner=self._verify_planner,
            gamma=self.gamma,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            tp_rank=self.ps.tp_rank,
            device=self.device,
            simulate_acc_len=self._simulate_acc_len,
        )

        if self._is_pd_prefill and not self._draft_is_moe:
            self.draft_model.prune_to_ctx_kv_injection()
            self._init_pp_context_feature_indices()

    @property
    def _draft_weight_sharded(self) -> bool:
        # Runtime weight sharding adds a node-local AllGather to the dense
        # draft's forward. Idle DP ranks must still run that forward (empty
        # batch) so all shard-group members enter the AG; probe live because
        # prune_to_ctx_kv_injection() drops the shard bookkeeping.
        return bool(getattr(self.draft_model, "_weight_shard_layers", None))

    def _resolve_target_embed_tokens(self, target_model):
        if hasattr(target_model, "get_input_embeddings"):
            return target_model.get_input_embeddings()
        return target_model.model.get_input_embeddings()

    def _init_pp_context_feature_indices(self) -> None:
        if self.ps.pp_size <= 1:
            return
        if not hasattr(self.draft_model, "project_target_hidden_partial"):
            return

        target_layer_ids = getattr(
            self.model_runner.spec_aux_config, "dflash_target_layer_ids", None
        )
        if not target_layer_ids:
            return

        target_model = self.target_worker.model_runner.model
        start_layer = int(getattr(target_model, "start_layer", 0))
        end_layer = int(getattr(target_model, "end_layer", 0))
        target_layer_to_feature = {
            int(layer_id): idx for idx, layer_id in enumerate(target_layer_ids)
        }
        local_feature_indices = [
            target_layer_to_feature[layer_id]
            for layer_id in range(start_layer, end_layer)
            if layer_id in target_layer_to_feature
        ]
        if not local_feature_indices:
            return

        self._pp_context_feature_indices = local_feature_indices
        if self.ps.tp_rank == 0:
            logger.info(
                "DSpark PP-local context projection will accumulate feature indices %s "
                "from target layers %s for PP rank %s local layers [%s, %s).",
                local_feature_indices,
                list(target_layer_ids),
                self.ps.pp_rank,
                start_layer,
                end_layer,
            )

    @property
    def carries_confidence(self) -> bool:
        return self._verify_planner.carries_confidence

    @property
    def spec_v2_attn_backends(self) -> tuple:
        return (
            self._target_worker.model_runner.attn_backend,
            self.draft_model_runner.attn_backend,
        )

    def __getattr__(self, name):
        if name == "_target_worker":
            raise AttributeError(name)
        return getattr(self.target_worker, name)

    def _draft_context(self):
        if self._draft_dp_context_enabled:
            return draft_tp_context(get_parallel().attn_tp_group)
        return nullcontext()

    def _vocab_graph_capture_context(self):
        """Put the lm_head shard group's pynccl comm in graph mode for the draft
        capture.

        The draft graphs are captured with the ambient TP group patched to
        attn-TP, so the module-level ``graph_capture()`` only switches THAT
        group into graph mode (pynccl enabled). The DP-gathered lm-head
        collectives run on the shard group (the global TP group), whose pynccl
        comm would stay disabled and fall back to torch.distributed -- which is
        not capturable and stalls the capture.
        """
        if not self._dp_vocab_sync:
            return nullcontext()
        group = getattr(self.draft_model, "lm_head_group", None)
        pynccl_comm = getattr(group, "pynccl_comm", None)
        if pynccl_comm is None:
            return nullcontext()
        return pynccl_comm.change_state(enable=True)

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        if (
            memory_pool_config is not None
            and self._is_pd_prefill
            and not self._draft_is_moe
            and not self.use_compact_draft_cache
            and self.ps.pp_rank < self.ps.pp_size - 1
        ):
            page_size = int(self.page_size)

            def _minimal_capacity(capacity):
                if capacity is None or capacity == 0:
                    return capacity
                return page_size

            memory_pool_config = MemoryPoolConfig(
                max_total_num_tokens=page_size,
                max_running_requests=memory_pool_config.max_running_requests,
                full_max_total_num_tokens=_minimal_capacity(
                    memory_pool_config.full_max_total_num_tokens
                ),
                swa_max_total_num_tokens=_minimal_capacity(
                    memory_pool_config.swa_max_total_num_tokens
                ),
                mem_fraction_static=memory_pool_config.mem_fraction_static,
            )
        # Without draft windowing, the draft worker aliases the target
        # request->token mapping and KV allocator. With windowing enabled, it
        # keeps a private compact req->token table over the target's global KV
        # index space AND a private, window-sized KV allocator: the compact
        # table stores draft-SWA ids (see draft_swa_pool.py), so the draft's KV
        # buffer only needs `max_running_requests * (window + 2*gamma + page)`
        # rows -- the whole point of the change. Passing None here forces the
        # draft's kv_cache_configurator to build a fresh, smaller allocator
        # from the shrunk MemoryPoolConfig below rather than aliasing target's.
        if self.use_compact_draft_cache and memory_pool_config is not None:
            memory_pool_config = self._shrink_memory_pool_config_to_window(
                memory_pool_config
            )
            token_to_kv_pool_allocator = None
        self._draft_worker.alloc_memory_pool(
            memory_pool_config=memory_pool_config,
            req_to_token_pool=(
                None if self.use_compact_draft_cache else req_to_token_pool
            ),
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )
        self._init_draft_slot_map()

    def _shrink_memory_pool_config_to_window(
        self, memory_pool_config: MemoryPoolConfig
    ) -> MemoryPoolConfig:
        """Resize the draft's KV pool from context-length down to window-length.

        The draft only ever addresses the swa-id space (see draft_swa_pool.py):
        ids fit in ``[1, num_swa_slots)`` and any id outside that range is a bug
        the injector catches. Sizing the physical KV buffer to
        ``num_swa_slots`` -- rather than the target's context-length pool --
        turns the index compression into an actual memory saving.

        Never grow the config. Two cases must survive:
          * a config quirk (small target, tiny max_reqs) that recommends more
            slots than the target has -- the window buys nothing;
          * the PD-prefill branch above already shrunk to ``page_size`` -- that
            branch minimises a mostly-unused pool, so growing it back to
            ``swa_size`` here would undo the intent.
        """
        # In PD mode Prefill is a producer of full prompt KV. Its temporary
        # source mappings live until RDMA completion and may overlap across
        # transfer rooms, so sizing this pool by the Decode window is unsafe.
        # Keep the original Prefill draft pool; Decode remains window-compact.
        if self.__dict__.get("_is_pd_prefill", False):
            return memory_pool_config

        max_running_requests = memory_pool_config.max_running_requests or int(
            self.model_runner.req_to_token_pool.req_to_token.shape[0]
        )
        page_size = int(self.page_size)
        # The recommendation is already the physical pool size: it is a whole
        # number of pages and includes the reserved page 0 (whose slot 0 is the
        # unmapped sentinel). The KV pool must stay page-aligned so trtllm_mha's
        # ``k_cache.view(-1, page_size, num_kv_heads, head_dim)`` reshape stays
        # valid; a mid-page pool size makes that view raise a shape mismatch.
        # compute_compact_draft_pool_slots is the single source of truth, shared
        # with the target-side pool budget (pool_configurator) so the Decode pool
        # and the reserved budget can never disagree. Its multiplier defaults to
        # the shared env headroom (SGLANG_DSPARK_SWA_POOL_MULTIPLIER).
        swa_pool_size = compute_compact_draft_pool_slots(
            max_num_reqs=max_running_requests,
            window_size=self.draft_window_size,
            gamma=int(self.verify_num_draft_tokens),
            page_size=page_size,
        )
        if swa_pool_size >= memory_pool_config.max_total_num_tokens:
            return memory_pool_config
        if self.ps.tp_rank == 0:
            logger.info(
                "DSpark draft KV pool: shrinking %d -> %d slots (%.1fx) "
                "for window=%d, gamma=%d, max_reqs=%d.",
                memory_pool_config.max_total_num_tokens,
                swa_pool_size,
                memory_pool_config.max_total_num_tokens / max(1, swa_pool_size),
                self.draft_window_size,
                int(self.verify_num_draft_tokens),
                max_running_requests,
            )
        return MemoryPoolConfig(
            max_total_num_tokens=swa_pool_size,
            max_running_requests=memory_pool_config.max_running_requests,
            full_max_total_num_tokens=memory_pool_config.full_max_total_num_tokens,
            swa_max_total_num_tokens=memory_pool_config.swa_max_total_num_tokens,
            mem_fraction_static=memory_pool_config.mem_fraction_static,
        )

    def _init_draft_slot_map(self) -> None:
        """Build the window-bounded draft SWA slot map (windowed mode only).

        Sized from the window, not the context length: that is what keeps the
        draft KV footprint bounded as sequences grow.

        The map's ``num_swa_slots`` MUST NOT exceed the draft pool's addressable
        slot count (``draft.token_to_kv_pool_allocator.size + page_size``). If
        the map hands out a slot id past the pool, ``store_kvcache`` asserts
        ``index < size_limit`` on the device -- the assertion fires
        asynchronously from a later Mamba free, making the true cause hard to
        find. Bound the map to what the pool can actually hold.
        """
        if not self.use_compact_draft_cache:
            return
        assert self.draft_window_size is not None

        target_req_to_token = self.model_runner.req_to_token_pool.req_to_token
        max_num_reqs = int(target_req_to_token.shape[0])
        target_allocator = self.model_runner.token_to_kv_pool_allocator
        # SWA allocators expose ``size`` as the compact physical pool size, but
        # req_to_token stores ids from the complete target slot address space.
        # DraftWindowSlotMap must therefore be indexed by size_full.
        size_full = getattr(target_allocator, "size_full", None)
        num_full_slots = int(
            size_full if size_full is not None else target_allocator.size
        )
        # The draft worker rebuilt its allocator against the shrunk pool
        # (`_shrink_memory_pool_config_to_window`); read its size, not the
        # target's, so the map's id space stays within the actual buffer.
        draft_allocator = self.draft_model_runner.token_to_kv_pool_allocator
        draft_pool_size = int(draft_allocator.size)
        page_size = int(self.page_size)
        # The recommended size is already the physical pool size (a whole number
        # of pages, including the reserved page 0). Bound it to the actual draft
        # pool and floor-align to page_size so DraftWindowSlotMap's page-space
        # invariant holds even if the allocator reports a mid-page size.
        # The recommended size is already the physical pool size (a whole number
        # of pages, including the reserved page 0). Scale it by the SAME pool
        # multiplier as the physical buffer (_shrink_memory_pool_config_to_window)
        # so the slot map is never the binding constraint, then bound it to the
        # actual draft pool and floor-align to page_size so DraftWindowSlotMap's
        # page-space invariant holds even if the allocator reports a mid-page size.
        recommended_swa_slots = int(
            DraftWindowSlotMap.recommended_num_swa_slots(
                max_num_reqs=max_num_reqs,
                window_size=self.draft_window_size,
                gamma=int(self.verify_num_draft_tokens),
                page_size=page_size,
            )
            * compact_draft_pool_multiplier()
        )
        num_swa_slots = (
            min(recommended_swa_slots, draft_pool_size) // page_size
        ) * page_size
        # The map is gathered against ``cache_loc`` / ``req_to_token``, both of
        # which live on the draft device; a CPU table would raise a device
        # mismatch on the first inject.
        self._draft_slot_map = DraftWindowSlotMap(
            num_full_slots=num_full_slots,
            num_swa_slots=num_swa_slots,
            max_num_reqs=max_num_reqs,
            window_size=self.draft_window_size,
            page_size=int(self.page_size),
            device=self.device,
        )
        draft_pool = getattr(self.draft_model_runner, "token_to_kv_pool", None)
        if draft_pool is not None:
            # PD queues receive the raw draft pool from kv_cache_builder. Expose
            # the compact map on that pool so both prefill source translation
            # and decode destination allocation use the same worker-owned map.
            draft_pool.slot_map = self._draft_slot_map
            draft_pool.full_to_swa_index_mapping = self._draft_slot_map.full_to_swa
        if self.ps.tp_rank == 0:
            logger.info(
                "DSpark draft SWA slot map: window=%d, gamma=%d, swa_slots=%d "
                "(%d per request) over %d global slots "
                "(allocator.size=%d, allocator.size_full=%s).",
                self.draft_window_size,
                self.verify_num_draft_tokens,
                num_swa_slots,
                num_swa_slots // max(1, max_num_reqs),
                num_full_slots,
                int(target_allocator.size),
                str(size_full),
            )

    def init_attention_backends(self):
        with self._draft_context():
            self._draft_worker.init_attention_backends()
        self._need_mamba_verify_commit = mambaish_config(
            self.model_runner.model_config
        ) is not None and hasattr(
            self.model_runner.attn_backend,
            "update_mamba_state_after_mtp_verify",
        )

    def init_cuda_graphs(self):
        capture_decode_cuda_graph = self._decode_graph_allowed
        if is_cuda() and capture_decode_cuda_graph:
            available_mem = get_available_gpu_memory(self.device, self.gpu_id)
            if available_mem < 1.0:
                capture_decode_cuda_graph = False
                logger.warning(
                    "Disable DSpark draft cuda graph because only %.2f GB GPU "
                    "memory is available after target backend initialization.",
                    available_mem,
                )
        with self._draft_context(), self._vocab_graph_capture_context():
            if capture_decode_cuda_graph:
                self._draft_sampler = self._maybe_build_draft_sampler()
                if self._draft_sampler is not None:
                    self.draft_model_runner.capture_tail_hooks.append(
                        make_draft_sampler_capture_hook(self._draft_sampler)
                    )
                self._proposer.attach_draft_sampler(self._draft_sampler)
            self._draft_worker.init_cuda_graphs(
                capture_decode_cuda_graph=capture_decode_cuda_graph
            )

    def _maybe_build_draft_sampler(self):
        return maybe_build_draft_sampler(
            draft_model=self.draft_model,
            gamma=self.gamma,
            max_bs=max(self.server_args.cuda_graph_config.decode.bs),
            device=self.device,
            tp_rank=self.ps.tp_rank,
            confidence_fn=(
                self._verify_planner.compute_confidence_tensor
                if self._verify_planner.carries_confidence
                else None
            ),
            out=(
                self._verify_epilogue.draft_tokens_buf
                if self._verify_epilogue is not None
                else None
            ),
        )

    def clear_cache_pool(self):
        pass

    def set_dspark_forced_budget_frac(self, frac: Optional[float]) -> None:
        self._forced_budget_frac = frac
        self._verify_planner.set_forced_budget_frac(frac)

    def dump_info_records(self) -> Optional[dict]:
        return self._observers.dump_info_records()

    def clear_info_records(self) -> None:
        self._observers.clear_info_records()

    def block_accept_estimate_log_suffix(self) -> Optional[str]:
        return self._observers.block_accept_estimate_log_suffix()

    def note_request_finished(self, *, rid: str, natural_stop: bool) -> None:
        self._observers.note_request_finished(rid=rid, natural_stop=natural_stop)

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish=None,
        grammar_barrier=None,
        pp_proxy_tensors=None,
    ) -> GenerationBatchResult:
        if getattr(batch, "return_logprob", False):
            raise ValueError(
                "DSpark speculative decoding does not support return_logprob yet."
            )

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            self._verify_planner.note_non_decode_step()
            self._observers.note_prefill_step()
            return self._forward_prefill(batch, on_publish, pp_proxy_tensors)

        return self._forward_decode(batch, on_publish, grammar_barrier)

    def _forward_prefill(
        self, batch: ScheduleBatch, on_publish, pp_proxy_tensors=None
    ) -> GenerationBatchResult:
        if batch.forward_mode.is_idle():
            if self.server_args.enable_dp_attention:
                self.target_worker.forward_batch_generation(
                    batch,
                    pp_proxy_tensors=pp_proxy_tensors,
                    capture_hidden_mode=CaptureHiddenMode.FULL,
                )
            return self._decode_idle_result(on_publish=on_publish)

        batch_output = self.target_worker.forward_batch_generation(
            batch,
            pp_proxy_tensors=pp_proxy_tensors,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )
        logits_output = batch_output.logits_output
        output_pp_proxy_tensors = batch_output.pp_hidden_states_proxy_tensors
        # The target aux hidden is consumed locally on this rank only: it feeds
        # this rank's context projection (dspark_ctx_acc), which is the sole
        # DSpark tensor the next PP rank reads. Pop it from the outgoing proxy
        # so it is not redundantly shipped across the PP wire (the downstream
        # rank reads hidden_states / residual / dspark_ctx_acc, never this).
        target_hidden = (
            logits_output.hidden_states
            if logits_output is not None
            else (
                output_pp_proxy_tensors.tensors.pop("dspark_aux_hidden_states", None)
                if output_pp_proxy_tensors is not None
                else None
            )
        )
        next_token_ids = batch_output.next_token_ids
        batch_output.new_seq_lens = batch.seq_lens
        if on_publish is not None:
            on_publish(batch_output.new_seq_lens)

        if target_hidden is None and self.ps.pp_size > 1:
            target_hidden = torch.empty(
                (0, 0), dtype=torch.float16, device=self.device
            )
        if target_hidden is None:
            raise RuntimeError(
                "DSpark requires target aux hidden capture for prefill, but got None. "
                "Make sure the target model has DFlash layers-to-capture configured."
            )
        has_local_target_hidden = target_hidden.numel() > 0
        if batch.extend_lens is None or batch.prefix_lens is None:
            raise RuntimeError(
                "DSpark expected extend_lens / prefix_lens in extend mode, got None."
            )
        if batch.out_cache_loc is None:
            raise RuntimeError("DSpark prefill expected out_cache_loc, but got None.")

        # Must inject before prefill returns: the scheduler may update radix
        # afterward, invalidating out_cache_loc.
        # Non-last PP ranks return PPProxyTensors and do not sample tokens, so
        # next_token_ids is None there. Positions are still needed for local
        # draft-KV injection, and should live on this worker's device.
        device = self.device
        num_real_tokens = int(sum(batch.extend_lens))
        # Build on host first, then upload with non_blocking=True (same idiom as
        # ForwardBatch.init_new). Passing device= to torch.tensor() makes PyTorch
        # take the pageable-H2D path, which appends a cudaStreamSynchronize: the
        # host would block until the whole target forward drains, leaving the GPU
        # queue empty for the PP consensus-ring waits that follow.
        ctx_lens = torch.tensor(batch.extend_lens, dtype=torch.int32).to(
            device, non_blocking=True
        )
        draft_seq_lens = torch.tensor(batch.prefix_lens, dtype=torch.int32).to(
            device, non_blocking=True
        )
        positions, _ = compute_position(
            self.model_runner.server_args.attention_backend,
            draft_seq_lens,
            ctx_lens,
            num_real_tokens,
        )
        # mlp-sync (DP attention / MoE a2a) pads extend batches to a multiple of
        # attn_tp_size, but only the last PP rank un-pads its output
        # (post_forward_mlp_sync_batch trims logits_output.hidden_states) —
        # earlier ranks publish the padded rows. Trim both sides to the real
        # tokens so the cross-stage accumulation shares one row convention.
        incoming_ctx = (
            pp_proxy_tensors.tensors.get("dspark_ctx_acc")
            if pp_proxy_tensors is not None
            else None
        )
        if incoming_ctx is not None and incoming_ctx.shape[0] > num_real_tokens:
            incoming_ctx = incoming_ctx[:num_real_tokens]
        local_ctx = None
        if (
            self.ps.pp_size > 1
            and has_local_target_hidden
            and self._pp_context_feature_indices is not None
        ):
            local_ctx = self.draft_model.project_target_hidden_partial(
                target_hidden, self._pp_context_feature_indices
            )
            if local_ctx.shape[0] > num_real_tokens:
                local_ctx = local_ctx[:num_real_tokens]
        ctx_acc = None
        if incoming_ctx is not None and local_ctx is not None:
            ctx_acc = (
                incoming_ctx.to(device=local_ctx.device, dtype=local_ctx.dtype)
                + local_ctx
            )
        elif incoming_ctx is not None:
            ctx_acc = incoming_ctx.to(device=self.device, non_blocking=True)
        elif local_ctx is not None:
            ctx_acc = local_ctx

        # Prefill writes draft KV at freshly-allocated slots, so they need a
        # draft-SWA mapping before the write. Decode maps inside
        # _maintain_draft_swa_window; prefill has no such pass. Only the
        # trailing `window` tokens of each extend segment are kept: the draft's
        # self-attention is window-truncated, so older extend tokens are never
        # read and must not consume draft-SWA slots (the pool is sized by the
        # window, not by the extend length).
        slot_map = self._draft_slot_map
        if slot_map is not None:
            keep = torch.zeros(num_real_tokens, dtype=torch.bool, device=device)
            off = 0
            for ext_len in batch.extend_lens:
                ext_len = int(ext_len)
                keep_len = min(ext_len, int(self.draft_window_size))
                if keep_len > 0:
                    keep[off + ext_len - keep_len : off + ext_len] = True
                off += ext_len
            prefill_cache_loc = batch.out_cache_loc[keep]
            prefill_positions = positions[keep]
        else:
            keep = None
            prefill_cache_loc = batch.out_cache_loc
            prefill_positions = positions

        # Prefill draft KV is written locally before PD transfer, so its active
        # tail must be mapped before injection. The per-request mapper records
        # ownership; Mooncake recycles these source slots after RDMA completes.
        self._map_prefill_draft_slots(batch=batch, cache_loc=prefill_cache_loc)

        if output_pp_proxy_tensors is not None:
            # Non-last PP ranks forward the FULL accumulated context to the next
            # rank; only the rank that actually injects KV trims to the window.
            if ctx_acc is not None:
                output_pp_proxy_tensors.tensors["dspark_ctx_acc"] = ctx_acc
        elif ctx_acc is not None:
            if keep is not None:
                ctx_acc = ctx_acc[keep]
            self._kv_injector.inject_projected_context(
                projected_context=ctx_acc,
                cache_loc=prefill_cache_loc,
                positions=prefill_positions,
            )
        elif has_local_target_hidden and not (self.ps.pp_size > 1 and not self._draft_is_moe):
            if keep is not None and target_hidden.shape[0] == num_real_tokens:
                target_hidden = target_hidden[keep]
            self._kv_injector.inject_target_hidden(
                target_hidden=target_hidden,
                cache_loc=prefill_cache_loc,
                positions=prefill_positions,
            )
        # Avoid copying large hidden-state buffers to CPU in overlap scheduling.
        if logits_output is not None:
            logits_output.hidden_states = None

        if next_token_ids is not None:
            batch_output.next_draft_input = make_next_draft_input(
                bonus_tokens=next_token_ids,
                new_seq_lens=batch.seq_lens,
            )
        return batch_output

    def _idle_verify_ragged_layout(self, batch: ScheduleBatch):
        if batch.global_num_tokens is None or not self._verify_planner.is_compact_mode:
            return None
        global_bs = max(batch.global_num_tokens)
        if global_bs <= 0:
            return None
        return idle_ragged_layout(
            tier_num_reqs=global_bs,
            dp_tier_num_tokens=self._dp_verify_tier_num_tokens(batch),
            device=self.device,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            model_runner=self.model_runner,
        )

    def _dp_verify_tier_num_tokens(self, batch: ScheduleBatch) -> Optional[int]:
        if not (
            self._draft_is_moe
            and self.server_args.enable_dp_attention
            and batch.global_num_tokens is not None
            and self._verify_planner.is_compact_mode
        ):
            return None
        return dp_global_verify_tier_num_tokens(
            global_tier_num_tokens=batch.global_spec_verify_tier_num_tokens
        )

    def _decode_idle_result(
        self,
        *,
        on_publish,
    ) -> GenerationBatchResult:
        next_draft_input = make_next_draft_input(
            bonus_tokens=torch.empty((0,), device=self.device, dtype=torch.int64),
            new_seq_lens=torch.empty((0,), device=self.device, dtype=torch.int64),
        )
        if on_publish is not None:
            on_publish(next_draft_input.new_seq_lens)
        return GenerationBatchResult(
            logits_output=None,
            next_token_ids=torch.empty((0,), dtype=torch.int64, device=self.device),
            accept_lens=torch.empty((0,), dtype=torch.int32, device=self.device),
            block_accept_lens=torch.empty((0,), dtype=torch.int32, device=self.device),
            next_draft_input=next_draft_input,
            can_run_cuda_graph=False,
            speculative_num_draft_tokens=int(self.verify_num_draft_tokens),
            new_seq_lens=next_draft_input.new_seq_lens,
        )

    def _map_prefill_draft_slots(self, *, batch: ScheduleBatch, cache_loc) -> None:
        """Map the slots prefill is about to write draft KV into.

        ``cache_loc`` is the window-trimmed write list prepared by
        ``_forward_prefill``: one tail window per extend segment, because the
        draft's window-truncated attention never reads the rest. Mapping
        ``batch.out_cache_loc`` instead would remap the whole prompt and exhaust
        the compact pool under long-context PD traffic.
        """
        slot_map = self._draft_slot_map
        if slot_map is None or batch.out_cache_loc is None:
            return
        # Map per request, carrying the owner id into the map.  A flattened
        # batch mapping cannot be released safely because the allocator then
        # has no request high-water mark to sweep on completion.
        if batch.extend_lens is None or batch.prefix_lens is None:
            slot_map.map_slots(cache_loc)
            return
        # Chunked prefill may call this repeatedly for the same request; evict
        # the previous chunk before retaining the new tail.
        compact_offset = 0
        for req, prefix_len, extend_len in zip(
            batch.reqs, batch.prefix_lens, batch.extend_lens
        ):
            ext_len = int(extend_len)
            keep_len = min(ext_len, int(self.draft_window_size))
            if keep_len > 0:
                committed = int(prefix_len) + ext_len
                slot_map.advance_window(
                    req_pool_indices_cpu=[int(req.req_pool_idx)],
                    committed_lens_cpu=[committed],
                    target_req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                )
                req_loc = cache_loc[compact_offset : compact_offset + keep_len]
                slot_map.map_slots(
                    req_loc,
                    req_pool_idx=int(req.req_pool_idx),
                    end_pos=committed,
                )
            compact_offset += keep_len

    def _translate_draft_slots(self, cache_loc):
        """Target-global slot ids -> draft-SWA slot ids; identity when off."""
        slot_map = self._draft_slot_map
        if slot_map is None:
            return cache_loc
        return slot_map.translate(cache_loc)

    @property
    def _translate_draft_slots_or_none(self):
        """The translator, or None when windowing is off.

        The compact-row rebuild skips its translation pass entirely on None,
        which keeps the non-windowed path free of an extra gather.
        """
        return None if self._draft_slot_map is None else self._translate_draft_slots

    def _maintain_draft_swa_window(
        self, *, batch: ScheduleBatch, draft_input: DFlashDraftInputV2
    ) -> None:
        """Advance the window, then map every slot the draft may touch this step.

        Both halves run here, on the host, before the draft forward:

        * Eviction uses ``req.kv_committed_len``, which the batch-result processor
          settles on the host, so it lags this step's verify by one. That lag is
          harmless -- the frontier is monotonic and the pool carries a block of
          slack -- and it avoids a D2H sync on ``commit_lens``. It also means the
          target's ``decode_batch_idx >= 1`` guard has no analogue here: the
          length we evict against is one the draft has already read past.
        * Mapping starts at the eviction frontier, never at 0: re-mapping an
          evicted position would silently undo the eviction. It must reach the
          RESERVED length, because ``prepare_for_decode`` over-allocates to
          ``committed + 2 * block_size`` and the draft may gather any of it --
          a reserved-but-unmapped slot would translate to 0 (the unmapped
          sentinel) and read the wrong KV.
        * ``reserved_seq_lens_cpu`` is computed one scheduling step ago (host
          metadata intentionally lags, see ``prepare_for_decode``), off last
          step's ``kv_committed_len``. When a step accepts an unusually long
          draft block, this step's ``kv_committed_len`` can already exceed
          that stale reserved bound. ``alloc_verify_window`` always gathers
          this step's real ``[committed, committed + block)`` regardless, so
          the map must cover at least that -- else the same reserved-but-
          unmapped-slot failure hits on the *current* step's own verify
          write, not just a future over-allocated one. ``committed[i] +
          block`` reads only already-resident host ints, so folding it in
          costs no extra D2H sync.
        """
        slot_map = self._draft_slot_map
        if slot_map is None:
            return
        target_req_to_token = self.model_runner.req_to_token_pool.req_to_token
        reserved = draft_input.reserved_seq_lens_cpu
        block = int(self.verify_num_draft_tokens)

        req_pool_indices = [int(req.req_pool_idx) for req in batch.reqs]
        committed = [int(req.kv_committed_len) for req in batch.reqs]
        slot_map.advance_window(
            req_pool_indices_cpu=req_pool_indices,
            committed_lens_cpu=committed,
            target_req_to_token=target_req_to_token,
        )

        # Batched window-tail mapping: ONE map_slots call per step. The former
        # per-request loop paid a full map_slots chain (~15 small CUDA ops plus
        # torch.unique) per request per step, which alone cost ~0.4ms/request
        # of pure CPU launch overhead -- the bulk of SWA's decode regression.
        window_tail = int(self.draft_window_size) + 2 * block
        starts = []
        ends = []
        active_reqs = []
        for i, req_pool_idx in enumerate(req_pool_indices):
            current_end = committed[i] + block
            end = max(int(reserved[i]), current_end) if reserved is not None else current_end
            # Only the active SWA tail can be read by draft attention.  On the
            # first decode step the eviction frontier is zero, but ``end`` may
            # be the entire prefill context; mapping from zero would allocate
            # one compact page per full-context page and defeat SWA entirely.
            # Keep enough history for the configured window and the reserved
            # verify block, while preserving the monotonic eviction frontier.
            window_start = max(0, end - window_tail)
            start = max(int(slot_map.evict_frontier[req_pool_idx]), window_start)
            if end > start:
                starts.append(start)
                ends.append(end)
                active_reqs.append(req_pool_idx)
        if starts:
            device_ = target_req_to_token.device
            starts_t = torch.tensor(starts, dtype=torch.int64, device=device_)
            lens_t = torch.tensor(
                [e - s for s, e in zip(starts, ends)], dtype=torch.int64, device=device_
            )
            span = int(max(e - s for s, e in zip(starts, ends)))
            pos = torch.arange(span, dtype=torch.int64, device=device_)
            idx = starts_t.unsqueeze(1) + pos.unsqueeze(0)
            keep = pos.unsqueeze(0) < lens_t.unsqueeze(1)
            reqs_t = torch.tensor(active_reqs, dtype=torch.int64, device=device_)
            gathered = target_req_to_token[reqs_t.unsqueeze(1), idx]
            # map_slots is idempotent per id, so one flattened call over every
            # request's [start, end) range is equivalent to the per-request
            # calls it replaces.
            slot_map.map_slots(gathered[keep])
            slot_map.note_mapped_extents(active_reqs, ends)

    def release_draft_kv(self, *, req_pool_idx: int) -> None:
        """Return every draft-SWA slot this request still holds.

        Called from the request-release path once the radix insert has happened
        but before the slot id is recycled, so a reused prefix never inherits a
        stale draft mapping.
        """
        slot_map = self._draft_slot_map
        if slot_map is None:
            return
        slot_map.release_request(
            req_pool_idx=req_pool_idx,
            target_req_to_token=self.model_runner.req_to_token_pool.req_to_token,
        )

    def _forward_decode(
        self, batch: ScheduleBatch, on_publish, grammar_barrier=None
    ) -> GenerationBatchResult:
        if batch.spec_info is None:
            batch.spec_info = DFlashDraftInputV2.create_idle_input(device=self.device)
        draft_input = batch.spec_info
        if not isinstance(draft_input, DFlashDraftInputV2):
            raise RuntimeError(
                "DSpark spec-v2 expected DFlashDraftInputV2 state on the running batch."
            )

        if batch.forward_mode.is_idle():
            self._observers.note_idle_decode_step()
            if self.server_args.enable_dp_attention:
                if (
                    self._draft_is_moe
                    or self._dp_vocab_sync
                    or self._draft_weight_sharded
                ):
                    # Same TP context as the busy path's propose, so the draft
                    # side of an idle rank issues its collectives on the same
                    # groups.
                    with self._draft_context():
                        self._proposer.run_idle_participation(batch)
                self._verify_executor.run_idle_participation(
                    batch=batch, idle_layout=self._idle_verify_ragged_layout(batch)
                )
                # Mirror the busy path: exactly one AG submission per step on
                # every rank (the proposer's prefetch is idempotent).
                if self._draft_weight_sharded:
                    self._proposer.prefetch_weight_shard()
            return self._decode_idle_result(on_publish=on_publish)

        batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )
        bs = len(batch.seq_lens)
        device = self.device
        prefix_lens = batch.seq_lens

        self._observers.begin_step()

        self._maintain_draft_swa_window(batch=batch, draft_input=draft_input)

        target_model = self.target_worker.model_runner.model

        verify_window = alloc_verify_window(
            batch=batch,
            bs=bs,
            device=device,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            block_pos_offsets=self._block_pos_offsets,
            model_runner=self.model_runner,
        )

        sampling_info = batch.sampling_info
        with self._draft_context(), self._observers.segment(InfoSegment.DRAFT):
            proposal = self._proposer.propose(
                batch=batch,
                draft_input=draft_input,
                verify_window=verify_window,
                bs=bs,
                device=device,
                target_model=target_model,
                sampling_info=sampling_info,
            )
        draft_block_ids = proposal.draft_block_ids
        draft_block = proposal.draft_block
        draft_tokens = draft_block.draft_tokens

        confidence = proposal.confidence
        if confidence is None:
            confidence = self._verify_planner.compute_confidence_tensor(
                draft_hidden=proposal.draft_hidden,
                anchor_tokens=draft_block_ids[:, 0],
                draft_tokens=draft_tokens,
                confidence_tap=proposal.confidence_tap,
            )

        verify_token_budget = self._verify_planner.resolve_verify_token_budget(
            draft_input=draft_input,
            confidence=confidence,
            prefix_lens=prefix_lens,
            req_pool_indices=batch.req_pool_indices,
        )

        global_num_reqs = (
            max(batch.global_num_tokens)
            if self._draft_is_moe
            and self.server_args.enable_dp_attention
            and batch.global_num_tokens is not None
            else None
        )
        layout = self._verify_planner.schedule_layout(
            req_pool_indices=batch.req_pool_indices,
            prefix_lens=prefix_lens,
            device=device,
            confidence=confidence,
            budget=verify_token_budget,
            global_num_reqs=global_num_reqs,
            dp_tier_num_tokens=self._dp_verify_tier_num_tokens(batch),
        )
        run_compact = self._verify_planner.should_run_compact(layout=layout)

        verify_ids_2d = torch.cat(
            [draft_block_ids[:, :1], draft_tokens], dim=1
        ).contiguous()

        # Must stay ahead of the target verify launch below.
        grammar_tree = (
            GrammarTree.from_linear_chain(verify_ids_2d) if batch.has_grammar else None
        )

        # A live grammar forces the eager path: the folded epilogue accepts inside
        # the cuda graph off its own buffers, where the mask below never lands.
        fold_eligible = (
            self._verify_executor.verify_epilogue is not None
            and proposal.folded
            # The epilogue's in-graph accept is greedy (accept_greedy_triton);
            # sampling batches must take the eager accept path even when the
            # draft proposal itself folded.
            and (sampling_info is None or sampling_info.is_all_greedy)
            and verify_logits_adjustments_are_noop(sampling_info)
            and self._simulate_acc_len <= 0
            and not batch.has_grammar
        )
        prepare_mamba_track_for_verify(batch)
        with self._observers.segment(InfoSegment.TARGET_VERIFY):
            if run_compact:
                target_verify, hidden_strided = self._verify_executor.run_compact(
                    batch=batch,
                    layout=layout,
                    draft_block_ids=draft_block_ids,
                    draft_tokens=draft_tokens,
                    bs=bs,
                    device=device,
                    sampling_info=sampling_info,
                    inject_gate=fold_eligible,
                )
            else:
                target_verify = self._verify_executor.run_non_compact(
                    batch=batch,
                    draft_input=draft_input,
                    verify_ids_2d=verify_ids_2d,
                    verify_window=verify_window,
                    sampling_info=sampling_info,
                )
                hidden_strided = None
        logits_output = target_verify.logits_output
        can_run_cuda_graph = target_verify.can_run_cuda_graph

        if batch.has_grammar:
            # Both the FSM advance over the previous batch's committed tokens and
            # the traversal below are host work, so they overlap the launch above.
            if grammar_barrier is not None:
                grammar_barrier()
            # run_compact scatters its rows back to (bs * chain_len), so the mask
            # lines up with the logits on both verify paths.
            vocab_mask = build_grammar_vocab_mask(
                reqs=batch.reqs,
                verify_input=draft_input,
                tree=grammar_tree,
                sampling_info=sampling_info,
                device=logits_output.next_token_logits.device,
            )
            if vocab_mask is not None:
                draft_input.grammar.apply_vocab_mask(
                    logits=logits_output.next_token_logits, vocab_mask=vocab_mask
                )

        epilogue = self._verify_executor.verify_epilogue
        folded_accept = fold_eligible and run_compact and can_run_cuda_graph
        # Fire AG 0 here, right where the accept-epilogue kernels
        # (finalize_accept_lens / build_out_tokens / index ops) are about to be
        # enqueued: the gate resolves at the epilogue start on the device, so
        # the AG covers [accept epilogue + ReplaySSM commit fold]. The proposer's
        # prefetch is idempotent per step, and the idle path mirrors this with
        # exactly one submission per step (NCCL pairs the shard-group AGs by
        # sequence number).
        if self._draft_weight_sharded:
            self._proposer.prefetch_weight_shard()
        accept = self._verify_executor.accept_and_finalize(
            folded_accept=folded_accept,
            bs=bs,
            verify_ids_2d=verify_ids_2d,
            target_logits=logits_output.next_token_logits,
            draft_block=draft_block,
            sampling_info=sampling_info,
            draft_input=draft_input,
            layout=layout,
            prefix_lens=prefix_lens,
            draft_tokens=draft_tokens,
        )
        if on_publish is not None:
            if confidence is not None:
                on_publish(accept.new_seq_lens, confidence=confidence)
            else:
                on_publish(accept.new_seq_lens)

        self._commit_target_mamba_states_after_verify(
            batch=batch,
            seq_lens_pre_verify=prefix_lens,
            seq_lens_post_verify=accept.new_seq_lens,
            commit_lens=accept.commit_lens,
        )

        folded_commit = folded_accept and epilogue.folds_commit
        if not folded_commit:
            self._verify_executor.commit_hidden(
                batch=batch,
                layout=layout,
                hidden_strided=hidden_strided,
                verify_window=verify_window,
                logits_output=logits_output,
                commit_lens=accept.commit_lens,
                bs=bs,
                run_compact=run_compact,
            )
        logits_output.hidden_states = None

        self._observers.observe_verify_step(
            forward_ct=int(batch.forward_iter),
            reqs=batch.reqs,
            bs=bs,
            proposal_folded=proposal.folded,
            verify_ids_2d=verify_ids_2d,
            target_logits=logits_output.next_token_logits,
            layout=layout,
            confidence=confidence,
            prefix_lens=prefix_lens,
            draft_tokens=draft_tokens,
            draft_block=draft_block,
            sampling_info=sampling_info,
            correct_len=accept.correct_len,
            cap_trim_lens=accept.cap_trim_lens,
            bonus=accept.bonus,
            commit_lens=accept.commit_lens,
            verify_token_budget=verify_token_budget,
            req_pool_indices=batch.req_pool_indices,
            verify_tier_num_tokens=int(batch.spec_verify_tier_num_tokens),
            dp_tier_num_tokens=self._dp_verify_tier_num_tokens(batch),
        )

        next_draft_input = make_next_draft_input(
            bonus_tokens=accept.bonus,
            new_seq_lens=accept.new_seq_lens,
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=accept.out_tokens.reshape(-1),
            accept_lens=accept.commit_lens,
            block_accept_lens=accept.commit_lens + accept.cap_trim_lens,
            cap_lens=(
                layout.verify_lens.to(torch.int32) if layout is not None else None
            ),
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=int(self.verify_num_draft_tokens),
            new_seq_lens=accept.new_seq_lens,
        )

    def _commit_target_mamba_states_after_verify(
        self,
        *,
        batch: ScheduleBatch,
        seq_lens_pre_verify: torch.Tensor,
        seq_lens_post_verify: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> None:
        """Commit the last accepted verify step's KDA/mamba state (chain
        layout: step index = commit_lens - 1) into the persistent caches."""
        if not self._need_mamba_verify_commit:
            return
        # Chain layout only: step index = commit_lens - 1. A tree (topk > 1)
        # layout would need the accept-index mapping the shared spec_utils
        # commit helper does.
        assert self.server_args.speculative_eagle_topk in (None, 1)
        attn_backend = self.target_worker.model_runner.attn_backend

        last_correct_step_indices = commit_lens.to(torch.int64) - 1
        mamba_steps_to_track = None

        if batch.mamba_track_indices is not None:
            mamba_track_interval = self.server_args.mamba_track_interval
            to_track_mask = (
                seq_lens_pre_verify // mamba_track_interval
                != seq_lens_post_verify // mamba_track_interval
            )
            tracking_point = (
                seq_lens_post_verify // mamba_track_interval * mamba_track_interval
            )
            to_track_ith = torch.clamp(tracking_point - seq_lens_pre_verify - 1, min=0)
            can_track_mask = to_track_mask & (
                to_track_ith < commit_lens.to(to_track_ith.dtype)
            )
            mamba_steps_to_track = torch.where(
                can_track_mask,
                to_track_ith.to(torch.int64),
                torch.full_like(to_track_ith, -1, dtype=torch.int64),
            )

        attn_backend.update_mamba_state_after_mtp_verify(
            last_correct_step_indices=last_correct_step_indices,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            model=self.target_worker.model_runner.model,
        )

    def get_confidence_budget_prepare(self):
        return self._verify_planner.confidence_budget_prepare()
