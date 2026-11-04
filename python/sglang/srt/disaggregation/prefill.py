"""
Life cycle of a request in the prefill server

1. Bootstrap Queue
    a. Initialize a sender for each request
    b. Use the queue to store requests whose bootstrap (handshake and preallocation) has not finished
    c. Poll senders to check bootstrap state
    d. Once bootstrap is complete, move request to Waiting Queue

2. Waiting Queue
    a. Use PrefillAdder to pop requests
    b. Run forward
    c. Add the request to Inflight Queue

3. Inflight Queue
    a. Poll (non-blocking) the sender of the request
    b. Once the transfer has finished, return the request
"""

from __future__ import annotations

import copy
import hashlib
import logging
from array import array
from collections import deque
from contextlib import contextmanager
from http import HTTPStatus
from typing import TYPE_CHECKING, Callable, List, Optional, Union

import msgspec
import numpy as np
import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.common.staging_buffer import (
    compute_grid_segments,
    staging_grid_tokens,
)
from sglang.srt.disaggregation.fake.conn import FakeKVSender
from sglang.srt.disaggregation.utils import (
    FAKE_BOOTSTRAP_HOST,
    DisaggregationMode,
    KVClassType,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    build_kv_layer_ids,
    build_staging_slot_metadata,
    get_dsa_tail_state_indices,
    get_dsv4_request_state_indices,
    get_kv_class,
    get_qsa_pending_state_indices,
    is_aborted,
    is_mla_backend,
    poll_and_all_reduce_attn_cp_tp_group,
    poll_and_all_reduce_pp,
    prepare_abort,
    setup_state_kv_args,
)
from sglang.srt.distributed import get_pp_group
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    NextBatchPlan,
    Req,
    ScheduleBatch,
)
from sglang.srt.mem_cache.common import (
    kv_to_page_indices,
    kv_to_page_indices_pinned,
    kv_to_page_num,
    maybe_cache_unfinished_req,
    release_kv_cache,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.srt.observability.req_time_stats import set_schedule_time_batch
from sglang.srt.observability.scheduler_stage_metrics import (
    SCHEDULER_STAGE_GET_NEXT_BATCH,
    SCHEDULER_STAGE_PROCESS_QUEUE,
    SchedulerStageMetricsRecorder,
    scheduler_stage_method,
)
from sglang.srt.runtime_context import (
    get_disagg,
    get_schedule,
)
from sglang.srt.utils import is_npu
from sglang.srt.utils.common import Range
from sglang.srt.utils.nvtx_utils import operations_nvtx_range

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler
    from sglang.srt.mem_cache.memory_pool import KVCache

logger = logging.getLogger(__name__)

_is_npu = is_npu()
class _DeferredChunkFinalize(msgspec.Struct, frozen=True):
    """A chunked req's chunk whose finalize the threaded AHEAD loop deferred
    (see SchedulerDisaggregationPrefillMixin.process_prefill_chunk)."""

    req: Req
    chunk_range: Range


# Deferred work queued by AHEAD admission: chunk finalizes, plus plain
# callables (a chunked abort's KV release) that always run in the serial window.
DeferredFinalize = List[Union[_DeferredChunkFinalize, Callable[[], None]]]


def should_force_retry(req: Req) -> bool:
    """Test hook to force a request into optimistic prefill retry."""
    retry_prob = envs.SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB.get()
    # Force only before/during the first attempt (count is 1 while it runs).
    if retry_prob <= 0 or req.prefill_attempt_count > 1 or req.is_retracted:
        return False

    digest = hashlib.sha256(str(req.rid).encode()).digest()
    return int.from_bytes(digest[:8], "big") < retry_prob * 2**64


def _transfer_start_layer(*, pool, hf_text_config) -> int:
    # Hybrid pools count all layers in start_layer, but peer KV lists contain only
    # full-attention layers, so translate to a full-attention-relative offset.
    if not isinstance(pool, HybridLinearKVPool):
        return pool.start_layer
    return sum(
        1 for lid in hf_text_config.full_attention_layer_ids if lid < pool.start_layer
    )


def maybe_release_metadata_buffer(
    req: Req, allocator: ReqToMetadataIdxAllocator
) -> None:
    """
    Release the metadata buffer index allocated for a request in prefill disaggregation mode.

    This function safely releases the metadata buffer index if it was allocated.

    Args:
        req: The request object that may have a metadata_buffer_index allocated
        allocator: The ReqToMetadataIdxAllocator instance to free the index
    """
    if req.metadata_buffer_index >= 0:
        allocator.free(req.metadata_buffer_index)
        req.metadata_buffer_index = -1
@contextmanager
def _batched_kv_release(allocator):
    """Defer the KV frees of one inflight-queue pass into a single free group (and one
    SWA segment batch), so releasing N finished requests costs a few kernels instead of
    a few per request on the scheduler thread. Gated by
    SGLANG_OPT_DISAGG_PREFILL_RELEASE_BATCH; a no-op inside an already open group."""
    if (
        not envs.SGLANG_OPT_DISAGG_PREFILL_RELEASE_BATCH.get()
        or getattr(allocator, "free_group", True) is not None
    ):
        yield
        return
    batch_swa = hasattr(allocator, "begin_swa_segment_batch")
    allocator.free_group_begin()
    if batch_swa:
        allocator.begin_swa_segment_batch()
    try:
        yield
    finally:
        if batch_swa:
            allocator.end_swa_segment_batch()
        allocator.free_group_end()


class PrefillBootstrapQueue:
    """
    Store the requests in bootstrapping
    """

    def __init__(
        self,
        token_to_kv_pool: KVCache,
        draft_token_to_kv_pool: Optional[KVCache],
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        metadata_buffers: MetadataBuffers,
        tp_rank: int,
        tp_size: int,
        gpu_id: int,
        bootstrap_port: int,
        gloo_group: ProcessGroup,
        max_total_num_tokens: int,
        scheduler: Scheduler,
        scheduler_stage_metrics: SchedulerStageMetricsRecorder,
        pp_rank: int,
        pp_size: int,
        transfer_backend: TransferBackend,
    ):
        self.token_to_kv_pool = token_to_kv_pool
        self.draft_token_to_kv_pool = draft_token_to_kv_pool
        self.is_mla_backend = is_mla_backend(token_to_kv_pool)
        self.metadata_buffers = metadata_buffers
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.gpu_id = gpu_id
        self.bootstrap_port = bootstrap_port
        self.queue: List[Req] = []
        self.gloo_group = gloo_group
        self.scheduler = scheduler
        self.scheduler_stage_metrics = scheduler_stage_metrics
        self.max_total_num_tokens = (
            self.scheduler.tp_worker.model_runner.effective_max_total_num_tokens
        )
        self.transfer_backend = transfer_backend
        if envs.SGLANG_DISAGG_STAGING_BUFFER.get():
            if self.is_mla_backend:
                raise RuntimeError(
                    "SGLANG_DISAGG_STAGING_BUFFER is designed for non-MLA models "
                    "(e.g. GQA, MHA). MLA models should not set this flag."
                )
            page_size = self.scheduler.token_to_kv_pool_allocator.page_size
            # Same source as send_kv_chunk's staging grid below, so validation
            # and the grid cannot disagree after a post-publish override.
            chunked_prefill_size = get_schedule().chunked_prefill_size
            cps = chunked_prefill_size or 8192
            # Staging slices each send into a fixed page-aligned grid, so an
            # unbounded (-1) or non-page-aligned chunk size has no valid grid.
            if cps <= 0 or cps % page_size != 0:
                raise RuntimeError(
                    f"SGLANG_DISAGG_STAGING_BUFFER requires a positive "
                    f"chunked_prefill_size that is a multiple of page_size "
                    f"({page_size}); got {chunked_prefill_size}."
                )
            if self.pp_size > 1 and self.transfer_backend != TransferBackend.MOONCAKE:
                raise RuntimeError(
                    "SGLANG_DISAGG_STAGING_BUFFER with pp_size > 1 is only "
                    "supported by Mooncake."
                )
        self.kv_manager = self._init_kv_manager()

    def _init_kv_manager(self) -> CommonKVManager:
        kv_args_class = get_kv_class(self.transfer_backend, KVClassType.KVARGS)
        kv_args = kv_args_class()
        kv_args.engine_rank = self.tp_rank
        kv_args.pp_rank = self.pp_rank
        kv_args.system_dp_rank = self.scheduler.ps.dp_rank
        kv_args.rust_http_port = (
            self.scheduler.rust_server.http_port
            if self.scheduler.rust_server is not None
            else None
        )
        kv_args.kv_cache_dtype_str = (
            self.scheduler.tp_worker.model_runner.kv_cache_dtype_str
        )
        layer_shard_enabled = getattr(
            self.token_to_kv_pool, "layer_shard_enabled", False
        )
        layer_shard_rank = getattr(self.token_to_kv_pool, "layer_shard_rank", None)
        layer_shard_size = getattr(self.token_to_kv_pool, "layer_shard_size", 1)
        transfer_draft_cache = (
            not layer_shard_enabled or layer_shard_rank == layer_shard_size - 1
        )
        kv_args.prefill_start_layer = (
            getattr(
                self.token_to_kv_pool,
                "layer_shard_start",
                self.token_to_kv_pool.start_layer,
            )
            if layer_shard_enabled
            else _transfer_start_layer(
                pool=self.token_to_kv_pool,
                hf_text_config=self.scheduler.model_config.hf_text_config,
            )
        )
        kv_data_ptrs, kv_data_lens, kv_item_lens = (
            self.token_to_kv_pool.get_contiguous_buf_infos()
        )
        kv_args.prefill_end_layer = (
            kv_args.prefill_start_layer + len(kv_data_ptrs)
            if layer_shard_enabled
            else getattr(self.token_to_kv_pool, "end_layer", None)
        )

        draft_kv_pool = (
            self.draft_token_to_kv_pool
            if transfer_draft_cache and (not _is_npu or get_pp_group().is_last_rank)
            else None
        )
        num_draft_entries = 0
        if draft_kv_pool is not None:
            # We should also transfer draft model kv cache. The indices are
            # always shared with a target model.
            draft_kv_data_ptrs, draft_kv_data_lens, draft_kv_item_lens = (
                draft_kv_pool.get_contiguous_buf_infos()
            )
            kv_data_ptrs += draft_kv_data_ptrs
            kv_data_lens += draft_kv_data_lens
            kv_item_lens += draft_kv_item_lens
            num_draft_entries = len(draft_kv_data_ptrs)

        kv_args.kv_data_ptrs = kv_data_ptrs
        kv_args.kv_data_lens = kv_data_lens
        kv_args.kv_item_lens = kv_item_lens
        kv_args.num_draft_entries = num_draft_entries
        kv_args.kv_layer_ids = build_kv_layer_ids(
            token_to_kv_pool=self.token_to_kv_pool,
            draft_token_to_kv_pool=draft_kv_pool,
            num_draft_entries=num_draft_entries,
            num_hidden_layers=self.scheduler.model_config.num_hidden_layers,
        )
        if not self.is_mla_backend:
            kv_args.kv_head_num = self.token_to_kv_pool.head_num
            kv_args.total_kv_head_num = (
                self.scheduler.model_config.get_total_num_kv_heads()
            )
        kv_args.page_size = self.token_to_kv_pool.page_size

        kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.aux_item_lens = (
            self.metadata_buffers.get_buf_infos()
        )
        kv_args.ib_device = get_disagg().disaggregation_ib_device
        kv_args.gpu_id = self.scheduler.ps.gpu_id

        req_to_token_pool = getattr(self.scheduler, "req_to_token_pool", None)
        setup_state_kv_args(
            kv_args,
            self.token_to_kv_pool,
            self.draft_token_to_kv_pool if transfer_draft_cache else None,
            self.scheduler.model_config.num_hidden_layers,
            req_to_token_pool=req_to_token_pool,
        )

        kv_manager_class = get_kv_class(self.transfer_backend, KVClassType.MANAGER)
        kv_manager = kv_manager_class(
            kv_args,
            DisaggregationMode.PREFILL,
            self.scheduler.server_args,
            self.is_mla_backend,
        )
        # Pass KV pool tensor refs to the manager for GPU gather (staging mode)
        if (
            envs.SGLANG_DISAGG_STAGING_BUFFER.get()
            and hasattr(kv_manager, "set_kv_buffer_tensors")
            and not self.is_mla_backend
        ):
            kv_pool = self.token_to_kv_pool
            if hasattr(kv_pool, "full_kv_pool"):
                kv_pool = kv_pool.full_kv_pool
            staging_slots = build_staging_slot_metadata(
                kv_layer_ids=kv_args.kv_layer_ids,
                num_draft_entries=num_draft_entries,
                kv_pool=kv_pool,
                draft_kv_pool=draft_kv_pool,
            )
            if staging_slots is not None:
                k_buffers, v_buffers, slot_layer_ids = staging_slots
                kv_manager.set_kv_buffer_tensors(
                    k_buffers,
                    v_buffers,
                    kv_pool.page_size,
                    slot_layer_ids=slot_layer_ids,
                )
        return kv_manager

    def create_sender(self, req: Req, num_kv_heads: int) -> bool:
        """Create a KV sender for the request without enqueuing it.
        Returns False if the request exceeds KV capacity."""
        if self._check_if_req_exceed_kv_capacity(req):
            return False

        backend = (
            TransferBackend.FAKE
            if req.bootstrap_host == FAKE_BOOTSTRAP_HOST
            else self.transfer_backend
        )
        kv_sender_class = get_kv_class(backend, KVClassType.SENDER)

        dest_tp_ranks = [self.tp_rank]

        req.disagg_kv_sender = kv_sender_class(
            mgr=self.kv_manager,
            bootstrap_addr=f"{req.bootstrap_host}:{self.bootstrap_port}",
            bootstrap_room=req.bootstrap_room,
            dest_tp_ranks=dest_tp_ranks,
            pp_rank=self.pp_rank,
            req_has_disagg_prefill_dp_rank=req.disagg_prefill_dp_rank is not None,
        )
        self._process_req(req)
        req.pending_bootstrap = True
        return True

    def ensure_metadata_buffer(self, req: Req) -> bool:
        if req.metadata_buffer_index >= 0:
            return True

        if self.req_to_metadata_buffer_idx_allocator.available_size() == 0:
            return False
        req.metadata_buffer_index = self.req_to_metadata_buffer_idx_allocator.alloc()
        assert req.metadata_buffer_index is not None
        return True

    def finalize_bootstrap(self, req: Req) -> bool:
        """Initialize the sender after bootstrap completes.
        Returns False if no metadata buffer is available (non-terminal)."""
        assert req.pending_bootstrap, "finalize_bootstrap is not idempotent"
        if not self.ensure_metadata_buffer(req):
            return False

        req.time_stats.set_bootstrap_done_time()
        decode_prefix_len = req.disagg_kv_sender.pop_decode_prefix_len()
        num_kv_indices = len(req.origin_input_ids)
        req.start_send_idx = decode_prefix_len
        # Base of the staging chunk grid (suffix-relative send coordinates).
        req.disagg_decode_prefix_len = decode_prefix_len
        num_kv_indices_to_send = num_kv_indices - decode_prefix_len
        num_pages = kv_to_page_num(
            num_kv_indices_to_send,
            self.scheduler.token_to_kv_pool_allocator.page_size,
        )
        req.disagg_kv_sender.init(num_pages, req.metadata_buffer_index)
        req.pending_bootstrap = False
        return True

    def add(self, req: Req, num_kv_heads: int) -> None:
        if not self.create_sender(req, num_kv_heads):
            return
        self.queue.append(req)

    def extend(self, reqs: List[Req], num_kv_heads: int) -> None:
        for req in reqs:
            self.add(req, num_kv_heads)

    def _check_if_req_exceed_kv_capacity(self, req: Req) -> bool:
        if len(req.origin_input_ids) > self.max_total_num_tokens:
            message = f"Request {req.rid} exceeds the maximum number of tokens: {len(req.origin_input_ids)} > {self.max_total_num_tokens}"
            logger.error(message)
            req.time_stats.trace_ctx.abort(abort_info={"reason": message})
            prepare_abort(req, message, status_code=HTTPStatus.BAD_REQUEST)
            self.scheduler.output_streamer.stream_output([req], req.return_logprob)
            return True
        return False

    def _process_req(self, req: Req) -> None:
        """
        Set max_new_tokens = 1, so PrefillAdder memory estimation is accurate
        """
        req.sampling_params.max_new_tokens = 1

    @scheduler_stage_method(SCHEDULER_STAGE_PROCESS_QUEUE)
    def pop_bootstrapped(
        self,
        return_failed_reqs: bool = False,
        pp_good_rids: Optional[List[str]] = None,
        pp_bad_rids: Optional[List[str]] = None,
    ) -> List[Req] | tuple[List[Req], List[Req]]:
        """
        pop the reqs which has finished bootstrapping

        return_failed_reqs: For PP, on rank 0, also return the failed reqs to notify the next rank
        pp_good_rids: RIDs that PP consensus determined as WaitingForInput.
        pp_bad_rids: RIDs that PP consensus determined as Failed.
        """

        bootstrapped_reqs = []
        failed_reqs = []
        indices_to_remove = set()

        if len(self.queue) == 0:
            if return_failed_reqs is False:
                return []
            else:
                return [], []

        if self.pp_size > 1:
            polls = poll_and_all_reduce_pp(
                (req.rid for req in self.queue),
                KVPoll.WaitingForInput,
                pp_good_rids,
                pp_bad_rids,
            )
            uncovered = [i for i, poll in enumerate(polls) if poll is None]
            if uncovered:
                local_polls = poll_and_all_reduce_attn_cp_tp_group(
                    [self.queue[i].disagg_kv_sender for i in uncovered],
                    self.scheduler.attn_cp_cpu_group,
                    self.scheduler.attn_tp_cpu_group,
                )
                for i, local_poll in zip(uncovered, local_polls):
                    if local_poll == KVPoll.Failed:
                        polls[i] = KVPoll.Failed
        else:
            polls = poll_and_all_reduce_attn_cp_tp_group(
                [req.disagg_kv_sender for req in self.queue],
                self.scheduler.attn_cp_cpu_group,
                self.scheduler.attn_tp_cpu_group,
            )

        for i, (req, poll) in enumerate(zip(self.queue, polls)):
            if poll is None:
                continue

            if poll == KVPoll.Failed:
                self.scheduler.handle_bootstrap_failure(req)
                indices_to_remove.add(i)
                failed_reqs.append(req)
            elif poll == KVPoll.Bootstrapping:
                if (
                    req.prefill_attempt_count < get_disagg().optimistic_prefill_attempts
                    and not req.is_retracted  # engine paused
                ):
                    if not self.ensure_metadata_buffer(req):
                        continue  # no more metadata buffer
                    req.prefill_attempt_count += 1
                    bootstrapped_reqs.append(req)
                    indices_to_remove.add(i)
                    req.time_stats.set_wait_queue_entry_time()
                    req.arrival_processed_tokens = (
                        self.scheduler.processed_tokens_counter
                    )
            elif poll == KVPoll.WaitingForInput:
                if should_force_retry(req):  # skip checking for testing
                    if not self.ensure_metadata_buffer(req):
                        continue  # no more metadata buffer
                    req.prefill_attempt_count += 1
                elif not self.finalize_bootstrap(req):
                    continue
                bootstrapped_reqs.append(req)
                indices_to_remove.add(i)
                req.time_stats.set_wait_queue_entry_time()
                req.arrival_processed_tokens = self.scheduler.processed_tokens_counter
            else:
                raise RuntimeError(
                    f"Unexpected poll state {poll} for req {req.rid} in pop_bootstrapped"
                )

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        if return_failed_reqs is False:
            return bootstrapped_reqs
        else:
            return bootstrapped_reqs, failed_reqs

    def release_memory_occupation(self):
        self.queue.clear()
        if hasattr(self.kv_manager, "deregister_buffer_to_engine"):
            self.kv_manager.deregister_buffer_to_engine()

    def resume_memory_occupation(self):
        if hasattr(self.kv_manager, "register_buffer_to_engine"):
            self.kv_manager.register_buffer_to_engine()


class SchedulerDisaggregationPrefillMixin:
    """
    Mixin for Scheduler to handle disaggregation prefill
    """

    def maybe_prefetch_staging_for_batch(self: Scheduler, batch: ScheduleBatch) -> None:
        """Pre-send STAGING_REQ so decode allocates staging during GPU forward."""
        kv_mgr = self.disagg_prefill_bootstrap_queue.kv_manager
        prefetch = getattr(kv_mgr, "_prefetch_staging_reqs", None)
        if prefetch is None:
            return
        for req in batch.reqs:
            room = getattr(req, "bootstrap_room", None)
            if room is not None and room in kv_mgr.transfer_infos:
                prefetch(room)

    @scheduler_stage_method(SCHEDULER_STAGE_PROCESS_QUEUE)
    def resolve_waiting_queue_bootstrap(self: Scheduler) -> None:
        """Resolve bootstrap status for waiting prefill requests before admission.

        Covers the window between leaving the bootstrap queue and being admitted
        into a running batch: aborts requests whose decode peer died, and
        finalizes optimistic requests whose bootstrap completed so they skip
        the post-forward bootstrap check.
        """
        candidates = [req for req in self.waiting_queue if not is_aborted(req)]
        if not candidates:
            return
        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender for req in candidates],
            self.attn_cp_cpu_group,
            self.attn_tp_cpu_group,
        )
        failed = set()
        for req, poll in zip(candidates, polls):
            if poll == KVPoll.Failed:
                self.handle_bootstrap_failure(req)
                failed.add(req)
            elif (
                poll == KVPoll.WaitingForInput
                and req.pending_bootstrap
                and not should_force_retry(req)
            ):
                # Optimistic requests reserved a metadata buffer when popped, so
                # finalize cannot fail here; if it ever does, the request stays
                # pending and the post-forward check resolves it.
                self.disagg_prefill_bootstrap_queue.finalize_bootstrap(req)
        if failed:
            self.waiting_queue = [
                req for req in self.waiting_queue if req not in failed
            ]

    def has_bootstrapped_waiting_req(self: Scheduler) -> bool:
        return any(
            not req.pending_bootstrap and not is_aborted(req)
            for req in self.waiting_queue
        )

    @scheduler_stage_method(SCHEDULER_STAGE_GET_NEXT_BATCH)
    def get_next_disagg_prefill_batch_to_run(
        self: Scheduler,
        running_batch: ScheduleBatch,
        last_batch: Optional[ScheduleBatch],
        *,
        deferred_finalize: Optional[DeferredFinalize] = None,
    ) -> NextBatchPlan:
        """``deferred_finalize``: see process_prefill_chunk (threaded AHEAD loop)."""
        self.process_pending_chunked_abort(deferred_release=deferred_finalize)

        # HACK (byronhsu): reset the batch_is_full flag because we never enter update_running_batch which resets it
        # Otherwise, it hangs under high concurrency
        running_batch.batch_is_full = False

        self.resolve_waiting_queue_bootstrap()

        self.process_prefill_chunk(
            last_batch=last_batch,
            running_batch=running_batch,
            deferred_finalize=deferred_finalize,
        )

        prefill_plan = self.get_new_batch_prefill(running_batch)
        batch = prefill_plan.batch_to_run
        running_batch = prefill_plan.running_batch
        batch = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(batch)

        if batch:
            set_schedule_time_batch(batch)

        return NextBatchPlan(batch_to_run=batch, running_batch=running_batch)

    @torch.no_grad()
    def event_loop_normal_disagg_prefill(self: Scheduler) -> None:
        """A normal scheduler loop for prefill worker in disaggregation mode."""
        while True:
            self.retire_ascache_writes()
            # Receive requests
            self.ingest_requests()
            if self._engine_paused:
                self._record_scheduler_state_for_paused_engine()
                continue
            self.waiting_queue.extend(
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
            )
            self.process_disagg_prefill_inflight_queue(
                prepolled=self._fused_inflight_polls
            )

            # Get the next batch to run
            plan = self.get_next_disagg_prefill_batch_to_run(
                running_batch=self.running_batch, last_batch=self.last_batch
            )
            self.running_batch = plan.running_batch
            batch = plan.batch_to_run
            batch = self.ngram_embedding_manager.prepare_for_forward(
                batch, chunked_req=self.chunked_req
            )
            self.cur_batch_for_debug = batch

            # Launch the current batch
            if batch:
                if self.enable_staging:
                    self.maybe_prefetch_staging_for_batch(batch)
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                self.on_idle()


            # Update last_batch
            self.last_batch = batch

    @torch.no_grad()
    def event_loop_overlap_disagg_prefill(self: Scheduler) -> None:
        self.result_queue = deque()

        while True:
            self.retire_ascache_writes()
            # Receive requests
            self.ingest_requests()
            if self._engine_paused:
                self._record_scheduler_state_for_paused_engine()
                continue
            self.waiting_queue.extend(
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
            )
            self.process_disagg_prefill_inflight_queue(
                prepolled=self._fused_inflight_polls
            )

            # Get the next batch to run
            plan = self.get_next_disagg_prefill_batch_to_run(
                running_batch=self.running_batch, last_batch=self.last_batch
            )
            self.running_batch = plan.running_batch
            batch = plan.batch_to_run
            batch = self.ngram_embedding_manager.prepare_for_forward(
                batch, chunked_req=self.chunked_req
            )
            self.cur_batch_for_debug = batch

            # Launch the current batch
            if batch:
                if self.enable_staging:
                    self.maybe_prefetch_staging_for_batch(batch)
                batch_result = self.run_batch(batch)
                self._apply_war_barrier()
                self.result_queue.append((batch.copy(), batch_result))
            else:
                batch_result = None

            # Process the last batch
            if self.last_batch:
                tmp_batch, tmp_result = self.result_queue.popleft()
                self.process_batch_result(tmp_batch, tmp_result)
            elif batch is None:
                # When the server is idle, do self-check and re-init some states
                self.on_idle()


            # Run sample of the current batch
            # It depends on the result of the last batch (e.g., grammar), so we run it after the last batch is processed.
            self.launch_batch_sample_if_needed(batch_result, batch)

            # Update last_batch
            self.last_batch = batch
    def _ensure_prefill_fwd_thread(self: Scheduler) -> None:
        """Lazily start the forward worker thread (never created when the
        SGLANG_ENABLE_PREFILL_FWD_THREAD gate is off, so the base loop is
        untouched)."""
        if getattr(self, "_fwd_thread", None) is not None:
            return
        if (
            torch.device(self.device).type == "cuda"
            and self.token_to_kv_pool_allocator.page_size > 1
        ):
            from sglang.srt.mem_cache.allocator.paged import warmup_alloc_extend_kernel

            warmup_alloc_extend_kernel(
                self.device,
                self.token_to_kv_pool_allocator.page_size,
                self.max_running_requests,
            )
            logger.info(
                "Prefill allocator kernels warmed: page_size=%d max_batch_size=%d",
                self.token_to_kv_pool_allocator.page_size,
                self.max_running_requests,
            )
        import queue as _queue
        import threading as _threading

        self._fwd_in_q: _queue.Queue = _queue.Queue(maxsize=1)
        self._fwd_out_q: _queue.Queue = _queue.Queue(maxsize=1)
        self._fwd_thread = _threading.Thread(
            target=self._prefill_fwd_thread_main, name="prefill_fwd", daemon=True
        )
        self._fwd_thread.start()

    # Grad mode is thread-local: the worker does not inherit the scheduler's
    # no_grad, and the CP extend path calls model.model() past its no_grad.
    @torch.no_grad()
    def _prefill_fwd_thread_main(self: Scheduler) -> None:
        """Worker thread: bind the CUDA device, then run each submitted forward.

        Safe to run off the scheduler thread because a prefill forward is
        read-only w.r.t. the KV allocator / radix tree / req_to_token (those
        mutate only on the scheduler thread), and the forward's NCCL
        device_group is a distinct communicator from the gloo cpu_group every
        scheduler-thread collective uses -- so the two threads never share a
        communicator. The WAR barrier is recorded here, right after the
        forward's shared reads, so it captures them before any slot reuse.
        """
        self.device_module.set_device(self.ps.gpu_id)
        batch_write_fence = envs.SGLANG_AS_SPAG_BATCH_WRITE_FENCE.get()
        while True:
            item = self._fwd_in_q.get()
            if item is None:  # shutdown sentinel
                return
            batch, early_send_prefix, extend_input_len_per_req = item
            events = None
            if self._fwd_record_events:
                events = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                events[0].record(self.forward_stream)
            try:
                # inference_mode skips autograd version/view bookkeeping per op; the
                # eager prefill body is host launch bound at small chunks.
                with torch.inference_mode(
                    envs.SGLANG_OPT_PREFILL_FWD_INFERENCE_MODE.get()
                ):
                    result = self.run_batch(
                        batch,
                        early_send_prefix=early_send_prefix,
                        extend_input_len_per_req=extend_input_len_per_req,
                    )
                if events is not None:
                    events[1].record(self.forward_stream)
                    self._fwd_events.append(events)
                if batch_write_fence:
                    batch.as_kv_ready_event = self.device_module.Event()
                    batch.as_kv_ready_event.record(stream=self.forward_stream)
                self._apply_war_barrier()
                # Snapshot on the worker, after the forward's in-place SB edits
                # and while only this thread owns the batch (the scheduler has
                # moved on to the next admission), so the copy is race-free.
                batch_copy = batch.copy()
                self._fwd_out_q.put((result, batch_copy, None))
            except BaseException as exc:  # relay to the scheduler thread
                self._fwd_out_q.put((None, None, exc))

    def _fwd_submit(
        self: Scheduler,
        batch,
        *,
        early_send_prefix: bool = True,
        extend_input_len_per_req: Optional[List[int]] = None,
    ) -> None:
        """Hand a batch to the forward thread (non-blocking)."""
        self._ensure_prefill_fwd_thread()
        self._fwd_in_q.put((batch, early_send_prefix, extend_input_len_per_req))

    def _fwd_collect(self: Scheduler):
        """Block for the oldest submitted forward; return (result, batch_copy)
        or re-raise its error."""
        result, batch_copy, exc = self._fwd_out_q.get()
        if exc is not None:
            raise exc
        if batch_copy.as_kv_ready_event is not None:
            for req in batch_copy.reqs:
                req.as_kv_ready_event = batch_copy.as_kv_ready_event
        return result, batch_copy

    def _run_batch_on_fwd_thread(self: Scheduler, batch):
        """Zero-overlap (Step 1): submit then block immediately, so the worker
        and scheduler never run concurrently -- proves the forward is correct
        off-thread with no perf change. Returns (result, batch_copy)."""
        self._fwd_submit(batch)
        return self._fwd_collect()

    def _snapshot_batch_for_forward(self: Scheduler, batch):
        """Independent-enough snapshot so the forward thread can read it while
        the scheduler admits the next batch (Step 2).

        A shallow ScheduleBatch copy already isolates every field admission
        rebinds out of place: get_next / filter_batch rebind the LIVE batch's
        attrs (reqs, seq_lens, req_pool_indices, sampling_info, ...), and this
        copy is a distinct object that keeps the old refs. Only the in-place
        mutations the forward actually reads need explicit isolation:

        - sampling_info: filter_batch mutates it in place. The forward-only copy
          drops the penalizer orchestrator (overlap-mode sampling reads the acc_*
          buffers), leaving the live sampling_info's orchestrator intact for
          admission's filter.
        - spec_info: filter_batch mutates it in place when present.

        The reqs list itself is NOT copied per-element: the forward reads only
        batch-level tensors (input_ids / seq_lens / out_cache_loc / ...), never a
        req's fill_ids, so admission's in-place edits to a chunked req don't race
        it. Crucially, process_batch_result must see the LIVE req objects here so
        its inflight_middle_chunks decrement lands on the same req the admission
        increment did -- isolating the chunked req into a throwaway copy leaked
        that decrement and left the live req wedged as a middle chunk forever
        (orphaned req_to_token slot + KV, and 1-token admission thereafter).
        """
        snap = copy.copy(batch)
        if batch.sampling_info is not None:
            snap.sampling_info = batch.sampling_info.copy_for_forward()
        if batch.spec_info is not None:
            snap.spec_info = copy.copy(batch.spec_info)
        return snap

    @torch.no_grad()
    def event_loop_overlap_disagg_prefill_threaded(self: Scheduler) -> None:
        """SGLANG_ENABLE_PREFILL_FWD_THREAD: run the forward on a worker thread.

        AHEAD off (Step 1): zero-overlap validation -- submit and block.
        AHEAD on (Step 2): admit the NEXT batch while the current forward is in
        flight, so the loop-top retire/ingest/admission overlap the previous
        forward. Safe on a prefill P instance (forward read-only w.r.t.
        allocator/tree/req_to_token; NCCL group disjoint from the scheduler
        thread's gloo collectives; read slots stay tree-locked until their
        pipelined process_batch_result).
        """
        ahead = envs.SGLANG_PREFILL_FWD_THREAD_AHEAD.get()
        self.result_queue = deque()
        inflight = None  # live prev batch whose forward is on the worker (ahead)

        while True:
            self.retire_ascache_writes()
            self.ingest_requests()
            if self._engine_paused:
                self._record_scheduler_state_for_paused_engine()
                continue
            self.waiting_queue.extend(
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
            )

            # AHEAD: admission below runs while the previous forward may still be
            # launching on the worker (no WAR barrier on schedule_stream yet), so
            # the work that reads / frees that forward's in-flight KV (the
            # chunked req's chunk finalize, a chunked abort's KV release) is
            # queued here and run once the forward is collected.
            deferred_finalize: Optional[DeferredFinalize] = (
                [] if ahead else None
            )
            plan = self.get_next_disagg_prefill_batch_to_run(
                running_batch=self.running_batch,
                last_batch=self.last_batch,
                deferred_finalize=deferred_finalize,
            )
            self.running_batch = plan.running_batch
            batch = plan.batch_to_run
            batch = self.ngram_embedding_manager.prepare_for_forward(
                batch, chunked_req=self.chunked_req
            )
            self.cur_batch_for_debug = batch

            if not ahead:
                if batch:
                    if self.enable_staging:
                        self.maybe_prefetch_staging_for_batch(batch)
                    batch_result, batch_copy = self._run_batch_on_fwd_thread(batch)
                    self.result_queue.append((batch_copy, batch_result))
                else:
                    batch_result = None
                if self.last_batch:
                    tmp_batch, tmp_result = self.result_queue.popleft()
                    self.process_batch_result(tmp_batch, tmp_result)
                elif batch is None:
                    self.on_idle()
                self.process_disagg_prefill_inflight_queue()
                self.launch_batch_sample_if_needed(batch_result, batch)
                self.last_batch = batch
                continue

            # Step 2: admit-ahead. This iteration's retire / ingest / get_next
            # above already ran WHILE the previous batch's forward was in flight;
            # what of it touches that forward's in-flight KV was deferred (see
            # deferred_finalize) and runs right after the collect below.
            # Only sampling must run with the worker idle: it uses forward_stream
            # and future_map, which run_batch (worker) uses too. So: collect the
            # previous forward, sample it, launch this batch's forward, and only
            # THEN process the previous result, which overlaps the new forward
            # (the base loop orders run_batch(N) before process(N-1) the same
            # way). process touches neither forward_stream nor future_map; its
            # radix inserts, SWA staging (side stream) and KV sends hit the
            # previous batch's reqs, never what the new forward reads. Admission
            # touches neither structure either (the future_map.stash paths are
            # gated to non-overlap / mixed-decode, which never fire here).
            prev_result = None
            if inflight is not None:
                prev_snap = inflight
                prev_result, prev_result_copy = self._fwd_collect()
                # prev_snap carries an independent sampling_info, so sampling the
                # previous batch is correct even though get_next above already
                # filtered its live batch.
                self.launch_batch_sample_if_needed(prev_result, prev_snap)

            # The previous forward is now fully enqueued and its WAR barrier is
            # on schedule_stream (the worker applies it before handing back the
            # result), and the worker is idle until the submit below. Work run
            # here is ordered after the previous forward on the GPU and lands
            # before the next forward reads req_to_token -- what the base loop
            # gets from running admission after run_batch + barrier. Only a
            # chunk finalize that would dedup-repoint the chunked req's row needs
            # that last property; the rest overlaps the next forward's launch.
            overlap_finalize = self._run_serial_deferred_finalize(deferred_finalize)

            if batch:
                if self.enable_staging:
                    self.maybe_prefetch_staging_for_batch(batch)
                snap = self._snapshot_batch_for_forward(batch)
                # Early-send the cached prefix here rather than inside run_batch,
                # so the forward thread never races the send_kv_chunk that
                # process_batch_result below issues for a continuing chunked req.
                for req in snap.reqs:
                    self.maybe_send_cached_prefix_chunk(req)
                self._fwd_submit(
                    snap,
                    early_send_prefix=False,
                    # Read on this thread: the next admission may advance a
                    # chunked req's extend_range while the worker is mid-run_batch.
                    extend_input_len_per_req=self._snapshot_extend_input_lens(snap),
                )
                new_inflight = snap
            else:
                new_inflight = None

            # Overlaps the forward just submitted; before process(prev) as in the
            # base loop, whose admission finalizes the chunk first.
            for item in overlap_finalize:
                self._finalize_prefill_chunk(item.req, chunk_range=item.chunk_range)

            if prev_result is not None:
                # Worker's post-forward copy of the snapshot; finished reqs are
                # the same objects, so KV-send / cache writes land on the real reqs.
                self.process_batch_result(prev_result_copy, prev_result)
            elif new_inflight is None:
                self.on_idle()

            # Drain KV transfers every iter (as the base loop does); pending
            # sends must complete even when the pipeline momentarily has nothing
            # in flight, or their requests never finish.
            self.process_disagg_prefill_inflight_queue()

            inflight = new_inflight
            # Admission state only (no forward result); the next iter's
            # get_next_batch must see this batch as last_batch.
            self.last_batch = batch


    def process_batch_result_disagg_prefill(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> None:
        """
        Transfer kv for prefill completed requests and add it into disagg_prefill_inflight_queue
        Adapted from process_batch_result_prefill
        """
        (
            logits_output,
            next_token_ids,
            extend_input_len_per_req,
            extend_logprob_start_len_per_req,
            copy_done,
        ) = (
            result.logits_output,
            result.next_token_ids,
            result.extend_input_len_per_req,
            result.extend_logprob_start_len_per_req,
            result.copy_done,
        )

        if copy_done is not None:
            copy_done.synchronize()
        auxiliary_output_starts = (
            self.batch_result_processor.snapshot_auxiliary_output_starts(batch, result)
        )
        auxiliary_output = result.auxiliary_host_output
        if result.routed_experts_output is not None:
            result.routed_experts_output.finalize()
            result.routed_experts_output = None
        if result.indexer_topk_output is not None:
            result.indexer_topk_output.finalize()
            result.indexer_topk_output = None

        logprob_pt = 0
        aborted_reqs: List[Req] = []
        assert batch.spec_info is result.next_draft_input
        draft_input = result.next_draft_input
        draft_hidden_states_cpu = None
        draft_dsa_topk_indices_cpu = None
        if self.spec_algorithm.is_eagle() and draft_input is not None:
            draft_hidden_states_cpu = draft_input.hidden_states.to(
                "cpu", non_blocking=False
            )
            if batch.spec_info.dsa_topk_indices is not None:
                draft_dsa_topk_indices_cpu = batch.spec_info.dsa_topk_indices.to(
                    "cpu", non_blocking=False
                )
        # Transfer kv for prefill completed requests and add it into disagg_prefill_inflight_queue
        next_token_ids = result.next_token_ids.tolist()
        self.batch_result_processor.move_logprobs_to_cpu(
            batch=batch,
            logits_output=logits_output,
        )
        if logits_output is not None and logits_output.sampling_mask_output is not None:
            self.batch_result_processor.materialize_sampling_mask_output(
                batch.reqs, logits_output
            )

        def advance_logprob_pt(i: int, req: Req) -> None:
            nonlocal logprob_pt
            if not req.return_logprob or extend_input_len_per_req is None:
                return
            extend_logprob_start_len = extend_logprob_start_len_per_req[i]
            extend_input_len = extend_input_len_per_req[i]
            if extend_logprob_start_len < extend_input_len:
                logprob_pt += extend_input_len - extend_logprob_start_len

        # Rescue every finished req's tail SWA window in ONE batched staging
        # pass, before the per-req loop below submits any AttentionStore write
        # (the write consumes the staged slots). The window is only HBM-resident
        # this forward, so this must run before any early-continue below.
        self.detect_and_may_copy_as_swa_batch(
            [req for req in batch.reqs if req.inflight_middle_chunks <= 0]
        )
        for i, (req, next_token_id) in enumerate(
            zip(batch.reqs, next_token_ids, strict=True)
        ):
            if req.inflight_middle_chunks <= 0:
                req.time_stats.set_prefill_finished_time()

                if is_aborted(req):
                    if self._retire_aborted_prefill_result(req):
                        req.time_stats.set_completion_time()
                        aborted_reqs.append(req)
                    advance_logprob_pt(i, req)
                    continue

                # Test hook: exercise the release/requeue retry path.
                if req.pending_bootstrap and should_force_retry(req):
                    self.optimistic_release_and_requeue(req)
                    advance_logprob_pt(i, req)
                    continue

                sampling_mask_finish_reason = None
                if req.return_sampling_mask:
                    assert logits_output is not None
                    statuses = logits_output.next_token_sampling_mask_status
                    status = None if statuses is None else statuses[i]
                    sampling_mask_finish_reason = (
                        self.batch_result_processor.get_sampling_mask_finish_reason(
                            status=status
                        )
                    )
                if sampling_mask_finish_reason is not None:
                    req.to_finish = sampling_mask_finish_reason
                    req.time_stats.trace_ctx.abort(
                        abort_info={"reason": sampling_mask_finish_reason.message}
                    )
                    if self._retire_aborted_prefill_result(req):
                        req.time_stats.set_completion_time()
                        aborted_reqs.append(req)
                    advance_logprob_pt(i, req)
                    continue

                req.output_ids.append(next_token_id)
                if req.grammar is not None:
                    try:
                        req.grammar.accept_token(next_token_id)
                    except ValueError as e:
                        error_message = f"Grammar accept_token failed for req {req.rid} with token {next_token_id}: {e}"
                        prepare_abort(
                            req,
                            error_message,
                            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                        )
                    req.grammar.finished = req.finished()
                    if is_aborted(req):
                        if self._retire_aborted_prefill_result(req):
                            req.time_stats.set_completion_time()
                            aborted_reqs.append(req)
                        advance_logprob_pt(i, req)
                        continue

                maybe_cache_unfinished_req(req, self.tree_cache)
                self.disagg_prefill_inflight_queue.append(req)
                if self.spec_algorithm.is_eagle() and draft_input is not None:
                    req.output_topk_p = draft_input.topk_p[i]
                    req.output_topk_index = draft_input.topk_index[i]
                    req.hidden_states_tensor = draft_hidden_states_cpu[i].clone()
                    if draft_dsa_topk_indices_cpu is not None:
                        req.output_dsa_topk_indices = draft_dsa_topk_indices_cpu[
                            i
                        ].clone()
                    else:
                        req.output_dsa_topk_indices = None
                else:
                    req.hidden_states_tensor = None
                    req.output_dsa_topk_indices = None
                if req.return_logprob:
                    assert extend_logprob_start_len_per_req is not None
                    assert extend_input_len_per_req is not None
                    extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                    extend_input_len = extend_input_len_per_req[i]
                    num_input_logprobs = extend_input_len - extend_logprob_start_len
                    self.batch_result_processor.logprob_result_processor.add_logprob_return_values(
                        i,
                        req,
                        logprob_pt,
                        next_token_ids,
                        num_input_logprobs,
                        logits_output,
                    )
                    logprob_pt += num_input_logprobs
                if req.return_sampling_mask:
                    self.batch_result_processor.add_sampling_mask_return_values(
                        i, req, logits_output
                    )
                if not req.pending_bootstrap and not self._prefill_fake_free_kv(req):
                    self.send_kv_chunk(req, last_chunk=True)
                req.time_stats.set_prefill_transfer_queue_entry_time()

            else:
                # being chunked reqs' prefill is not finished
                req.inflight_middle_chunks -= 1

                # Still chunking iff its next chunk was launched: either it is
                # still self.chunked_req, or its final chunk (extend_range
                # reaching the end of the input) is in flight. A yielded req
                # is neither, so do its deferred release here.
                still_chunking = self.chunked_req is req or (
                    req.extend_range is not None
                    and req.extend_range.end >= len(req.origin_input_ids)
                )
                # Abort is terminal. Do not requeue an aborted optimistic
                # request merely because bootstrap is still pending.
                if is_aborted(req):
                    if not still_chunking and self._retire_aborted_prefill_result(req):
                        req.time_stats.set_completion_time()
                        aborted_reqs.append(req)
                    advance_logprob_pt(i, req)
                    req.time_stats.set_last_chunked_prefill_finish_time()
                    continue

                if req.pending_bootstrap and not still_chunking:
                    self.optimistic_release_and_requeue(req)
                    advance_logprob_pt(i, req)
                    req.time_stats.set_last_chunked_prefill_finish_time()
                    continue

                if req.return_logprob:
                    extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                    extend_input_len = extend_input_len_per_req[i]
                    if extend_logprob_start_len < extend_input_len:
                        num_input_logprobs = extend_input_len - extend_logprob_start_len
                        self.batch_result_processor.logprob_result_processor.add_input_logprob_return_values(
                            i,
                            req,
                            logits_output,
                            logprob_pt,
                            num_input_logprobs,
                            last_prefill_chunk=False,
                        )
                        logprob_pt += num_input_logprobs

                # In non-overlap-mode, KV is sent in process_prefill_chunk
                # Only send when req's sender is initialized
                if self.enable_overlap and not req.pending_bootstrap:
                    assert req.metadata_buffer_index >= 0, (
                        f"Req {req.rid} does not have metadata buffer allocated"
                    )
                    self.send_kv_chunk(req, last_chunk=False, end_idx=req.tmp_end_idx)
                req.time_stats.set_last_chunked_prefill_finish_time()

        if auxiliary_output is not None:
            self.batch_result_processor.consume_auxiliary_output(
                batch,
                auxiliary_output,
                auxiliary_output_starts,
            )

        if aborted_reqs:
            self.output_streamer.stream_output(
                aborted_reqs,
                any(req.return_logprob for req in aborted_reqs),
            )

        can_run_cuda_graph = result.can_run_cuda_graph
        self.metrics_reporter.report_prefill_stats(
            batch=batch,
            prefill_stats=batch.prefill_stats,
            can_run_cuda_graph=can_run_cuda_graph,
            dp_cooperation_info=batch.dp_cooperation_info,
        )
        self.maybe_send_health_check_signal()

    def _prefill_fake_free_kv(self: Scheduler, req) -> bool:
        """Benchmark single-press: free a fake-bootstrap request's KV right after
        prefill. aiakperf --fake-bootstrap sets req.bootstrap_host=FAKE_BOOTSTRAP_HOST
        and no decode peer ever pulls the KV, so it would otherwise leak until an
        impossible transfer concludes (oversubscribes SWA -> stall). Gated by env
        SGLANG_PREFILL_FAKE_FREE_KV=1; never affects real PD (real bootstrap_host).
        """
        return (
            self._fake_transfer_bypass
            and req.bootstrap_host == FAKE_BOOTSTRAP_HOST
        )

    @scheduler_stage_method(SCHEDULER_STAGE_PROCESS_QUEUE)
    def process_disagg_prefill_inflight_queue(
        self: Scheduler,
        rids_to_check: Optional[List[str]] = None,
        prepolled: Optional[List[int]] = None,
    ) -> List[Req]:
        """
        Poll the requests in the middle of transfer. If done, return the request.
        rids_to_check: For PP, on rank > 0, check the rids from the previous rank has consensus with the current rank.
        prepolled: poll states already MIN all-reduced together with the bootstrap
            and waiting queues by fuse_bootstrap_polls (one all_reduce instead of
            three). Positionally aligned with disagg_prefill_inflight_queue.
        """
        if len(self.disagg_prefill_inflight_queue) == 0:
            return []

        done_reqs = []

        polls = (
            prepolled
            if prepolled is not None
            and len(prepolled) == len(self.disagg_prefill_inflight_queue)
            else poll_and_all_reduce_attn_cp_tp_group(
                [req.disagg_kv_sender for req in self.disagg_prefill_inflight_queue],
                self.attn_cp_cpu_group,
                self.attn_tp_cpu_group,
            )
        )

        undone_reqs: List[Req] = []
        # Check .poll() for the reqs in disagg_prefill_inflight_queue. If Success, respond to the client and remove it from the queue
        with _batched_kv_release(self.token_to_kv_pool_allocator):
            for req, poll in zip(self.disagg_prefill_inflight_queue, polls):
                if self._prefill_fake_free_kv(req):
                    # Synthetic single-press: no decode peer pulls this KV; free it now.
                    if not isinstance(req.finished_reason, FINISH_ABORT):
                        req.finished_reason = FINISH_LENGTH(length=0)
                    release_kv_cache(req, self.tree_cache, is_insert=False)
                    try:
                        req.disagg_kv_sender.clear()
                    except Exception:
                        pass
                    done_reqs.append(req)
                    continue
                if rids_to_check is not None:
                    if req.rid not in rids_to_check:
                        undone_reqs.append(req)
                        continue

                    # In PP mode, the previous rank may have reached a terminal
                    # state (Success/Failed) while this rank's local poll is still
                    # in a transient state due to clock skew or propagation delay.
                    # Treat non-terminal states as undone instead of crashing.
                    if poll not in (
                        KVPoll.Success,
                        KVPoll.Failed,
                    ):
                        logger.warning_once(
                            f"PP rank {self.ps.pp_rank}: unexpected poll state {poll} for rid {req.rid} "
                            f"from consensus; treating as undone",
                        )
                        undone_reqs.append(req)
                        continue

                if req.pending_bootstrap:
                    # Parked: prefill finished before bootstrap completed.
                    if self.handle_pending_bootstrap(req, poll):
                        self.send_kv_chunk(req, last_chunk=True)
                        undone_reqs.append(req)
                    elif poll != KVPoll.Failed:
                        undone_reqs.append(req)
                    continue

                if poll in [KVPoll.WaitingForInput, KVPoll.Transferring]:
                    # todo: set Transferring correctly in backend
                    undone_reqs.append(req)
                elif poll == KVPoll.Success:  # transfer done
                    if not isinstance(req.finished_reason, FINISH_ABORT):
                        req.finished_reason = FINISH_LENGTH(length=0)
                    release_kv_cache(req, self.tree_cache)  # unlock the tree
                    # FIXME: clean up req's data in transfer engine
                    req.disagg_kv_sender.clear()
                    done_reqs.append(req)
                    req.time_stats.set_prefill_kv_transfer_finish_time()
                elif poll == KVPoll.Failed:
                    self.handle_inflight_transfer_failure(req)
                    done_reqs.append(req)
                else:
                    raise RuntimeError(
                        f"Unexpected poll state {poll} for req {req.rid} in inflight queue"
                    )

        for req in done_reqs:
            req.time_stats.set_completion_time()

        for req in done_reqs:
            if isinstance(req.finished_reason, FINISH_ABORT):
                continue
            if req.bootstrap_host == FAKE_BOOTSTRAP_HOST:
                continue
            kv_mgr = getattr(req.disagg_kv_sender, "kv_mgr", None)
            if kv_mgr and getattr(kv_mgr, "is_dummy_cp_rank", False):
                continue
            metrics = req.time_stats.compute_and_observe_kv_transfer_metrics(
                req.disagg_kv_sender.get_transfer_metric()
            )
            if metrics:
                # Update last-value for REST API
                if "latency_ms" in metrics:
                    self.metrics_reporter.kv_transfer_latency_ms = metrics["latency_ms"]
                if "speed_gb_s" in metrics:
                    self.metrics_reporter.kv_transfer_speed_gb_s = metrics["speed_gb_s"]

        # Stream requests which have finished transfer
        self.output_streamer.stream_output(
            done_reqs,
            any(req.return_logprob for req in done_reqs),
            None,
        )
        for req in done_reqs:
            req: Req

            maybe_release_metadata_buffer(
                req, self.req_to_metadata_buffer_idx_allocator
            )

        self.disagg_prefill_inflight_queue = undone_reqs

        return done_reqs

    def handle_inflight_transfer_failure(
        self: Scheduler, req: Req
    ) -> Optional[Exception]:
        """Conclude an inflight request whose KV transfer failed."""
        error_message = (
            f"Prefill transfer failed for request rank={self.ps.tp_rank} "
            f"{req.rid=} {req.bootstrap_room=}"
        )
        exc: Optional[Exception] = None
        try:
            req.disagg_kv_sender.failure_exception()
        except Exception as e:
            exc = e
            error_message += f" with exception {e}"
        # Mute error message for propagated exceptions to avoid duplicate logging
        if getattr(exc, "is_from_another_rank", False):
            logger.debug(error_message)
        else:
            logger.warning(error_message)
        req.time_stats.trace_ctx.abort(abort_info={"reason": error_message})
        release_kv_cache(req, self.tree_cache)  # unlock the tree
        if not isinstance(req.finished_reason, FINISH_ABORT):
            prepare_abort(
                req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR
            )
        if self.metrics_reporter.enable_metrics:
            self.metrics_collector.increment_transfer_failed_reqs()
        return exc

    def clear_pending_chunk_send(self: Scheduler, req: Req) -> None:
        """Drop `req` from the sent-but-unconcluded chunk set.

        Every path that retires a request without a `last_chunk=True` send must
        call this: a stale entry holds the unified-memory compaction gate closed
        for the process lifetime.
        """
        self.disagg_prefill_pending_chunk_rids.discard(req.rid)

    def _retire_aborted_prefill_result(self: Scheduler, req: Req) -> bool:
        """Release an aborted request when its last prefill result is safe."""
        self.clear_pending_chunk_send(req)
        owns_resources = (
            req.kv.holds_kv or req.kv.holds_mamba or req.metadata_buffer_index >= 0
        )
        if not owns_resources:
            # A bootstrap failure or earlier abort already retired it.
            return False

        sender = req.disagg_kv_sender
        if sender is not None:
            try:
                sender.abort()
            except Exception:
                # Transport notification is best effort; local ownership must
                # still be released or the next idle invariant check will fail.
                logger.exception("Failed to notify KV sender of abort for %s", req.rid)

        if req.to_finish is not None and not req.finished():
            req.update_finish_state()
        maybe_release_metadata_buffer(req, self.req_to_metadata_buffer_idx_allocator)
        req.pending_bootstrap = False
        if self.enable_hicache_storage:
            self.tree_cache.release_aborted_request(req.rid)
        if req.kv.holds_kv or req.kv.holds_mamba:
            release_kv_cache(req, self.tree_cache, is_insert=False)
        return True

    def handle_bootstrap_failure(self: Scheduler, req: Req) -> None:
        self.clear_pending_chunk_send(req)
        error_message = (
            f"Prefill bootstrap failed for request rank={self.ps.tp_rank} "
            f"{req.rid=} {req.bootstrap_room=}"
        )
        is_propagated = False
        try:
            req.disagg_kv_sender.failure_exception()
        except Exception as e:
            error_message += f" with exception {e}"
            is_propagated = getattr(e, "is_from_another_rank", False)
        # Mute error message for propagated exceptions to avoid duplicate logging
        if is_propagated:
            logger.debug(error_message)
        else:
            logger.warning(error_message)
        req.time_stats.trace_ctx.abort(abort_info={"reason": error_message})
        if req.kv.holds_kv or req.kv.holds_mamba:
            release_kv_cache(req, self.tree_cache)
        maybe_release_metadata_buffer(req, self.req_to_metadata_buffer_idx_allocator)
        req.pending_bootstrap = False
        prepare_abort(req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
        self.output_streamer.stream_output([req], req.return_logprob)
        if self.metrics_reporter.enable_metrics:
            self.metrics_collector.increment_bootstrap_failed_reqs()
        if self.enable_hicache_storage:
            self.tree_cache.release_aborted_request(req.rid)

    def handle_pending_bootstrap(self: Scheduler, req: Req, poll: KVPoll) -> bool:
        """Return True when bootstrap is finalized and KV transfer can proceed."""
        if poll == KVPoll.Failed:
            self.handle_bootstrap_failure(req)
            return False
        elif poll == KVPoll.Bootstrapping:
            return False
        elif poll == KVPoll.WaitingForInput:
            if should_force_retry(req):  # test hook
                return False
            # Metadata buffer was allocated in pop_bootstrapped before
            # the request entered the waiting queue, so finalize should not fail.
            assert self.disagg_prefill_bootstrap_queue.finalize_bootstrap(req)
            return True
        else:
            raise RuntimeError(
                f"Unexpected poll state {poll} for req {req.rid} in handle_pending_bootstrap"
            )

    def check_bootstrap(self: Scheduler, req: Req) -> bool:
        """Check bootstrap status for an optimistic prefilled request.
        Returns True if bootstrap is finished."""
        if not req.pending_bootstrap:
            return True
        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender],
            self.attn_cp_cpu_group,
            self.attn_tp_cpu_group,
        )
        return self.handle_pending_bootstrap(req, polls[0])

    def process_prefill_chunk(
        self: Scheduler,
        last_batch: Optional[ScheduleBatch],
        running_batch: ScheduleBatch,
        *,
        deferred_finalize: Optional[DeferredFinalize] = None,
    ) -> None:
        """``deferred_finalize`` (threaded AHEAD loop only): instead of
        finalizing the chunk the previous forward computed (SWA checkpoint
        staging + tree insert / AttentionStore write / dedup / out-of-window SWA
        free), queue it there. That forward may still be launching on the worker
        with no WAR barrier yet, so those reads and frees of its in-flight KV
        must wait until the caller has collected it."""
        chunked_req_to_exclude = set()
        if (req := self.chunked_req) is not None:
            chunked_req_to_exclude.add(req)
            if deferred_finalize is not None:
                deferred_finalize.append(
                    _DeferredChunkFinalize(req=req, chunk_range=req.extend_range)
                )
                # The next chunk's admission needs the full computed prefix;
                # take it from the req's own row (the tree insert that would
                # normally provide it is deferred). The deferred finalize then
                # repoints it exactly as cache_unfinished_req does in-line.
                req.prefix_indices = self.req_to_token_pool.req_to_token[
                    req.kv.req_pool_idx, : req.extend_range.end
                ].to(dtype=torch.int64, copy=True)
            else:
                self._finalize_prefill_chunk(req, chunk_range=req.extend_range)

            if not self.check_bootstrap(req):
                if is_aborted(req):
                    # bootstrap failed
                    self.chunked_req = None
                elif self.has_bootstrapped_waiting_req():
                    # optimistic request yields to waiting requests
                    self.chunked_req = None
                    if not self.enable_overlap:
                        self.optimistic_release_and_requeue(req)
                # else: still bootstrapping, keep computing without sending
            elif self.enable_overlap:
                # Delay KV transfer to process_batch_result_disagg_prefill when overlap is enabled to ensure results are resolved
                req.tmp_end_idx = min(
                    req.extend_range.end,
                    len(req.origin_input_ids),
                )
            else:
                self.send_kv_chunk(req)

            if self.chunked_req is not None:
                running_batch.batch_is_full = False

        if last_batch and last_batch.forward_mode.is_extend():
            if last_batch.chunked_req:
                # In the context pipeline parallelism, after the last chunk, the current microbatch still track outdated chunked_req.
                # We need to discard it.
                chunked_req_to_exclude.add(last_batch.chunked_req)

            last_bs = last_batch.batch_size()
            last_batch.filter_batch(chunked_req_to_exclude=list(chunked_req_to_exclude))
            if last_batch.batch_size() < last_bs:
                running_batch.batch_is_full = False
    def _run_serial_deferred_finalize(
        self: Scheduler, deferred_finalize: DeferredFinalize
    ) -> List[_DeferredChunkFinalize]:
        """Run the deferred work that must land before the next forward is
        submitted; return the chunk finalizes that may instead overlap it."""
        overlap_finalize: List[_DeferredChunkFinalize] = []
        for item in deferred_finalize:
            if not isinstance(item, _DeferredChunkFinalize):
                item()
            elif self._chunk_finalize_keeps_row(
                item.req, chunk_range=item.chunk_range
            ):
                overlap_finalize.append(item)
            else:
                self._finalize_prefill_chunk(item.req, chunk_range=item.chunk_range)
        return overlap_finalize

    def _chunk_finalize_keeps_row(self: Scheduler, req: Req, *, chunk_range) -> bool:
        """Whether finalizing this chunk provably leaves the req's req_to_token
        row and every slot in it in place, so it may run after the next forward
        (which reads that row as its prefix) is submitted.

        The only part of a chunk finalize that touches what the next forward
        reads is the tree insert's dedup: where the inserted key overlaps
        existing nodes past ``cache_protected_len``, the insert frees the req's
        duplicate slots and the caller repoints its row at the tree's. So: walk
        the tree read-only and require no overlap past ``cache_protected_len``
        (an evicted node counts as overlap -- conservative). The rest of the
        finalize is safe to overlap: SWA staging and the AttentionStore write
        only read this chunk; the out-of-window SWA free stops a page below the
        sliding window the next chunk attends. Nothing else mutates the tree
        between this check and the finalize (scheduler thread only).

        The unfiltered chunk key is a superset of what the insert uses (an SWA
        branch insert only shortens it), so the check stays conservative.
        """
        from sglang.srt.mem_cache.radix_cache import RadixKey
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        if req.skip_radix_cache_insert:
            return True  # maybe_cache_unfinished_req inserts nothing
        tree = self.tree_cache
        if not isinstance(tree, UnifiedRadixCache):
            return False
        if tree.disable:
            return True  # cache_unfinished_req only snapshots prefix_indices
        if tree.tree_core.is_eagle or req.session is not None:
            # Bigram keys / streaming-session commits: not mirrored here.
            return False
        key = RadixKey(
            req.full_untruncated_fill_ids,
            req.extra_key,
            limit=chunk_range.end,
            cache_salt=req.cache_salt,
        ).page_aligned(tree.page_size)
        overlap_len = tree.tree_core.peek_insert_overlap_len(key)
        return overlap_len <= req.kv.cache_protected_len

    def _finalize_prefill_chunk(self: Scheduler, req: Req, *, chunk_range) -> None:
        """Commit the chunk of ``req`` ending at ``chunk_range.end``: stage its
        SWA checkpoints, then insert it into the tree (which also submits the
        AttentionStore write and frees out-of-window SWA).

        Deferred (threaded AHEAD loop), this runs after the next chunk was
        already admitted, so ``req.extend_range`` points one chunk ahead; both
        callees size the committed prefix by ``get_fill_ids()``, so rewind the
        range to the computed chunk for the duration. Safe even while the next
        forward runs on the worker: the forward path reads no req's
        extend_range (its per-req lengths are batch-level, and
        extend_input_len_per_req is precomputed before submit).
        """
        live_range = req.extend_range
        req.extend_range = chunk_range
        try:
            # Strictly before maybe_cache_unfinished_req: that call evicts SWA
            # beyond pre_len - max(sliding_window, page_size), which is exactly the
            # window these checkpoints need. The commit that follows submits them,
            # because cache_unfinished_req runs the AttentionStore write path.
            self.detect_and_may_copy_as_swa(req)
            maybe_cache_unfinished_req(req, self.tree_cache, chunked=True)
        finally:
            req.extend_range = live_range

    def maybe_send_cached_prefix_chunk(self: Scheduler, req: Req) -> None:
        if not envs.SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX.get():
            return

        # Staging sends into positional grid slots, so the early-send boundary
        # must stay stable across the request's batches: snapshot the at-rest
        # prefix on the first batch. Non-staging reads the live prefix.
        if self.enable_staging and req.early_send_prefix_end is None:
            req.early_send_prefix_end = max(
                0, len(req.prefix_indices) - req.host_hit_length
            )

        if req.pending_bootstrap:
            return

        # Device-resident prefix only; page-aligned so start_send_idx stays exact.
        cached_end = (
            req.early_send_prefix_end
            if self.enable_staging
            else len(req.prefix_indices) - req.host_hit_length
        )
        if cached_end <= req.start_send_idx:
            return
        if cached_end % self.token_to_kv_pool_allocator.page_size != 0:
            # DCP radix hits can end on a logical cache-page boundary that is
            # not a complete physical DCP page. The regular final send covers
            # the full range; only skip this optional early-send optimization.
            return
        # Early-send issues the KV read before this step's forward is enqueued,
        # but under overlap scheduling the PRIOR step's prefill forward may still
        # be writing these prefix pages on forward_stream. Record a completion
        # event now so the transfer worker can wait on those writes before the
        # RDMA read, instead of racing them.
        if self.enable_overlap:
            ev = torch.cuda.Event()
            ev.record(self.forward_stream)
            req.disagg_kv_sender._early_send_wait_event = ev
        self.send_kv_chunk(req, last_chunk=False, end_idx=cached_end)

    def _sendkv_pinned_pages(
        self: Scheduler, num_pages: int, kv_indices: torch.Tensor
    ) -> torch.Tensor:
        """Pinned staging buffer for one segment's page indices, reused across
        calls. Only safe while ``self._sendkv_gather_live`` is False (the
        previous send already drained its D2H); the segment loop enforces that
        by syncing before the buffer can be handed to the next send."""
        buf = self._sendkv_pinned
        if buf is None or buf.numel() < num_pages:
            buf = torch.empty(num_pages, dtype=torch.int32, pin_memory=True)
            self._sendkv_pinned = buf
        return buf

    def _sendkv_fake_transfer(self: Scheduler, req: Req) -> bool:
        """Fake-backend benchmark shortcut: the sender discards indices, so the
        payload builders and per-segment page-index D2H (the dominant
        scheduler-thread cost, measurable even in the nocache baseline) can be
        skipped entirely. Gated with _prefill_fake_free_kv so real PD keeps the
        full path; the sender's page counter still advances via token math."""
        return (
            self._fake_transfer_bypass
            and isinstance(req.disagg_kv_sender, FakeKVSender)
        )

    def send_kv_chunk(
        self: Scheduler,
        req: Req,
        last_chunk: bool = False,
        end_idx: Optional[int] = None,
    ) -> None:
        """
        Send a prefilled chunk to the decode server
        """
        page_size = self.token_to_kv_pool_allocator.page_size
        start_idx = req.start_send_idx
        transfer_input_len = len(req.origin_input_ids)
        end_idx = (
            end_idx
            if end_idx is not None
            else min(req.extend_range.end, transfer_input_len)
        )

        if not last_chunk:
            # if not the last chunk and the last page is partial, delay the last partial page to the next send
            end_idx = end_idx - end_idx % page_size
            if self.enable_staging:
                # Staging identifies chunks positionally against a uniform
                # prefetched grid, so non-last sends must end on a grid
                # boundary; the remainder rides with the next send.
                grid_tokens = staging_grid_tokens(
                    get_schedule().chunked_prefill_size, page_size
                )
                base = req.disagg_decode_prefix_len
                end_idx = base + ((end_idx - base) // grid_tokens) * grid_tokens

        if end_idx < start_idx:
            logger.debug(
                "send_kv_chunk skip: rid=%s start_send_idx=%s end_idx=%s",
                req.rid,
                start_idx,
                end_idx,
            )
            return

        if self._sendkv_fake_transfer(req):
            # FakeKVSender.send only counts pages; drive the same accounting
            # from token math instead of gathering req_to_token rows to host.
            # curr_idx must reach num_pages for poll() to conclude, and the
            # last chunk must land exactly at num_pages (it is the equality
            # poll checks), so the final call closes any residual gap.
            total_pages = kv_to_page_num(
                transfer_input_len - req.disagg_decode_prefix_len, page_size
            )
            new_idx = min(
                req.disagg_kv_sender.curr_idx
                + (end_idx - start_idx) // page_size,
                total_pages,
            )
            if last_chunk and new_idx < total_pages:
                new_idx = total_pages
            req.disagg_kv_sender.curr_idx = new_idx
            req.start_send_idx = end_idx
            if last_chunk:
                self.disagg_prefill_pending_chunk_rids.discard(req.rid)
            else:
                self.disagg_prefill_pending_chunk_rids.add(req.rid)
            return

        state_indices: Optional[List] = None
        if last_chunk:
            self.disagg_metadata_buffers.set_buf(req)

            # Most state payloads read token-pool rows and should match the KV
            # range actually materialized on prefill. C128 state is request
            # scoped, so its transfer index must use the logical input length
            # that decode used to register the destination row.
            seq_len = min(req.extend_range.end, transfer_input_len)
            c128_seq_len = transfer_input_len

            def _mamba_payload():
                return [
                    self.req_to_token_pool.translate_mamba_indices(
                        self.req_to_token_pool.req_index_to_mamba_index_mapping[
                            req.kv.req_pool_idx
                        ]
                    )
                    .cpu()
                    .numpy()
                ]

            def _swa_payload():
                window_size = self.sliding_window_size
                window_start = max(req.disagg_decode_prefix_len, seq_len - window_size)
                window_start = (window_start // page_size) * page_size
                window_kv_indices_full = self.req_to_token_pool.req_to_token[
                    req.kv.req_pool_idx, window_start:seq_len
                ]
                window_kv_indices_swa = (
                    self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                        window_kv_indices_full
                    )
                )
                return kv_to_page_indices(window_kv_indices_swa, page_size)

            def _full_kv_pages_payload():
                kv_indices_full = self.req_to_token_pool.req_to_token[
                    req.kv.req_pool_idx, :seq_len
                ]
                return kv_to_page_indices(kv_indices_full, page_size)

            def _dsa_tail_payload():
                return get_dsa_tail_state_indices(
                    self.token_to_kv_pool_allocator.get_kvcache(),
                    req.kv.req_pool_idx,
                    seq_len,
                )

            def _qsa_pending_payload():
                # Raw index-K/RoPE state is one full compression-group ring per
                # request, addressed by the request-pool slot rather than KV pages.
                return get_qsa_pending_state_indices(req)

            def _swa_ring_payload():
                # Unified_kv SWA ring rows (req_pool_idx*ring_stride + pos%ring_stride)
                # for the last `window` positions, in ascending position order so
                # decode (its own req_pool_idx) matches positionally.
                _pool = self.token_to_kv_pool_allocator.get_kvcache()
                ring_stride = _pool.unified_swa_ring_size
                window_size = _pool.unified_swa_window
                window_start = max(0, seq_len - window_size)
                positions = np.arange(window_start, seq_len, dtype=np.int64)
                state_slot = int(req.kv.req_pool_idx)
                ring_rows = state_slot * ring_stride + (positions % ring_stride)
                return ring_rows.astype(np.int32)

            def _c128_state_payload():
                return get_dsv4_request_state_indices(
                    self.token_to_kv_pool_allocator.get_kvcache(),
                    int(req.kv.req_pool_idx),
                    c128_seq_len,
                )

            state_types = (
                self.disagg_prefill_bootstrap_queue.kv_manager.kv_args.state_types
            )
            payloads = {
                StateType.MAMBA: _mamba_payload,
                StateType.QSA_PENDING: _qsa_pending_payload,
                StateType.QSA_COMPRESSED: _full_kv_pages_payload,
                StateType.SWA: _swa_payload,
                StateType.DSA: _full_kv_pages_payload,
                StateType.DSA_TAIL: _dsa_tail_payload,
                StateType.MINIMAX_INDEX_K: _full_kv_pages_payload,
                StateType.SWA_RING: _swa_ring_payload,
                StateType.DSV4_REQUEST_STATE: _c128_state_payload,
                StateType.BLOCK_SCALE: _full_kv_pages_payload,
                StateType.BLOCK_SCALE_SWA: _swa_payload,
            }
            if _is_npu and isinstance(
                self.token_to_kv_pool_allocator.get_kvcache(),
                DeepSeekV4TokenToKVPool,
            ):
                from sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (
                    dsv4_state_payloads,
                )

                payloads.update(
                    dsv4_state_payloads(
                        self.req_to_token_pool,
                        req.kv.req_pool_idx,
                        seq_len,
                        page_size,
                        prefix_len=req.disagg_decode_prefix_len,
                    )
                )
            state_indices = [
                payloads[st]() if st in payloads else None for st in state_types
            ]

        if self.enable_staging:
            # One sender.send per grid slot; the sender's cumulative page
            # counter marks only the final sub-send of the final chunk as
            # is_last, routing aux/state correctly.
            segments = compute_grid_segments(
                start_idx,
                end_idx,
                req.disagg_decode_prefix_len,
                staging_grid_tokens(get_schedule().chunked_prefill_size, page_size),
            )
        else:
            segments = [(start_idx, end_idx)]

        # The page-index D2H runs on a side stream, not the schedule stream:
        # send_kv runs in process_batch_result behind the WAR barrier, so a
        # pageable .cpu() there blocks until the in-flight forward drains
        # (measured up to 61.8ms per call). This lane only waits on
        # prep_done_event (recorded after this batch's prep writes, before the
        # barrier), so the sync below ends once its own gather completes.
        gather_stream = None
        if (
            self.enable_overlap
            and self.swa_copy_stream is not None
            and not self._sendkv_gather_live
        ):
            gather_stream = self.swa_copy_stream
            gather_stream.wait_event(self.prep_done_event)
            self._sendkv_gather_done = self.device_module.Event()
            self._sendkv_gather_live = True

        for seg_start, seg_end in segments:
            is_final_segment = seg_end == end_idx
            kv_indices = self.req_to_token_pool.req_to_token[
                req.kv.req_pool_idx, seg_start:seg_end
            ]
            # Unified memory: req_to_token holds VIRTUAL ids; the transfer needs
            # physical ones. Per segment, since each is its own gather.
            kv_indices = (
                self.token_to_kv_pool_allocator.translate_kv_indices_for_transfer(
                    kv_indices
                )
            )
            if gather_stream is not None:
                num_pages = kv_to_page_num(kv_indices.numel(), page_size)
                host_buf = self._sendkv_pinned_pages(num_pages, kv_indices)
                with self.device_module.stream(gather_stream):
                    page_indices = kv_to_page_indices_pinned(
                        kv_indices,
                        page_size,
                        stream=gather_stream,
                        host_buf=host_buf,
                    )
            else:
                page_indices = kv_to_page_indices(kv_indices, page_size)
            segment_is_last = last_chunk and is_final_segment
            if not req.disagg_kv_sender.should_send_kv_chunk(
                len(page_indices), segment_is_last
            ):
                continue
            if gather_stream is not None:
                # The pinned view is only valid once the side-stream D2H
                # completes; the record+sync below fences every segment's copy.
                with self.device_module.stream(gather_stream):
                    self._sendkv_gather_done.record()
                self._sendkv_gather_done.synchronize()
                gather_stream = None
                self._sendkv_gather_live = False
                # Snapshot before send(): the transfer queue keeps the ndarray
                # beyond this call, but the pinned buffer is reused next call.
                page_indices = page_indices.copy()
            req.disagg_kv_sender.send(
                page_indices,
                state_indices if segment_is_last else None,
                num_kv_tokens=seg_end - seg_start,
            )
        req.start_send_idx = end_idx
        # A last chunk needs no entry: every `last_chunk=True` call site has
        # already put the request on `disagg_prefill_inflight_queue`.
        if last_chunk:
            self.disagg_prefill_pending_chunk_rids.discard(req.rid)
        else:
            self.disagg_prefill_pending_chunk_rids.add(req.rid)

    def optimistic_release_and_requeue(self: Scheduler, req: Req) -> None:
        """Release KV cache and requeue an optimistic prefill request."""
        max_attempts = get_disagg().optimistic_prefill_attempts
        maybe_cache_unfinished_req(req, self.tree_cache)
        release_kv_cache(req, self.tree_cache)
        req.reset_for_retract()
        req.output_ids = array("q")
        req.start_send_idx = 0
        self.clear_pending_chunk_send(req)  # re-sends from scratch
        req.tmp_end_idx = -1
        req.disagg_decode_prefix_len = 0
        req.early_send_prefix_end = None
        req.hidden_states_tensor = None
        req.output_dsa_topk_indices = None
        req.pending_bootstrap = True
        req.time_stats.reset_prefill_retry_time()
        if req.prefill_attempt_count >= max_attempts:
            logger.info(
                f"Req {req.rid} exhausted optimistic prefill attempts "
                "falling back to bootstrap queue"
            )
            # Reset it so the next real bootstrap done can be recorded.
            req.time_stats.bootstrap_done_time = 0.0
            self.disagg_prefill_bootstrap_queue.queue.append(req)
        else:
            req.prefill_attempt_count += 1
            logger.info(
                f"Req {req.rid} optimistic prefill yielded "
                f"({req.prefill_attempt_count}/{max_attempts} attempts used)"
            )
            if self.metrics_reporter.enable_metrics:
                self.metrics_collector.increment_prefill_retries(1)
            req.time_stats.set_wait_queue_entry_time()
            req.arrival_processed_tokens = self.processed_tokens_counter
            self.waiting_queue.insert(0, req)
