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

import hashlib
import logging
import time
from array import array
from collections import deque
from http import HTTPStatus
from typing import TYPE_CHECKING, List, Optional, Tuple

import msgspec
import numpy as np
import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.utils import (
    FAKE_BOOTSTRAP_HOST,
    DisaggregationMode,
    KVClassType,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    get_dsv4_c128_state_indices,
    get_kv_class,
    get_transfer_draft_kv_layer_ids,
    get_transfer_kv_layer_ids,
    is_aborted,
    is_dsv4_c128_online_enabled,
    is_mla_backend,
    poll_and_all_reduce_attn_cp_tp_group,
    poll_and_all_reduce_pp,
    prepare_abort,
    setup_state_kv_args,
)
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
    kv_to_page_num,
    maybe_cache_unfinished_req,
    release_kv_cache,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.observability.req_time_stats import set_schedule_time_batch
from sglang.srt.utils.nvtx_utils import scheduler_nvtx_method

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler
    from sglang.srt.mem_cache.memory_pool import KVCache

logger = logging.getLogger(__name__)


def should_force_retry(req: Req) -> bool:
    """Test hook to force a request into optimistic prefill retry."""
    retry_prob = envs.SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB.get()
    # Force only before/during the first attempt (count is 1 while it runs).
    if retry_prob <= 0 or req.prefill_attempt_count > 1 or req.is_retracted:
        return False

    digest = hashlib.sha256(str(req.rid).encode()).digest()
    return int.from_bytes(digest[:8], "big") < retry_prob * 2**64


class PendingFinalKV(msgspec.Struct, omit_defaults=True):
    """A request whose final prefill KV/state is waiting to be sent early.

    Enqueued right after `_pp_launch_batch` on non-PP0 ranks; flushed from the
    next iteration's CPU phase once `forward_done` has actually completed.
    `end_idx` is snapshotted at enqueue time because `req.extend_range` may be
    rewritten by later scheduling before the flush runs.
    """

    req: Req
    end_idx: int
    forward_done: torch.Event


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
            server_args = self.scheduler.server_args
            page_size = self.scheduler.token_to_kv_pool_allocator.page_size
            cps = server_args.chunked_prefill_size or 8192
            # Staging slices each send into a fixed page-aligned grid, so an
            # unbounded (-1) or non-page-aligned chunk size has no valid grid.
            if cps <= 0 or cps % page_size != 0:
                raise RuntimeError(
                    f"SGLANG_DISAGG_STAGING_BUFFER requires a positive "
                    f"chunked_prefill_size that is a multiple of page_size "
                    f"({page_size}); got {server_args.chunked_prefill_size}."
                )
            if self.pp_size > 1:
                # Staging writer accounting has no pp dimension.
                raise RuntimeError(
                    "SGLANG_DISAGG_STAGING_BUFFER does not support pp_size > 1."
                )
            if server_args.enable_prefill_context_parallel:
                # CP rewrites index_slice per rank, breaking the chunk grid.
                raise RuntimeError(
                    "SGLANG_DISAGG_STAGING_BUFFER does not support "
                    "prefill context parallelism."
                )
        self.kv_manager = self._init_kv_manager()

    def _init_kv_manager(self) -> CommonKVManager:
        kv_args_class = get_kv_class(self.transfer_backend, KVClassType.KVARGS)
        kv_args = kv_args_class()
        kv_args.engine_rank = self.tp_rank
        kv_args.pp_rank = self.pp_rank
        kv_args.system_dp_rank = self.scheduler.ps.dp_rank
        layer_shard_enabled = getattr(
            self.token_to_kv_pool, "layer_shard_enabled", False
        )
        layer_shard_rank = getattr(self.token_to_kv_pool, "layer_shard_rank", None)
        layer_shard_size = getattr(self.token_to_kv_pool, "layer_shard_size", 1)
        transfer_draft_cache = (
            self.pp_size <= 1 or self.pp_rank == self.pp_size - 1
        ) and (not layer_shard_enabled or layer_shard_rank == layer_shard_size - 1)
        kv_args.prefill_start_layer = (
            getattr(
                self.token_to_kv_pool,
                "layer_shard_start",
                self.token_to_kv_pool.start_layer,
            )
            if layer_shard_enabled
            else self.token_to_kv_pool.start_layer
        )
        kv_args.mla_compression_ratios = None
        kv_data_ptrs, kv_data_lens, kv_item_lens = (
            self.token_to_kv_pool.get_contiguous_buf_infos()
        )
        kv_layer_ids = get_transfer_kv_layer_ids(
            self.token_to_kv_pool, len(kv_data_ptrs)
        )
        kv_args.prefill_end_layer = (
            kv_args.prefill_start_layer + len(kv_data_ptrs)
            if layer_shard_enabled
            else getattr(self.token_to_kv_pool, "end_layer", None)
        )

        # Boundary before the draft pool's entries are appended below. Draft
        # entries take the head-sliced replicated transfer path under PD DCP.
        num_mla_kv_entries = len(kv_data_ptrs)
        draft_kv_head_num = 0
        if self.draft_token_to_kv_pool is not None and transfer_draft_cache:
            # We should also transfer draft model kv cache. The indices are
            # always shared with a target model.
            draft_kv_data_ptrs, draft_kv_data_lens, draft_kv_item_lens = (
                self.draft_token_to_kv_pool.get_contiguous_buf_infos()
            )
            kv_layer_ids += get_transfer_draft_kv_layer_ids(len(draft_kv_data_ptrs))
            kv_data_ptrs += draft_kv_data_ptrs
            kv_data_lens += draft_kv_data_lens
            kv_item_lens += draft_kv_item_lens
            draft_kv_head_num = self.draft_token_to_kv_pool.head_num

        kv_args.num_mla_kv_entries = num_mla_kv_entries
        kv_args.draft_kv_head_num = draft_kv_head_num

        kv_args.kv_data_ptrs = kv_data_ptrs
        kv_args.kv_data_lens = kv_data_lens
        kv_args.kv_item_lens = kv_item_lens
        kv_args.kv_layer_ids = (
            kv_layer_ids if len(kv_layer_ids) == len(kv_data_ptrs) else []
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
        kv_args.ib_device = self.scheduler.server_args.disaggregation_ib_device
        kv_args.gpu_id = self.scheduler.ps.gpu_id

        req_to_token_pool = getattr(self.scheduler, "req_to_token_pool", None)
        setup_state_kv_args(
            kv_args,
            self.token_to_kv_pool,
            self.draft_token_to_kv_pool if transfer_draft_cache else None,
            self.scheduler.model_config.num_hidden_layers,
            req_to_token_pool=req_to_token_pool,
        )

        if isinstance(self.token_to_kv_pool, DeepSeekV4TokenToKVPool):
            # V4's KVCache is organized by compression-ratio
            # buckets rather than by layer.
            kv_args.mla_compression_ratios = list(
                self.token_to_kv_pool.compression_ratios
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
            if hasattr(kv_pool, "k_buffer") and hasattr(kv_pool, "v_buffer"):
                kv_manager.set_kv_buffer_tensors(
                    kv_pool.k_buffer,
                    kv_pool.v_buffer,
                    kv_pool.page_size,
                )
        if draft_kv_head_num:
            # The draft pool is head-sharded on the decode side, so its transfer
            # needs a GPU gather to stay descriptor-efficient; hand over the pool
            # tensors in the same order their pointers were registered above.
            kv_manager.set_draft_kv_entry_tensors(
                self.draft_token_to_kv_pool._pd_registerable_tensors()
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
                    req.prefill_attempt_count
                    < self.scheduler.server_args.optimistic_prefill_attempts
                    and not req.is_retracted  # engine paused
                ):
                    if not self.ensure_metadata_buffer(req):
                        continue  # no more metadata buffer
                    req.prefill_attempt_count += 1
                    bootstrapped_reqs.append(req)
                    indices_to_remove.add(i)
                    req.time_stats.set_wait_queue_entry_time()
            elif poll == KVPoll.WaitingForInput:
                if should_force_retry(req):  # skip checking for testing
                    if not self.ensure_metadata_buffer(req):
                        continue  # no more metadata buffer
                    req.prefill_attempt_count += 1
                elif not self.finalize_bootstrap(req):
                    if self.scheduler._pp_pd_sideband_bootstrap:
                        # The admission table comes from PP0, which already
                        # reserved one metadata credit per admitted rid, and every
                        # rank must admit exactly that set. Silently skipping this
                        # req would diverge this rank's batch from the others', so
                        # surface the broken quota lock-step instead.
                        raise RuntimeError(
                            f"No metadata buffer for req {req.rid} admitted by the "
                            f"PP bootstrap consensus (pp_rank={self.pp_rank}, "
                            "available_size="
                            f"{self.req_to_metadata_buffer_idx_allocator.available_size()})"
                            "; the metadata buffer quota is out of lock-step "
                            "across PP ranks."
                        )
                    continue
                bootstrapped_reqs.append(req)
                indices_to_remove.add(i)
                req.time_stats.set_wait_queue_entry_time()
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

    def get_ready_bootstrapped_rids_for_pp(self) -> Tuple[List[str], List[str]]:
        """Return ordered PP candidates without reserving local resources."""
        good_rids: List[str] = []
        failed_rids: List[str] = []
        if len(self.queue) == 0:
            return good_rids, failed_rids

        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender for req in self.queue],
            self.scheduler.attn_cp_cpu_group,
            self.scheduler.attn_tp_cpu_group,
        )
        metadata_credits = self.req_to_metadata_buffer_idx_allocator.available_size()

        for req, poll in zip(self.queue, polls):
            if poll == KVPoll.Failed:
                failed_rids.append(req.rid)
            elif poll == KVPoll.WaitingForInput:
                metadata_cost = 1 if req.metadata_buffer_index < 0 else 0
                if metadata_cost > metadata_credits:
                    break
                metadata_credits -= metadata_cost
                good_rids.append(req.rid)
            elif poll == KVPoll.Bootstrapping:
                continue
            else:
                raise RuntimeError(
                    f"Unexpected poll state {poll} for req {req.rid} "
                    "in get_ready_bootstrapped_rids_for_pp"
                )

        return good_rids, failed_rids

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

    @scheduler_nvtx_method("scheduler.get_next_batch_to_run")
    def get_next_disagg_prefill_batch_to_run(
        self: Scheduler,
        running_batch: ScheduleBatch,
        last_batch: Optional[ScheduleBatch],
    ) -> NextBatchPlan:
        self.process_pending_chunked_abort()

        # HACK (byronhsu): reset the batch_is_full flag because we never enter update_running_batch which resets it
        # Otherwise, it hangs under high concurrency
        running_batch.batch_is_full = False

        self.resolve_waiting_queue_bootstrap()

        self.process_prefill_chunk(last_batch=last_batch, running_batch=running_batch)

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
            # Receive requests
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue
            self.waiting_queue.extend(
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
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

            self.process_disagg_prefill_inflight_queue()

            # Update last_batch
            self.last_batch = batch

    @torch.no_grad()
    def event_loop_overlap_disagg_prefill(self: Scheduler) -> None:
        self.result_queue = deque()

        while True:
            # Receive requests
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue
            self.waiting_queue.extend(
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
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

            self.process_disagg_prefill_inflight_queue()

            # Run sample of the current batch
            # It depends on the result of the last batch (e.g., grammar), so we run it after the last batch is processed.
            self.launch_batch_sample_if_needed(batch_result, batch)

            # Update last_batch
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
        if result.routed_experts_output is not None:
            result.routed_experts_output.finalize()
            result.routed_experts_output = None
        if result.indexer_topk_output is not None:
            result.indexer_topk_output.finalize()
            result.indexer_topk_output = None

        if self.enable_metrics:
            self.metrics_reporter.iter_forward_finish_time = time.time()
            run_batch_time = (
                self.metrics_reporter.iter_forward_finish_time
                - self.metrics_reporter.iter_forward_start_time
            )
            self.metrics_reporter.stats.run_batch_time = run_batch_time

        logprob_pt = 0
        # A skipped output comm hands back _pp_make_skip_output_result's
        # placeholder, which carries no draft input, while batch.spec_info still
        # holds what run_batch assigned at launch -- a real draft input on the
        # last rank, which samples even though it no longer post-processes its own
        # output row. So the identity only holds for a rank that consumed a real
        # output row (PP0 via _pp_prep_batch_result, or every rank with the
        # early-last-chunk path off). See
        # docs/design/pp-prefill-last-chunk-early-send-phase2.md section 3.1.
        if result.skipped_output_comm:
            assert result.next_draft_input is None
        else:
            assert batch.spec_info is result.next_draft_input
        # None on a skipped-output rank; only the is_eagle() branch below reads
        # it and the early path admits DSPARK alone.
        draft_input = result.next_draft_input
        # Transfer kv for prefill completed requests and add it into disagg_prefill_inflight_queue
        next_token_ids = result.next_token_ids.tolist()
        self.batch_result_processor.move_logprobs_to_cpu(
            batch=batch,
            logits_output=logits_output,
        )

        def advance_logprob_pt(i: int, req: Req) -> None:
            nonlocal logprob_pt
            if not req.return_logprob or extend_input_len_per_req is None:
                return
            extend_logprob_start_len = extend_logprob_start_len_per_req[i]
            extend_input_len = extend_input_len_per_req[i]
            if extend_logprob_start_len < extend_input_len:
                logprob_pt += extend_input_len - extend_logprob_start_len

        for i, (req, next_token_id) in enumerate(
            zip(batch.reqs, next_token_ids, strict=True)
        ):
            if req.inflight_middle_chunks <= 0:
                req.time_stats.set_prefill_finished_time()

                # Test hook: exercise the release/requeue retry path.
                if req.pending_bootstrap and should_force_retry(req):
                    self.optimistic_release_and_requeue(req)
                    advance_logprob_pt(i, req)
                    continue

                req.output_ids.append(next_token_id)
                if req.pp_early_final_kv_done:
                    # _pp_flush_pending_final_kv already did the radix insert, the
                    # send and the inflight-queue append; next_token_id here is the
                    # placeholder zero from the skipped PP output ring.
                    advance_logprob_pt(i, req)
                    continue
                # Still queued: this post-process synchronized the forward that the
                # flush's non-blocking query had not yet seen, so the regular path
                # below sends it. Cancel the queue entry to keep the send single.
                self._pp_drop_pending_final_kv(req)
                maybe_cache_unfinished_req(req, self.tree_cache)
                self.disagg_prefill_inflight_queue.append(req)
                if self.spec_algorithm.is_eagle() and draft_input is not None:
                    req.output_topk_p = draft_input.topk_p[i]
                    req.output_topk_index = draft_input.topk_index[i]
                    req.hidden_states_tensor = (
                        draft_input.hidden_states[i].cpu().clone()
                    )
                    dsa_topk_indices = batch.spec_info.dsa_topk_indices
                    if dsa_topk_indices is not None:
                        req.output_dsa_topk_indices = dsa_topk_indices[i].cpu().clone()
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
                if not req.pending_bootstrap:
                    # _pp_pd_owns_aux is False only on non-PP0 ranks with the
                    # early-last-chunk switch on: their output_ids[0] is the
                    # placeholder zero, so they must never write the aux row.
                    self.send_kv_chunk(
                        req, last_chunk=True, send_aux=self._pp_pd_owns_aux
                    )
                req.time_stats.set_prefill_transfer_queue_entry_time()
                if self.enable_metrics:
                    self.metrics_collector.observe_prompt_calculation_time(
                        req.time_stats.prefill_transfer_queue_entry_time
                        - req.time_stats.forward_entry_time
                    )

                # Only the aux owner holds the real sampled token; a non-owner's
                # next_token_id is _pp_make_skip_output_result's placeholder zero,
                # which a real xgrammar rejects (single-rank prepare_abort, so the
                # PP ranks disagree on the request's fate) and a strict-thinking
                # wrapper silently mis-advances. The prefill-side advance has no
                # downstream consumer -- decode rebuilds and advances its own FSM
                # from the real token in process_prebuilt. See
                # docs/design/pp-prefill-last-chunk-early-send-phase2.md section 4.
                if req.grammar is not None and self._pp_pd_owns_aux:
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
                if req.pending_bootstrap and not still_chunking:
                    self.optimistic_release_and_requeue(req)
                    advance_logprob_pt(i, req)
                    req.time_stats.set_last_chunked_prefill_finish_time()
                    continue

                # Optimistic bootstrap can fail while this overlapped chunk is
                # already running. Drop aborted chunks instead of sending KV.
                if is_aborted(req):
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
                    assert (
                        req.metadata_buffer_index >= 0
                    ), f"Req {req.rid} does not have metadata buffer allocated"
                    self.send_kv_chunk(req, last_chunk=False, end_idx=req.tmp_end_idx)
                req.time_stats.set_last_chunked_prefill_finish_time()

        can_run_cuda_graph = result.can_run_cuda_graph
        self.metrics_reporter.report_prefill_stats(
            batch=batch,
            prefill_stats=batch.prefill_stats,
            can_run_cuda_graph=can_run_cuda_graph,
            dp_cooperation_info=batch.dp_cooperation_info,
        )

    def process_disagg_prefill_inflight_queue(
        self: Scheduler,
        transfer_status: Optional[Tuple[List[str], List[str]]] = None,
    ) -> List[Req]:
        """
        Poll the requests in the middle of transfer. If done, return the request.
        transfer_status: For PP, the consensus success and failure request ids.
        """
        if len(self.disagg_prefill_inflight_queue) == 0:
            return []

        done_reqs = []
        success_rids = set(transfer_status[0]) if transfer_status is not None else set()
        failed_rids = set(transfer_status[1]) if transfer_status is not None else set()

        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender for req in self.disagg_prefill_inflight_queue],
            self.attn_cp_cpu_group,
            self.attn_tp_cpu_group,
        )

        undone_reqs: List[Req] = []
        # Check .poll() for the reqs in disagg_prefill_inflight_queue. If Success, respond to the client and remove it from the queue
        for req, poll in zip(self.disagg_prefill_inflight_queue, polls):
            if transfer_status is not None:
                if req.rid in failed_rids:
                    self.handle_inflight_transfer_failure(req)
                    done_reqs.append(req)
                    continue

                if req.rid not in success_rids:
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
                    self.send_kv_chunk(
                        req, last_chunk=True, send_aux=self._pp_pd_owns_aux
                    )
                    undone_reqs.append(req)
                elif poll != KVPoll.Failed:
                    undone_reqs.append(req)
                continue

            if poll in [KVPoll.WaitingForInput, KVPoll.Transferring]:
                # todo: set Transferring correctly in backend
                undone_reqs.append(req)
            elif poll == KVPoll.Success:  # transfer done
                release_kv_cache(req, self.tree_cache)  # unlock the tree
                if not isinstance(req.finished_reason, FINISH_ABORT):
                    req.finished_reason = FINISH_LENGTH(length=0)
                # FIXME: clean up req's data in transfer engine
                req.disagg_kv_sender.clear()
                done_reqs.append(req)
                req.time_stats.set_prefill_kv_transfer_finish_time()
                if self.enable_metrics:
                    self.metrics_collector.observe_kvcache_transfer_time(
                        req.time_stats.prefill_kv_transfer_finish_time
                        - req.time_stats.prefill_transfer_queue_entry_time
                    )
                    self.metrics_collector.observe_time_to_first_token(
                        req.time_stats.prefill_kv_transfer_finish_time
                        - req.time_stats.prefill_bootstrap_queue_entry_time
                    )
            elif poll == KVPoll.Failed:
                self.handle_inflight_transfer_failure(req)
                done_reqs.append(req)
            else:
                raise RuntimeError(
                    f"Unexpected poll state {poll} for req {req.rid} in inflight queue"
                )

        for req in done_reqs:
            req.time_stats.set_completion_time()
            if self.enable_metrics and not isinstance(
                req.finished_reason, FINISH_ABORT
            ):
                self.metrics_collector.observe_request_first_token_forward_time(
                    req.time_stats.get_request_first_token_forward_time()
                )

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

    def get_transferred_rids(
        self: Scheduler,
    ) -> Tuple[List[str], List[str]]:
        """
        Used by PP, inspect terminal transfer states without popping requests.
        """
        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender for req in self.disagg_prefill_inflight_queue],
            self.attn_cp_cpu_group,
            self.attn_tp_cpu_group,
        )

        success_rids: List[str] = []
        failed_rids: List[str] = []

        for req, poll in zip(self.disagg_prefill_inflight_queue, polls):
            if poll == KVPoll.Success:
                success_rids.append(req.rid)
            elif poll == KVPoll.Failed:
                failed_rids.append(req.rid)

        return success_rids, failed_rids

    def handle_bootstrap_failure(self: Scheduler, req: Req) -> None:
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
        if (
            req.req_pool_idx is not None
            or req.kv is not None
            or req.mamba_pool_idx is not None
        ):
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
        planned_chunked_action: Optional[str] = None,
    ) -> None:
        chunked_req_to_exclude = set()
        if (req := self.chunked_req) is not None:
            chunked_req_to_exclude.add(req)
            maybe_cache_unfinished_req(req, self.tree_cache, chunked=True)

            if planned_chunked_action == "abort":
                if not is_aborted(req):
                    self.handle_bootstrap_failure(req)
                self.chunked_req = None
                bootstrap_ready = False
            else:
                bootstrap_ready = self.check_bootstrap(req)

            if not bootstrap_ready and is_aborted(req):
                # bootstrap failed
                self.chunked_req = None
            elif planned_chunked_action == "yield":
                # Replay PP0's scheduler transition. Local bootstrap readiness
                # must not select a different request for this microbatch.
                self.chunked_req = None
                if not self.enable_overlap:
                    self.optimistic_release_and_requeue(req)
            elif not bootstrap_ready:
                if planned_chunked_action in ("resume", "hold"):
                    pass
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

    def _pp_enqueue_pending_final_kv(
        self: Scheduler, batch: ScheduleBatch, mb_id: int
    ) -> None:
        """Queue this microbatch's final-chunk requests for an early KV send.

        Only non-PP0 ranks reach the body (see `_pp_pd_early_send_final_kv`): with
        PP0 owning the aux row they no longer need the sampled token for the last
        chunk, so the KV/state can leave from the next iteration's CPU phase
        instead of waiting a full PP round for the output ring to come back.
        """
        if not self._pp_pd_early_send_final_kv or not batch.forward_mode.is_extend():
            return

        # KV pages must not be read by RDMA before the forward that writes them;
        # the flush gates on this event.
        forward_done = self.mb_metadata[mb_id].forward_done
        for req in batch.reqs:
            # Inverse of process_batch_result_disagg_prefill's still_chunking
            # test: this chunk reaches the end of the input and no later chunk of
            # the request was scheduled.
            if (
                req is self.chunked_req
                or req.extend_range is None
                or req.extend_range.end < len(req.origin_input_ids)
            ):
                continue
            # Parked requests keep the handle_pending_bootstrap resend path.
            if req.pending_bootstrap or is_aborted(req):
                continue
            self.disagg_prefill_pending_final_kv.append(
                PendingFinalKV(
                    req=req,
                    # Snapshot: later scheduling may rewrite req.extend_range.
                    end_idx=min(req.extend_range.end, len(req.origin_input_ids)),
                    forward_done=forward_done,
                )
            )

    def _pp_flush_pending_final_kv(self: Scheduler) -> None:
        """Send the final KV/state of every queued request whose forward is done.

        Called once per PP loop iteration, right after `process_prefill_chunk` and
        before `get_new_batch_prefill`, so the pages released here are already
        visible to this iteration's scheduling. `forward_done.query()` is
        non-blocking on purpose: a microbatch that has not finished is retried
        next iteration instead of stalling the CPU inside the pipeline.
        """
        if not self.disagg_prefill_pending_final_kv:
            return

        still_pending: List[PendingFinalKV] = []
        for entry in self.disagg_prefill_pending_final_kv:
            req = entry.req
            if is_aborted(req):
                continue
            if not entry.forward_done.query():
                still_pending.append(entry)
                continue
            # Radix dedup first: cache_unfinished_req may free pages and rewrite
            # this request's req_to_token rows, so the send has to read them after.
            maybe_cache_unfinished_req(req, self.tree_cache)
            self.send_kv_chunk(
                req,
                last_chunk=True,
                end_idx=entry.end_idx,
                send_aux=False,
            )
            # Tells process_batch_result_disagg_prefill not to send it again.
            req.pp_early_final_kv_done = True
            req.time_stats.set_prefill_transfer_queue_entry_time()
            if self.enable_metrics:
                self.metrics_collector.observe_prompt_calculation_time(
                    req.time_stats.prefill_transfer_queue_entry_time
                    - req.time_stats.forward_entry_time
                )
            self.disagg_prefill_inflight_queue.append(req)

        self.disagg_prefill_pending_final_kv = still_pending

    def _pp_drop_pending_final_kv(self: Scheduler, req: Req) -> None:
        """Cancel a queued early send because the regular path is taking over.

        The flush only sends once `forward_done.query()` is true, and it runs
        before this iteration's post-process. A microbatch whose forward is still
        running at flush time therefore stays queued, yet post-process
        (`d2h_event.synchronize()`) may resolve it in the very same iteration and
        send it the regular way. Without this cancel the next flush would send the
        final chunk a second time, pushing the sender's cursor past its index
        count and hanging the room forever.
        """
        if not self.disagg_prefill_pending_final_kv:
            return
        self.disagg_prefill_pending_final_kv = [
            entry
            for entry in self.disagg_prefill_pending_final_kv
            if entry.req is not req
        ]

    def maybe_send_cached_prefix_chunk(self: Scheduler, req: Req) -> None:
        # Only bootstrap-finalized requests; staging excluded.
        if (
            not envs.SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX.get()
            or self.enable_staging
            or req.pending_bootstrap
        ):
            return

        # Device-resident prefix only; page-aligned so start_send_idx stays exact.
        cached_end = len(req.prefix_indices) - req.host_hit_length
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

    def send_kv_chunk(
        self: Scheduler,
        req: Req,
        last_chunk: bool = False,
        end_idx: Optional[int] = None,
        send_aux: bool = True,
    ) -> None:
        """
        Send a prefilled chunk to the decode server

        send_aux only applies to the last chunk: False means this rank is not the
        aux owner, so the metadata buffer is neither written nor transferred.
        Used by the PP+PD early-last-chunk path where PP0 owns the aux row.
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

        if end_idx < start_idx:
            logger.debug(
                "send_kv_chunk skip: rid=%s start_send_idx=%s end_idx=%s",
                req.rid,
                start_idx,
                end_idx,
            )
            return

        state_indices: Optional[List] = None
        if last_chunk:
            if send_aux:
                self.disagg_metadata_buffers.set_buf(req)
            else:
                # The buffer stays allocated on every rank so the bootstrap
                # metadata credit (and therefore the PP admission set) is
                # identical across ranks; it is simply left unwritten here.
                assert (
                    req.metadata_buffer_index >= 0
                ), f"metadata buffer must stay allocated on non-aux ranks, {req.rid=}"

            # Most state payloads read token-pool rows and should match the KV
            # range actually materialized on prefill. C128 state is request
            # scoped, so its transfer index must use the logical input length
            # that decode used to register the destination row.
            # end_idx is the snapshot taken by the caller, so this stays correct
            # even when req.extend_range was rewritten by later scheduling.
            seq_len = end_idx
            c128_seq_len = transfer_input_len

            def _mamba_payload():
                return [
                    self.req_to_token_pool.req_index_to_mamba_index_mapping[
                        req.req_pool_idx
                    ]
                    .cpu()
                    .numpy()
                ]

            def _swa_payload():
                window_size = self.sliding_window_size
                window_start = max(0, seq_len - window_size)
                window_start = (window_start // page_size) * page_size
                window_kv_indices_full = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, window_start:seq_len
                ]
                window_kv_indices_swa = (
                    self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                        window_kv_indices_full
                    )
                )
                return kv_to_page_indices(window_kv_indices_swa, page_size)

            def _dsa_payload():
                kv_indices_full = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, :seq_len
                ]
                return kv_to_page_indices(kv_indices_full, page_size)

            def _swa_ring_payload():
                # Unified_kv SWA ring rows (req_pool_idx*ring_stride + pos%ring_stride)
                # for the last `window` positions, in ascending position order so
                # decode (its own req_pool_idx) matches positionally.
                _pool = self.token_to_kv_pool_allocator.get_kvcache()
                ring_stride = _pool.unified_swa_ring_size
                window_size = _pool.unified_swa_window
                window_start = max(0, seq_len - window_size)
                positions = np.arange(window_start, seq_len, dtype=np.int64)
                state_slot = int(req.req_pool_idx)
                ring_rows = state_slot * ring_stride + (positions % ring_stride)
                return ring_rows.astype(np.int32)

            def _c128_state_payload():
                online = is_dsv4_c128_online_enabled()
                ring_size = (
                    1
                    if online
                    else self.token_to_kv_pool_allocator.get_kvcache().get_ring_size(
                        128
                    )
                )
                return get_dsv4_c128_state_indices(
                    int(req.req_pool_idx),
                    c128_seq_len,
                    online=online,
                    ring_size=ring_size,
                )

            state_types = (
                self.disagg_prefill_bootstrap_queue.kv_manager.kv_args.state_types
            )
            state_indices = []
            for st in state_types:
                if st == StateType.MAMBA:
                    state_indices.append(_mamba_payload())
                elif st == StateType.SWA:
                    state_indices.append(_swa_payload())
                elif st == StateType.DSA:
                    state_indices.append(_dsa_payload())
                elif st == StateType.MINIMAX_INDEX_K:
                    # Index rows live at the same loc as main KV on the same
                    # page_size, so reuse the full-seq page-ids.
                    state_indices.append(_dsa_payload())
                elif st == StateType.SWA_RING:
                    state_indices.append(_swa_ring_payload())
                elif st == StateType.C128_STATE:
                    state_indices.append(_c128_state_payload())
                else:
                    state_indices.append(None)

        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, start_idx:end_idx
        ]
        page_indices = kv_to_page_indices(kv_indices, page_size)
        if not req.disagg_kv_sender.should_send_kv_chunk(len(page_indices), last_chunk):
            return
        req.disagg_kv_sender.send(
            page_indices,
            state_indices,
            num_kv_tokens=end_idx - start_idx,
            send_aux=send_aux,
        )
        req.start_send_idx = end_idx

    def optimistic_release_and_requeue(self: Scheduler, req: Req) -> None:
        """Release KV cache and requeue an optimistic prefill request."""
        max_attempts = self.server_args.optimistic_prefill_attempts
        maybe_cache_unfinished_req(req, self.tree_cache)
        release_kv_cache(req, self.tree_cache)
        req.reset_for_retract()
        req.output_ids = array("q")
        req.start_send_idx = 0
        req.tmp_end_idx = -1
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
            self.waiting_queue.insert(0, req)
