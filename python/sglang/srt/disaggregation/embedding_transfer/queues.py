"""
Staged queues that drive multimodal embedding reception on the prefill server.

Mirrors the PD KV-transfer staging in ``disaggregation/decode.py``: the
scheduler main loop only ever does a non-blocking poll, while the transport
makes progress in the background.  The embedding rows themselves are never
moved here — attn-tp rank 0 broadcasts them inside model forward
(see ``mm_utils._broadcast_precomputed_embedding``).
"""

import logging
import time
from http import HTTPStatus
from typing import TYPE_CHECKING, List, Tuple

import torch

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.disaggregation.embedding_transfer.base import WaitingImageRequestStatus
from sglang.srt.disaggregation.embedding_transfer.mooncake_transport import (
    _MooncakeBufferManager,
)
from sglang.srt.disaggregation.embedding_transfer.request import WaitingImageRequest
from sglang.srt.managers.io_struct import TokenizedGenerateReqInput

if TYPE_CHECKING:
    from sglang.srt.disaggregation.encode_receiver_aiak import MMReceiverHTTP
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Life cycle of a multimodal request on the prefill server (EPD mode)
#
# 1. MMPreallocQueue:
#     a. Kick off the pull requests to the encoder (no-op on non-first ranks);
#        transport.send() acquires the pinned buffer from the Mooncake pool
#     b. Once every pull request is out (agreed across attn-tp so the ranks
#        stay in lockstep), move the request to MMTransferQueue
#
# 2. MMTransferQueue:
#     a. Poll the transport to check the embedding transfer state
#     b. Sync SUCCESS/FAIL/TIMEOUT across attn-tp on the CPU group
#     c. On SUCCESS, non-first ranks assemble metadata-only mm_inputs locally
#        (offsets / hash / pad_value, no rows) and the request is handed over
#        to the scheduler waiting queue
#
# 3. WaitingQueue:
#     a. Normal prefill forward path, which fans this chunk's embedding rows
#        out from attn-tp rank 0 inside embed_mm_inputs
# ---------------------------------------------------------------------------


class MMPreallocQueue:
    """Requests waiting for their pinned buffer and encoder pull to go out."""

    def __init__(self, mm_receiver: "MMReceiverHTTP"):
        self.queue: List[WaitingImageRequest] = []
        self.mm_receiver = mm_receiver

    def add(self, waiting_req: WaitingImageRequest) -> None:
        """Kick off one request's embedding pull and enqueue it."""
        waiting_req.advance()
        self.queue.append(waiting_req)

    def extend(self, waiting_reqs: List[WaitingImageRequest]) -> None:
        """Register and enqueue a batch of requests, in order."""
        for req in waiting_reqs:
            self.add(req)

    def pop_preallocated(self) -> List[WaitingImageRequest]:
        """Return the reqs ready to move to MMTransferQueue.

        Only the first rank sends pulls, so readiness has to be agreed across
        attn-tp before popping, or the per-rank queues drift apart.
        """
        if not self.queue:
            return []

        local_flags: List[int] = []
        for waiting_req in self.queue:
            # Promote failures too, so the transfer queue emits every abort
            # through one uniform path.
            if waiting_req.status != WaitingImageRequestStatus.PENDING:
                local_flags.append(1)
                continue
            # Retry: the pinned pool may have been full on the previous poll.
            if not waiting_req.all_pulls_sent:
                waiting_req.advance()
            local_flags.append(
                1
                if (
                    waiting_req.all_pulls_sent
                    or waiting_req.status != WaitingImageRequestStatus.PENDING
                )
                else 0
            )

        synced_flags = self.mm_receiver.sync_ready_flags(local_flags)

        ready: List[WaitingImageRequest] = []
        pending: List[WaitingImageRequest] = []
        for waiting_req, is_ready in zip(self.queue, synced_flags):
            (ready if is_ready else pending).append(waiting_req)

        self.queue = pending
        return ready


class MMTransferQueue:
    """Requests whose embedding transfer is in flight."""

    def __init__(self, mm_receiver: "MMReceiverHTTP"):
        self.queue: List[WaitingImageRequest] = []
        self.mm_receiver = mm_receiver
        self.wait_timeout = mm_receiver.wait_timeout

    def add(self, waiting_req: WaitingImageRequest) -> None:
        """Enqueue one request whose pull requests are already on the wire."""
        self.queue.append(waiting_req)

    def extend(self, waiting_reqs: List[WaitingImageRequest]) -> None:
        """Enqueue a batch of requests promoted from ``MMPreallocQueue``."""
        self.queue.extend(waiting_reqs)

    def pop_transferred(
        self,
    ) -> Tuple[List[TokenizedGenerateReqInput], List[Tuple["Req", str, int]]]:
        """Poll the in-flight transfers once.

        Returns:
            (ready_reqs, abort_reqs) — requests whose mm_inputs are assembled
            and can enter waiting_queue, and (req, error_msg, error_code) for
            those that failed or timed out during transfer / assembly.
        """
        # Drain before the empty-queue exit: the last request can be the one
        # whose cleanup() left buffers here.
        buf_mgr = _MooncakeBufferManager.get_instance_if_exists()
        if buf_mgr is not None:
            buf_mgr.process_inflight_queue()

        if not self.queue:
            return [], []

        # Step 1: first-rank polls transports; status synced across attn-tp.
        synced_status, synced_err = self._poll_and_sync_status(time.time())

        # Step 2: non-first ranks assemble metadata-only mm_inputs locally from
        # task_results.  No embedding rows move here; rank 0 broadcasts them
        # inside model forward.  Keyed on synced_status, so the decision is
        # identical on every rank.
        for i, waiting_req in enumerate(self.queue):
            if (
                int(synced_status[i].item()) == int(WaitingImageRequestStatus.SUCCESS)
                and not waiting_req.is_first_rank
            ):
                waiting_req.assemble_metadata_only()

        ready_reqs = []
        abort_reqs = []
        indices_to_remove = set()
        for i, waiting_req in enumerate(self.queue):
            status_value = int(synced_status[i].item())
            should_remove = self._settle_synced_status(
                waiting_req,
                status_value,
                int(synced_err[i]),
                ready_reqs,
                abort_reqs,
            )
            if should_remove:
                indices_to_remove.add(i)

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        if self.mm_receiver.mm_receiver_metrics_collector is not None:
            self.mm_receiver.mm_receiver_metrics_collector.set_waiting_requests(
                len(self.queue)
            )

        if ready_reqs:
            rids = [
                r.rid
                for r in ready_reqs
                if not r.rid.startswith(HEALTH_CHECK_RID_PREFIX)
            ]
            if rids:
                logger.info(f"[trace] recv embedding completed: {rids}")
        if abort_reqs:
            rids = [
                t[0].rid
                for t in abort_reqs
                if not t[0].rid.startswith(HEALTH_CHECK_RID_PREFIX)
            ]
            if rids:
                logger.info(f"[trace] recv embedding failed: {rids}")

        return ready_reqs, abort_reqs

    def _poll_and_sync_status(self, now: float) -> Tuple[torch.Tensor, List[int]]:
        """First-rank polls transports; broadcast (status, err_code) on cpu group.

        Both returned sequences are len(self.queue) long on every rank.
        """
        sync_group = self.mm_receiver._get_sync_group()
        n = len(self.queue)

        if sync_group is None or sync_group.world_size <= 1:
            # Single-rank: pull locally, no distributed collectives.
            for waiting_req in self.queue:
                waiting_req.advance()
                waiting_req.assemble_if_ready()
                waiting_req.check_timeout(now)
            status_arr = [int(w.status) for w in self.queue]
            err_arr = [int(w.error_code or 0) for w in self.queue]
            return (
                torch.tensor(status_arr, dtype=torch.int32, device="cpu"),
                err_arr,
            )

        is_first_rank = sync_group.rank_in_group == 0
        global_src = sync_group.ranks[0]

        if is_first_rank:
            for waiting_req in self.queue:
                waiting_req.advance()
                waiting_req.assemble_if_ready()
                waiting_req.check_timeout(now)
            status_arr = [int(w.status) for w in self.queue]
            err_arr = [int(w.error_code or 0) for w in self.queue]
        else:
            status_arr = [0] * n
            err_arr = [0] * n

        pack = torch.tensor(status_arr + err_arr, dtype=torch.int32, device="cpu")
        torch.distributed.broadcast(pack, src=global_src, group=sync_group.cpu_group)
        return pack[:n].clone(), pack[n:].tolist()

    def _settle_synced_status(
        self,
        waiting_req: WaitingImageRequest,
        status_value: int,
        err_code: int,
        ready_reqs: list,
        abort_reqs: list,
    ) -> bool:
        """Turn one synced status into a ready / abort entry.

        Returns True when the request should leave the queue (ready, failed or
        timed out), False while it is still PENDING.
        """
        if status_value == int(WaitingImageRequestStatus.SUCCESS):
            # Rank 0 succeeded but this rank has no mm_inputs: metadata-only
            # assembly can fail on its own.  Abort rather than hand the
            # scheduler a request without mm_inputs.
            if waiting_req.recv_req.mm_inputs is None:
                metrics = self.mm_receiver.mm_receiver_metrics_collector
                if metrics is not None:
                    metrics.increment_failed_requests()
                error_msg = waiting_req.error_msg or "Image embedding assembly failed"
                logger.error(
                    f"Waiting request {waiting_req.recv_req.rid} assembly failed: "
                    f"{error_msg} error_code={waiting_req.error_code}"
                )
                abort_reqs.append(
                    (
                        self.mm_receiver.create_req(waiting_req.recv_req),
                        error_msg,
                        waiting_req.error_code or HTTPStatus.BAD_GATEWAY,
                    )
                )
                return True
            ready_reqs.append(waiting_req.recv_req)
            metrics = self.mm_receiver.mm_receiver_metrics_collector
            if metrics is not None:
                metrics.observe_embedding_transfer_time(
                    time.time() - waiting_req.start_time
                )
            return True

        if status_value == int(WaitingImageRequestStatus.FAIL):
            # Transfer failed on rank 0 — adopt its verdict locally so every
            # rank emits the same abort.
            if waiting_req.status == WaitingImageRequestStatus.PENDING:
                waiting_req.mark_failed(
                    waiting_req.error_msg or "Image embedding failed on another rank",
                    waiting_req.error_code or err_code or HTTPStatus.BAD_GATEWAY,
                )
            metrics = self.mm_receiver.mm_receiver_metrics_collector
            if metrics is not None:
                metrics.increment_failed_requests()
            logger.error(
                f"Waiting request {waiting_req.recv_req.rid} failed: "
                f"{waiting_req.error_msg} error_code={waiting_req.error_code}"
            )
            abort_reqs.append(
                (
                    self.mm_receiver.create_req(waiting_req.recv_req),
                    waiting_req.error_msg,
                    waiting_req.error_code,
                )
            )
            return True

        if status_value == int(WaitingImageRequestStatus.TIMEOUT):
            # Same abort path as FAIL, distinct code.
            timeout_msg = (
                waiting_req.error_msg
                or f"Timeout waiting for image embedding after {self.wait_timeout}s"
            )
            if waiting_req.status == WaitingImageRequestStatus.PENDING:
                waiting_req.mark_timeout(timeout_msg)
            metrics = self.mm_receiver.mm_receiver_metrics_collector
            if metrics is not None:
                metrics.increment_timeout_requests()
            logger.error(
                f"Timed out waiting for image embeddings for request "
                f"{waiting_req.recv_req.rid}"
            )
            abort_reqs.append(
                (
                    self.mm_receiver.create_req(waiting_req.recv_req),
                    timeout_msg,
                    waiting_req.error_code or HTTPStatus.REQUEST_TIMEOUT,
                )
            )
            return True

        return False
