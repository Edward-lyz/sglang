"""Parallel per-target KV transfer for the mooncake prefill path.

Opt-in via ``SGLANG_PP_PD_KV_TRANSFER_TARGET_PARALLEL``. When enabled, a chunk's
multi-destination fan-out is expanded at enqueue time into one work unit per
decode destination, so the existing worker threads transfer to different decode
ranks in parallel.

This covers **only** the fan-out of one room. Spreading *different* rooms over the
worker threads is a separate concern, owned by
``SGLANG_PP_PD_KV_TRANSFER_ROOM_SHARDING`` and applied in
``MooncakeKVManager._transfer_queue_shard_idx``; the fan-out shards by room as
well, so enabling it forces that knob on. Keeping them apart matters because the
deployments that need them are disjoint: a decode fleet with one destination per
room (plain DP) needs only the room sharding, while the fan-out only pays off when
a room really has several destinations (e.g. decode DCP). Accordingly a room with
fewer than two real destinations is declined here and takes the whole-room path.

**This module deliberately duplicates the per-destination body of
``MooncakeKVManager.transfer_worker``.** The legacy whole-room path in `conn.py`
is left byte-identical so that turning the flag off restores exactly the
pre-existing code path; the two bodies are meant to be read side by side. Only
two things differ, and both are forced by the fan-out:

- room convergence is cross-thread (``_RoomCompletion``) instead of a thread-local
  ``polls`` list;
- a failure is broadcast to every destination of the room, not only to the one
  that failed -- every decode rank tracks this prefill rank's response
  independently, so a rank that is never told waits until its own abort timeout.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import logging
import threading
import time
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.utils import FastQueue, TransferKVChunk
from sglang.srt.observability.mooncake_trace import (
    MooncakeRequestStage,
    mooncake_trace_slice,
)
from sglang.srt.observability.trace import TraceNullContext, TraceReqContext
from sglang.srt.utils.network import NetworkAddress

if TYPE_CHECKING:
    from sglang.srt.disaggregation.mooncake.conn import (
        KVArgsRegisterInfo,
        MooncakeKVManager,
        TransferInfo,
    )

logger = logging.getLogger(__name__)

TraceCtx = Union["TraceReqContext", TraceNullContext]

class _RoomCompletion:
    """Cross-thread convergence state for one room's last chunk.

    The N units of a room's last chunk run on N worker threads, but the room may
    only be concluded once: exactly one thread sets the terminal status and syncs
    it to every decode endpoint. Decode counts one response per prefill rank, so
    a second publication is a duplicate response and a missing one is a hang
    until the client's abort timeout.

    The object is built by the enqueue loop and handed to every sibling unit, so
    all of them agree on how many arrivals to wait for; a worker building its own
    would see ``arrived == 1 == n_targets`` and publish early. It is owned by the
    chunks that reference it -- there is no registry to clean up, so the abort /
    vanished-destination / exception paths can simply drop their unit.
    """

    __slots__ = ("lock", "n_targets", "arrived", "all_ok", "dst_ranks_infos")

    def __init__(self, n_targets: int):
        self.lock = threading.Lock()
        self.n_targets = n_targets
        self.arrived = 0
        self.all_ok = True
        # (endpoint, dst_port, room) of every target that reported in, used by
        # the completer to broadcast the final status.
        self.dst_ranks_infos: List[Tuple[str, int, int]] = []

    def report(self, *, ok: bool, dst_rank_info: Tuple[str, int, int]) -> bool:
        """Record one target's outcome. True for the last arrival, i.e. the
        single thread that owns concluding the room."""
        with self.lock:
            self.arrived += 1
            self.all_ok = self.all_ok and ok
            self.dst_ranks_infos.append(dst_rank_info)
            return self.arrived == self.n_targets


@dataclasses.dataclass
class PerTargetKVChunk(TransferKVChunk):
    """A work unit covering exactly one decode destination of a room.

    A subclass rather than an extra field on ``TransferKVChunk`` so that the
    shared container (also used by the nixl backend) stays untouched, and so the
    worker can dispatch on the type. ``completion`` is set on the last chunk only
    and is shared by that chunk's sibling units.
    """

    target_session_id: str = ""
    completion: Optional[_RoomCompletion] = None


def maybe_enqueue_target_parallel_chunks(
    mgr: "MooncakeKVManager",
    *,
    bootstrap_room: int,
    kv_indices: npt.NDArray[np.int32],
    index_slice: slice,
    is_last_chunk: bool,
    aux_index: Optional[int],
    state_indices: Optional[List],
    num_kv_tokens: Optional[int],
    trace_ctx: TraceCtx,
) -> bool:
    """Expand a chunk into one work unit per real decode destination.

    Returns True once the units are enqueued, False to decline -- the caller then
    falls back to the whole-room path. **A room with fewer than two real
    destinations is always declined**, even with the flag on: there is no fan-out
    to parallelize, so the units would only add a ``_RoomCompletion`` and a second
    code path over the well-trodden one. That covers both the single-destination
    room (plain decode DP -- the common case, where the win comes from room
    sharding instead) and the dummy-only room, whose Success the whole-room worker
    already concludes.

    A room cannot flip between the two paths mid-flight: ``transfer_infos[room]``
    is complete and ``is_dummy`` resolved before the room reaches
    ``WaitingForInput``, i.e. before its first chunk is enqueued.

    Sharding is ``(hash(room) + ordinal) % n_queues``: ``hash(room)`` spreads
    different rooms over the queues -- the same key the whole-room path uses under
    ``SGLANG_PP_PD_KV_TRANSFER_ROOM_SHARDING`` -- and the ordinal walks a room's own
    destinations onto consecutive queues. Two properties fall out of that:

    - as long as ``n_targets <= n_queues``, the fan-out occupies exactly
      n_targets distinct queues -- pigeonhole, not probability. Hashing each
      destination independently (``hash((room, session_id))``) only gets there
      ~55% of the time for 8 targets over 50 queues: birthday collisions put two
      of a room's transfers back on one worker thread.
    - the ordinal comes from ``sorted()``, so it is stable across the chunks of a
      room. A destination must not move between queues mid-room: only the last
      chunk concludes the room, so the earlier chunks of the same destination
      have to stay ordered behind it on one queue.

    Sorting (rather than dict order) is what makes the ordinal independent of the
    order in which decode ranks happened to register.
    """
    room_infos = mgr.transfer_infos[bootstrap_room]
    real_sessions = sorted(
        sid for sid, info in room_infos.items() if not info.is_dummy
    )
    if len(real_sessions) < 2:
        return False

    # Shared by the sibling units of this chunk only; the middle chunks never
    # conclude a room and so need no tracker.
    completion = _RoomCompletion(len(real_sessions)) if is_last_chunk else None

    num_queues = len(mgr.transfer_queues)
    base_shard_idx = hash(bootstrap_room) % num_queues
    for ordinal, session_id in enumerate(real_sessions):
        shard_idx = (base_shard_idx + ordinal) % num_queues
        mgr.transfer_queues[shard_idx].put(
            PerTargetKVChunk(
                room=bootstrap_room,
                prefill_kv_indices=kv_indices,
                index_slice=index_slice,
                is_last_chunk=is_last_chunk,
                prefill_aux_index=aux_index,
                state_indices=state_indices,
                num_kv_tokens=num_kv_tokens,
                # thread_context is thread-bound and the sibling units now run on
                # different threads, so each needs its own context. Copying an
                # already-copied context is well-defined (thread_context is None
                # on a copy), which is why the sender can keep passing one copy.
                trace_ctx=trace_ctx.copy_for_thread(),
                target_session_id=session_id,
                completion=completion,
            )
        )
    return True


def process_target_parallel_chunk(
    mgr: "MooncakeKVManager",
    *,
    kv_chunk: PerTargetKVChunk,
    executor: concurrent.futures.ThreadPoolExecutor,
    queue: FastQueue,
    staging_strategy,
) -> None:
    """Worker-side counterpart of ``enqueue_target_parallel_chunks``.

    One unit, one destination: this is the body of the legacy
    ``for req in reqs_to_be_processed`` loop with the loop peeled off. Callers
    keep the legacy ``except Exception`` handler of ``transfer_worker`` around
    this -- it pops the room's ``transfer_infos``, after which the sibling units
    resolve no destination and drop out, which is the intended blast radius for
    an exception.
    """
    if mgr.enable_trace:
        kv_chunk.trace_ctx.rebuild_thread_context()
        kv_chunk.trace_ctx.trace_slice_start(
            MooncakeRequestStage.MOONCAKE_WORKER_SEND.stage_name,
            MooncakeRequestStage.MOONCAKE_WORKER_SEND.level,
        )

    if (
        kv_chunk.room not in mgr.request_status
        or mgr.check_status(kv_chunk.room) == KVPoll.Failed
    ):
        logger.debug(
            f"Skipping chunk for room {kv_chunk.room} because it has already failed or been aborted"
        )
        _end_worker_trace(mgr, kv_chunk)
        return

    # None means the room's bookkeeping is gone (concluded, aborted, or dropped by
    # a sibling unit's exception handler): nothing to transfer, nothing to settle.
    req = mgr.transfer_infos.get(kv_chunk.room, {}).get(kv_chunk.target_session_id)
    if req is None:
        _end_worker_trace(mgr, kv_chunk)
        return

    # Unique id per prefill sender so decode's response set size matches expected_response_num.
    prefill_unique_rank = (
        mgr.attn_tp_rank * (mgr.pp_size * mgr.attn_cp_size)
        + mgr.pp_rank * mgr.attn_cp_size
        + mgr.attn_cp_rank
    )
    start_ts = time.perf_counter()

    outcome, poll_ok = _transfer_one_target(
        mgr,
        kv_chunk=kv_chunk,
        req=req,
        executor=executor,
        queue=queue,
        staging_strategy=staging_strategy,
        prefill_unique_rank=prefill_unique_rank,
    )

    if mgr.enable_trace:
        mooncake_trace_slice(
            kv_chunk.trace_ctx,
            MooncakeRequestStage.MOONCAKE_WORKER_SEND_SESSION,
            start_ts,
        )

    if outcome != _OK:
        # _DEFERRED: staging re-enqueued this unit, it reports on the retry.
        # _ROOM_FAILED: the room is already concluded Failed, and this unit
        # deliberately does not report -- the arrival count can then never reach
        # n_targets, so no sibling publishes a status over the failure. Neither
        # case cleans up transfer_infos, matching the legacy path (its cleanup is
        # gated on a Success/absent status too).
        _end_worker_trace(mgr, kv_chunk)
        return

    if kv_chunk.is_last_chunk:
        _report_and_maybe_conclude(
            mgr,
            kv_chunk=kv_chunk,
            req=req,
            ok=poll_ok,
            prefill_unique_rank=prefill_unique_rank,
        )

    _end_worker_trace(mgr, kv_chunk)
    # Gated on the room's terminal status, so a unit that is not the last arrival
    # leaves the bookkeeping its siblings still need.
    _cleanup_finished_room(mgr, kv_chunk.room)


def _end_worker_trace(mgr: "MooncakeKVManager", kv_chunk: PerTargetKVChunk) -> None:
    if mgr.enable_trace:
        kv_chunk.trace_ctx.trace_slice_end(
            MooncakeRequestStage.MOONCAKE_WORKER_SEND.stage_name,
            MooncakeRequestStage.MOONCAKE_WORKER_SEND.level,
            thread_finish_flag=True,
        )


# _transfer_one_target outcomes.
_OK = 0  # transfer ran to the end; the returned poll says whether aux succeeded
_DEFERRED = 1  # staging re-enqueued this unit; it will report on the retry
_ROOM_FAILED = 2  # this unit already concluded the room as Failed


def _transfer_one_target(
    mgr: "MooncakeKVManager",
    *,
    kv_chunk: PerTargetKVChunk,
    req: "TransferInfo",
    executor: concurrent.futures.ThreadPoolExecutor,
    queue: FastQueue,
    staging_strategy,
    prefill_unique_rank: int,
) -> Tuple[int, bool]:
    """Transfer one chunk to one decode destination.

    Body of the legacy per-req loop with the loop peeled off; the `break`s become
    ``_ROOM_FAILED`` returns. Failure handling keeps the legacy split: a dead
    session / KV / state failure concludes the room immediately, while an aux
    failure only contributes a False poll (the room then concludes Failed once
    every sibling has reported).
    """
    # Early exit if the request has failed: release the session once its
    # cooldown expires (or, fail-open, when it was blacklisted without a
    # timestamp), otherwise fail the room.
    if not mgr._check_session_cooldown(req.mooncake_session_id):
        _fail_room(
            mgr,
            room=kv_chunk.room,
            reason=(
                f"Decode instance could be dead, remote mooncake "
                f"session {req.mooncake_session_id} is not alive"
            ),
            prefill_unique_rank=prefill_unique_rank,
        )
        return (_ROOM_FAILED, False)

    target_rank_registration_info: "KVArgsRegisterInfo" = mgr.decode_kv_args_table[
        req.mooncake_session_id
    ]
    is_dcp_transfer = target_rank_registration_info.requires_dcp_relayout
    if is_dcp_transfer:
        chunked_dst_kv_indice = req.dst_kv_indices
    else:
        chunked_dst_kv_indice = req.dst_kv_indices[kv_chunk.index_slice]

        # NOTE: This is temporarily a workaround to deal with the case where the
        # prefill_kv_indices is mismatched with the dst_kv_indices when page size
        # > 1, this should never happen. Rebinding the field is safe even though
        # sibling units share the underlying array: each unit owns its own chunk
        # object and the right-hand side is a fresh slice view.
        if len(chunked_dst_kv_indice) < len(kv_chunk.prefill_kv_indices):
            logger.warning(
                f"len(chunked_dst_kv_indice) = {len(chunked_dst_kv_indice)}, "
                f"len(kv_chunk.prefill_kv_indices) = {len(kv_chunk.prefill_kv_indices)}"
            )
            kv_chunk.prefill_kv_indices = kv_chunk.prefill_kv_indices[
                : len(chunked_dst_kv_indice)
            ]

    skip_kv, skip_state = mgr._get_dsa_cache_transfer_skip_flags(
        target_rank_registration_info
    )
    if (
        len(kv_chunk.prefill_kv_indices) == 0
        or not mgr.kv_args.kv_data_ptrs
        or skip_kv
    ):
        ret = 0
    elif is_dcp_transfer:
        ret = _send_kvcache_dcp_with_draft_tail(
            mgr,
            kv_chunk=kv_chunk,
            req=req,
            target_rank_registration_info=target_rank_registration_info,
            chunked_dst_kv_indice=chunked_dst_kv_indice,
            executor=executor,
        )
    elif (
        mgr.is_mla_backend
        or mgr.is_hybrid_mla_backend
        or mgr.attn_tp_size == target_rank_registration_info.dst_attn_tp_size
    ):
        # Same MLA/draft split as the legacy worker in conn.py: the whole-entry
        # flat transfer cannot head-slice the draft (MHA) KV when attn_tp differs
        # (e.g. prefill attn_tp=1 -> decode TP8), nor address a window-compact
        # decode draft pool.
        src_mla_entry_count, dst_mla_entry_count = mgr._mla_entry_counts(
            target_rank_registration_info
        )
        has_draft_entries = src_mla_entry_count < len(mgr.kv_args.kv_item_lens)
        ret = mgr.send_kvcache(
            req.mooncake_session_id,
            kv_chunk.prefill_kv_indices,
            target_rank_registration_info.dst_kv_ptrs,
            chunked_dst_kv_indice,
            executor,
            target_rank_registration_info.dst_kv_layer_ids,
            src_entry_count=src_mla_entry_count if has_draft_entries else None,
            dst_entry_count=dst_mla_entry_count if has_draft_entries else None,
        )
        if ret == 0 and has_draft_entries:
            ret = mgr.send_kvcache_draft(
                req.mooncake_session_id,
                kv_chunk.prefill_kv_indices,
                target_rank_registration_info.dst_kv_ptrs,
                (
                    req.draft_dst_kv_indices
                    if req.draft_dst_kv_indices is not None
                    else chunked_dst_kv_indice
                ),
                src_mla_entry_count=src_mla_entry_count,
                dst_mla_entry_count=dst_mla_entry_count,
                dst_attn_tp_size=target_rank_registration_info.dst_attn_tp_size,
                dst_tp_rank=target_rank_registration_info.dst_tp_rank,
                draft_total_kv_heads=mgr.kv_args.draft_kv_head_num * mgr.attn_tp_size,
                # No DCP relayout on this branch, so the no-window fallback plan
                # must not stride.
                dst_dcp_size=1,
                src_page_offset=kv_chunk.index_slice.start or 0,
                decode_prefix_len=req.decode_prefix_len or 0,
                num_kv_tokens=kv_chunk.num_kv_tokens,
                draft_window_start=req.draft_window_start,
                executor=executor,
            )
    elif (
        mgr.enable_staging
        and staging_strategy is not None
        and target_rank_registration_info.staging is not None
    ):
        ret, deferred = mgr._do_staging_transfer(
            staging_strategy,
            kv_chunk,
            req,
            target_rank_registration_info,
            chunked_dst_kv_indice,
            executor,
            queue,
            prefill_unique_rank,
        )
        if deferred:
            # Re-enqueued just this unit, so -- unlike the legacy path's `break`
            # -- the room's other destinations are unaffected.
            return (_DEFERRED, False)
    else:
        ret = mgr.send_kvcache_slice(
            req.mooncake_session_id,
            kv_chunk.prefill_kv_indices,
            target_rank_registration_info.dst_kv_ptrs,
            chunked_dst_kv_indice,
            target_rank_registration_info.dst_tp_rank,
            target_rank_registration_info.dst_attn_tp_size,
            target_rank_registration_info.dst_kv_item_len,
            executor,
        )

    if ret != 0:
        mgr._blacklist_session(req.mooncake_session_id)
        _fail_room(
            mgr,
            room=kv_chunk.room,
            reason=(
                f"Failed to send kv chunk of {kv_chunk.room} to "
                f"{NetworkAddress(req.endpoint, req.dst_port).to_host_port_str()}"
            ),
            prefill_unique_rank=prefill_unique_rank,
        )
        return (_ROOM_FAILED, False)

    if not kv_chunk.is_last_chunk:
        return (_OK, True)

    if kv_chunk.state_indices and not skip_state:
        state_rc = mgr.maybe_send_extra(
            req,
            kv_chunk.state_indices,
            executor,
            target_rank_registration_info,
        )
        if state_rc != 0:
            # Same blacklist contract as the KV failure above; this branch used
            # to forget the cooldown timestamp, blacklisting the session forever.
            mgr._blacklist_session(req.mooncake_session_id)
            _fail_room(
                mgr,
                room=kv_chunk.room,
                reason=(
                    f"Failed to send state components of {kv_chunk.room} to "
                    f"{NetworkAddress(req.endpoint, req.dst_port).to_host_port_str()}"
                ),
                prefill_unique_rank=prefill_unique_rank,
            )
            return (_ROOM_FAILED, False)

    # Only the last chunk we need to send the aux data. prefill_aux_index is None
    # when this rank is not the aux owner (PP+PD early-last-chunk); the room still
    # concludes here so decode's per-rank response tracker still collects this
    # rank.
    if kv_chunk.prefill_aux_index is None:
        return (_OK, True)
    aux_rc = mgr.send_aux(
        req, kv_chunk.prefill_aux_index, target_rank_registration_info.dst_aux_ptrs
    )
    return (_OK, aux_rc == 0)


def _send_kvcache_dcp_with_draft_tail(
    mgr: "MooncakeKVManager",
    *,
    kv_chunk: PerTargetKVChunk,
    req: "TransferInfo",
    target_rank_registration_info: "KVArgsRegisterInfo",
    chunked_dst_kv_indice: npt.NDArray[np.int64],
    executor: concurrent.futures.ThreadPoolExecutor,
) -> int:
    """DCP relayout send, duplicated verbatim from the legacy worker."""
    dcp_token_item_lens = target_rank_registration_info.dcp_token_item_lens
    assert dcp_token_item_lens is not None
    # Split entries at the MLA/draft boundary: the MLA prefix takes DCP token
    # relayout; the head-sharded draft tail (DCP-replicated) takes a head-slice
    # whole-token transfer. The boundary is per-side: a PP prefill stage only owns
    # its own layers' MLA entries while the decode peer owns all of them.
    src_mla_entry_count = mgr.kv_args.num_mla_kv_entries or len(
        mgr.kv_args.kv_item_lens
    )
    dst_mla_entry_count = (
        target_rank_registration_info.dst_num_mla_kv_entries
        or len(target_rank_registration_info.dst_kv_ptrs)
    )
    ret = mgr.send_kvcache_dcp(
        req.mooncake_session_id,
        kv_chunk.prefill_kv_indices,
        target_rank_registration_info.dst_kv_ptrs,
        chunked_dst_kv_indice,
        dcp_token_item_lens=dcp_token_item_lens,
        dst_dcp_size=target_rank_registration_info.dst_dcp_size,
        dst_dcp_rank=target_rank_registration_info.dst_dcp_rank,
        src_page_offset=kv_chunk.index_slice.start or 0,
        decode_prefix_len=req.decode_prefix_len or 0,
        num_kv_tokens=kv_chunk.num_kv_tokens,
        executor=executor,
        dst_layer_ids=target_rank_registration_info.dst_kv_layer_ids,
        src_mla_entry_count=src_mla_entry_count,
        dst_mla_entry_count=dst_mla_entry_count,
    )
    if ret == 0 and src_mla_entry_count < len(mgr.kv_args.kv_item_lens):
        draft_total_kv_heads = mgr.kv_args.draft_kv_head_num * mgr.attn_tp_size
        ret = mgr.send_kvcache_draft(
            req.mooncake_session_id,
            kv_chunk.prefill_kv_indices,
            target_rank_registration_info.dst_kv_ptrs,
            (
                req.draft_dst_kv_indices
                if req.draft_dst_kv_indices is not None
                else chunked_dst_kv_indice
            ),
            src_mla_entry_count=src_mla_entry_count,
            dst_mla_entry_count=dst_mla_entry_count,
            dst_attn_tp_size=target_rank_registration_info.dst_attn_tp_size,
            dst_tp_rank=target_rank_registration_info.dst_tp_rank,
            draft_total_kv_heads=draft_total_kv_heads,
            dst_dcp_size=target_rank_registration_info.dst_dcp_size,
            src_page_offset=kv_chunk.index_slice.start or 0,
            decode_prefix_len=req.decode_prefix_len or 0,
            num_kv_tokens=kv_chunk.num_kv_tokens,
            draft_window_start=req.draft_window_start,
            executor=executor,
        )
    return ret


def _report_and_maybe_conclude(
    mgr: "MooncakeKVManager",
    *,
    kv_chunk: PerTargetKVChunk,
    req: "TransferInfo",
    ok: bool,
    prefill_unique_rank: int,
) -> None:
    """Report this target's outcome; the last arrival concludes the room.

    Replaces the legacy thread-local ``polls`` list, whose ``len(polls) ==
    req.required_dst_info_num`` test only worked because one thread walked every
    destination. ``required_dst_info_num`` is not usable as the target count here:
    it counts all registered decode ranks including dummies, whereas only the real
    sessions ever produce a unit.
    """
    assert kv_chunk.completion is not None, "last chunk must carry a completion"
    is_completer = kv_chunk.completion.report(
        ok=ok, dst_rank_info=(req.endpoint, req.dst_port, req.room)
    )
    if is_completer:
        _publish_room_status(mgr, kv_chunk=kv_chunk, completion=kv_chunk.completion)


def _publish_room_status(
    mgr: "MooncakeKVManager",
    *,
    kv_chunk: PerTargetKVChunk,
    completion: _RoomCompletion,
) -> None:
    """Set the room's terminal status and broadcast it to every destination.

    Only the completer runs this, which is what keeps ``update_status``'s
    read-modify-write (``max()`` for non-Failed) single-writer, and keeps decode's
    per-prefill-rank response tracker at exactly one response from this rank.
    Reading ``completion`` without the lock is safe here: every reporter has
    already returned from ``report()`` before the completer was chosen.
    """
    prefill_unique_rank = (
        mgr.attn_tp_rank * (mgr.pp_size * mgr.attn_cp_size)
        + mgr.pp_rank * mgr.attn_cp_size
        + mgr.attn_cp_rank
    )
    status = KVPoll.Success if completion.all_ok else KVPoll.Failed
    mgr.update_status(kv_chunk.room, status)
    for endpoint, dst_port, room in completion.dst_ranks_infos:
        mgr.sync_status_to_decode_endpoint(
            endpoint, dst_port, room, status, prefill_unique_rank
        )


def _fail_room(
    mgr: "MooncakeKVManager",
    *,
    room: int,
    reason: str,
    prefill_unique_rank: int,
) -> None:
    """Conclude a room as Failed and tell *every* destination.

    The legacy path only notified the destination it was working on, which was
    enough because the same thread then abandoned the remaining ones and no
    Success could follow. With the fan-out, the siblings are separate threads:
    each decode rank tracks this prefill rank's response on its own, so a rank
    that is not told would wait for its own abort timeout. ``Failed`` is an
    unconditional assignment in ``update_status``, so concurrent failures on
    several targets are harmless.
    """
    mgr.record_failure(room, reason)
    mgr.update_status(room, KVPoll.Failed)
    for target in list(mgr.transfer_infos.get(room, {}).values()):
        if target.is_dummy:
            continue
        try:
            mgr.sync_status_to_decode_endpoint(
                target.endpoint,
                target.dst_port,
                target.room,
                KVPoll.Failed,
                prefill_unique_rank,
            )
        except Exception:
            logger.error(
                f"Failed to notify {target.endpoint}:{target.dst_port} that room "
                f"{room} failed",
                exc_info=True,
            )


def _cleanup_finished_room(mgr: "MooncakeKVManager", room: int) -> None:
    """Legacy end-of-chunk cleanup, lifted as-is.

    Still gated on the room's status being Success or gone, which is what makes it
    safe to call from every unit: a unit that is not the last arrival finds the
    room still in flight and leaves the bookkeeping its siblings read.
    """
    if room not in mgr.request_status or mgr.check_status(room) == KVPoll.Success:
        if room in mgr.transfer_infos:
            mgr.transfer_infos.pop(room)
        mgr.req_to_decode_prefix_len.pop(room, None)
        if mgr.enable_staging:
            # Purge prefetch bookkeeping for the finished room. Snapshot first:
            # the scheduler thread adds concurrently.
            for key in list(mgr._staging_ctx.prefetch_requested):
                if key[0] == room:
                    mgr._staging_ctx.prefetch_requested.discard(key)
            mgr._staging_ctx.prefetched_rooms.discard(room)
