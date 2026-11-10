"""Coordinate pipeline-parallel scheduling, transport, and output handling."""

from __future__ import annotations

import logging
import math
import time
from array import array
from collections import defaultdict, deque
from contextlib import nullcontext
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.distributed
from tqdm import tqdm

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.embedding_transfer.request import metadata_only_recv_req
from sglang.srt.disaggregation.utils import (
    is_aborted,
    poll_and_all_reduce_attn_cp_tp_group,
    prepare_abort,
)
from sglang.srt.distributed.parallel_state import P2PWork
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    get_attention_dp_rank,
    get_attention_dp_size,
    is_dp_attention_enabled,
    set_is_extend_in_batch,
)
from sglang.srt.managers.overlap_utils import RelayPayload
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import PlannedAdmission
from sglang.srt.managers.scheduler_components.request_receiver import PPBatchPlanMessage
from sglang.srt.managers.utils import (
    GenerationBatchResult,
    get_logprob_dict_from_result,
    get_logprob_from_pp_outputs,
)
from sglang.srt.mem_cache.storage.asradix.types import AttentionStoreWriteOwner
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.observability.req_time_stats import set_time_batch
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils import DynamicGradMode, broadcast_pyobj, point_to_point_pyobj
from sglang.srt.utils.common import get_device_module, is_xpu

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

PPTransferStatus = Tuple[List[str], List[str]]
PPReleasePayload = Union[List[str], PPTransferStatus]
_PP_BATCH_CONTROL_KEY = "__pp_batch_control__"


def _pp_can_skip_output_comm(batch: ScheduleBatch) -> bool:
    """Check if output send/recv can be skipped for this batch."""
    return (
        envs.SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM.get()
        and batch is not None
        and batch.forward_mode == ForwardMode.EXTEND
        and len(batch.reqs) == 1
        and not batch.contains_last_prefill_chunk
        and not batch.return_logprob
    )


def _pp_ordered_intersection(left: List[str], right: List[str]) -> List[str]:
    right_set = set(right)
    return [rid for rid in left if rid in right_set]


def _pp_ordered_union(left: List[str], right: List[str]) -> List[str]:
    seen = set(left)
    merged = list(left)
    for rid in right:
        if rid not in seen:
            seen.add(rid)
            merged.append(rid)
    return merged


def _pp_route_aborts_to_bad(
    good_rids: List[str],
    bad_rids: List[str],
    aborted_rids: Set[str],
) -> Tuple[List[str], List[str]]:
    """Move aborted rids out of the good (intersection) set and into the bad
    (union) set, so PP consensus fails them uniformly on every rank.

    This also flushes aborted reqs that never reached good/bad consensus
    (e.g. stuck in Bootstrapping with a sender whose abort() does not drive
    the poll to Failed).
    """
    if not aborted_rids:
        return good_rids, bad_rids
    good_rids = [rid for rid in good_rids if rid not in aborted_rids]
    # Sort the newcomers so every rank appends them in the same order, keeping
    # the consensus payloads byte-comparable across ranks in logs.
    bad_rids = _pp_ordered_union(bad_rids, sorted(aborted_rids))
    return good_rids, bad_rids


def _pp_merge_transfer_status(
    previous: PPTransferStatus,
    current: PPTransferStatus,
) -> PPTransferStatus:
    previous_success, previous_failed = previous
    current_success, current_failed = current
    failed = _pp_ordered_union(previous_failed, current_failed)
    success = _pp_ordered_intersection(previous_success, current_success)
    if failed:
        failed_set = set(failed)
        success = [rid for rid in success if rid not in failed_set]
    return success, failed


def _pp_merge_abort_records(
    *record_groups: Tuple[Dict[str, Any], ...],
) -> Tuple[Dict[str, Any], ...]:
    """Merge request aborts deterministically, with the lowest PP rank winning."""
    by_request: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for records in record_groups:
        for record in records:
            rid = str(record["rid"])
            candidate = dict(record)
            key = (int(candidate.get("pp_batch_plan_id", -1)), rid)
            previous = by_request.get(key)
            if previous is None or (
                int(candidate.get("origin_pp_rank", -1)),
                str(candidate.get("reason_code", "")),
                str(candidate.get("reason", "")),
            ) < (
                int(previous.get("origin_pp_rank", -1)),
                str(previous.get("reason_code", "")),
                str(previous.get("reason", "")),
            ):
                by_request[key] = candidate
    return tuple(by_request[key] for key in sorted(by_request))


def _pp_control_has_current_aborts(control: Optional[Dict[str, Any]]) -> bool:
    """Return whether control carries aborts for its current batch plan."""

    if control is None:
        return False
    pp_batch_plan_id = int(control.get("pp_batch_plan_id", -1))
    return any(
        int(record.get("pp_batch_plan_id", -1)) == pp_batch_plan_id
        for record in control.get("aborts", ())
    )


def _pp_project_contract(
    contract: Dict[str, Any], active_rids: Tuple[str, ...]
) -> Dict[str, Any]:
    """Project a PP batch contract onto an ordered subset of requests."""
    remote_rids = tuple(contract.get("rids", ()))
    active_set = set(active_rids)
    projected_order = tuple(rid for rid in remote_rids if rid in active_set)
    if projected_order != active_rids or len(active_set) != len(active_rids):
        raise RuntimeError(
            "PP active requests are not an ordered subset of the upstream contract: "
            f"upstream={remote_rids} active={active_rids}"
        )

    prefix_lens = tuple(contract.get("prefix_lens", ()))
    extend_ranges = tuple(contract.get("extend_ranges", ()))
    if len(prefix_lens) != len(remote_rids) or len(extend_ranges) != len(remote_rids):
        raise RuntimeError(f"invalid PP batch contract request vectors: {contract}")

    keep_indices = [i for i, rid in enumerate(remote_rids) if rid in active_set]
    projected_ranges = tuple(extend_ranges[i] for i in keep_indices)
    decisions = tuple(
        decision
        for decision in contract.get("as_decisions", ())
        if decision and decision[0] in active_set
    )
    return {
        "version": contract.get("version"),
        "mb_id": contract.get("mb_id"),
        "rids": active_rids,
        "prefix_lens": tuple(prefix_lens[i] for i in keep_indices),
        "extend_ranges": projected_ranges,
        "extend_num_tokens": sum(end - start for start, end in projected_ranges),
        "chunked_rid": (
            contract.get("chunked_rid")
            if contract.get("chunked_rid") in active_set
            else None
        ),
        "as_decisions": decisions,
    }


def _pp_filter_proxy_rows(
    pp_proxy_tensors: PPProxyTensors,
    upstream_contract: Dict[str, Any],
    active_rids: Tuple[str, ...],
) -> Dict[str, Any]:
    """Drop aborted request rows from token-row-aligned proxy tensors."""
    tensors = pp_proxy_tensors.tensors
    upstream_rids = tuple(upstream_contract.get("rids", ()))
    if active_rids == upstream_rids:
        return dict(tensors)
    _pp_project_contract(upstream_contract, active_rids)

    ranges = tuple(upstream_contract.get("extend_ranges", ()))
    if len(ranges) != len(upstream_rids):
        raise RuntimeError(f"invalid PP proxy row contract: {upstream_contract}")
    row_lengths = [end - start for start, end in ranges]
    total_rows = sum(row_lengths)
    active_set = set(active_rids)
    spans = []
    offset = 0
    for rid, row_count in zip(upstream_rids, row_lengths):
        if rid in active_set:
            spans.append((offset, offset + row_count))
        offset += row_count

    for key, value in tensors.items():
        if key.startswith("__") or not isinstance(value, torch.Tensor):
            continue
        if value.ndim == 0:
            continue
        if value.shape[0] < total_rows:
            raise RuntimeError(
                "PP proxy tensor cannot be projected after request abort: "
                f"key={key} rows={value.shape[0]} expected_at_least={total_rows}"
            )

    filtered: Dict[str, Any] = {}
    for key, value in tensors.items():
        if key.startswith("__") or not isinstance(value, torch.Tensor):
            filtered[key] = value
            continue
        if value.ndim == 0:
            filtered[key] = value
            continue
        if not spans:
            filtered[key] = value[:0]
        elif len(spans) == 1:
            start, end = spans[0]
            filtered[key] = value[start:end]
        else:
            filtered[key] = torch.cat([value[start:end] for start, end in spans], dim=0)
    return filtered


@dataclass
class PPBatchMetadata:
    can_run_cuda_graph: bool
    forward_done: torch.Event


class SchedulerPPMixin:
    """Provide pipeline-parallel scheduling and communication operations."""

    def _pp_extract_authoritative_batch_plan(
        self, mb_id: int, recv_reqs: List[Any]
    ) -> Tuple[List[Any], Optional[Dict[str, Any]]]:
        plan_messages = [
            req for req in recv_reqs if isinstance(req, PPBatchPlanMessage)
        ]
        work_reqs = [
            req for req in recv_reqs if not isinstance(req, PPBatchPlanMessage)
        ]
        if self.pp_group.is_first_rank:
            if plan_messages:
                raise RuntimeError(
                    f"PP0 received an unexpected PP batch plan at mb_id={mb_id}"
                )
            return work_reqs, None
        if len(plan_messages) != 1:
            raise RuntimeError(
                "expected exactly one PP batch plan in request relay: "
                f"mb_id={mb_id} received={len(plan_messages)}"
            )
        message = plan_messages[0]
        if message.mb_id != mb_id:
            raise RuntimeError(
                f"PP batch plan relay mismatch: local={mb_id} remote={message.mb_id}"
            )
        return work_reqs, message.plan

    @staticmethod
    def _pp_batch_contract(
        batch: Optional[ScheduleBatch], mb_id: int
    ) -> Dict[str, Any]:
        if batch is None:
            return {
                "version": 1,
                "mb_id": int(mb_id),
                "rids": (),
                "prefix_lens": (),
                "extend_ranges": (),
                "extend_num_tokens": 0,
                "chunked_rid": None,
                "as_decisions": (),
            }

        decisions = []
        for req in batch.reqs:
            decision = getattr(req, "_as_pp_match_decision", None)
            if isinstance(decision, dict):
                decisions.append(
                    (
                        req.rid,
                        decision.get("hash_digest"),
                        int(decision.get("matched", -1)),
                        tuple(decision.get("generation", ())),
                    )
                )
        return {
            "version": 1,
            "mb_id": int(mb_id),
            "rids": tuple(req.rid for req in batch.reqs),
            "prefix_lens": tuple(len(req.prefix_indices) for req in batch.reqs),
            "extend_ranges": tuple(
                (int(req.extend_range.start), int(req.extend_range.end))
                for req in batch.reqs
            ),
            "extend_num_tokens": int(batch.extend_num_tokens or 0),
            "chunked_rid": (
                batch.chunked_req.rid
                if getattr(batch, "chunked_req", None) is not None
                else None
            ),
            "as_decisions": tuple(decisions),
        }

    def _pp_make_authoritative_batch_plan(
        self,
        mb_id: int,
        batch: Optional[ScheduleBatch],
        prior_chunked_req: Optional[Req] = None,
        next_chunked_req: Optional[Req] = None,
        local_aborts: Tuple[Dict[str, Any], ...] = (),
        transport: Optional[bool] = None,
    ) -> Dict[str, Any]:
        requests = []
        planned_admissions = tuple(
            getattr(batch, "_pp_prefill_admissions", ()) if batch is not None else ()
        )
        if not all(
            isinstance(admission, PlannedAdmission) for admission in planned_admissions
        ):
            raise RuntimeError(
                "PP batch admissions must be PlannedAdmission objects: "
                f"admissions={planned_admissions}"
            )
        admissions = tuple(admission.to_wire() for admission in planned_admissions)
        if batch is not None:
            for req in batch.reqs:
                req._pp_batch_plan_id = int(self._pp_next_batch_plan_id)
                decision = getattr(req, "_as_pp_match_decision", None)
                requests.append(
                    {
                        "rid": req.rid,
                        "as_match": (
                            dict(decision) if isinstance(decision, dict) else None
                        ),
                    }
                )

        if prior_chunked_req is None:
            chunked_transition = {"rid": None, "action": "none"}
        elif any(
            admission.source == "chunked" and admission.rid == prior_chunked_req.rid
            for admission in planned_admissions
        ):
            chunked_transition = {
                "rid": prior_chunked_req.rid,
                "action": "resume",
            }
        elif any(
            admission.source == "waiting" and admission.rid == prior_chunked_req.rid
            for admission in planned_admissions
        ):
            # Non-overlap mode may release/requeue a yielded chunk and admit it
            # again in the same scheduler pass. It is still a yield transition;
            # the new admission source is explicitly waiting.
            chunked_transition = {
                "rid": prior_chunked_req.rid,
                "action": "yield",
            }
        elif prior_chunked_req is next_chunked_req:
            chunked_transition = {
                "rid": prior_chunked_req.rid,
                "action": "hold",
            }
        elif is_aborted(prior_chunked_req):
            chunked_transition = {
                "rid": prior_chunked_req.rid,
                "action": "abort",
            }
        else:
            chunked_transition = {
                "rid": prior_chunked_req.rid,
                "action": "yield",
            }

        pp_batch_plan_id = self._pp_next_batch_plan_id
        self._pp_next_batch_plan_id += 1
        tombstones = self._pp_abort_tombstones_for_plan()
        local_aborts = tuple(
            {**record, "pp_batch_plan_id": int(pp_batch_plan_id)}
            for record in local_aborts
        )
        for record in local_aborts:
            logger.error(
                "PP request cancellation originated: pp_batch_plan_id=%s mb_id=%s "
                "rid=%s origin_pp_rank=%s reason_code=%s reason=%s",
                pp_batch_plan_id,
                mb_id,
                record["rid"],
                record["origin_pp_rank"],
                record["reason_code"],
                record["reason"],
            )
        return {
            "version": 4,
            "mb_id": int(mb_id),
            "maintenance_round": int(self._pp_current_cache_maintenance_round),
            "pp_batch_plan_id": int(pp_batch_plan_id),
            "transport": batch is not None if transport is None else bool(transport),
            "chunked_transition": chunked_transition,
            "admissions": admissions,
            "requests": tuple(requests),
            "contract": self._pp_batch_contract(batch, mb_id),
            "aborts": _pp_merge_abort_records(tombstones, local_aborts),
        }

    def _pp_abort_tombstones_for_plan(self) -> Tuple[Dict[str, Any], ...]:
        """Collect active abort tombstones and age them by one batch plan."""

        records = []
        expired = []
        for key, entry in self._pp_abort_tombstones.items():
            records.append(entry["record"])
            entry["remaining"] -= 1
            if entry["remaining"] <= 0:
                expired.append(key)
        for key in expired:
            del self._pp_abort_tombstones[key]
        return _pp_merge_abort_records(tuple(records))

    def _pp_register_abort_tombstones(
        self, records: Tuple[Dict[str, Any], ...]
    ) -> None:
        """Retain finalized abort records for propagation across the PP ring."""

        ttl = self.ps.pp_size + self.pp_loop_size
        for record in records:
            rid = str(record["rid"])
            key = (int(record.get("pp_batch_plan_id", -1)), rid)
            previous = self._pp_abort_tombstones.get(key)
            merged = _pp_merge_abort_records(
                (() if previous is None else (previous["record"],)),
                (record,),
            )[0]
            self._pp_abort_tombstones[key] = {
                "record": merged,
                "remaining": ttl,
            }

    def _pp_apply_abort_records_to_local_queue(
        self, records: Tuple[Dict[str, Any], ...], current_batch_plan_id: int
    ) -> None:
        """Remove locally queued requests identified by propagated abort records."""

        if not records:
            return
        by_request = {
            (int(record.get("pp_batch_plan_id", -1)), record["rid"]): record
            for record in records
        }
        kept_waiting = []
        for req in self.waiting_queue:
            req_pp_batch_plan_id = int(getattr(req, "_pp_batch_plan_id", -1))
            record = by_request.get((req_pp_batch_plan_id, req.rid))
            if record is None and req_pp_batch_plan_id == -1:
                record = by_request.get((current_batch_plan_id, req.rid))
            if record is None:
                kept_waiting.append(req)
                continue
            if not req.finished():
                req._as_cache_read_abort_reason = record.get("reason")
                self._abort_failed_cache_read(req)
        self.waiting_queue = kept_waiting

        if self.chunked_req is not None:
            chunked_pp_batch_plan_id = int(
                getattr(self.chunked_req, "_pp_batch_plan_id", -1)
            )
            record = by_request.get((chunked_pp_batch_plan_id, self.chunked_req.rid))
            if record is None and chunked_pp_batch_plan_id == -1:
                record = by_request.get((current_batch_plan_id, self.chunked_req.rid))
            if record is not None:
                if not self.chunked_req.finished():
                    self.chunked_req._as_cache_read_abort_reason = record.get("reason")
                    self._abort_failed_cache_read(self.chunked_req)
                self.chunked_req = None

    def _pp_apply_local_aborts_to_plan(
        self,
        mb_id: int,
        batch: Optional[ScheduleBatch],
        plan: Dict[str, Any],
        local_aborts: Tuple[Dict[str, Any], ...],
    ) -> Dict[str, Any]:
        """Project an upstream batch plan after request-local stage aborts."""

        upstream_contract = plan["contract"]
        local_contract = self._pp_batch_contract(batch, mb_id)
        upstream_rids = tuple(upstream_contract.get("rids", ()))
        local_rids = tuple(local_contract.get("rids", ()))
        local_aborts = tuple(
            {**record, "pp_batch_plan_id": int(plan["pp_batch_plan_id"])}
            for record in local_aborts
        )
        chunked_rid = upstream_contract.get("chunked_rid")
        for record in local_aborts:
            if record["rid"] == chunked_rid:
                # Retain the abort until already-relayed chunk plans drain.
                self._pp_local_chunk_abort_records.setdefault(
                    str(record["rid"]), dict(record)
                )
        local_abort_rids = {record["rid"] for record in local_aborts}
        for record in local_aborts:
            logger.error(
                "PP request cancellation originated: pp_batch_plan_id=%s mb_id=%s "
                "rid=%s origin_pp_rank=%s reason_code=%s reason=%s",
                plan["pp_batch_plan_id"],
                mb_id,
                record["rid"],
                record["origin_pp_rank"],
                record["reason_code"],
                record["reason"],
            )
        removed_rids = {rid for rid in upstream_rids if rid not in set(local_rids)}
        if removed_rids != local_abort_rids:
            raise RuntimeError(
                "PP batch admission diverged without matching request aborts: "
                f"planned={upstream_rids} local={local_rids} "
                f"aborted={tuple(sorted(local_abort_rids))}"
            )
        projected_contract = _pp_project_contract(upstream_contract, local_rids)
        if projected_contract != local_contract:
            raise RuntimeError(
                "PP authoritative batch plan mismatch after applying request aborts: "
                f"local={local_contract} projected={projected_contract}"
            )

        active_set = set(local_rids)
        outgoing = dict(plan)
        outgoing["admissions"] = tuple(
            admission
            for admission in plan.get("admissions", ())
            if admission.get("rid") in active_set
        )
        outgoing["requests"] = tuple(
            request
            for request in plan.get("requests", ())
            if request.get("rid") in active_set
        )
        outgoing["contract"] = local_contract
        outgoing["aborts"] = _pp_merge_abort_records(
            tuple(plan.get("aborts", ())), local_aborts
        )
        return outgoing

    def _pp_project_locally_aborted_chunked_requests(
        self, plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Project in-flight plans that still reference a locally aborted chunk."""
        if not self._pp_local_chunk_abort_records:
            return plan

        known_abort_keys = {
            (int(record.get("pp_batch_plan_id", -1)), str(record.get("rid", "")))
            for record in plan.get("aborts", ())
        }
        acknowledged_rids = [
            rid
            for rid, record in self._pp_local_chunk_abort_records.items()
            if (int(record["pp_batch_plan_id"]), rid) in known_abort_keys
        ]
        for rid in acknowledged_rids:
            del self._pp_local_chunk_abort_records[rid]

        if not self._pp_local_chunk_abort_records:
            return plan

        transition = plan.get("chunked_transition", {})
        referenced_rids = {
            str(entry.get("rid"))
            for key in ("admissions", "requests")
            for entry in plan.get(key, ())
            if entry.get("rid") is not None
        }
        referenced_rids.update(str(rid) for rid in plan["contract"].get("rids", ()))
        if transition.get("rid") is not None:
            referenced_rids.add(str(transition["rid"]))

        aborted_rids = set(self._pp_local_chunk_abort_records).intersection(
            referenced_rids
        )
        if not aborted_rids:
            return plan

        active_rids = tuple(
            rid for rid in plan["contract"].get("rids", ()) if rid not in aborted_rids
        )
        current_batch_plan_id = int(plan["pp_batch_plan_id"])
        current_aborts = tuple(
            {
                **self._pp_local_chunk_abort_records[rid],
                "pp_batch_plan_id": current_batch_plan_id,
            }
            for rid in sorted(aborted_rids)
        )
        outgoing = dict(plan)
        outgoing["admissions"] = tuple(
            entry
            for entry in plan.get("admissions", ())
            if entry.get("rid") not in aborted_rids
        )
        outgoing["requests"] = tuple(
            entry
            for entry in plan.get("requests", ())
            if entry.get("rid") not in aborted_rids
        )
        outgoing["contract"] = _pp_project_contract(plan["contract"], active_rids)
        if transition.get("rid") in aborted_rids:
            outgoing["chunked_transition"] = {"rid": None, "action": "none"}
        outgoing["aborts"] = _pp_merge_abort_records(
            tuple(plan.get("aborts", ())), current_aborts
        )
        logger.error(
            "PP projected in-flight plans for locally aborted chunk: "
            "pp_batch_plan_id=%s rids=%s",
            current_batch_plan_id,
            tuple(sorted(aborted_rids)),
        )
        self._pp_validate_authoritative_plan_schema(int(plan["mb_id"]), outgoing)
        return outgoing

    def _pp_defer_chunked_cache_read_abort(
        self, req: Req, record: Dict[str, Any]
    ) -> None:
        """Stop future chunks now and release KV at the next safe loop point."""
        req._as_cache_read_abort_reason = record.get("reason")
        if self.chunked_req is req:
            self.chunked_req = None
        self._pp_pending_cache_read_aborts[id(req)] = (req, dict(record))

    def _pp_drain_pending_cache_read_aborts(self) -> None:
        """Release deferred chunk resources after the prior forward was drained."""
        pending = tuple(self._pp_pending_cache_read_aborts.values())
        self._pp_pending_cache_read_aborts.clear()
        for req, record in pending:
            if req.finished():
                continue
            req._as_cache_read_abort_reason = record.get("reason")
            self._abort_failed_cache_read(req)

    def _pp_install_authoritative_batch_plan(
        self, mb_id: int, plan: Dict[str, Any]
    ) -> None:
        if not isinstance(plan, dict) or plan.get("version") != 4:
            raise RuntimeError(f"invalid PP batch plan at mb_id={mb_id}: {plan}")
        if int(plan.get("mb_id", -1)) != mb_id:
            raise RuntimeError(
                f"PP batch plan mb_id mismatch: local={mb_id} remote={plan.get('mb_id')}"
            )
        self._pp_validate_authoritative_plan_schema(mb_id, plan)
        pp_batch_plan_id = int(plan["pp_batch_plan_id"])
        expected_round = getattr(self, "_pp_current_cache_maintenance_round", None)
        if (
            expected_round is not None
            and int(plan["maintenance_round"]) != expected_round
        ):
            raise RuntimeError(
                "PP cache maintenance round mismatch: "
                f"local={expected_round} remote={plan['maintenance_round']} mb_id={mb_id}"
            )
        if pp_batch_plan_id <= self._pp_last_received_batch_plan_id:
            raise RuntimeError(
                "stale or reordered PP batch plan: "
                f"last={self._pp_last_received_batch_plan_id} received={pp_batch_plan_id}"
            )
        self._pp_last_received_batch_plan_id = pp_batch_plan_id
        self._pp_apply_abort_records_to_local_queue(
            tuple(plan.get("aborts", ())), pp_batch_plan_id
        )

        local_reqs = list(self.waiting_queue) + list(self.running_batch.reqs)
        if self.chunked_req is not None:
            local_reqs.append(self.chunked_req)
        by_rid = {req.rid: req for req in local_reqs}
        for entry in plan.get("requests", ()):
            rid = entry.get("rid")
            req = by_rid.get(rid)
            if req is None:
                raise RuntimeError(
                    f"PP batch plan references unavailable rid={rid} mb_id={mb_id}"
                )
            req._as_pp_match_decision_required = True
            req._pp_batch_plan_id = int(plan["pp_batch_plan_id"])
            decision = entry.get("as_match")
            if decision is None:
                if hasattr(req, "_as_pp_match_decision"):
                    del req._as_pp_match_decision
            else:
                req._as_pp_match_decision = dict(decision)

    @staticmethod
    def _pp_validate_authoritative_plan_schema(
        mb_id: int, plan: Dict[str, Any]
    ) -> None:
        try:
            pp_batch_plan_id = int(plan["pp_batch_plan_id"])
            maintenance_round = int(plan["maintenance_round"])
            transport = plan["transport"]
            transition = plan["chunked_transition"]
            action = transition["action"]
            transition_rid = transition["rid"]
            admissions = tuple(
                PlannedAdmission.from_wire(entry) for entry in plan["admissions"]
            )
            admission_pairs = tuple((entry.rid, entry.source) for entry in admissions)
            request_rids = {entry["rid"] for entry in plan["requests"]}
            contract = plan["contract"]
            contract_rids = tuple(contract["rids"])
            contract_prefix_lens = tuple(
                int(value) for value in contract["prefix_lens"]
            )
            contract_extend_ranges = tuple(
                (int(value[0]), int(value[1])) for value in contract["extend_ranges"]
            )
            contract_extend_num_tokens = int(contract["extend_num_tokens"])
            contract["chunked_rid"]
            aborts = tuple(plan.get("aborts", ()))
            abort_keys = tuple(
                (int(record["pp_batch_plan_id"]), record["rid"]) for record in aborts
            )
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"invalid PP batch plan shape at mb_id={mb_id}: {plan}"
            ) from exc

        admission_rids = tuple(rid for rid, _ in admission_pairs)
        chunked_admissions = [
            (index, rid)
            for index, (rid, source) in enumerate(admission_pairs)
            if source == "chunked"
        ]
        if (
            pp_batch_plan_id < 0
            or maintenance_round < 0
            or not isinstance(transport, bool)
            or action not in ("none", "resume", "hold", "yield", "abort")
            or (action == "none") != (transition_rid is None)
            or any(
                not isinstance(rid, str)
                or not rid
                or source not in ("waiting", "chunked")
                for rid, source in admission_pairs
            )
            or len(set(admission_rids)) != len(admission_rids)
            or admission_rids != contract_rids
            or set(contract_rids) != request_rids
            or len(set(contract_rids)) != len(contract_rids)
            or len(contract_prefix_lens) != len(contract_rids)
            or len(contract_extend_ranges) != len(contract_rids)
            or any(
                prefix_len < 0 or start != prefix_len or end <= start
                for prefix_len, (start, end) in zip(
                    contract_prefix_lens, contract_extend_ranges
                )
            )
            or contract_extend_num_tokens
            != sum(end - start for start, end in contract_extend_ranges)
            or tuple(admission.extend_range for admission in admissions)
            != contract_extend_ranges
            or len(set(abort_keys)) != len(abort_keys)
            or any(not isinstance(rid, str) or not rid for _, rid in abort_keys)
            or not {
                rid for abort_seq, rid in abort_keys if abort_seq == pp_batch_plan_id
            }.isdisjoint(contract_rids)
            or chunked_admissions not in ([], [(0, transition_rid)])
            or ((action == "resume") != bool(chunked_admissions))
            or not set(admission_rids).issubset(request_rids)
            or (not transport and bool(contract_rids))
        ):
            raise RuntimeError(
                f"invalid PP transition/admissions at mb_id={mb_id}: "
                f"transition={transition} admissions={admissions}"
            )

    def _pp_validate_chunked_transition(
        self,
        mb_id: int,
        plan: Dict[str, Any],
        local_chunked_req: Optional[Req],
    ) -> str:
        transition = plan["chunked_transition"]
        action = transition["action"]
        rid = transition["rid"]
        local_rid = local_chunked_req.rid if local_chunked_req is not None else None
        if rid != local_rid:
            raise RuntimeError(
                "PP chunked state mismatch before replay: "
                f"mb_id={mb_id} planned={rid} local={local_rid} action={action}"
            )
        return action

    def _pp_select_authoritative_waiting_reqs(
        self,
        planned_admissions: Tuple[PlannedAdmission, ...],
        resumed_chunked_req: Optional[Req],
    ) -> List[Req]:
        """Resolve PP0's explicit admission sources in their committed order."""
        waiting_by_rid = {req.rid: req for req in self.waiting_queue}
        candidates = []
        for admission in planned_admissions:
            rid = admission.rid
            source = admission.source
            if source == "chunked":
                local_rid = (
                    resumed_chunked_req.rid if resumed_chunked_req is not None else None
                )
                if rid != local_rid:
                    raise RuntimeError(
                        "PP0 batch plan selected unavailable chunked request: "
                        f"planned={rid} local={local_rid}"
                    )
                continue
            req = waiting_by_rid.get(rid)
            if req is None:
                raise RuntimeError(
                    f"PP0 batch plan selected unavailable waiting rid={rid}"
                )
            candidates.append(req)
        return candidates

    @staticmethod
    def _pp_prefill_admissions(
        can_run_list: List[Req],
        resumed_chunked_req: Optional[Req],
    ) -> Tuple[PlannedAdmission, ...]:
        return tuple(
            PlannedAdmission.from_req(
                req,
                source=("chunked" if req is resumed_chunked_req else "waiting"),
            )
            for req in can_run_list
        )

    def _pp_validate_authoritative_batch_plan(
        self,
        mb_id: int,
        batch: Optional[ScheduleBatch],
        plan: Dict[str, Any],
    ) -> None:
        local_contract = self._pp_batch_contract(batch, mb_id)
        remote_contract = plan.get("contract")
        if local_contract != remote_contract:
            raise RuntimeError(
                "PP authoritative batch plan mismatch before forward: "
                f"local={local_contract} remote={remote_contract}"
            )

    @staticmethod
    def _pp_control_from_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
        """Build transport control metadata from an authoritative batch plan."""

        return {
            "version": 2,
            "maintenance_round": int(plan["maintenance_round"]),
            "pp_batch_plan_id": int(plan["pp_batch_plan_id"]),
            "mb_id": int(plan["mb_id"]),
            "transport": bool(plan["transport"]),
            "contract": plan["contract"],
            "aborts": tuple(plan.get("aborts", ())),
        }

    def _pp_prepare_received_proxy(
        self,
        mb_id: int,
        batch: Optional[ScheduleBatch],
        pp_proxy_tensors: PPProxyTensors,
        upstream_plan: Dict[str, Any],
    ) -> PPProxyTensors:
        """Validate and project a received proxy for the local survivor batch."""

        tensors = pp_proxy_tensors.tensors
        remote_control = tensors.get(_PP_BATCH_CONTROL_KEY)
        expected_control = self._pp_control_from_plan(upstream_plan)
        if remote_control != expected_control:
            raise RuntimeError(
                "PP proxy control does not match the relayed plan: "
                f"mb_id={mb_id} proxy={remote_control} plan={expected_control}"
            )

        expected_contract = expected_control["contract"]
        local_contract = self._pp_batch_contract(batch, mb_id)
        active_rids = tuple(local_contract.get("rids", ()))
        filtered = _pp_filter_proxy_rows(
            pp_proxy_tensors,
            expected_contract,
            active_rids,
        )
        filtered.pop(_PP_BATCH_CONTROL_KEY, None)
        prepared = PPProxyTensors(filtered)
        if active_rids != tuple(expected_contract.get("rids", ())):
            prepared._mark_pp_projected_token_rows(
                int(local_contract.get("extend_num_tokens", 0))
            )
        return prepared

    def _pp_validate_authoritative_plan_configuration(self) -> None:
        unsupported = []
        if self.enable_priority_preemption:
            unsupported.append("priority preemption")
        if self.enable_dynamic_chunking:
            unsupported.append("dynamic chunking")
        if self.prefill_delayer is not None:
            unsupported.append("prefill delayer")
        if unsupported:
            raise RuntimeError(
                "PP authoritative batch plan requires deterministic prefill settings; "
                f"unsupported: {', '.join(unsupported)}"
            )

    @DynamicGradMode()
    def event_loop_pp(self: Scheduler):
        """
        A scheduler loop for pipeline parallelism.
        Notes:
        1. Each stage runs in the same order and is notified by the previous stage.
        2. We use async send but sync recv to avoid desynchronization while minimizing the communication overhead.
        3. We can use async batch depth to buffer the outputs in the last stage for to allow overlapping the GPU computation and CPU processing and avoid last PP rank staggler.

        Unified Schedule:
        ====================================================================
        Stage P
        recv ith req from previous stage
        recv ith proxy from previous stage
        run ith batch
        recv prev (i+1)% mb_size th outputs
        process batch result of prev (i+1)% mb_size th batch (can be run in parallel with the curr batch GPU computation)
        send ith req to next stage
        send ith proxy to next stage
        send current stage's outputs to next stage(can be stashed and delayed to send later)

        the above order can be optimized and reordered to minimize communication-related CPU stall and overhead bubbles.

        ====================================================================
        """
        self.init_pp_loop_state()
        while True:
            server_is_idle = True
            for mb_id in range(self.pp_loop_size):
                if self.enable_asradix_cache:
                    self._pp_current_cache_maintenance_round = (
                        self._pp_next_cache_maintenance_round
                    )
                    self._pp_next_cache_maintenance_round += 1
                    self._pp_bind_cache_write_owner(None, mb_id)
                self.running_batch = self.running_mbs[mb_id]
                self.last_batch = self.last_mbs[mb_id]
                next_first_rank_mb_id = (mb_id + self.ps.pp_size) % self.pp_loop_size
                next_mb_id = (mb_id + 1) % self.pp_loop_size
                with torch.profiler.record_function("recv_requests"):
                    recv_reqs = self.request_receiver.recv_requests()
                    self.process_input_requests(recv_reqs)
                if not self.pp_group.is_last_rank:
                    self._pp_commit_comm_work(self.send_req_work)
                    with torch.profiler.record_function("send_reqs_to_next_stage"):
                        self.send_req_work = self._pp_send_pyobj_to_next_stage(
                            recv_reqs,
                            async_send=True,
                        )
                with torch.profiler.record_function("get_next_batch_to_run"):
                    plan = self.get_next_batch_to_run(
                        running_batch=self.running_batch, last_batch=self.last_batch
                    )
                    self.running_batch = plan.running_batch
                    self.mbs[mb_id] = plan.batch_to_run
                self.running_mbs[mb_id] = self.running_batch
                cur_batch: Optional[ScheduleBatch] = self.mbs[mb_id]
                self._pp_bind_cache_write_owner(cur_batch, mb_id)
                self.cur_batch_for_debug = cur_batch
                if cur_batch:
                    server_is_idle = False
                    pp_proxy_tensors = self._pp_recv_proxy_tensors()
                next_pp_outputs = None
                next_batch_result = None
                d2h_event = None
                if self.server_args.pp_async_batch_depth > 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )
                self._pp_commit_comm_work(self.send_proxy_work)
                if cur_batch:
                    result, self.launch_event = self._pp_launch_batch(
                        mb_id,
                        cur_batch,
                        pp_proxy_tensors,
                        self.mb_metadata,
                        self.last_rank_comm_queue,
                    )
                if self.server_args.pp_async_batch_depth == 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )
                if self.mbs[next_mb_id] is not None:
                    d2h_event.synchronize()
                    with torch.profiler.record_function("process_batch_result"):
                        self._pp_process_batch_result(
                            self.mbs[next_mb_id],
                            next_batch_result,
                        )
                    self.last_mbs[next_mb_id] = self.mbs[next_mb_id]
                if not self.pp_group.is_last_rank:
                    if cur_batch:
                        self.device_module.current_stream().wait_event(
                            self.launch_event
                        )
                        with torch.profiler.record_function(
                            "send_proxy_dict_to_next_stage"
                        ):
                            self.send_proxy_work = self._pp_send_dict_to_next_stage(
                                result.pp_hidden_states_proxy_tensors.tensors,
                                async_send=True,
                                msg_type="proxy",
                            )

                self.pp_outputs = next_pp_outputs

            # When the server is idle, self-check and re-init some states
            if server_is_idle:
                self.on_idle()

    @DynamicGradMode()
    def event_loop_pp_disagg_prefill(self: Scheduler):
        """
        This is the prefill server event loop for pipeline parallelism.

        Notes:
        1. Following the same rules as the event_loop_pp.
        2. Adds extra steps for KV transfer process: bootstrap + release.

        Prefill Server Schedule:
        ====================================================================
        Stage P
        recv ith req from previous stage
        recv ith bootstrap req from previous stage
        recv ith transferred req from previous stage
        recv ith proxy from previous stage
        run ith batch
        recv prev (i+1) % mb_size th consensus bootstrapped req from previous stage
        local consensus on bootstrapped req
        recv prev (i+1) % mb_size th release req from previous stage
        local consensus on release req
        recv prev (i+1) % mb_size th outputs
        process batch result of prev (i+1)% mb_size th batch (can be run in parallel with the curr batch GPU computation)
        send ith req to next stage
        send ith bootstrap req to next stage
        send ith transferred req to next stage
        send ith proxy to next stage
        send current stage's outputs to next stage (can be stashed and delayed to send later)

        the above order can be optimized and reordered to minimize communication-related CPU stall and overhead bubbles.
        ====================================================================

        There are two additional elements compared to the regular schedule:

        Bootstrap Requests + Release Requests:
        - Both can have local failure and need to be consensus on. PP needs to guarantee eventual consistency of local failure and flush malfunc requests out as soft error.

        """
        self.init_pp_loop_state()
        # PPBatchPlan is currently an AttentionStore correctness protocol. Keep
        # the ordinary PP and non-AS disaggregated paths byte-for-byte compatible.
        self._pp_batch_contract_guard_enabled = bool(self.enable_asradix_cache)
        if self._pp_batch_contract_guard_enabled:
            self._pp_validate_authoritative_plan_configuration()

        # PD additional state initialization
        bmbs = [None] * self.pp_loop_size
        tmbs = [None] * self.pp_loop_size
        consensus_bootstrapped_rids: Optional[List[str]] = None
        transferred_rids: PPTransferStatus = ([], [])
        release_rids: Optional[PPTransferStatus] = None
        send_bootstrapped_work = []
        send_transfer_work = []
        send_consensus_bootstrapped_work = []
        send_release_work = []

        while True:
            server_is_idle = True
            for mb_id in range(self.pp_loop_size):
                if self.enable_asradix_cache:
                    self._pp_current_cache_maintenance_round = (
                        self._pp_next_cache_maintenance_round
                    )
                    self._pp_next_cache_maintenance_round += 1
                    self._pp_bind_cache_write_owner(None, mb_id)
                self.running_batch = self.running_mbs[mb_id]
                self.last_batch = self.last_mbs[mb_id]
                next_first_rank_mb_id = (mb_id + self.ps.pp_size) % self.pp_loop_size
                next_mb_id = (mb_id + 1) % self.pp_loop_size

                next_pp_outputs = None
                next_release_rids = None
                next_consensus_bootstrapped_rids = None
                d2h_event = None
                next_batch_result = None

                recv_reqs = self.request_receiver.recv_requests()
                authoritative_plan = None
                upstream_plan = None
                if self._pp_batch_contract_guard_enabled:
                    recv_reqs, authoritative_plan = (
                        self._pp_extract_authoritative_batch_plan(mb_id, recv_reqs)
                    )
                    upstream_plan = authoritative_plan
                self.process_input_requests(recv_reqs)
                recv_reqs = self._pp_pd_get_mm_ready_reqs(recv_reqs)

                if not self.pp_group.is_last_rank:
                    self._pp_commit_comm_work(self.send_req_work)

                # Finish any in-flight proxy send before releasing aborted KV.
                if self._pp_pending_cache_read_aborts:
                    self._pp_commit_comm_work(self.send_proxy_work)
                    self._pp_drain_pending_cache_read_aborts()

                bootstrapped_rids = self._pp_pd_get_bootstrapped_ids()
                bmbs[mb_id] = bootstrapped_rids
                self._pp_commit_comm_work(send_bootstrapped_work)

                transferred_rids = self._pp_pd_get_prefill_transferred_ids()
                self._pp_commit_comm_work(send_transfer_work)
                tmbs[mb_id] = transferred_rids

                if (
                    self._pp_batch_contract_guard_enabled
                    and not self.pp_group.is_first_rank
                ):
                    authoritative_plan = (
                        self._pp_project_locally_aborted_chunked_requests(
                            authoritative_plan
                        )
                    )
                    self._pp_install_authoritative_batch_plan(mb_id, authoritative_plan)

                prior_chunked_req = self.chunked_req
                planned_chunked_action = None
                if authoritative_plan is not None:
                    planned_chunked_action = self._pp_validate_chunked_transition(
                        mb_id, authoritative_plan, prior_chunked_req
                    )
                self.process_prefill_chunk(
                    last_batch=self.last_batch,
                    running_batch=self.running_batch,
                    planned_chunked_action=planned_chunked_action,
                )
                # Before get_new_batch_prefill so the pages this releases are
                # already in flight when scheduling looks at them.
                self._pp_flush_pending_final_kv()
                prefill_plan = self.get_new_batch_prefill(
                    self.running_batch, authoritative_plan=authoritative_plan
                )
                batch = prefill_plan.batch_to_run
                self.running_batch = prefill_plan.running_batch
                scheduled_batch = batch
                if (
                    self._pp_batch_contract_guard_enabled
                    and not self.pp_group.is_first_rank
                ):
                    authoritative_plan = self._pp_apply_local_aborts_to_plan(
                        mb_id,
                        batch,
                        authoritative_plan,
                        tuple(prefill_plan.pp_local_aborts),
                    )
                    self._pp_validate_authoritative_batch_plan(
                        mb_id, batch, authoritative_plan
                    )
                batch = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(batch)
                if (
                    self._pp_batch_contract_guard_enabled
                    and self.pp_group.is_first_rank
                ):
                    authoritative_plan = self._pp_make_authoritative_batch_plan(
                        mb_id,
                        scheduled_batch,
                        prior_chunked_req=prior_chunked_req,
                        next_chunked_req=self.chunked_req,
                        local_aborts=tuple(prefill_plan.pp_local_aborts),
                        transport=batch is not None,
                    )
                if self._pp_batch_contract_guard_enabled:
                    self._pp_mb_controls[mb_id] = self._pp_control_from_plan(
                        authoritative_plan
                    )
                    self._pp_bind_cache_write_owner(
                        batch,
                        mb_id,
                        pp_batch_plan_id=int(authoritative_plan["pp_batch_plan_id"]),
                    )
                self.mbs[mb_id] = batch
                self.running_mbs[mb_id] = self.running_batch

                cur_batch: Optional[ScheduleBatch] = self.mbs[mb_id]
                self._pp_bind_cache_write_owner(cur_batch, mb_id)
                self.cur_batch_for_debug = cur_batch
                transport_expected = bool(
                    self._pp_batch_contract_guard_enabled
                    and self._pp_mb_controls[mb_id]["transport"]
                )
                if cur_batch or transport_expected:
                    server_is_idle = False
                if cur_batch or (
                    transport_expected and not self.pp_group.is_first_rank
                ):
                    pp_proxy_tensors = self._pp_recv_proxy_tensors()
                    if (
                        self._pp_batch_contract_guard_enabled
                        and not self.pp_group.is_first_rank
                    ):
                        pp_proxy_tensors = self._pp_prepare_received_proxy(
                            mb_id,
                            cur_batch,
                            pp_proxy_tensors,
                            upstream_plan,
                        )

                if self.server_args.pp_async_batch_depth > 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )
                self._pp_commit_comm_work(self.send_proxy_work)
                if cur_batch:
                    result, self.launch_event = self._pp_launch_batch(
                        mb_id,
                        cur_batch,
                        pp_proxy_tensors,
                        self.mb_metadata,
                        self.last_rank_comm_queue,
                    )
                    # Needs mb_metadata[mb_id].forward_done, recorded just above.
                    self._pp_enqueue_pending_final_kv(batch=cur_batch, mb_id=mb_id)

                if transport_expected and self.pp_group.is_last_rank:
                    control = self._pp_mb_controls[mb_id]
                    if cur_batch:
                        output_tensors = self.last_rank_comm_queue[-1][1].tensors
                        output_tensors[_PP_BATCH_CONTROL_KEY] = control
                    else:
                        event = self.device_module.Event()
                        event.record(self.device_module.current_stream())
                        self.mb_metadata[mb_id] = PPBatchMetadata(
                            can_run_cuda_graph=False,
                            forward_done=event,
                        )
                        self.last_rank_comm_queue.append(
                            (
                                event,
                                PPProxyTensors({_PP_BATCH_CONTROL_KEY: control}),
                            )
                        )
                if self.server_args.pp_async_batch_depth == 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )
                send_consensus_bootstrapped_work, consensus_bootstrapped_rids = (
                    self._pp_pd_send_consensus_bootstrapped_ids(
                        bmbs,
                        next_first_rank_mb_id,
                        consensus_bootstrapped_rids,
                        bootstrapped_rids,
                    )
                )
                send_release_work, release_rids = (
                    self._pp_pd_send_consensus_release_ids(
                        tmbs, next_first_rank_mb_id, release_rids, transferred_rids
                    )
                )

                if bmbs[next_mb_id] is not None:
                    next_consensus_bootstrapped_rids = (
                        self._pp_recv_pyobj_from_prev_stage()
                    )
                    next_consensus_bootstrapped_rids = self.process_bootstrapped_queue(
                        next_consensus_bootstrapped_rids
                    )
                self._pp_commit_comm_work(send_consensus_bootstrapped_work)
                if tmbs[next_mb_id] is not None:
                    next_release_rids = self._pp_recv_pyobj_from_prev_stage()
                self._pp_commit_comm_work(send_release_work)
                # post-process the coming microbatch
                if self.mbs[next_mb_id] is not None:
                    d2h_event.synchronize()
                    # d2h_event misses forward_stream, which reads the pinned pages.
                    if self.disagg_prefill_mm_prealloc_queue is not None:
                        self.mb_metadata[next_mb_id].forward_done.synchronize()
                    self._pp_process_batch_result(
                        self.mbs[next_mb_id],
                        next_batch_result,
                    )
                    self.last_mbs[next_mb_id] = self.mbs[next_mb_id]
                elif (
                    self._pp_batch_contract_guard_enabled
                    and self._pp_mb_controls[next_mb_id] is not None
                    and self._pp_mb_controls[next_mb_id]["transport"]
                ):
                    # Clear stale state after a cancellation-only output.
                    self.last_mbs[next_mb_id] = None

                if tmbs[next_mb_id] is not None:
                    self.process_disagg_prefill_inflight_queue(next_release_rids)
                if not self.pp_group.is_last_rank:
                    relay_reqs = recv_reqs
                    if self._pp_batch_contract_guard_enabled:
                        relay_reqs = [
                            *recv_reqs,
                            PPBatchPlanMessage(
                                mb_id=mb_id,
                                plan=authoritative_plan,
                            ),
                        ]
                    self.send_req_work = self._pp_send_pyobj_to_next_stage(
                        relay_reqs, async_send=True
                    )
                    send_bootstrapped_work = self._pp_send_pyobj_to_next_stage(
                        bootstrapped_rids, async_send=True
                    )
                    send_transfer_work = self._pp_send_pyobj_to_next_stage(
                        transferred_rids, async_send=True
                    )
                    if cur_batch or transport_expected:
                        if cur_batch:
                            self.device_module.current_stream().wait_event(
                                self.launch_event
                            )
                            proxy_tensors = (
                                dict(result.pp_hidden_states_proxy_tensors.tensors)
                                if self._pp_batch_contract_guard_enabled
                                else result.pp_hidden_states_proxy_tensors.tensors
                            )
                        else:
                            proxy_tensors = {}
                        if self._pp_batch_contract_guard_enabled:
                            proxy_tensors[_PP_BATCH_CONTROL_KEY] = self._pp_mb_controls[
                                mb_id
                            ]
                        self.send_proxy_work = self._pp_send_dict_to_next_stage(
                            proxy_tensors,
                            async_send=True,
                            msg_type="proxy",
                        )

                self.pp_outputs = next_pp_outputs
                release_rids = next_release_rids
                consensus_bootstrapped_rids = next_consensus_bootstrapped_rids

                self.running_batch.batch_is_full = False

            # When the server is idle, self-check and re-init some states
            if server_is_idle and len(self.disagg_prefill_inflight_queue) == 0:
                self.on_idle()

    @DynamicGradMode()
    def event_loop_pp_disagg_decode(self: Scheduler):
        self.init_pp_loop_state()

        # PD additional state initialization
        rmbs = [None] * self.pp_loop_size
        pmbs = [None] * self.pp_loop_size
        tmbs = [None] * self.pp_loop_size
        consensus_retract_rids: Optional[List[str]] = None
        consensus_prealloc_rids: Optional[List[str]] = None
        release_rids: Optional[List[str]] = None  # consensus transferred rids
        send_retract_work = []
        send_prealloc_work = []
        send_transfer_work = []
        send_consensus_retract_work = []
        send_consensus_prealloc_work = []
        send_release_work = []

        while True:
            server_is_idle = True
            for mb_id in range(self.pp_loop_size):
                if self.enable_asradix_cache:
                    self._pp_current_cache_maintenance_round = (
                        self._pp_next_cache_maintenance_round
                    )
                    self._pp_next_cache_maintenance_round += 1
                    self._pp_bind_cache_write_owner(None, mb_id)
                self.running_batch = self.running_mbs[mb_id]
                self.last_batch = self.last_mbs[mb_id]
                next_first_rank_mb_id = (mb_id + self.ps.pp_size) % self.pp_loop_size
                next_mb_id = (mb_id + 1) % self.pp_loop_size

                next_pp_outputs = None
                next_consensus_retract_rids = None
                next_consensus_prealloc_rids = None
                next_release_rids = None
                d2h_event = None
                next_batch_result = None

                recv_reqs = self.request_receiver.recv_requests()
                self.process_input_requests(recv_reqs)

                if not self.pp_group.is_last_rank:
                    self._pp_commit_comm_work(self.send_req_work)

                # reaching consensus through PP ranks
                retract_rids = self._pp_pd_get_retract_ids(mb_id)
                rmbs[mb_id] = retract_rids
                self._pp_commit_comm_work(send_retract_work)

                prealloc_rids = self._pp_pd_get_prealloc_ids()
                pmbs[mb_id] = prealloc_rids
                self._pp_commit_comm_work(send_prealloc_work)

                transferred_rids = self._pp_pd_get_decode_transferred_ids()
                tmbs[mb_id] = transferred_rids
                self._pp_commit_comm_work(send_transfer_work)

                # get batch to run and proxy tensors if needed
                plan = self.get_next_disagg_decode_batch_to_run(
                    running_batch=self.running_batch
                )
                self.running_batch = plan.running_batch
                batch = plan.batch_to_run
                self.mbs[mb_id] = batch
                self.running_mbs[mb_id] = self.running_batch

                cur_batch: Optional[ScheduleBatch] = self.mbs[mb_id]
                self._pp_bind_cache_write_owner(cur_batch, mb_id)
                self.cur_batch_for_debug = cur_batch
                if cur_batch:
                    server_is_idle = False
                    pp_proxy_tensors = None
                    if not cur_batch.forward_mode.is_prebuilt():
                        pp_proxy_tensors = self._pp_recv_proxy_tensors()

                # early send output if possible
                if self.server_args.pp_async_batch_depth > 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )
                self._pp_commit_comm_work(self.send_proxy_work)

                if cur_batch:
                    result, self.launch_event = self._pp_launch_batch(
                        mb_id,
                        cur_batch,
                        pp_proxy_tensors,
                        self.mb_metadata,
                        self.last_rank_comm_queue,
                    )

                if self.server_args.pp_async_batch_depth == 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )

                # reach consensus on last rank and send to PP=0
                # otherwise, just pass along previous consensus
                send_consensus_retract_work, consensus_retract_rids = (
                    self._pp_pd_send_consensus_bootstrapped_ids(
                        rmbs,
                        next_first_rank_mb_id,
                        consensus_retract_rids,
                        retract_rids,
                    )
                )

                send_consensus_prealloc_work, consensus_prealloc_rids = (
                    self._pp_pd_send_consensus_bootstrapped_ids(
                        pmbs,
                        next_first_rank_mb_id,
                        consensus_prealloc_rids,
                        prealloc_rids,
                    )
                )

                send_release_work, release_rids = (
                    self._pp_pd_send_consensus_release_ids(
                        tmbs, next_first_rank_mb_id, release_rids, transferred_rids
                    )
                )

                if self.server_args.disaggregation_decode_enable_offload_kvcache:
                    self.decode_offload_manager.check_offload_progress()

                if rmbs[next_mb_id] is not None:
                    next_consensus_retract_rids = self._pp_recv_pyobj_from_prev_stage()
                    next_consensus_retract_rids = self.process_retract_queue(
                        next_consensus_retract_rids
                    )
                self._pp_commit_comm_work(send_consensus_retract_work)

                if pmbs[next_mb_id] is not None:
                    next_consensus_prealloc_rids = self._pp_recv_pyobj_from_prev_stage()
                    next_consensus_prealloc_rids = self.process_prealloc_queue(
                        next_consensus_prealloc_rids
                    )
                self._pp_commit_comm_work(send_consensus_prealloc_work)

                if tmbs[next_mb_id] is not None:
                    next_release_rids = self._pp_recv_pyobj_from_prev_stage()
                    next_release_rids = self.process_decode_transfer_queue(
                        next_release_rids
                    )
                self._pp_commit_comm_work(send_release_work)

                # post-process the coming microbatch
                if self.mbs[next_mb_id] is not None:
                    if not self.mbs[next_mb_id].forward_mode.is_prebuilt():
                        d2h_event.synchronize()
                        self._pp_process_batch_result(
                            self.mbs[next_mb_id],
                            next_batch_result,
                        )
                    self.last_mbs[next_mb_id] = self.mbs[next_mb_id]

                if not self.pp_group.is_last_rank:
                    self.send_req_work = self._pp_send_pyobj_to_next_stage(
                        recv_reqs, async_send=True
                    )
                    send_retract_work = self._pp_send_pyobj_to_next_stage(
                        retract_rids, async_send=True
                    )
                    send_prealloc_work = self._pp_send_pyobj_to_next_stage(
                        prealloc_rids, async_send=True
                    )
                    send_transfer_work = self._pp_send_pyobj_to_next_stage(
                        transferred_rids, async_send=True
                    )
                    if cur_batch and not cur_batch.forward_mode.is_prebuilt():
                        self.device_module.current_stream().wait_event(
                            self.launch_event
                        )
                        self.send_proxy_work = self._pp_send_dict_to_next_stage(
                            result.pp_hidden_states_proxy_tensors.tensors,
                            async_send=True,
                            msg_type="proxy",
                        )

                self.pp_outputs = next_pp_outputs
                release_rids = next_release_rids
                consensus_retract_rids = next_consensus_retract_rids
                consensus_prealloc_rids = next_consensus_prealloc_rids

                self.running_batch.batch_is_full = False

            # When the server is idle, self-check and re-init some states
            queue_size = (
                len(self.waiting_queue)
                + len(self.disagg_decode_transfer_queue.queue)
                + len(self.disagg_decode_prealloc_queue.queue)
            )
            if self.server_args.disaggregation_decode_enable_offload_kvcache:
                queue_size += len(self.decode_offload_manager.ongoing_offload)

            if server_is_idle and queue_size == 0:
                self.on_idle()

    def init_pp_loop_state(self: Scheduler):
        self.pp_loop_size: int = self.ps.pp_size + self.server_args.pp_async_batch_depth
        self._pp_batch_contract_guard_enabled = False
        # In CP mode, attention weights are duplicated, eliminating the need for the attention TP all-gather operation.
        self.require_attn_tp_allgather = (
            not self.server_args.enable_dsa_prefill_context_parallel
        )
        self.mbs = [None] * self.pp_loop_size
        self.last_mbs = [None] * self.pp_loop_size
        self.running_mbs = [
            ScheduleBatch(reqs=[], batch_is_full=False)
            for _ in range(self.pp_loop_size)
        ]
        self.mb_metadata: List[Optional[PPBatchMetadata]] = [None] * self.pp_loop_size
        self._pp_mb_controls: List[Optional[Dict[str, Any]]] = [
            None
        ] * self.pp_loop_size
        self._pp_next_batch_plan_id = 0
        self._pp_last_received_batch_plan_id = -1
        self._pp_next_cache_maintenance_round = 0
        self._pp_current_cache_maintenance_round = None
        self._pp_abort_tombstones: Dict[Tuple[int, str], Dict[str, Any]] = {}
        self._pp_local_chunk_abort_records: Dict[str, Dict[str, Any]] = {}
        self._pp_pending_cache_read_aborts: Dict[int, Tuple[Req, Dict[str, Any]]] = {}
        self.pp_outputs: Optional[PPProxyTensors] = None
        self.last_rank_comm_queue: deque[Tuple[torch.Event, PPProxyTensors]] = deque()

        self.send_req_work = []
        self.send_proxy_work = []
        self.send_output_work = []
        self.launch_event = None
        self._pp_tensor_dict_inbox: Dict[str, deque[Dict[str, torch.Tensor]]] = (
            defaultdict(deque)
        )

    def profile_and_init_predictor(self: Scheduler):
        """
        Profile prefill latency for dynamic chunk sizing.

        Only runs on PP0 (first rank), then broadcasts data to all ranks.
        All ranks fit coefficients using the same data.
        """
        seq_lens: List[int] = []
        latencies: List[float] = []

        if self.pp_group.is_first_rank:
            model_runner = self.tp_worker.model_runner
            model_config = model_runner.model_config
            input_ids_list: List[array[int]] = []
            for i in range(128):
                chunk_size = int(
                    self.chunked_prefill_size * 1.25
                    - i * (self.chunked_prefill_size * 1.25 // 128)
                )
                if chunk_size <= 0:
                    break
                input_ids = array(
                    "q",
                    np.random.randint(
                        0, 10000, size=chunk_size, dtype=np.int64
                    ).tobytes(),
                )
                input_ids_list.append(input_ids)

            sampling_params = SamplingParams(
                temperature=0,
                max_new_tokens=1,
            )
            # Create and profile requests
            for i, input_ids in enumerate(
                tqdm(
                    input_ids_list,
                    desc="Profiling prefill latency for dynamic chunking",
                )
            ):
                req = Req(
                    rid=str(i),
                    origin_input_text="",
                    origin_input_ids=input_ids,
                    sampling_params=sampling_params,
                )
                req.full_untruncated_fill_ids = req.origin_input_ids
                req.logprob_start_len = -1
                req.set_extend_range(
                    len(req.prefix_indices), len(req.full_untruncated_fill_ids)
                )

                # Prepare batch
                batch = ScheduleBatch.init_new(
                    [req],
                    self.req_to_token_pool,
                    self.token_to_kv_pool_allocator,
                    self.tree_cache,
                    self.model_config,
                    False,
                    self.spec_algorithm,
                )

                current_seq_len = req.extend_range.end

                if is_dp_attention_enabled():
                    # For profiling, we only have one request on PP0
                    # Set global_num_tokens to indicate this rank has tokens, others have 0
                    dp_size = get_attention_dp_size()
                    global_num_tokens = [0] * dp_size
                    dp_rank = get_attention_dp_rank()
                    global_num_tokens[dp_rank] = current_seq_len
                    batch.global_num_tokens = global_num_tokens
                    batch.global_num_tokens_for_logprob = global_num_tokens

                hs = (
                    getattr(model_config, "hc_hidden_size", None)
                    or model_config.hidden_size
                )
                proxy_tensors = {
                    "hidden_states": torch.zeros(
                        (current_seq_len, hs),
                        dtype=model_config.dtype,
                        device=self.device,
                    ),
                    "residual": torch.zeros(
                        (current_seq_len, model_config.hidden_size),
                        dtype=model_config.dtype,
                        device=self.device,
                    ),
                }
                pp_proxy_topk_size = model_runner.get_pp_proxy_topk_size()
                if pp_proxy_topk_size is not None:
                    proxy_tensors["topk_indices"] = torch.zeros(
                        (current_seq_len, pp_proxy_topk_size),
                        dtype=torch.int32,
                        device=self.device,
                    )

                pp_proxy = PPProxyTensors(proxy_tensors)

                # Measure latency with device synchronization for accurate timing
                device_module = get_device_module()
                # Synchronize before starting timing to ensure clean measurement
                device_module.synchronize()

                start = time.perf_counter()
                batch.prepare_for_extend()

                # Resolve deferred H2D: prepare_for_extend now leaves input_ids=None
                if batch.input_ids is None and batch.prefill_input_ids_cpu is not None:
                    batch.input_ids = batch.prefill_input_ids_cpu.to(
                        self.device, non_blocking=True
                    )
                    batch.prefill_input_ids_cpu = None

                forward_batch = ForwardBatch.init_new(
                    batch,
                    model_runner,
                    return_hidden_states_before_norm=False,
                )
                set_is_extend_in_batch(batch.forward_mode.is_extend())

                _ = model_runner.forward(
                    forward_batch=forward_batch, pp_proxy_tensors=pp_proxy
                )

                # Synchronize after forward to ensure GPU operations complete
                device_module.synchronize()

                latency_seconds = time.perf_counter() - start
                latency_ms = latency_seconds * 1e3  # Convert to milliseconds
                seq_lens.append(len(input_ids))
                latencies.append(latency_ms)

                # Release KV and Mamba cache
                if req.req_pool_idx is not None:
                    kv_indices = self.req_to_token_pool.req_to_token[
                        req.req_pool_idx, : req.extend_range.end
                    ]
                    self.token_to_kv_pool_allocator.free(kv_indices)
                    if req.mamba_pool_idx is not None:
                        self.req_to_token_pool.free_mamba_cache(req)
                    self.req_to_token_pool.free(req)
                    req.kv = None

            logger.info(
                f"[PP Dynamic Chunk] [PP0] Profiled {len(seq_lens)} samples: "
                f"seq_lens={seq_lens}, latencies_ms={latencies}"
            )

            if self.ps.attn_tp_size > 1:
                data_to_sync_tp = [seq_lens, latencies]
                data_to_sync_tp = broadcast_pyobj(
                    data_to_sync_tp,
                    self.attn_tp_group.rank,
                    self.attn_tp_cpu_group,
                    src=self.attn_tp_group.ranks[0],
                )
                seq_lens, latencies = data_to_sync_tp

            if self.ps.attn_cp_size > 1:
                data_to_sync_tp = [seq_lens, latencies]
                data_to_sync_tp = broadcast_pyobj(
                    data_to_sync_tp,
                    self.attn_cp_group.rank,
                    self.attn_cp_cpu_group,
                    src=self.attn_cp_group.ranks[0],
                )

        # Broadcast data to all ranks
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            data_to_sync = [seq_lens, latencies]
            self.pp_group.broadcast_object_list(data_to_sync, src=0)
            seq_lens, latencies = data_to_sync

        # Quadratic model: f(l) = al^2 + bl + c
        self.length_predictor = ChunkSizePredictor()
        self.length_predictor.fit(seq_lens, latencies)
        self.length_predictor.set_target_latency(self.chunked_prefill_size)
        self.length_predictor.is_ready = True
        logger.info(
            f"[PP Dynamic Chunk] [PP{self.ps.pp_rank}] Predictor ready (quadratic). "
            f"Target latency: {self.length_predictor.target_latency:.2f}ms"
        )

    def predict_next_chunk_size(self: Scheduler, history_len: int) -> Optional[int]:
        """
        Predict next chunk size dynamically based on current history length.

        Args:
            history_len: Current sequence length

        Returns:
            Predicted chunk size, or None to use default chunked_prefill_size
        """
        if (
            not self.enable_dynamic_chunking
            or self.length_predictor is None
            or not self.length_predictor.is_ready
        ):
            return None

        max_chunk_size = self.max_prefill_tokens
        predicted_size = self.length_predictor.predict_next_chunk_size(
            history_len=history_len,
            base_chunk_size=self.chunked_prefill_size,
            page_size=self.page_size,
            context_len=self.model_config.context_len,
            max_chunk_size=max_chunk_size,
        )

        if predicted_size is not None:
            logger.debug(
                f"[PP Dynamic Chunk] [PP{self.ps.pp_rank}] Predicted chunk size: "
                f"{predicted_size} (history_len={history_len})"
            )

        return predicted_size

    def process_bootstrapped_queue(
        self: Scheduler, bootstrapped_rids: Optional[List[str]]
    ):
        # finished consensus bootstrapped reqs and prepare the waiting queue
        if bootstrapped_rids is not None:
            (
                good_consensus_bootstrapped_rids,
                bad_consensus_bootstrapped_rids,
            ) = bootstrapped_rids
            good_reqs, failed_reqs = (
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped(
                    return_failed_reqs=True,
                    pp_good_rids=good_consensus_bootstrapped_rids,
                    pp_bad_rids=bad_consensus_bootstrapped_rids,
                )
            )
            self.waiting_queue.extend(good_reqs)
            return [
                [req.rid for req in good_reqs],
                _pp_ordered_union(
                    bad_consensus_bootstrapped_rids,
                    [req.rid for req in failed_reqs],
                ),
            ]
        return None

    def _pp_pd_get_mm_ready_reqs(self: Scheduler, recv_reqs: List) -> List:
        """Admit EPD requests locally and append the copies to relay downstream."""
        mm_ready = self.process_disagg_prefill_mm_queues()
        if not mm_ready:
            return recv_reqs

        relayed = []
        admitted = []
        for req in mm_ready:
            if not self.pp_group.is_last_rank:
                try:
                    relayed.append(metadata_only_recv_req(req))
                except ValueError as e:
                    logger.error("EPD relay rejected %s: %s", req.rid, e)
                    self._pp_abort_unrelayable_mm_req(req, str(e))
                    continue
            admitted.append(req)

        if admitted:
            self.process_input_requests(admitted)
        if not relayed:
            return recv_reqs
        return recv_reqs + relayed

    def _pp_abort_unrelayable_mm_req(self: Scheduler, recv_req, error_msg: str) -> None:
        """Fail one un-relayable EPD request instead of every rank."""
        req = self.mm_receiver.create_req(recv_req)
        prepare_abort(req, error_msg, status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
        self.output_streamer.stream_output([req], req.return_logprob)

    def _pp_pd_get_bootstrapped_ids(self: Scheduler):
        # communicate pre-consensus bootstrapp reqs
        if self.pp_group.is_first_rank:
            good_bootstrapped_rids, bad_bootstrapped_rids = (
                self.disagg_prefill_bootstrap_queue.get_ready_bootstrapped_rids_for_pp()
            )
        else:
            # Other ranks, receive the bootstrap reqs info from the previous rank and ensure the consensus
            prev_bootstrapped_rids = self._pp_recv_pyobj_from_prev_stage()
            prev_good_bootstrapped_rids, prev_bad_bootstrapped_rids = (
                prev_bootstrapped_rids
            )
            curr_good_bootstrapped_rids, curr_bad_bootstrapped_rids = (
                self.disagg_prefill_bootstrap_queue.get_ready_bootstrapped_rids_for_pp()
            )
            good_bootstrapped_rids = _pp_ordered_intersection(
                prev_good_bootstrapped_rids,
                curr_good_bootstrapped_rids,
            )
            bad_bootstrapped_rids = _pp_ordered_union(
                prev_bad_bootstrapped_rids,
                curr_bad_bootstrapped_rids,
            )
        # Route locally-aborted reqs through the bad-union consensus so every PP
        # rank flushes them in the same consensus round, regardless of when the
        # AbortReq reaches each rank and regardless of whether
        # disagg_kv_sender.abort() drives the poll to Failed (it is optional).
        aborted_rids = {
            req.rid
            for req in self.disagg_prefill_bootstrap_queue.queue
            if isinstance(req.finished_reason, FINISH_ABORT)
        }
        good_bootstrapped_rids, bad_bootstrapped_rids = _pp_route_aborts_to_bad(
            good_bootstrapped_rids, bad_bootstrapped_rids, aborted_rids
        )
        return [good_bootstrapped_rids, bad_bootstrapped_rids]

    def _pp_pd_get_prefill_transferred_ids(
        self: Scheduler,
    ) -> PPTransferStatus:
        # get the current stage transfer success
        current_status = self.get_transferred_rids()
        if self.pp_group.is_first_rank:
            transferred_rids = current_status
        # if other ranks, do intersection with the previous rank's transferred rids
        else:
            # 2 (Release): Receive the transferred rids from the previous rank
            # 1. recv previous stage's transferred reqs info
            previous_status = self._pp_recv_pyobj_from_prev_stage()
            transferred_rids = _pp_merge_transfer_status(
                previous=previous_status,
                current=current_status,
            )
        return transferred_rids

    def _pp_pd_send_consensus_bootstrapped_ids(
        self: Scheduler,
        bmbs: List[List[str]],
        next_first_rank_mb_id: int,
        consensus_bootstrapped_rids: List[str],
        bootstrapped_rids: List[str],
    ):
        # 3 (Release): send the release rids from last stage to the first stage
        send_consensus_bootstrapped_work = []
        if self.pp_group.is_last_rank:
            if bmbs[next_first_rank_mb_id] is not None:
                consensus_bootstrapped_rids = bootstrapped_rids
                send_consensus_bootstrapped_work = self._pp_send_pyobj_to_next_stage(
                    consensus_bootstrapped_rids, async_send=True
                )
        # 4 (Release): send the release rids from non last rank to the next rank
        else:
            if consensus_bootstrapped_rids is not None:
                send_consensus_bootstrapped_work = self._pp_send_pyobj_to_next_stage(
                    consensus_bootstrapped_rids, async_send=True
                )
        return send_consensus_bootstrapped_work, consensus_bootstrapped_rids

    def _pp_pd_send_consensus_release_ids(
        self: Scheduler,
        tmbs: List[Optional[PPReleasePayload]],
        next_first_rank_mb_id: int,
        release_rids: Optional[PPReleasePayload],
        transferred_rids: PPReleasePayload,
    ):
        send_release_work = []
        if self.pp_group.is_last_rank:
            if tmbs[next_first_rank_mb_id] is not None:
                release_rids = transferred_rids
                send_release_work = self._pp_send_pyobj_to_next_stage(
                    release_rids, async_send=True
                )
        # 4 (Release): send the release rids from non last rank to the next rank
        else:
            if release_rids is not None:
                send_release_work = self._pp_send_pyobj_to_next_stage(
                    release_rids, async_send=True
                )
        return send_release_work, release_rids

    def _pp_commit_comm_work(self: Scheduler, work: List[P2PWork]) -> None:
        for p2p_work in work:
            p2p_work.work.wait()
        work.clear()

    def _pp_commit_send_output_work_and_preprocess_output_tensors(
        self: Scheduler,
        next_first_rank_mb_id: int,
        next_mb_id: int,
    ) -> Tuple[
        Optional[PPProxyTensors],
        Optional[GenerationBatchResult],
        Optional[torch.Event],
    ]:
        self._pp_commit_comm_work(work=self.send_output_work)
        (
            next_pp_outputs,
            next_batch_result,
            d2h_event,
            self.send_output_work,
        ) = self._pp_send_recv_and_preprocess_output_tensors(
            next_first_rank_mb_id,
            next_mb_id,
            self.mbs,
            self.mb_metadata,
            self.last_rank_comm_queue,
            self.pp_outputs,
        )
        return next_pp_outputs, next_batch_result, d2h_event

    def _pp_send_pyobj_to_next_stage(self: Scheduler, data, async_send: bool = False):
        p2p_work = []
        if self.ps.attn_tp_rank == 0 and self.ps.attn_cp_rank == 0:
            dp_offset = (
                self.ps.attn_dp_rank * self.ps.attn_cp_size * self.ps.attn_tp_size
            )
            p2p_work = point_to_point_pyobj(
                data,
                self.ps.pp_rank * self.ps.tp_size + dp_offset,
                self.world_group.cpu_group,
                self.ps.pp_rank * self.ps.tp_size + dp_offset,
                ((self.ps.pp_rank + 1) % self.ps.pp_size) * self.ps.tp_size + dp_offset,
                async_send=async_send,
            )
        return p2p_work

    def _pp_recv_pyobj_from_prev_stage(self: Scheduler):
        if self.ps.attn_tp_rank == 0 and self.ps.attn_cp_rank == 0:
            dp_offset = (
                self.ps.attn_dp_rank * self.ps.attn_cp_size * self.ps.attn_tp_size
            )
            data = point_to_point_pyobj(
                [],
                self.ps.pp_rank * self.ps.tp_size + dp_offset,
                self.world_group.cpu_group,
                ((self.ps.pp_rank - 1) % self.ps.pp_size) * self.ps.tp_size + dp_offset,
                self.ps.pp_rank * self.ps.tp_size + dp_offset,
            )
        else:
            data = None

        if self.ps.attn_tp_size > 1:
            data = broadcast_pyobj(
                data,
                self.attn_tp_group.rank,
                self.attn_tp_cpu_group,
                src=self.attn_tp_group.ranks[0],
            )

        if self.ps.attn_cp_size > 1:
            data = broadcast_pyobj(
                data,
                self.attn_cp_group.rank,
                self.attn_cp_cpu_group,
                src=self.attn_cp_group.ranks[0],
            )

        return data

    def _pp_prepare_tensor_dict(
        self: Scheduler, result: GenerationBatchResult, batch: ScheduleBatch
    ) -> Dict[str, torch.Tensor]:
        tensor_dict = {
            "next_token_ids": result.next_token_ids,
        }

        if batch.return_logprob:
            logprob_dict = get_logprob_dict_from_result(result)
            tensor_dict = {
                **tensor_dict,
                **logprob_dict,
            }
        return tensor_dict

    def _pp_send_dict_to_next_stage(
        self: Scheduler,
        tensor_dict: Dict[str, Any],
        async_send: bool = True,
        msg_type: str = "default",
    ):
        # Warn once if using default untyped messages
        if msg_type == "default":
            logger.warning_once(
                "PP send: using default untyped message. "
                "Consider adding msg_type='proxy' or 'output' to avoid recv conflicts."
            )
        tensor_dict["__msg_type__"] = msg_type
        p2p_work = []
        p2p_work.extend(
            self.pp_group.send_tensor_dict(
                tensor_dict=tensor_dict,
                all_gather_group=(
                    self.attn_tp_group if self.require_attn_tp_allgather else None
                ),
                async_send=async_send,
            )
        )
        return p2p_work

    def _pp_recv_typed_dict(
        self: Scheduler,
        expected_kind: str = "default",
        all_gather_group: Optional = None,
    ) -> Dict[str, Any]:
        """Receive a typed tensor dict, demultiplexing by msg_type.

        If a message of the wrong kind is received, it's stashed in the queue
        and we continue receiving until we get the expected kind.
        """
        if expected_kind in self._pp_tensor_dict_inbox:
            inbox_queue = self._pp_tensor_dict_inbox[expected_kind]
            if inbox_queue:
                return inbox_queue.popleft()

        while True:
            tensor_dict = self.pp_group.recv_tensor_dict(
                all_gather_group=all_gather_group
            )
            received_kind = tensor_dict.get("__msg_type__", "default")
            if received_kind == expected_kind:
                if received_kind == "default":
                    logger.warning_once(
                        f"PP recv: got default untyped message. Content keys: {tensor_dict.keys()}"
                        "Consider adding msg_type='proxy' or 'output' to avoid recv conflicts."
                    )
                return tensor_dict
            else:
                logger.debug(
                    f"PP recv: expected {expected_kind}, got {received_kind}, stashing"
                )
                self._pp_tensor_dict_inbox[received_kind].append(tensor_dict)

    def _pp_recv_proxy_tensors(self: Scheduler) -> Optional[PPProxyTensors]:
        pp_proxy_tensors = None
        if not self.pp_group.is_first_rank:
            pp_proxy_tensors = PPProxyTensors(
                self._pp_recv_typed_dict(
                    expected_kind="proxy",
                    all_gather_group=(
                        self.attn_tp_group if self.require_attn_tp_allgather else None
                    ),
                )
            )
        return pp_proxy_tensors

    def _pp_recv_dict_from_prev_stage(
        self: Scheduler,
    ) -> Dict[str, Any]:
        return self._pp_recv_typed_dict(
            expected_kind="output",
            all_gather_group=(
                self.attn_tp_group if self.require_attn_tp_allgather else None
            ),
        )

    def _pp_make_skip_output_result(
        self: Scheduler,
        batch: ScheduleBatch,
        mb_metadata: Optional[PPBatchMetadata],
    ):
        bs = len(batch.reqs)
        placeholder = torch.zeros(bs, dtype=torch.int64, device=self.device)
        # next_pp_outputs = None so non-last ranks skip forwarding
        # (pp_outputs is None gate). Placeholder carried in
        # batch_result.next_token_ids for process_batch_result_prefill.
        batch_result = GenerationBatchResult(
            logits_output=None,
            pp_hidden_states_proxy_tensors=None,
            next_token_ids=placeholder,
            can_run_cuda_graph=(
                mb_metadata.can_run_cuda_graph if mb_metadata else False
            ),
            skipped_output_comm=True,
        )
        d2h_event = self.device_module.Event()
        d2h_event.record(self.device_module.current_stream())
        return None, batch_result, d2h_event

    def _pp_prep_batch_result(
        self: Scheduler,
        batch: ScheduleBatch,
        mb_metadata: PPBatchMetadata,
        pp_outputs: PPProxyTensors,
    ):
        from sglang.srt.managers.scheduler import GenerationBatchResult

        logits_output = None
        extend_input_len_per_req = None
        extend_logprob_start_len_per_req = None

        if batch.return_logprob:
            (
                logits_output,
                extend_input_len_per_req,
                extend_logprob_start_len_per_req,
            ) = get_logprob_from_pp_outputs(pp_outputs)
        # PP outputs may be on CPU after result processing, while draft state is
        # filtered with device-resident batch indices.
        next_token_ids = pp_outputs["next_token_ids"].to(
            device=batch.device,
            dtype=torch.int64,
            non_blocking=True,
        )
        # Fail loud on a PP batch divergence: if the last stage sampled a
        # different number of tokens than this rank's batch holds, the relay
        # below would either raise an opaque broadcast error or, when the two
        # sizes happen to match with different members, silently write tokens
        # into the wrong slots.
        if next_token_ids.shape[0] != batch.req_pool_indices.shape[0]:
            raise RuntimeError(
                "PP batch size mismatch between stages: "
                f"next_token_ids={next_token_ids.shape[0]} "
                f"req_pool_indices={batch.req_pool_indices.shape[0]}"
            )
        # PP rank 0 also relays into output_tokens_buf so the next iter's
        # resolve_forward_inputs finds these tokens for the decode portion
        # of mixed-chunk batches (which gather via mix_running_indices).
        self.future_map.stash(
            batch.req_pool_indices, RelayPayload(bonus_tokens=next_token_ids)
        )
        next_draft_input = None
        if batch.spec_algorithm.is_dspark():
            from sglang.srt.speculative.dspark_components.dspark_draft import (
                make_next_draft_input,
            )

            next_draft_input = make_next_draft_input(
                bonus_tokens=next_token_ids,
                new_seq_lens=batch.seq_lens,
            )
            batch.spec_info = next_draft_input
        batch.input_ids = None
        output_result = GenerationBatchResult(
            logits_output=logits_output,
            pp_hidden_states_proxy_tensors=None,
            next_token_ids=pp_outputs["next_token_ids"],
            extend_input_len_per_req=extend_input_len_per_req,
            extend_logprob_start_len_per_req=extend_logprob_start_len_per_req,
            can_run_cuda_graph=mb_metadata.can_run_cuda_graph,
            next_draft_input=next_draft_input,
        )
        return output_result

    def _pp_bind_cache_write_owner(
        self: Scheduler,
        batch: Optional[ScheduleBatch],
        mb_id: int,
        pp_batch_plan_id: Optional[int] = None,
    ) -> None:
        """Bind cache commits made in this scheduler round to PP work."""
        if not self.enable_asradix_cache:
            return
        maintenance_round = getattr(self, "_pp_current_cache_maintenance_round", None)
        if maintenance_round is None:
            raise RuntimeError("missing PP cache maintenance round for batch owner")
        existing_owner = getattr(batch, "cache_write_owner", None)
        if pp_batch_plan_id is None and existing_owner is not None:
            pp_batch_plan_id = existing_owner.pp_batch_plan_id
        owner = AttentionStoreWriteOwner(
            maintenance_round=int(maintenance_round),
            pp_batch_plan_id=(
                None if pp_batch_plan_id is None else int(pp_batch_plan_id)
            ),
            mb_id=int(mb_id),
        )
        # Keep a round owner active during scheduler-side cache commits (for
        # example, stashing a previous chunk before admission). The result
        # path installs the more specific batch owner temporarily below.
        self.tree_cache.set_attention_store_write_owner(owner)
        if batch is None:
            return
        batch.cache_write_owner = owner
        for req in batch.reqs:
            req._as_write_owner = AttentionStoreWriteOwner(
                maintenance_round=owner.maintenance_round,
                pp_batch_plan_id=owner.pp_batch_plan_id,
                mb_id=owner.mb_id,
                rid=req.rid,
            )

    def _pp_process_batch_result(
        self: Scheduler, batch: ScheduleBatch, output_result: GenerationBatchResult
    ):
        owner = getattr(batch, "cache_write_owner", None)
        current_round = getattr(self, "_pp_current_cache_maintenance_round", None)
        if owner is not None and current_round is not None:
            # A result can be delayed by PP async depth.  The logical owner
            # remains the batch plan, but retirement ordering must use the
            # round in which this commit is actually enqueued.
            owner = AttentionStoreWriteOwner(
                maintenance_round=int(current_round),
                pp_batch_plan_id=owner.pp_batch_plan_id,
                mb_id=owner.mb_id,
            )
        cache = self.tree_cache
        context = (
            cache.attention_store_write_context(owner)
            if owner is not None and hasattr(cache, "attention_store_write_context")
            else nullcontext()
        )
        with context:
            self.process_batch_result(batch, output_result)

    def _pp_send_output_to_next_stage(
        self: Scheduler,
        next_first_rank_mb_id: int,
        mbs: List[ScheduleBatch],
        last_rank_comm_queue: deque,
        pp_outputs: PPProxyTensors | None,
    ) -> List[P2PWork]:
        send_output_work = []
        if self.pp_group.is_last_rank:
            # send ready PP output to rank 0
            target = mbs[next_first_rank_mb_id]
            target_control = self._pp_mb_controls[next_first_rank_mb_id]
            control_transport = bool(
                self._pp_batch_contract_guard_enabled
                and target_control is not None
                and target_control["transport"]
            )
            if target is not None or control_transport:
                q_event, pp_outputs_to_send = last_rank_comm_queue.popleft()
                if target is None or (
                    not target.forward_mode.is_prebuilt()
                    and (
                        not _pp_can_skip_output_comm(target)
                        or _pp_control_has_current_aborts(target_control)
                    )
                ):
                    self.device_module.current_stream().wait_event(q_event)
                    with torch.profiler.record_function("send_res_dict_to_next_stage"):
                        send_output_work = self._pp_send_dict_to_next_stage(
                            pp_outputs_to_send.tensors,
                            async_send=True,
                            msg_type="output",
                        )
        # send the outputs from the last round to let the next stage worker run post processing
        # The early-last-chunk path collapses the ring to a single last-rank -> PP0
        # hop: no rank other than the last one needs to relay outputs, because no
        # rank other than PP0 consumes next_token_ids any more.
        if not self.pp_group.is_last_rank and not self._pp_pd_early_last_chunk_enabled:
            if pp_outputs:
                with torch.profiler.record_function("send_res_dict_to_next_stage"):
                    send_output_work = self._pp_send_dict_to_next_stage(
                        pp_outputs.tensors,
                        async_send=True,
                        msg_type="output",
                    )
        return send_output_work

    def _pp_reconcile_output_aborts(
        self,
        mb_id: int,
        batch: Optional[ScheduleBatch],
        pp_outputs: PPProxyTensors,
    ) -> Optional[ScheduleBatch]:
        """Apply finalized PP aborts and return the surviving output batch."""

        expected_control = self._pp_mb_controls[mb_id]
        remote_control = pp_outputs.tensors.get(_PP_BATCH_CONTROL_KEY)
        if expected_control is None or remote_control is None:
            raise RuntimeError(f"missing PP cancellation control for mb_id={mb_id}")
        if (
            remote_control.get("version") != expected_control.get("version")
            or remote_control.get("pp_batch_plan_id")
            != expected_control.get("pp_batch_plan_id")
            or remote_control.get("maintenance_round")
            != expected_control.get("maintenance_round")
            or remote_control.get("mb_id") != mb_id
            or not remote_control.get("transport")
        ):
            raise RuntimeError(
                "PP output cancellation control mismatch: "
                f"local={expected_control} remote={remote_control}"
            )

        final_contract = remote_control.get("contract")
        local_contract = expected_control.get("contract")
        final_rids = tuple(final_contract.get("rids", ()))
        projected = _pp_project_contract(local_contract, final_rids)
        if projected != final_contract:
            raise RuntimeError(
                "PP output contract is not a valid survivor projection: "
                f"local={local_contract} remote={final_contract}"
            )

        abort_records = tuple(remote_control.get("aborts", ()))
        remote_abort_by_key = {
            (int(record.get("pp_batch_plan_id", -1)), record.get("rid")): record
            for record in abort_records
        }
        missing_known_aborts = tuple(
            record
            for record in expected_control.get("aborts", ())
            if remote_abort_by_key.get(
                (int(record.get("pp_batch_plan_id", -1)), record.get("rid"))
            )
            != record
        )
        if missing_known_aborts:
            raise RuntimeError(
                "PP output lost previously known abort records: "
                f"missing={missing_known_aborts} remote={abort_records}"
            )
        current_abort_records = tuple(
            record
            for record in abort_records
            if int(record.get("pp_batch_plan_id", -1))
            == int(expected_control["pp_batch_plan_id"])
        )
        abort_by_rid = {record["rid"]: record for record in current_abort_records}
        surviving_aborts = set(final_rids).intersection(abort_by_rid)
        if surviving_aborts:
            raise RuntimeError(
                "PP output kept requests marked aborted: "
                f"rids={tuple(sorted(surviving_aborts))}"
            )
        removed_rids = set(local_contract.get("rids", ())) - set(final_rids)
        if not removed_rids.issubset(abort_by_rid):
            raise RuntimeError(
                "PP output removed requests without abort evidence: "
                f"removed={tuple(sorted(removed_rids))} aborts={abort_records}"
            )
        if self.pp_group.is_first_rank and current_abort_records:
            self._pp_register_abort_tombstones(current_abort_records)
            logger.error(
                "PP request cancellation finalized: pp_batch_plan_id=%s mb_id=%s "
                "aborted_rids=%s surviving_rids=%s",
                expected_control["pp_batch_plan_id"],
                mb_id,
                tuple(record["rid"] for record in current_abort_records),
                final_rids,
            )

        if batch is None:
            if final_rids:
                raise RuntimeError(
                    "PP output revived requests after the local batch was cancelled: "
                    f"mb_id={mb_id} final={final_rids}"
                )
            return None

        had_requests = bool(batch.reqs)
        keep_indices = []
        for index, req in enumerate(batch.reqs):
            record = abort_by_rid.get(req.rid)
            if record is None:
                keep_indices.append(index)
                continue
            is_chunked = self.chunked_req is req or batch.chunked_req is req
            if is_chunked:
                # Defer KV release until its in-flight proxy send completes.
                self._pp_defer_chunked_cache_read_abort(req, record)
                if batch.chunked_req is req:
                    batch.chunked_req = None
            elif not req.finished():
                req._as_cache_read_abort_reason = record.get("reason")
                self._abort_failed_cache_read(req)

        if len(keep_indices) != len(batch.reqs):
            # filter_batch expects request rows, not prefill token rows.
            batch.input_ids = None
            batch.filter_batch(keep_indices=keep_indices)
            batch.prefix_lens = list(final_contract.get("prefix_lens", ()))
            batch.extend_lens = [
                end - start for start, end in final_contract.get("extend_ranges", ())
            ]
            batch.extend_num_tokens = int(final_contract.get("extend_num_tokens", 0))
        if had_requests and not batch.reqs:
            return None
        return batch

    def _pp_send_recv_and_preprocess_output_tensors(
        self: Scheduler,
        next_first_rank_mb_id: int,
        next_mb_id: int,
        mbs: List[ScheduleBatch],
        mb_metadata: List[PPBatchMetadata],
        last_rank_comm_queue: deque[Tuple[torch.Event, PPProxyTensors]],
        pp_outputs: PPProxyTensors | None,
    ) -> Tuple[
        Optional[PPProxyTensors],
        Optional[GenerationBatchResult],
        Optional[torch.Event],
        List[P2PWork],
    ]:
        next_pp_outputs = None
        d2h_event = None
        batch_result = None
        send_output_work = []

        # Keep all TP lanes on the same side of the PP communication boundary.
        # Otherwise one lane can enqueue the next PP recv while its peers enter
        # model TP collectives, creating a cyclic CUDA-stream dependency.
        if self.ps.attn_tp_size > 1:
            torch.distributed.barrier(group=self.attn_tp_cpu_group)

        # On CUDA, isend is async: it enqueues to the stream and returns,
        # so every rank can send first safely. On some backends isend is
        # effectively blocking and does not return until the peer posts a
        # matching recv; if every PP rank sends first, all ranks block
        # waiting for a receiver and the ring deadlocks. Order send/recv
        # by pp_rank parity (even: send->recv, odd: recv->send) so each
        # adjacent pair has one sender and one receiver posted at the
        # same time.

        # CUDA: send first
        # XPU: even ranks send first, odd ranks recv first.
        send_first = (not is_xpu()) or ((self.ps.pp_rank % 2) == 0)

        def _do_send():
            return self._pp_send_output_to_next_stage(
                next_first_rank_mb_id,
                mbs,
                last_rank_comm_queue,
                pp_outputs,
            )

        def _do_recv():
            nonlocal next_pp_outputs, batch_result, d2h_event
            target = mbs[next_mb_id]
            target_control = self._pp_mb_controls[next_mb_id]
            control_transport = bool(
                self._pp_batch_contract_guard_enabled
                and target_control is not None
                and target_control["transport"]
            )
            if (target is None and not control_transport) or (
                target is not None and target.forward_mode.is_prebuilt()
            ):
                return
            if (
                
                (
                    target is not None
                    and _pp_can_skip_output_comm(target)
                    and not _pp_control_has_current_aborts(target_control)
                ) or (
                    self._pp_pd_early_last_chunk_enabled
                    and not self.pp_group.is_first_rank
                )
            ):
                next_pp_outputs, batch_result, d2h_event = (
                    self._pp_make_skip_output_result(target, mb_metadata[next_mb_id])
                )
                return
            with torch.profiler.record_function("recv_res_dict_from_prev_stage"):
                next_pp_outputs = PPProxyTensors(self._pp_recv_dict_from_prev_stage())
            if self._pp_batch_contract_guard_enabled:
                target = self._pp_reconcile_output_aborts(
                    next_mb_id, target, next_pp_outputs
                )
                mbs[next_mb_id] = target
                if target is None:
                    return
            with self.copy_stream_ctx:
                self.copy_stream.wait_stream(self.schedule_stream)
                batch_result = self._pp_prep_batch_result(
                    target, mb_metadata[next_mb_id], next_pp_outputs
                )
                d2h_event = self.device_module.Event()
                d2h_event.record(self.device_module.current_stream())

        if send_first:
            send_output_work = _do_send()
            _do_recv()
        else:
            _do_recv()
            send_output_work = _do_send()

        return next_pp_outputs, batch_result, d2h_event, send_output_work

    def _pp_launch_batch(
        self: Scheduler,
        mb_id: int,
        cur_batch: ScheduleBatch,
        pp_proxy_tensors: PPProxyTensors,
        mb_metadata: List[Optional[PPBatchMetadata]],
        last_rank_comm_queue: deque,
    ):
        with torch.profiler.record_function("run_batch"):
            with self.forward_stream_ctx:
                self.forward_stream.wait_stream(self.schedule_stream)
                set_time_batch(
                    cur_batch.reqs,
                    "set_run_batch_cpu_start_time",
                    trace_only=True,
                )
                result = self.run_batch(cur_batch, pp_proxy_tensors)
                set_time_batch(
                    cur_batch.reqs,
                    "set_run_batch_cpu_end_time",
                    trace_only=True,
                    attrs={"pp_mb_id": mb_id},
                )
                event = self.device_module.Event()
                event.record(self.device_module.current_stream())
                mb_metadata[mb_id] = PPBatchMetadata(
                    can_run_cuda_graph=result.can_run_cuda_graph,
                    forward_done=event,
                )
                if self.pp_group.is_last_rank:
                    # (last rank) buffer the outputs for async batch depth
                    last_rank_comm_queue.append(
                        (
                            event,
                            PPProxyTensors(
                                self._pp_prepare_tensor_dict(result, cur_batch)
                            ),
                        )
                    )
        return result, event

    def get_rids(
        self: Scheduler, req_queue: List[Req], is_send: bool, *poll_statuses_group
    ):
        """
        Used by PP, get the required rids with the given poll statuses.
        """
        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender if is_send else req.kv_receiver for req in req_queue],
            self.attn_cp_cpu_group,
            self.attn_tp_cpu_group,
        )
        rids: List = []
        for poll_statuses in poll_statuses_group:
            rids.append(
                [
                    req.rid if is_send else req.req.rid
                    for req, poll in zip(req_queue, polls)
                    if poll in poll_statuses
                ]
            )
        return tuple(rids) if len(rids) > 1 else rids[0]

    def _pp_pd_get_retract_ids(self: Scheduler, mb_id: int):
        # communicate pre-consensus retracted reqs
        for req in self.disagg_decode_prealloc_queue.retracted_queue:
            # assign retracted reqs to the current microbatch
            if req.retraction_mb_id is None:
                req.retraction_mb_id = mb_id
        curr_retract_rids = [
            req.rid
            for req in self.disagg_decode_prealloc_queue.retracted_queue
            if req.retraction_mb_id == mb_id
        ]
        if self.pp_group.is_first_rank:
            # First rank, get all retracted req ids for the microbatch
            return curr_retract_rids
        else:
            # Other ranks, receive the retracted reqs info from the previous rank and ensure the consensus
            prev_retract_rids = self._pp_recv_pyobj_from_prev_stage()
            return list(set(prev_retract_rids) & set(curr_retract_rids))

    def _pp_pd_get_prealloc_ids(self: Scheduler):
        # communicate pre-consensus prealloc reqs
        if self.pp_group.is_first_rank:
            # First rank, pop the preallocated reqs from the prealloc queue
            good_prealloc_rids, bad_prealloc_rids = self.get_rids(
                self.disagg_decode_prealloc_queue.queue,
                False,
                [KVPoll.WaitingForInput],
                [KVPoll.Failed],
            )
        else:
            # Other ranks, receive the preallocated reqs info from the previous rank and ensure the consensus
            prev_prealloc_rids = self._pp_recv_pyobj_from_prev_stage()
            prev_good_prealloc_rids, prev_bad_prealloc_rids = prev_prealloc_rids
            curr_good_prealloc_rids, curr_bad_prealloc_rids = self.get_rids(
                self.disagg_decode_prealloc_queue.queue,
                False,
                [KVPoll.WaitingForInput],
                [KVPoll.Failed],
            )
            good_prealloc_rids = list(
                set(prev_good_prealloc_rids) & set(curr_good_prealloc_rids)
            )
            bad_prealloc_rids = list(
                set(prev_bad_prealloc_rids) | set(curr_bad_prealloc_rids)
            )
        # Same abort routing as the prefill bootstrap consensus above.
        aborted_rids = {
            decode_req.req.rid
            for decode_req in self.disagg_decode_prealloc_queue.queue
            if isinstance(decode_req.req.finished_reason, FINISH_ABORT)
        }
        good_prealloc_rids, bad_prealloc_rids = _pp_route_aborts_to_bad(
            good_prealloc_rids, bad_prealloc_rids, aborted_rids
        )
        return [good_prealloc_rids, bad_prealloc_rids]

    def _pp_pd_get_decode_transferred_ids(self: Scheduler):
        # get the current stage transfer success
        if self.pp_group.is_first_rank:
            transferred_rids = self.get_rids(
                self.disagg_decode_transfer_queue.queue,
                False,
                [KVPoll.Success, KVPoll.Failed],
            )
        # if other ranks, do intersection with the previous rank's transferred rids
        else:
            # 2 (Release): Receive the transferred rids from the previous rank
            # 1. recv previous stage's transferred reqs info
            prev_transferred_rids = self._pp_recv_pyobj_from_prev_stage()
            # 2. get the current stage's transferred reqs info
            curr_transferred_rids = self.get_rids(
                self.disagg_decode_transfer_queue.queue,
                False,
                [KVPoll.Success, KVPoll.Failed],
            )
            # 3. new consensus rids = intersection(previous consensus rids, transfer finished rids)
            transferred_rids = list(
                set(prev_transferred_rids) & set(curr_transferred_rids)
            )
        return transferred_rids

    def process_retract_queue(self: Scheduler, retract_rids: Optional[List[str]]):
        if retract_rids is not None:
            # try to resume retracted requests if there are enough space for another `num_reserved_decode_tokens` decode steps
            resumed_reqs = self.disagg_decode_prealloc_queue.resume_retracted_reqs(
                retract_rids
            )
            self.waiting_queue.extend(resumed_reqs)
            return [req.rid for req in resumed_reqs]
        return None

    def process_prealloc_queue(self: Scheduler, prealloc_rids: Optional[List[str]]):
        if len(self.disagg_decode_prealloc_queue.retracted_queue) > 0:
            # if there are still retracted requests, we do not allocate new requests
            return [[], []]

        if prealloc_rids is not None:
            (
                good_consensus_prealloc_rids,
                bad_consensus_prealloc_rids,
            ) = prealloc_rids
            good_reqs, failed_reqs = self.disagg_decode_prealloc_queue.pop_preallocated(
                pp_good_rids=good_consensus_prealloc_rids,
                pp_bad_rids=bad_consensus_prealloc_rids,
            )
            self.disagg_decode_transfer_queue.extend(good_reqs)
            return [
                [req.req.rid for req in good_reqs],
                [req.req.rid for req in failed_reqs],
            ]
        return None

    def process_decode_transfer_queue(
        self: Scheduler, release_rids: Optional[List[str]]
    ):
        if release_rids is not None:
            released_reqs = self.disagg_decode_transfer_queue.pop_transferred(
                release_rids
            )
            if self.enable_hisparse:
                for req in released_reqs:
                    self.hisparse_coordinator.admit_request_direct(req)
            self.waiting_queue.extend(released_reqs)
            return [req.rid for req in released_reqs]
        return None


class ChunkSizePredictor:
    """
    Predictor for dynamic chunk size based on quadratic latency model.

    Models latency as: f(l) = a*l^2 + b*l + c
    Predicts next chunk size x such that: f(L+x) - f(L) = target_latency
    """

    def __init__(self):
        self.quadratic_coeff_a = 0.0
        self.linear_coeff_b = 0.0
        self.constant_coeff_c = 0.0
        self.target_latency: Optional[float] = None
        self.is_ready = False

    def fit(self, seq_lens: List[int], latencies: List[float]):
        """Fit quadratic coefficients f(l) = al^2 + bl + c from data points."""
        # Skip the first data point to reduce fitting bias, as the first run is slower without warmup
        L = np.array(seq_lens[1:], dtype=np.float64)
        T = np.array(latencies[1:], dtype=np.float64)

        if len(L) < 8:
            raise ValueError(
                f"Not enough data points for quadratic fitting ({len(L)} < 8). "
                "Need at least 8 samples with different sequence lengths."
            )

        # Build design matrix for f(l) = al^2 + bl + c
        X = np.column_stack([L * L, L, np.ones_like(L)])  # [l^2, l, 1]

        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(X, T, rcond=None)
            if len(coeffs) >= 3:
                fitted_a = float(coeffs[0])  # quadratic coefficient
                fitted_b = float(coeffs[1])  # linear coefficient
                fitted_c = float(coeffs[2])  # constant coefficient
            else:
                raise ValueError("Failed to fit coefficients: insufficient rank")
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Failed to fit f(l) = al^2 + bl + c: {e}")

        # Validate coefficients
        if fitted_a <= 0:
            raise ValueError(
                f"Fitted quadratic coefficient a={fitted_a:.2e} is not positive. "
                "Attention has O(n^2) complexity, so a must be positive. "
                "Check warmup data quality."
            )

        if fitted_b < 0:
            logger.warning(
                f"Fitted linear coefficient b={fitted_b:.2e} is negative. Setting b=0."
            )
            fitted_b = 0.0

        self.quadratic_coeff_a = fitted_a
        self.linear_coeff_b = fitted_b
        self.constant_coeff_c = fitted_c

        logger.info(
            f"[ChunkSizePredictor] Fitted coefficients: a={fitted_a:.2e}, "
            f"b={fitted_b:.2e}, c={fitted_c:.2e}"
        )

    def set_target_latency(self, base_chunk_size: int):
        """Set target latency based on base chunk size: target = f(base_chunk_size) - f(0)."""

        def f(length: float) -> float:
            """Total latency function: f(length) = a*length^2 + b*length + c."""
            return (
                self.quadratic_coeff_a * length * length
                + self.linear_coeff_b * length
                + self.constant_coeff_c
            )

        self.target_latency = f(float(base_chunk_size)) - f(0.0)

        if self.target_latency <= 0:
            raise ValueError(
                f"Calculated target_latency={self.target_latency:.2f}ms is not positive. "
                "Check warmup data quality."
            )

        logger.info(
            f"[ChunkSizePredictor] Target latency: {self.target_latency:.2f}ms "
            f"(base_chunk_size={base_chunk_size})"
        )

    def predict_next_chunk_size(
        self,
        history_len: int,
        base_chunk_size: int,
        page_size: int,
        context_len: int,
        max_chunk_size: Optional[int] = None,
    ) -> Optional[int]:
        """
        Predict next chunk size x such that f(history_len + x) - f(history_len) = target_latency.

        Args:
            history_len: Current sequence length (L)
            base_chunk_size: Base chunk size
            page_size: Page size for alignment
            context_len: Maximum context length
            max_chunk_size: Maximum allowed chunk size (optional)

        Returns:
            Predicted chunk size, or None if prediction fails
        """
        if not self.is_ready or self.target_latency is None:
            return None

        # Handle quadratic model: f(l) = al^2 + bl + c
        if self.quadratic_coeff_a <= 0:
            return None

        # Solve f(L+x) - f(L) = T
        # where f(L) = a*L^2 + b*L + c
        # This expands to: ax^2 + (2aL+b)x - T = 0
        # A = a, B = 2aL + b, C = -T
        A = self.quadratic_coeff_a
        B = 2 * self.quadratic_coeff_a * history_len + self.linear_coeff_b
        C = -self.target_latency

        discriminant = B * B - 4 * A * C

        if discriminant < 0:
            logger.warning(
                f"Discriminant is negative ({discriminant:.2e}). "
                f"No real solution for chunk size. L={history_len}, T={self.target_latency:.2f}ms."
            )
            return None

        sqrt_discriminant = math.sqrt(discriminant)
        calculated_chunk_size_float = (-B + sqrt_discriminant) / (2 * A)

        if calculated_chunk_size_float <= 0:
            logger.warning(
                f"Calculated chunk size is non-positive ({calculated_chunk_size_float:.2f}). "
                f"L={history_len}, T={self.target_latency:.2f}ms."
            )
            return None

        # Use a smooth coefficient to reduce the abrupt decrease in chunk size
        smooth_coeff = envs.SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR.get()
        smoothed_chunk_size = base_chunk_size + smooth_coeff * (
            calculated_chunk_size_float - base_chunk_size
        )
        # Make sure the dynamic chunk size is at least 1/4 of the base chunk size
        calculated_chunk_size = max(int(smoothed_chunk_size), base_chunk_size // 4)

        # Align to page_size (minimum alignment size is 64)
        alignment_size = max(page_size, 64)
        dynamic_chunk_size = (calculated_chunk_size // alignment_size) * alignment_size

        # Ensure aligned size is at least alignment_size
        if dynamic_chunk_size < alignment_size:
            dynamic_chunk_size = alignment_size

        # Apply constraints
        max_allowed = context_len - history_len - 100  # Leave 100 tokens margin
        if max_chunk_size is not None:
            max_allowed = min(max_allowed, max_chunk_size)
        dynamic_chunk_size = min(dynamic_chunk_size, max_allowed)

        # Align again after min operation
        dynamic_chunk_size = (dynamic_chunk_size // alignment_size) * alignment_size

        if dynamic_chunk_size < alignment_size:
            return None

        return dynamic_chunk_size
