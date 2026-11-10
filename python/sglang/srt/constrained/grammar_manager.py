from __future__ import annotations

import logging
import time
from concurrent import futures
from threading import Event
from typing import TYPE_CHECKING, List

import torch

from sglang.srt.constrained.base_grammar_backend import (
    InvalidGrammarObject,
    create_grammar_backend,
)
from sglang.srt.constrained.reasoner_grammar_backend import ReasonerGrammarObject
from sglang.srt.distributed.communication_tags import P2PTag
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.managers.io_struct import AbortReq
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class GrammarManager:
    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler
        self.server_args = scheduler.server_args
        self.grammar_queue: List[Req] = []
        if not self.server_args.skip_tokenizer_init:
            self.grammar_backend = create_grammar_backend(
                self.server_args,
                scheduler.tokenizer,
                scheduler.model_config.vocab_size,
                scheduler.model_config.hf_eos_token_id,
                think_end_id=scheduler.model_config.think_end_id,
            )
        else:
            self.grammar_backend = None

        self._enable_strict_thinking = (
            self.grammar_backend.enable_strict_thinking
            if self.grammar_backend is not None
            else False
        )

        self.grammar_sync_group = scheduler.dp_tp_cpu_group
        self.grammar_sync_size = scheduler.dp_tp_group.world_size
        self.grammar_sync_entry = scheduler.dp_tp_group.first_rank
        self.is_grammar_sync_entry = scheduler.dp_tp_group.is_first_rank
        self.pp_rank = scheduler.ps.pp_rank
        self.pp_size = scheduler.ps.pp_size
        self.pp_group = scheduler.pp_group
        self.grammar_pp_sync_work_list = []

        self.SGLANG_GRAMMAR_POLL_INTERVAL = envs.SGLANG_GRAMMAR_POLL_INTERVAL.get()
        self.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS = (
            envs.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS.get()
        )
        self.max_grammar_replay_tokens = envs.SGLANG_MAX_GRAMMAR_REPLAY_TOKENS.get()
        self._replay_executor = futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="grammar-replay"
        )
        self._replay_cancel_events: dict[int, Event] = {}

    def _replay_continuation_tokens(
        self,
        rid: str,
        grammar,
        continuation_token_ids: tuple[int, ...],
        cancel_event: Event,
    ):
        """Advance grammar state to match continuation position."""
        if not continuation_token_ids or grammar is None:
            return grammar

        logger.debug(
            "Grammar replay start (rid=%s): grammar type=%s, "
            "continuation_token_ids length=%d",
            rid,
            type(grammar).__name__,
            len(continuation_token_ids),
        )

        phase = "initial replay"
        try:
            for index, token_id in enumerate(continuation_token_ids):
                if index % 256 == 0 and cancel_event.is_set():
                    logger.debug(
                        "Grammar replay cancelled (rid=%s, tokens=%d)", rid, index
                    )
                    return None
                grammar.accept_token(token_id)

            # Reset current_token so that process_prebuilt on the D node
            # will accept P's first generated token into the grammar FSM.
            phase = "current_token reset"
            if hasattr(grammar, "current_token"):
                grammar.current_token = None
            if isinstance(grammar, ReasonerGrammarObject) and hasattr(
                grammar.grammar, "current_token"
            ):
                grammar.grammar.current_token = None
            logger.debug(
                "Grammar replay completed (rid=%s, tokens=%d)",
                rid,
                len(continuation_token_ids),
            )
            return grammar
        except Exception as e:
            logger.warning(
                "Grammar continuation replay failed "
                "(rid=%s, phase=%s, tokens=%d): %s. "
                "Continuing without grammar constraint.",
                rid,
                phase,
                len(continuation_token_ids),
                e,
            )
            return None

    def _submit_continuation_replay(self, req: Req, grammar):
        continuation_token_ids = tuple(req.continuation_token_ids or ())
        cancel_event = Event()
        self._replay_cancel_events[id(req)] = cancel_event
        return self._replay_executor.submit(
            self._replay_continuation_tokens,
            req.rid,
            grammar,
            continuation_token_ids,
            cancel_event,
        )

    def _check_replay_limit(self, req: Req) -> bool:
        continuation_len = len(req.continuation_token_ids or ())
        if (
            continuation_len == 0
            or self.max_grammar_replay_tokens < 0
            or continuation_len <= self.max_grammar_replay_tokens
        ):
            return True

        error_msg = (
            "Grammar continuation replay exceeds the configured limit: "
            f"{continuation_len} > {self.max_grammar_replay_tokens} tokens. "
            "Set SGLANG_MAX_GRAMMAR_REPLAY_TOKENS to a larger value or a "
            "negative value to disable the limit."
        )
        logger.warning("Grammar replay rejected (rid=%s): %s", req.rid, error_msg)
        req.set_finish_with_abort(error_msg)
        return False

    def __len__(self):
        return len(self.grammar_queue)

    def clear(self):
        if self.grammar_backend:
            self.grammar_backend.reset()

    def has_waiting_grammars(self) -> bool:
        return len(self.grammar_queue) > 0

    def _drain_pp_sync_work(self):
        for p2p_work in self.grammar_pp_sync_work_list:
            p2p_work.work.wait()
        self.grammar_pp_sync_work_list.clear()

    def _pp_sync_ready_failed(
        self,
        ready_req_idxs: set[int],
        failed_req_idxs: set[int],
    ) -> tuple[set[int], set[int]]:
        """
        Synchronize ready/failed grammar request indexes across the PP pipeline.

        PP0 provides the data. Each later PP rank receives it from the previous
        rank and asynchronously forwards it to the next rank.
        """
        if self.pp_size <= 1 or self.pp_group is None:
            return ready_req_idxs, failed_req_idxs

        self._drain_pp_sync_work()
        data = (ready_req_idxs, failed_req_idxs)
        if self.pp_rank > 0:
            data = self.pp_group.recv_object(
                src=self.pp_rank - 1,
                tag=P2PTag.GRAMMAR_PP_SYNC,
            )
        if self.pp_rank + 1 < self.pp_size:
            self.grammar_pp_sync_work_list.extend(
                self.pp_group.send_object(
                    data,
                    dst=self.pp_rank + 1,
                    async_send=True,
                    tag=P2PTag.GRAMMAR_PP_SYNC,
                )
            )
        return data

    def abort_requests(self, recv_req: AbortReq):
        for req in self.grammar_queue:
            if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                logger.debug(f"Abort grammar queue request. {req.rid=}")
                cancel_event = self._replay_cancel_events.get(id(req))
                if cancel_event is not None:
                    cancel_event.set()
                if isinstance(req.grammar, futures.Future) and req.grammar:
                    req.grammar.cancel()
                req.set_finish_with_abort("Aborted by AbortReq.")

    def _get_request_thinking_budget(self, req: Req) -> int | None:
        custom_params = req.sampling_params.custom_params
        if not isinstance(custom_params, dict):
            return None
        thinking_budget = custom_params.get("thinking_budget")
        return thinking_budget if isinstance(thinking_budget, int) else None

    def _apply_request_reasoning_budget(self, req: Req) -> None:
        thinking_budget = self._get_request_thinking_budget(req)
        if thinking_budget is None:
            return
        if isinstance(req.grammar, ReasonerGrammarObject):
            req.grammar.max_think_tokens = thinking_budget

    def process_req_with_grammar(self, req: Req) -> bool:
        # Init grammar cache for this request
        add_to_grammar_queue = False
        if (
            req.sampling_params.json_schema is not None
            or req.sampling_params.regex is not None
            or req.sampling_params.ebnf is not None
            or req.sampling_params.structural_tag is not None
        ):
            if self.grammar_backend is None:
                error_msg = "Grammar-based generation (json_schema, regex, ebnf, structural_tag) is not supported when the server is launched with --grammar-backend none"
                req.set_finish_with_abort(error_msg)
            elif not self._check_replay_limit(req):
                pass
            else:
                if req.sampling_params.json_schema is not None:
                    key = ("json", req.sampling_params.json_schema)
                elif req.sampling_params.regex is not None:
                    key = ("regex", req.sampling_params.regex)
                elif req.sampling_params.ebnf is not None:
                    key = ("ebnf", req.sampling_params.ebnf)
                elif req.sampling_params.structural_tag:
                    key = ("structural_tag", req.sampling_params.structural_tag)

                value, cache_hit = self.grammar_backend.get_cached_or_future_value(
                    key, req.require_reasoning
                )
                req.grammar = value
                req.grammar_key = key

                if not cache_hit:
                    add_to_grammar_queue = True
                else:
                    if isinstance(
                        value, InvalidGrammarObject
                    ):  # We hit a cached invalid grammar.
                        error_msg = (
                            f"Failed to compile {key[0]} grammar: {value.error_message}"
                        )
                        req.set_finish_with_abort(error_msg)
                    else:
                        self._apply_request_reasoning_budget(req)
                        if req.continuation_token_ids:
                            req.grammar = self._submit_continuation_replay(req, value)
                            add_to_grammar_queue = True
        elif self._enable_strict_thinking:
            grammar_obj = self.grammar_backend.init_strict_reasoning_grammar(
                req.require_reasoning
            )
            if grammar_obj is not None:
                req.grammar = grammar_obj
                self._apply_request_reasoning_budget(req)
                if req.require_reasoning and req.continuation_token_ids:
                    if self._check_replay_limit(req):
                        req.grammar_key = ("reasoning_only", "")
                        req.grammar = self._submit_continuation_replay(req, grammar_obj)
                        add_to_grammar_queue = True

        if add_to_grammar_queue:
            self.grammar_queue.append(req)

        return add_to_grammar_queue

    def get_ready_grammar_requests(self) -> List[Req]:
        """
        Move requests whose grammar objects are ready from grammar_queue to waiting_queue.

        For PP0, DP/TP group rank i returns two sets ready_reqs_i,
        failed_reqs_i. ready_reqs_all = all_gather(ready_reqs_i) within
        PP0's DP/TP group. failed_reqs_all = all_gather(failed_reqs_i)
        within PP0's DP/TP group.

        ready_reqs = intersect(ready_reqs_all)
        failed_reqs = union(failed_reqs_all)

        PP0 then propagates the synced result to later PP ranks. Later PP
        ranks receive and apply the propagated ready/failed decision.
        """
        assert self.grammar_backend
        ready_req_idxs: set[int] = set()
        failed_req_idxs: set[int] = set()

        if self.pp_rank == 0:
            # Poll for ready requests
            start_time = time.perf_counter()
            while time.perf_counter() - start_time < self.SGLANG_GRAMMAR_POLL_INTERVAL:
                for i, req in enumerate(self.grammar_queue):
                    if i in ready_req_idxs:
                        continue

                    if (
                        req.finished() or req.grammar is None
                    ):  # It is aborted by AbortReq
                        ready_req_idxs.add(i)
                        continue

                    assert isinstance(req.grammar, futures.Future), f"{req=}"
                    if req.grammar.done():
                        ready_req_idxs.add(i)

                if len(ready_req_idxs) == len(self.grammar_queue):
                    break

                # Sleep a bit to avoid busy waiting
                time.sleep(self.SGLANG_GRAMMAR_POLL_INTERVAL / 10)

            # Check failed requests
            for i, req in enumerate(self.grammar_queue):
                if i not in ready_req_idxs:
                    # grammar_wait_ct is only updated on PP0; later PP ranks
                    # receive PP0's ready/failed decision through PP sync.
                    self.grammar_queue[i].grammar_wait_ct += 1
                    if (
                        self.grammar_queue[i].grammar_wait_ct
                        >= self.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS
                    ):
                        # Timeout after max poll iterations
                        # The actual waiting time is SGLANG_GRAMMAR_MAX_POLL_ITERATIONS * max(SGLANG_GRAMMAR_POLL_INTERVAL, GPU_forward_batch_latency)
                        failed_req_idxs.add(i)

            # Sync ready and failed requests across all TP ranks in PP0.
            if self.grammar_sync_size == 1:
                synced_ready_req_idxs = ready_req_idxs
                synced_failed_req_idxs = failed_req_idxs
            else:
                all_gather_output = [None] * self.grammar_sync_size
                torch.distributed.all_gather_object(
                    all_gather_output,
                    (ready_req_idxs, failed_req_idxs),
                    group=self.grammar_sync_group,
                )
                synced_ready_req_idxs = set.intersection(
                    *[x[0] for x in all_gather_output]
                )
                synced_failed_req_idxs = set.union(*[x[1] for x in all_gather_output])
        else:
            synced_ready_req_idxs = ready_req_idxs
            synced_failed_req_idxs = failed_req_idxs

        # Propagate PP0's grammar queue decision to later PP ranks.
        (
            synced_ready_req_idxs,
            synced_failed_req_idxs,
        ) = self._pp_sync_ready_failed(
            synced_ready_req_idxs,
            synced_failed_req_idxs,
        )

        # Return ready requests
        return_reqs: List[Req] = []
        finalized_req_idxs: set[int] = set()
        for i in synced_ready_req_idxs:
            req = self.grammar_queue[i]
            if req.finished() or req.grammar is None:  # It is aborted by AbortReq
                return_reqs.append(req)
                finalized_req_idxs.add(i)
                self._replay_cancel_events.pop(id(req), None)
                continue

            assert isinstance(req.grammar, futures.Future) and req.grammar_key
            is_replay = id(req) in self._replay_cancel_events
            try:
                grammar = req.grammar.result()
            except Exception as e:
                phase = "replay" if is_replay else "compilation"
                logger.error(
                    "Grammar %s raised an exception: %s, grammar_key=%s",
                    phase,
                    e,
                    req.grammar_key,
                )
                grammar = (
                    None
                    if is_replay
                    else InvalidGrammarObject(f"Grammar compilation failed: {e}")
                )

            if is_replay:
                req.grammar = grammar
                self._replay_cancel_events.pop(id(req), None)
                return_reqs.append(req)
                finalized_req_idxs.add(i)
                continue

            self.grammar_backend.set_cache(req.grammar_key, grammar.copy())
            req.grammar = grammar
            self._apply_request_reasoning_budget(req)
            if isinstance(grammar, InvalidGrammarObject):
                error_msg = (
                    f"Failed to compile {req.grammar_key[0]} grammar: "
                    f"{grammar.error_message}"
                )
                req.set_finish_with_abort(error_msg)
                return_reqs.append(req)
                finalized_req_idxs.add(i)
            else:
                if req.continuation_token_ids:
                    # Compile completed; replay asynchronously before admitting
                    # the request to the prefill queue.
                    req.grammar = self._submit_continuation_replay(req, grammar)
                    req.grammar_wait_ct = 0
                else:
                    req.grammar = grammar
                    return_reqs.append(req)
                    finalized_req_idxs.add(i)

        # Return failed requests
        for i in synced_failed_req_idxs:
            req = self.grammar_queue[i]
            return_reqs.append(req)
            finalized_req_idxs.add(i)

            assert isinstance(req.grammar, futures.Future) and req.grammar_key
            cancel_event = self._replay_cancel_events.pop(id(req), None)
            is_replay_timeout = cancel_event is not None
            if is_replay_timeout:
                cancel_event.set()
            req.grammar.cancel()
            if is_replay_timeout:
                error_msg = (
                    "Grammar continuation replay timed out: "
                    f"{req.grammar_key=}, tokens={len(req.continuation_token_ids or ())}"
                )
            else:
                self.grammar_backend.set_cache(
                    req.grammar_key,
                    InvalidGrammarObject("Grammar preprocessing timed out"),
                )
                error_msg = f"Grammar preprocessing timed out: {req.grammar_key=}"
            req.set_finish_with_abort(error_msg)

        # Remove finished requests from grammar_queue
        self.grammar_queue = [
            req
            for i, req in enumerate(self.grammar_queue)
            if i not in finalized_req_idxs and i not in synced_failed_req_idxs
        ]
        return return_reqs
