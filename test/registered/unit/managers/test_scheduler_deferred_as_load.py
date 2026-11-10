"""Deferred AS restores must finish before committing chunked admission."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.prefill_admission import PrefillAdmission

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _req(rid):
    req = MagicMock()
    req.rid = rid
    req.time_stats.forward_entry_time = 0
    req.inflight_middle_chunks = 0
    req.last_node = object()
    req.extend_range = SimpleNamespace(length=64)
    return req


class TestDeferredAsLoadAdmission(unittest.TestCase):
    PATHS = ("continuous", "ordinary")

    def _admit(
        self, path, reqs, *, candidate=None, outcomes=(), existing=None, as_enabled=True
    ):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.chunked_req = existing
        scheduler.waiting_queue = list(reqs)
        scheduler.enable_asradix_cache = as_enabled
        scheduler.enable_hierarchical_cache = False
        scheduler.enable_priority_preemption = False
        scheduler.enable_priority_scheduling = False
        scheduler.enable_hicache_storage = False
        scheduler.enable_dynamic_chunking = False
        scheduler.enable_lora = False
        scheduler.is_hybrid_swa = False
        scheduler.is_mixed_chunk = False
        scheduler.dllm_config = None
        scheduler.min_free_slots_delayer = None
        scheduler.server_args = SimpleNamespace(enable_flexkv=False, prefill_max_requests=16)
        scheduler.disaggregation_mode = None
        scheduler.grammar_manager = MagicMock()
        scheduler.grammar_manager.has_waiting_grammars.return_value = False
        scheduler.policy = MagicMock()
        scheduler.tree_cache = MagicMock()
        scheduler.tree_cache.deferred_as_load_conflict.return_value = False
        scheduler.tree_cache.flush_as_load_handoffs.return_value = list(outcomes)
        scheduler.req_to_token_pool = SimpleNamespace()
        scheduler.token_to_kv_pool_allocator = MagicMock()
        scheduler.get_num_allocatable_reqs = MagicMock(return_value=16)
        scheduler.page_size = 64
        scheduler.new_token_ratio_tracker = SimpleNamespace(current=1.0)
        scheduler.max_prefill_tokens = 4096
        scheduler.chunked_prefill_size = 64
        scheduler.priority_scheduling_preemption_threshold = 0
        scheduler.max_prefill_bs = 0
        scheduler.max_running_requests = 16
        scheduler.truncation_align_size = None
        scheduler.metrics_reporter = SimpleNamespace(enable_metrics=True)
        scheduler.metrics_collector = MagicMock()
        scheduler.model_config = None
        scheduler.enable_overlap = True
        scheduler.spec_algorithm = None
        scheduler.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(prefill_aware_swa=False)
        )
        scheduler.load_inquirer = MagicMock()

        running = MagicMock()
        running.reqs = []
        running.batch_is_full = False
        running.is_empty.return_value = True
        adder = MagicMock()
        adder.can_run_list = [existing] if existing is not None else []
        adder.preempt_list = []
        adder.new_chunked_req = None
        adder.add_chunked_req.return_value = existing

        def add_one_req(req, **kwargs):
            adder.can_run_list.append(req)
            if req is reqs[-1]:
                adder.new_chunked_req = candidate
                return scheduler_module.AddReqResult.OTHER
            return scheduler_module.AddReqResult.CONTINUE

        adder.add_one_req.side_effect = add_one_req
        if path == "continuous":
            for req in reqs:
                add_one_req(req)
        batch = MagicMock()
        initial_chunks = existing.inflight_middle_chunks if existing is not None else 0

        def build_batch(can_run_list, *args, chunked_req):
            self.assertIs(chunked_req, scheduler.chunked_req)
            self.assertEqual(
                scheduler.chunked_req.inflight_middle_chunks
                if scheduler.chunked_req is not None
                else 0,
                initial_chunks + 1 if scheduler.chunked_req is not None else 0,
            )
            batch.reqs = list(can_run_list)
            return batch

        with (
            patch.dict("os.environ", {"AS_DIAG_EARLY_WRITE_RECLAIM": "0"}),
            patch.object(scheduler_module, "PrefillAdder", return_value=adder),
            patch.object(
                scheduler_module.ScheduleBatch, "init_new", side_effect=build_batch
            ) as init_batch,
            patch.object(scheduler_module.PrefillStats, "from_adder"),
        ):
            if path == "continuous":
                result, returned_running = scheduler._seal_prefill_admission(
                    PrefillAdmission(running_batch=running, adder=adder)
                )
            else:
                result, returned_running = scheduler._get_new_batch_prefill_raw(
                    None, running
                )
        self.assertIs(returned_running, running)
        return scheduler, result, init_batch

    def test_failed_first_chunk_is_requeued_without_forward_state(self):
        for path in self.PATHS:
            with self.subTest(path=path):
                req = _req("failed-first-chunk")
                outcome = SimpleNamespace(req=req, old_node=req.last_node, ok=False)
                scheduler, batch, init_batch = self._admit(
                    path, [req], candidate=req, outcomes=[outcome]
                )

                self.assertIsNone(batch)
                self.assertIsNone(scheduler.chunked_req)
                self.assertEqual(scheduler.waiting_queue, [req])
                self.assertEqual(req.inflight_middle_chunks, 0)
                req.time_stats.set_forward_entry_time.assert_not_called()
                wait_metric = scheduler.metrics_collector.observe_request_waiting_time
                wait_metric.assert_not_called()
                scheduler.tree_cache.dec_lock_ref.assert_called_once_with(
                    outcome.old_node
                )
                init_batch.assert_not_called()

    def test_failed_chunk_candidate_does_not_block_successful_peer(self):
        for path in self.PATHS:
            with self.subTest(path=path):
                peer, failed = _req("peer"), _req("failed-first-chunk")
                peer_new_node = object()
                outcomes = [
                    SimpleNamespace(
                        req=peer,
                        old_node=peer.last_node,
                        new_node=peer_new_node,
                        ok=True,
                    ),
                    SimpleNamespace(req=failed, old_node=failed.last_node, ok=False),
                ]
                scheduler, batch, _ = self._admit(
                    path, [peer, failed], candidate=failed, outcomes=outcomes
                )

                self.assertEqual(batch.reqs, [peer])
                self.assertIsNone(scheduler.chunked_req)
                self.assertEqual(scheduler.waiting_queue, [failed])
                self.assertIs(peer.last_node, peer_new_node)
                scheduler.tree_cache.inc_lock_ref.assert_called_once_with(peer_new_node)
                self.assertEqual(scheduler.tree_cache.dec_lock_ref.call_count, 2)
                peer.time_stats.set_forward_entry_time.assert_called_once()
                failed.time_stats.set_forward_entry_time.assert_not_called()
                self.assertEqual(failed.inflight_middle_chunks, 0)
                wait_metric = scheduler.metrics_collector.observe_request_waiting_time
                wait_metric.assert_called_once()

    def test_successful_chunk_candidate_survives_failed_peer(self):
        for path in self.PATHS:
            with self.subTest(path=path):
                failed, candidate = _req("failed-peer"), _req("first-chunk")
                outcomes = [
                    SimpleNamespace(req=failed, old_node=failed.last_node, ok=False),
                    SimpleNamespace(
                        req=candidate,
                        old_node=candidate.last_node,
                        new_node=candidate.last_node,
                        ok=True,
                    ),
                ]
                scheduler, batch, _ = self._admit(
                    path, [failed, candidate], candidate=candidate, outcomes=outcomes
                )

                self.assertEqual(batch.reqs, [candidate])
                self.assertIs(scheduler.chunked_req, candidate)
                self.assertEqual(candidate.inflight_middle_chunks, 1)
                self.assertEqual(scheduler.waiting_queue, [failed])
                failed.time_stats.set_forward_entry_time.assert_not_called()

    def test_successful_first_chunk_is_marked_before_batch_preparation(self):
        for path in self.PATHS:
            for as_enabled in (False, True):
                with self.subTest(path=path, as_enabled=as_enabled):
                    req = _req("first-chunk")
                    outcome = SimpleNamespace(
                        req=req, old_node=req.last_node, new_node=req.last_node, ok=True
                    )
                    scheduler, batch, _ = self._admit(
                        path,
                        [req],
                        candidate=req,
                        outcomes=[outcome],
                        as_enabled=as_enabled,
                    )

                    self.assertEqual(batch.reqs, [req])
                    self.assertIs(scheduler.chunked_req, req)
                    self.assertEqual(req.inflight_middle_chunks, 1)
                    self.assertEqual(scheduler.waiting_queue, [])
                    req.time_stats.set_forward_entry_time.assert_called_once()
                    batch.prepare_for_extend.assert_called_once()

    def test_existing_chunk_continues_when_new_request_load_fails(self):
        for path in self.PATHS:
            with self.subTest(path=path):
                existing, failed = _req("continuation"), _req("failed-peer")
                existing.inflight_middle_chunks = 2
                outcome = SimpleNamespace(
                    req=failed, old_node=failed.last_node, ok=False
                )
                scheduler, batch, _ = self._admit(
                    path, [failed], existing=existing, outcomes=[outcome]
                )

                self.assertEqual(batch.reqs, [existing])
                self.assertIs(scheduler.chunked_req, existing)
                self.assertEqual(existing.inflight_middle_chunks, 3)
                self.assertEqual(scheduler.waiting_queue, [failed])
                existing.time_stats.set_forward_entry_time.assert_called_once()
                failed.time_stats.set_forward_entry_time.assert_not_called()


if __name__ == "__main__":
    unittest.main()
