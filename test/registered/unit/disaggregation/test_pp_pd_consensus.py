import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.base import KVPoll  # noqa: E402
from sglang.srt.disaggregation.common.conn import CommonKVManager  # noqa: E402
from sglang.srt.disaggregation.decode import DecodePreallocQueue  # noqa: E402
from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue  # noqa: E402
from sglang.srt.disaggregation.utils import (  # noqa: E402
    _DRAFT_KV_LAYER_ID_BASE,
    build_transfer_entry_pairs,
    poll_and_all_reduce_pp,
    resolve_dcp_dst_entry_indices,
)
from sglang.srt.managers.schedule_batch import FINISH_ABORT  # noqa: E402
from sglang.srt.managers.scheduler_pp_mixin import (  # noqa: E402
    SchedulerPPMixin,
    _pp_merge_transfer_status,
    _pp_route_aborts_to_bad,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPPPDConsensus(CustomTestCase):
    @staticmethod
    def _make_prefill_queue(pp_rank):
        class FakeKVArgs:
            pass

        class FakeManager:
            def __init__(self, kv_args, *_args):
                self.kv_args = kv_args

        class FakePool:
            start_layer = 4
            end_layer = 6
            head_num = 1
            page_size = 64

            @staticmethod
            def get_contiguous_buf_infos():
                return [10, 11], [100, 100], [10, 10]

        class FakeDraftPool:
            start_layer = 0
            end_layer = 1

            @staticmethod
            def get_contiguous_buf_infos():
                return [20], [100], [10]

        queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
        queue.transfer_backend = "mooncake"
        queue.tp_rank = 0
        queue.pp_rank = pp_rank
        queue.pp_size = 2
        queue.scheduler = SimpleNamespace(
            ps=SimpleNamespace(dp_rank=0, gpu_id=0),
            server_args=SimpleNamespace(disaggregation_ib_device=None),
            model_config=SimpleNamespace(
                num_hidden_layers=8,
                get_total_num_kv_heads=lambda: 1,
            ),
            req_to_token_pool=None,
        )
        queue.token_to_kv_pool = FakePool()
        queue.draft_token_to_kv_pool = FakeDraftPool()
        queue.metadata_buffers = SimpleNamespace(get_buf_infos=lambda: ([], [], []))
        queue.is_mla_backend = False
        return queue, FakeKVArgs, FakeManager

    def test_transfer_failure_overrides_ordered_success_intersection(self):
        """A failure on one PP rank must terminate an otherwise successful rid."""
        status = _pp_merge_transfer_status(
            previous=(["req-a", "req-b", "req-c"], ["req-x"]),
            current=(["req-c", "req-a", "req-b"], ["req-b", "req-y"]),
        )

        self.assertEqual(
            status,
            (["req-a", "req-c"], ["req-x", "req-b", "req-y"]),
        )

    def test_bootstrap_probe_respects_local_metadata_credit_prefix(self):
        """A slower PP rank must not advertise requests it cannot admit."""
        queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
        queue.queue = [
            SimpleNamespace(
                rid="req-failed",
                metadata_buffer_index=-1,
                disagg_kv_sender=object(),
            ),
            SimpleNamespace(
                rid="req-ready",
                metadata_buffer_index=-1,
                disagg_kv_sender=object(),
            ),
            SimpleNamespace(
                rid="req-blocked",
                metadata_buffer_index=-1,
                disagg_kv_sender=object(),
            ),
        ]
        queue.scheduler = SimpleNamespace(
            attn_cp_cpu_group=object(),
            attn_tp_cpu_group=object(),
        )
        queue.req_to_metadata_buffer_idx_allocator = SimpleNamespace(
            available_size=lambda: 1
        )

        with patch(
            "sglang.srt.disaggregation.prefill." "poll_and_all_reduce_attn_cp_tp_group",
            return_value=[
                KVPoll.Failed,
                KVPoll.WaitingForInput,
                KVPoll.WaitingForInput,
            ],
        ):
            good_rids, failed_rids = queue.get_ready_bootstrapped_rids_for_pp()

        self.assertEqual(good_rids, ["req-ready"])
        self.assertEqual(failed_rids, ["req-failed"])
        self.assertEqual(
            [req.metadata_buffer_index for req in queue.queue],
            [-1, -1, -1],
        )

    def test_only_last_pp_registers_draft_kv_for_transfer(self):
        """Draft KV has one prefill owner even though every PP rank has a worker."""
        layer_ids_by_rank = []
        for pp_rank in (0, 1):
            queue, fake_args, fake_manager = self._make_prefill_queue(pp_rank)

            def get_kv_class(_backend, class_type):
                return fake_args if class_type.value == "kvargs" else fake_manager

            with (
                patch(
                    "sglang.srt.disaggregation.prefill.get_kv_class",
                    side_effect=get_kv_class,
                ),
                patch(
                    "sglang.srt.disaggregation.prefill.setup_state_kv_args",
                ),
            ):
                manager = queue._init_kv_manager()
            layer_ids_by_rank.append(manager.kv_args.kv_layer_ids)

        self.assertEqual(layer_ids_by_rank[0], [4, 5])
        self.assertEqual(
            layer_ids_by_rank[1],
            [4, 5, _DRAFT_KV_LAYER_ID_BASE],
        )

    def test_pp_prefill_entries_pair_with_pp1_decode_entries(self):
        """Each PP source maps target layers plus the unique draft entry by id."""
        decode_layer_ids = [
            0,
            1,
            2,
            3,
            4,
            5,
            _DRAFT_KV_LAYER_ID_BASE,
        ]

        rank_0_pairs = build_transfer_entry_pairs(
            src_layer_ids=[0, 1, 2, 3],
            dst_layer_ids=decode_layer_ids,
            n_src=4,
            n_dst=7,
        )
        rank_1_pairs = build_transfer_entry_pairs(
            src_layer_ids=[4, 5, _DRAFT_KV_LAYER_ID_BASE],
            dst_layer_ids=decode_layer_ids,
            n_src=3,
            n_dst=7,
        )

        self.assertEqual(rank_0_pairs, [(0, 0), (1, 1), (2, 2), (3, 3)])
        self.assertEqual(rank_1_pairs, [(0, 4), (1, 5), (2, 6)])

    def test_pp_prefill_dcp_registration_accepts_pp1_decode_boundary(self):
        """A PP prefill stage owns only part of the model's MLA KV entries.

        Regression: the PD DCP relayout path required the prefill and decode
        MLA/draft entry boundaries to be equal. Under PP the prefill boundary
        is per-stage (1-3 entries) while a pp_size=1 decode peer reports the
        whole model's, so registration raised inside the prefill bootstrap
        thread, killing it and stalling every subsequent KV transfer.
        """
        decode_mla_entries = 24
        dst_page_item_lens = [640] * decode_mla_entries + [320]
        dst_layer_ids = list(range(decode_mla_entries)) + [_DRAFT_KV_LAYER_ID_BASE]
        # Last PP stage: two own MLA layers plus the single draft entry.
        manager = SimpleNamespace(
            kv_args=SimpleNamespace(
                page_size=64,
                kv_item_lens=[640, 640, 640],
                kv_layer_ids=[4, 5, _DRAFT_KV_LAYER_ID_BASE],
                num_mla_kv_entries=2,
            )
        )

        src_boundary, dst_boundary = CommonKVManager._resolve_mla_entry_boundaries(
            manager, dst_num_mla_kv_entries=decode_mla_entries
        )
        self.assertEqual((src_boundary, dst_boundary), (2, decode_mla_entries))

        paired_dst_page_item_lens = [
            dst_page_item_lens[j]
            for j in resolve_dcp_dst_entry_indices(
                manager.kv_args.kv_layer_ids,
                dst_layer_ids,
                len(manager.kv_args.kv_item_lens),
                len(dst_page_item_lens),
            )
        ]
        self.assertEqual(
            CommonKVManager.prepare_dcp_token_item_lens(
                manager,
                paired_dst_page_item_lens,
                num_mla_kv_entries=src_boundary,
            ),
            [10, 10, 10],
        )

        # A decode peer reporting fewer MLA entries than the prefill stage owns
        # cannot be paired, so the sanity check must still reject it.
        with self.assertRaises(RuntimeError):
            CommonKVManager._resolve_mla_entry_boundaries(
                manager, dst_num_mla_kv_entries=1
            )


class TestPPBootstrapConsensusAuthority(CustomTestCase):
    """PP consensus, not the local poll, decides who is admitted.

    Regression: `pop_bootstrapped` used to receive one flat
    `rids_to_check = good + bad` list and re-poll locally. A rid the ring had
    already agreed to fail was still "to check" on a rank whose own sender
    lagged at `WaitingForInput`, so that rank finalized and admitted it while
    every other rank drained it -- the microbatch rosters diverged.
    """

    @staticmethod
    def _make_queue(reqs):
        queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
        queue.pp_size = 2
        queue.queue = list(reqs)
        queue.req_to_metadata_buffer_idx_allocator = SimpleNamespace(
            available_size=lambda: 8,
            alloc=lambda: 3,
        )
        queue.scheduler = SimpleNamespace(
            attn_cp_cpu_group=object(),
            attn_tp_cpu_group=object(),
            server_args=SimpleNamespace(optimistic_prefill_attempts=0),
            token_to_kv_pool_allocator=SimpleNamespace(page_size=64),
            handle_bootstrap_failure=MagicMock(),
        )
        return queue

    @staticmethod
    def _make_req(rid):
        return SimpleNamespace(
            rid=rid,
            disagg_kv_sender=MagicMock(**{"pop_decode_prefix_len.return_value": 0}),
            metadata_buffer_index=-1,
            pending_bootstrap=True,
            prefill_attempt_count=0,
            is_retracted=False,
            origin_input_ids=[1, 2, 3],
            start_send_idx=0,
            time_stats=MagicMock(),
        )

    def test_consensus_failure_wins_over_healthy_local_poll(self):
        """A rid the ring failed must not be admitted by a lagging rank."""
        req = self._make_req("req-bad")
        queue = self._make_queue([req])

        with patch(
            "sglang.srt.disaggregation.prefill." "poll_and_all_reduce_attn_cp_tp_group",
            return_value=[KVPoll.WaitingForInput],
        ) as local_poll:
            good, failed = queue.pop_bootstrapped(
                return_failed_reqs=True,
                pp_good_rids=[],
                pp_bad_rids=["req-bad"],
            )

        # The consensus decision is authoritative, so no local poll is needed.
        local_poll.assert_not_called()
        self.assertEqual(good, [])
        self.assertEqual([r.rid for r in failed], ["req-bad"])
        req.disagg_kv_sender.init.assert_not_called()
        self.assertTrue(req.pending_bootstrap)
        queue.scheduler.handle_bootstrap_failure.assert_called_once_with(req)

    def test_unconsensused_rid_stays_queued_instead_of_being_admitted(self):
        """A rid in neither consensus list is undecided, not implicitly good."""
        req = self._make_req("req-pending")
        queue = self._make_queue([req])

        with patch(
            "sglang.srt.disaggregation.prefill." "poll_and_all_reduce_attn_cp_tp_group",
            return_value=[KVPoll.WaitingForInput],
        ):
            good, failed = queue.pop_bootstrapped(
                return_failed_reqs=True,
                pp_good_rids=[],
                pp_bad_rids=[],
            )

        self.assertEqual((good, failed), ([], []))
        self.assertEqual([r.rid for r in queue.queue], ["req-pending"])
        req.disagg_kv_sender.init.assert_not_called()

    def test_local_failure_of_unconsensused_rid_is_still_drained(self):
        """An undecided rid whose own sender died must not linger forever.

        Only the ranks that observe the local failure drain it here; the rid
        never entered a plan, so no roster can reference it.
        """
        req = self._make_req("req-dead")
        queue = self._make_queue([req])

        with patch(
            "sglang.srt.disaggregation.prefill." "poll_and_all_reduce_attn_cp_tp_group",
            return_value=[KVPoll.Failed],
        ):
            good, failed = queue.pop_bootstrapped(
                return_failed_reqs=True,
                pp_good_rids=[],
                pp_bad_rids=[],
            )

        self.assertEqual(good, [])
        self.assertEqual([r.rid for r in failed], ["req-dead"])
        self.assertEqual(queue.queue, [])

    def test_consensus_good_rid_is_finalized_without_repolling(self):
        """A rid the ring admitted is initialized from the decision alone."""
        req = self._make_req("req-good")
        queue = self._make_queue([req])

        with patch(
            "sglang.srt.disaggregation.prefill." "poll_and_all_reduce_attn_cp_tp_group",
            return_value=[KVPoll.Bootstrapping],
        ) as local_poll:
            good, failed = queue.pop_bootstrapped(
                return_failed_reqs=True,
                pp_good_rids=["req-good"],
                pp_bad_rids=[],
            )

        local_poll.assert_not_called()
        self.assertEqual([r.rid for r in good], ["req-good"])
        self.assertEqual(failed, [])
        req.disagg_kv_sender.init.assert_called_once_with(1, 3)
        self.assertFalse(req.pending_bootstrap)

    def test_poll_and_all_reduce_pp_requires_both_consensus_lists(self):
        """A missing decision must fail loudly, never degrade to "all good"."""
        with self.assertRaises(ValueError):
            poll_and_all_reduce_pp(["a"], KVPoll.WaitingForInput, ["a"], None)
        with self.assertRaises(ValueError):
            poll_and_all_reduce_pp(["a"], KVPoll.WaitingForInput, None, [])

    def test_bad_list_overrides_good_list_for_the_same_rid(self):
        """Ranks disagreeing about a rid resolve to Failed, not to ready."""
        self.assertEqual(
            poll_and_all_reduce_pp(
                ["a", "b"], KVPoll.WaitingForInput, ["a", "b"], ["b"]
            ),
            [KVPoll.WaitingForInput, KVPoll.Failed],
        )


class TestPPAbortRoutedThroughConsensus(CustomTestCase):
    """A local abort must reach the ring even if the sender's poll never fails.

    Regression: `disagg_kv_sender.abort()` only records a failure reason and
    flips `conclude_state`; it does not guarantee the next poll returns
    `Failed`. An aborted request could therefore stay in the good intersection
    on the rank that had not yet received the AbortReq, while the rank that had
    already dropped it voted it away -- the rosters diverged again.
    """

    def test_aborted_bootstrap_req_is_forced_into_the_bad_union(self):
        aborted = SimpleNamespace(
            rid="req-aborted", finished_reason=FINISH_ABORT("Aborted by AbortReq.")
        )
        healthy = SimpleNamespace(rid="req-ok", finished_reason=None)
        scheduler = SimpleNamespace(
            pp_group=SimpleNamespace(is_first_rank=True),
            disagg_prefill_bootstrap_queue=SimpleNamespace(
                queue=[aborted, healthy],
                get_ready_bootstrapped_rids_for_pp=lambda: (
                    ["req-aborted", "req-ok"],
                    [],
                ),
            ),
        )

        good, bad = SchedulerPPMixin._pp_pd_get_bootstrapped_ids(scheduler)

        self.assertEqual(good, ["req-ok"])
        self.assertEqual(bad, ["req-aborted"])

    def test_aborted_prealloc_req_is_forced_into_the_bad_union(self):
        aborted = SimpleNamespace(
            req=SimpleNamespace(
                rid="req-aborted", finished_reason=FINISH_ABORT("Aborted by AbortReq.")
            )
        )
        scheduler = SimpleNamespace(
            pp_group=SimpleNamespace(is_first_rank=True),
            disagg_decode_prealloc_queue=SimpleNamespace(queue=[aborted]),
            get_rids=lambda *_args: (["req-aborted"], []),
        )

        good, bad = SchedulerPPMixin._pp_pd_get_prealloc_ids(scheduler)

        self.assertEqual(good, [])
        self.assertEqual(bad, ["req-aborted"])

    def test_abort_routing_appends_newcomers_in_a_rank_stable_order(self):
        """Every rank must build the same list from the same set.

        Ranks reach `_pp_route_aborts_to_bad` with the aborted rids in
        iteration order of their own bootstrap queue, so an unsorted append
        would make otherwise-identical decisions differ between ranks and
        break the byte-for-byte comparison of relayed payloads.
        """
        rank_a = _pp_route_aborts_to_bad(["g"], ["x"], {"b", "a", "c"})
        rank_b = _pp_route_aborts_to_bad(["g"], ["x"], {"c", "a", "b"})

        self.assertEqual(rank_a, rank_b)
        self.assertEqual(rank_a, (["g"], ["x", "a", "b", "c"]))

    def test_abort_routing_does_not_duplicate_an_already_bad_rid(self):
        self.assertEqual(
            _pp_route_aborts_to_bad([], ["dup"], {"dup"}),
            ([], ["dup"]),
        )


class TestPPPreallocConsensusContract(CustomTestCase):
    def test_pp_mode_refuses_to_run_without_a_consensus_decision(self):
        """Silently polling locally under PP is what caused the divergence."""
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 2

        with self.assertRaises(ValueError):
            queue.pop_preallocated()
        with self.assertRaises(ValueError):
            queue.pop_preallocated(pp_good_rids=[], pp_bad_rids=None)
        # The flat single-list protocol is gone; passing it must not be ignored.
        with self.assertRaises(ValueError):
            queue.pop_preallocated(rids_to_check=["a"], pp_good_rids=[], pp_bad_rids=[])


if __name__ == "__main__":
    unittest.main()
