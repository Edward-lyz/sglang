import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from sglang.srt.disaggregation.embedding_transfer.queues import MMTransferQueue
from sglang.srt.disaggregation.embedding_transfer.request import (
    metadata_only_recv_req,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputFormat,
    MultimodalInputs,
    MultimodalProcessorOutput,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _mm_item(start: int, end: int, hash_value: int) -> MultimodalDataItem:
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        offsets=[(start, end)],
        format=MultimodalInputFormat.PRECOMPUTED_EMBEDDING,
        precomputed_embeddings=torch.zeros(end - start + 1, 4),
    )
    item.hash = hash_value
    return item


def _recv_req(items=None, pool_buffer_addrs=None) -> TokenizedGenerateReqInput:
    items = items if items is not None else [_mm_item(1, 3, 111), _mm_item(5, 6, 222)]
    mm_inputs = MultimodalProcessorOutput(
        mm_items=items,
        input_ids=[0, 9, 9, 9, 4, 9, 9],
        im_token_id=9,
        pool_buffer_addrs=pool_buffer_addrs,
    )
    return TokenizedGenerateReqInput(
        rid="rid-0",
        input_text="",
        input_ids=[0, 9, 9, 9, 4, 9, 9],
        input_embeds=None,
        mm_inputs=mm_inputs,
        token_type_ids=None,
        sampling_params=SamplingParams(),
        return_logprob=False,
        logprob_start_len=0,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
        need_wait_for_mm_inputs=True,
        task_results=[{"task_id": "t0"}],
        mm_data_mooncake=["http://example/0.jpg"],
        encoder_urls=["http://encoder"],
    )


class TestMetadataOnlyRecvReq(CustomTestCase):
    def test_drops_embedding_and_pool_addrs(self):
        recv_req = _recv_req(pool_buffer_addrs=[4096, 8192])

        out = metadata_only_recv_req(recv_req)

        self.assertIsNone(out.mm_inputs.pool_buffer_addrs)
        for item, source in zip(out.mm_inputs.mm_items, recv_req.mm_inputs.mm_items):
            self.assertIsNone(item.precomputed_embeddings)
            self.assertIsNone(item.feature)
            self.assertEqual(item.offsets, source.offsets)
            self.assertEqual(item.hash, source.hash)
            self.assertEqual(item.modality, source.modality)
            self.assertTrue(item.is_precomputed_embedding())
        self.assertEqual(out.input_ids, recv_req.input_ids)
        self.assertEqual(out.mm_inputs.im_token_id, 9)

    def test_does_not_mutate_source(self):
        """mm_items is aliased, so nulling fields in place would blank stage 0."""
        recv_req = _recv_req(pool_buffer_addrs=[4096])
        mm_inputs = MultimodalInputs.from_processor_output(recv_req.mm_inputs)

        metadata_only_recv_req(recv_req)

        for item in recv_req.mm_inputs.mm_items:
            self.assertIsNotNone(item.precomputed_embeddings)
        for item in mm_inputs.mm_items:
            self.assertIsNotNone(item.precomputed_embeddings)
        self.assertEqual(recv_req.mm_inputs.pool_buffer_addrs, [4096])

    def test_preserves_mm_token_counts(self):
        recv_req = _recv_req()
        expected = MultimodalInputs.from_processor_output(
            recv_req.mm_inputs
        ).compute_mm_token_counts()

        out = metadata_only_recv_req(recv_req)

        actual = MultimodalInputs.from_processor_output(
            out.mm_inputs
        ).compute_mm_token_counts()
        self.assertEqual(actual, expected)
        self.assertEqual(actual[0], 5)

    def test_preserves_pad_value(self):
        """Downstream stages re-derive pad_value from the relayed hash."""
        recv_req = _recv_req()

        out = metadata_only_recv_req(recv_req)

        expected = MultimodalInputs.from_processor_output(recv_req.mm_inputs)
        actual = MultimodalInputs.from_processor_output(out.mm_inputs)
        self.assertEqual(
            [item.pad_value for item in actual.mm_items],
            [item.pad_value for item in expected.mm_items],
        )

    def test_rejects_item_without_hash(self):
        item = _mm_item(1, 3, 111)
        item.hash = None

        with self.assertRaisesRegex(ValueError, "no hash"):
            metadata_only_recv_req(_recv_req(items=[item]))

    def test_rejects_device_tensor_in_model_specific_data(self):
        item = _mm_item(1, 3, 111)
        item.set("extra_state", torch.zeros(2))

        with self.assertRaisesRegex(ValueError, "model_specific_data"):
            metadata_only_recv_req(_recv_req(items=[item]))

    def test_marks_request_already_admitted(self):
        out = metadata_only_recv_req(_recv_req())

        self.assertFalse(out.need_wait_for_mm_inputs)
        self.assertIsNone(out.task_results)
        self.assertIsNone(out.mm_data_mooncake)
        self.assertIsNone(out.num_items_assigned)
        self.assertIsNone(out.encoder_urls)


def _scheduler(mm_ready=(), mm_abort=()) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.disagg_prefill_mm_prealloc_queue = SimpleNamespace(
        pop_preallocated=lambda: []
    )
    scheduler.disagg_prefill_mm_transfer_queue = SimpleNamespace(
        extend=lambda reqs: None,
        pop_transferred=lambda: (list(mm_ready), list(mm_abort)),
    )
    scheduler.process_input_requests = MagicMock()
    scheduler.output_streamer = MagicMock()
    return scheduler


class TestProcessDisaggPrefillMMQueues(CustomTestCase):
    def test_returns_admitted_reqs(self):
        recv_req = _recv_req()
        scheduler = _scheduler(mm_ready=[recv_req])

        self.assertEqual(scheduler.process_disagg_prefill_mm_queues(), [recv_req])
        scheduler.process_input_requests.assert_not_called()

    def test_noop_when_queues_absent(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.disagg_prefill_mm_prealloc_queue = None

        self.assertEqual(scheduler.process_disagg_prefill_mm_queues(), [])

    def test_streams_aborts(self):
        req = SimpleNamespace(return_logprob=False)
        scheduler = _scheduler(mm_abort=[(req, "timeout", 408)])

        with patch(
            "sglang.srt.disaggregation.prefill.prepare_abort"
        ) as mock_prepare_abort:
            self.assertEqual(scheduler.process_disagg_prefill_mm_queues(), [])

        mock_prepare_abort.assert_called_once()
        scheduler.output_streamer.stream_output.assert_called_once_with([req], False)


class TestPPPDGetMMReadyReqs(CustomTestCase):
    def test_appends_metadata_only_copies(self):
        recv_req = _recv_req()
        scheduler = _scheduler(mm_ready=[recv_req])
        scheduler.pp_group = SimpleNamespace(is_last_rank=False)
        other = object()

        out = scheduler._pp_pd_get_mm_ready_reqs([other])

        scheduler.process_input_requests.assert_called_once_with([recv_req])
        self.assertEqual(len(out), 2)
        self.assertIs(out[0], other)
        self.assertIsNot(out[1], recv_req)
        for item in out[1].mm_inputs.mm_items:
            self.assertIsNone(item.precomputed_embeddings)

    def test_skips_forward_on_last_rank(self):
        scheduler = _scheduler(mm_ready=[_recv_req()])
        scheduler.pp_group = SimpleNamespace(is_last_rank=True)
        recv_reqs = [object()]

        self.assertEqual(scheduler._pp_pd_get_mm_ready_reqs(recv_reqs), recv_reqs)
        scheduler.process_input_requests.assert_called_once()

    def test_returns_input_unchanged_when_nothing_admitted(self):
        scheduler = _scheduler()
        scheduler.pp_group = SimpleNamespace(is_last_rank=False)
        recv_reqs = [object()]

        self.assertIs(scheduler._pp_pd_get_mm_ready_reqs(recv_reqs), recv_reqs)
        scheduler.process_input_requests.assert_not_called()

    def test_aborts_unrelayable_req_without_admitting_it(self):
        """An un-relayable payload must not enter any stage's waiting_queue."""
        bad_item = _mm_item(1, 3, 111)
        bad_item.hash = None
        bad = _recv_req(items=[bad_item])
        good = _recv_req()
        scheduler = _scheduler(mm_ready=[bad, good])
        scheduler.pp_group = SimpleNamespace(is_last_rank=False)
        aborted_req = SimpleNamespace(return_logprob=False)
        scheduler.mm_receiver = SimpleNamespace(create_req=lambda req: aborted_req)

        with patch(
            "sglang.srt.managers.scheduler_pp_mixin.prepare_abort"
        ) as mock_prepare_abort:
            out = scheduler._pp_pd_get_mm_ready_reqs([])

        mock_prepare_abort.assert_called_once()
        scheduler.output_streamer.stream_output.assert_called_once_with(
            [aborted_req], False
        )
        scheduler.process_input_requests.assert_called_once_with([good])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].rid, good.rid)


class TestMMTransferQueueInflightDrain(CustomTestCase):
    def test_drains_inflight_queue_when_queue_is_empty(self):
        """An abort can be the last entry, so the drain must not be gated on it."""
        transfer_queue = MMTransferQueue(SimpleNamespace(wait_timeout=1.0))
        buf_mgr = MagicMock()

        with patch(
            "sglang.srt.disaggregation.embedding_transfer.queues._MooncakeBufferManager"
        ) as mock_manager:
            mock_manager.get_instance_if_exists.return_value = buf_mgr
            self.assertEqual(transfer_queue.pop_transferred(), ([], []))

        buf_mgr.process_inflight_queue.assert_called_once_with()


class TestAbortFailedCacheRead(CustomTestCase):
    def test_releases_pool_buffers(self):
        """An AS-read abort drops the req before prefill, so it must free the pages."""
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_hicache_storage = False
        scheduler.disaggregation_mode = DisaggregationMode.PREFILL
        scheduler.req_to_metadata_buffer_idx_allocator = MagicMock()
        scheduler.ipc_channels = MagicMock()
        scheduler.ps = SimpleNamespace(pp_rank=0)
        req = SimpleNamespace(
            rid="rid-0",
            time_stats=SimpleNamespace(trace_ctx=MagicMock()),
            multimodal_inputs=MagicMock(),
            metadata_buffer_index=-1,
            pending_bootstrap=True,
            mamba_cow_src_index=None,
            mamba_needs_clear=False,
            req_pool_idx=None,
            mamba_pool_idx=None,
        )

        record = scheduler._abort_failed_cache_read(req)

        req.multimodal_inputs.release_pool_buffers.assert_called_once_with()
        self.assertEqual(record["reason_code"], "AS_CACHE_READ_FAILED")


if __name__ == "__main__":
    unittest.main()
