"""after_local_commit must floor the SPA write to the device-resident prefix.

On a warm hit the committed prefix can be partially offloaded to host/L2, so the
post-commit re-match only publishes the device-resident page-aligned front.
The write must floor to those published pages (truncating block_hash_ids and
re-deriving the checkpoint for the published terminal page) instead of raising
"SPA local commit did not publish the write prefix". A genuinely missing node
(node is None) must still be treated as a failure.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.storage.asradix import checkpoint_coordinator as cc
from sglang.srt.mem_cache.storage.asradix.checkpoint_coordinator import (
    SPACacheCoordinator,
    SPAGCacheCoordinator,
    _SPAWriteTask,
)
from sglang.srt.mem_cache.storage.asradix.cache import AttentionStoreUnifiedRadixCache
from sglang.test.test_utils import CustomTestCase

PAGE = 256


class _FakeSnapshot:
    def __init__(self, tag):
        self.tag = tag
        self.released = False

    def release(self):
        self.released = True


class _FakePageIds:
    def __init__(self, slots):
        self.slots = slots
        self.released = False

    def release(self):
        self.released = True


class _FakeCheckpointStore:
    def __init__(self):
        self.snapshot_calls = []

    def snapshot(self, state_slots):
        self.snapshot_calls.append(dict(state_slots))
        return _FakeSnapshot(("terminal", tuple(sorted(state_slots.items()))))


class _FakeCache:
    def __init__(self, page_size=PAGE):
        self.page_size = page_size
        self.tree_core = SimpleNamespace(is_eagle=False)
        self.ascli = SimpleNamespace(layout=SimpleNamespace(groups=["swa_page"]))
        # identity translation: swa slot == full slot
        self.token_to_kv_pool_allocator = SimpleNamespace(
            translate_loc_from_full_to_swa=lambda t: t
        )
        self.failures = []

    def _is_prefill_role(self):
        return True

    def inc_lock_ref(self, node):
        return SimpleNamespace(to_dec_params=lambda: "dec")

    def dec_lock_ref(self, node, params):
        pass

    def _log_as_failure(self, *args, **kwargs):
        self.failures.append((args, kwargs))


def _make_context(token_len, checkpoint_snapshot, block_hash_ids):
    req = SimpleNamespace(
        rid="r1",
        extra_key=None,
        full_untruncated_fill_ids=list(range(token_len * 4)),
        extend_range=SimpleNamespace(end=token_len),
        kv=SimpleNamespace(req_pool_idx=7),
    )
    task = _SPAWriteTask(
        rid="r1",
        page_endpoint=None,
        checkpoint_store=_FakeCheckpointStore(),
        block_hash_ids=list(block_hash_ids),
        page_ids=None,
        checkpoint_snapshot=checkpoint_snapshot,
        params={},
    )
    context = SimpleNamespace(
        req=req,
        token_ids=list(range(token_len)),
        snapshot=torch.arange(token_len, dtype=torch.int64),
        is_insert=True,
        prepared_write=task,
    )
    return context, task


def _run_after_local_commit(coordinator, cache, context, device_indices):
    node = object()
    match = SimpleNamespace(last_device_node=node, device_indices=device_indices)
    submitted = []
    discarded = []
    dummies = []
    prepared_slots = []

    coordinator.submit_write = lambda c, t: submitted.append(t)
    coordinator._discard_unsubmitted = lambda c, t: discarded.append(t)
    coordinator.enqueue_dummy = lambda c, rid: dummies.append(rid)

    def _fake_prepare_page_ids(c, slots):
        prepared_slots.append(slots)
        return _FakePageIds(slots)

    coordinator.prepare_page_ids = _fake_prepare_page_ids

    with mock.patch.object(cc.UnifiedRadixCache, "match_prefix", return_value=match):
        coordinator.after_local_commit(cache, context)

    return SimpleNamespace(
        submitted=submitted,
        discarded=discarded,
        dummies=dummies,
        prepared_slots=prepared_slots,
    )


class TestAfterLocalCommitFloor(unittest.TestCase):
    def test_floors_write_to_published_pages_on_partial_publish(self):
        coordinator = SPACacheCoordinator()
        cache = _FakeCache()
        old_snapshot = _FakeSnapshot("full-prefix")
        # key = 2 pages (512 tokens), block_hash_ids for both pages
        context, task = _make_context(
            token_len=2 * PAGE,
            checkpoint_snapshot=old_snapshot,
            block_hash_ids=[0xAA, 0xBB],
        )
        # only 1 page is device-resident (host-offloaded tail)
        device_indices = torch.arange(PAGE, dtype=torch.int64)

        result = _run_after_local_commit(coordinator, cache, context, device_indices)

        # write proceeds, not discarded
        self.assertEqual(len(result.submitted), 1)
        self.assertEqual(result.discarded, [])
        self.assertEqual(result.dummies, [])
        self.assertEqual(cache.failures, [])
        # floored to the single published page
        self.assertEqual(len(task.block_hash_ids), 1)
        self.assertEqual(list(task.block_hash_ids), [0xAA])
        self.assertEqual(len(result.prepared_slots[0]), PAGE)
        # checkpoint re-derived for the published terminal page; old released
        self.assertEqual(len(task.checkpoint_store.snapshot_calls), 1)
        self.assertTrue(old_snapshot.released)
        self.assertIsNot(task.checkpoint_snapshot, old_snapshot)

    def test_full_publish_keeps_write_unchanged(self):
        coordinator = SPACacheCoordinator()
        cache = _FakeCache()
        old_snapshot = _FakeSnapshot("full-prefix")
        context, task = _make_context(
            token_len=2 * PAGE,
            checkpoint_snapshot=old_snapshot,
            block_hash_ids=[0xAA, 0xBB],
        )
        # both pages device-resident: no floor
        device_indices = torch.arange(2 * PAGE, dtype=torch.int64)

        result = _run_after_local_commit(coordinator, cache, context, device_indices)

        self.assertEqual(len(result.submitted), 1)
        self.assertEqual(result.discarded, [])
        self.assertEqual(cache.failures, [])
        self.assertEqual(list(task.block_hash_ids), [0xAA, 0xBB])
        # no re-derivation, original checkpoint retained
        self.assertEqual(len(task.checkpoint_store.snapshot_calls), 0)
        self.assertFalse(old_snapshot.released)
        self.assertIs(task.checkpoint_snapshot, old_snapshot)

    def test_missing_node_is_still_a_failure(self):
        coordinator = SPACacheCoordinator()
        cache = _FakeCache()
        context, task = _make_context(
            token_len=2 * PAGE,
            checkpoint_snapshot=_FakeSnapshot("full-prefix"),
            block_hash_ids=[0xAA, 0xBB],
        )
        submitted = []
        discarded = []
        dummies = []
        coordinator.submit_write = lambda c, t: submitted.append(t)
        coordinator._discard_unsubmitted = lambda c, t: discarded.append(t)
        coordinator.enqueue_dummy = lambda c, rid: dummies.append(rid)
        coordinator.prepare_page_ids = lambda c, slots: _FakePageIds(slots)

        match = SimpleNamespace(
            last_device_node=None, device_indices=torch.arange(0, dtype=torch.int64)
        )
        with mock.patch.object(cc.UnifiedRadixCache, "match_prefix", return_value=match):
            coordinator.after_local_commit(cache, context)

        self.assertEqual(submitted, [])
        self.assertEqual(len(discarded), 1)
        self.assertEqual(dummies, ["r1"])
        self.assertEqual(len(cache.failures), 1)


class TestSPAGFinishedWriteReuse(CustomTestCase):
    def setUp(self):
        self.coordinator = SPAGCacheCoordinator()
        self.coordinator._reuse_finished_write = True
        self.req = SimpleNamespace(
            rid="reuse-request",
            origin_input_ids=list(range(2 * PAGE)),
            output_ids=[],
            effective_kv_committed_len=lambda: 2 * PAGE,
            kv=SimpleNamespace(req_pool_idx=0),
            as_block_hashs=SimpleNamespace(hash_ids=[11, 22]),
            as_kv_cache_params={},
            as_spag_submitted_write=((11, 22), {}),
        )
        self.cache = object.__new__(AttentionStoreUnifiedRadixCache)
        self.cache.tree_core = SimpleNamespace()
        self.cache.pp_size = 1
        self.cache.page_size = PAGE
        self.cache.disable = False
        self.cache.disable_finished_insert = False
        self.cache._as_active_write_owner = None
        self.cache._as_coordinator = self.coordinator
        self.cache.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(2 * PAGE).reshape(1, -1)
        )

    def test_reuse_keeps_local_commit_and_avoids_second_retention(self):
        with mock.patch.object(
            cc.UnifiedRadixCache, "cache_finished_req"
        ) as local_commit, mock.patch.object(
            self.coordinator, "prepare_write"
        ) as prepare_write, mock.patch.object(
            self.coordinator, "enqueue_dummy"
        ) as enqueue_dummy:
            self.cache.cache_finished_req(self.req, is_insert=True)
        local_commit.assert_called_once_with(self.req, True)
        prepare_write.assert_not_called()
        enqueue_dummy.assert_not_called()

    def test_changed_prefix_namespace_or_pp_still_writes(self):
        for hashes, params, pp_size in (
            ([11, 23], {}, 1),
            ([11, 22], {"namespace": "other"}, 1),
            ([11, 22], {}, 2),
        ):
            with self.subTest(hashes=hashes, params=params, pp_size=pp_size):
                self.req.as_block_hashs.hash_ids = hashes
                self.req.as_kv_cache_params = params
                self.cache.pp_size = pp_size
                context = self.coordinator.capture_finished(self.cache, self.req, True)
                self.assertTrue(context.is_insert)

    def test_failed_write_allows_retry_without_clearing_newer_prefix(self):
        task = _SPAWriteTask(
            rid=self.req.rid,
            page_endpoint=None,
            checkpoint_store=None,
            block_hash_ids=(11, 22),
            page_ids=None,
            checkpoint_snapshot=None,
            params={},
            request=self.req,
        )
        self.coordinator.finish(self.cache, task, -1)
        self.assertIsNone(self.req.as_spag_submitted_write)
        context = self.coordinator.capture_finished(self.cache, self.req, True)
        self.assertTrue(context.is_insert)
        self.req.as_spag_submitted_write = ((11, 22, 33), {})
        self.coordinator.finish(self.cache, task, -1)
        self.assertEqual(self.req.as_spag_submitted_write, ((11, 22, 33), {}))

    def test_batch_fence_covers_producer_and_scheduler_staging(self):
        self.coordinator._batch_write_fence = True
        producer = object()
        self.req.as_kv_ready_event = producer
        task = SimpleNamespace(request=self.req, swa_slot_ids=(), producer_ready=None)
        stream = mock.Mock()
        staged_ready = object()
        cache = SimpleNamespace(
            device="cuda",
            token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: object()),
        )
        with mock.patch.object(
            SPACacheCoordinator, "_make_release_after_write"
        ), mock.patch.object(
            cc.torch.cuda, "current_stream", return_value=stream
        ), mock.patch.object(
            self.coordinator, "record_device_event", return_value=staged_ready
        ) as record_event:
            self.coordinator._make_release_after_write(cache, task, object())
        stream.wait_event.assert_called_once_with(producer)
        record_event.assert_called_once_with("cuda", stream)
        self.assertIs(task.producer_ready, staged_ready)

    def test_reused_write_retries_before_releasing_staging(self):
        actions = []
        task = SimpleNamespace(
            requested_blocks=2,
            release_after_write=lambda: actions.append("release"),
        )
        results = iter((-1, 2))

        def write(cache, task):
            actions.append("write")
            return next(results)

        with mock.patch.object(self.coordinator, "do_write", side_effect=write):
            result = self.coordinator._run_write(self.cache, task)
        self.assertEqual(result, 2)
        self.assertEqual(actions, ["write", "write", "release"])

    def test_writer_uses_captured_fence_without_waiting_on_later_forward(self):
        captured_ready = object()
        store = mock.Mock()
        task = _SPAWriteTask(
            rid=self.req.rid,
            page_endpoint=SimpleNamespace(store=store),
            checkpoint_store=None,
            block_hash_ids=(11, 22),
            page_ids=SimpleNamespace(materialize=lambda: (7, 8)),
            checkpoint_snapshot=None,
            params={},
            request=self.req,
            producer_ready=captured_ready,
        )
        cache = SimpleNamespace(_log_as_result=mock.Mock(), _log_as_failure=mock.Mock())
        with mock.patch.object(
            self.coordinator, "_match_pages", side_effect=(0, 2)
        ), mock.patch.object(
            self.coordinator, "_write_chunks", return_value=[(0, 2, (), None)]
        ), mock.patch.object(self.coordinator, "record_device_event") as late_record:
            result = self.coordinator.do_write(cache, task)
        self.assertEqual(result, 2)
        late_record.assert_not_called()
        self.assertIs(store.write.call_args.kwargs["producer_ready"], captured_ready)
        cache._log_as_failure.assert_not_called()


if __name__ == "__main__":
    unittest.main()
