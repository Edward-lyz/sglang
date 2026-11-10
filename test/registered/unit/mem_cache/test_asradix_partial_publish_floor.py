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
    _SPAWriteTask,
)

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


if __name__ == "__main__":
    unittest.main()
