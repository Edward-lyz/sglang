"""Lifecycle tests for the draft-side SWA slot map (PR2): slot conservation,
window-bounded live set, and high-water-mark release sweeping."""

import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.draft_swa_pool import DraftWindowSlotMap
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

_W = 8
_GAMMA = 3
_MAX_REQS = 4
_ROW_WIDTH = 256


def _make(window_size=_W, page_size=1, num_swa_slots=None):
    if num_swa_slots is None:
        num_swa_slots = DraftWindowSlotMap.recommended_num_swa_slots(
            max_num_reqs=_MAX_REQS,
            window_size=window_size,
            gamma=_GAMMA,
            page_size=page_size,
        )
    slot_map = DraftWindowSlotMap(
        num_full_slots=4096,
        num_swa_slots=num_swa_slots,
        max_num_reqs=_MAX_REQS,
        window_size=window_size,
        page_size=page_size,
    )
    # Disjoint global slot ids per request.
    req_to_token = torch.zeros((_MAX_REQS, _ROW_WIDTH), dtype=torch.int64)
    nxt = 1
    for r in range(_MAX_REQS):
        req_to_token[r] = torch.arange(nxt, nxt + _ROW_WIDTH)
        nxt += _ROW_WIDTH
    return slot_map, req_to_token


def _run_decode(slot_map, req_to_token, *, req, start_pos, steps, accept):
    """Drive `steps` decode steps for one request; return the committed length."""
    pos = start_pos
    for _ in range(steps):
        slot_map.map_slots(
            req_to_token[req, pos : pos + _GAMMA], req_pool_idx=req, end_pos=pos + _GAMMA
        )
        pos += accept
        slot_map.advance_window(
            req_pool_indices_cpu=[req],
            committed_lens_cpu=[pos],
            target_req_to_token=req_to_token,
        )
    return pos


class TestSlotConservation(CustomTestCase):
    def test_single_request_returns_every_slot(self):
        slot_map, req_to_token = _make()
        start_free = slot_map.num_free_slots

        slot_map.map_slots(req_to_token[2, :4], req_pool_idx=2, end_pos=4)
        _run_decode(slot_map, req_to_token, req=2, start_pos=4, steps=15, accept=2)
        slot_map.release_request(req_pool_idx=2, target_req_to_token=req_to_token)

        self.assertEqual(slot_map.num_free_slots, start_free)
        self.assertEqual(slot_map.num_live_slots, 0)

    def test_concurrent_requests_return_every_slot(self):
        """Interleaved requests must not leak or free each other's slots."""
        slot_map, req_to_token = _make()
        start_free = slot_map.num_free_slots
        reqs = [0, 1, 3]
        pos = {}
        for r in reqs:
            slot_map.map_slots(req_to_token[r, :5], req_pool_idx=r, end_pos=5)
            pos[r] = 5

        for _ in range(10):
            for r in reqs:
                slot_map.map_slots(
                    req_to_token[r, pos[r] : pos[r] + _GAMMA],
                    req_pool_idx=r,
                    end_pos=pos[r] + _GAMMA,
                )
                pos[r] += 1
            slot_map.advance_window(
                req_pool_indices_cpu=reqs,
                committed_lens_cpu=[pos[r] for r in reqs],
                target_req_to_token=req_to_token,
            )

        # Every request still owns a live window while the others are running.
        for r in reqs:
            live = slot_map.translate(req_to_token[r, : pos[r]])
            self.assertTrue(bool((live > 0).any()), f"req {r} lost its whole window")

        for r in reqs:
            slot_map.release_request(req_pool_idx=r, target_req_to_token=req_to_token)
        self.assertEqual(slot_map.num_free_slots, start_free)
        self.assertEqual(slot_map.num_live_slots, 0)

    def test_slot_reuse_after_release_is_disjoint(self):
        """A released slot must be reusable and not remain mapped anywhere."""
        slot_map, req_to_token = _make()
        first = slot_map.map_slots(req_to_token[0, :6], req_pool_idx=0, end_pos=6)
        slot_map.release_request(req_pool_idx=0, target_req_to_token=req_to_token)

        self.assertTrue(
            bool((slot_map.translate(req_to_token[0, :6]) == 0).all()),
            "released full ids still translate to a live swa slot",
        )
        second = slot_map.map_slots(req_to_token[1, :6], req_pool_idx=1, end_pos=6)
        self.assertEqual(sorted(first.tolist()), sorted(second.tolist()))


class TestWindowBound(CustomTestCase):
    def test_live_set_stays_window_bounded(self):
        """The live set must stay window-bounded, not grow with sequence length."""
        slot_map, req_to_token = _make()
        bound = max(_W, 1) + _GAMMA + 1

        slot_map.map_slots(req_to_token[2, :4], req_pool_idx=2, end_pos=4)
        pos = 4
        peak = 0
        for _ in range(30):
            slot_map.map_slots(
                req_to_token[2, pos : pos + _GAMMA], req_pool_idx=2, end_pos=pos + _GAMMA
            )
            pos += 2
            slot_map.advance_window(
                req_pool_indices_cpu=[2],
                committed_lens_cpu=[pos],
                target_req_to_token=req_to_token,
            )
            peak = max(peak, slot_map.num_live_slots)
            self.assertLessEqual(
                slot_map.num_live_slots,
                bound,
                f"live set {slot_map.num_live_slots} exceeds window bound {bound} "
                f"at committed_len={pos}",
            )
        # Sanity: the run really did go far past the window.
        self.assertGreater(pos, 4 * _W)

    def test_window_contents_survive_eviction(self):
        """Eviction must never free a slot still inside the window."""
        slot_map, req_to_token = _make()
        slot_map.map_slots(req_to_token[2, :4], req_pool_idx=2, end_pos=4)
        pos = _run_decode(
            slot_map, req_to_token, req=2, start_pos=4, steps=12, accept=2
        )

        window_start = pos - min(pos, _W)
        in_window = slot_map.translate(req_to_token[2, window_start:pos])
        self.assertTrue(
            bool((in_window > 0).all()),
            f"eviction dropped in-window slots: {in_window.tolist()}",
        )

    def test_eviction_frontier_is_monotonic_and_idempotent(self):
        slot_map, req_to_token = _make()
        slot_map.map_slots(req_to_token[2, :40], req_pool_idx=2, end_pos=40)

        freed_first = slot_map.advance_window(
            req_pool_indices_cpu=[2],
            committed_lens_cpu=[40],
            target_req_to_token=req_to_token,
        )
        self.assertGreater(freed_first, 0)
        frontier = int(slot_map.evict_frontier[2])

        # Same length again: nothing left to free, frontier unmoved.
        self.assertEqual(
            slot_map.advance_window(
                req_pool_indices_cpu=[2],
                committed_lens_cpu=[40],
                target_req_to_token=req_to_token,
            ),
            0,
        )
        self.assertEqual(int(slot_map.evict_frontier[2]), frontier)

        # A SHORTER length must not walk the frontier backwards (a retract or a
        # stale host length would otherwise re-free already-freed slots).
        slot_map.advance_window(
            req_pool_indices_cpu=[2],
            committed_lens_cpu=[10],
            target_req_to_token=req_to_token,
        )
        self.assertEqual(int(slot_map.evict_frontier[2]), frontier)


class TestRejectedTail(CustomTestCase):
    def test_release_sweeps_mapped_high_water_not_committed_len(self):
        """Release must free mapped-but-uncommitted (rejected tail) slots too."""
        slot_map, req_to_token = _make()
        start_free = slot_map.num_free_slots

        slot_map.map_slots(req_to_token[1, :4], req_pool_idx=1, end_pos=4)
        pos = 4
        # Final step accepts 1 of gamma, so 2 block slots stay above committed.
        slot_map.map_slots(
            req_to_token[1, pos : pos + _GAMMA], req_pool_idx=1, end_pos=pos + _GAMMA
        )
        committed = pos + 1
        self.assertGreater(int(slot_map.mapped_high_water[1]), committed)

        slot_map.release_request(req_pool_idx=1, target_req_to_token=req_to_token)
        self.assertEqual(slot_map.num_free_slots, start_free)
        self.assertEqual(slot_map.num_live_slots, 0)

    def test_remap_is_idempotent(self):
        """Re-mapping the same slot must not consume a second swa slot."""
        slot_map, req_to_token = _make()
        rows = req_to_token[0, :5]
        first = slot_map.map_slots(rows, req_pool_idx=0, end_pos=5)
        free_after_first = slot_map.num_free_slots
        second = slot_map.map_slots(rows, req_pool_idx=0, end_pos=5)

        self.assertTrue(torch.equal(first, second))
        self.assertEqual(slot_map.num_free_slots, free_after_first)


class TestFailureModes(CustomTestCase):
    def test_exhaustion_raises_instead_of_returning_slot_zero(self):
        """Exhausting the pool must raise, not silently hand out slot 0."""
        slot_map, req_to_token = _make(num_swa_slots=6)  # 5 usable
        slot_map.map_slots(req_to_token[0, :5], req_pool_idx=0, end_pos=5)
        with self.assertRaises(RuntimeError):
            slot_map.map_slots(req_to_token[1, :2], req_pool_idx=1, end_pos=2)

    def test_negative_sentinel_is_dropped_not_mapped(self):
        """-1 is the 'no last_loc' sentinel and must never consume a slot."""
        slot_map, _ = _make()
        free_before = slot_map.num_free_slots
        self.assertEqual(slot_map.free_slots(torch.tensor([-1], dtype=torch.int64)), 0)
        self.assertEqual(slot_map.num_free_slots, free_before)


class TestSyncFreeBookkeeping(CustomTestCase):
    """The map/free paths keep no host-side counts, so the invariants the old
    ``torch.unique`` + readback chain got for free need pinning down."""

    def test_free_stack_stays_a_permutation_of_the_draft_pages(self):
        """Masked pushes must not duplicate, drop, or alias a draft page."""
        for page_size in (1, 4):
            with self.subTest(page_size=page_size):
                slot_map, req_to_token = _make(page_size=page_size)
                for r in range(_MAX_REQS):
                    slot_map.map_slots(req_to_token[r, :12], req_pool_idx=r, end_pos=12)
                    _run_decode(
                        slot_map, req_to_token, req=r, start_pos=12, steps=20, accept=2
                    )
                    # No swa slot may be handed to two full ids at once.
                    live = slot_map.full_to_swa[:-1]
                    live = live[live > 0]
                    self.assertEqual(int(live.numel()), int(live.unique().numel()))
                for r in range(_MAX_REQS):
                    slot_map.release_request(
                        req_pool_idx=r, target_req_to_token=req_to_token
                    )
                self.assertTrue(
                    torch.equal(
                        slot_map._free_pages[: int(slot_map._free_top)].sort().values,
                        torch.arange(1, slot_map.num_draft_pages),
                    ),
                    "free stack is no longer a permutation of the draft pages",
                )

    def test_duplicate_ids_are_freed_once(self):
        """Dedup is by scatter race now; a repeated id must not double-push."""
        slot_map, req_to_token = _make(page_size=4)
        ids = req_to_token[0, :8]
        free_before = slot_map.num_free_slots
        slot_map.map_slots(ids.repeat(3), req_pool_idx=0, end_pos=8)
        self.assertLess(slot_map.num_free_slots, free_before)
        slot_map.free_slots(
            torch.cat([ids, ids, torch.full((5,), -1, dtype=torch.int64)])
        )
        self.assertEqual(slot_map.num_free_slots, free_before)

    def test_unchecked_exhaustion_raises_on_the_next_checked_call(self):
        """assert_capacity=False cannot raise in place; the latch must not be lost."""
        slot_map, req_to_token = _make(num_swa_slots=6)  # 5 usable
        slot_map.map_slots(req_to_token[0, :5], assert_capacity=False)
        slot_map.map_slots(req_to_token[1, :3], assert_capacity=False)
        with self.assertRaises(RuntimeError):
            slot_map.map_slots(req_to_token[2, :1], req_pool_idx=2, end_pos=1)


class TestPagedGeometry(CustomTestCase):
    def test_paged_frontier_is_page_aligned_and_keeps_the_window(self):
        """With page_size > 1 the eviction frontier must land on a page boundary."""
        page = 4
        slot_map, req_to_token = _make(window_size=_W, page_size=page)
        for committed in range(1, 80):
            threshold = slot_map.evict_threshold(committed)
            self.assertEqual(threshold % page, 0, f"committed={committed}")
            # Never evicts into the window.
            self.assertLessEqual(threshold, max(0, committed - _W))

    def test_recommended_size_scales_with_window_not_context(self):
        small = DraftWindowSlotMap.recommended_num_swa_slots(
            max_num_reqs=48, window_size=128, gamma=7
        )
        large_window = DraftWindowSlotMap.recommended_num_swa_slots(
            max_num_reqs=48, window_size=1024, gamma=7
        )
        self.assertLess(small, large_window)
        # Pin the order of magnitude: cost scales with window, not context.
        self.assertLess(small, 48 * 200)


class TestPageAlignment(CustomTestCase):
    """With page_size > 1, translation must preserve within-page offsets and
    contiguity so the page-table build (trtllm_mha) stays correct."""

    def test_within_page_offset_is_preserved(self):
        page = 8
        _, req_to_token = _make(window_size=_W, page_size=page)
        num_swa_slots = DraftWindowSlotMap.recommended_num_swa_slots(
            max_num_reqs=_MAX_REQS, window_size=_W, gamma=_GAMMA, page_size=page
        )
        # Fresh map per request: testing per-token offset preservation only.
        for r in range(_MAX_REQS):
            slot_map = DraftWindowSlotMap(
                num_full_slots=4096,
                num_swa_slots=num_swa_slots,
                max_num_reqs=_MAX_REQS,
                window_size=_W,
                page_size=page,
            )
            full = req_to_token[r, :40]
            swa = slot_map.map_slots(full, req_pool_idx=r, end_pos=40)
            self.assertTrue(
                bool(((swa % page) == (full % page)).all()),
                f"req {r}: within-page offset changed: "
                f"{list(zip(full.tolist(), swa.tolist()))}",
            )

    def test_contiguous_full_ids_map_to_contiguous_swa_ids(self):
        page = 8
        slot_map, req_to_token = _make(window_size=_W, page_size=page)
        full = req_to_token[1, :40]
        swa = slot_map.map_slots(full, req_pool_idx=1, end_pos=40)
        self.assertTrue(
            bool((torch.diff(swa) == 1).all()),
            f"contiguous full ids produced non-contiguous swa ids: {swa.tolist()}",
        )

    def test_same_full_page_shares_one_draft_page(self):
        page = 8
        slot_map, req_to_token = _make(window_size=_W, page_size=page)
        full = req_to_token[2, :64]
        swa = slot_map.map_slots(full, req_pool_idx=2, end_pos=64)
        full_pages = full // page
        draft_pages = swa // page
        for fp in torch.unique(full_pages).tolist():
            ids = draft_pages[full_pages == fp]
            self.assertEqual(
                int(ids.unique().numel()),
                1,
                f"full page {fp} split across draft pages {ids.unique().tolist()}",
            )

    def test_lifecycle_keeps_offsets_across_reuse(self):
        page = 8
        slot_map, req_to_token = _make(window_size=_W, page_size=page)
        full = req_to_token[0, :32]
        swa_first = slot_map.map_slots(full, req_pool_idx=0, end_pos=32)
        self.assertTrue(bool(((swa_first % page) == (full % page)).all()))
        slot_map.release_request(req_pool_idx=0, target_req_to_token=req_to_token)

        other = req_to_token[1, :32]
        swa_second = slot_map.map_slots(other, req_pool_idx=1, end_pos=32)
        self.assertTrue(bool(((swa_second % page) == (other % page)).all()))


class TestWiredWorkerPath(CustomTestCase):
    """Worker-side wiring (no GPU/model): reservation mapping and frontier
    resurrection guards, driven directly."""

    def _worker(self, *, window=_W, page_size=1):
        from types import SimpleNamespace

        from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
            DSparkWorkerV2,
        )

        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker.draft_window_size = window
        worker.use_compact_draft_cache = True
        worker.verify_num_draft_tokens = _GAMMA
        worker.page_size = page_size
        worker.device = "cpu"

        _, req_to_token = _make(window_size=window, page_size=page_size)
        worker.model_runner = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
            token_to_kv_pool_allocator=SimpleNamespace(size=4096),
        )
        # _init_draft_slot_map reads the DRAFT allocator's (already shrunk) size.
        worker.draft_model_runner = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(
                size=DraftWindowSlotMap.recommended_num_swa_slots(
                    max_num_reqs=_MAX_REQS,
                    window_size=window,
                    gamma=_GAMMA,
                    page_size=page_size,
                )
            )
        )
        worker.ps = SimpleNamespace(tp_rank=1)
        worker._init_draft_slot_map()
        return worker, req_to_token

    @staticmethod
    def _batch(entries):
        from types import SimpleNamespace

        return SimpleNamespace(
            reqs=[
                SimpleNamespace(req_pool_idx=r, kv_committed_len=c) for r, c in entries
            ]
        )

    def test_every_reserved_slot_is_mapped(self):
        """The over-allocated reservation range must be fully mapped."""
        from types import SimpleNamespace

        worker, req_to_token = self._worker()
        slot_map = worker._draft_slot_map
        committed = 4
        for _ in range(20):
            reserved_end = committed + 2 * _GAMMA
            worker._maintain_draft_swa_window(
                batch=self._batch([(2, committed)]),
                draft_input=SimpleNamespace(
                    reserved_seq_lens_cpu=torch.tensor(
                        [reserved_end], dtype=torch.int32
                    )
                ),
            )
            frontier = int(slot_map.evict_frontier[2])
            live_range = slot_map.translate(req_to_token[2, frontier:reserved_end])
            self.assertTrue(
                bool((live_range > 0).all()),
                f"unmapped slot in [{frontier}, {reserved_end}) at committed={committed}",
            )
            committed += 2

    def test_mapping_does_not_resurrect_evicted_positions(self):
        """Mapping must start at the eviction frontier, not resurrect evicted slots."""
        from types import SimpleNamespace

        worker, req_to_token = self._worker()
        slot_map = worker._draft_slot_map
        committed = 4
        for _ in range(25):
            worker._maintain_draft_swa_window(
                batch=self._batch([(2, committed)]),
                draft_input=SimpleNamespace(
                    reserved_seq_lens_cpu=torch.tensor(
                        [committed + 2 * _GAMMA], dtype=torch.int32
                    )
                ),
            )
            committed += 2

        frontier = int(slot_map.evict_frontier[2])
        self.assertGreater(frontier, 0, "window never advanced")
        evicted = slot_map.translate(req_to_token[2, :frontier])
        self.assertTrue(
            bool((evicted == 0).all()),
            "an evicted position was re-mapped by a later step",
        )

    def test_wired_release_returns_every_slot(self):
        from types import SimpleNamespace

        worker, _ = self._worker()
        slot_map = worker._draft_slot_map
        start_free = slot_map.num_free_slots

        committed = 4
        for _ in range(20):
            worker._maintain_draft_swa_window(
                batch=self._batch([(2, committed)]),
                draft_input=SimpleNamespace(
                    reserved_seq_lens_cpu=torch.tensor(
                        [committed + 2 * _GAMMA], dtype=torch.int32
                    )
                ),
            )
            committed += 2

        worker.release_draft_kv(req_pool_idx=2)
        self.assertEqual(slot_map.num_free_slots, start_free)
        self.assertEqual(slot_map.num_live_slots, 0)

    def test_disabled_window_builds_no_slot_map(self):
        """Windowing off must leave the whole mechanism inert."""
        from types import SimpleNamespace

        from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
            DSparkWorkerV2,
        )

        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker.draft_window_size = None
        worker.use_compact_draft_cache = False
        worker._draft_slot_map = None
        worker._init_draft_slot_map()
        self.assertIsNone(worker._draft_slot_map)
        # The maintenance and release hooks must be no-ops, not crashes.
        worker._maintain_draft_swa_window(
            batch=self._batch([(0, 4)]),
            draft_input=SimpleNamespace(reserved_seq_lens_cpu=None),
        )
        worker.release_draft_kv(req_pool_idx=0)

class TestPhysicalPoolShrink(CustomTestCase):
    """The draft's KV buffer must physically shrink, not just alias the target's."""

    def _worker(self, *, window, max_running_requests, target_size, page_size=1):
        from types import SimpleNamespace

        from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
            DSparkWorkerV2,
        )

        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker.draft_window_size = window
        worker.use_compact_draft_cache = True
        worker.verify_num_draft_tokens = _GAMMA
        worker.page_size = page_size
        worker.device = "cpu"
        req_to_token = torch.zeros(
            (max_running_requests, _ROW_WIDTH), dtype=torch.int64
        )
        worker.model_runner = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
            token_to_kv_pool_allocator=SimpleNamespace(size=target_size),
        )
        # Shrink path rebuilds the draft allocator; _init_draft_slot_map reads it.
        worker.draft_model_runner = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(
                size=DraftWindowSlotMap.recommended_num_swa_slots(
                    max_num_reqs=max_running_requests,
                    window_size=window,
                    gamma=_GAMMA,
                    page_size=page_size,
                )
            )
        )
        worker.ps = SimpleNamespace(tp_rank=1, pp_rank=0, pp_size=1)
        return worker

    def test_shrunk_max_total_tokens_scale_with_window_not_context(self):
        from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig

        worker = self._worker(
            window=128, max_running_requests=32, target_size=4_200_000
        )
        cfg = MemoryPoolConfig(
            max_total_num_tokens=4_200_000, max_running_requests=32
        )
        shrunk = worker._shrink_memory_pool_config_to_window(cfg)
        # Draft pool sized by window * max_reqs, not by target's context pool.
        self.assertLess(shrunk.max_total_num_tokens, cfg.max_total_num_tokens // 20)
        # And it must match what DraftWindowSlotMap wants -- otherwise the
        # allocator hands out ids the slot map can't hold, or the pool wastes
        # bytes past what the map addresses.
        expected = DraftWindowSlotMap.recommended_num_swa_slots(
            max_num_reqs=32, window_size=128, gamma=_GAMMA, page_size=1
        )
        self.assertEqual(shrunk.max_total_num_tokens, expected)

    def test_shrink_is_a_noop_when_window_would_grow_the_pool(self):
        """If the recommended pool would exceed the target's, keep the target size."""
        from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig

        worker = self._worker(
            window=100_000, max_running_requests=128, target_size=1024
        )
        cfg = MemoryPoolConfig(
            max_total_num_tokens=1024, max_running_requests=128
        )
        shrunk = worker._shrink_memory_pool_config_to_window(cfg)
        self.assertIs(shrunk, cfg)  # same object -- no shrink applied

    def test_alloc_passes_none_allocator_when_shrinking(self):
        """The target allocator must not be passed through, or the shrink is defeated."""
        from types import SimpleNamespace

        from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig

        worker = self._worker(
            window=_W, max_running_requests=_MAX_REQS, target_size=4096
        )
        recorded = {}

        def fake_alloc(*, memory_pool_config, req_to_token_pool, token_to_kv_pool_allocator):
            recorded["memory_pool_config"] = memory_pool_config
            recorded["req_to_token_pool"] = req_to_token_pool
            recorded["token_to_kv_pool_allocator"] = token_to_kv_pool_allocator

        worker._draft_worker = SimpleNamespace(alloc_memory_pool=fake_alloc)
        worker._is_pd_prefill = False
        worker._draft_is_moe = False

        original_alloc = SimpleNamespace(size=4096, name="target_alloc")
        original_pool = SimpleNamespace(name="target_pool")
        worker.alloc_memory_pool(
            memory_pool_config=MemoryPoolConfig(
                max_total_num_tokens=4096, max_running_requests=_MAX_REQS
            ),
            req_to_token_pool=original_pool,
            token_to_kv_pool_allocator=original_alloc,
        )

        # Decisive check: target allocator is NOT forwarded when shrinking.
        self.assertIsNone(recorded["token_to_kv_pool_allocator"])
        self.assertLess(
            recorded["memory_pool_config"].max_total_num_tokens, 4096
        )
        # req_to_token_pool is always aliased when windowing is on (a view).
        self.assertIsNone(recorded["req_to_token_pool"])

    def test_slot_map_num_full_slots_matches_shrunk_pool(self):
        """The slot map must be built against the shrunk pool, not the target's."""
        from types import SimpleNamespace

        worker = self._worker(
            window=_W, max_running_requests=_MAX_REQS, target_size=4096
        )
        # _init_draft_slot_map currently reads the OUTER (target) allocator;
        # pin that invariant so a future refactor doesn't silently regress it.
        expected_swa = DraftWindowSlotMap.recommended_num_swa_slots(
            max_num_reqs=_MAX_REQS,
            window_size=_W,
            gamma=_GAMMA,
            page_size=1,
        )
        worker._init_draft_slot_map()
        self.assertGreaterEqual(
            worker._draft_slot_map.num_swa_slots, expected_swa
        )
        # full_to_swa must cover every id the target allocator can hand out.
        self.assertGreaterEqual(
            worker._draft_slot_map.full_to_swa.numel(),
            worker.model_runner.token_to_kv_pool_allocator.size,
        )


    def test_slot_map_lives_on_worker_device(self):
        """The slot map must be created on the worker's device, not default to CPU."""
        from types import SimpleNamespace

        from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
            DSparkWorkerV2,
        )

        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker.draft_window_size = _W
        worker.use_compact_draft_cache = True
        worker.verify_num_draft_tokens = _GAMMA
        worker.page_size = 1
        worker.device = "meta"
        req_to_token = torch.zeros(
            (_MAX_REQS, _ROW_WIDTH), dtype=torch.int64, device="meta"
        )
        worker.model_runner = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
            token_to_kv_pool_allocator=SimpleNamespace(size=4096),
        )
        worker.draft_model_runner = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(
                size=DraftWindowSlotMap.recommended_num_swa_slots(
                    max_num_reqs=_MAX_REQS,
                    window_size=_W,
                    gamma=_GAMMA,
                    page_size=1,
                )
            )
        )
        worker.ps = SimpleNamespace(tp_rank=1)
        worker._init_draft_slot_map()
        self.assertEqual(worker._draft_slot_map.full_to_swa.device.type, "meta")


class TestWritePathTranslation(CustomTestCase):
    """The inject path must write at DRAFT-swa ids, not target-global ones."""

    def _wired(self):
        from types import SimpleNamespace

        from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
            DSparkWorkerV2,
        )

        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker.draft_window_size = _W
        worker.use_compact_draft_cache = True
        worker.verify_num_draft_tokens = _GAMMA
        worker.page_size = 1
        worker.device = "cpu"
        _, req_to_token = _make()
        worker.model_runner = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
            token_to_kv_pool_allocator=SimpleNamespace(size=4096),
            device="cpu",
        )
        worker.draft_model_runner = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(
                size=DraftWindowSlotMap.recommended_num_swa_slots(
                    max_num_reqs=_MAX_REQS,
                    window_size=_W,
                    gamma=_GAMMA,
                    page_size=1,
                )
            )
        )
        worker.ps = SimpleNamespace(tp_rank=1)
        worker._init_draft_slot_map()
        return worker, req_to_token

    def _injector(self, translator):
        from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
            TargetHiddenKvInjector,
        )

        injector = TargetHiddenKvInjector.__new__(TargetHiddenKvInjector)
        injector._slot_translator = translator
        return injector

    def test_prefill_maps_out_cache_loc_and_records_extent(self):
        from types import SimpleNamespace

        worker, req_to_token = self._wired()
        slot_map = worker._draft_slot_map
        worker._map_prefill_draft_slots(
            batch=SimpleNamespace(
                out_cache_loc=req_to_token[2, :5],
                reqs=[SimpleNamespace(req_pool_idx=2)],
                prefix_lens=[0],
                extend_lens=[5],
            ),
            # window (8) >= extend_len (5), so _forward_prefill's trimmed write
            # list is the whole extend segment.
            cache_loc=req_to_token[2, :5],
        )
        swa = worker._translate_draft_slots(req_to_token[2, :5])
        self.assertTrue(bool((swa > 0).all()), "prefill slots left unmapped")
        # release sweeps by position, so the extent must be recorded too.
        self.assertEqual(int(slot_map.mapped_high_water[2]), 5)

    def test_translated_ids_live_in_the_draft_space_not_the_global_one(self):
        """Draft ids must be far smaller than global ids, not identity-mapped."""
        from types import SimpleNamespace

        worker, req_to_token = self._wired()
        worker._map_prefill_draft_slots(
            batch=SimpleNamespace(
                out_cache_loc=req_to_token[3, :6],
                reqs=[SimpleNamespace(req_pool_idx=3)],
                prefix_lens=[0],
                extend_lens=[6],
            ),
            cache_loc=req_to_token[3, :6],
        )
        global_ids = req_to_token[3, :6]
        injector = self._injector(worker._translate_draft_slots)
        # Go through the INJECTOR (what the write path actually calls), not the helper.
        swa = injector._to_draft_slots(global_ids)
        self.assertLess(int(swa.max()), worker._draft_slot_map.num_swa_slots)
        self.assertFalse(
            bool(torch.equal(swa, global_ids)), "translation degraded to identity"
        )

    def test_translation_is_identity_when_windowing_is_off(self):
        injector = self._injector(None)
        loc = torch.tensor([7, 8, 9], dtype=torch.int64)
        self.assertTrue(torch.equal(injector._to_draft_slots(loc), loc))


@unittest.skipUnless(torch.cuda.is_available(), "compact rebuild is a triton kernel")
class TestReadWriteIdSpaceAgreement(CustomTestCase):
    """The compact row (READ side) must hold the same ids the inject path WRITES."""

    def test_row_holds_translated_swa_ids(self):
        from sglang.srt.speculative.draft_window import (
            compact_draft_seq_lens,
            rebuild_compact_draft_req_to_token,
        )

        device = "cuda"
        num_swa_slots = DraftWindowSlotMap.recommended_num_swa_slots(
            max_num_reqs=_MAX_REQS, window_size=_W, gamma=_GAMMA, page_size=1
        )
        slot_map = DraftWindowSlotMap(
            num_full_slots=4096,
            num_swa_slots=num_swa_slots,
            max_num_reqs=_MAX_REQS,
            window_size=_W,
            page_size=1,
            device=device,
        )
        target = torch.zeros((_MAX_REQS, 128), dtype=torch.int64, device=device)
        nxt = 1
        for r in range(_MAX_REQS):
            target[r] = torch.arange(nxt, nxt + 128, device=device)
            nxt += 128
        draft = torch.full((_MAX_REQS, 128), -9, dtype=torch.int64, device=device)

        reqs = [2, 0]
        prefix_lens = torch.tensor([20, 6], dtype=torch.int64, device=device)
        compact_lens = compact_draft_seq_lens(
            seq_lens=prefix_lens, window_size=_W, page_size=1, device=device
        )
        verify = torch.stack(
            [target[r, int(p) : int(p) + _GAMMA] for r, p in zip(reqs, prefix_lens)]
        )
        for i, r in enumerate(reqs):
            end = int(prefix_lens[i]) + _GAMMA
            slot_map.map_slots(target[r, :end], req_pool_idx=r, end_pos=end)

        rebuild_compact_draft_req_to_token(
            draft_req_to_token=draft,
            target_req_to_token=target,
            req_pool_indices=torch.tensor(reqs, dtype=torch.int64, device=device),
            prefix_lens=prefix_lens,
            draft_prefix_lens=compact_lens,
            verify_cache_loc_2d=verify,
            bs=len(reqs),
            block_size=_GAMMA,
            slot_translator=slot_map.translate,
        )
        torch.cuda.synchronize()

        for i, r in enumerate(reqs):
            span = int(compact_lens[i])
            live = draft[r, : span + _GAMMA]
            self.assertTrue(
                bool((live > 0).all()), f"req {r}: row has unmapped (0) entries"
            )
            self.assertLess(
                int(live.max()),
                slot_map.num_swa_slots,
                f"req {r}: row id outside the draft pool -- read would go OOB",
            )
            window_start = int(prefix_lens[i]) - span
            torch.testing.assert_close(
                draft[r, :span],
                slot_map.translate(
                    target[r, window_start : int(prefix_lens[i])]
                ),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                draft[r, span : span + _GAMMA],
                slot_map.translate(verify[i]),
                rtol=0,
                atol=0,
            )
            # The decisive check: the row is NOT the global ids it was built from.
            self.assertFalse(
                bool(
                    torch.equal(
                        draft[r, :span],
                        target[r, window_start : int(prefix_lens[i])],
                    )
                ),
                f"req {r}: row still holds target-global ids",
            )


class TestReleaseOrdering(CustomTestCase):
    """req_pool_idx must be snapshotted BEFORE release_kv_cache nulls it,
    else the draft sweep is silently skipped and slots leak."""

    def test_snapshot_before_release_prevents_draft_leak(self):
        from types import SimpleNamespace

        from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
            DSparkWorkerV2,
        )

        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker.draft_window_size = _W
        worker.use_compact_draft_cache = True
        worker.verify_num_draft_tokens = _GAMMA
        worker.page_size = 1
        worker.device = "cpu"
        _, req_to_token = _make()
        worker.model_runner = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
            token_to_kv_pool_allocator=SimpleNamespace(size=4096),
        )
        worker.draft_model_runner = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(
                size=DraftWindowSlotMap.recommended_num_swa_slots(
                    max_num_reqs=_MAX_REQS,
                    window_size=_W,
                    gamma=_GAMMA,
                    page_size=1,
                )
            )
        )
        worker.ps = SimpleNamespace(tp_rank=1)
        worker._init_draft_slot_map()
        slot_map = worker._draft_slot_map
        start_free = slot_map.num_free_slots

        slot_map.map_slots(req_to_token[2, :6], req_pool_idx=2, end_pos=6)
        req = SimpleNamespace(req_pool_idx=2)

        # Correct ordering: read the idx, then run the (destructive) release_kv_cache.
        draft_req_pool_idx = req.req_pool_idx
        req.req_pool_idx = None  # what ReqToTokenPool.free would do
        if draft_req_pool_idx is not None:
            worker.release_draft_kv(req_pool_idx=int(draft_req_pool_idx))
        self.assertEqual(slot_map.num_free_slots, start_free)

        # Wrong ordering: reading after release_kv_cache sees None and leaks.
        slot_map.map_slots(req_to_token[3, :6], req_pool_idx=3, end_pos=6)
        leaked = SimpleNamespace(req_pool_idx=3)
        leaked.req_pool_idx = None  # read-after-release equivalent
        if leaked.req_pool_idx is not None:
            worker.release_draft_kv(req_pool_idx=int(leaked.req_pool_idx))
        self.assertLess(slot_map.num_free_slots, start_free)


if __name__ == "__main__":
    unittest.main()


class TestBatchedMaintainDraftSwaWindow(CustomTestCase):
    """The batched window-tail mapping must match the per-request mapping it
    replaced: same covered ids, high-water marks, and eviction frontier."""

    def _wired(self, page_size):
        from types import SimpleNamespace

        from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
            DSparkWorkerV2,
        )

        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker.draft_window_size = _W
        worker.use_compact_draft_cache = True
        worker.verify_num_draft_tokens = _GAMMA
        worker.page_size = page_size
        worker.device = "cpu"
        _, req_to_token = _make(page_size=page_size)
        worker.model_runner = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
            token_to_kv_pool_allocator=SimpleNamespace(size=4096),
            device="cpu",
        )
        slots = DraftWindowSlotMap.recommended_num_swa_slots(
            max_num_reqs=_MAX_REQS,
            window_size=_W,
            gamma=_GAMMA,
            page_size=page_size,
        )
        worker.draft_model_runner = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(size=slots)
        )
        worker.ps = SimpleNamespace(tp_rank=1)
        worker._init_draft_slot_map()
        return worker, req_to_token

    def _batch(self, req_pool_indices, committed, reserved):
        from types import SimpleNamespace

        return SimpleNamespace(
            reqs=[
                SimpleNamespace(req_pool_idx=r, kv_committed_len=c)
                for r, c in zip(req_pool_indices, committed)
            ],
        ), SimpleNamespace(reserved_seq_lens_cpu=reserved)

    def test_batched_maintain_matches_per_request_reference(self):
        for page_size in (1, 4):
            with self.subTest(page_size=page_size):
                worker, req_to_token = self._wired(page_size)
                slot_map = worker._draft_slot_map
                # req 1 has a longer reserved extent than committed + gamma.
                reqs = [0, 1]
                committed = [_W + 2 * _GAMMA + 3, _GAMMA + 1]
                reserved = torch.tensor(
                    [committed[0] + 2 * _GAMMA, committed[1] + 2 * _GAMMA],
                    dtype=torch.int64,
                )
                # Distinct full ids per row so the covered extent is observable.
                for r in reqs:
                    req_to_token[r, :_ROW_WIDTH] = torch.arange(
                        1, _ROW_WIDTH + 1, dtype=torch.int64
                    ) + r * 1000
                batch, draft_input = self._batch(reqs, committed, reserved)
                worker._maintain_draft_swa_window(
                    batch=batch, draft_input=draft_input
                )

                # Reference: what the old per-request loop must have covered.
                for i, r in enumerate(reqs):
                    end = max(int(reserved[i]), committed[i] + _GAMMA)
                    window_start = max(0, end - _W - 2 * _GAMMA)
                    start = max(int(slot_map.evict_frontier[r]), window_start)
                    covered = slot_map.translate(req_to_token[r, start:end])
                    self.assertTrue(
                        bool((covered > 0).all()),
                        f"page={page_size} req={r}: unmapped slot in "
                        f"[{start},{end}) -> {covered.tolist()}",
                    )
                    self.assertEqual(
                        int(slot_map.mapped_high_water[r]),
                        end,
                        f"page={page_size} req={r}: high-water must reach the mapped extent",
                    )

    def test_batched_maintain_maps_only_once_per_step(self):
        worker, req_to_token = self._wired(1)
        slot_map = worker._draft_slot_map
        reqs = [0, 1, 2]
        committed = [_W] * 3
        reserved = torch.tensor([_W + _GAMMA] * 3, dtype=torch.int64)
        for r in reqs:
            req_to_token[r, :64] = torch.arange(1, 65, dtype=torch.int64) + r * 1000
        batch, draft_input = self._batch(reqs, committed, reserved)

        calls = []
        real_map = slot_map.map_slots

        def counting_map(full_indices, **kwargs):
            calls.append(int(full_indices.numel()))
            return real_map(full_indices, **kwargs)

        slot_map.map_slots = counting_map
        try:
            worker._maintain_draft_swa_window(
                batch=batch, draft_input=draft_input
            )
        finally:
            slot_map.map_slots = real_map
        # One batched call, not one per request (the pre-fix behaviour).
        self.assertEqual(len(calls), 1)
        self.assertEqual(sum(calls), 3 * (_W + _GAMMA))
