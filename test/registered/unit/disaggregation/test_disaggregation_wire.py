import unittest
from types import SimpleNamespace

import numpy as np
import torch

from sglang.srt.disaggregation.base.conn import KVArgs, StateType
from sglang.srt.disaggregation.common.utils import (
    build_dcp_replicated_token_transfer_plan,
    build_dcp_token_transfer_plan,
    group_concurrent_contiguous,
    pack_int_lists,
    pack_list_of_buffers,
    unpack_int_lists,
    unpack_list_of_buffers,
)
from sglang.srt.disaggregation.prefill import bounded_send_chunk_ends
from sglang.srt.disaggregation.utils import (
    MetadataBuffers,
    get_dsv4_c128_state_indices,
    setup_state_kv_args,
)
from sglang.srt.managers.overlap_utils import FutureMap, RelayPayload
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.speculative.eagle_disaggregation import (
    build_eagle_disagg_draft_input,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDisaggregationWire(unittest.TestCase):
    def test_int_lists_roundtrip(self):
        cases = [
            ("Q", [[1, 2, 3], [4]]),
            ("I", [[10, 20], [30, 40, 50]]),
            ("i", [[-1, 2], [3, -4, 5]]),
        ]
        for fmt, sample in cases:
            packed = pack_int_lists(sample, fmt)
            self.assertEqual(unpack_int_lists(packed, fmt), sample, msg=fmt)

    def test_pack_accepts_ndarray(self):
        arrs = [
            np.array([1, 2, 3], dtype=np.int32),
            np.array([4, 5], dtype=np.int32),
        ]
        packed = pack_int_lists(arrs, "i")
        self.assertEqual(unpack_int_lists(packed, "i"), [[1, 2, 3], [4, 5]])

    def test_empty_outer_list(self):
        self.assertEqual(pack_int_lists([], "Q"), b"")
        self.assertEqual(unpack_int_lists(b"", "Q"), [])

    def test_empty_inner_list(self):
        packed = pack_int_lists([[]], "I")
        self.assertEqual(unpack_int_lists(packed, "I"), [[]])

    def test_list_of_buffers_roundtrip(self):
        bufs = [b"abc", b"", b"de", b"x" * 17]
        self.assertEqual(unpack_list_of_buffers(pack_list_of_buffers(bufs)), bufs)


class TestGroupConcurrentContiguous(unittest.TestCase):
    @staticmethod
    def _arr(values):
        return np.array(values, dtype=np.int32)

    def test_single_contiguous_group(self):
        src = self._arr([10, 11, 12])
        dst = self._arr([5, 6, 7])
        self.assertEqual(
            group_concurrent_contiguous(src, dst),
            ([[10, 11, 12]], [[5, 6, 7]]),
        )

    def test_splits_on_discontiguous_indices(self):
        src = self._arr([10, 11, 20])
        dst = self._arr([5, 6, 7])
        self.assertEqual(
            group_concurrent_contiguous(src, dst),
            ([[10, 11], [20]], [[5, 6], [7]]),
        )

    def test_empty_src_nonempty_dst(self):
        self.assertEqual(
            group_concurrent_contiguous(self._arr([]), self._arr([1, 2])), ([], [])
        )

    def test_nonempty_src_empty_dst(self):
        # Regression: a non-empty source paired with an empty destination must not
        # raise a NumPy broadcast error (observed transferring DSA sparse-attention
        # state on a disaggregated GLM deployment when decode registered zero dst indices).
        self.assertEqual(
            group_concurrent_contiguous(self._arr([1, 2]), self._arr([])), ([], [])
        )

    def test_mismatched_nonempty_lengths_raise(self):
        with self.assertRaises(ValueError):
            group_concurrent_contiguous(self._arr([1, 2, 3]), self._arr([1, 2]))


class TestEagleDsaSeedTransfer(unittest.TestCase):
    @staticmethod
    def _make_req(seed, metadata_buffer_index=0):
        return SimpleNamespace(
            metadata_buffer_index=metadata_buffer_index,
            output_ids=[101],
            cached_tokens=0,
            cached_tokens_device=0,
            cached_tokens_host=0,
            cached_tokens_storage=0,
            multimodal_inputs=None,
            return_logprob=False,
            return_sampling_mask=False,
            hidden_states_tensor=torch.tensor([1.0, 2.0]),
            output_topk_p=torch.tensor([1.0]),
            output_topk_index=torch.tensor([7]),
            output_dsa_topk_indices=seed,
            bootstrap_room=9,
        )

    def test_metadata_buffer_copies_seed_and_uses_invalid_sentinel(self):
        buffers = MetadataBuffers(
            size=2,
            hidden_size=2,
            hidden_states_dtype=torch.float32,
            output_dsa_topk_indices_dim=3,
        )
        seed = torch.tensor([4, 5, 6], dtype=torch.int32)
        buffers.set_buf(self._make_req(seed))
        buffers.set_buf(self._make_req(None, metadata_buffer_index=1))

        self.assertTrue(torch.equal(buffers.output_dsa_topk_indices[0], seed))
        self.assertEqual(buffers.output_dsa_topk_indices[1].tolist(), [-1, -1, -1])
        ptrs, data_lens, item_lens = buffers.get_buf_infos()
        self.assertEqual(ptrs[-2], buffers.output_dsa_topk_indices.data_ptr())
        self.assertEqual(data_lens[-2], buffers.output_dsa_topk_indices.nbytes)
        self.assertEqual(item_lens[-2], buffers.output_dsa_topk_indices[0].nbytes)

    def test_decode_input_requires_valid_seed_for_every_request(self):
        seeds = (
            torch.tensor([1, 2, 3], dtype=torch.int32),
            torch.tensor([4, 5, 6], dtype=torch.int32),
        )
        batch = SimpleNamespace(
            reqs=[self._make_req(seed) for seed in seeds],
            device="cpu",
            enable_overlap=False,
        )
        server_args = SimpleNamespace(
            speculative_eagle_topk=1,
            speculative_num_steps=5,
            enable_multi_layer_eagle=False,
        )
        last_tokens = torch.tensor([11, 12], dtype=torch.int64)

        draft_input = build_eagle_disagg_draft_input(
            batch, server_args, last_tokens, None
        )
        self.assertTrue(torch.equal(draft_input.dsa_topk_indices, torch.stack(seeds)))

        for invalid_seed in (
            None,
            torch.full((3,), -1, dtype=torch.int32),
        ):
            batch.reqs[1].output_dsa_topk_indices = invalid_seed
            draft_input = build_eagle_disagg_draft_input(
                batch, server_args, last_tokens, None
            )
            self.assertIsNone(draft_input.dsa_topk_indices)

    def test_future_map_initializes_seed_buffer_after_seedless_payload(self):
        future_map = object.__new__(FutureMap)
        future_map.dsa_topk_indices_buf = None
        future_map.req_pool_size = 4
        future_map.device = "cpu"
        future_map._maybe_init_dsa_topk_indices_buf(
            RelayPayload(bonus_tokens=torch.zeros((2,), dtype=torch.int64))
        )
        self.assertIsNone(future_map.dsa_topk_indices_buf)

        seeds = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
        future_map._maybe_init_dsa_topk_indices_buf(
            RelayPayload(
                bonus_tokens=torch.zeros((2,), dtype=torch.int64),
                dsa_topk_indices=seeds,
            )
        )
        self.assertEqual(future_map.dsa_topk_indices_buf.shape, (4, 3))
        self.assertEqual(future_map.dsa_topk_indices_buf.dtype, torch.int32)


class TestDSV4C128StateIndices(unittest.TestCase):
    def test_online_aligned_boundary_has_no_partial_state(self):
        np.testing.assert_array_equal(
            get_dsv4_c128_state_indices(7, 256, online=True, ring_size=1),
            np.empty((0,), dtype=np.int32),
        )

    def test_online_partial_boundary_uses_request_slot(self):
        np.testing.assert_array_equal(
            get_dsv4_c128_state_indices(7, 257, online=True, ring_size=1),
            np.array([7], dtype=np.int32),
        )

    def test_offline_aligned_boundary_has_no_partial_state(self):
        np.testing.assert_array_equal(
            get_dsv4_c128_state_indices(7, 256, online=False, ring_size=128),
            np.empty((0,), dtype=np.int32),
        )

    def test_offline_partial_boundary_uses_request_local_page(self):
        np.testing.assert_array_equal(
            get_dsv4_c128_state_indices(7, 129, online=False, ring_size=256),
            np.array([15], dtype=np.int32),
        )


def _buf_infos(*ptrs):
    return list(ptrs), [ptr + 100 for ptr in ptrs], [ptr + 200 for ptr in ptrs]


def _make_dsv4_target(*, unified, mapping=None):
    pool = object.__new__(DeepSeekV4TokenToKVPool)
    pool._unified_kv = unified
    pool.page_size = 256
    pool.sliding_window = 128
    pool.full_to_swa_index_mapping = mapping
    pool.unified_swa_window = 128
    pool.unified_swa_ring_size = 131
    pool.unified_swa_pages = 524
    pool.get_state_buf_infos = lambda: _buf_infos(11)
    pool.get_unified_swa_ring_buf_infos = lambda: (
        _buf_infos(12) if unified else ([], [], [])
    )
    pool.get_c128_state_buf_infos = lambda: ([], [], [])
    return pool


def _make_dsv4_draft(*, unified, mapping=None):
    pool = object.__new__(DeepSeekV4TokenToKVPool)
    pool._unified_kv = unified
    pool.compression_ratios = [0]
    pool.page_size = 256
    pool.sliding_window = 128
    pool.full_to_swa_index_mapping = mapping
    pool.unified_swa_window = 128
    pool.unified_swa_ring_size = 131
    pool.unified_swa_pages = 524
    pool.compress_state_pools = [None]
    pool.indexer_compress_state_pools = [None]
    if unified:
        pool.unified_kv_pool = SimpleNamespace(
            swa_pages=524,
            kv_buffer=[torch.empty((524, 16), dtype=torch.uint8)],
        )
    else:
        pool.swa_kv_pool = SimpleNamespace(
            kv_buffer=[torch.empty((2, 16), dtype=torch.uint8)]
        )
    return pool


class TestDSV4DraftStateRegistration(unittest.TestCase):
    def test_draft_state_is_a_separate_component(self):
        mapping = torch.arange(16)
        cases = [
            (
                "paged",
                _make_dsv4_target(unified=False, mapping=mapping),
                _make_dsv4_draft(unified=False, mapping=mapping),
                [StateType.SWA, StateType.SWA],
                [[11]],
            ),
            (
                "unified",
                _make_dsv4_target(unified=True),
                _make_dsv4_draft(unified=True),
                [StateType.SWA, StateType.SWA_RING, StateType.SWA_RING],
                [[11], [12]],
            ),
        ]

        for name, target, draft, expected_types, target_ptrs in cases:
            with self.subTest(name=name):
                if draft._unified_kv:
                    expected_infos = draft.get_unified_swa_ring_buf_infos()
                else:
                    expected_infos = draft.get_state_buf_infos()
                kv_args = KVArgs()

                setup_state_kv_args(kv_args, target, draft)

                self.assertEqual(kv_args.state_types, expected_types)
                self.assertEqual(kv_args.state_data_ptrs[:-1], target_ptrs)
                self.assertEqual(kv_args.state_data_ptrs[-1], expected_infos[0])
                self.assertEqual(kv_args.state_data_lens[-1], expected_infos[1])
                self.assertEqual(kv_args.state_item_lens[-1], expected_infos[2])


class TestBoundedSendChunkEnds(CustomTestCase):
    """Boundary math of the KV send-chunk splitter.

    ``send_kv_chunk`` relies on three derived properties: the last boundary is
    exactly ``end_idx`` (only that sub-chunk carries aux/state and the decode-side
    last-chunk bookkeeping), a range that needs no split yields exactly one
    boundary (so the unbounded path is bit-identical to the pre-split behavior),
    and no sub-range exceeds the budget (that is the whole point of the cap).
    """

    def test_no_split_cases_yield_single_boundary(self):
        cases = {
            "cap disabled": (0, 100, 0),
            "cap negative": (0, 100, -1),
            "range fits": (0, 4096, 4096),
            "empty range carries aux only": (4096, 4096, 4096),
        }
        for name, (start, end, budget) in cases.items():
            with self.subTest(name=name):
                self.assertEqual(bounded_send_chunk_ends(start, end, budget), [end])

    def test_split_boundaries(self):
        self.assertEqual(bounded_send_chunk_ends(0, 4097, 4096), [4096, 4097])
        self.assertEqual(
            bounded_send_chunk_ends(1000, 3000, 512), [1512, 2024, 2536, 3000]
        )

    def test_derived_properties_hold_across_ranges(self):
        page_size = 64
        budget = 4 * page_size
        for end in range(0, 2000, 37):
            for start in (0, page_size, 5 * page_size):
                if start > end:
                    continue
                ends = bounded_send_chunk_ends(start, end, budget)
                self.assertEqual(ends[-1], end)
                self.assertEqual(ends, sorted(set(ends)))
                bounds = [start] + ends
                for lo, hi in zip(bounds, bounds[1:]):
                    self.assertLessEqual(hi - lo, budget)
                # Intermediate boundaries stay on the page grid, so every
                # sub-chunk but the last maps to a whole number of KV pages.
                for boundary in ends[:-1]:
                    self.assertEqual(boundary % page_size, 0)

    def test_split_dcp_plans_equal_unsplit_plan(self):
        """Chunking must not change any source or destination RDMA address."""
        page_size = 4
        num_tokens = 29
        src_pages = np.arange(10, 18, dtype=np.int32)
        dst_pages = np.arange(100, 104, dtype=np.int32)
        chunk_ends = bounded_send_chunk_ends(0, num_tokens, 2 * page_size)

        plan_builders = (
            (
                "sharded MLA",
                build_dcp_token_transfer_plan,
                {"dcp_rank": 1},
            ),
            (
                "replicated draft",
                build_dcp_replicated_token_transfer_plan,
                {},
            ),
        )
        for name, build_plan, extra_args in plan_builders:
            with self.subTest(layout=name):
                unsplit = build_plan(
                    src_pages,
                    dst_pages,
                    physical_page_size=page_size,
                    dcp_size=2,
                    src_page_offset=0,
                    decode_prefix_len=0,
                    num_kv_tokens=num_tokens,
                    **extra_args,
                )

                split_src = []
                split_dst = []
                chunk_start = 0
                for chunk_end in chunk_ends:
                    first_page = chunk_start // page_size
                    last_page = (chunk_end + page_size - 1) // page_size
                    chunk_plan = build_plan(
                        src_pages[first_page:last_page],
                        dst_pages,
                        physical_page_size=page_size,
                        dcp_size=2,
                        src_page_offset=first_page,
                        decode_prefix_len=0,
                        num_kv_tokens=chunk_end - chunk_start,
                        **extra_args,
                    )
                    split_src.append(chunk_plan.src_token_indices)
                    split_dst.append(chunk_plan.dst_token_indices)
                    chunk_start = chunk_end

                np.testing.assert_array_equal(
                    np.concatenate(split_src), unsplit.src_token_indices
                )
                np.testing.assert_array_equal(
                    np.concatenate(split_dst), unsplit.dst_token_indices
                )


if __name__ == "__main__":
    unittest.main()
