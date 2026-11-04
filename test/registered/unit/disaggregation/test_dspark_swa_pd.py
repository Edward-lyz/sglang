import concurrent.futures
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager


class TestDSparkSWAPD(unittest.TestCase):
    def test_chunked_prefill_transfers_each_chunk_overlapping_draft_window(self):
        """A draft SWA window must not be populated from the last chunk only.

        This is intentionally a contract test for the transfer integration.  A
        10-token prompt split into [0:6] and [6:10] chunks has a window [2:10],
        so both chunks overlap the window and must contribute KV entries.
        """
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.kv_args = SimpleNamespace(
            page_size=1,
            kv_data_ptrs=[10_000, 20_000],
            kv_item_lens=[256, 256],
            engine_rank=0,
        )
        manager.attn_tp_size = 1
        manager.server_args = SimpleNamespace(speculative_draft_window_size=None)
        manager.engine = Mock()
        manager.engine.batch_transfer_sync.return_value = 0

        dst_window = np.arange(1002, 1010, dtype=np.int32)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            for start, stop in ((0, 6), (6, 10)):
                status = manager.send_kvcache_draft(
                    "session",
                    np.arange(2000 + start, 2000 + stop, dtype=np.int32),
                    [30_000, 40_000],
                    dst_window,
                    src_mla_entry_count=0,
                    dst_mla_entry_count=0,
                    dst_attn_tp_size=1,
                    dst_tp_rank=0,
                    draft_total_kv_heads=1,
                    dst_dcp_size=1,
                    src_page_offset=start,
                    decode_prefix_len=0,
                    num_kv_tokens=stop - start,
                    draft_window_start=2,
                    executor=executor,
                )
                self.assertEqual(status, 0)

        # Both chunks must have generated a transfer.  Current code returns 0
        # before doing any work for the first chunk.
        self.assertEqual(manager.engine.batch_transfer_sync.call_count, 4)

    # ---- helpers --------------------------------------------------------

    @staticmethod
    def _manager(page_size, *, attn_tp_size=1, engine_rank=0):
        """A MooncakeKVManager stub whose draft KV token stride is 256 bytes.

        The stride is PER RANK, so a prefill rank always owns 256 B/token no
        matter how many attn-TP ranks share the heads. ``attn_tp_size`` is the
        PREFILL side's; the decode side's is passed to ``_send`` as
        ``dst_attn_tp_size``. A mismatch (e.g. 8 -> 1 for decode DP16) is what
        exercises the M-to-N head slicing.
        """
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.kv_args = SimpleNamespace(
            page_size=page_size,
            kv_data_ptrs=[10_000, 20_000],
            kv_item_lens=[page_size * 256, page_size * 256],
            engine_rank=engine_rank,
        )
        manager.attn_tp_size = attn_tp_size
        manager.server_args = SimpleNamespace(speculative_draft_window_size=None)
        manager.engine = Mock()
        manager.engine.batch_transfer_sync.return_value = 0
        return manager

    @staticmethod
    def _send(manager, src_pages, dst_slots, **kwargs):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            return manager.send_kvcache_draft(
                "session",
                src_pages,
                [30_000, 40_000],
                dst_slots,
                src_mla_entry_count=0,
                dst_mla_entry_count=0,
                dst_attn_tp_size=kwargs.pop("dst_attn_tp_size", 1),
                dst_tp_rank=kwargs.pop("dst_tp_rank", 0),
                draft_total_kv_heads=kwargs.pop("draft_total_kv_heads", 1),
                dst_dcp_size=kwargs.pop("dst_dcp_size", 1),
                decode_prefix_len=0,
                executor=executor,
                **kwargs,
            )

    def _assert_single_run(self, manager, *, src_slot, dst_slot, num_tokens):
        """Every draft KV entry got one coalesced descriptor at the given slots.

        Layer futures complete in nondeterministic order, so compare the address
        SETS against both entry base pointers rather than indexing by position.
        """
        calls = manager.engine.batch_transfer_sync.call_args_list
        self.assertEqual(len(calls), 2)  # one future per draft KV entry
        src_seen, dst_seen = set(), set()
        for call in calls:
            _, src_addresses, dst_addresses, lengths = call.args
            self.assertEqual(len(src_addresses), 1)
            self.assertEqual(len(dst_addresses), 1)
            self.assertEqual(lengths, [num_tokens * 256])
            src_seen.add(src_addresses[0])
            dst_seen.add(dst_addresses[0])
        self.assertEqual(
            src_seen, {10_000 + src_slot * 256, 20_000 + src_slot * 256}
        )
        self.assertEqual(
            dst_seen, {30_000 + dst_slot * 256, 40_000 + dst_slot * 256}
        )

    def test_window_shorter_than_chunk_still_transfers(self):
        """A tail window narrower than the chunk must not collapse to nothing.

        Regression: the chunk extent was derived from ``min(len(dst), len(src))``
        instead of the chunk's real token span, so a 1000-token chunk was treated
        as ``[start, start+128)``. Intersecting that with a tail window
        ``[872, 1000)`` is empty, so the draft KV was silently never sent while
        the function still reported success.
        """
        manager = self._manager(page_size=1)
        status = self._send(
            manager,
            np.arange(2000, 3000, dtype=np.int32),
            np.arange(5000, 5128, dtype=np.int32),
            src_page_offset=0,
            num_kv_tokens=1000,
            draft_window_start=872,
        )
        self.assertEqual(status, 0)
        self._assert_single_run(manager, src_slot=2872, dst_slot=5000, num_tokens=128)

    def test_page_indices_expand_to_token_slots(self):
        """``prefill_kv_indices`` are PAGE ids; ``src_page_offset`` a PAGE offset.

        Both come from ``kv_to_page_indices`` in ``send_kv_chunk``, while the
        draft window (``num_kv_tokens`` / ``draft_window_start`` / the compact
        destinations) is token space. Treating the pages as token slots shrinks
        every source address by ``page_size`` and reads the wrong KV.
        """
        manager = self._manager(page_size=64)
        status = self._send(
            manager,
            np.arange(700, 710, dtype=np.int32),  # 10 pages = tokens [6400, 7040)
            np.arange(5000, 5128, dtype=np.int32),
            src_page_offset=100,
            num_kv_tokens=640,
            draft_window_start=6912,  # last 128 tokens: chunk offsets [512, 640)
        )
        self.assertEqual(status, 0)
        # chunk offset 512 -> page index 8 -> page id 708 -> token slot 708*64.
        self._assert_single_run(
            manager, src_slot=708 * 64, dst_slot=5000, num_tokens=128
        )

    def _assert_per_token_slice(
        self,
        manager,
        *,
        first_src_slot,
        first_dst_slot,
        num_tokens,
        src_token_item_len,
        dst_token_item_len,
        dst_head_slice_offset,
        slice_bytes,
    ):
        """A head slice cannot coalesce: one descriptor per token, strided.

        Layer futures complete in nondeterministic order, so compare the FIRST
        address of each call against both entry base pointers as a set.
        """
        calls = manager.engine.batch_transfer_sync.call_args_list
        self.assertEqual(len(calls), 2)  # one future per draft KV entry
        src_first, dst_first = set(), set()
        for call in calls:
            _, src_addresses, dst_addresses, lengths = call.args
            self.assertEqual(len(src_addresses), num_tokens)
            self.assertEqual(len(dst_addresses), num_tokens)
            self.assertEqual(lengths, [slice_bytes] * num_tokens)
            # Contiguous source slots stay contiguous; the destination strides by
            # the peer's (wider) per-token slot.
            self.assertEqual(
                src_addresses[1] - src_addresses[0], src_token_item_len
            )
            self.assertEqual(
                dst_addresses[1] - dst_addresses[0], dst_token_item_len
            )
            src_first.add(src_addresses[0])
            dst_first.add(dst_addresses[0])
        self.assertEqual(
            src_first,
            {
                base + first_src_slot * src_token_item_len
                for base in (10_000, 20_000)
            },
        )
        self.assertEqual(
            dst_first,
            {
                base + first_dst_slot * dst_token_item_len + dst_head_slice_offset
                for base in (30_000, 40_000)
            },
        )

    def test_plain_dp_decode_head_slice_lands_on_compact_window_slots(self):
        """Prefill attn_tp=8 -> decode DP16 (attn_tp=1), DCP off, compact window.

        The decode rank owns ALL 8 draft KV heads while this prefill rank owns
        one, so its slice must land at head offset 3 (= its attn-TP rank) inside
        the peer's wider per-token slot, at the COMPACT swa slot the decode side
        sent -- not at the target's page id. ``dst_dcp_size=1`` throughout: the
        window plan never consults it.
        """
        manager = self._manager(page_size=64, attn_tp_size=8, engine_rank=3)
        status = self._send(
            manager,
            np.arange(700, 710, dtype=np.int32),  # 10 pages = 640 tokens
            np.arange(5000, 5128, dtype=np.int32),  # 128 compact swa slots
            dst_attn_tp_size=1,
            dst_tp_rank=0,
            draft_total_kv_heads=8,
            dst_dcp_size=1,
            src_page_offset=0,
            num_kv_tokens=640,
            draft_window_start=512,  # tail 128 tokens of this chunk
        )
        self.assertEqual(status, 0)
        # chunk offset 512 -> page index 8 -> page id 708 -> token slot 708*64,
        # and 708*64+63+1 == 709*64, so the 128 source slots are contiguous.
        self._assert_per_token_slice(
            manager,
            first_src_slot=708 * 64,
            first_dst_slot=5000,
            num_tokens=128,
            src_token_item_len=256,  # this rank owns 1 of 8 heads
            dst_token_item_len=8 * 256,  # the decode rank owns all 8
            dst_head_slice_offset=3 * 256,  # attn-TP rank 3
            slice_bytes=256,
        )

    def test_plain_dp_decode_without_window_uses_full_locs(self):
        """Same topology, decode draft pool NOT window-compact.

        With no ``draft_window_start`` the whole chunk is sent at the target's
        full locs. ``dst_dcp_size=1`` must reduce the replicated plan to a plain
        page -> token expansion (no stride), while the head slicing still
        applies.
        """
        manager = self._manager(page_size=64, attn_tp_size=8, engine_rank=3)
        status = self._send(
            manager,
            np.arange(700, 710, dtype=np.int32),  # 10 pages = 640 tokens
            np.arange(900, 910, dtype=np.int32),  # 10 dst pages
            dst_attn_tp_size=1,
            dst_tp_rank=0,
            draft_total_kv_heads=8,
            dst_dcp_size=1,
            src_page_offset=0,
            num_kv_tokens=640,
        )
        self.assertEqual(status, 0)
        self._assert_per_token_slice(
            manager,
            first_src_slot=700 * 64,
            first_dst_slot=900 * 64,
            num_tokens=640,
            src_token_item_len=256,
            dst_token_item_len=8 * 256,
            dst_head_slice_offset=3 * 256,
            slice_bytes=256,
        )

    def test_legacy_peer_without_window_uses_replicated_plan(self):
        """No ``draft_window_start`` means the peer runs a full-size draft pool.

        That path must keep the DCP-replicated page->token expansion: all
        ``num_kv_tokens`` tokens of the chunk are sent, addressed by the target's
        virtual page ids.
        """
        manager = self._manager(page_size=64)
        status = self._send(
            manager,
            np.arange(700, 710, dtype=np.int32),  # 10 pages = 640 tokens
            np.arange(900, 910, dtype=np.int32),  # 10 virtual dst pages
            src_page_offset=0,
            num_kv_tokens=640,
        )
        self.assertEqual(status, 0)
        for call in manager.engine.batch_transfer_sync.call_args_list:
            self.assertEqual(sum(call.args[3]), 640 * 256)


if __name__ == "__main__":
    unittest.main()
