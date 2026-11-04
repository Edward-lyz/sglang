"""Check snapshot eligibility before publishing KDA checkpoint metadata."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.overrides import _mamba_radix_cache_resolution
from sglang.srt.configs.hybrid_arch import uses_kda_attentionstore_checkpoints
from sglang.srt.configs.kimi_linear import KimiLinearConfig
from sglang.srt.layers.attention import hybrid_linear_attn_backend as hybrid_backend
from sglang.srt.managers import schedule_batch
from sglang.srt.managers.schedule_batch import (
    ScheduleBatch,
    _latest_checkpoint_in_extend,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def model_config(arch="KimiLinearForCausalLM"):
    config = object.__new__(KimiLinearConfig)
    config.architectures = [arch]
    return SimpleNamespace(hf_config=config, linear_attn_registry_result=None)


class TestKDAExactCheckpointSafety(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            enable_asradix_cache=True,
            disable_radix_cache=False,
            mamba_cache_chunk_size=64,
            enable_mamba_extra_buffer_lazy=lambda: False,
        )
        self.batch = SimpleNamespace(
            model_config=model_config(),
            req_to_token_pool=SimpleNamespace(
                mamba_pool=SimpleNamespace(
                    mamba_cache=SimpleNamespace(conv=[torch.empty(1, 2, 3, 1)])
                ),
                get_mamba_ping_pong_other_idx=lambda index: 1 - index,
            ),
        )
        self.backend = object.__new__(hybrid_backend.MambaAttnBackendBase)
        self.backend.device = "cpu"
        self.backend.checkpoint_chunk_size = 256
        self.backend.intermediate_state_chunk_size = 64
        self.backend.exact_checkpoint_tracking = True
        self.backend.conv_states_shape = (1, 3)
        for module in (schedule_batch, hybrid_backend):
            patcher = patch.object(module, "get_server_args", return_value=self.args)
            patcher.start()
            self.addCleanup(patcher.stop)

    def prepare(self, prefix, extend, branching=None):
        req = SimpleNamespace(
            prefix_indices=range(prefix),
            extend_range=SimpleNamespace(length=extend),
            mamba_ping_pong_track_buffer=torch.tensor([20, 21]),
            mamba_next_track_idx=0,
            mamba_last_track_seqlen=128,
            mamba_branching_seqlen=branching,
        )
        entry = ScheduleBatch._mamba_radix_cache_v2_req_prepare_for_extend(
            self.batch, req
        )
        return req, entry

    def metadata(self, prefixes, extends, entries):
        return SimpleNamespace(
            extend_prefix_lens=torch.tensor(prefixes),
            extend_seq_lens=torch.tensor(extends),
            mamba_track_seqlens=torch.tensor([e.track_seqlen for e in entries]),
            mamba_track_mask=torch.tensor([e.track_mask for e in entries]),
            mamba_track_indices=torch.tensor([e.track_index for e in entries]),
        )

    def test_unavailable_snapshots_do_not_mutate_request_tracking(self):
        for prefix, extend in ((160, 160), (255, 1), (254, 2)):
            with self.subTest(prefix=prefix, extend=extend):
                req, entry = self.prepare(prefix, extend)
                self.assertFalse(entry.track_mask)
                self.assertEqual(entry.track_seqlen, -1)
                self.assertEqual(req.mamba_next_track_idx, 0)
                self.assertEqual(req.mamba_last_track_seqlen, 128)

    def test_non_aligned_request_does_not_fail_the_other_batch_rows(self):
        _, good = self.prepare(0, 300)
        _, skipped = self.prepare(160, 160)
        forward = self.metadata([0, 160], [300, 160], [good, skipped])
        h_src, h_dst, final_src, _ = self.backend._init_track_ssm_indices(
            torch.tensor([10, 11]), forward
        )
        self.assertEqual(h_src.tolist(), [4])
        self.assertEqual(h_dst.tolist(), [20])
        self.assertEqual(final_src.tolist(), [])
        self.assertEqual(
            self.backend._init_track_conv_indices(
                torch.tensor([0, 300, 460]), forward
            ).tolist(),
            [[253, 254, 255]],
        )

    def test_short_conv_window_is_skipped_in_a_packed_batch(self):
        _, good = self.prepare(0, 256)
        _, skipped = self.prepare(255, 1)
        forward = self.metadata([0, 255], [256, 1], [good, skipped])
        indices = self.backend._init_track_conv_indices(
            torch.tensor([0, 256, 257]), forward
        )
        self.assertEqual(indices.tolist(), [[253, 254, 255]])
        _, _, final_src, final_dst = self.backend._init_track_ssm_indices(
            torch.tensor([10, 11]), forward
        )
        self.assertEqual(final_src.tolist(), [10])
        self.assertEqual(final_dst.tolist(), [20])

    def test_backend_rejects_a_window_crossing_request_boundaries(self):
        # Defensive check for malformed callers; the scheduler skips this row.
        for start in (0, 64):
            forward = SimpleNamespace(
                extend_prefix_lens=torch.tensor([255]),
                extend_seq_lens=torch.tensor([1]),
                mamba_track_seqlens=torch.tensor([256]),
                mamba_track_mask=torch.tensor([True]),
            )
            with self.assertRaisesRegex(ValueError, "inside its own extend"):
                self.backend._init_track_conv_indices(
                    torch.tensor([start, start + 1]), forward
                )

    def test_valid_final_state_with_non_aligned_prefix_is_kept(self):
        req, entry = self.prepare(160, 96)
        self.assertTrue(entry.track_mask)
        self.assertEqual(req.mamba_last_track_seqlen, 256)
        self.assertEqual(req.mamba_next_track_idx, 1)
        forward = self.metadata([160], [96], [entry])
        self.assertEqual(
            [
                t.tolist()
                for t in self.backend._init_track_ssm_indices(
                    torch.tensor([10]), forward
                )
            ],
            [[], [], [10], [20]],
        )

    def test_unsafe_branch_does_not_replace_a_valid_final_snapshot(self):
        req, entry = self.prepare(255, 257, branching=256)
        self.assertEqual(entry.track_seqlen, 512)
        self.assertEqual(req.mamba_last_track_seqlen, 512)

    def test_valid_branch_still_takes_precedence(self):
        req, entry = self.prepare(64, 600, branching=256)
        self.assertEqual(entry.track_seqlen, 256)
        self.assertEqual(req.mamba_last_track_seqlen, 256)

    def test_conv_window_size_comes_from_the_pool(self):
        self.batch.req_to_token_pool.mamba_pool.mamba_cache.conv = [
            torch.empty(1, 2, 5, 1)
        ]
        _, entry = self.prepare(252, 4)
        self.assertFalse(entry.track_mask)

    def test_lazy_skip_does_not_publish_a_checkpoint(self):
        self.args.enable_mamba_extra_buffer_lazy = lambda: True
        req, entry = self.prepare(160, 160)
        self.assertFalse(entry.track_mask)
        self.assertEqual(req.mamba_last_track_seqlen, 128)
        self.assertEqual(req.mamba_next_track_idx, 0)

    def test_chunk_sizes_24k_and_36k_keep_aligned_snapshots(self):
        for size in (24 * 1024, 36 * 1024):
            for prefix in range(0, 512, 64):
                req, entry = self.prepare(prefix, size)
                self.assertTrue(entry.track_mask)
                self.assertEqual(
                    req.mamba_last_track_seqlen, (prefix + size) // 256 * 256
                )

    def test_prefix_and_extend_boundary_matrix(self):
        for prefix in range(512):
            for extend in (1, 2, 3, 63, 64, 65, 255, 256, 257, 512):
                req, entry = self.prepare(prefix, extend)
                # Materialized SSM states: each internal 64-token boundary,
                # plus the final state. Conv needs three preceding input rows.
                states = set(range(prefix + 64, prefix + extend, 64))
                states.add(prefix + extend)
                snapshots = {end for end in states if end - 3 >= prefix}
                target = _latest_checkpoint_in_extend(prefix, extend, 256)
                self.assertEqual(entry.track_mask, target in snapshots)
                self.assertEqual(
                    req.mamba_last_track_seqlen, target if entry.track_mask else 128
                )

    def test_scheduler_alias_uses_exact_semantics(self):
        self.batch.model_config = model_config("AliasForKDA")
        req, entry = self.prepare(0, 300)
        self.assertEqual(entry.track_seqlen, 256)
        self.assertEqual(req.mamba_last_track_seqlen, 256)

    def test_legacy_non_kda_tracking_is_unchanged(self):
        self.batch.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["Qwen3NextForCausalLM"]),
            linear_attn_registry_result=None,
        )
        req, entry = self.prepare(64, 300, branching=128)
        self.assertTrue(entry.track_mask)
        self.assertEqual(entry.track_seqlen, 129)
        self.assertEqual(req.mamba_last_track_seqlen, 128)

    def test_shared_gate_handles_text_multimodal_alias_and_registered_kda(self):
        config = model_config("AliasForKDA")
        self.assertTrue(uses_kda_attentionstore_checkpoints(self.args, config))
        config.hf_config = SimpleNamespace(
            text_config=config.hf_config, architectures=["MultimodalAlias"]
        )
        self.assertTrue(uses_kda_attentionstore_checkpoints(self.args, config))
        config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["RegisteredKDA"]),
            linear_attn_registry_result=(
                SimpleNamespace(
                    backend_class_name="sglang.srt.layers.attention.linear.kda_backend.KDAAttnBackend"
                ),
                object(),
            ),
        )
        self.assertTrue(uses_kda_attentionstore_checkpoints(self.args, config))
        self.args.disable_radix_cache = True
        self.assertFalse(uses_kda_attentionstore_checkpoints(self.args, config))
        self.args.disable_radix_cache = False
        self.args.enable_asradix_cache = False
        self.assertFalse(uses_kda_attentionstore_checkpoints(self.args, config))

    def test_resolution_requires_buffers_for_kda_config(self):
        self.args.get_model_config = lambda: model_config("AliasForKDA")
        self.args.mamba_radix_cache_strategy = "auto"
        self.assertEqual(
            _mamba_radix_cache_resolution(self.args)["mamba_radix_cache_strategy"],
            "extra_buffer",
        )
        self.args.mamba_radix_cache_strategy = "no_buffer"
        with self.assertRaisesRegex(ValueError, "requires mamba_radix_cache_strategy"):
            _mamba_radix_cache_resolution(self.args)


if __name__ == "__main__":
    unittest.main()
