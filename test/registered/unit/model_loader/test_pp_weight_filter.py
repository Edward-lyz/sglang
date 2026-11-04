"""
Unit tests for pipeline-parallel weight-load filtering.

Verifies that a PP rank only reads the checkpoint shards / tensors holding the
decoder layers it owns, and that the filter is disabled in the cases where it
would drop weights the rank actually needs.
"""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import safetensors.torch
import torch

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.environ import envs
from sglang.srt.model_loader.loader import DefaultModelLoader, _resolve_pp_layer_range
from sglang.srt.model_loader.weight_utils import (
    buffered_multi_thread_safetensors_weights_iterator,
    filter_safetensors_files_by_name,
    make_pp_layer_name_filter,
    multi_thread_safetensors_weights_iterator,
    safetensors_weights_iterator,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

INDEX_FILE = "model.safetensors.index.json"


class TestPPLayerNameFilter(CustomTestCase):
    def setUp(self):
        self.keep = make_pp_layer_name_filter(
            start_layer=6, end_layer=12, num_hidden_layers=93
        )

    def test_owned_decoder_layers_are_kept(self):
        self.assertTrue(self.keep("model.layers.6.self_attn.q_proj.weight"))
        self.assertTrue(self.keep("model.layers.11.mlp.experts.3.w1.weight"))
        self.assertTrue(self.keep("language_model.model.layers.7.mlp.gate.weight"))

    def test_unowned_decoder_layers_are_dropped(self):
        self.assertFalse(self.keep("model.layers.5.self_attn.q_proj.weight"))
        self.assertFalse(self.keep("model.layers.12.mlp.gate.weight"))
        self.assertFalse(self.keep("language_model.model.layers.3.mlp.gate.weight"))

    def test_non_decoder_weights_are_kept(self):
        # No layer index at all.
        self.assertTrue(self.keep("model.embed_tokens.weight"))
        self.assertTrue(self.keep("model.norm.weight"))
        self.assertTrue(self.keep("lm_head.weight"))
        # Separate layer-index namespaces must not be filtered.
        self.assertTrue(self.keep("model.mtp.layers.0.eh_proj.weight"))
        self.assertTrue(self.keep("vision_tower.encoder.layers.3.attn.qkv.weight"))
        # Beyond the instantiated stack: left to the model to skip.
        self.assertTrue(self.keep("model.layers.95.mlp.gate.weight"))


class TestPPWeightFileFilter(CustomTestCase):
    def _make_index(self, tmpdir, layers_per_shard=6, num_layers=93, num_shards=16):
        weight_map = {
            f"model.layers.{lid}.w": f"shard-{lid // layers_per_shard:02d}.safetensors"
            for lid in range(num_layers)
        }
        weight_map["model.embed_tokens.weight"] = "shard-00.safetensors"
        weight_map["lm_head.weight"] = f"shard-{num_shards - 1:02d}.safetensors"
        with open(os.path.join(tmpdir, INDEX_FILE), "w") as f:
            json.dump({"weight_map": weight_map}, f)
        return [os.path.join(tmpdir, name) for name in sorted(set(weight_map.values()))]

    def test_only_shards_with_owned_weights_are_kept(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = self._make_index(tmpdir)
            keep = make_pp_layer_name_filter(6, 12, 93)
            kept = filter_safetensors_files_by_name(files, tmpdir, INDEX_FILE, keep)
            self.assertEqual(
                [os.path.basename(f) for f in kept],
                # shard-01 holds layers 6-11; shard-00 / shard-15 are kept for
                # embed_tokens / lm_head.
                [
                    "shard-00.safetensors",
                    "shard-01.safetensors",
                    "shard-15.safetensors",
                ],
            )

    def test_unindexed_shard_is_kept(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = self._make_index(tmpdir)
            extra = os.path.join(tmpdir, "mtp.safetensors")
            keep = make_pp_layer_name_filter(6, 12, 93)
            kept = filter_safetensors_files_by_name(
                files + [extra], tmpdir, INDEX_FILE, keep
            )
            self.assertIn(extra, kept)

    def test_missing_index_falls_back_to_all_shards(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = [os.path.join(tmpdir, "model-00001-of-00002.safetensors")]
            keep = make_pp_layer_name_filter(6, 12, 93)
            self.assertEqual(
                filter_safetensors_files_by_name(files, tmpdir, INDEX_FILE, keep),
                files,
            )

    def test_empty_index_falls_back_to_all_shards(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, INDEX_FILE), "w") as f:
                json.dump({"weight_map": {}}, f)
            files = [os.path.join(tmpdir, "model-00001-of-00002.safetensors")]
            keep = make_pp_layer_name_filter(6, 12, 93)
            self.assertEqual(
                filter_safetensors_files_by_name(files, tmpdir, INDEX_FILE, keep),
                files,
            )


class TestPPWeightIteratorFilter(CustomTestCase):
    def _make_shard(self, tmpdir):
        path = os.path.join(tmpdir, "model-00001-of-00001.safetensors")
        safetensors.torch.save_file(
            {
                "model.layers.5.w": torch.ones(2),
                "model.layers.7.w": torch.ones(2) * 2,
                "lm_head.weight": torch.ones(2) * 3,
            },
            path,
        )
        return path

    def test_iterators_yield_only_owned_weights(self):
        keep = make_pp_layer_name_filter(6, 12, 93)
        expected = ["lm_head.weight", "model.layers.7.w"]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_shard(tmpdir)
            iterators = [
                safetensors_weights_iterator([path], name_filter=keep),
                safetensors_weights_iterator(
                    [path], disable_mmap=True, name_filter=keep
                ),
                multi_thread_safetensors_weights_iterator(
                    [path], max_workers=2, name_filter=keep
                ),
                buffered_multi_thread_safetensors_weights_iterator(
                    [path], max_workers=2, name_filter=keep
                ),
                buffered_multi_thread_safetensors_weights_iterator(
                    [path], max_workers=2, disable_mmap=True, name_filter=keep
                ),
            ]
            for iterator in iterators:
                names = sorted(name for name, _ in iterator)
                self.assertEqual(names, expected)

    def test_no_filter_yields_everything(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_shard(tmpdir)
            self.assertEqual(len(list(safetensors_weights_iterator([path]))), 3)


class TestPPFilterGating(CustomTestCase):
    def _source(self, pp_layer_range, num_hidden_layers=93):
        return DefaultModelLoader.Source(
            model_or_path="/does/not/exist",
            revision=None,
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(num_hidden_layers=num_hidden_layers)
            ),
            pp_layer_range=pp_layer_range,
        )

    def _loader(self, draft_model_idx=None):
        return DefaultModelLoader(
            LoadConfig(model_loader_extra_config={}, draft_model_idx=draft_model_idx)
        )

    def test_filter_built_for_pp_sharded_source(self):
        loader = self._loader()
        self.assertIsNotNone(loader._maybe_build_pp_name_filter(self._source((6, 12))))

    def test_no_filter_without_pp_sharding(self):
        loader = self._loader()
        self.assertIsNone(loader._maybe_build_pp_name_filter(self._source(None)))
        # Rank owns the whole stack.
        self.assertIsNone(loader._maybe_build_pp_name_filter(self._source((0, 93))))

    def test_no_filter_for_draft_model(self):
        loader = self._loader(draft_model_idx=0)
        self.assertIsNone(loader._maybe_build_pp_name_filter(self._source((6, 12))))

    def test_out_of_range_layer_span_raises(self):
        loader = self._loader()
        for bad_range in ((-5, 100), (6, 6), (12, 6), (6, 94)):
            with self.assertRaises(ValueError):
                loader._maybe_build_pp_name_filter(self._source(bad_range))

    def test_layer_range_resolution_requires_env_flag(self):
        model = SimpleNamespace(start_layer=6, end_layer=12)
        self.assertIsNone(_resolve_pp_layer_range(model))
        with envs.SGLANG_ENABLE_PP_WEIGHT_FILTER.override(True):
            self.assertEqual(_resolve_pp_layer_range(model), (6, 12))
            # Range exposed on the inner decoder module.
            self.assertEqual(
                _resolve_pp_layer_range(SimpleNamespace(model=model)), (6, 12)
            )
            self.assertIsNone(_resolve_pp_layer_range(SimpleNamespace()))


if __name__ == "__main__":
    unittest.main()
