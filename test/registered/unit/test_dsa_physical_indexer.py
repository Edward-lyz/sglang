"""Regression tests for physical DSA indexer cache accounting and transfer.

Run with the matching SGLang runtime dependencies. These CPU tensor tests do
not load model weights or create an inference server.
"""

import os
import unittest
from contextlib import ExitStack
from types import SimpleNamespace as N
from unittest.mock import patch

import torch

from sglang.srt.configs.model_config import (
    dsa_layer_skips_topk,
    get_dsa_indexer_layer_pattern,
)
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, MLATokenToKVPool
from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator

FULL_LAYERS = [0, 1, 2, *range(6, 78, 4)]


class TestPhysicalIndexer(unittest.TestCase):
    def setUp(self):
        self.config = N(
            architectures=["GlmMoeDsaForCausalLM"],
            num_hidden_layers=78,
            index_head_dim=128,
            index_topk=2048,
            index_topk_freq=4,
            index_skip_topk_offset=3,
            indexer_types=["full" if i in FULL_LAYERS else "shared" for i in range(78)],
        )

    def configurator(
        self, *, compact=True, dtype="fp8_e4m3", start=0, layers=78, **flags
    ):
        args = N(
            kv_cache_dtype=dtype,
            enable_dsa_cache_layer_split=False,
            enable_hisparse=False,
            enable_hierarchical_cache=False,
        )
        for key, value in flags.items():
            setattr(args, key, value)
        kvc = N(
            model_config=N(
                hf_config=self.config,
                hf_text_config=self.config,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
            ),
            kv_cache_dtype=(
                torch.float8_e4m3fn if dtype == "fp8_e4m3" else torch.uint8
            ),
            server_args=args,
            layer_info=N(
                start_layer=start,
                end_layer=start + layers,
                num_effective_layers=layers,
            ),
            use_mla_backend=True,
            is_draft_worker=False,
            spec_aux_config=N(eagle_draft_num_layers=1),
            spec_algorithm=N(
                is_eagle=lambda: True,
                is_standalone=lambda: False,
                is_dflash_family=lambda: False,
            ),
        )
        with ExitStack() as stack:
            stack.enter_context(
                patch.dict(os.environ, {"SGLANG_INDEXER_CACHE_OPT": str(int(compact))})
            )
            stack.enter_context(
                patch(
                    "sglang.srt.model_executor.pool_configurator.mambaish_config",
                    return_value=None,
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.model_executor.pool_configurator.get_parallel",
                    return_value=N(attn_tp_size=1),
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.layers.cp.utils.get_glm_dsa_layer_split_effective_num_layers",
                    side_effect=lambda _, count: count,
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.mem_cache.memory_pool.use_dsa_fp4_indexer",
                    return_value=False,
                )
            )
            return DefaultPoolConfigurator(kvc)

    def test_glm52_producers_match_forward_topk_ownership(self):
        pattern = get_dsa_indexer_layer_pattern(self.config)
        self.assertEqual(
            [i for i, item in enumerate(pattern) if item == "F"], FULL_LAYERS
        )
        self.assertEqual(
            [item == "S" for item in pattern],
            [dsa_layer_skips_topk(self.config, i) for i in range(78)],
        )

    def test_explicit_pattern_fallback(self):
        self.config.indexer_types = None
        self.config.index_topk_pattern = ["F", "S", "F"]
        self.config.num_hidden_layers = 3
        self.assertEqual(get_dsa_indexer_layer_pattern(self.config), ["F", "S", "F"])

    def test_frequency_fallback(self):
        self.config.indexer_types = None
        self.assertEqual(
            get_dsa_indexer_layer_pattern(self.config),
            ["F" if i in FULL_LAYERS else "S" for i in range(78)],
        )
        self.config.index_skip_topk_offset = None
        self.assertEqual(
            get_dsa_indexer_layer_pattern(self.config)[:7],
            ["F", "F", "S", "S", "S", "F", "S"],
        )

    def test_fp8_budget_preserves_existing_proportional_draft_accounting(self):
        self.assertEqual(self.configurator(compact=False)._cell_size, 55932)
        self.assertEqual(self.configurator()._cell_size, 48311)
        self.assertEqual(self.configurator(dtype="mxfp4")._cell_size, 34407)

    def test_fp8_capacity_for_same_memory_budget(self):
        budget = 1733312 * 55932
        before = self.configurator(compact=False).calculate_pool_sizes(budget, 64)
        after = self.configurator().calculate_pool_sizes(budget, 64)
        self.assertGreater(after.max_total_num_tokens, before.max_total_num_tokens)
        self.assertEqual(before.max_total_num_tokens // 61440, 28)
        self.assertEqual(after.max_total_num_tokens // 61440, 32)
        self.assertLessEqual(after.max_total_num_tokens * 48311, budget)

    def test_incompatible_cache_modes_keep_full_accounting(self):
        for flag in (
            "enable_dsa_cache_layer_split",
            "enable_hisparse",
            "enable_hierarchical_cache",
        ):
            with self.subTest(flag=flag):
                self.assertEqual(self.configurator(**{flag: True})._cell_size, 55932)

    def test_partial_pipeline_slice_counts_local_producers(self):
        # Accounting only: this does not validate PP forward boundary handling.
        self.assertEqual(
            self.configurator(start=6, layers=4)._cell_size,
            int((4 * 576 + 132) * 5 / 4),
        )

    def test_incomplete_pattern_falls_back_to_full_accounting(self):
        self.config.indexer_types = ["full"]
        self.assertEqual(self.configurator()._cell_size, 55932)

    def pool(self):
        pool = DSATokenToKVPool.__new__(DSATokenToKVPool)
        pool.layer_num = 78
        pool.indexer_layer_num = 21
        pool.start_layer = 0
        pool.page_size = 64
        pool.cpu_offloading_chunk_size = 64
        pool.indexer_layer_mapping = {
            layer: index for index, layer in enumerate(FULL_LAYERS)
        }
        pool.index_k_with_scale_buffer = [
            torch.full((4, 8448), index + 1, dtype=torch.uint8) for index in range(21)
        ]
        return pool

    def test_pd_metadata_contains_only_physical_buffers(self):
        pool = self.pool()
        ptrs, sizes, items = pool.get_state_buf_infos()
        self.assertEqual(len(ptrs), 21)
        self.assertEqual(len(set(ptrs)), 21)
        self.assertEqual(sizes, [4 * 8448] * 21)
        self.assertEqual(items, [8448] * 21)
        self.assertIs(
            pool._get_index_buf_by_layer(74), pool.index_k_with_scale_buffer[20]
        )
        with self.assertRaises(ValueError):
            pool._get_index_buf_by_layer(75)

    def test_retract_restore_all_physical_layers_and_preserve_other_pages(self):
        pool = self.pool()
        indices = torch.arange(64, 128)
        expected = [tensor.clone() for tensor in pool.index_k_with_scale_buffer]
        with (
            patch.object(MLATokenToKVPool, "get_cpu_copy", return_value=[]),
            patch.object(MLATokenToKVPool, "load_cpu_copy"),
            patch("torch.cuda.synchronize"),
        ):
            saved = pool.get_cpu_copy(indices)
            self.assertEqual(len(saved["index_k"]), 21)
            saved["index_k"] = [
                [chunk.clone() for chunk in chunks] for chunks in saved["index_k"]
            ]
            for tensor in pool.index_k_with_scale_buffer:
                tensor[1].zero_()
            pool.load_cpu_copy(saved, indices)
        for actual, original in zip(pool.index_k_with_scale_buffer, expected):
            self.assertTrue(torch.equal(actual, original))


if __name__ == "__main__":
    unittest.main(verbosity=2)
