"""CPU tests for page-level KV trace collection."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.srt.managers.trace_manager import TraceManager

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class FakeTokenKVPool:
    kv_trace_layout = "mha_gqa"

    def __init__(self, page_size=4, layers=1, rows=12):
        self.page_size = page_size
        self.start_layer = 0
        self.layer_num = layers
        self.k = [torch.arange(rows * 2, dtype=torch.uint8).reshape(rows, 2) + i for i in range(layers)]
        self.v = [torch.arange(rows * 3, dtype=torch.uint8).reshape(rows, 3) + 10 + i for i in range(layers)]

    def get_kv_buffer(self, layer_id):
        return self.k[layer_id], self.v[layer_id]


class FakeSWA(FakeTokenKVPool):
    kv_trace_layout = "swa"

    def __init__(self):
        super().__init__(page_size=4, layers=2)
        self.full_kv_pool = FakeTokenKVPool(page_size=4, layers=1)
        self.swa_kv_pool = FakeTokenKVPool(page_size=2, layers=1)
        self.layers_mapping = {0: (0, False), 1: (0, True)}
        self.full_to_swa_index_mapping = torch.tensor(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        )

    def translate_loc_from_full_to_swa(self, indices):
        return self.full_to_swa_index_mapping[indices]


class FakeDeepSeekV4:
    kv_trace_layout = "deepseek_v4"

    def __init__(self):
        self.page_size = 4
        self.swa_page_size = 4
        self.start_layer = 0
        self.layer_num = 1
        self.swa_kv_pool = SimpleNamespace(
            page_size=4,
            kv_buffer=[torch.arange(12, dtype=torch.uint8).reshape(3, 4)],
        )
        self.c4_kv_pool = SimpleNamespace(
            page_size=1,
            kv_buffer=[torch.arange(6, dtype=torch.uint8).reshape(3, 2)],
        )
        self.c4_indexer_kv_pool = SimpleNamespace(page_size=1)
        self.index_buffer = [torch.arange(9, dtype=torch.uint8).reshape(3, 3)]
        self.layer_mapping = [
            SimpleNamespace(
                compress_ratio=4,
                compress_layer_id=0,
                compress_kv_pool=self.c4_kv_pool,
            )
        ]

    def translate_loc_from_full_to_swa(self, indices):
        return indices

    def get_index_k_page_size(self):
        return 1

    def get_index_k_with_scale_buffer(self, layer_id):
        return self.index_buffer[0]


class FakeDSA(FakeTokenKVPool):
    kv_trace_layout = "dsa"

    def __init__(self):
        super().__init__(page_size=4, layers=1, rows=8)
        self.index_k_with_scale_buffer = [
            torch.arange(6, dtype=torch.uint8).reshape(2, 3)
        ]

    def get_index_k_with_scale_buffer(self, layer_id):
        mapping = getattr(self, "indexer_layer_mapping", None)
        return self.index_k_with_scale_buffer[mapping[layer_id] if mapping else layer_id]


class FakeMLA(FakeTokenKVPool):
    kv_trace_layout = "mla"
    kv_lora_rank = 16


class TestTraceManagerKV(unittest.TestCase):
    def setUp(self):
        self.manager = TraceManager(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            num_hidden_layers=2,
        )

    def test_mha_pages_include_partial_page(self):
        pool = FakeTokenKVPool(page_size=4, rows=8)
        indices = torch.tensor([0, 1, 4, 5, 7], dtype=torch.int32)
        result = self.manager._collect_kv("r1", indices, pool, "mha", [0])

        self.assertEqual(result["layout"], "mha_gqa")
        self.assertEqual(result["total_tokens"], 5)
        self.assertEqual(result["logical_page_count"], 2)
        self.assertNotIn("match_len", result)
        self.assertNotIn("num_covered_pages", result)
        k_pages = result["layers"][0]["sources"][0]["pages"]
        self.assertEqual([p["physical_page_ids"] for p in k_pages], [[0], [1]])
        expected = hashlib.md5(pool.k[0][0:4].numpy().tobytes()).hexdigest()
        self.assertEqual(k_pages[0]["md5"], expected)
        self.assertEqual(k_pages[1]["source_token_count"], 3)

    def test_dsa_compact_indexer_keeps_latent_only_layers(self):
        pool = FakeDSA()
        pool.layer_num = 2
        pool.k.append(pool.k[0] + 9)
        pool.v.append(pool.v[0] + 9)
        pool.indexer_layer_mapping = {0: 0}
        result = self.manager._collect_kv(
            "r-dsa-compact", torch.tensor([0, 1, 2, 3]), pool, "dsa", [0, 1]
        )
        self.assertEqual(result["collection_status"], "ok")
        self.assertEqual(result["warnings"], [])
        self.assertEqual([layer["layer_id"] for layer in result["layers"]], [0, 1])
        self.assertEqual(
            [source["name"] for source in result["layers"][1]["sources"]], ["k"]
        )
        self.assertEqual(
            result["layers"][1]["sources"][0]["pages"][0]["md5"],
            hashlib.md5(pool.k[1][:4].numpy().tobytes()).hexdigest(),
        )

    def test_fp8_pages_hash_raw_storage_bytes(self):
        page = torch.tensor(
            [[1.0, -2.0], [3.0, -4.0]], dtype=torch.float8_e4m3fn
        )

        digest = self.manager._page_md5(page)

        self.assertEqual(
            digest,
            hashlib.md5(page.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
        )

    def test_swa_layer_translates_slots(self):
        pool = FakeSWA()
        result = self.manager._collect_kv(
            "r2", torch.tensor([0, 2, 4]), pool, "swa", layer_ids=[0, 1]
        )

        self.assertEqual(result["layout"], "swa")
        self.assertEqual(result["layers"][0]["pool"], "full")
        self.assertEqual(result["layers"][1]["pool"], "swa")
        swa_page = result["layers"][1]["sources"][0]["pages"][0]
        self.assertEqual(swa_page["physical_page_ids"], [1])
        self.assertEqual(swa_page["source_token_count"], 1)

    def test_swa_unmapped_slots_are_warnings(self):
        pool = FakeSWA()
        result = self.manager._collect_kv(
            "r-swa-warning",
            torch.tensor([0, 1, 2], dtype=torch.int32),
            pool,
            "swa",
            layer_ids=[0, 1],
        )

        self.assertEqual(len(result["warnings"]), 2)
        self.assertTrue(
            all(warning["code"] == "unmapped_token_slot" for warning in result["warnings"])
        )
        self.assertEqual(
            {warning["source"] for warning in result["warnings"]}, {"k", "v"}
        )
        self.assertEqual(result["collection_status"], "partial")

    def test_explicit_layers_limit_collection(self):
        pool = FakeTokenKVPool(page_size=4, layers=2)
        result = self.manager._collect_kv(
            "r-selected",
            torch.tensor([0, 1, 4], dtype=torch.int32),
            pool,
            "mha",
            layer_ids=[1],
        )

        self.assertEqual([layer["layer_id"] for layer in result["layers"]], [1])

    def test_trace_layer_selector_defaults_to_global_zero(self):
        pool = FakeTokenKVPool(page_size=4, layers=2)

        self.assertEqual(
            self.manager._get_kv_trace_layers(SimpleNamespace(trace_config={}), pool),
            [0],
        )

    def test_trace_layer_selector_filters_to_local_pp_layers(self):
        pool = FakeTokenKVPool(page_size=4, layers=2)
        pool.start_layer = 1

        self.assertEqual(
            self.manager._get_kv_trace_layers(
                SimpleNamespace(trace_config={"kv": {"layers": [0, 1, 2, 2]}}),
                pool,
            ),
            [1, 2],
        )

    def test_mla_only_collects_compressed_kv(self):
        result = self.manager._collect_kv(
            "r-mla", torch.tensor([0, 1, 2, 3]), FakeMLA(), "mla", [0]
        )

        self.assertEqual(
            [source["name"] for source in result["layers"][0]["sources"]], ["k"]
        )

    def test_dsa_collects_mla_kv_and_indexer_without_mha_value(self):
        result = self.manager._collect_kv(
            "r-dsa", torch.tensor([0, 1, 2, 3]), FakeDSA(), "dsa", [0]
        )

        self.assertEqual(
            [source["name"] for source in result["layers"][0]["sources"]],
            ["k", "indexer_k"],
        )

    def test_trace_kv_defaults_to_layer_zero(self):
        pool = FakeTokenKVPool(page_size=4, layers=2)
        self.manager.token_to_kv_pool_allocator = SimpleNamespace(
            get_kvcache=lambda: pool
        )
        self.manager.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
        )
        req = SimpleNamespace(
            req_pool_idx=0,
            origin_input_ids=[11, 12],
            output_ids=[],
            prefix_indices=torch.tensor([], dtype=torch.int32),
            trace_config={"enable_trace": True, "trace_type": "kv"},
        )
        recv = SimpleNamespace(
            rid="r-default",
            unique_trace_id="u-default",
            return_online=True,
            dump_dir=tempfile.mkdtemp(),
            trace_type="kv",
        )

        self.manager._trace_kv(recv, req)

        self.assertEqual(
            [layer["layer_id"] for layer in req.trace_info["kv_match_md5"]["layers"]],
            [0],
        )

    def test_pd_trace_filter(self):
        req = SimpleNamespace(trace_config={"pd": "p"})
        self.assertTrue(self.manager._should_trace(req))

        self.manager.disaggregation_mode = "decode"
        self.assertFalse(self.manager._should_trace(req))
        self.assertTrue(
            self.manager._should_trace(SimpleNamespace(trace_config={"pd": "d"}))
        )
        self.assertTrue(
            self.manager._should_trace(SimpleNamespace(trace_config={}))
        )

    def test_rank_metadata_includes_disaggregation_mode(self):
        self.manager.disaggregation_mode = "prefill"
        self.assertEqual(
            self.manager._rank_metadata()["disaggregation_mode"], "prefill"
        )

    def test_deepseek_v4_reports_unsupported_layout(self):
        pool = FakeDeepSeekV4()
        indices = torch.tensor([0, 1, 2, 3, 4, 7], dtype=torch.int32)
        result = self.manager._collect_kv("r3", indices, pool, "deepseek-v4", [0])

        self.assertEqual(result["layout"], "deepseek_v4")
        self.assertEqual(result["layers"], [])
        self.assertEqual(result["collection_status"], "partial")
        self.assertEqual(result["warnings"][0]["code"], "unsupported_kv_layout")

    def test_dump_is_json_only(self):
        pool = FakeTokenKVPool(page_size=4)
        self.manager.token_to_kv_pool_allocator = SimpleNamespace(get_kvcache=lambda: pool)
        self.manager.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
        )
        req = SimpleNamespace(
            req_pool_idx=0,
            origin_input_ids=[11, 12],
            output_ids=[13, 14],
            prefix_indices=torch.tensor([], dtype=torch.int32),
            trace_config={"model_identity": "fake"},
        )
        recv = SimpleNamespace(
            rid="r4",
            unique_trace_id="u4",
            return_online=False,
            dump_dir=tempfile.mkdtemp(),
            trace_type="kv",
        )
        self.manager._trace_kv(recv, req)
        files = list(Path(recv.dump_dir).iterdir())
        self.assertEqual([path.suffix for path in files], [".json"])
        self.assertIn("kv_match_md5", files[0].read_text())


if __name__ == "__main__":
    unittest.main()
