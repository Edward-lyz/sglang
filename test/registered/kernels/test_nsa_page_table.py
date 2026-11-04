import unittest

import torch

from sglang.srt.layers.attention.nsa.page_table import (
    gather_block_table_into,
    gather_page_table_into,
)
from sglang.srt.layers.attention.nsa.transform_index import (
    transform_index_page_table_decode_fast,
    transform_index_page_table_decode_ref,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=1, suite="stage-b-test-small-1-gpu")


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestNSAPageTable(unittest.TestCase):
    def test_gather_page_and_block_tables(self):
        page_size = 64
        sentinel = -12345

        for max_len in (1, 63, 64, 65, 438272, 516096):
            aligned_len = ((max_len + page_size - 1) // page_size) * page_size
            cols = torch.arange(max_len, dtype=torch.int32, device="cuda")
            req_to_token = torch.stack(
                [cols + row * aligned_len for row in range(4)]
            )
            req_pool_indices = torch.tensor(
                [2, 0], dtype=torch.int32, device="cuda"
            )
            source = req_to_token.index_select(0, req_pool_indices.long())

            for repeats in (1, 4):
                with self.subTest(max_len=max_len, repeats=repeats):
                    expected = torch.repeat_interleave(source, repeats, dim=0)
                    output = torch.full(
                        (expected.shape[0] + 1, max_len + 3),
                        sentinel,
                        dtype=torch.int32,
                        device="cuda",
                    )
                    actual = gather_page_table_into(
                        req_to_token,
                        req_pool_indices,
                        output,
                        max_len,
                        repeats=repeats,
                    )
                    self.assertEqual(actual.data_ptr(), output.data_ptr())
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertTrue(
                        torch.all(output[expected.shape[0] :] == sentinel).item()
                    )
                    self.assertTrue(
                        torch.all(output[: expected.shape[0], max_len:] == sentinel).item()
                    )

                    expected_blocks = expected[:, ::page_size] // page_size
                    block_output = torch.full(
                        (expected.shape[0] + 1, expected_blocks.shape[1] + 2),
                        sentinel,
                        dtype=torch.int32,
                        device="cuda",
                    )
                    actual_blocks = gather_block_table_into(
                        req_to_token,
                        req_pool_indices,
                        block_output,
                        max_len,
                        page_size,
                        repeats=repeats,
                    )
                    self.assertEqual(
                        actual_blocks.data_ptr(), block_output.data_ptr()
                    )
                    torch.testing.assert_close(
                        actual_blocks,
                        expected_blocks,
                        rtol=0,
                        atol=0,
                    )
                    self.assertTrue(
                        torch.all(block_output[expected.shape[0] :] == sentinel).item()
                    )
                    self.assertTrue(
                        torch.all(
                            block_output[
                                : expected.shape[0], expected_blocks.shape[1] :
                            ]
                            == sentinel
                        ).item()
                    )

    def test_int64_request_indices(self):
        req_to_token = torch.arange(
            3 * 65, dtype=torch.int32, device="cuda"
        ).reshape(3, 65)
        req_pool_indices = torch.tensor([2, 0], dtype=torch.int64, device="cuda")
        output = torch.empty(4, 65, dtype=torch.int32, device="cuda")

        actual = gather_page_table_into(
            req_to_token,
            req_pool_indices,
            output,
            max_len=65,
            repeats=2,
        )
        expected = torch.repeat_interleave(
            req_to_token[req_pool_indices], repeats=2, dim=0
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_block_table_topk_transform(self):
        page_size = 64
        physical_pages = torch.tensor(
            [[101, 203, 307], [401, 509, 601]],
            dtype=torch.int32,
            device="cuda",
        )
        topk_indices = torch.full(
            (2, 2048), -1, dtype=torch.int32, device="cuda"
        )
        logical = torch.tensor(
            [0, 1, 63, 64, 65, 127, 128, 129],
            dtype=torch.int32,
            device="cuda",
        )
        topk_indices[:, : logical.numel()] = logical

        expected = torch.full_like(topk_indices, -1)
        expected[:, : logical.numel()] = (
            physical_pages[:, logical.long() // page_size] * page_size
            + logical.remainder(page_size)
        )

        ref = transform_index_page_table_decode_ref(
            physical_pages, topk_indices, page_size=page_size
        )
        actual = transform_index_page_table_decode_fast(
            physical_pages, topk_indices, page_size=page_size
        )
        torch.testing.assert_close(ref, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_page_size_one_transform_compatibility(self):
        page_table = torch.arange(
            2 * 130, dtype=torch.int32, device="cuda"
        ).reshape(2, 130)
        topk_indices = torch.full(
            (2, 2048), -1, dtype=torch.int32, device="cuda"
        )
        logical = torch.tensor(
            [0, 1, 63, 64, 129], dtype=torch.int32, device="cuda"
        )
        topk_indices[:, : logical.numel()] = logical
        expected = torch.full_like(topk_indices, -1)
        expected[:, : logical.numel()] = page_table[:, logical.long()]

        actual = transform_index_page_table_decode_fast(
            page_table, topk_indices, page_size=1
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_block_table_topk_transform_variable_width_and_invalid_pages(self):
        page_table = torch.tensor(
            [[2, -1]], dtype=torch.int32, device="cuda"
        )
        topk_indices = torch.tensor(
            [[0, 63, 64, 65, 127, 128, -1]],
            dtype=torch.int64,
            device="cuda",
        )
        expected = torch.tensor(
            [[128, 191, -1, -1, -1, -1, -1]],
            dtype=torch.int32,
            device="cuda",
        )

        ref = transform_index_page_table_decode_ref(
            page_table, topk_indices, page_size=64
        )
        actual = transform_index_page_table_decode_fast(
            page_table, topk_indices, page_size=64
        )
        torch.testing.assert_close(ref, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_empty_block_table_topk_transform(self):
        page_table = torch.empty((2, 0), dtype=torch.int32, device="cuda")
        topk_indices = torch.tensor(
            [[0, -1, 64], [1, 2, 3]], dtype=torch.int32, device="cuda"
        )
        expected = torch.full_like(topk_indices, -1)

        ref = transform_index_page_table_decode_ref(
            page_table, topk_indices, page_size=64
        )
        actual = transform_index_page_table_decode_fast(
            page_table, topk_indices, page_size=64
        )
        torch.testing.assert_close(ref, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
