import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

_DEVICE = "cuda"


def _num_variants(kernel) -> int:
    return len(kernel.device_caches[torch.cuda.current_device()][0])


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestDsv4LengthAgnosticKernels(CustomTestCase):
    """After the startup warmup, serving-path Triton kernels compile nothing new.

    A variant first compiled while serving costs ~75 ms on every rank, inside the
    TTFT of whichever request first hits that prompt length.
    """

    def test_candidate_mask_no_compile_after_warmup(self):
        from sglang.kernels.ops.attention.dsv4 import candidate_block_score as cbs
        from sglang.srt.layers.attention.dsv4.candidate_indexer import (
            candidate_block_mask,
        )

        block_size, max_topk = 8, 2048
        cbs.warmup_mask_candidate_logits(
            max_topk_blocks=max_topk,
            block_size=block_size,
            device=torch.cuda.current_device(),
        )
        kernels = (cbs._publish_block_membership, cbs._mask_candidate_scores_inplace)
        before = [_num_variants(k) for k in kernels]
        gen = torch.Generator(device=_DEVICE).manual_seed(0)
        # Widths and block counts cover the ==1, %16 and odd classes; the row
        # slice [1:4] of an odd-width block tensor misaligns the ids.
        for width, stride, topk in (
            (1, 4, 1),
            (17, 20, 3),
            (128, 128, 16),
            (129, 132, 17),
            (16384, 16384, max_topk),
            (25345, 25348, 77),
        ):
            logits = torch.randn(5, stride, device=_DEVICE, generator=gen)[1:4, :width]
            nblocks = (width + block_size - 1) // block_size
            blocks = torch.randint(
                -1,
                nblocks,
                (5, topk),
                device=_DEVICE,
                dtype=torch.int32,
                generator=gen,
            )[1:4]
            expected = logits.masked_fill(
                ~candidate_block_mask(
                    blocks=blocks, width=width, block_size=block_size
                ),
                -torch.inf,
            )
            cbs.mask_candidate_logits_(logits, blocks, block_size)
            torch.testing.assert_close(logits, expected, rtol=0, atol=0)
        self.assertEqual([_num_variants(k) for k in kernels], before)

    def test_scatter_kv_rows_no_compile_after_warmup(self):
        from sglang.kernels.ops.attention.dsv4 import cp_kv_store
        from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout

        page_size, layout, cp_size = 4, KVLayout.V41, 2
        data_words, scale_words = layout.data_bytes // 4, layout.scale_bytes // 4
        page_bytes = layout.page_bytes(page_size)
        cp_kv_store.warmup_scatter_kv_rows(
            page_size=page_size,
            layout=layout,
            cp_size=cp_size,
            device=torch.cuda.current_device(),
        )
        kernel = cp_kv_store._scatter_kv_rows_kernel
        before = _num_variants(kernel)
        gen = torch.Generator(device=_DEVICE).manual_seed(0)
        # (total, loc offset) covering ==1, %16 and odd sizes, aligned or not.
        for total, offset in ((1, 0), (32, 1), (33, 0), (64, 3)):
            rank_len = -(-total // cp_size)
            npages = -(-rank_len // page_size)
            gathered = torch.randint(
                0,
                255,
                (cp_size * npages, page_bytes),
                device=_DEVICE,
                dtype=torch.uint8,
                generator=gen,
            )
            num_dst_pages = npages * cp_size + 1
            loc = torch.randperm(
                num_dst_pages * page_size, device=_DEVICE, generator=gen
            )[: total + offset].to(torch.int32)[offset:]
            loc[::5] = -1
            token = torch.arange(total, device=_DEVICE, dtype=torch.int64)
            row_map = (token % cp_size) * rank_len + token // cp_size
            for interleave, src_row in ((True, loc), (False, row_map)):
                kvcache = torch.zeros(
                    num_dst_pages, page_bytes, device=_DEVICE, dtype=torch.uint8
                )
                cp_kv_store._launch_scatter_kv_rows(
                    gathered=gathered,
                    kvcache=kvcache,
                    loc=loc,
                    src_row=src_row,
                    interleave=interleave,
                    total=total,
                    rank_len=rank_len,
                    npages=npages,
                    page_bytes=page_bytes,
                    page_size=page_size,
                    layout=layout,
                    cp_size=cp_size,
                )
                expected = _reference_scatter(
                    gathered=gathered.view(torch.int32).flatten(),
                    loc=loc,
                    row_map=row_map,
                    rank_len=rank_len,
                    npages=npages,
                    page_words=page_bytes // 4,
                    page_size=page_size,
                    data_words=data_words,
                    scale_words=scale_words,
                    num_dst_pages=num_dst_pages,
                )
                self.assertTrue(
                    torch.equal(kvcache.view(torch.int32).flatten(), expected)
                )
        self.assertEqual(_num_variants(kernel), before)


def _reference_scatter(
    *,
    gathered,
    loc,
    row_map,
    rank_len,
    npages,
    page_words,
    page_size,
    data_words,
    scale_words,
    num_dst_pages,
):
    scale_base = page_size * data_words
    out = torch.zeros(num_dst_pages * page_words, device=_DEVICE, dtype=torch.int32)
    for t, d in enumerate(loc.tolist()):
        if d < 0:
            continue
        r, j = divmod(int(row_map[t]), rank_len)
        s_base = (r * npages + j // page_size) * page_words
        s_slot = j % page_size
        d_base, d_slot = (d // page_size) * page_words, d % page_size
        sd, ss = d_base + d_slot * data_words, s_base + s_slot * data_words
        out[sd : sd + data_words] = gathered[ss : ss + data_words]
        sd = d_base + scale_base + d_slot * scale_words
        ss = s_base + scale_base + s_slot * scale_words
        out[sd : sd + scale_words] = gathered[ss : ss + scale_words]
    return out


if __name__ == "__main__":
    unittest.main()
