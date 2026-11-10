"""SGLANG_K31_DSA_VERIFY_META_ROW_PROGS: the strided target-verify metadata grid
must write exactly what the default grid writes over every row's live prefix."""
import unittest

import torch

from sglang.kernels.ops.attention import dsa_metadata
from sglang.srt.environ import envs


def _run(row_progs, seq_lens, req_pool_indices, req_to_token, width, next_n, page):
    bs = seq_lens.numel()
    e = bs * next_n
    dev = seq_lens.device
    out = dict(
        cache_seqlens=torch.full((bs,), -7, dtype=torch.int32, device=dev),
        cu_seqlens_k=torch.full((bs + 1,), -7, dtype=torch.int32, device=dev),
        page_table_1=torch.full((e, width), -7, dtype=torch.int32, device=dev),
        seqlens_expanded=torch.full((e,), -7, dtype=torch.int32, device=dev),
        dsa_cache_seqlens=torch.full((e,), -7, dtype=torch.int32, device=dev),
        dsa_cu_seqlens_k=torch.full((e + 1,), -7, dtype=torch.int32, device=dev),
        real_page_table=torch.full((e, width // page), -7, dtype=torch.int32, device=dev),
    )
    with envs.SGLANG_K31_DSA_VERIFY_META_ROW_PROGS.override(row_progs):
        dsa_metadata.fused_dsa_target_verify_metadata(
            seq_lens=seq_lens, req_pool_indices=req_pool_indices, req_to_token=req_to_token,
            bs=bs, max_seqlen_k=width, dsa_index_topk=2048, real_page_size=page,
            next_n=next_n, **out,
        )
    torch.cuda.synchronize()
    return out


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestK31DsaVerifyMetaRowProgs(unittest.TestCase):
    def test_strided_matches_default_on_live_prefix(self):
        torch.manual_seed(0)
        width, next_n, page, pool = 1 << 17, 6, 64, 40
        req_to_token = torch.randint(0, 1 << 22, (pool, width), dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([1, 63, 64, 1000, 4097, 70000, width - next_n, 129],
                                dtype=torch.int64, device="cuda")
        req_pool_indices = torch.randperm(pool, device="cuda")[: seq_lens.numel()]
        ref = _run(0, seq_lens, req_pool_indices, req_to_token, width, next_n, page)
        for row_progs in (1, 4, 16):
            got = _run(row_progs, seq_lens, req_pool_indices, req_to_token, width, next_n, page)
            for k in ("cache_seqlens", "cu_seqlens_k", "seqlens_expanded",
                      "dsa_cache_seqlens", "dsa_cu_seqlens_k"):
                self.assertTrue(torch.equal(got[k], ref[k]), k)
            for r in range(seq_lens.numel() * next_n):
                live = int(seq_lens[r // next_n]) + next_n
                self.assertTrue(torch.equal(got["page_table_1"][r, :live],
                                            ref["page_table_1"][r, :live]), (row_progs, r))
                lp = (live + page - 1) // page
                self.assertTrue(torch.equal(got["real_page_table"][r, :lp],
                                            ref["real_page_table"][r, :lp]), (row_progs, r))


if __name__ == "__main__":
    unittest.main()
