from __future__ import annotations

import pytest
import torch

from sglang.kernels.ops.attention.dsa.index_buf_accessor import GetKAndS, SetKAndS
from sglang.kernels.ops.attention.dsa.triton_kernel import act_quant_fp4
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

HEAD_DIM = 128
FP4_DIM = HEAD_DIM // 2
GROUP_SIZE = 32
SCALE_GROUPS = HEAD_DIM // GROUP_SIZE
SCALE_BYTES = 4
PAGE_SIZE = 64
E2M1_MAX = 6.0


class MockFP4DSATokenToKVPool:
    """Minimal pool object for the FP4 68B/token index-K layout."""

    def __init__(self, page_size: int = PAGE_SIZE):
        self.page_size = page_size
        self.index_head_dim = HEAD_DIM
        self.quant_block_size = HEAD_DIM
        self.device = "cuda"
        self.use_fp4_indexer = True
        self.index_k_data_bytes_per_token = FP4_DIM
        self.index_k_scale_bytes_per_token = SCALE_BYTES
        self.index_k_bytes_per_token = FP4_DIM + SCALE_BYTES
        self.index_k_scale_offset_in_page = page_size * FP4_DIM


def _ceil_ue8m0_exp_ref(x: torch.Tensor) -> torch.Tensor:
    bits = x.to(torch.float32).contiguous().view(torch.int32)
    exp = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    return (exp + (mantissa != 0).to(torch.int32)).clamp(1, 254)


def _fp4_e2m1_code_ref(x: torch.Tensor) -> torch.Tensor:
    ax = torch.minimum(x.abs(), torch.tensor(E2M1_MAX, device=x.device))
    idx = torch.zeros_like(ax, dtype=torch.uint8)
    for threshold in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        idx += (ax > threshold).to(torch.uint8)
    return idx | (((x < 0) & (idx != 0)).to(torch.uint8) * 8)


def _ref_quantize_fp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.contiguous().view(-1, HEAD_DIM).float()
    groups = x.view(-1, SCALE_GROUPS, GROUP_SIZE)
    scale_raw = (groups.abs().amax(dim=-1) / E2M1_MAX).clamp_min(1.0e-4)
    scale_exp = _ceil_ue8m0_exp_ref(scale_raw)
    scale = (scale_exp << 23).contiguous().view(torch.float32)

    scaled = (groups / scale.unsqueeze(-1)).view(-1, HEAD_DIM)
    code = _fp4_e2m1_code_ref(scaled)
    packed = (code[:, 0::2].to(torch.int16) | (code[:, 1::2].to(torch.int16) << 4)).to(
        torch.uint8
    )

    packed_sf = scale_exp[:, 0].clone()
    for group_id in range(1, SCALE_GROUPS):
        packed_sf |= scale_exp[:, group_id] << (8 * group_id)
    return packed, packed_sf


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("num_tokens", [1, 7, 96])
def test_act_quant_fp4_matches_reference(num_tokens: int) -> None:
    torch.manual_seed(num_tokens)
    device = get_device()
    x = torch.randn(num_tokens, HEAD_DIM, device=device, dtype=torch.bfloat16)
    x[0, :8] = torch.tensor(
        [-8.0, -6.0, -3.0, -1.5, 0.0, 0.5, 2.0, 8.0],
        device=device,
        dtype=torch.bfloat16,
    )

    packed, sf = act_quant_fp4(x)
    ref_packed, ref_sf = _ref_quantize_fp4(x)

    torch.testing.assert_close(packed.view(torch.uint8), ref_packed)
    torch.testing.assert_close(sf.squeeze(-1), ref_sf)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("num_tokens", [1, 64, 96])
def test_fp4_index_cache_roundtrip(num_tokens: int) -> None:
    torch.manual_seed(num_tokens)
    device = get_device()
    pool = MockFP4DSATokenToKVPool()
    num_pages = max(1, (num_tokens + PAGE_SIZE - 1) // PAGE_SIZE)
    buf = torch.zeros(
        num_pages,
        PAGE_SIZE * (FP4_DIM + SCALE_BYTES),
        device=device,
        dtype=torch.uint8,
    )

    x = torch.randn(num_tokens, HEAD_DIM, device=device, dtype=torch.bfloat16)
    index_k, index_k_scale = act_quant_fp4(x)
    loc = torch.arange(num_tokens, device=device, dtype=torch.int64)

    SetKAndS.execute(
        pool=pool,
        buf=buf,
        loc=loc,
        index_k=index_k,
        index_k_scale=index_k_scale,
    )

    # Per-page layout: page_size * 64B packed data, then page_size * 4B scales.
    for token_id in range(num_tokens):
        page = token_id // PAGE_SIZE
        offset = token_id % PAGE_SIZE
        torch.testing.assert_close(
            buf[page, offset * FP4_DIM : (offset + 1) * FP4_DIM],
            index_k[token_id].view(torch.uint8),
        )
        sf_start = PAGE_SIZE * FP4_DIM + offset * SCALE_BYTES
        torch.testing.assert_close(
            buf[page, sf_start : sf_start + SCALE_BYTES],
            index_k_scale[token_id].view(torch.uint8),
        )

    gathered_k, gathered_s = GetKAndS.triton(
        pool,
        buf,
        page_indices=torch.arange(num_pages, device=device)[None, :],
        seq_len_tensor=torch.tensor([num_tokens], device=device),
        seq_len_sum=num_tokens,
        max_seq_len=num_tokens,
    )
    torch.testing.assert_close(gathered_k, index_k.view(torch.uint8))
    torch.testing.assert_close(gathered_s, index_k_scale.view(torch.uint8))
