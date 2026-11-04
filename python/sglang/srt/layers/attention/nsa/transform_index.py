from typing import List, Optional

import torch
import triton
import triton.language as tl


def transform_index_page_table_prefill(**kwargs):
    return transform_index_page_table_prefill_ref(**kwargs)


def transform_index_page_table_decode(**kwargs):
    return transform_index_page_table_decode_ref(**kwargs)


@triton.jit
def transform_index_page_table_decode_kernel(
    page_table_ptr: torch.Tensor,
    topk_indices_ptr: torch.Tensor,
    result_ptr: torch.Tensor,
    page_table_stride_0: tl.constexpr,
    page_table_stride_1: tl.constexpr,
    topk_indices_stride_0: tl.constexpr,
    topk_indices_stride_1: tl.constexpr,
    result_stride_0: tl.constexpr,
    result_stride_1: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MAX_NUM_PAGES: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_id = tl.program_id(0)
    page_table_ptr = page_table_ptr + req_id * page_table_stride_0
    topk_indices_ptr = topk_indices_ptr + req_id * topk_indices_stride_0
    result_ptr = result_ptr + req_id * result_stride_0

    offset = tl.arange(0, BLOCK_SIZE)
    topk_mask = offset < TOPK
    loaded_topk_indices = tl.load(
        topk_indices_ptr + offset * topk_indices_stride_1,
        mask=topk_mask,
        other=-1,
    )
    logical_page_indices = loaded_topk_indices // PAGE_SIZE
    valid = (
        topk_mask
        & (loaded_topk_indices >= 0)
        & (logical_page_indices < MAX_NUM_PAGES)
    )
    physical_page_indices = tl.load(
        page_table_ptr + logical_page_indices * page_table_stride_1,
        mask=valid,
        other=-1,
    )
    valid = valid & (physical_page_indices >= 0)
    physical_slots = (
        physical_page_indices * PAGE_SIZE + loaded_topk_indices % PAGE_SIZE
    )
    tl.store(
        result_ptr + offset * result_stride_1,
        tl.where(valid, physical_slots, -1),
        mask=topk_mask,
    )


def transform_index_page_table_decode_fast(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    result: Optional[torch.Tensor] = None,
    page_size: int = 1,
) -> torch.Tensor:
    """
    Transform the page table according to topk indices for sparse topk attention.
    Args:
        page_table: [qo_len, ceil(max_seqlen_k / page_size)] page table
        topk_indices: [qo_len, topk], the topk indices for each query position
    Returns:
        transformed_page_table: [qo_len, topk], the transformed page table
        For out-of-bound indices in topk_indices, this should be filled with -1.
    """
    assert page_size > 0
    assert page_table.shape[0] == topk_indices.shape[0]
    assert page_table.dtype == torch.int32
    assert topk_indices.dtype in (torch.int32, torch.int64)
    qo_len = topk_indices.shape[0]
    topk = topk_indices.shape[1]
    if result is None:
        result = torch.empty_like(topk_indices, dtype=torch.int32)
    assert result.dtype == torch.int32 and result.shape == topk_indices.shape
    if qo_len == 0 or topk == 0:
        return result
    if page_table.shape[1] == 0:
        result.fill_(-1)
        return result
    block_size = triton.next_power_of_2(topk)
    num_warps = 8 if block_size >= 2048 else 4 if block_size >= 256 else 1
    # Launch triton kernel
    grid = (qo_len,)
    transform_index_page_table_decode_kernel[grid](
        page_table,
        topk_indices,
        result,
        page_table.stride(0),
        page_table.stride(1),
        topk_indices.stride(0),
        topk_indices.stride(1),
        result.stride(0),
        result.stride(1),
        PAGE_SIZE=page_size,
        MAX_NUM_PAGES=page_table.shape[1],
        TOPK=topk,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return result


def transform_index_page_table_prefill_fast(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    extend_lens_cpu: List[int],
    page_size: int = 1,
) -> torch.Tensor:
    # TODO(baizhou): can be implemented with another triton kernel
    assert page_size > 0
    result = torch.empty_like(topk_indices, dtype=torch.int32)
    assert len(extend_lens_cpu) == page_table.shape[0]
    offset = 0
    for i, l in enumerate(extend_lens_cpu):
        transform_index_page_table_decode_fast(
            page_table[i].unsqueeze(0).expand(l, -1),
            topk_indices[offset : offset + l],
            result=result[offset : offset + l],
            page_size=page_size,
        )
        offset += l
    assert offset == topk_indices.shape[0]
    return result


def transform_index_page_table_decode_ref(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    result: Optional[torch.Tensor] = None,
    page_size: int = 1,
) -> torch.Tensor:
    assert page_size > 0
    assert page_table.shape[0] == topk_indices.shape[0]
    assert page_table.dtype == torch.int32
    assert topk_indices.dtype in (torch.int32, torch.int64)
    if result is None:
        result = torch.empty_like(topk_indices, dtype=torch.int32)
    assert result.dtype == torch.int32 and result.shape == topk_indices.shape
    if page_table.shape[1] == 0:
        result.fill_(-1)
        return result
    safe_indices = topk_indices.clamp(min=0).to(torch.int64)
    logical_page_indices = torch.div(safe_indices, page_size, rounding_mode="floor")
    valid = (topk_indices >= 0) & (logical_page_indices < page_table.shape[1])
    torch.gather(
        page_table.to(result.dtype),
        dim=1,
        index=logical_page_indices.clamp(max=page_table.shape[1] - 1).to(torch.int64),
        out=result,
    )
    valid &= result >= 0
    result.mul_(page_size).add_(safe_indices.remainder(page_size).to(result.dtype))
    result.masked_fill_(~valid, -1)
    return result


def transform_index_page_table_prefill_ref(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    extend_lens_cpu: List[int],
    page_size: int = 1,
) -> torch.Tensor:
    assert page_size > 0
    result = torch.empty_like(topk_indices, dtype=torch.int32)
    assert len(extend_lens_cpu) == page_table.shape[0]
    offset = 0
    for i, l in enumerate(extend_lens_cpu):
        transform_index_page_table_decode_ref(
            page_table[i].unsqueeze(0).expand(l, -1),
            topk_indices[offset : offset + l],
            result=result[offset : offset + l],
            page_size=page_size,
        )
        offset += l
    assert offset == topk_indices.shape[0]
    return result


if __name__ == "__main__":
    bs, topk, max_seqlen = 10, 2048, 3000
    page_table = torch.randint(0, 100, (bs, max_seqlen), device="cuda")
    topk_indices = torch.full((bs, topk), -1, device="cuda")
    topk_indices[:, :1600] = torch.arange(1600).unsqueeze(0).repeat(bs, 1)
    ref_result = transform_index_page_table_decode_ref(page_table, topk_indices)
    result = transform_index_page_table_decode_fast(page_table, topk_indices)
    assert torch.all(result == ref_result)
    print("Passed")
