# TODO(candidate): retire with the last path that still selects through masks
# (Hopper decode, prefill); the paged fp4 decode path already has the DeepGEMM one.
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.dsv4.candidate_indexer import (
    CandidateMetadata,
    IndexerInputs,
    PrefillIndexerInputs,
    mask_topk_scores,
)
from sglang.srt.layers.attention.dsv4.indexer import (
    fp4_paged_mqa_logits,
    fp32_jit_paged_topk,
    select_candidate_blocks,
)


@dataclass
class CandidateMasks(CandidateMetadata):
    mask: Optional[torch.Tensor] = None  # decode: [rows, width] bool
    request_masks: Optional[List[torch.Tensor]] = None  # prefill: [rows_b, lc_b] each


@dataclass
class PrefillCandidateBlocks(CandidateMetadata):
    """Prefill level-one candidates as int32 block ids ``[rows_b, <= topk_blocks]``
    per request (-1 = none), instead of bool ``[rows_b, lc_b]`` position masks:
    the published size no longer grows with the prompt length."""

    request_blocks: List[torch.Tensor]

    def tail(self, lengths: List[int]) -> "PrefillCandidateBlocks":
        """Like ``tail`` for requests, but returning blocks instead of masks."""
        return PrefillCandidateBlocks(
            request_blocks=[
                blocks[blocks.shape[0] - length :]
                for blocks, length in zip(self.request_blocks, lengths)
            ]
        )


def published_prefill_blocks(candidate) -> PrefillCandidateBlocks:
    """The forward's ``candidate_metadata`` as the block ids the source published."""
    assert isinstance(candidate, PrefillCandidateBlocks), "candidate blocks missing"
    return candidate


def _candidate_block_topk(
    logits: torch.Tensor,
    compress_lens,
    topk_blocks: int,
    block_size: int,
):
    width = logits.size(-1)
    padding = -width % block_size
    scores = F.pad(logits, (0, padding), value=-torch.inf) if padding else logits
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    # The block holding the query's newest position is always kept.
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(
        torch.arange(num_blocks, device=logits.device) == last, torch.inf
    )
    return scores.topk(min(topk_blocks, num_blocks), dim=-1)


def select_candidate_block_ids(
    logits: torch.Tensor,
    compress_lens,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """``select_candidate_blocks`` as block ids: the same kept blocks, -1 where the
    mask would keep nothing (all -inf block)."""
    top = _candidate_block_topk(logits, compress_lens, topk_blocks, block_size)
    return top.indices.to(torch.int32).masked_fill_(~(top.values > -torch.inf), -1)


def candidate_block_mask(
    blocks: torch.Tensor, width: int, block_size: int
) -> torch.Tensor:
    """Expand block ids back to the bool position mask over ``width`` columns."""
    num_blocks = (width + block_size - 1) // block_size
    keep = torch.zeros(
        (*blocks.shape[:-1], num_blocks + 1), dtype=torch.bool, device=blocks.device
    )
    keep.scatter_(-1, blocks.to(torch.int64).masked_fill(blocks < 0, num_blocks), True)
    return keep[..., :num_blocks].repeat_interleave(block_size, dim=-1)[..., :width]


def published_masks(candidate) -> CandidateMasks:
    """The forward's ``candidate_metadata`` as the masks the source published."""
    assert isinstance(candidate, CandidateMasks), "candidate masks missing"
    return candidate


def two_level_decode_logits(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    is_candidate_source: bool,
    uses_candidates: bool,
    topk_blocks: int,
    block_size: int,
    published: Optional[torch.Tensor],
    _consumer_live_prefix: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Apply candidate-block filtering and return logits plus an optional published mask.

    Mask columns past each sequence length to -inf before selection: the paged
    logits kernel leaves that tail uninitialized, and an all -inf block means
    unreachable. This path is graph-captured and must not synchronize with the host.
    Work normally scales with allocated page-table capacity. The private decode
    consumer opt-in writes only its live prefix and vector padding on CUDA.
    """
    if not (is_candidate_source or uses_candidates):
        return logits, None

    if (
        logits.is_cuda
        and torch.version.cuda is not None
        and logits.ndim == 2
        and logits.stride(1) == 1
        and seq_lens.device == logits.device
        and seq_lens.dtype in (torch.int32, torch.int64)
        and seq_lens.is_contiguous()
        and seq_lens.shape in ((logits.shape[0],), (logits.shape[0], 1))
        and logits.numel() > 0
        and 0 < block_size <= 1024
    ):
        from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
            candidate_block_logits,
        )

        if not is_candidate_source:
            assert (
                torch.is_tensor(published)
                and published.shape[0] == logits.shape[0]
                and published.shape[1] >= logits.shape[1]
            ), "candidate mask missing for decode"
        return candidate_block_logits(
            logits,
            seq_lens,
            topk_blocks=topk_blocks,
            block_size=block_size,
            published=None if is_candidate_source else published,
            _consumer_live_prefix=_consumer_live_prefix and not is_candidate_source,
        )

    lens_col = seq_lens if seq_lens.dim() > 1 else seq_lens.unsqueeze(-1)
    reachable = torch.arange(logits.shape[-1], device=logits.device) < lens_col
    logits = logits.float().masked_fill(~reachable, -torch.inf)

    if is_candidate_source:
        # The source scores over every reachable position itself and only publishes,
        # which is what the reference does.
        return logits, select_candidate_blocks(
            logits, lens_col, topk_blocks=topk_blocks, block_size=block_size
        )

    assert torch.is_tensor(published) and published.shape[0] == logits.shape[0], (
        "candidate mask missing for decode"
    )
    return logits.masked_fill(~published[:, : logits.shape[-1]], -torch.inf), None


def cut_request_masks(masks: CandidateMasks, tail_lens: List[int]) -> CandidateMasks:
    """The last ``tail_lens[b]`` rows of each request's mask."""
    assert masks.request_masks is not None, "prefill masks missing"
    return CandidateMasks(
        request_masks=[
            mask[mask.shape[0] - t :] for mask, t in zip(masks.request_masks, tail_lens)
        ]
    )


class TorchCandidateIndexer:
    def __init__(self, topk_blocks: int, block_size: int, prefill_dense=None):
        self.topk_blocks = topk_blocks
        self.block_size = block_size
        # the dense-score prefill implementation (DenseCandidateIndexer)
        self._prefill_dense = prefill_dense

    def publish_prefill(
        self, inputs: PrefillIndexerInputs, out_positions: torch.Tensor
    ) -> CandidateMetadata:
        """Publish the chunk's candidates through the dense prefill indexer."""
        return self._prefill_dense.publish_prefill(inputs, out_positions)

    def select_prefill(
        self,
        published: CandidateMetadata,
        inputs: PrefillIndexerInputs,
        out_positions: torch.Tensor,
    ) -> None:
        """A consumer layer's top-k through the dense prefill indexer."""
        self._prefill_dense.select_prefill(published, inputs, out_positions)

    def prefill_tail(
        self, published: CandidateMetadata, tail_lens: List[int]
    ) -> CandidateMetadata:
        """The published candidates of the late-layer tail rows."""
        return self._prefill_dense.prefill_tail(published, tail_lens)

    def publish_decode(
        self,
        inputs: IndexerInputs,
        page_indices: torch.Tensor,
        raw_indices: Optional[torch.Tensor] = None,
    ) -> CandidateMasks:
        """Layer 20 end to end: dense logits, its own plain top-k into
        ``page_indices`` (and ``raw_indices`` when given), and the mask for the
        layers after it; the backend stores the mask on the forward metadata."""
        metadata = inputs.metadata
        logits = fp4_paged_mqa_logits(
            (inputs.q_fp4, inputs.q_sf),
            inputs.k_cache,
            inputs.weights,
            metadata.compressed_seq_lens,
            metadata.page_table,
            metadata.deep_gemm_metadata,
            metadata.max_compressed_seq_len,
            request_ids=metadata.request_ids,
        )
        logits, mask = two_level_decode_logits(
            logits,
            metadata.compressed_seq_lens,
            is_candidate_source=True,
            uses_candidates=False,
            topk_blocks=self.topk_blocks,
            block_size=self.block_size,
            published=None,
        )
        fp32_jit_paged_topk(logits, metadata, page_indices, raw_indices)
        return CandidateMasks(mask=mask)

    def select_decode(
        self,
        candidate_metadata: CandidateMasks,
        inputs: IndexerInputs,
        page_indices: torch.Tensor,
        raw_indices: Optional[torch.Tensor] = None,
    ) -> None:
        """A consumer: dense logits, masked, top-k, written as slots through the
        page table with ``-1`` past the valid count (and as positions into
        ``raw_indices`` when given)."""
        metadata = inputs.metadata
        page_size = metadata.compressed_page_size
        assert isinstance(candidate_metadata, CandidateMasks)
        logits = fp4_paged_mqa_logits(
            (inputs.q_fp4, inputs.q_sf),
            inputs.k_cache,
            inputs.weights,
            metadata.compressed_seq_lens,
            metadata.page_table,
            metadata.deep_gemm_metadata,
            metadata.max_compressed_seq_len,
            request_ids=metadata.request_ids,
        )
        logits, _ = two_level_decode_logits(
            logits,
            metadata.compressed_seq_lens,
            is_candidate_source=False,
            uses_candidates=True,
            topk_blocks=self.topk_blocks,
            block_size=self.block_size,
            published=candidate_metadata.mask,
            _consumer_live_prefix=metadata.use_topk_v2,
        )
        # raw positions into `selected`; `page_indices` gets the unfiltered slots
        # here and is rewritten with the masked selection just below
        selected = torch.empty_like(page_indices)
        fp32_jit_paged_topk(logits, metadata, page_indices, raw_indices=selected)
        if logits.is_cuda and torch.version.cuda is not None:
            # fused: drop the selections the mask zeroed, page-transform the rest
            from sglang.kernels.ops.attention.dsv4.indexer_postprocess import (
                filter_topk_pages,
            )

            filter_topk_pages(
                logits,
                selected,
                metadata.page_table,
                page_indices,
                page_size,
                raw_indices,
            )
        else:
            selected = mask_topk_scores(logits, selected)
            columns = selected.clamp_min(0).to(torch.int64)
            slots = metadata.page_table.gather(1, columns // page_size) * page_size
            slots = slots + columns % page_size
            page_indices.copy_(torch.where(selected >= 0, slots, -1))
            if raw_indices is not None:
                raw_indices.copy_(selected)
