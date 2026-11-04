from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import List, Optional

import torch

from sglang.kernels.ops.attention.dsv4.candidate_blocks import candidate_row_lens
from sglang.kernels.ops.attention.dsv4.topk import (
    amax8_varlen,
    plan_topk_v2,
    sort_candidate_blocks,
    topk_transform_bf16_small,
    topk_transform_paged_v2,
    topk_transform_ragged_v2,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.candidate_indexer import (
    CandidateMetadata,
    IndexerInputs,
    PrefillIndexerInputs,
    expand_index_page_table,
)
from sglang.srt.layers.attention.dsv4.dense_prefill_indexer import (
    DenseCandidateIndexer,
    score_tiles,
)
from sglang.srt.layers.attention.dsv4.indexer import (
    fp4_paged_mqa_logits,
)

CANDIDATE_BLOCK_SIZE = 8  # positions per block; DeepGEMM accepts 8 or 16

# Longest row DeepGEMM's sparse MQA schedule takes. Measured on SM100: a row of
# 2**20 positions runs, one of 2**20 + 1 trips the device assert
# `logical_page_idx < block_table_stride` in sm100_sparse_mqa_logits_metadata.cuh
# (reported as `unspecified launch failure`) whatever the page table's width.
# A verify row can sit past the context limit (a request at context_len - 2 plus
# the draft block), and those positions are never committed.
MAX_SPARSE_SEQ_LEN = 1 << 20


@dataclass
class SparseBlockTable(CandidateMetadata):
    # [rows, topk_blocks] int32: ascending logical block ids, valid for the first
    # min(topk_blocks, ceil(seq_len / 8)) entries of a row; DeepGEMM reads only those
    blocks: torch.Tensor
    # DeepGEMM's schedule metadata (uint8) for them
    schedule: torch.Tensor
    # [rows, topk_blocks] int32: the same blocks as pool slots / 8, so a consumer's
    # top-k maps column j of the sparse row to slot phys_blocks[b, j // 8] * 8 + j % 8
    # with the plain page-table transform at page size 8
    phys_blocks: torch.Tensor
    # [rows] int32: length of each row of the sparse logits (see `valid_lens`)
    valid_lens: torch.Tensor


def valid_lens(seq_lens: torch.Tensor, topk_blocks: int) -> torch.Tensor:
    """Length of each row of the sparse logits: the published blocks laid out
    block by block, the newest (highest) block possibly partial. Torch reference
    of ``candidate_row_lens``; the decode path takes the kernel's value."""
    block = CANDIDATE_BLOCK_SIZE
    num = ((seq_lens + block - 1) // block).clamp_max(topk_blocks)
    return block * (num - 1) + (seq_lens - 1) % block + 1


def amax_topk_blocks(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    nblocks: torch.Tensor,
    topk_blocks: int,
    max_seq_len: Optional[int] = None,
) -> torch.Tensor:
    """Per row the ``topk_blocks`` blocks of 8 positions with the largest block
    maximum among its first ``seq_lens[b]`` positions, the newest block always
    included: block ids in no particular order, ``-1`` past the row's count
    (``sort_candidate_blocks`` turns them into the published table). ``nblocks``
    is ``ceil(seq_lens / 8)`` as int32, from ``candidate_row_lens``."""
    rows = logits.shape[0]
    block = CANDIDATE_BLOCK_SIZE
    if max_seq_len is None:
        max_seq_len = logits.shape[1]
    # NOTE: plan cannot be the previous kernel of topk_transform_paged_v2
    plan = plan_topk_v2(nblocks)
    # block maxima, the newest block +inf; the top-k reads each row up to nblocks
    # only, so nothing past a row's keys is initialised (v2 needs stride % 4 == 0)
    keys = logits.new_empty(rows, -(-max_seq_len // (4 * block)) * 4)
    amax8_varlen(logits, seq_lens, out=keys)
    blocks = torch.empty(rows, topk_blocks, dtype=torch.int32, device=logits.device)
    topk_transform_paged_v2(keys, nblocks, None, blocks, 1, plan)
    return blocks


_ROW_IDS: dict = {}


def _row_ids(rows: int, device: torch.device) -> torch.Tensor:
    """``arange(rows)`` int32 from a cached buffer (grown in steps of 8192), so the
    every-row-its-own-request case costs no launch."""
    buf = _ROW_IDS.get(device)
    if buf is None or buf.numel() < rows:
        size = max(8192, -(-rows // 8192) * 8192)
        buf = _ROW_IDS[device] = torch.arange(size, dtype=torch.int32, device=device)
    return buf[:rows]


def build_sparse_indexer_schedule(
    blocks: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
    q_dtype: torch.dtype,
    request_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """DeepGEMM's schedule for the published blocks: ``seq_lens`` ``[rows]``
    int32, ``page_table`` ``[rows, pages]`` int32 at the index pool's page size.
    ``request_ids`` ``[rows]``, one id per row with a request's rows consecutive
    (verify: its draft tokens), lets DeepGEMM pair two rows of a request on one
    KV pass; each row keeps its own block list and output layout. Paired rows
    must share their page-table row. None: every row is its own request."""
    import deep_gemm

    rows = blocks.shape[0]
    if request_ids is None:
        request_ids = _row_ids(rows, blocks.device)  # cached, no launch
    else:
        # the scheduler keeps request indices as int64; one small cast per publish
        request_ids = request_ids[:rows].to(torch.int32).contiguous()
    return deep_gemm.get_paged_sparse_mqa_logits_metadata(
        seq_lens.contiguous(),
        page_table,
        request_ids,
        page_size,
        blocks,
        q_dtype,
        CANDIDATE_BLOCK_SIZE,
    )


def sparse_logits(
    q_fp4: torch.Tensor,
    q_sf: torch.Tensor,
    k_cache: torch.Tensor,
    weights: torch.Tensor,
    table: SparseBlockTable,
) -> torch.Tensor:
    """bf16 logits ``[rows, topk_blocks * 8]`` of the published blocks: ``q_fp4``
    ``[rows, 1, heads, 64]`` int8 with ``q_sf`` ``[rows, 1, heads]`` int32 (packed
    ue8m0), ``k_cache`` ``[pages, page_size, 1, 68]`` uint8 whose page stride is
    a multiple of 512 bytes, ``weights`` ``[rows, heads]`` bf16."""
    import deep_gemm

    return deep_gemm.fp8_fp4_paged_sparse_mqa_logits(
        (q_fp4, q_sf),
        k_cache,
        weights,
        table.schedule,
        table.blocks.shape[1],
        CANDIDATE_BLOCK_SIZE,
    )


def topk_transform_sparse(
    logits: torch.Tensor,
    valid_lens: torch.Tensor,
    table: SparseBlockTable,
    page_indices: torch.Tensor,
) -> None:
    """Top-``k`` (``k = page_indices.shape[1]``) of every row of the sparse
    ``logits`` (bf16 ``[rows, topk_blocks * 8]``) within its first ``valid_lens[b]``
    columns, written as pool slots, ``-1`` where a row has fewer than ``k`` valid
    columns, in no particular order. Column ``j`` is slot
    ``phys_blocks[b, j // 8] * 8 + j % 8``: the bf16 top-k kernel's page-table
    transform at page size 8 over the physical block table."""
    topk_transform_bf16_small(
        logits, valid_lens, table.phys_blocks, page_indices, CANDIDATE_BLOCK_SIZE
    )


def warmup(rows: int, topk_blocks: int, page_size: int, q_dtype: torch.dtype, device):
    """Build one schedule so DeepGEMM allocates its per-stream metadata workspace
    outside any CUDA-graph capture."""
    seq_lens = torch.full(
        (rows,), CANDIDATE_BLOCK_SIZE, dtype=torch.int32, device=device
    )
    blocks = torch.zeros(rows, topk_blocks, dtype=torch.int32, device=device)
    page_table = torch.zeros(rows, 1, dtype=torch.int32, device=device)
    build_sparse_indexer_schedule(blocks, seq_lens, page_table, page_size, q_dtype)


@dataclass
class PrefillSparseBlockTable(SparseBlockTable):
    """Published by a prefill chunk's candidate source: one row per query token.
    The inputs stay attached so the late-layer tail can rebuild the table for
    its rows (the DeepGEMM schedule is per row set, it cannot be sliced)."""

    compress_lens: torch.Tensor  # [rows] int32
    page_table: torch.Tensor  # [rows, index pages] int32
    request_ids: torch.Tensor  # [rows] int32
    q_dtype: torch.dtype
    page_size: int  # index-K pool page size (slots)
    rows_per_request: List[int]  # rows of each request, in row order


class DeepGemmCandidateIndexer:
    def __init__(
        self,
        topk_blocks: int,
        block_size: int,
        prefill_dense: Optional[DenseCandidateIndexer] = None,
    ):
        assert block_size == CANDIDATE_BLOCK_SIZE, block_size
        self.topk_blocks = topk_blocks
        self.block_size = block_size
        # set for the forwards whose prefill rows the page table does not describe
        self._prefill_dense = prefill_dense
        self.alt_stream = None
        self.need_wait = False
        # TODO: maybe lower this priority
        if envs.SGLANG_DSV41_DEEP_GEMM_CANDIDATE_OVERLAP.get():
            self.alt_stream = torch.cuda.Stream()

    def publish_decode(
        self,
        inputs: IndexerInputs,
        page_indices: torch.Tensor,
        raw_indices: Optional[torch.Tensor] = None,
    ) -> SparseBlockTable:
        """Layer 20 end to end: dense logits, its own plain top-k into
        ``page_indices`` (and ``raw_indices`` when given), the block table from
        the same logits and DeepGEMM's schedule for it; the backend stores the
        table on the forward metadata."""
        metadata = inputs.metadata
        seq_lens = metadata.compressed_seq_lens.reshape(-1)
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
        if self.alt_stream is not None:
            self.alt_stream.wait_stream(torch.cuda.current_stream())
            # the level-one chain reads `logits` on the side stream while the main
            # stream moves on: keep the allocator from handing its memory to a
            # later main-stream allocation before those reads ran
            logits.record_stream(self.alt_stream)
            self.need_wait = True
            stream_ctx = torch.cuda.stream(self.alt_stream)
        else:
            stream_ctx = contextlib.nullcontext()
        # TODO(candidate): one kernel for both selections below (dense logits read once)
        topk_transform_paged_v2(
            logits,
            seq_lens,
            metadata.page_table,
            page_indices,
            metadata.compressed_page_size,
            metadata.topk_metadata,
            out_raw_indices=raw_indices,
        )
        # per row: block count for the block top-k, sparse-row length for the consumers
        with stream_ctx:
            # the sparse chain never sees a row longer than DeepGEMM's limit; the
            # dense selection above keeps the true length
            sparse_lens = seq_lens.clamp_max(MAX_SPARSE_SEQ_LEN)
            nblocks, row_valid_lens = candidate_row_lens(sparse_lens, self.topk_blocks)
            blocks = amax_topk_blocks(logits, sparse_lens, nblocks, self.topk_blocks)
            # in place: ascending, INT32_MAX padded, plus the blocks as pool slots / 8
            phys_blocks = sort_candidate_blocks(
                blocks,
                sparse_lens,
                metadata.page_table,
                metadata.compressed_page_size,
            )
            schedule = build_sparse_indexer_schedule(
                blocks,
                sparse_lens,
                metadata.page_table,
                metadata.compressed_page_size,
                inputs.q_fp4.dtype,
                inputs.request_ids,
            )
            return SparseBlockTable(
                blocks=blocks,
                schedule=schedule,
                phys_blocks=phys_blocks,
                valid_lens=row_valid_lens,
            )

    def scores(self, table: SparseBlockTable, inputs: IndexerInputs) -> torch.Tensor:
        """A consumer: bf16 logits ``[rows, topk_blocks * 8]`` of the published
        blocks only."""
        return sparse_logits(
            inputs.q_fp4,
            inputs.q_sf,
            inputs.k_cache,
            inputs.weights.to(torch.bfloat16),
            table,
        )

    def select_decode(
        self,
        candidate_metadata: SparseBlockTable,
        inputs: IndexerInputs,
        page_indices: torch.Tensor,
        raw_indices: Optional[torch.Tensor] = None,
    ) -> None:
        """A consumer: top-``k`` (``k = page_indices.shape[1]``) inside the
        published blocks, written ascending as slots through the page table with
        ``-1`` past the valid count (and as positions into ``raw_indices`` when
        given)."""
        assert raw_indices is None
        table = candidate_metadata
        if self.alt_stream is not None and self.need_wait:
            self.need_wait = False
            torch.cuda.current_stream().wait_stream(self.alt_stream)
        logits = self.scores(table, inputs)
        # decode carries no raw_indices; the kernel writes slots only
        topk_transform_sparse(logits, table.valid_lens, table, page_indices)

    def publish_prefill(
        self, inputs: PrefillIndexerInputs, out_positions: torch.Tensor
    ) -> CandidateMetadata:
        """The source layer's own top-k and the block table of the chunk from
        one pass over its dense scores, tile by tile: per tile the plain top-k
        into ``out_positions`` and one read for the block keys."""
        if self._prefill_dense is not None:
            return self._prefill_dense.publish_prefill(inputs, out_positions)
        rows = inputs.num_rows
        device = inputs.q_fp4.device
        nblocks, valid_lens = candidate_row_lens(inputs.compress_lens, self.topk_blocks)
        blocks = torch.empty(rows, self.topk_blocks, dtype=torch.int32, device=device)
        # the block keys read the score rows through 32-byte vectors
        for tile, logits in score_tiles(inputs, width_align=8):
            lens = inputs.compress_lens[tile]
            topk_transform_ragged_v2(
                logits,
                lens,
                out_offsets=inputs.request_starts[tile],
                out_indices=out_positions[tile],
            )
            # keys past a row's block count stay unset; the top-k reads a row
            # up to its block count only (v2 wants the stride a multiple of 4)
            width_blocks = (
                logits.shape[1] + CANDIDATE_BLOCK_SIZE - 1
            ) // CANDIDATE_BLOCK_SIZE
            keys = logits.new_empty(logits.shape[0], (width_blocks + 3) // 4 * 4)
            amax8_varlen(logits, lens, out=keys)
            topk_transform_ragged_v2(
                keys,
                nblocks[tile],
                out_offsets=torch.zeros(
                    logits.shape[0], dtype=torch.int32, device=device
                ),
                out_indices=blocks[tile],
            )
        request_ids = torch.repeat_interleave(
            torch.arange(
                len(inputs.rows_per_request), dtype=torch.int32, device=device
            ),
            torch.tensor(inputs.rows_per_request, dtype=torch.int64, device=device),
            output_size=rows,
        )
        # index-K pool pages of each row's request
        page_table = expand_index_page_table(
            inputs.kv_page_table,
            full_page_size=inputs.kv_page_size,
            compress_ratio=inputs.compress_ratio,
            index_page_size=inputs.page_size,
        ).contiguous()
        return self._prefill_table(
            blocks,
            inputs.compress_lens,
            page_table,
            inputs.page_size,
            request_ids,
            inputs.rows_per_request,
            inputs.q_fp4.dtype,
            valid_lens,
        )

    def _prefill_table(
        self,
        blocks,
        compress_lens,
        page_table,
        page_size,
        request_ids,
        rows_per_request,
        q_dtype,
        valid_lens,
    ) -> PrefillSparseBlockTable:
        # in place: ascending, INT32_MAX padded, plus the blocks as pool slots / 8
        phys_blocks = sort_candidate_blocks(
            blocks, compress_lens, page_table, page_size
        )
        schedule = build_sparse_indexer_schedule(
            blocks, compress_lens, page_table, page_size, q_dtype, request_ids
        )
        return PrefillSparseBlockTable(
            blocks=blocks,
            schedule=schedule,
            phys_blocks=phys_blocks,
            valid_lens=valid_lens,
            compress_lens=compress_lens,
            page_table=page_table,
            request_ids=request_ids,
            q_dtype=q_dtype,
            page_size=page_size,
            rows_per_request=rows_per_request,
        )

    def prefill_tail(
        self, published: CandidateMetadata, tail_lens: List[int]
    ) -> CandidateMetadata:
        """Rebuild the block table for the last ``tail_lens[b]`` rows per request."""
        if self._prefill_dense is not None:
            return self._prefill_dense.prefill_tail(published, tail_lens)
        table = published
        assert isinstance(table, PrefillSparseBlockTable), "prefill block table missing"
        rows, start = [], 0
        for n, t in zip(table.rows_per_request, tail_lens):
            rows.append(torch.arange(start + n - t, start + n))
            start += n
        rows = torch.cat(rows).to(table.blocks.device)
        compress_lens = table.compress_lens[rows].contiguous()
        _, valid_lens = candidate_row_lens(compress_lens, self.topk_blocks)
        return self._prefill_table(
            table.blocks[rows].contiguous(),
            compress_lens,
            table.page_table[rows].contiguous(),
            table.page_size,
            table.request_ids[rows].contiguous(),
            tail_lens,
            table.q_dtype,
            valid_lens,
        )

    def select_prefill(
        self,
        published: CandidateMetadata,
        inputs: PrefillIndexerInputs,
        out_positions: torch.Tensor,
    ) -> None:
        """A consumer layer's top-k, scored sparsely over the published blocks."""
        if self._prefill_dense is not None:
            return self._prefill_dense.select_prefill(published, inputs, out_positions)
        table = published
        assert isinstance(table, PrefillSparseBlockTable), "prefill block table missing"
        rows, heads = inputs.q_sf.shape
        logits = sparse_logits(
            inputs.q_fp4.view(rows, 1, heads, 64),
            inputs.q_sf.view(rows, 1, heads),
            inputs.k_cache,
            inputs.weights.to(torch.bfloat16),
            table,
        )
        # request-relative compressed positions, -1 padded
        topk_transform_bf16_small(
            logits, table.valid_lens, table.blocks, out_positions, CANDIDATE_BLOCK_SIZE
        )
        out_positions.add_(
            torch.where(out_positions >= 0, inputs.request_starts[:, None], 0)
        )
