"""Per-query sparse-index combiner for the FlashMLA sparse prefill path.

Adapts vllm's ``combine_topk_swa_indices`` to sglang's flat-workspace layout.
Reference:
https://github.com/vllm-project/vllm/blob/124fac10cb0ea83aee2ffeabac0b413d6b759b26/vllm/models/deepseek_v4/common/ops/cache_utils.py#L476

For each
query token in a prefill chunk, emits one row of combined indices into the
chunk's bf16 KV workspace:

    [ topk indices into compressed cache (rebased)   ]
    [ swa positional indices (rebased)               ]
    [ -1 padding up to a multiple of 128             ]

The workspace is a single flat ``(total_workspace_tokens, 512)`` tensor
formed by concatenating, per request, that request's compressed-region
gather followed by all requests' SWA-region gathers. Two per-request
offset tensors describe the layout:

  * ``compressed_base[r]`` — flat index where request r's compressed
    region begins. Topk indices ``topk_indices[token, j]`` are local to
    request r's compressed region (in ``[0, compressed_gather_len[r])``)
    and get rebased to flat space by adding ``compressed_base[r]``.
  * ``swa_base[r]`` — flat index where request r's SWA region begins.
    Per-query SWA indices are computed positionally as
    ``swa_base[r] + (pos - swa_len + 1 - gather_start) + j``.

This is the natural layout for ``flash_mla_sparse_fwd``'s ``kv: (s_kv, 1,
d_qk)`` argument, where ``s_kv`` is the total flat workspace length.

For SWA-only layers callers pass ``topk=0``, ``compressed_base = 0`` (the
compressed branch becomes a no-op) and any ``compress_ratio >= 1``.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import triton

from sglang.kernels.ops.attention.dsv4.dequant_k_cache import DIM_NOPE, DIM_ROPE
from sglang.srt.environ import envs
from sglang.srt.utils import ceil_align
from sglang.srt.utils.common import async_h2d
from sglang.srt.utils.community_opt import log_community_opt_once

# FlashMLA sparse prefill asserts ``params.topk % B_TOPK == 0``. B_TOPK is 64
# for the h_q=64 kernel and 128 for h_q=128; pad to 128 to satisfy both.
SPARSE_PREFILL_TOPK_ALIGNMENT = 128
# Bf16 workspace per-token width, matching ``dequantize_k_cache_paged``'s
# output: 448 fp8 nope (dequanted) + 64 bf16 rope = 512.
WORKSPACE_DIM = DIM_NOPE + DIM_ROPE
DSV4_Q8KV8_PREFILL_ENV = "SGLANG_DSV4_Q8KV8_PREFILL"
DSV4_Q8KV8_PREFILL_LOG_ENV = "SGLANG_DSV4_Q8KV8_PREFILL_LOG"

from sglang.kernels.ops.attention.dsv4.sparse_prefill_kernels import (
    _build_swa_token_ids_kernel,
    _combine_topk_swa_indices_kernel,
)

# Binary-search depth floor for the combine kernel: covers up to 127 requests per
# chunk with one compiled variant per (ratio, block) instead of one per bit length.
_COMBINE_MIN_SEARCH_STEPS = 7


def use_dsv4_q8kv8_sparse_prefill(dsv4_prefill_backend: str = "auto") -> bool:
    """Return whether DeepSeek-V4 sparse prefill should use Q8KV8.

    ``dsv4_prefill_backend`` is the production configuration. The environment
    variable remains as a debug override while the runtime path is being
    hardened: truthy values force Q8 on, falsy values force it off.
    """
    env_value = os.getenv(DSV4_Q8KV8_PREFILL_ENV)
    if env_value is not None:
        return env_value.lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    return dsv4_prefill_backend == "flashmla_sparse_q8"


class SparsePrefillWorkspace:
    """Backend-owned scratch storage for sparse prefill KV dequantization.

    The workspace contents are fully overwritten before every attention call,
    so token buckets and compression ratios can safely share one buffer. Sparse
    prefill executes eagerly and serially on the supported paths, which makes it
    safe to replace the scratch allocation when a larger extent is needed.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self._buffer: Optional[torch.Tensor] = None
        # Bumped on every reallocation, so contents tracked across layers (the
        # chunk cache's ``resident`` compressed gather) are known to be gone.
        self.generation = 0

    def get(
        self,
        num_tokens: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        assert num_tokens > 0
        current_capacity = self._buffer.shape[0] if self._buffer is not None else 0
        current_dtype = self._buffer.dtype if self._buffer is not None else None
        if num_tokens > current_capacity or dtype != current_dtype:
            self._buffer = torch.empty(
                (num_tokens, 1, WORKSPACE_DIM),
                dtype=dtype,
                device=self.device,
            )
            self.generation += 1
        return self._buffer[:num_tokens]


def combined_topk_width(topk: int, window_size: int) -> int:
    """Width of the padded combined_indices last dim that
    ``combine_topk_swa_indices`` would produce for these args."""
    return ceil_align(topk + window_size, SPARSE_PREFILL_TOPK_ALIGNMENT)


def combine_topk_swa_indices(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    query_pos: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    compressed_base: torch.Tensor,
    swa_base: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    out_indices: Optional[torch.Tensor] = None,
    out_lens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine top-k and SWA indices for flash_mla_sparse_fwd.

    Top-k indices are int32 [num_tokens, K] in each request's compressed region.
    Invalid entries within topk_len must be -1; later entries are ignored.
    query_start_loc can include a cross-chunk offset; query_pos is absolute.
    compressed_base and swa_base address the flat workspace. SWA-only layers use
    topk == 0, but compress_ratio must still be positive.

    Returns int32 indices [num_tokens, padded_topk_swa] and per-token scanned-prefix
    lengths, including -1 entries skipped by attention. Width is padded to 128.
    Rows are written whole and -1 padded, so preallocated out_indices / out_lens
    are overwritten entirely; rows past the last request are -1 with length 0.
    """
    assert topk_indices.dtype == torch.int32
    assert query_start_loc.dtype == torch.int32
    assert query_pos.dtype == torch.int32
    assert seq_lens.dtype == torch.int32
    assert gather_lens.dtype == torch.int32
    assert compressed_base.dtype == torch.int32
    assert swa_base.dtype == torch.int32
    assert compress_ratio >= 1, "compress_ratio must be >= 1 (use topk=0 for SWA-only)"
    assert topk_indices.shape[-1] >= topk, (
        f"topk_indices width {topk_indices.shape[-1]} must be >= topk {topk}"
    )

    num_tokens = topk_indices.shape[0]
    num_reqs = seq_lens.shape[0]
    combined_topk = combined_topk_width(topk, window_size)
    if out_indices is None:
        # The kernel writes whole -1 padded rows.
        combined_indices = torch.empty(
            (num_tokens, combined_topk),
            dtype=torch.int32,
            device=topk_indices.device,
        )
    else:
        assert out_indices.shape == (num_tokens, combined_topk)
        assert out_indices.dtype == torch.int32
        combined_indices = out_indices
    if out_lens is None:
        combined_lens = torch.empty(
            num_tokens, dtype=torch.int32, device=topk_indices.device
        )
    else:
        assert out_lens.shape == (num_tokens,)
        assert out_lens.dtype == torch.int32
        combined_lens = out_lens

    if num_tokens == 0:
        return combined_indices, combined_lens
    from sglang.srt.utils.community_opt import log_community_opt_once

    log_community_opt_once(
        "#41658 combine_topk_swa_indices",
        "token-parallel combine kernel enabled (tokens=%d, reqs=%d)",
        num_tokens,
        num_reqs,
    )
    block_t = 2  # Measured on B200 with num_warps=2, 16384 tokens, bs 1 to 64.
    _combine_topk_swa_indices_kernel[(triton.cdiv(num_tokens, block_t),)](
        combined_indices,
        combined_indices.stride(0),
        combined_lens,
        topk_indices,
        topk_indices.stride(0),
        query_start_loc,
        query_pos,
        seq_lens,
        gather_lens,
        compressed_base,
        swa_base,
        num_tokens,
        num_reqs,
        top_k=topk,
        COMPRESS_RATIO=compress_ratio,
        WINDOW_SIZE=window_size,
        WIDTH=combined_topk,
        PADDED_WIDTH=triton.next_power_of_2(combined_topk),
        BLOCK_T=block_t,
        # Bucketed so a new request count does not compile a new variant at serve
        # time; the extra steps are masked out once lo == hi.
        SEARCH_STEPS=max(max(num_reqs, 1).bit_length(), _COMBINE_MIN_SEARCH_STEPS),
        num_warps=2,
    )
    return combined_indices, combined_lens


def build_swa_token_ids(
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    full_to_swa: torch.Tensor,
    swa_window: int,
    total_swa: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a flat list of physical SWA-cache token IDs covering each
    request's positional union of every query's SWA window.

    Per request, the union spans seq positions
    ``[max(0, seq_len - extend - W + 1), seq_len)``, of length
    ``min(seq_len, extend + W - 1)``. Each position is translated through
    ``req_to_token`` (full kv-cache id) and then ``full_to_swa`` (SWA
    cache id) to land in the SWA-cache token-id space that
    ``dequantize_k_cache_paged`` consumes.

    Args:
        seq_lens: (num_reqs,) int32, per-request total sequence length.
        extend_seq_lens: (num_reqs,) int32, per-request query length.
        req_pool_indices: (num_reqs,) int32, per-request row in
            ``req_to_token``.
        req_to_token: (num_reqs_max, max_seq_len) int32. Full kv-cache id
            per (request, seq position).
        full_to_swa: (full_pool_size + extra,) int64. Maps full kv id to
            SWA-cache id.
        swa_window: int. SWA window size.
        total_swa: Number of token IDs to allocate, computed from CPU lengths.

    Returns:
        swa_token_ids: (total_swa,) int32, flat physical SWA-cache token IDs.
        swa_first_pos: (num_reqs,) int32, first seq position covered per req.
        swa_gather_lens: (num_reqs,) int32, gather length per request.
        swa_offsets: (num_reqs+1,) int32, exclusive cumsum of swa_gather_lens.
    """
    assert seq_lens.dtype == torch.int32
    assert extend_seq_lens.dtype == torch.int32
    assert req_pool_indices.dtype == torch.int32
    assert req_to_token.dtype == torch.int32
    assert full_to_swa.dtype == torch.int64

    num_reqs = seq_lens.shape[0]
    device = seq_lens.device

    swa_gather_lens = torch.minimum(seq_lens, extend_seq_lens + (swa_window - 1)).to(
        torch.int32
    )
    swa_first_pos = (seq_lens - swa_gather_lens).to(torch.int32)
    swa_offsets = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
    swa_offsets[1:] = torch.cumsum(swa_gather_lens, dim=0).to(torch.int32)
    swa_token_ids = torch.empty(total_swa, dtype=torch.int32, device=device)
    if total_swa == 0:
        return swa_token_ids, swa_first_pos, swa_gather_lens, swa_offsets

    NUM_WORKERS = 128
    _build_swa_token_ids_kernel[(num_reqs, NUM_WORKERS)](
        swa_token_ids,
        swa_first_pos,
        swa_gather_lens,
        swa_offsets,
        req_pool_indices,
        req_to_token,
        req_to_token.stride(0),
        full_to_swa,
    )
    return swa_token_ids, swa_first_pos, swa_gather_lens, swa_offsets


@dataclass
class CompressedGather:
    """Positional layout of one top-k compressed cache inside the workspace."""

    # (num_reqs * c_max,) int32, or (sum of per-request rows,) in the compact layout
    flat_token_ids: torch.Tensor
    page_size: int
    compressed_base: torch.Tensor  # (num_reqs,) int32
    swa_base: torch.Tensor  # (num_reqs,) int32
    # Output buffers of the last combine; the kernel rewrites whole rows.
    combined_indices: Optional[torch.Tensor] = None
    combined_lens: Optional[torch.Tensor] = None
    # (top-k version, raw_indices data_ptr) the buffers above were combined from.
    combined_key: Optional[tuple] = None


@dataclass
class SparsePrefillChunkCache:
    """Cache prefill-chunk metadata shared across layers.

    Fields depend on request/token mappings and compressed page tables,
    not per-layer k_cache; layer-specific top-k combinations are not cached.
    """

    # Geometry computed once per chunk.
    num_reqs: int
    num_qo_tokens: int
    # Actual maximum sequence length in this forward. CUDA-graph metadata may
    # have a much wider page table sized for the capture limit; gather only the
    # live sequence extent instead of materializing that padded capacity.
    max_seq_len: int
    # Model's SWA window — the per-query attention range. Used by
    # combine_topk_swa_indices' WINDOW_SIZE and by build_swa_token_ids's
    # gather_lens. Must match SWA_WINDOW from the backend (e.g. 128), NOT
    # the SWA pool's storage page size (often 256).
    swa_window_size: int
    # SWA cache pool's storage page size — used as the dequant kernel's
    # ``page_size`` so that ``slot // page_size`` recovers the right page.
    swa_page_size: int
    seq_lens: torch.Tensor  # (num_reqs,) int32
    query_start_loc: torch.Tensor  # (num_reqs+1,) int32, query rows per request
    query_pos: torch.Tensor  # (num_qo_tokens,) int32, sequence position per row

    # SWA-side (every layer needs these, all chunk-invariant).
    swa_token_ids: torch.Tensor  # (total_swa,) int32
    swa_first_pos: torch.Tensor  # (num_reqs,) int32
    swa_gather_lens: torch.Tensor  # (num_reqs,) int32
    swa_offsets: torch.Tensor  # (num_reqs+1,) int32

    # c0 pre-computed combine output (entire input set is chunk-invariant).
    c0_combined_indices: torch.Tensor = field(default=None)
    c0_combined_lens: torch.Tensor = field(default=None)
    # c128: positional layout of the c128 cache + pre-computed combine.
    c128_flat_token_ids: Optional[torch.Tensor] = None  # (num_reqs * c128_max,) int32
    c128_combined_indices: Optional[torch.Tensor] = None
    c128_combined_lens: Optional[torch.Tensor] = None

    # Top-k compressed caches (c1 / c2 / c4), keyed by compress ratio.
    compressed: Dict[int, CompressedGather] = field(default_factory=dict)

    # Host lengths for the compact compressed layout (vllm#59131); None keeps
    # the num_reqs * c_max padded layout.
    seq_lens_cpu: Optional[List[int]] = None
    query_lens_cpu: Optional[List[int]] = None
    # The compressed gather the workspace head currently holds, as
    # (ratio, source layer, rows, workspace generation, data_ptr, dtype); None
    # once anything else was written there. Lives on the per-forward chunk cache.
    resident: Optional[tuple] = None

    @classmethod
    def build(
        cls,
        seq_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        query_lens: torch.Tensor,
        query_pos: torch.Tensor,
        req_pool_indices: torch.Tensor,
        req_to_token: torch.Tensor,
        full_to_swa: torch.Tensor,
        swa_window_size: int,
        swa_page_size: int,
        num_qo_tokens: int,
        max_seq_len: int,
        total_swa: int,
        seq_lens_cpu: Optional[List[int]] = None,
        query_lens_cpu: Optional[List[int]] = None,
    ) -> "SparsePrefillChunkCache":
        """``query_lens`` / ``query_pos``: the rows this forward runs (the extend, or
        a CP rank's interleaved share of it); the SWA gather spans the whole extend.
        ``seq_lens_cpu`` / ``query_lens_cpu`` mirror ``seq_lens`` / ``query_lens``
        on the host and enable the compact compressed layout."""
        device = seq_lens.device
        num_reqs = seq_lens.shape[0]

        query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
        query_start_loc[1:] = torch.cumsum(query_lens, dim=0).to(torch.int32)

        swa_token_ids, swa_first_pos, swa_gather_lens, swa_offsets = (
            build_swa_token_ids(
                seq_lens=seq_lens,
                extend_seq_lens=extend_seq_lens,
                req_pool_indices=req_pool_indices,
                req_to_token=req_to_token,
                full_to_swa=full_to_swa,
                swa_window=swa_window_size,
                total_swa=total_swa,
            )
        )

        cache = cls(
            num_reqs=num_reqs,
            num_qo_tokens=num_qo_tokens,
            max_seq_len=max_seq_len,
            swa_window_size=swa_window_size,
            swa_page_size=swa_page_size,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            query_pos=query_pos,
            swa_token_ids=swa_token_ids,
            swa_first_pos=swa_first_pos,
            swa_gather_lens=swa_gather_lens,
            swa_offsets=swa_offsets,
            seq_lens_cpu=seq_lens_cpu,
            query_lens_cpu=query_lens_cpu,
        )

        # Pre-compute the c0 combine output: TOPK=0, compressed_base=0,
        # swa_base = swa_offsets[:-1]. All inputs are chunk-invariant.
        zero_topk = torch.zeros((num_qo_tokens, 1), dtype=torch.int32, device=device)
        zero_compressed_base = torch.zeros(num_reqs, dtype=torch.int32, device=device)
        c0_swa_base = swa_offsets[:-1].to(torch.int32)
        cache.c0_combined_indices, cache.c0_combined_lens = combine_topk_swa_indices(
            topk_indices=zero_topk,
            query_start_loc=query_start_loc,
            query_pos=query_pos,
            seq_lens=seq_lens,
            gather_lens=swa_gather_lens,
            compressed_base=zero_compressed_base,
            swa_base=c0_swa_base,
            window_size=swa_window_size,
            compress_ratio=1,
            topk=0,
        )
        return cache

    def ensure_c128(self, c128_page_indices: torch.Tensor) -> None:
        """Populate c128-side fields from per-query c128 page indices.

        ``c128_page_indices[q, j]`` carries slot ids derived from
        ``page_table[q]`` (request-keyed; same across queries of a request)
        but masked per-token by ``j < seq_lens_casual[q] // 128`` — entries
        beyond that are -1. We need a row whose mask covers every j the
        combine kernel might reference, i.e. up to ``seq_lens[r] // 128``;
        that's the *last* query's mask. Pulling the first query in a fresh
        prefill (``seq_lens_casual = 1``) yields an all-`-1` row that
        clamp_min(0) collapses to slot 0, sending dequant to a polluted
        slot and producing garbage c128 entries.
        """
        if self.c128_flat_token_ids is not None:
            return
        device = self.seq_lens.device
        c128_max = max(self.max_seq_len // 128, 1)
        assert c128_max <= c128_page_indices.shape[-1], (
            f"live c128 extent {c128_max} exceeds metadata capacity "
            f"{c128_page_indices.shape[-1]}"
        )
        # a request without rows on this rank gathers into a region nothing reads
        last_q_per_req = (self.query_start_loc[1:] - 1).clamp_min(0).long()
        per_req_c128 = c128_page_indices.narrow(1, 0, c128_max).index_select(
            0, last_q_per_req
        )
        # Clamp -1 -> 0 so dequant doesn't OOB; combine masks the invalid
        # tail via topk_len.
        flat_c128_ids = per_req_c128.reshape(-1).clamp_min(0).to(torch.int32)
        compressed_base = (
            torch.arange(self.num_reqs, dtype=torch.int32, device=device) * c128_max
        ).to(torch.int32)
        total_compressed = self.num_reqs * c128_max
        # Pre-compute the c128 combine output. topk_indices[q, j] = j is the
        # arange-broadcast pattern; we materialize it once here so the
        # combine kernel can read it like any other topk tensor.
        topk_indices = (
            torch.arange(c128_max, dtype=torch.int32, device=device)[None, :]
            .expand(self.num_qo_tokens, -1)
            .contiguous()
        )
        swa_base = (total_compressed + self.swa_offsets[:-1]).to(torch.int32)
        combined_indices, combined_lens = combine_topk_swa_indices(
            topk_indices=topk_indices,
            query_start_loc=self.query_start_loc,
            query_pos=self.query_pos,
            seq_lens=self.seq_lens,
            gather_lens=self.swa_gather_lens,
            compressed_base=compressed_base,
            swa_base=swa_base,
            window_size=self.swa_window_size,
            compress_ratio=128,
            topk=c128_max,
        )

        self.c128_flat_token_ids = flat_c128_ids
        self.c128_combined_indices = combined_indices
        self.c128_combined_lens = combined_lens

    def ensure_compressed(
        self,
        compress_ratio: int,
        page_table: torch.Tensor,
        c_page_size: int,
    ) -> CompressedGather:
        """``page_table`` is (num_qo_tokens, max_blocks); rows within a request
        are duplicates. The combine output is per-layer (depends on the
        layer's remapped topk_indices), so we only cache the gather-side
        scaffolding plus compressed/swa bases.
        """
        gather = self.compressed.get(compress_ratio)
        if gather is not None:
            return gather
        if (
            self.seq_lens_cpu is not None
            and self.query_lens_cpu is not None
            and envs.SGLANG_OPT_DSV4_COMPACT_COMPRESSED_GATHER.get()
        ):
            gather = self._compact_compressed_gather(
                compress_ratio, page_table, c_page_size
            )
            self.compressed[compress_ratio] = gather
            return gather
        device = self.seq_lens.device
        c_max = max(self.max_seq_len // compress_ratio, 1)
        c_capacity = page_table.shape[-1] * c_page_size
        assert c_max <= c_capacity, (
            f"live c{compress_ratio} extent {c_max} exceeds metadata capacity {c_capacity}"
        )
        first_q_per_req = (
            self.query_start_loc[:-1].clamp_max(self.num_qo_tokens - 1).long()
        )
        num_blocks = (c_max + c_page_size - 1) // c_page_size
        assert num_blocks <= page_table.shape[1]
        per_req_page_table = page_table.narrow(1, 0, num_blocks).index_select(
            0, first_q_per_req
        )

        k_arange = torch.arange(c_max, dtype=torch.int32, device=device)
        block_idx = (k_arange // c_page_size).long()
        in_page = (k_arange % c_page_size).to(torch.int32)
        token_ids_2d = (
            per_req_page_table.index_select(1, block_idx) * c_page_size + in_page
        ).to(torch.int32)
        flat_ids = token_ids_2d.reshape(-1).clamp_min(0)
        total_compressed = self.num_reqs * c_max
        compressed_base = (
            torch.arange(self.num_reqs, dtype=torch.int32, device=device) * c_max
        ).to(torch.int32)
        swa_base = (total_compressed + self.swa_offsets[:-1]).to(torch.int32)

        gather = CompressedGather(
            flat_token_ids=flat_ids,
            page_size=c_page_size,
            compressed_base=compressed_base,
            swa_base=swa_base,
        )
        self.compressed[compress_ratio] = gather
        return gather

    def _compact_compressed_gather(
        self,
        compress_ratio: int,
        page_table: torch.Tensor,
        c_page_size: int,
    ) -> CompressedGather:
        """vllm#59131 layout: request r owns ``seq_len[r] // ratio`` rows (0 when it
        has no query rows here, e.g. under CP), packed back to back. The combine
        kernel only emits top-k values below ``(pos + 1) // ratio <= seq_len //
        ratio``, so every referenced row is the same KV row the padded layout
        holds, at a different offset."""
        device = self.seq_lens.device
        c_lens = [
            s // compress_ratio if q > 0 else 0
            for s, q in zip(self.seq_lens_cpu, self.query_lens_cpu, strict=True)
        ]
        bases = [0] * self.num_reqs
        for r in range(1, self.num_reqs):
            bases[r] = bases[r - 1] + c_lens[r - 1]
        total = bases[-1] + c_lens[-1]
        c_live = max(c_lens)
        c_capacity = page_table.shape[-1] * c_page_size
        assert c_live <= c_capacity, (
            f"live c{compress_ratio} extent {c_live} exceeds metadata capacity {c_capacity}"
        )
        lens_bases = async_h2d([c_lens, bases], dtype=torch.int32, device=device)
        c_lens_dev, compressed_base = lens_bases[0], lens_bases[1]
        first_q_per_req = (
            self.query_start_loc[:-1].clamp_max(self.num_qo_tokens - 1).long()
        )
        num_blocks = (c_live + c_page_size - 1) // c_page_size
        per_req_page_table = page_table.narrow(1, 0, num_blocks).index_select(
            0, first_q_per_req
        )
        req = torch.repeat_interleave(
            torch.arange(self.num_reqs, device=device), c_lens_dev, output_size=total
        )
        p = torch.arange(total, dtype=torch.int32, device=device) - compressed_base[req]
        block = per_req_page_table.reshape(-1)[req * num_blocks + (p // c_page_size)]
        flat_ids = (block * c_page_size + p % c_page_size).to(torch.int32).clamp_min(0)
        swa_base = (total + self.swa_offsets[:-1]).to(torch.int32)
        padded = self.num_reqs * max(self.max_seq_len // compress_ratio, 1)
        log_community_opt_once(
            "vllm#59131 compact_compressed_gather",
            "c%d gathers %d compressed rows instead of %d (reqs=%d)",
            compress_ratio,
            total,
            padded,
            self.num_reqs,
        )
        return CompressedGather(
            flat_token_ids=flat_ids,
            page_size=c_page_size,
            compressed_base=compressed_base,
            swa_base=swa_base,
        )

    def combine_compressed(
        self,
        compress_ratio: int,
        sparse_raw_indices: torch.Tensor,
        topk_version: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``sparse_raw_indices`` is the top-k as request-local compressed
        positions, already the workspace coordinate ``combine_topk_swa_indices``
        expects, so no remap is needed.

        ``topk_version`` identifies the top-k the index_source layer last wrote;
        while it and the buffer are unchanged, the previous combine is returned
        (vllm#59131). None always recombines.
        """
        gather = self.compressed[compress_ratio]
        topk = sparse_raw_indices.shape[-1]
        key = (
            None
            if topk_version is None
            else (topk_version, sparse_raw_indices.data_ptr(), topk)
        )
        if key is not None and gather.combined_key == key:
            return gather.combined_indices, gather.combined_lens
        if gather.combined_indices is None:
            # The kernel writes whole -1 padded rows and every length.
            device = self.seq_lens.device
            gather.combined_indices = torch.empty(
                (self.num_qo_tokens, combined_topk_width(topk, self.swa_window_size)),
                dtype=torch.int32,
                device=device,
            )
            gather.combined_lens = torch.empty(
                self.num_qo_tokens, dtype=torch.int32, device=device
            )
        gather.combined_key = key
        return combine_topk_swa_indices(
            topk_indices=sparse_raw_indices,
            query_start_loc=self.query_start_loc,
            query_pos=self.query_pos,
            seq_lens=self.seq_lens,
            gather_lens=self.swa_gather_lens,
            compressed_base=gather.compressed_base,
            swa_base=gather.swa_base,
            window_size=self.swa_window_size,
            compress_ratio=compress_ratio,
            topk=topk,
            out_indices=gather.combined_indices,
            out_lens=gather.combined_lens,
        )
