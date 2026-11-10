from __future__ import annotations

from enum import Enum, IntEnum, auto
from typing import Callable, Dict, List, Optional, Tuple

import torch

from sglang.srt.environ import envs

_FLASHINFER_TIE_BREAK_VALUES = {
    "small": 1,
    "large": 2,
}


class TopkTransformMethod(IntEnum):
    # Transform topk indices to indices to the page table (page_size = 1)
    PAGED = auto()
    # Transform topk indices to indices to ragged kv (non-paged)
    RAGGED = auto()


class DSATopKBackend(Enum):
    SGL_KERNEL = "sgl-kernel"
    TORCH = "torch"
    FLASHINFER = "flashinfer"

    def is_sgl_kernel(self) -> bool:
        return self == DSATopKBackend.SGL_KERNEL

    def is_torch(self) -> bool:
        return self == DSATopKBackend.TORCH

    def is_flashinfer(self) -> bool:
        return self == DSATopKBackend.FLASHINFER

    def topk_func(
        self,
        score: torch.Tensor,
        lengths: torch.Tensor,
        topk: int,
        row_starts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.is_sgl_kernel():
            from sgl_kernel import fast_topk_v2

            return fast_topk_v2(score, lengths, topk, row_starts=row_starts)
        if self.is_torch():
            return _topk_unfused(
                score,
                lengths,
                topk,
                row_starts=row_starts,
                topk_op=torch.topk,
                topk_op_kwargs={"dim": -1},
            )
        if self.is_flashinfer():
            import flashinfer

            return _topk_unfused(
                score,
                lengths,
                topk,
                row_starts=row_starts,
                topk_op=flashinfer.top_k,
                topk_op_kwargs={
                    "sorted": False,
                    "deterministic": envs.SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC.get(),
                    "tie_break": _flashinfer_tie_break_value(),
                    "dsa_graph_safe": True,
                },
            )
        raise RuntimeError(f"Unsupported {self = }.")

    def topk_transform(
        self,
        logits: torch.Tensor,
        lengths: torch.Tensor,
        topk: int,
        topk_transform_method: TopkTransformMethod,
        attn_metadata,
        cu_seqlens_q_topk: Optional[torch.Tensor] = None,
        topk_indices_offset: Optional[torch.Tensor] = None,
        row_starts: Optional[torch.Tensor] = None,
        batch_idx_list: Optional[List[int]] = None,
        force_unfused_topk: bool = False,
        row_range: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if not envs.SGLANG_DSA_FUSE_TOPK.get() or force_unfused_topk:
            return self.topk_func(logits, lengths, topk, row_starts=row_starts)

        # Decode-shaped PAGED top-k (plain decode AND spec verify / draft-extend,
        # whose expanded rows match the same shape) routes to the DeepSeek-V4 top-k
        # v2 JIT kernel, which fuses top-k selection and the page-table transform in
        # one launch and consumes the indexer's own page_size>=1 table directly, so
        # no page_size=1 table is materialized. Shared by DeepSeek-V3.2 and GLM DSA.
        # This is a deterministic dispatch on the work shape, not a best-effort
        # attempt: the fused-decode CUDA graph drops the page_size=1 table for
        # exactly this case (see dsa_drop_wide_page_table), so once the shape
        # matches we commit to v2 and never silently fall back to the legacy
        # page_size=1 path from here.
        if (
            envs.SGLANG_OPT_USE_TOPK_V2.get()
            and topk_transform_method == TopkTransformMethod.PAGED
            and row_starts is None
            and batch_idx_list is None
            and 0 < topk <= 2048
            and lengths.shape[0]
            == logits.shape[0]
            == attn_metadata.real_page_table.shape[0]
        ):
            return _topk_transform_v2_paged(logits, lengths, topk, attn_metadata)

        # Single-request PAGED prefill: every row shares the one page-table row
        # and its scores start at column 0 (ks == 0), so the v2 kernel can read
        # a stride-0 broadcast of that row. batch_idx_list, if any, is all zeros.
        # A one-request slice of a larger batch (the K3 MLA CP indexer, which
        # passes no row_starts because its K starts at column 0) uses its own
        # page-table row the same way.
        page_table = getattr(attn_metadata, "real_page_table", None)
        single_row = None
        if page_table is not None and page_table.shape[0] == 1:
            single_row = 0
        elif (
            page_table is not None
            and row_starts is None
            and batch_idx_list is not None
            and len(batch_idx_list) == 1
        ):
            single_row = int(batch_idx_list[0])
        if (
            envs.SGLANG_DSA_PREFILL_TOPK_V2.get()
            and topk_transform_method == TopkTransformMethod.PAGED
            and 0 < topk <= 2048
            and single_row is not None
            and page_table.dtype == torch.int32
            and logits.dtype == torch.float32
            and logits.stride(1) == 1
            and logits.stride(0) % 4 == 0
        ):
            return _topk_transform_v2_paged_prefill(
                logits, lengths, topk, attn_metadata, single_row
            )

        # Multi-request PAGED prefill: rows of request b are contiguous and its
        # scores start at column ks_b = sum(seq_lens[:b]); run v2 once per
        # request on that column window with the request's page-table row.
        # ``row_range`` locates a row-chunk of the logits (the indexer splits
        # them under memory pressure) within the batch's rows. Batches with more
        # than _V2_PREFILL_MAX_SEGMENTS requests send only their long requests to
        # v2 and the rest to the legacy kernel (sgl-kernel only), so a long
        # request never goes through the [rows, max_seq_len] page_table_1 gather.
        if (
            envs.SGLANG_DSA_PREFILL_TOPK_V2.get()
            and topk_transform_method == TopkTransformMethod.PAGED
            and (batch_idx_list is None or row_range is not None)
            and row_starts is not None
            and 0 < topk <= 2048
            and logits.dtype == torch.float32
            and logits.stride(1) == 1
            and logits.stride(0) % 4 == 0
        ):
            segments = _v2_prefill_segments(
                logits, attn_metadata, row_range, allow_mixed=self.is_sgl_kernel()
            )
            if segments is not None:
                return _topk_transform_v2_paged_prefill_multi(
                    logits, lengths, topk, attn_metadata, segments, row_starts
                )

        # The legacy transforms below read attn_metadata.page_table_1 (page_size=1),
        # which is always present here: the fold only drops it for the decode case
        # dispatched to v2 above.
        assert attn_metadata.page_table_1 is not None

        if self.is_sgl_kernel():
            from sgl_kernel import (
                fast_topk_transform_fused,
                fast_topk_transform_ragged_fused,
            )

            if topk_transform_method == TopkTransformMethod.PAGED:
                page_table_size_1 = (
                    attn_metadata.page_table_1[batch_idx_list]
                    if batch_idx_list is not None
                    else attn_metadata.page_table_1
                )
                return fast_topk_transform_fused(
                    score=logits,
                    lengths=lengths,
                    page_table_size_1=page_table_size_1,
                    cu_seqlens_q=cu_seqlens_q_topk,
                    topk=topk,
                    row_starts=row_starts,
                )
            if topk_transform_method == TopkTransformMethod.RAGGED:
                if topk_indices_offset is None:
                    raise RuntimeError(
                        "RAGGED topk_transform requires topk_indices_offset; "
                        "expected extend-without-speculative metadata."
                    )
                return fast_topk_transform_ragged_fused(
                    score=logits,
                    lengths=lengths,
                    topk_indices_offset=topk_indices_offset,
                    topk=topk,
                    row_starts=row_starts,
                )
            raise RuntimeError(f"Unsupported {topk_transform_method = }.")

        if self.is_flashinfer():
            import flashinfer

            if topk_transform_method == TopkTransformMethod.PAGED:
                row_to_batch, local_row_starts = _build_flashinfer_paged_args(
                    attn_metadata=attn_metadata,
                    row_starts=row_starts,
                    cu_seqlens_q_topk=cu_seqlens_q_topk,
                    batch_idx_list=batch_idx_list,
                    device=logits.device,
                    num_rows=logits.shape[0],
                )
                return flashinfer.top_k_page_table_transform(
                    logits.contiguous(),
                    attn_metadata.page_table_1.contiguous(),
                    lengths.contiguous(),
                    topk,
                    row_to_batch=row_to_batch,
                    deterministic=envs.SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC.get(),
                    tie_break=_flashinfer_tie_break_value(),
                    dsa_graph_safe=True,
                    row_starts=local_row_starts,
                )
            if topk_transform_method == TopkTransformMethod.RAGGED:
                if topk_indices_offset is None:
                    raise RuntimeError(
                        "RAGGED topk_transform requires topk_indices_offset; "
                        "expected extend-without-speculative metadata."
                    )
                return flashinfer.top_k_ragged_transform(
                    logits.contiguous(),
                    topk_indices_offset.contiguous(),
                    lengths.contiguous(),
                    topk,
                    deterministic=envs.SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC.get(),
                    tie_break=_flashinfer_tie_break_value(),
                    dsa_graph_safe=True,
                    row_starts=row_starts,
                )
            raise RuntimeError(f"Unsupported {topk_transform_method = }.")

        raise RuntimeError(f"Unsupported {self = } for SGLANG_DSA_FUSE_TOPK.")


def _topk_unfused(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
    topk_op: Callable[..., Tuple[torch.Tensor, torch.Tensor]] = torch.topk,
    topk_op_kwargs: Optional[Dict[str, object]] = None,
) -> torch.Tensor:
    batch_size, max_score_len = score.shape
    topk_indices = score.new_full((batch_size, topk), -1, dtype=torch.int32)
    if batch_size == 0 or topk == 0 or max_score_len == 0:
        return topk_indices

    if row_starts is None:
        row_starts = torch.zeros_like(lengths, dtype=torch.int32, device=score.device)
    else:
        row_starts = row_starts.to(dtype=torch.int32, device=score.device)
    lengths = lengths.to(dtype=torch.int32, device=score.device)

    col_indices = torch.arange(max_score_len, dtype=torch.int32, device=score.device)
    col_indices = col_indices.unsqueeze(0)
    row_starts_unsqueezed = row_starts.unsqueeze(1)
    row_ends_unsqueezed = (row_starts + lengths).unsqueeze(1)
    valid_mask = (col_indices >= row_starts_unsqueezed) & (
        col_indices < row_ends_unsqueezed
    )

    masked_logits = score.masked_fill(~valid_mask, float("-inf"))
    valid_topk = min(topk, max_score_len)
    topk_kwargs = topk_op_kwargs or {}
    topk_scores, topk_col_indices = topk_op(masked_logits, valid_topk, **topk_kwargs)
    topk_local_indices = topk_col_indices.to(torch.int32) - row_starts_unsqueezed
    topk_local_indices = topk_local_indices.masked_fill(
        topk_scores == float("-inf"), -1
    )
    topk_indices[:, :valid_topk] = topk_local_indices

    return topk_indices


def _topk_transform_v2_paged(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    attn_metadata,
) -> torch.Tensor:
    """Fused top-k + page-table transform via the DeepSeek-V4 v2 JIT kernel.

    Returns the transformed page indices ``(num_rows, topk)`` int32 (physical
    page_size=1 KV slots, ``-1`` padded) -- identical in meaning to
    ``fast_topk_transform_fused`` / ``flashinfer.top_k_page_table_transform``.
    The kernel selects, per row, the top-k of ``logits[row, :lengths[row]]`` and
    maps each selected position ``p`` through the page table as
    ``real_page_table[row, p // page_size] * page_size + (p % page_size)``. Feeding
    it the indexer's compact ``real_page_table`` (page_size = pool page size,
    typically 64) yields the same physical slots as gathering the page_size=1
    table, without materializing that wide table.

    This is a committed contract, not a best-effort path: ``topk_transform`` routes
    here only for the decode-shaped PAGED case, and the fused-decode CUDA graph
    drops the page_size=1 table for exactly this case (see
    ``dsa_drop_wide_page_table``). The preconditions below are therefore
    invariants the caller must uphold -- they assert (raise) on violation rather
    than fall back to the slow legacy path (which may not even have a page_size=1
    table to fall back to) or silently paper over bad input (padding, recomputing
    the plan) at the cost of the performance this path exists to deliver.

    ``lengths`` entries must be NON-NEGATIVE: the kernel reads them as
    ``uint32_t``, so a negative row length (DP-padded / idle-companion rows)
    reinterprets as ~4e9 tokens and illegal-addresses. Metadata producers clamp
    padded rows to 0 (see ``fused_dsa_draft_extend_metadata`` /
    ``seqlens_expand_kernel``); 0 takes the trivial all-(-1) output path.
    """
    from sglang.kernels.ops.attention.dsv4.topk import topk_transform_512_v2
    from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

    num_rows = logits.shape[0]

    # The indexer (DeepGEMM) emits fp32 scores with unit row stride and a 16B-aligned
    # row stride (a multiple of 4), which is exactly the kernel's ABI (it checks
    # score_stride % 4 == 0 with strides {S, 1}). This holds even though the scores
    # may be a padded view (stride(0) > width, so not `is_contiguous()`); assert the
    # real requirement rather than force a contiguous copy of the wide score buffer.
    assert (
        logits.dtype == torch.float32
        and logits.stride(1) == 1
        and logits.stride(0) % 4 == 0
    ), f"v2 top-k expects fp32 scores with unit row stride and 16B-aligned score_stride, got {logits.dtype=} {logits.stride()=}"
    assert 0 < topk <= 2048, f"v2 top-k supports 0 < topk <= 2048, got {topk=}"

    page_table = attn_metadata.real_page_table
    assert page_table.dtype == torch.int32
    lengths_i32 = lengths.to(torch.int32)

    # The plan is preprocessed once per forward (DSAMetadata.topk_v2_plan,
    # refreshed in-place under CUDA graph) and reused across layers. A missing or
    # mismatched plan means the caller skipped that preprocessing -- fail loudly
    # rather than silently recompute it per layer.
    plan = attn_metadata.topk_v2_plan
    assert (
        plan is not None and plan.shape[0] == num_rows + 1
    ), "topk_v2_plan must be preprocessed per forward (see DSAMetadata.topk_v2_plan)"

    page_size = get_token_to_kv_pool().page_size
    out = logits.new_full((num_rows, topk), -1, dtype=torch.int32)
    topk_transform_512_v2(logits, lengths_i32, page_table, out, page_size, plan)
    return out


def _topk_transform_v2_paged_prefill(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    attn_metadata,
    page_table_row: int = 0,
) -> torch.Tensor:
    """Prefill variant of :func:`_topk_transform_v2_paged` for rows of one
    request: the plan is built from this call's row lengths (prefill metadata
    carries no ``topk_v2_plan``) and the request's page-table row is broadcast
    per row."""
    from sglang.kernels.ops.attention.dsv4.topk import (
        plan_topk_v2,
        topk_transform_512_v2,
    )
    from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

    num_rows = logits.shape[0]
    lengths_i32 = lengths.to(torch.int32)
    page_table = attn_metadata.real_page_table[
        page_table_row : page_table_row + 1
    ].expand(num_rows, -1)
    plan = plan_topk_v2(lengths_i32)
    out = logits.new_full((num_rows, topk), -1, dtype=torch.int32)
    topk_transform_512_v2(
        logits, lengths_i32, page_table, out, get_token_to_kv_pool().page_size, plan
    )
    return out


# Per-request v2 costs ~3 launches per request, so it only pays on long rows
# (B200, topk 2048: [492K+1792, 0+1792] 2.36 -> 1.15 ms; 8 x fresh 2048
# 0.08 -> 0.18 ms). Short batches stay on the legacy kernel; batches with more
# than _V2_PREFILL_MAX_SEGMENTS requests send only requests of at least
# _V2_PREFILL_MIN_MAX_SEQ tokens to v2.
_V2_PREFILL_MAX_SEGMENTS = 8
_V2_PREFILL_MIN_MAX_SEQ = 8192


def _v2_prefill_segments(
    logits: torch.Tensor,
    attn_metadata,
    row_range: Optional[Tuple[int, int]] = None,
    allow_mixed: bool = False,
):
    """Host-side (row_start, row_end, col_start, col_len, batch, req_row_start,
    extend_len, use_v2) per request of a multi-request PAGED prefill, in
    ``logits`` row coordinates, or None when the batch does not qualify.
    ``logits`` holds batch rows ``row_range`` (default: all of them). Segments
    with use_v2=False go to the legacy kernel; that split is only produced when
    ``allow_mixed``. Uses only CPU metadata, so no device sync."""
    page_table = getattr(attn_metadata, "real_page_table", None)
    seq_lens_cpu = getattr(attn_metadata, "indexer_seq_lens_cpu", None)
    extend_lens = getattr(attn_metadata, "dsa_extend_seq_lens_list", None)
    if page_table is None or seq_lens_cpu is None or extend_lens is None:
        return None
    bs = page_table.shape[0]
    if (
        page_table.dtype != torch.int32
        or bs <= 1
        or len(extend_lens) != bs
        or seq_lens_cpu.shape[0] != bs
        or int(seq_lens_cpu.max()) < _V2_PREFILL_MIN_MAX_SEQ
    ):
        return None
    mixed = bs > _V2_PREFILL_MAX_SEGMENTS
    if mixed and not allow_mixed:
        return None
    lo, hi = (0, sum(extend_lens)) if row_range is None else row_range
    if hi - lo != logits.shape[0] or hi > sum(extend_lens):
        return None
    segments = []
    row = col = 0
    for b, (q, k) in enumerate(zip(extend_lens, seq_lens_cpu.tolist())):
        r0, r1 = max(row, lo), min(row + q, hi)
        if r0 < r1:
            if col + k > logits.shape[1]:
                return None
            use_v2 = not mixed or k >= _V2_PREFILL_MIN_MAX_SEQ
            segments.append((r0 - lo, r1 - lo, col, k, b, r0 - row, q, use_v2))
        row += q
        col += k
    # A window that does not start on a 16 B boundary is read from the aligned
    # column below it through the page_size=1 table.
    if any(s[7] and s[2] % 4 != 0 for s in segments) and (
        getattr(attn_metadata, "page_table_1", None) is None
    ):
        return None
    return segments


def _topk_transform_v2_paged_prefill_multi(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    attn_metadata,
    segments,
    row_starts: torch.Tensor,
) -> torch.Tensor:
    from sglang.kernels.ops.attention.dsv4.topk import (
        plan_topk_v2,
        topk_transform_512_v2,
    )
    from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

    page_size = get_token_to_kv_pool().page_size
    lengths_i32 = lengths.to(torch.int32)
    out = logits.new_full((logits.shape[0], topk), -1, dtype=torch.int32)
    legacy_run = []
    for seg in segments:
        r0, r1, c0, k, b, req_row, q, use_v2 = seg
        if not use_v2:
            legacy_run.append(seg)
            continue
        _legacy_paged_prefill_rows(
            logits, lengths_i32, topk, attn_metadata, legacy_run, row_starts, out
        )
        legacy_run = []
        pad = c0 % 4
        if pad == 0:
            seg_lens = lengths_i32[r0:r1]
            page_table = attn_metadata.real_page_table[b : b + 1].expand(r1 - r0, -1)
            topk_transform_512_v2(
                logits[r0:r1, c0 : c0 + k],
                seg_lens,
                page_table,
                out[r0:r1],
                page_size,
                plan_topk_v2(seg_lens),
            )
            continue
        # The window starts off a 16 B boundary: read it from column c0 - pad
        # with the pad columns at -inf (they hold the previous request's K,
        # which these rows never use) and lengths + pad, and map positions
        # through this request's page_size=1 row shifted right by pad.
        # Row j of the request has length k - q + j + 1; rows of length
        # <= topk take the kernel's trivial path, which never reads scores
        # and must see the unshifted table, so they are launched apart.
        n_trivial = min(max(topk - (k - q) - req_row, 0), r1 - r0)
        rm = r0 + n_trivial
        if n_trivial > 0:
            seg_lens = lengths_i32[r0:rm]
            topk_transform_512_v2(
                logits[r0:rm, c0 - pad : c0 - pad + k],
                seg_lens,
                attn_metadata.real_page_table[b : b + 1].expand(n_trivial, -1),
                out[r0:rm],
                page_size,
                plan_topk_v2(seg_lens),
            )
        if rm < r1:
            window = logits[rm:r1, c0 - pad : c0 + k]
            window[:, :pad] = float("-inf")
            page_table_1 = attn_metadata.page_table_1
            shifted_row = torch.cat(
                [page_table_1.new_full((pad,), -1), page_table_1[b, :k]]
            )
            seg_lens = lengths_i32[rm:r1] + pad
            topk_transform_512_v2(
                window,
                seg_lens,
                shifted_row[None].expand(r1 - rm, -1),
                out[rm:r1],
                1,
                plan_topk_v2(seg_lens),
            )
    _legacy_paged_prefill_rows(
        logits, lengths_i32, topk, attn_metadata, legacy_run, row_starts, out
    )
    return out


def _legacy_paged_prefill_rows(
    logits: torch.Tensor,
    lengths_i32: torch.Tensor,
    topk: int,
    attn_metadata,
    run,
    row_starts: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """sgl-kernel top-k over a run of row-adjacent short segments. Each row's
    page_size=1 table is cut to the run's longest request, so the per-row copy
    stays [rows, < _V2_PREFILL_MIN_MAX_SEQ] instead of [rows, max_seq_len]."""
    if not run:
        return
    from sgl_kernel import fast_topk_transform_fused

    r0, r1 = run[0][0], run[-1][1]
    width = max(seg[3] for seg in run)
    page_table_1 = attn_metadata.page_table_1[:, :width]
    page_table = torch.cat(
        [page_table_1[seg[4] : seg[4] + 1].expand(seg[1] - seg[0], -1) for seg in run]
    )
    out[r0:r1] = fast_topk_transform_fused(
        score=logits[r0:r1],
        lengths=lengths_i32[r0:r1],
        page_table_size_1=page_table,
        cu_seqlens_q=torch.arange(
            r1 - r0 + 1, dtype=torch.int32, device=logits.device
        ),
        topk=topk,
        row_starts=row_starts[r0:r1],
    )


def _build_flashinfer_paged_args(
    attn_metadata,
    row_starts: Optional[torch.Tensor],
    cu_seqlens_q_topk: Optional[torch.Tensor],
    batch_idx_list: Optional[List[int]],
    device: torch.device,
    num_rows: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    row_to_batch = (
        torch.as_tensor(batch_idx_list, dtype=torch.int32, device=device)
        if batch_idx_list is not None
        else None
    )

    if (
        row_to_batch is not None
        and cu_seqlens_q_topk is not None
        and row_to_batch.shape[0] != num_rows
    ):
        q_lens = (cu_seqlens_q_topk[1:] - cu_seqlens_q_topk[:-1]).to(
            dtype=torch.int32, device=device
        )
        row_to_batch = torch.repeat_interleave(row_to_batch, q_lens)

    if row_to_batch is None and cu_seqlens_q_topk is not None:
        # Decode-like case (one query row per batch) does not need an explicit mapping.
        # Avoid dynamic tensor construction in this branch to keep CUDA graph capture safe.
        num_batches = cu_seqlens_q_topk.shape[0] - 1
        if not (row_starts is None and num_rows == num_batches):
            q_lens = (cu_seqlens_q_topk[1:] - cu_seqlens_q_topk[:-1]).to(
                dtype=torch.int32, device=device
            )
            row_to_batch = torch.repeat_interleave(
                torch.arange(q_lens.shape[0], dtype=torch.int32, device=device),
                q_lens,
            )

    if row_starts is not None and row_to_batch is None:
        raise RuntimeError(
            "PAGED topk_transform with row_starts requires cu_seqlens_q metadata."
        )

    local_row_starts = row_starts
    if local_row_starts is not None and row_to_batch is not None:
        local_row_starts = (
            local_row_starts - attn_metadata.cu_seqlens_k[:-1][row_to_batch]
        )

    return row_to_batch, local_row_starts


def _flashinfer_tie_break_value() -> int:
    mode = envs.SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK.get()
    if mode is None:
        return 0
    mode = mode.lower()
    if mode not in _FLASHINFER_TIE_BREAK_VALUES:
        raise RuntimeError(
            "SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK must be one of "
            f"{tuple(_FLASHINFER_TIE_BREAK_VALUES)} or unset, got {mode!r}."
        )
    return _FLASHINFER_TIE_BREAK_VALUES[mode]
