"""Draft-side SWA slot map: bounds draft KV by the window, not by context length.

The draft writes its KV at slots the TARGET allocator handed out, so it cannot
own the global ("full") index space. It can own a much smaller SWA index space
plus a full -> swa mapping over it -- DeepSeek-V4's index-addressed SWA layout,
driven from the draft side:

  * a slot is mapped lazily, when the draft actually writes it
  * a slot whose position falls out of the window is unmapped and freed
  * live swa slots per request stay bounded by the window, so the draft KV pool
    is sized by ``max_running_requests * (window + page + gamma)`` rather than by
    the context length

Slots stay index-addressed and therefore content-stable: a swa id keeps its KV
until something explicitly frees it. That is what lets a prefix cache -- and the
attention store -- reason about draft KV at all; a per-request ring, whose slot
meaning drifts with position, could not.

Mirrors ``SWATokenToKVPoolAllocator``'s conventions: mapping entry 0 means
"unmapped", and the table carries a trailing -1 sentinel so a -1 full id
translates to -1.

Page alignment (the trtllm_mha contract)
-----------------------------------------
The trtllm_mha backend reads KV through a *page table*: it samples
``req_to_token`` only at page boundaries and derives each block id with
``slot // page_size``, then the attention kernel reads token ``pos`` at
``page_table[pos // page_size]`` slot ``pos % page_size``. That is only correct
when a token's translated slot id satisfies

    draft_id % page_size == full_id % page_size   (== pos % page_size)

i.e. the low ``log2(page_size)`` bits must survive translation. The target
allocator already hands out page-aligned full ids, so the draft side must hand
out page-aligned draft ids: the free list is therefore a *page* list, and a full
page is mapped to one draft page whose within-page offsets are reused verbatim.
Slot 0 stays the unmapped sentinel, so the whole page 0 is reserved and ids are
handed out from page 1 upward.

Sync discipline
---------------
Every table here (including the free-page stack and its top) lives on the draft
device, and no method reads any of it back on the hot path. Two idioms buy that:

  * never compact. ``x[mask]``, ``torch.unique`` and ``mask.sum()`` all have a
    data-dependent output, so the host must learn a count -- a D2H copy queued
    behind the whole stream, i.e. one that waits out the PREVIOUS forward.
    Instead the arrays keep their original length and everything that must not be
    touched is parked on a *sink* index (a pad entry whose value is already the
    value being written).
  * dedup by racing, not by sorting. ``_first_occurrence`` picks one
    representative per distinct page using scatter semantics.
  * no Python scalars in the hot path. As the value of an indexed assignment a
    scalar becomes a CPU 0-dim tensor that ATen copies to the device, and a
    pageable H2D copy appends a cudaStreamSynchronize -- measured at 1.6ms,
    because it waits out the queued forward. As a ``torch.where`` operand it
    costs an extra fill_ kernel. The ``_c_*`` constants exist for both.

The only readback left is ``_assert_capacity``, which the decode path opts out of
(``assert_capacity=False``) and pays for by deferring the exhaustion raise to the
next prefill.
"""

from __future__ import annotations

import functools
import math
import os
import threading

import torch


def _synchronized(method):
    """Serialize a DraftWindowSlotMap method under the instance ``_lock``.

    The map is read by the PD Decode preallocation path (``send_metadata``'s
    ``map_slots`` for the transfer window) and mutated by the forward thread
    (``_maintain_draft_swa_window``, ``release_draft_kv``). The free-page stack
    and ``full_to_swa`` updates are multi-step read-modify-writes, so an
    unguarded concurrent call corrupts the pool (double-handed page, lost page,
    torn translate). A reentrant lock keeps ``advance_window``/``release_request``
    -> ``free_slots`` re-entry safe.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


def upload_host_ints(values, *, device) -> torch.Tensor:
    """Upload a small host int list to ``device`` without stalling the host.

    ``torch.tensor(values, device=cuda)`` takes the pageable-H2D path, which
    appends a cudaStreamSynchronize: the host then blocks until everything already
    queued drains, so a few-byte index upload ends up serializing the step behind
    the previous forward (same idiom note as DSpark's prefill uploads). Staging
    through pinned memory keeps the copy asynchronous; torch's caching host
    allocator records the copy event, so the staging block is not recycled before
    the copy lands.
    """
    device = torch.device(device)
    host = torch.tensor(values, dtype=torch.int64, pin_memory=device.type == "cuda")
    return host.to(device=device, non_blocking=True)


# 单个页检查网格的元素上限。int64 网格最多约 32 MiB；函数总峰值还包括
# gather 结果与布尔掩码，不能把这个值当作整段代码的显存上限。
_FREE_SLOTS_GRID_ELEMENTS = 1 << 22


def validate_token_ranges(target_req_to_token, req_pool_indices, starts, ends):
    """Validate host-side row/range metadata before indexing or mutating state."""
    if not (len(req_pool_indices) == len(starts) == len(ends)):
        raise ValueError("Draft SWA request indices and range lengths differ")
    num_rows, width = target_req_to_token.shape
    for req, start, end in zip(req_pool_indices, starts, ends):
        if not (0 <= int(req) < num_rows and 0 <= int(start) <= int(end) <= width):
            raise ValueError(
                f"Draft SWA range out of bounds: req_pool_idx={req}, "
                f"range=[{start}, {end}), table_shape=({num_rows}, {width})"
            )


def gather_token_ranges(target_req_to_token, req_pool_indices, starts, ends):
    """Gather validated ragged ranges, padding unused entries with -1."""
    validate_token_ranges(target_req_to_token, req_pool_indices, starts, ends)
    device = target_req_to_token.device
    lengths = [int(end) - int(start) for start, end in zip(starts, ends)]
    span = max(lengths, default=0)
    if span == 0:
        return target_req_to_token.new_empty((0,))
    reqs_t = upload_host_ints(req_pool_indices, device=device)
    starts_t = upload_host_ints(starts, device=device)
    lens_t = upload_host_ints(lengths, device=device)
    pos = torch.arange(span, dtype=torch.int64, device=device)
    keep = pos.unsqueeze(0) < lens_t.unsqueeze(1)
    # 补齐的列仍会参与 gather，事后 where 保护不了这次读取。
    # 只把无效补齐列改成 0；有效区间已在 CPU 上校验，不用 clamp 掩盖错误。
    indices = torch.where(keep, starts_t.unsqueeze(1) + pos.unsqueeze(0), 0)
    gathered = target_req_to_token[reqs_t.unsqueeze(1), indices]
    return torch.where(keep, gathered, -1).reshape(-1)


class DraftWindowSlotMap:
    """Full (target-global) slot id -> draft-local SWA slot id, window-bounded.

    Sync discipline: the mapping tables AND the free-page stack (top included)
    live on the draft device, and neither ``map_slots`` nor ``free_slots`` reads
    any of it back -- see the module docstring. Only ``_assert_capacity`` and the
    count properties sync, and none of them is on the decode path.
    """

    def __init__(
        self,
        *,
        num_full_slots: int,
        num_swa_slots: int,
        max_num_reqs: int,
        window_size: int,
        page_size: int = 1,
        device: str | torch.device = "cpu",
    ) -> None:
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}.")
        # Reentrant: guards the free-page stack / full_to_swa against concurrent
        # forward-thread and mooncake transfer-thread mutation (see _synchronized).
        self._lock = threading.RLock()
        self.num_full_slots = int(num_full_slots)
        self.num_swa_slots = int(num_swa_slots)
        self.window_size = int(window_size)
        self.page_size = max(1, int(page_size))
        self.device = device

        # The swa id space must cover whole pages: page 0 is reserved (slot 0 is
        # the unmapped sentinel), so usable ids start at page_size.
        if self.num_swa_slots <= self.page_size:
            raise ValueError(
                f"num_swa_slots must exceed page_size (page 0 is the sentinel), "
                f"got num_swa_slots={self.num_swa_slots}, page_size={self.page_size}."
            )
        if (self.num_swa_slots % self.page_size) != 0:
            raise ValueError(
                f"num_swa_slots must be a multiple of page_size so the swa id "
                f"space is a whole number of pages, got num_swa_slots="
                f"{self.num_swa_slots}, page_size={self.page_size}."
            )

        # full -> swa. 0 = unmapped; trailing entry maps a -1 full id to -1.
        self.full_to_swa = torch.zeros(
            self.num_full_slots + self.page_size + 1, dtype=torch.int64, device=device
        )
        self.full_to_swa[-1] = -1
        # Masked-write sinks. The last entry is the -1 sentinel; the one before it
        # is a pad entry that is never a real full id and must stay 0, so writing
        # 0 there is the no-op that replaces compaction.
        self._sentinel_idx = self.full_to_swa.numel() - 1
        self._zero_sink = self.full_to_swa.numel() - 2

        # full page -> draft page. 0 = unmapped. A full page's within-page slots
        # must all resolve to the SAME draft page, so page alignment survives the
        # full -> swa translation. The trailing entry is this table's sink: it
        # holds -1, so parked elements never look like they need a draft page and
        # writing -1 back leaves it intact.
        self.num_full_pages = (self.num_full_slots + 2 * self.page_size) // self.page_size
        self._poison_page = self.num_full_pages
        self.full_page_to_draft_page = torch.zeros(
            self.num_full_pages + 1, dtype=torch.int64, device=device
        )
        self.full_page_to_draft_page[self._poison_page] = -1
        # Scratch for the sync-free dedup (see _first_occurrence). Lock-protected
        # like the tables, and never read at an index it was not just written at,
        # so it needs no clearing between calls.
        self._page_claim = torch.zeros(
            self.num_full_pages + 1, dtype=torch.int64, device=device
        )

        # Free draft pages as a stack. Page 0 is reserved; ids run from page 1.
        # The top lives ON THE DEVICE so pop/push positions are computed with
        # device ops -- that is what keeps map_slots/free_slots readback-free. One
        # trailing slot is the push sink; a stack of at most num_draft_pages - 1
        # entries never reaches it.
        self.num_draft_pages = self.num_swa_slots // self.page_size
        self._free_pages = torch.zeros(
            self.num_draft_pages + 1, dtype=torch.int64, device=device
        )
        self._free_pages[: self.num_draft_pages - 1] = torch.arange(
            1, self.num_draft_pages, dtype=torch.int64, device=device
        )
        self._stack_sink = self.num_draft_pages
        self._free_top = torch.full(
            (), self.num_draft_pages - 1, dtype=torch.int64, device=device
        )
        # Sticky: set when a pop wanted more pages than the stack held. Read only
        # by _assert_capacity, i.e. never on the decode path.
        self._exhausted = torch.zeros((), dtype=torch.bool, device=device)

        # Device-resident constants. Two reasons never to spell these as Python
        # scalars in the hot path:
        #   * as the VALUE of an indexed assignment, ATen builds a CPU 0-dim
        #     tensor and copies it over -- a pageable H2D, which appends a
        #     cudaStreamSynchronize and waits out the previous forward (measured:
        #     1.6ms). Same trap ``upload_host_ints`` documents.
        #   * as a ``torch.where`` operand, it materializes a 0-dim device tensor
        #     with an extra fill_ kernel. No sync, but a launch per call.
        def _const(value: int) -> torch.Tensor:
            return torch.full((), value, dtype=torch.int64, device=device)

        self._c_zero = _const(0)
        self._c_minus_one = _const(-1)
        self._c_zero_sink = _const(self._zero_sink)
        self._c_sentinel = _const(self._sentinel_idx)
        self._c_poison_page = _const(self._poison_page)
        self._c_stack_sink = _const(self._stack_sink)
        # Within-page offsets, kept resident: free_slots builds a
        # [num_candidate_pages, page_size] index grid from it on every eviction.
        self._within_page = torch.arange(
            self.page_size, dtype=torch.int64, device=device
        )

        # Per-request eviction frontier: committed positions below this have had
        # their swa slots freed. Host-side, mirroring ``req.kv.swa_evicted_seqlen``.
        self.evict_frontier = torch.zeros(int(max_num_reqs), dtype=torch.int64)
        # Per-request high-water mark of positions this map has ever mapped.
        # Release must sweep up to HERE, not to the committed length: the draft
        # forward maps the whole verify block, and a rejected tail sits above the
        # committed length. Sweeping a fixed committed_len+gamma instead would be
        # unsafe -- req_to_token entries past a request's own extent are stale
        # leftovers that may now name slots owned by a DIFFERENT live request.
        self.mapped_high_water = torch.zeros(int(max_num_reqs), dtype=torch.int64)

    # ---- capacity -------------------------------------------------------

    @property
    def num_free_slots(self) -> int:
        """Number of free slots across all free pages.

        Reads the device-resident stack top, so this syncs -- diagnostics and
        tests only, never the decode path.
        """
        return int(self._free_top) * self.page_size

    @property
    def num_live_slots(self) -> int:
        """Number of live (allocated) slots excluding the reserved page."""
        # Usable ids start at page_size (page 0 is reserved).
        return (self.num_swa_slots - self.page_size) - self.num_free_slots

    @staticmethod
    def recommended_num_swa_slots(
        *, max_num_reqs: int, window_size: int, gamma: int, page_size: int = 1
    ) -> int:
        """Slots needed so every concurrent request can hold a full window.

        Per request the live set is the window frontier (``max(window, page)``,
        the eviction threshold), plus the over-allocated reservation the draft may
        gather from (``2 * gamma`` -- see ``prepare_for_decode``), plus a page of
        alignment slack. Each request gets a whole number of pages (so a request
        whose first full id is not page-aligned still has enough pages), plus one
        extra page for the reserved page 0 (whose slot 0 is the unmapped
        sentinel). The result is always a multiple of ``page_size``.
        """
        page_size = max(1, int(page_size))
        per_req = (
            max(int(window_size), page_size)
            + 2 * int(gamma)
            + page_size
        )
        pages_per_req = (per_req + page_size - 1) // page_size
        num_pages = max_num_reqs * pages_per_req + 1
        return num_pages * page_size

    # ---- mapping --------------------------------------------------------

    @_synchronized
    def translate(self, full_indices: torch.Tensor) -> torch.Tensor:
        """Full ids -> swa ids. Unmapped ids come back as 0."""
        return self.full_to_swa[full_indices]

    def _first_occurrence(self, keys: torch.Tensor, sel: torch.Tensor) -> torch.Tensor:
        """One True per distinct key among ``sel``, without a readback.

        Deduping by compaction (``torch.unique``) forces a sync: its output size
        is data-dependent. Instead every selected element races to claim its key
        in a scratch table -- scatter semantics guarantee exactly ONE writer per
        index survives -- and an element is the representative iff the surviving
        writer is itself. Elements outside ``sel`` are parked on the poison key
        and dropped, so they can never steal a real page's claim.
        """
        ids = torch.arange(keys.numel(), dtype=torch.int64, device=keys.device)
        keys = torch.where(sel, keys, self._c_poison_page)
        self._page_claim[keys] = ids
        return sel & (self._page_claim[keys] == ids)

    def _assert_capacity(self, num_new: torch.Tensor) -> None:
        """Turn pool exhaustion into a raise. Costs one readback; not hot-path.

        Called before any table is touched, so a raise leaves the map intact.
        ``_exhausted`` carries over an overflow from an earlier
        ``assert_capacity=False`` call, which had no way to raise at the time.
        """
        num_new, num_free, exhausted = (
            torch.stack([num_new, self._free_top, self._exhausted.long()])
            .cpu()
            .tolist()
        )
        if not exhausted and num_new <= num_free:
            return
        latched = (
            " (an earlier unchecked allocation already overflowed)" if exhausted else ""
        )
        raise RuntimeError(
            f"Draft SWA slot pool exhausted: need {num_new} pages, "
            f"{num_free} free of {self.num_draft_pages - 1}{latched}. "
            f"Raise the draft pool size or lower --speculative-draft-window-size."
        )

    @_synchronized
    def map_slots(
        self,
        full_indices: torch.Tensor,
        *,
        req_pool_idx=None,
        end_pos=None,
        assert_capacity: bool = True,
    ) -> torch.Tensor:
        """Assign a swa slot to each full id and return the swa ids.

        Idempotent per id: an already-mapped full id keeps its swa slot, so a
        re-injected token does not leak.

        The assigned swa id preserves the full id's within-page offset
        (``swa_id % page_size == full_id % page_size``), which the paged
        attention backend relies on.

        ``req_pool_idx`` / ``end_pos`` record how far into a request's
        ``req_to_token`` row this call mapped, so ``release_request`` knows how far
        to sweep. Pass them for every per-request write (prefill and verify
        block); omitting them leaves the high-water mark untouched.

        ``assert_capacity=False`` skips the exhaustion check, which is the one
        readback in this path; the overflow is then latched on the device and
        raised by the next checked call. Pass it on the decode path, where a sync
        would wait out the previous forward.
        """
        if full_indices.numel() == 0:
            return full_indices.new_empty((0,), dtype=torch.int64)
        if req_pool_idx is not None and end_pos is not None:
            idx = int(req_pool_idx)
            self.mapped_high_water[idx] = max(
                int(self.mapped_high_water[idx]), int(end_pos)
            )

        full_indices = full_indices.to(
            dtype=torch.int64, device=self.full_to_swa.device
        )
        # A -1 full id resolves through the trailing sentinel to -1, so it is
        # never in the `== 0` set and never consumes a slot.
        need = self.full_to_swa[full_indices] == 0
        pages = torch.where(need, full_indices // self.page_size, self._c_poison_page)
        # A full page already backed by a draft page keeps it (that is what
        # makes within-page offsets survive translation); only never-mapped
        # pages allocate, one per distinct page in this batch.
        new_pages = self._first_occurrence(
            pages, need & (self.full_page_to_draft_page[pages] == 0)
        )
        rank = torch.cumsum(new_pages, 0) - 1
        num_new = new_pages.sum()
        if assert_capacity:
            self._assert_capacity(num_new)
        else:
            self._exhausted.logical_or_(num_new > self._free_top)

        # Pop num_new pages off the stack. The slice runs upward from the new
        # top so ascending full pages get ascending draft pages, keeping
        # contiguous full ids contiguous in the draft space.
        take = self._free_pages[
            (self._free_top - num_new + rank).clamp_(0, self._stack_sink)
        ]
        self.full_page_to_draft_page[
            torch.where(new_pages, pages, self._c_poison_page)
        ] = torch.where(new_pages, take, self._c_minus_one)
        self._free_top.sub_(num_new).clamp_(min=0)

        # Re-read: this picks up both the pages just allocated and the ones
        # this batch reused, so duplicates inside the batch agree for free.
        draft_pages = self.full_page_to_draft_page[pages]
        swa = draft_pages * self.page_size + full_indices % self.page_size
        self.full_to_swa[torch.where(need, full_indices, self._c_zero_sink)] = (
            torch.where(need, swa, self._c_zero)
        )
        return self.full_to_swa[full_indices]

    @_synchronized
    def note_mapped_extents(self, req_pool_indices, end_positions) -> None:
        """Raise several requests' mapped high-water marks in one acquisition.

        The decode hot path maps every request's window tail in ONE
        ``map_slots`` call and records all extents here afterwards; raising
        the high-water per request inside the mapping loop would re-introduce
        a per-request lock round-trip over the batch.
        """
        hw = self.mapped_high_water
        for req_pool_idx, end_pos in zip(req_pool_indices, end_positions):
            idx = int(req_pool_idx)
            end_pos = int(end_pos)
            if end_pos > int(hw[idx]):
                hw[idx] = end_pos

    @_synchronized
    def free_slots(
        self, full_indices: torch.Tensor, *, return_count: bool = True
    ) -> int:
        """Unmap the given full ids and return their swa slots to the pool.

        Only mapped ids consume work; duplicates and unmapped ids are dropped, so
        calling this twice on the same range is a no-op the second time.

        Draft pages are recycled only once every full page's slots are freed, so
        a partially-freed page never leaves a gap in another request's page.

        ``return_count=False`` skips the freed-slot count, the one readback in
        this path -- pass it on the decode path, where a sync would wait out the
        previous forward. The returned count is then 0, not the real count.
        """
        if full_indices.numel() == 0:
            return 0
        full_indices = full_indices.reshape(-1).to(
            dtype=torch.int64, device=self.full_to_swa.device
        )
        chunk_size = max(1, _FREE_SLOTS_GRID_ELEMENTS // self.page_size)
        if full_indices.numel() <= chunk_size:
            # 常态路径：一次处理，保留原来的返回语义。
            return self._free_slots_chunk(full_indices, return_count=return_count)

        # 【修复】分片处理，避免一次性 materialize 整个索引网格。
        # 下面是 (n, page_size) 的 int64 网格，n 由调用方决定：请求的第一个
        # decode step 会把整段 prefill 前缀交给 sweep（根因见 advance_window），
        # n 到 10 万级时 n × page_size 64 ≈ 1.6 GiB 瞬时显存，在 KV cache 已吃满
        # 的卡上直接 CUDA OOM。按「网格元素数」分片，把峰值压到
        # _FREE_SLOTS_GRID_ELEMENTS × 8 字节；span 小的常态路径只跑一个分片。
        # 语义完全不变：每个分片仍按原顺序做 unmap / dedup / 推栈，而一个 full
        # page 只有在「最后一个仍 mapped 的槽位被解映射的那个分片」里才会满足
        # released（要求整页已 unmapped），所以不会重复入栈、也不会漏释放；
        # 映射表、页表、空闲栈 top 都与单次调用一致。
        # 唯一差异：跨分片的页会在「补完它的那个分片」里入栈，因此空闲栈里页的
        # 排列顺序可能与单次调用不同（只在 n 超过分片阈值时发生）。栈里的空闲页
        # 彼此等价，只影响下次先取谁，不影响正确性。
        count = 0
        if return_count:
            # 原接口按释放前的每次出现计数。先整体算一遍再开始 unmap，
            # 防止跨分片重复 id 少算；return_count=False 不增加设备读回。
            total = torch.zeros((), dtype=torch.int64, device=self.full_to_swa.device)
            for part in full_indices.split(chunk_size):
                part = part.to(dtype=torch.int64, device=self.full_to_swa.device)
                total.add_((self.full_to_swa[part] > 0).sum())
            count = int(total)
        for part in full_indices.split(chunk_size):
            self._free_slots_chunk(part, return_count=False)
        return count

    def _free_slots_chunk(self, full_indices, *, return_count: bool) -> int:
        """One slice of ``free_slots``: unmap, then recycle completed pages.

        Split out only so the caller can bound the transient ``(n, page_size)``
        index grid. The caller holds the instance lock and decides the count
        semantics: ``free_slots`` pre-counts over the whole input for its chunked
        path, while this single-slice path counts its own input directly.
        """
        swa = self.full_to_swa[full_indices]
        # Unmapped ids read 0 and the -1 sentinel reads -1, so both drop out.
        mapped = swa > 0

        # Unmap. Writing 0 is idempotent, so duplicate ids need no dedup, and
        # everything not being freed is parked on the always-zero sink.
        self.full_to_swa[torch.where(mapped, full_indices, self._c_zero_sink)] = (
            self._c_zero
        )

        # Recycle a page only when its whole full page is unmapped. One
        # candidate element per distinct page, so a page is pushed once.
        pages = torch.where(mapped, full_indices // self.page_size, self._c_poison_page)
        candidates = self._first_occurrence(pages, mapped)
        # Non-candidates read the -1 sentinel row, which is never all-zero.
        rows = torch.where(
            candidates, pages * self.page_size, self._c_sentinel
        ).unsqueeze(1)
        rows = (rows + self._within_page).clamp_(max=self._sentinel_idx)
        released = candidates & (self.full_to_swa[rows] == 0).all(dim=1)

        draft_pages = self.full_page_to_draft_page[
            torch.where(released, pages, self._c_poison_page)
        ]
        valid = released & (draft_pages > 0)
        # Push at device-computed stack positions; the rest go to the sink.
        rank = torch.cumsum(valid, 0) - 1
        self._free_pages[
            torch.where(valid, self._free_top + rank, self._c_stack_sink).clamp_(
                0, self._stack_sink
            )
        ] = draft_pages
        self._free_top.add_(valid.sum()).clamp_(max=self.num_draft_pages - 1)
        self.full_page_to_draft_page[torch.where(valid, pages, self._c_poison_page)] = (
            torch.where(valid, self._c_zero, self._c_minus_one)
        )

        # Counts each occurrence of a duplicated id; the pool state does not.
        return int(mapped.sum()) if return_count else 0

    # ---- window eviction ------------------------------------------------

    def evict_threshold(self, committed_len: int) -> int:
        """Committed positions strictly below this may be freed.

        ``max(window, page)`` mirrors ``free_swa_out_of_window_slots``: keeping at
        least one page means the frontier stays page-aligned below the window.
        """
        threshold = int(committed_len) - max(self.window_size, self.page_size)
        if self.page_size > 1:
            threshold = (threshold // self.page_size) * self.page_size
        return max(0, threshold)

    @_synchronized
    def advance_window(
        self,
        *,
        req_pool_indices_cpu,
        committed_lens_cpu,
        target_req_to_token: torch.Tensor,
        return_count: bool = True,
    ) -> int:
        """Free every swa slot that fell out of the window; return the count.

        The frontier is monotonic per request, so this is safe to call on any
        cadence -- every step, or every Nth step like the target's SWA eviction.

        The stale ranges are gathered across requests in ONE device op and freed
        with a single ``free_slots`` call, which slices the page grid internally.
        The former per-request loop paid
        ``free_slots``'s full chain (``torch.unique`` plus several D2H syncs)
        once per request per step, which dominated the decode hot path at batch.

        ``return_count=False`` forwards to ``free_slots``: no readback, and the
        returned count is 0.
        """
        # 使用真实表宽（含 speculative 预留空间），不是模型标称上下文。
        # 参数都在 CPU 上，校验不引入 GPU 同步。
        validate_token_ranges(
            target_req_to_token,
            req_pool_indices_cpu,
            [0] * len(req_pool_indices_cpu),
            committed_lens_cpu,
        )
        # Host-side frontier advance; collect the ranges that actually moved.
        #
        # 【根因】请求的第一个 decode step 上 old_frontier 必然是 0（release_request
        # 会把它重置为 0），而 committed_len 此时已经是整段 prefill 前缀，于是
        # new_frontier = committed_len - max(window, page)（10 万前缀 → 97920），
        # 下面就把 [0, 97920) 整段交给 sweep。正常稳态这里只有 gamma 量级。
        #
        # 这一段绝大多数位置 draft 从没映射过：forward 侧的映射是从
        # max(frontier, end - window_tail) 起步的，PD transfer 侧（decode.py 的
        # send_metadata 预分配）也只映射 [origin_input_len - window, origin_input_len)，
        # 两者都贴着 new_frontier 附近。所以这个 sweep 真正要释放的只有顶部
        # 一页左右，其余全是空转。
        #
        # 空转本身不算错（未映射的位置会被 free_slots 判为 no-op），但代价是：
        # 下面把整批的 [start, end) 拼成 (R, span) 的 gather，span 被撑到整个
        # 前缀长度，free_slots 再据此造一个 (R×span, page_size) 的 int64 索引网格
        # —— 35 个 10 万前缀的请求 × page_size 64 ≈ 1.6 GiB 瞬时显存，在 KV cache
        # 已吃满的卡上直接 CUDA OOM。
        #
        # 这里刻意不「跳过」这段 sweep：mapped_high_water 只记录映射终点、不记录
        # 起点（map_slots 带 req_pool_idx/end_pos 时会抬升它，PD transfer 与 prefill
        # 分块路径都传了），所以推不出本请求最低被映射过的位置；而 transfer 预分配
        # 会映射到 frontier 下方一页左右，跳过 sweep 就会漏掉那一页、泄漏 draft page。
        # 因此改为在 free_slots 里对页网格分片，把峰值显存压住。
        starts: list[int] = []
        ends: list[int] = []
        advancing_reqs: list[int] = []
        for req_pool_idx, committed_len in zip(req_pool_indices_cpu, committed_lens_cpu):
            req_pool_idx = int(req_pool_idx)
            new_frontier = self.evict_threshold(int(committed_len))
            old_frontier = int(self.evict_frontier[req_pool_idx])
            if new_frontier <= old_frontier:
                continue
            starts.append(old_frontier)
            ends.append(new_frontier)
            advancing_reqs.append(req_pool_idx)
        if not advancing_reqs:
            return 0

        # 一次批量 gather 覆盖全批 [start, end)；补齐的无效列在 gather 前已改成
        # 安全下标、之后置 -1，free_slots 会丢弃它们。free_slots 内部再对
        # (n, page_size) 页网格分片。
        stale = gather_token_ranges(target_req_to_token, advancing_reqs, starts, ends)

        # Free 成功后才推进 frontier：中途失败不能让请求报告已淘汰但仍映射的槽位。
        num_freed = self.free_slots(stale, return_count=return_count)
        for req_pool_idx, new_frontier in zip(advancing_reqs, ends):
            self.evict_frontier[req_pool_idx] = new_frontier
        return num_freed

    @_synchronized
    def release_request(
        self,
        *,
        req_pool_idx: int,
        target_req_to_token: torch.Tensor,
        return_count: bool = True,
    ) -> int:
        """Free everything still mapped for a finished request and reset it.

        Sweeps ``[evict_frontier, high_water)`` -- from the eviction frontier up to
        the furthest position ever mapped, and not the committed length: the draft
        forward maps the whole verify block, so a rejected tail sits above the
        committed length and would otherwise leak. Starting at the frontier rather
        than at 0 is equivalent (everything below it is already freed or was never
        mapped) and avoids handing the whole prefill prefix to ``free_slots``.

        Draft KV is never shared across requests (it is a function of this
        request's target hidden states), so a finished request keeps nothing.

        ``return_count=False`` forwards to ``free_slots``: no readback, and the
        returned count is 0.
        """
        idx = int(req_pool_idx)
        # 【优化】起点从 0 改成 evict_frontier。
        # 所有映射路径（forward 的 _maintain_draft_swa_window、PD transfer 的
        # send_metadata 预分配、prefill 的 _map_prefill_draft_slots）都只映射
        # frontier 及其之上的位置，而 advance_window 每次推进 frontier 时已经把
        # 下面那一段 sweep 干净了。所以剩下的活槽位只可能在
        # [frontier, high_water) 里，从 0 开始扫纯属白干：长上下文下会白造一个
        # (high_water, page_size) 的 int64 索引网格（128k × page_size 64 ≈ 64 MiB
        # 瞬时显存/次释放）。high_water 仍是「映射过的最远位置」，语义不变，
        # 所以 rejected tail 依旧不会漏。
        low_water = int(self.evict_frontier[idx])
        high_water = int(self.mapped_high_water[idx])
        # frontier 以下已淘汰；旧 req_to_token 项可能仍指向已复用的 full id。
        # 只释放剩余范围，并用 high_water 覆盖 rejected tail。
        validate_token_ranges(
            target_req_to_token, [idx], [0], [max(low_water, high_water)]
        )
        num_freed = 0
        if high_water > low_water:
            num_freed = self.free_slots(
                target_req_to_token[idx, low_water:high_water],
                return_count=return_count,
            )
        self.evict_frontier[idx] = 0
        self.mapped_high_water[idx] = 0
        return num_freed


def compact_draft_pool_multiplier() -> float:
    """Headroom multiplier for the draft SWA pool (and its slot map).

    High-concurrency PD Decode overlaps transfer/retry lifetimes, so the physical
    KV buffer and the DraftWindowSlotMap page count must be scaled together --
    scaling only the buffer leaves the slot map as the binding constraint and it
    exhausts. Read here once so the Decode-side worker and the target-side pool
    budget can never disagree.
    """
    try:
        return max(
            1.0, float(os.environ.get("SGLANG_DSPARK_SWA_POOL_MULTIPLIER", "1"))
        )
    except ValueError:
        return 1.0


def compute_compact_draft_pool_slots(
    *,
    max_num_reqs: int,
    window_size: int,
    gamma: int,
    page_size: int = 1,
    multiplier: float | None = None,
) -> int:
    """Window-bounded draft KV pool size in slots, page-aligned.

    Single source of truth for how many slots the compact (SWA) draft pool needs,
    shared by the Decode-side worker (`_shrink_memory_pool_config_to_window`) and
    the target-side pool budget (`pool_configurator`) so the two sides can never
    drift. The recommended size is already a whole number of pages (including the
    reserved page 0); ``multiplier`` adds headroom for high-concurrency PD Decode
    where transfer/retry lifetimes overlap, then the result is re-aligned to
    ``page_size`` so trtllm_mha's ``k_cache.view(-1, page_size, ...)`` reshape
    stays valid. When ``multiplier`` is omitted it reads the shared env var.
    """
    page_size = max(1, int(page_size))
    if multiplier is None:
        multiplier = compact_draft_pool_multiplier()
    multiplier = max(1.0, float(multiplier))
    recommended = DraftWindowSlotMap.recommended_num_swa_slots(
        max_num_reqs=max_num_reqs,
        window_size=window_size,
        gamma=gamma,
        page_size=page_size,
    )
    return int(math.ceil(recommended * multiplier / page_size) * page_size)
