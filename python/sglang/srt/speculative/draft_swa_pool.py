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
        full_indices = full_indices.to(
            dtype=torch.int64, device=self.full_to_swa.device
        )
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
        with a single ``free_slots`` call. The former per-request loop paid
        ``free_slots``'s full chain (``torch.unique`` plus several D2H syncs)
        once per request per step, which dominated the decode hot path at batch.

        ``return_count=False`` forwards to ``free_slots``: no readback, and the
        returned count is 0.
        """
        # Host-side frontier advance; collect the ranges that actually moved.
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

        # One batched gather of every request's stale range (mirrors the map-side
        # batching in _maintain_draft_swa_window). ``keep`` trims each row to its
        # own [start, end) span; the trim pads with -1 rather than compacting with
        # a boolean-mask select, which would be a host-device sync right at the
        # top of the step. ``free_slots`` drops -1 for free.
        device = target_req_to_token.device
        starts_t = upload_host_ints(starts, device=device)
        lens_t = upload_host_ints([e - s for s, e in zip(starts, ends)], device=device)
        reqs_t = upload_host_ints(advancing_reqs, device=device)
        span = int(max(e - s for s, e in zip(starts, ends)))
        pos = torch.arange(span, dtype=torch.int64, device=device)
        idx = starts_t.unsqueeze(1) + pos.unsqueeze(0)
        keep = pos.unsqueeze(0) < lens_t.unsqueeze(1)
        gathered = target_req_to_token[reqs_t.unsqueeze(1), idx]
        # Scalar -1 (not ``_c_minus_one``): this row is built on the TARGET
        # device, which the map's own constants need not share.
        stale = torch.where(keep, gathered, -1).reshape(-1)

        # Free the union once, then advance the frontiers (host-only) only after
        # the free succeeds: a mid-free failure must not leave a request reporting
        # slots as freed that are still mapped.
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

        Sweeps ``[0, high_water)`` -- the furthest position ever mapped -- not the
        committed length: the draft forward maps the whole verify block, so a
        rejected tail sits above the committed length and would otherwise leak.

        Draft KV is never shared across requests (it is a function of this
        request's target hidden states), so a finished request keeps nothing.

        ``return_count=False`` forwards to ``free_slots``: no readback, and the
        returned count is 0.
        """
        idx = int(req_pool_idx)
        high_water = int(self.mapped_high_water[idx])
        num_freed = 0
        if high_water > 0:
            num_freed = self.free_slots(
                target_req_to_token[idx, :high_water], return_count=return_count
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
