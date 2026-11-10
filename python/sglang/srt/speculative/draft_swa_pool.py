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


class DraftWindowSlotMap:
    """Full (target-global) slot id -> draft-local SWA slot id, window-bounded.

    Sync discipline: the mapping tables live on the draft device, so any host
    branch on their contents is a D2H sync. Only counts required by the
    host-managed free-page stack are read back.
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

        # full page -> draft page. 0 = unmapped. A full page's within-page slots
        # must all resolve to the SAME draft page, so page alignment survives the
        # full -> swa translation.
        self.num_full_pages = (self.num_full_slots + 2 * self.page_size) // self.page_size
        self.full_page_to_draft_page = torch.zeros(
            self.num_full_pages, dtype=torch.int64, device=device
        )

        # Free draft pages as a stack. Page 0 is reserved; ids run from page 1.
        self.num_draft_pages = self.num_swa_slots // self.page_size
        self._free_pages = torch.arange(
            1, self.num_draft_pages, dtype=torch.int64, device=device
        )
        self._num_free_pages = int(self._free_pages.numel())
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
        """Number of free slots across all free pages."""
        return self._num_free_pages * self.page_size

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

    def _alloc_pages(self, num_pages: int) -> torch.Tensor:
        if num_pages > self._num_free_pages:
            raise RuntimeError(
                f"Draft SWA slot pool exhausted: need {num_pages} pages, "
                f"{self._num_free_pages} free of {self.num_draft_pages - 1}. "
                f"Raise the draft pool size or lower --speculative-draft-window-size."
            )
        take = self._free_pages[self._num_free_pages - num_pages : self._num_free_pages]
        self._num_free_pages -= num_pages
        return take

    @_synchronized
    def map_slots(
        self, full_indices: torch.Tensor, *, req_pool_idx=None, end_pos=None
    ) -> torch.Tensor:
        """Assign a swa slot to each full id and return the swa ids.

        Idempotent per id: an already-mapped full id keeps its swa slot, so a
        re-injected token does not leak. Raises when the pool is exhausted --
        silently dropping a mapping would make the draft read slot 0.

        The assigned swa id preserves the full id's within-page offset
        (``swa_id % page_size == full_id % page_size``), which the paged
        attention backend relies on.

        ``req_pool_idx`` / ``end_pos`` record how far into a request's
        ``req_to_token`` row this call mapped, so ``release_request`` knows how far
        to sweep. Pass them for every per-request write (prefill and verify
        block); omitting them leaves the high-water mark untouched.
        """
        if full_indices.numel() == 0:
            return full_indices.new_empty((0,), dtype=torch.int64)

        if req_pool_idx is not None and end_pos is not None:
            idx = int(req_pool_idx)
            self.mapped_high_water[idx] = max(
                int(self.mapped_high_water[idx]), int(end_pos)
            )

        full_indices = full_indices.to(dtype=torch.int64, device=self.full_to_swa.device)
        current = self.full_to_swa[full_indices]
        need_mask = current == 0

        # need_full never contains -1: full_to_swa[-1] == -1, so a -1 full id is
        # never in the `current == 0` set.
        need_full = full_indices[need_mask]

        if self.page_size == 1:
            num_need = int(need_mask.sum())  # host count needed only by this branch
            if num_need == 0:
                return current
            self.full_to_swa[need_full] = self._alloc_pages(num_need)
            return self.full_to_swa[full_indices]

        full_pages = need_full // self.page_size
        offsets = need_full % self.page_size

        # Reuse the draft page already assigned to a full page; allocate a new one
        # only for never-mapped pages (unconditional: the old .any() guard was a sync).
        assigned = self.full_page_to_draft_page[full_pages]
        new_full_pages = torch.unique(full_pages[assigned == 0])
        # The one unavoidable readback: _alloc_pages slices the host-side free
        # stack, so it needs the page count as a Python int.
        self.full_page_to_draft_page[new_full_pages] = self._alloc_pages(
            int(new_full_pages.numel())
        )

        draft_pages = self.full_page_to_draft_page[full_pages]
        self.full_to_swa[need_full] = draft_pages * self.page_size + offsets
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
    def free_slots(self, full_indices: torch.Tensor) -> int:
        """Unmap the given full ids and return their swa slots to the pool.

        Only mapped ids consume work; duplicates and unmapped ids are dropped, so
        calling this twice on the same range is a no-op the second time.

        Draft pages are recycled only once every full page's slots are freed, so
        a partially-freed page never leaves a gap in another request's page.
        """
        if full_indices.numel() == 0:
            return 0
        full_indices = full_indices.to(dtype=torch.int64, device=self.full_to_swa.device)
        full_indices = torch.unique(full_indices[full_indices >= 0])
        swa = self.full_to_swa[full_indices]
        mapped = swa > 0
        num_freed = int(mapped.sum())
        if num_freed == 0:
            return 0

        freed_full = full_indices[mapped]

        if self.page_size == 1:
            self._free_pages[self._num_free_pages : self._num_free_pages + num_freed] = (
                swa[mapped]
            )
            self._num_free_pages += num_freed
            self.full_to_swa[freed_full] = 0
            return num_freed

        self.full_to_swa[freed_full] = 0

        # Recycle a page only when its whole full page is unmapped (vectorized:
        # the old per-page Python loop paid one `.all()` sync per page).
        candidates = torch.unique(freed_full // self.page_size)
        # Clamp away from the trailing -1 sentinel (repeated index is all()-safe).
        rows = (candidates * self.page_size).unsqueeze(1) + self._within_page
        rows = rows.clamp_(max=self.full_to_swa.numel() - 2)
        released = candidates[(self.full_to_swa[rows] == 0).all(dim=1)]

        draft_pages = self.full_page_to_draft_page[released]
        valid = draft_pages > 0
        # Second and last readback: the host-side free stack needs the count.
        num_valid = int(valid.sum())
        if num_valid:
            self._free_pages[
                self._num_free_pages : self._num_free_pages + num_valid
            ] = draft_pages[valid]
            self._num_free_pages += num_valid
            self.full_page_to_draft_page[released[valid]] = 0
        return num_freed

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
    ) -> int:
        """Free every swa slot that fell out of the window; return the count.

        The frontier is monotonic per request, so this is safe to call on any
        cadence -- every step, or every Nth step like the target's SWA eviction.

        The stale ranges are gathered across requests in ONE device op and freed
        with a single ``free_slots`` call. The former per-request loop paid
        ``free_slots``'s full chain (``torch.unique`` plus several D2H syncs)
        once per request per step, which dominated the decode hot path at batch.
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
        # own [start, end) span, so positions beyond a shorter request's end are
        # never read.
        device = target_req_to_token.device
        starts_t = torch.tensor(starts, dtype=torch.int64, device=device)
        lens_t = torch.tensor(
            [e - s for s, e in zip(starts, ends)], dtype=torch.int64, device=device
        )
        reqs_t = torch.tensor(advancing_reqs, dtype=torch.int64, device=device)
        span = int(max(e - s for s, e in zip(starts, ends)))
        pos = torch.arange(span, dtype=torch.int64, device=device)
        idx = starts_t.unsqueeze(1) + pos.unsqueeze(0)
        keep = pos.unsqueeze(0) < lens_t.unsqueeze(1)
        stale = target_req_to_token[reqs_t.unsqueeze(1), idx][keep]

        # Free the union once, then advance the frontiers (host-only) only after
        # the free succeeds: a mid-free failure must not leave a request reporting
        # slots as freed that are still mapped.
        num_freed = self.free_slots(stale)
        for req_pool_idx, new_frontier in zip(advancing_reqs, ends):
            self.evict_frontier[req_pool_idx] = new_frontier
        return num_freed

    @_synchronized
    def release_request(
        self, *, req_pool_idx: int, target_req_to_token: torch.Tensor
    ) -> int:
        """Free everything still mapped for a finished request and reset it.

        Sweeps ``[0, high_water)`` -- the furthest position ever mapped -- not the
        committed length: the draft forward maps the whole verify block, so a
        rejected tail sits above the committed length and would otherwise leak.

        Draft KV is never shared across requests (it is a function of this
        request's target hidden states), so a finished request keeps nothing.
        """
        idx = int(req_pool_idx)
        high_water = int(self.mapped_high_water[idx])
        num_freed = 0
        if high_water > 0:
            num_freed = self.free_slots(target_req_to_token[idx, :high_water])
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
