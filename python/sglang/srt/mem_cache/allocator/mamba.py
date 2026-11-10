"""
Copyright 2026 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Slot allocator for the Mamba state pool.

Mamba caches one whole state tensor per request, so the allocator hands out
fixed-size slots (1 per request) rather than paged token KV indices.  The
underlying tensor storage lives in ``MambaPool``; this class owns only the
free-slot bookkeeping.
"""

from __future__ import annotations

import time
from typing import Iterator, Optional, Sequence

import torch


class MambaSlotAllocator:
    """Manages the free-list of request-level Mamba slots."""

    _HOST_FREE_WAIT_TIMEOUT_S = 1.0

    def __init__(self, size: int, device: str):
        self.size = size
        self.device = device
        self._alloc_iter: Optional[Iterator] = None
        self._alloc_host_iter: Optional[Iterator[int]] = None
        self._last_alloc_host_ids: Optional[tuple[int, ...]] = None
        self._free_slots_cpu: list[int] = []
        # Pending slots are published to both free lists together.
        self._pending_host_frees: list[tuple[object, torch.Tensor, torch.Tensor]] = []
        self._retired_host_frees: list[tuple[object, torch.Tensor, torch.Tensor]] = []
        self.clear()

    @staticmethod
    def _event_completed(event: object) -> bool:
        query = getattr(event, "query", None)
        if query is None:
            return False
        try:
            return bool(query())
        except Exception:
            return False

    def _reap_pending_host_frees(self, timeout: float = 0.0) -> None:
        """Publish completed CUDA frees, polling for at most ``timeout``."""
        deadline = time.monotonic() + max(0.0, timeout)
        pending = []
        for event, host_ids, source in self._pending_host_frees:
            while not self._event_completed(event):
                if time.monotonic() >= deadline:
                    pending.append((event, host_ids, source))
                    break
                time.sleep(0.001)
            else:
                ids = [int(slot) for slot in host_ids.tolist()]
                if len(ids) != source.numel():
                    raise RuntimeError("Mamba free metadata length mismatch")
                self.free_slots = torch.cat((self.free_slots, source))
                self._free_slots_cpu.extend(ids)
        self._pending_host_frees = pending
        self._retired_host_frees = [
            entry
            for entry in self._retired_host_frees
            if not self._event_completed(entry[0])
        ]

    def _prepare_alloc(self, need_size: int) -> None:
        self._reap_pending_host_frees()
        pending_size = sum(entry[2].numel() for entry in self._pending_host_frees)
        if len(self.free_slots) < need_size <= len(self.free_slots) + pending_size:
            self._reap_pending_host_frees(self._HOST_FREE_WAIT_TIMEOUT_S)

    def available_size(self) -> int:
        """Return the logical free count, including pending CUDA frees."""
        self._reap_pending_host_frees()
        return len(self.free_slots) + sum(
            entry[2].numel() for entry in self._pending_host_frees
        )

    def schedulable_available_size(self) -> int:
        """Return the number of slots immediately allocatable."""
        self._reap_pending_host_frees()
        return len(self.free_slots)

    def alloc_group_begin(self, num_reqs: int):
        """Pre-allocate a batch of slots for match_prefix to amortize overhead."""
        self._prepare_alloc(num_reqs)
        self._alloc_iter = None
        self._alloc_host_iter = None
        if num_reqs > 0:
            result = self._do_alloc(num_reqs)
            if result is not None:
                self._alloc_iter = iter(result.split(1))
                self._alloc_host_iter = iter(self._last_alloc_host_ids or ())
                self._last_alloc_host_ids = None

    def alloc_group_end(self):
        """Return any unused pre-allocated slots from the current group."""
        if self._alloc_iter is not None:
            remaining = list(self._alloc_iter)
            remaining_host = (
                list(self._alloc_host_iter)
                if self._alloc_host_iter is not None
                else None
            )
            if remaining:
                self.free(torch.cat(remaining), host_ids=remaining_host)
        self._alloc_iter = None
        self._alloc_host_iter = None

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        self._prepare_alloc(need_size)
        if self._alloc_iter is not None and need_size == 1:
            slot = next(self._alloc_iter, None)
            if slot is not None:
                if self._alloc_host_iter is not None:
                    next(self._alloc_host_iter, None)
                return slot
        value = self._do_alloc(need_size)
        self._last_alloc_host_ids = None
        return value

    def alloc_with_host(
        self, need_size: int
    ) -> tuple[Optional[torch.Tensor], Optional[tuple[int, ...]]]:
        """Allocate slots together with their host-side IDs."""
        self._prepare_alloc(need_size)
        if self._alloc_iter is not None and need_size == 1:
            slot = next(self._alloc_iter, None)
            if slot is not None:
                host_iter = self._alloc_host_iter
                host_id = None if host_iter is None else next(host_iter, None)
                return slot, None if host_id is None else (int(host_id),)
        value = self._do_alloc(need_size)
        if value is None:
            return None, None
        host_value = self._last_alloc_host_ids
        self._last_alloc_host_ids = None
        return value, host_value

    def _do_alloc(self, need_size: int) -> Optional[torch.Tensor]:
        if len(self._free_slots_cpu) != len(self.free_slots):
            raise RuntimeError("Mamba device and host free lists are out of sync")
        if need_size > len(self.free_slots):
            self._last_alloc_host_ids = None
            return None
        self._last_alloc_host_ids = tuple(self._free_slots_cpu[:need_size])
        del self._free_slots_cpu[:need_size]
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index

    def free(
        self,
        free_index: torch.Tensor,
        *,
        host_ids: Optional[Sequence[int]] = None,
    ):
        """Return slots, staging CUDA IDs asynchronously when needed."""
        if free_index is None or free_index.numel() == 0:
            return
        if host_ids is not None:
            if len(host_ids) != free_index.numel():
                raise ValueError(
                    f"Mamba free metadata length {len(host_ids)} does not match "
                    f"slot count {free_index.numel()}"
                )
            self.free_slots = torch.cat((self.free_slots, free_index))
            self._free_slots_cpu.extend(int(slot) for slot in host_ids)
            return
        if free_index.device.type == "cuda":
            device_ids = free_index.detach().reshape(-1).clone()
            host_tensor = torch.empty(
                device_ids.numel(), dtype=torch.int64, device="cpu", pin_memory=True
            )
            current_stream = torch.cuda.current_stream(free_index.device)
            host_tensor.copy_(device_ids, non_blocking=True)
            event = torch.cuda.Event()
            event.record(current_stream)
            device_ids.record_stream(current_stream)
            self._pending_host_frees.append((event, host_tensor, device_ids))
            return
        ids = [
            int(slot)
            for slot in free_index.detach().to(device="cpu").reshape(-1).tolist()
        ]
        self.free_slots = torch.cat((self.free_slots, free_index))
        self._free_slots_cpu.extend(ids)

    def free_with_host(self, free_index: torch.Tensor, host_ids: Sequence[int]) -> None:
        """Return slots using known host-side IDs."""
        self.free(free_index, host_ids=host_ids)

    def clear(self):
        # Slot 0 is reserved as a dummy write target for padded tokens.
        self._retired_host_frees.extend(self._pending_host_frees)
        self._pending_host_frees = []
        self._reap_pending_host_frees()
        self._free_slots_cpu = list(range(1, self.size + 1))
        self._last_alloc_host_ids = None
        self._alloc_iter = None
        self._alloc_host_iter = None
        self.free_slots = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )
