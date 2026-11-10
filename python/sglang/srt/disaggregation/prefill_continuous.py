"""Rank-consistent controller for continuous prefill admission."""

from __future__ import annotations

import queue
from typing import Callable

import torch
import torch.distributed as dist


class ContinuousPrefillController:
    """Coordinate the forward completion poll across scheduler ranks.

    Each scheduler rank owns one forward worker and therefore must consume its
    own worker result.  The collective only decides whether every rank has
    reached the same completion point; it must not let one fast rank release
    the group early.
    """

    def __init__(
        self,
        *,
        is_leader: bool,
        attn_tp_cpu_group,
        attn_cp_cpu_group,
        poll_s: float,
    ):
        self.is_leader = is_leader
        groups = []
        seen = set()
        for group in (attn_tp_cpu_group, attn_cp_cpu_group):
            if group is None:
                continue
            try:
                world_size = dist.get_world_size(group=group)
            except Exception:
                world_size = 1
            if world_size > 1 and id(group) not in seen:
                groups.append(group)
                seen.add(id(group))
        self._groups = groups
        self.poll_s = max(float(poll_s), 1.0e-4)

    def poll_forward_done(self, try_get: Callable[[float], bool]) -> bool:
        """Poll every local worker and reduce completion with MIN.

        The previous implementation polled only the CP0/TP0 worker and used
        MAX.  That allowed one rank to enter ``_fwd_collect`` while another
        rank was still executing its forward, which deadlocked the scheduler
        group under uneven GPU progress.
        """

        # Results are rank-local.  Every rank must take its own item from the
        # bounded queue so that _fwd_collect() can consume it later.
        local_done = bool(try_get(self.poll_s))
        if not self._groups:
            return local_done

        status = torch.tensor([1 if local_done else 0], dtype=torch.int32)
        for group in self._groups:
            # Keep the same TP-then-CP order used by the existing
            # poll_and_all_reduce_attn_cp_tp_group helper.  MIN means all
            # participants have received their local forward result.
            dist.all_reduce(status, op=dist.ReduceOp.MIN, group=group)
        return bool(status.item())


def make_try_get(
    out_q: "queue.Queue",
    stash: Callable[[object], None],
    has_stash: Callable[[], bool] | None = None,
) -> Callable[[float], bool]:
    """Return a bounded wait that removes at most one worker result.

    A rank may finish its local forward before another rank.  Once its result
    is stashed, it must keep reporting ``done`` while the other ranks catch
    up; otherwise the MIN all-reduce can never observe all ranks as complete.
    """

    def try_get(timeout: float) -> bool:
        if has_stash is not None and has_stash():
            return True
        try:
            item = out_q.get(timeout=timeout)
        except queue.Empty:
            return False
        stash(item)
        return True

    return try_get
