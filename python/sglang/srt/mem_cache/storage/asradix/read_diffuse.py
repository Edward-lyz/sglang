"""Sharded AttentionStore read + NVLink diffuse orchestration.

Under ``--enable-prefill-cp --cp-strategy interleave`` (attn_tp=1, attn_cp=N) the
SPAG KV a rank stores/reads from AttentionStore is *replicated* across the N
attention-CP ranks: every rank issues the same block hashes and pulls the same
bytes, so the aggregate AS->HBM H2D is N x redundant.

This orchestrates: each CP rank reads only its 1/N shard of the blocks via the
SDK, then the ranks *diffuse* their shards to each other over NVLink with a
raw-IPC SM scatter kernel (``kernel_diffuse``), so every rank ends up with the
full set at ~1/N of the redundant H2D.

Correctness preconditions (must hold; validate on device with the AS content
digest ``SGLANG_LOG_AS_CACHE_DIGEST``):
  * the N ranks are lockstep on the same requests/allocator, so the loaded page
    slots (``page_ids``) are identical on every rank -- the scatter kernel copies
    page k at the SAME byte offset ``slot*item_bytes`` in src and dst windows;
  * the KV content is genuinely replicated across the N ranks (true for MLA
    latent + indexer + SWA under CP interleave with attn_tp=1).

Transport (kernel + raw IPC) lives in ``kernel_diffuse``; this module owns handle
exchange, peer-window open, shard planning, and the pull-diffuse.
"""

from __future__ import annotations

import logging
import hashlib
from array import array
from typing import List, Optional, Sequence

import torch

from sglang.srt.environ import envs

from . import kernel_diffuse as kd

logger = logging.getLogger(__name__)


def shard_owner(block_index: int, cp_size: int, num_blocks: int) -> int:
    """Which CP rank owns (reads) a block, in balanced *contiguous* chunks.

    Contiguous (rank r owns a single run of blocks) rather than interleaved: the
    blocks are uniform so load balances either way, the pool pages are scattered
    either way (diffuse always uses the discrete scatter kernel), and a
    contiguous split makes rank 0's sublist a true prefix -- the friendliest case
    if the Agent ever falls back to longest-prefix matching.
    """
    return min(block_index * cp_size // num_blocks, cp_size - 1)


def local_block_indices(num_blocks: int, cp_rank: int, cp_size: int) -> List[int]:
    """Block indices this rank reads from AS (its contiguous 1/N chunk)."""
    return [
        k for k in range(num_blocks) if shard_owner(k, cp_size, num_blocks) == cp_rank
    ]


class AsReadDiffuser:
    """Exchange KV-pool IPC windows once, then pull peer shards over NVLink."""

    def __init__(self, pool, cp_group, device) -> None:
        self._pool = pool
        self._group = cp_group
        # Normalize to a concrete cuda device (the pool may report a bare
        # "cuda" with no index); pin every op to this process's device.
        dev = device if isinstance(device, torch.device) else torch.device(device)
        if dev.index is None:
            dev = torch.device(dev.type, torch.cuda.current_device())
        self._device = dev
        self.cp_size = torch.distributed.get_world_size(group=cp_group)
        self.cp_rank = torch.distributed.get_rank(group=cp_group)
        # Per-buffer (per-layer, per-segment) local base pointer + page stride.
        ptrs, _data_lens, item_lens = pool.get_contiguous_buf_infos()
        tensor_infos = pool.get_contiguous_buf_tensor_infos()
        if len(ptrs) != len(tensor_infos):
            raise RuntimeError(
                "AS read-diffuse buffer/tensor info length mismatch: "
                f"{len(ptrs)} vs {len(tensor_infos)}"
            )
        self._local_base: List[int] = [int(p) for p in ptrs]
        self._page_bytes: List[int] = [int(b) for b in item_lens]
        # Extra byte offset applied on top of each buffer's data_ptr (the page
        # region can start past a leading SWA ring). Same on every rank.
        self._offset_bytes: List[int] = [
            int(off_b) for _k, _buf, off_b, _ib in tensor_infos
        ]
        self._num_buffers = len(ptrs)
        self._batched = envs.SGLANG_AS_READ_DIFFUSE_BATCHED.get()
        if self._batched and any(
            base % 16 or stride <= 0 or stride % 16
            for base, stride in zip(self._local_base, self._page_bytes)
        ):
            raise RuntimeError("AS batched diffuse requires 16-byte aligned pool pages")
        # Share each local buffer's IPC window (framed handle + byte offset to the
        # buffer's data start; the page region then sits at + offset_bytes).
        local_share = [
            kd.share_torch_window(buf) for _kind, buf, _off, _ib in tensor_infos
        ]
        dev_index = self._device_index()
        gathered: List[Optional[list]] = [None] * self.cp_size
        torch.distributed.all_gather_object(gathered, local_share, group=cp_group)
        self._peer_base: List[Optional[List[int]]] = [None] * self.cp_size
        for peer in range(self.cp_size):
            if peer == self.cp_rank:
                self._peer_base[peer] = list(self._local_base)
                continue
            shares = gathered[peer]
            if shares is None or len(shares) != self._num_buffers:
                raise RuntimeError(
                    f"AS read-diffuse missing peer {peer} window handles"
                )
            self._peer_base[peer] = [
                kd.ipc_open_handle(handle, dev_index)
                + int(share_off)
                + self._offset_bytes[i]
                for i, (handle, share_off) in enumerate(shares)
            ]
        # Stable pointer/stride tables are allocated once, outside request IO.
        self._buffer_tables = {}
        if self._batched:
            for peer in range(self.cp_size):
                if peer == self.cp_rank:
                    continue
                bases = self._peer_base[peer]
                if any(base % 16 for base in bases):
                    raise RuntimeError("AS batched diffuse requires aligned IPC windows")
                self._buffer_tables[peer] = torch.tensor(
                    list(zip(self._local_base, bases, self._page_bytes)),
                    dtype=torch.int64,
                    device=self._device,
                )
        logger.info(
            "AS read-diffuse: opened %d peer windows x %d buffers "
            "(cp_rank=%d/%d, batched=%s)",
            self.cp_size - 1,
            self._num_buffers,
            self.cp_rank,
            self.cp_size,
            self._batched,
        )
        logger.info(
            "AS read-diffuse geometry: cp_rank=%d page_bytes=%s offsets=%s",
            self.cp_rank, self._page_bytes, self._offset_bytes,
        )
# PLACEHOLDER_APPEND

    def _device_index(self) -> int:
        dev = self._device
        if not isinstance(dev, torch.device):
            dev = torch.device(dev)
        if dev.index is not None:
            return dev.index
        # e.g. torch.device("cuda") with no index -> this process's current device.
        return torch.cuda.current_device()

    def local_shard(self, num_blocks: int) -> List[int]:
        """Block indices this rank reads from AS."""
        return local_block_indices(num_blocks, self.cp_rank, self.cp_size)

    def diffuse(
        self,
        page_slots: Sequence[int],
        num_blocks: int,
        extra_local_blocks: Sequence[int] = (),
        peers_ready: bool = False,
    ) -> None:
        """Pull every peer's owned page slots into the local pool over NVLink.

        ``page_slots[k]`` is the pool page slot of block ``k`` (identical on all
        ranks). This rank already holds its own shard plus ``extra_local_blocks``
        (blocks it read redundantly, e.g. the terminal SWA block); those are not
        pulled. Pull model: this rank only writes its own buffers, so no rank
        writes another's memory.

        ``peers_ready``: the caller already ran a CP-group collective after the
        (host-synchronous) shard reads, so every peer's read has landed in HBM --
        skip the redundant pre-barrier. A post-barrier still guards against a
        peer reusing a buffer this rank is still reading.
        """
        self.diffuse_many(
            [(page_slots, num_blocks, extra_local_blocks)],
            peers_ready=peers_ready,
        )

    def diffuse_many(self, batches, *, peers_ready: bool = False) -> None:
        """Pull several completed reads before a single completion barrier.

        Shard ownership restarts at each request boundary. Pending allocations
        stay owned until this call finishes, before any request is published or
        its slots can be reused.
        """
        prepared = [
            ([int(slot) for slot in slots], int(count), set(map(int, extra)))
            for slots, count, extra in batches
        ]
        for slots, count, extra in prepared:
            if len(slots) != count or count < 0:
                raise ValueError("AS diffuse page slots must cover the requested blocks")
        if not peers_ready:
            # Ensure peers' shard H2D reads landed before we pull them.
            torch.distributed.barrier(group=self._group)
        from .transport import cache_digest_enabled

        if cache_digest_enabled():
            # Debug-only proof of the identical physical-slot precondition.
            # Keep it off in performance runs, just like the content digests.
            identities_local = [
                hashlib.sha256(array("q", slots).tobytes()).hexdigest()
                for slots, _, _ in prepared
            ]
            identity = tuple(identities_local)
            identities = [None] * self.cp_size
            torch.distributed.all_gather_object(identities, identity, group=self._group)
            if len(set(identities)) != 1:
                raise RuntimeError("AS diffuse CP ranks disagree on physical page slots")
            for (_, count, _), slot_hash in zip(prepared, identities_local):
                logger.info(
                    "AS read-diffuse slot-map: cp_rank=%d pages=%d sha256=%s",
                    self.cp_rank, count, slot_hash,
                )
        stream = torch.cuda.current_stream(self._device)
        keepalive: List[torch.Tensor] = []
        with torch.cuda.stream(stream):
            for peer in range(self.cp_size):
                if peer == self.cp_rank:
                    continue
                owned = [
                    slots[k]
                    for slots, count, extra in prepared
                    for k in range(count)
                    if shard_owner(k, self.cp_size, count) == peer
                    and k not in extra
                ]
                if not owned:
                    continue
                idx = torch.tensor(owned, dtype=torch.int32, device=self._device)
                keepalive.append(idx)  # keep alive until kernels complete
                idx_ptr = int(idx.data_ptr())
                npages = idx.numel()
                if self._batched:
                    kd.scatter_buffers(self._buffer_tables[peer], idx)
                    continue
                peer_base = self._peer_base[peer]
                for i in range(self._num_buffers):
                    kd.scatter_pages(
                        self._local_base[i],
                        peer_base[i],
                        idx_ptr,
                        self._page_bytes[i],
                        npages,
                    )
        stream.synchronize()
        # Everyone finished pulling before any buffer is reused.
        torch.distributed.barrier(group=self._group)

