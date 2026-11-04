"""Factory embedding-hold lifecycle: MR register/deregister + retention + TTL.

Binds the real MMEncoder hold methods to a minimal stand-in (no GPU/model) with
a fake mooncake engine, so the ref-counting, retain-on-release (so a sibling
sharing the content-addressed task_id can still pull), and TTL eviction of
un-pulled holds are all checked directly.
"""

import asyncio
import os
from collections import OrderedDict

from sglang.srt.disaggregation.encoder.server import MMEncoder


class _FakeEngine:
    """Records mooncake MR register/deregister calls in place of real RDMA."""

    def __init__(self):
        """Start with empty register/deregister sets."""
        self.registered = set()
        self.deregistered = set()

    def register(self, ptr, length):
        """Record a registered MR pointer."""
        self.registered.add(ptr)

    def deregister(self, ptr):
        """Record a deregistered MR pointer."""
        self.deregistered.add(ptr)


class _Holder:
    """Minimal object carrying just the state the hold methods touch."""

    def __init__(self):
        """Bind the real MMEncoder hold methods onto this stand-in state."""
        self.task_embeddings = OrderedDict()
        self.task_embeddings_lock = asyncio.Lock()
        self.task_embeddings_bytes = 0
        # 0 disables the LRU capacity cap: these tests exercise the
        # retain-on-release and TTL-backstop paths, not size eviction.
        self.task_embeddings_max_bytes = 0
        self.use_mooncake = True
        self.engine = _FakeEngine()
        for name in (
            "store_task_embedding",
            "acquire_task_embedding",
            "release_task_embedding",
            "_drop_task_embedding_locked",
            "_evict_stale_task_embeddings_locked",
            "_evict_to_fit_locked",
            "_mark_hold_reserved_locked",
            "_check_hold_pool_budget",
        ):
            setattr(self, name, getattr(MMEncoder, name).__get__(self, _Holder))


def _emb():
    """A tiny bf16 embedding tensor to hold."""
    import torch

    return torch.zeros((4, 8), dtype=torch.bfloat16)


def test_release_retains_the_hold_and_mr_for_sibling_pulls():
    """A pull's release keeps the hold and its MR, not tears them down.

    ``task_id`` is content-addressed (md5(model_id + media_url)), so a sibling
    request sharing the same image pulls the same slot. Dropping it on the first
    release is what let a sibling's pull hit a 404; the hold is retained (refs
    back to 0) and only reclaimed by TTL / LRU later.
    """

    async def run():
        """Drive one store -> acquire -> release cycle."""
        h = _Holder()
        await h.store_task_embedding("t1", _emb(), image_grid_thw=[[1, 2, 2]])
        entry = h.task_embeddings["t1"]
        assert entry.mr_ptr is not None
        ptr = entry.mr_ptr
        assert ptr in h.engine.registered

        assert await h.acquire_task_embedding("t1") is entry
        await h.release_task_embedding("t1")  # refs back to 0, but retained
        assert "t1" in h.task_embeddings
        assert entry.refs == 0
        assert ptr not in h.engine.deregistered

    asyncio.run(run())


def test_ttl_evicts_unpulled_holds_and_frees_their_mr():
    """A hold nobody pulls is evicted after the TTL, freeing its MR."""

    async def run():
        """Store a hold, let it age past the TTL, then store another to sweep."""
        os.environ["SGLANG_ENCODER_TASK_EMB_TTL_S"] = "0.01"
        try:
            h = _Holder()
            await h.store_task_embedding("stale", _emb())
            stale_ptr = h.task_embeddings["stale"].mr_ptr
            await asyncio.sleep(0.05)
            # storing another task runs the eviction sweep
            await h.store_task_embedding("fresh", _emb())
            assert "stale" not in h.task_embeddings
            assert "fresh" in h.task_embeddings
            assert stale_ptr in h.engine.deregistered
        finally:
            del os.environ["SGLANG_ENCODER_TASK_EMB_TTL_S"]

    asyncio.run(run())

