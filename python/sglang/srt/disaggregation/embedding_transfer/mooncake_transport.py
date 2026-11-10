"""
Mooncake RDMA embedding transport for EPD disaggregation.

Holds the pinned host buffer pool singleton (``_MooncakeBufferManager``) and the
async, non-blocking transport that submits transfers to a shared thread pool.
"""

import logging
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from http import HTTPStatus
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from sglang.srt.disaggregation.embedding_transfer.base import (
    EmbeddingTransport,
    PullTarget,
    TransportResult,
    parse_torch_dtype,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mooncake buffer manager (ported from bench_encode.py TransferContext)
# ---------------------------------------------------------------------------


class _MooncakeBufferManager:
    """CPU pinned-memory buffer pool for Mooncake RDMA embedding transfers.

    Pre-allocates a contiguous CPU pinned-memory buffer and registers it with
    the Mooncake engine once.  Individual allocations are page-based pointer
    bumps — no per-request register/deregister overhead.

    Using CPU pinned memory instead of GPU memory because:
      - Frees GPU memory for KV cache (the scarcest resource in prefill)
      - Pinned memory enables fast DMA when prefill copies the embedding H2D

    The embedding stays resident in the pool as a zero-copy view (no clone)
    until prefill consumes it, so the pool size is a hard upper bound on
    embedding host memory and doubles as backpressure: when full or too
    fragmented, acquire_buffer() returns None and the caller retries later
    (after other buffers are freed).

    Configuration via environment variables:
        SGLANG_MC_POOL_GB      – pool size in GB (default: 2)
        SGLANG_MC_PAGE_SIZE_MB – page size in MB (default: 2)
    """

    _instance: Optional["_MooncakeBufferManager"] = None
    _init_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "_MooncakeBufferManager":
        """Get the singleton instance of MooncakeBufferManager.

        Creates the instance on first call using environment variables:
            - SGLANG_MC_POOL_GB: Pool size in GB (default: 2)
            - SGLANG_MC_PAGE_SIZE_MB: Page size in MB (default: 2)

        Returns:
            The singleton MooncakeBufferManager instance.
        """
        if cls._instance is not None:
            return cls._instance
        with cls._init_lock:
            if cls._instance is not None:
                return cls._instance
            pool_gb = float(os.environ.get("SGLANG_MC_POOL_GB", "2"))
            page_mb = int(os.environ.get("SGLANG_MC_PAGE_SIZE_MB", "2"))
            pool_bytes = int(pool_gb * (1024**3))
            page_size = page_mb * 1024 * 1024
            cls._instance = cls(pool_bytes, page_size)
            return cls._instance

    @classmethod
    def get_instance_if_exists(cls) -> Optional["_MooncakeBufferManager"]:
        """Return the singleton only if already created, else None (so a rank
        that never pulled does not lazily allocate the multi-GB pool)."""
        return cls._instance

    def __init__(self, pool_bytes: int, page_size: int):
        from sglang.srt.disaggregation.gpu_buffer_pool import GpuBufferPool

        self._pool = GpuBufferPool(pool_bytes, page_size, device="cpu", pin_memory=True)

        # Buffers whose transfer was still running at cleanup; freed only after
        # future.done() (process_inflight_queue), like disagg_prefill_inflight_queue.
        self._inflight_queue: List[Tuple[Future, int]] = []

        engine = self._get_engine()
        engine.register(self._pool.data_ptr, self._pool.total_bytes)

        logger.info(
            "MooncakeBufferManager: registered %.1f MB CPU pinned buffer "
            "(%d pages x %.1f MB)",
            self._pool.total_bytes / 1e6,
            self._pool.num_pages,
            page_size / 1e6,
        )

    @staticmethod
    def _get_engine():
        from sglang.srt.distributed.parallel_state import get_mooncake_embedding_engine

        return get_mooncake_embedding_engine()

    def acquire_buffer(self, byte_size: int) -> Optional[int]:
        """Try to allocate from pool; return None if pool is full."""
        return self._pool.try_allocate(byte_size)

    def release_buffer(self, ptr: int) -> None:
        """Return buffer to pool."""
        self._pool.free(ptr)

    def add_inflight_queue(self, future: "Future", buf_addr: int) -> None:
        """Queue a still-running transfer's buffer for release once it finishes."""
        self._inflight_queue.append((future, buf_addr))

    def process_inflight_queue(self) -> None:
        """Free buffers whose transfer has finished; keep the rest for a later cycle."""
        if not self._inflight_queue:
            return
        undone: List[Tuple[Future, int]] = []
        for future, buf_addr in self._inflight_queue:
            if future.done():
                try:
                    self._pool.free(buf_addr)
                except Exception as e:
                    logger.warning(
                        "process_inflight_queue: failed to free %s: %s", buf_addr, e
                    )
            else:
                undone.append((future, buf_addr))
        self._inflight_queue = undone

    def view_buffer_as_tensor(
        self, ptr: int, byte_size: int, dtype: torch.dtype
    ) -> torch.Tensor:
        """Zero-copy typed view of a buffer region."""
        offset = ptr - self._pool.data_ptr
        raw = self._pool._buf[offset : offset + byte_size]
        return raw.view(dtype)


# ---------------------------------------------------------------------------
# Mooncake transport
# ---------------------------------------------------------------------------


class MooncakeEmbeddingTransport(EmbeddingTransport):
    """Pull embedding data via Mooncake RDMA (async, non-blocking).

    Uses _MooncakeBufferManager for pinned-buffer pool allocation — the entire
    pool is pre-registered with the Mooncake engine once, so per-request
    buffer acquisition is just a pointer bump (no register/deregister).

    send() acquires a pinned buffer from the pool and submits the HTTP transfer
    request to a shared thread pool.  poll() checks futures without blocking.
    _resolve_future() returns the buffer as a zero-copy typed-tensor view and
    hands buf_addr ownership to the caller; the buffer is freed back to the
    pool only on error (otherwise after prefill consumes the embedding).
    """

    # Shared across all instances — avoids per-request thread-pool overhead.
    # Threads are created lazily and mostly block on I/O, so a large pool is safe.
    _pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix="mc-transfer")

    def __init__(self):
        # task_id -> (Future, PullTarget, buf_addr, byte_size, dtype_str)
        self._pending: Dict[str, Tuple[Future, PullTarget, int, int, str]] = {}

    def send(self, pull_target: PullTarget) -> bool:
        """Acquire a pinned buffer and submit the RDMA transfer to the pool.

        Returns False without side effects when the pinned buffer pool is
        exhausted, so the caller can retry on a later poll.
        """
        meta = pull_target.embedding_meta
        shape = meta["shape"]
        dtype_str = meta["dtype"]
        byte_size = (
            meta.get("byte_size") or int(np.prod(shape)) * np.dtype(dtype_str).itemsize
        )

        # Try to acquire buffer from pre-registered pool
        buf_mgr = _MooncakeBufferManager.get_instance()
        buf_addr = buf_mgr.acquire_buffer(byte_size)
        if buf_addr is None:
            logger.warning(
                "[trace] mc-transfer acquire_buffer=None task_id=%s byte_size=%d "
                "free=%d total=%d frag=%.2f live_alloc=%d",
                pull_target.task_id,
                byte_size,
                buf_mgr._pool.available_bytes(),
                buf_mgr._pool.total_bytes,
                buf_mgr._pool.fragmentation_ratio(),
                buf_mgr._pool.num_allocations(),
            )
            return False

        self._submit_transfer(pull_target, buf_addr, byte_size, dtype_str)
        return True

    def _submit_transfer(
        self,
        pull_target: PullTarget,
        buf_addr: int,
        byte_size: int,
        dtype_str: str,
    ) -> None:
        """Submit the HTTP transfer request to the thread pool."""
        buf_mgr = _MooncakeBufferManager.get_instance()
        try:
            engine = self._get_engine()
            payload = {
                "req_id": pull_target.request_id,
                "task_id": pull_target.task_id,
                # Encoder-shard index; /epd/transfer only echoes it back.
                "part_idx": 0,
                "session_id": engine.session_id,
                "buffer_address": buf_addr,
                "byte_size": byte_size,
                "dtype": dtype_str,
                "shape": pull_target.embedding_meta["shape"],
                "image_url": pull_target.image_url,
                "modality": pull_target.modality,
            }
            if pull_target.modality == "video":
                payload["img_grid_thw"] = pull_target.embedding_meta.get("img_grid_thw")
                payload["chunk_timestamps"] = pull_target.chunk_timestamps
            url = f"{pull_target.endpoint}/epd/transfer"

            # Submit to thread pool — non-blocking
            future = self._pool.submit(self._execute_transfer, url, payload)
            self._pending[pull_target.task_id] = (
                future,
                pull_target,
                buf_addr,
                byte_size,
                dtype_str,
            )
        except Exception:
            buf_mgr.release_buffer(buf_addr)
            raise

    def poll(self) -> List[TransportResult]:
        """Return results for transfers whose future has completed.

        Non-blocking: unfinished futures are left in ``self._pending``.
        """
        if not self._pending:
            return []

        results: List[TransportResult] = []
        done_keys: List[str] = []

        for task_id, (
            future,
            pull_target,
            buf_addr,
            byte_size,
            dtype_str,
        ) in self._pending.items():
            if not future.done():
                continue

            done_keys.append(task_id)
            result = self._resolve_future(
                task_id, future, pull_target, buf_addr, byte_size, dtype_str
            )
            results.append(result)

        for key in done_keys:
            del self._pending[key]

        return results

    def cleanup(self) -> None:
        """Cancel outstanding transfers and give their pinned buffers back.

        Buffers whose transfer is already in flight cannot be released yet, so
        they are handed to the buffer manager's inflight queue and freed later
        by ``process_inflight_queue``.
        """
        buf_mgr = _MooncakeBufferManager.get_instance()
        done = 0
        undone = 0
        for task_id, (future, _, buf_addr, byte_size, _) in self._pending.items():
            if future.cancel():
                try:
                    buf_mgr.release_buffer(buf_addr)
                    done += 1
                except Exception as e:
                    logger.warning("Failed to release buffer for %s: %s", task_id, e)
            else:
                # transfer may still be writing; free it later in process_inflight_queue
                buf_mgr.add_inflight_queue(future, buf_addr)
                undone += 1

        if self._pending:
            logger.info(
                "[trace] mc-transfer cleanup done=%d undone=%d",
                done,
                undone,
            )
        self._pending.clear()

    # -- internal --

    @staticmethod
    def _get_engine():
        from sglang.srt.distributed.parallel_state import get_mooncake_embedding_engine

        return get_mooncake_embedding_engine()

    @staticmethod
    def _execute_transfer(url: str, payload: dict) -> dict:
        """Synchronous HTTP POST — runs in thread pool worker."""
        import requests

        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _resolve_future(
        task_id: str,
        future: Future,
        pull_target: PullTarget,
        buf_addr: int,
        byte_size: int,
        dtype_str: str,
    ) -> TransportResult:
        """Convert a completed Future into a TransportResult.

        Returns the embedding as a zero-copy view into the pinned pool buffer
        and hands buf_addr ownership to the caller (freed after prefill
        consumes it). The buffer is released here only on error paths.
        """
        buf_mgr = _MooncakeBufferManager.get_instance()

        try:
            ack = future.result()
        except Exception as e:
            buf_mgr.release_buffer(buf_addr)
            return TransportResult(
                task_id=task_id,
                error_msg=f"Mooncake transfer request failed for {task_id}: {e}",
                error_code=HTTPStatus.BAD_GATEWAY,
            )

        if ack.get("status_code") != 200:
            buf_mgr.release_buffer(buf_addr)
            return TransportResult(
                task_id=task_id,
                error_msg=f"Mooncake transfer rejected: {ack.get('message')}",
                error_code=HTTPStatus.BAD_GATEWAY,
            )

        meta = pull_target.embedding_meta
        shape = meta["shape"]

        try:
            torch_dtype = parse_torch_dtype(dtype_str)
            recv_tensor = buf_mgr.view_buffer_as_tensor(
                buf_addr, byte_size, torch_dtype
            )
            embedding = recv_tensor[: int(np.prod(shape))].reshape(shape)
            grid_thw = torch.tensor(meta["img_grid_thw"])
        except Exception as e:
            buf_mgr.release_buffer(buf_addr)
            return TransportResult(
                task_id=task_id,
                error_msg=f"Failed to read embedding from buffer for {task_id}: {e}",
                error_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

        return TransportResult(
            task_id=task_id,
            embedding=embedding,
            grid_thw=grid_thw,
            buf_addr=buf_addr,
        )
