"""Multimodal embedding cache interfaces and a reference-counted LRU cache."""

import abc
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator

logger = logging.getLogger(__name__)


class MultimodalCache(abc.ABC):
    """Define the lookup and lifetime interface for multimodal embeddings."""

    @abc.abstractmethod
    def __init__(
        self,
    ):
        """Initialize the cache implementation."""
        ...

    @staticmethod
    def combine_hashes(mm_hashes: List[int]) -> Optional[int]:
        """
        Get a combined hash from individual mm item hashes
        """
        if not mm_hashes:
            return None
        return hash(tuple(mm_hashes))

    @abc.abstractmethod
    def get(
        self, mm_hashes: List[int], combined_hash: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        """
        Extract the embedding with the hash-ids of the queried items. Try combined hash first, if missed, fallback to individual hashes
        The returned tensor may not be contiguous
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def set(
        self,
        mm_hash: int,
        embedding: torch.Tensor,
        mm_embedding_allocator: BaseTokenToKVPoolAllocator,
    ) -> bool:
        """
        Set the embedding to the pre-allocated locations with a hash id
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def has(self, mm_hash: int) -> bool:
        """Return whether an embedding is available for the given hash."""
        raise NotImplementedError()

    @abc.abstractmethod
    def free(
        self, mm_hash: int, mm_embedding_allocator: BaseTokenToKVPoolAllocator
    ) -> bool:
        """Free the cached embedding and its allocated locations."""
        raise NotImplementedError()

    @abc.abstractmethod
    def clear(self):
        """Remove cached embeddings according to the implementation's policy."""
        raise NotImplementedError()

    @abc.abstractmethod
    def available_size(self):
        """Return the cache size reported by the implementation."""
        raise NotImplementedError()


def _get_tensor_size(embedding: torch.Tensor):
    """Return the size of the tensor's elements in bytes."""
    return embedding.element_size() * embedding.numel()


@dataclass(kw_only=True)
class EmbeddingResult:
    """Store an embedding, its media metadata, and cache lifetime state."""

    embedding: torch.Tensor
    image_grid_thw: Optional[any] = None  # Image grid dimensions for RoPE
    image_url: Optional[str] = None  # Source image URL
    modality: str = "image"
    chunk_timestamps: Optional[List[str]] = (
        None  # per-chunk start timestamps (video only)
    )
    # Original [width, height] per image, before any resize (Kimi-K3 needs it
    # to rebuild its resolution-carrying media framing on a cache hit).
    original_image_sizes: Optional[List[List[int]]] = None
    ref_count: int = 0  # Reference count for in-use protection
    _pending_delete: bool = False  # Marked for deletion when ref_count reaches 0
    _invalidated: bool = False  # Hidden from new readers; existing holders may finish

    @property
    def nbytes(self) -> int:
        """Byte size of the embedding, compatible with both Tensor and ScatteredTensor."""
        emb = self.embedding
        if isinstance(emb, torch.Tensor):
            return emb.numel() * emb.element_size()
        # ScatteredTensor
        if hasattr(emb, "nbytes"):
            return emb.nbytes
        return 0


class MultiModalStaticCache(MultimodalCache):
    """
    A server-level cache for multimodal embedding.
    Embeddings are computed prior, and this cache does not really pre-alloc.

    This cache implements reference counting to prevent embeddings from being
    evicted while they are being transferred (via Mooncake or ZMQ).
    """

    def __init__(
        self,
        max_size: int,
    ):
        """Initialize an LRU cache with a capacity of max_size bytes."""
        super().__init__()
        self.max_size = max_size
        self.mm_cache: OrderedDict[int, EmbeddingResult] = OrderedDict()
        self.current_size = 0
        self._pending_delete: Dict[int, EmbeddingResult] = {}  # Delayed deletion queue
        self._lock = threading.Lock()

    def get(self, mm_hashes, combined_hash: Optional[int] = None):
        """Look up a cached embedding.

        Accepts either a list/tuple of item hashes (looked up by their
        combined hash, matching the key used on ``set`` for bundled items) or
        a single raw hash key (e.g. an encode-server ``task_id``).
        """
        if isinstance(mm_hashes, (list, tuple)):
            key = (
                combined_hash
                if combined_hash is not None
                else self.combine_hashes(list(mm_hashes))
            )
        else:
            key = mm_hashes
        with self._lock:
            entry = self.mm_cache.get(key)
            if entry is not None and entry._invalidated:
                return None
            if entry is not None:
                self.mm_cache.move_to_end(key)
            return entry

    def get_single(self, mm_hash: int) -> Optional[EmbeddingResult]:
        """Get a single cached embedding by its raw hash (no combine_hashes)."""
        with self._lock:
            entry = self.mm_cache.get(mm_hash)
            if entry is not None and entry._invalidated:
                return None
            if entry is not None:
                self.mm_cache.move_to_end(mm_hash)
            return entry

    def acquire(self, mm_hash: int) -> Optional[EmbeddingResult]:
        """
        Acquire a reference to an embedding to prevent it from being evicted.

        This should be called before using an embedding for transfer operations.
        Must be paired with a call to release() when done.

        Args:
            mm_hash: The hash key of the embedding to acquire

        Returns:
            The EmbeddingResult if found, None otherwise
        """
        with self._lock:
            # Check main cache
            entry = self.mm_cache.get(mm_hash)
            if entry is not None and entry._invalidated:
                return None
            if entry is None:
                # Check pending delete queue - restore if found
                entry = self._pending_delete.get(mm_hash)
                if entry is not None and entry._invalidated:
                    return None
                if entry is not None:
                    self._pending_delete.pop(mm_hash)
                    # Restore to main cache
                    entry._pending_delete = False
                    data_size = _get_tensor_size(entry.embedding)
                    # 如果恢复后超过缓存上限，单次遍历逐出无引用的条目。
                    self._evict_to_fit(data_size)
                    self.mm_cache[mm_hash] = entry
                    self.current_size += data_size

            if entry is not None:
                entry.ref_count += 1
                # Move to end (most recently used)
                self.mm_cache.move_to_end(mm_hash)
            return entry

    def release(self, mm_hash: int) -> bool:
        """
        Release a reference to an embedding.

        When ref_count reaches 0 and the entry is marked for deletion,
        it will be moved to the pending delete queue or fully deleted.

        Args:
            mm_hash: The hash key of the embedding to release

        Returns:
            True if the entry was found and released, False otherwise
        """
        with self._lock:
            entry = self.mm_cache.get(mm_hash)
            if entry is None:
                entry = self._pending_delete.get(mm_hash)
            if entry is None:
                return False

            entry.ref_count = max(0, entry.ref_count - 1)

            # If marked for deletion and no more references, actually delete
            if entry.ref_count == 0 and entry._pending_delete:
                self._do_delete(mm_hash)

            return True

    def invalidate(self, mm_hash, *, expected=None) -> bool:
        """Stop new reads without freeing an embedding used by a transfer."""
        with self._lock:
            entry = self.mm_cache.get(mm_hash)
            if entry is None:
                entry = self._pending_delete.get(mm_hash)
            if entry is None or (expected is not None and entry is not expected):
                return False
            entry._invalidated = True
            entry._pending_delete = True
            if entry.ref_count == 0:
                self._do_delete(mm_hash)
            return True

    def _evict_to_fit(self, needed_size: int) -> bool:
        """
        Evict LRU entries (ref_count == 0) in a single pass until enough space is freed.

        Returns:
            True if enough space was freed (or already sufficient), False otherwise
        """
        if self.current_size + needed_size <= self.max_size:
            return True

        freed = 0
        target = self.current_size + needed_size - self.max_size
        to_evict = []
        for key, entry in self.mm_cache.items():
            if entry.ref_count == 0:
                freed += _get_tensor_size(entry.embedding)
                to_evict.append(key)
                if freed >= target:
                    break

        for key in to_evict:
            lru_size = _get_tensor_size(self.mm_cache[key].embedding)
            self._do_delete(key)
            logger.info(
                f"[evict] task={key}, size={lru_size}, new_size={self.current_size}"
            )

        return self.current_size + needed_size <= self.max_size

    def _do_delete(self, mm_hash: int) -> bool:
        """
        Actually delete an entry from the cache.

        Args:
            mm_hash: The hash key of the entry to delete

        Returns:
            True if deleted, False if not found
        """
        entry = self.mm_cache.pop(mm_hash, None)
        if entry is None:
            entry = self._pending_delete.pop(mm_hash, None)
            if entry is None:
                return False
        elif self._pending_delete.get(mm_hash) is entry:
            self._pending_delete.pop(mm_hash)

        self.current_size -= _get_tensor_size(entry.embedding)
        return True

    def set(
        self,
        mm_hash: int,
        embedding: EmbeddingResult,
        loc: Optional[torch.Tensor] = None,
    ) -> bool:
        """Set embedding to cache."""
        assert isinstance(embedding, EmbeddingResult), embedding

        with self._lock:
            pending = self._pending_delete.get(mm_hash)
            if pending is not None and pending.ref_count > 0:
                # release() is keyed by task ID. Do not rebind it while an old
                # invalidated/cleared generation is still being transferred.
                return False
            if mm_hash in self.mm_cache:
                existing = self.mm_cache[mm_hash]
                # Don't overwrite if currently being transferred
                if existing.ref_count > 0:
                    if existing._invalidated:
                        return False
                    self.mm_cache.move_to_end(mm_hash)
                    return True
                else:
                    # Remove old entry, fall through to normal insert path
                    self._do_delete(mm_hash)

            data_size = _get_tensor_size(embedding.embedding)
            logger.info(
                f"[set] task={mm_hash}, data_size={data_size}, "
                f"current={self.current_size}, max={self.max_size}"
            )

            if not self._evict_to_fit(data_size):
                logger.warning(
                    f"[set] task={mm_hash} - Cache full (current={self.current_size}, needed={data_size}, "
                    f"max={self.max_size}) but all entries are in use. Cannot cache new embedding."
                )
                return False

            self.mm_cache[mm_hash] = embedding
            self.current_size += data_size
            logger.info(
                f"[set] task={mm_hash} stored successfully, new_size={self.current_size}"
            )
            return True

    def has(self, mm_hash: int) -> bool:
        """Return whether the main cache contains a valid entry for the hash."""
        with self._lock:
            entry = self.mm_cache.get(mm_hash)
            return entry is not None and not entry._invalidated

    def clear(
        self, wait: bool = False, timeout: Optional[float] = None
    ) -> Dict[str, int]:
        """
        Clear entries from the cache.

        Args:
            wait: If True, wait for all in-use entries to be released
            timeout: Maximum time to wait (seconds), None means no timeout

        Returns:
            Stats dict with {'deleted': int, 'remaining': int, 'pending': int}
        """
        import time

        start_time = time.monotonic()
        stats = {"deleted": 0, "remaining": 0, "pending": 0}

        with self._lock:
            # 第一遍：删除无引用的条目
            for mm_hash, entry in list(self.mm_cache.items()):
                if entry.ref_count == 0:
                    self._do_delete(mm_hash)
                    stats["deleted"] += 1
                else:
                    entry._pending_delete = True
                    self._pending_delete[mm_hash] = entry
                    stats["pending"] += 1

            # 清空主缓存（只保留待删除队列中的）
            self.mm_cache.clear()

            if not wait:
                # 不等待，返回统计信息
                stats["remaining"] = len(self._pending_delete)
                self.current_size = sum(
                    _get_tensor_size(e.embedding) for e in self._pending_delete.values()
                )
                return stats

        # 等待模式：轮询等待所有条目释放（锁外）
        while self._pending_delete:
            # 检查超时
            if timeout is not None and (time.monotonic() - start_time) > timeout:
                logger.warning(
                    f"[clear] Timeout after {timeout}s, "
                    f"{len(self._pending_delete)} entries remaining"
                )
                break

            with self._lock:
                # 尝试清理已释放的条目
                released = []
                for mm_hash, entry in list(self._pending_delete.items()):
                    if entry.ref_count == 0:
                        self.current_size -= _get_tensor_size(entry.embedding)
                        del entry.embedding
                        released.append(mm_hash)
                        stats["deleted"] += 1

                # 从待删除队列移除已清理的
                for mm_hash in released:
                    del self._pending_delete[mm_hash]

            if self._pending_delete:
                time.sleep(0.01)  # 10ms 避免忙等

        stats["remaining"] = len(self._pending_delete)
        return stats

    def __len__(self):
        """Return the number of entries in the main cache."""
        with self._lock:
            return len(self.mm_cache)

    def snapshot_items(self) -> list:
        """Return a snapshot copy of all cache items, safe for iteration outside the lock."""
        with self._lock:
            return [
                (key, entry)
                for key, entry in self.mm_cache.items()
                if not entry._invalidated
            ]

    def snapshot(self) -> dict:
        """Return a shallow copy of cache entries visible to new readers."""
        return dict(self.snapshot_items())

    def available_size(self):
        """Return the number of entries in the main cache."""
        return self.__len__()

    def free(
        self, mm_hash: int, mm_embedding_allocator: BaseTokenToKVPoolAllocator
    ) -> bool:
        """
        Free an entry from the cache.

        For MultiModalStaticCache, the mm_embedding_allocator parameter is unused
        as this cache manages its own memory.
        """
        pass
