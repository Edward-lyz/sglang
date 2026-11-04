"""
Scatter-based multimodal embedding cache.

Uses a pre-allocated buffer + page allocator to store embeddings as
ScatteredTensor (scattered across non-contiguous pages), enabling
zero-copy RDMA transfer via mooncake.
"""

import ctypes
import logging
import math
import threading
import weakref
from collections import OrderedDict
from typing import Optional

import torch

from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult

logger = logging.getLogger(__name__)

# ═══════════════════════ type aliases ═════════════════════════════

Ptr = int  # 内存起始地址
Nbytes = int  # 字节数

# ═══════════════════════ helpers ══════════════════════════════════


class _DummyKVCache:
    pass


def _tensor_from_ptr(ptr: int, n_elems: int, dtype: torch.dtype) -> torch.Tensor:
    """将裸指针 + 元素数包装为一个 1-D torch.Tensor（零拷贝）。

    利用 ctypes 将 ptr 包装为 Python buffer，再通过 torch.frombuffer 创建 tensor。
    返回的 tensor 直接引用原始内存，不做任何拷贝。
    """
    elem_size = torch.tensor([], dtype=dtype).element_size()
    nbytes = n_elems * elem_size
    c_buf = (ctypes.c_char * nbytes).from_address(ptr)
    return torch.frombuffer(c_buf, dtype=dtype, count=n_elems)


# ═══════════════════════ ScatteredTensor (RAII) ═══════════════════


class ScatteredTensor:
    """拥有多段不相邻连续内存的 tensor，RAII 语义。

    构造时由 ScatterMemPool 分配页面，析构时自动归还。

    Parameters
    ----------
    mems : list[(ptr, nbytes)]
        每段连续内存的 (起始地址, 字节数)。各段按顺序拼接即为完整 tensor 数据。
    shape : tuple[int, ...]
        逻辑 shape。
    dtype : torch.dtype
    allocated_bytes : int
        实际分配的字节数（含页面对齐冗余）。
    pool : ScatterMemPool
        分配来源，通过 weakref 持有，析构时归还页面。
    """

    def __init__(
        self,
        mems: list[tuple[Ptr, Nbytes]],
        shape: torch.Size,
        dtype: torch.dtype,
        allocated_bytes: int,
        pool: "ScatterMemPool",
    ):
        self._mems = mems.copy()
        self.shape = shape
        self.dtype = dtype
        self.allocated_bytes = allocated_bytes
        self._pool_ref = weakref.ref(pool)

    def __del__(self):
        pool = self._pool_ref()
        if pool is not None:
            pool.free(self)

    @property
    def _elem_size(self) -> int:
        return torch.tensor([], dtype=self.dtype).element_size()

    @property
    def nbytes(self) -> int:
        """逻辑数据的字节数（shape 决定）。"""
        return math.prod(self.shape) * self._elem_size

    def as_tensor(self) -> torch.Tensor:
        """创建新 tensor，将各段数据拷贝到其中，返回目标 shape 的连续 tensor。"""
        td = self.dtype
        elem_size = self._elem_size
        out = torch.empty(self.shape, dtype=td)
        flat = out.view(-1)

        offset = 0
        for ptr, nbytes in self._mems:
            n_elems = nbytes // elem_size
            chunk = _tensor_from_ptr(ptr, n_elems, td)
            flat[offset : offset + n_elems].copy_(chunk, non_blocking=True)
            offset += n_elems

        return out

    def from_tensor(self, other: torch.Tensor):
        """将 other 的数据按序写入自己的各段 mem 中。"""
        if other.dtype != self.dtype:
            raise ValueError(
                f"dtype mismatch: expected {self.dtype}, got {other.dtype}"
            )
        if other.shape != self.shape:
            raise ValueError(
                f"shape mismatch: expected {self.shape}, got {other.shape}"
            )
        flat = other.contiguous().view(-1)
        elem_size = self._elem_size

        offset = 0
        for ptr, nbytes in self._mems:
            n_elems = nbytes // elem_size
            dst = _tensor_from_ptr(ptr, n_elems, self.dtype)
            dst.copy_(flat[offset : offset + n_elems], non_blocking=True)
            offset += n_elems


# ═══════════════════════ ScatterMemPool ══════════════════════════


class ScatterMemPool:
    """Pre-allocated buffer + page allocator for scattered memory.

    纯内存池，不涉及缓存策略、引用计数或淘汰逻辑。

    ⚠️ **线程不安全**：所有方法必须在同一线程（或 asyncio 单线程）中调用。
    当前在 encode server 中，所有调用都在 asyncio 事件循环内，满足此约束。
    """

    def __init__(
        self,
        max_bytes: int,
        embed_dim: int,
        page_size: int,
        dtype: torch.dtype,
        device: str = "cpu",
    ):
        elem_size = torch.tensor([], dtype=dtype).element_size()

        self._bytes_per_page = page_size * embed_dim * elem_size
        self._num_pages = max_bytes // self._bytes_per_page
        self._page_size = page_size
        self._embed_dim = embed_dim
        self._dtype = dtype
        self._device = device

        from sglang.srt.mem_cache.allocator import (
            TokenToKVPoolAllocator as FreeListAllocator,
        )

        self._allocator = FreeListAllocator(
            size=self._num_pages,
            dtype=torch.int64,
            device="cpu",
            kvcache=_DummyKVCache(),
            need_sort=False,
        )

        total_tokens = (self._num_pages + 1) * page_size
        if device == "cpu":
            self._buf = torch.zeros(
                total_tokens, embed_dim, dtype=dtype, pin_memory=True
            )
        else:
            self._buf = torch.zeros(total_tokens, embed_dim, dtype=dtype, device=device)

    def get_buffer_info(self) -> tuple[int, int]:
        """返回 buffer 的 (起始地址, 字节数)，用于 RDMA 注册。

        Returns:
            tuple[int, int]: (data_ptr, nbytes) 起始地址和字节数
        """
        data_ptr = self._buf.data_ptr()
        nbytes = self._buf.numel() * self._buf.element_size()
        return data_ptr, nbytes

    # ─────────────── alloc / free ────────────────

    def alloc(self, shape: torch.Size, dtype: torch.dtype) -> ScatteredTensor:
        """**⚠️ 线程不安全**：分配页面，构造 ScatteredTensor 并返回。"""
        elem_size = torch.tensor([], dtype=dtype).element_size()
        nbytes = math.prod(shape) * elem_size
        page_bytes = self._bytes_per_page
        need_pages = math.ceil(nbytes / page_bytes)

        page_indices = self._allocator.alloc(need_pages)
        if page_indices is None:
            raise RuntimeError(
                f"no free pages (need={need_pages}, "
                f"avail={self._allocator.available_size()})"
            )

        idx_list = page_indices.tolist()
        base_ptr = self._buf.data_ptr()
        remaining_bytes = nbytes

        mems: list[tuple[Ptr, Nbytes]] = []
        i = 0
        while i < len(idx_list):
            start_idx = idx_list[i]
            run_len = 1
            while (
                i + run_len < len(idx_list)
                and idx_list[i + run_len] == start_idx + run_len
            ):
                run_len += 1
            ptr = base_ptr + start_idx * page_bytes
            chunk_bytes = min(run_len * page_bytes, remaining_bytes)
            mems.append((ptr, chunk_bytes))
            remaining_bytes -= chunk_bytes
            i += run_len

        return ScatteredTensor(
            mems=mems,
            shape=shape,
            dtype=dtype,
            allocated_bytes=need_pages * page_bytes,
            pool=self,
        )

    def free(self, tensor: ScatteredTensor):
        """**⚠️ 线程不安全**：归还 tensor 持有的页面到 free list。"""
        page_bytes = self._bytes_per_page
        base_ptr = self._buf.data_ptr()

        page_indices: list[int] = []
        for ptr, nbytes in tensor._mems:
            start_page = (ptr - base_ptr) // page_bytes
            n_pages = math.ceil(nbytes / page_bytes)
            page_indices.extend(range(start_page, start_page + n_pages))

        if page_indices:
            self._allocator.free(torch.tensor(page_indices, dtype=torch.int64))

    def place(self, embedding: torch.Tensor) -> ScatteredTensor:
        """**⚠️ 线程不安全**：将 Tensor 拷贝到 pool 中，返回 handler。

        如果已经是 ScatteredTensor 则直接返回（无需拷贝），
        否则分配内存并拷贝数据。

        Args:
            embedding: 输入 tensor，可以是 torch.Tensor 或 ScatteredTensor

        Returns:
            ScatteredTensor handler，用于操作 pool 中的内存
        """
        if isinstance(embedding, ScatteredTensor):
            return embedding
        st = self.alloc(embedding.shape, embedding.dtype)
        st.from_tensor(embedding)
        return st

    # ─────────────── capacity queries ────────────

    @property
    def bytes_per_page(self) -> int:
        """每页字节数。"""
        return self._bytes_per_page

    @property
    def total_pages(self) -> int:
        """总页数。"""
        return self._num_pages

    @property
    def max_size(self) -> int:
        """总 bytes 数。"""
        return self._num_pages * self._bytes_per_page

    @property
    def current_size(self) -> int:
        """已使用的 bytes 数。"""
        return (
            self._num_pages - self._allocator.available_size()
        ) * self._bytes_per_page

    def available_size(self) -> int:
        """可用 page 数。"""
        return self._allocator.available_size()


# ═══════════════════════ ScatterMultiModalCache ═══════════════════


class ScatterMultiModalCache:
    """LRU cache with ref-counting, backed by ScatterMemPool.

    Embeddings are stored as ScatteredTensor (scattered across pages),
    enabling zero-copy RDMA transfer.
    """

    def __init__(self, mem_pool: ScatterMemPool):
        self.pool = mem_pool
        self.mm_cache: OrderedDict[str, EmbeddingResult] = OrderedDict()
        self._cache_lock = threading.Lock()

    # ─────────────── internal helpers ────────────────

    def _entry_pages(self, entry: EmbeddingResult) -> int:
        """返回 entry 占用的 page 数。"""
        emb = entry.embedding
        assert isinstance(emb, ScatteredTensor)
        return math.ceil(emb.allocated_bytes / self.pool.bytes_per_page)

    def _cache_pages(self) -> int:
        """返回 mm_cache 中所有可驱逐条目占用的 page 总数（含锁）。"""
        with self._cache_lock:
            return sum(self._entry_pages(e) for e in self.mm_cache.values())

    def _evict_to_fit(self, need_pages: int) -> bool:
        """逐出 LRU 端条目直到腾出足够 page。"""
        if self.pool.available_size() >= need_pages:
            return True

        if need_pages > self.pool.total_pages:
            return False

        freed = 0
        cache_empty = False
        avail = self.pool.available_size()

        with self._cache_lock:
            while avail + freed < need_pages:
                if not self.mm_cache:
                    cache_empty = True
                    break
                _, entry = self.mm_cache.popitem(last=False)
                freed += self._entry_pages(entry)
                del entry  # release pages before the final availability check

        if cache_empty and avail + freed < need_pages:
            logger.warning(
                "_evict_to_fit: cache empty but still need %d pages, "
                "%d pages in use outside cache. Consider increasing pool size.",
                need_pages - (avail + freed),
                self.pool.total_pages - avail,
            )

        return self.pool.available_size() >= need_pages

    # ─────────────── public API ──────────────────────

    def has(self, mm_hash: str) -> bool:
        """返回缓存中是否存在该 mm_hash。"""
        return mm_hash in self.mm_cache

    def get(self, mm_hash: str) -> Optional[EmbeddingResult]:
        """获取条目并将其移到 LRU 末端；不存在时返回 None。"""
        with self._cache_lock:
            entry = self.mm_cache.get(mm_hash)
            if entry is not None:
                self.mm_cache.move_to_end(mm_hash)
            return entry

    def _alloc_scattered(
        self, shape: torch.Size, dtype: torch.dtype
    ) -> Optional[ScatteredTensor]:
        """从 pool 分配 ScatteredTensor，空间不足时淘汰缓存条目。

        返回 ScatteredTensor（已分配，未写入数据），失败返回 None。
        """
        nbytes: int = math.prod(shape) * torch.tensor([], dtype=dtype).element_size()
        need_pages = math.ceil(nbytes / self.pool.bytes_per_page)
        if not self._evict_to_fit(need_pages):
            avail_pages = self.pool.available_size()
            total_pages = self.pool.total_pages
            used_pages = total_pages - avail_pages
            evictable_pages = self._cache_pages()
            in_flight_pages = used_pages - evictable_pages
            logger.warning(
                "[alloc] cache full, need %d pages | "
                "pool: total=%d avail=%d used=%d | "
                "evictable(in cache)=%d in-flight(outside cache)=%d",
                need_pages,
                total_pages,
                avail_pages,
                used_pages,
                evictable_pages,
                in_flight_pages,
            )
            return None
        return self.pool.alloc(shape, dtype)

    def set(self, mm_hash: str, es: EmbeddingResult) -> bool:
        """将 EmbeddingResult 存入缓存。

        如果 es.embedding 是 torch.Tensor，会从 pool 分配 ScatteredTensor
        并将数据拷贝进去；如果已经是 ScatteredTensor 则直接存入。
        """
        embedding = es.embedding
        if isinstance(embedding, torch.Tensor):
            scattered = self._alloc_scattered(embedding.shape, embedding.dtype)
            if scattered is None:
                return False
            scattered.from_tensor(embedding)
            es.embedding = scattered

        assert isinstance(
            es.embedding, ScatteredTensor
        ), f"expected ScatteredTensor, got {type(es.embedding)}"
        with self._cache_lock:
            if mm_hash in self.mm_cache:
                self.mm_cache.pop(mm_hash, None)
            self.mm_cache[mm_hash] = es
        return True

    def free(self, mm_hash: str) -> bool:
        """删除条目。"""
        with self._cache_lock:
            return self.mm_cache.pop(mm_hash, None) is not None

    def invalidate(self, mm_hash, *, expected=None) -> bool:
        """Drop the cache reference; in-flight ScatteredTensor holders keep pages."""
        with self._cache_lock:
            entry = self.mm_cache.get(mm_hash)
            if entry is None or (expected is not None and entry is not expected):
                return False
            entry._invalidated = True
            self.mm_cache.pop(mm_hash)
            return True

    def clear(self):
        """清空所有缓存条目。"""
        with self._cache_lock:
            self.mm_cache.clear()

    @property
    def max_size(self) -> int:
        """返回总 bytes 数。"""
        return self.pool.max_size

    @property
    def current_size(self) -> int:
        """返回已使用的 bytes 数。"""
        return self.pool.current_size

    def available_size(self) -> int:
        """返回可用 page 数。"""
        return self.pool.available_size()

    def __len__(self):
        return len(self.mm_cache)

    def snapshot(self) -> dict:
        """返回 cache 所有条目的快照副本。"""
        with self._cache_lock:
            return self.mm_cache.copy()
