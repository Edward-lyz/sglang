# Mixin class for metadata management of Deepseek MHA forward (chunked prefix cache)
# More details can be found in python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py

import logging
from typing import List, Optional

import torch

from sglang.kernels.ops.kvcache.kv_indices import (
    create_chunked_prefix_cache_kv_indices,
    create_flashinfer_kv_indices_triton,
)
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_context import (
    get_req_to_token_pool,
    get_token_to_kv_pool,
)

logger = logging.getLogger(__name__)


class MLAExpandedPrefixKV:
    """FP8 K/V of one long request's MLA prefix, kept across its prefill chunks.

    The KV pool stores only the MLA latent, so every chunk of a chunked prefill
    re-expands the whole prefix through kv_b_proj: L tokens per chunk, ~L^2/x
    per request. This keeps the expanded FP8 K/V of every MLA layer on this
    stage for the request being prefilled, so a chunk only expands the tokens
    added since the previous chunk and attends to the prefix in one segment.

    One request at a time, keyed by rid; a different rid restarts expansion
    from token 0, which costs what the latent path costs for that chunk.
    Buffers are allocated once every local MLA layer has registered, and only
    if SGLANG_MLA_EXPANDED_KV_CACHE_MIN_FREE_GB stays free afterwards;
    otherwise the feature is off for the process.
    """

    def __init__(self):
        self.capacity = envs.SGLANG_MLA_EXPANDED_KV_CACHE_TOKENS.get()
        self.state = "probe" if self.capacity > 0 else "off"
        self.layer_shapes = {}  # layer_id -> (num_heads, k_head_dim, v_head_dim)
        self.slot = {}  # layer_id -> index into k_buf / v_buf
        self.k_buf: Optional[torch.Tensor] = None
        self.v_buf: Optional[torch.Tensor] = None
        self.rid = {}
        self.valid = {}

    @property
    def ready(self) -> bool:
        """Whether the buffers are allocated and the cache can be used."""
        return self.state == "ready"

    def register_layer(self, layer_id, num_heads, k_head_dim, v_head_dim, supported):
        """Record a local MLA layer's K/V shape; an unsupported layer disables."""
        if self.state != "probe":
            return
        if not supported:
            logger.info("[MLA EKV] layer %d unsupported; disabled", layer_id)
            self.state = "off"
            return
        self.layer_shapes.setdefault(layer_id, (num_heads, k_head_dim, v_head_dim))

    def try_allocate(self, device: torch.device):
        """Allocate the buffers once every layer registered and memory allows."""
        if self.state != "probe":
            return
        pool = get_token_to_kv_pool()
        pool = getattr(pool, "full_kv_pool", pool)
        if pool.dtype != torch.float8_e4m3fn:
            logger.info("[MLA EKV] KV cache dtype %s is not fp8; disabled", pool.dtype)
            self.state = "off"
            return
        n = len(pool.kv_buffer)
        if len(self.layer_shapes) < n:
            return  # not every local MLA layer has run yet
        shapes = set(self.layer_shapes.values())
        if len(shapes) != 1:
            self.state = "off"
            return
        num_heads, k_dim, v_dim = shapes.pop()
        need = n * self.capacity * num_heads * (k_dim + v_dim)
        free, _ = torch.cuda.mem_get_info(device)
        free += torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
        min_free = int(envs.SGLANG_MLA_EXPANDED_KV_CACHE_MIN_FREE_GB.get() * (1 << 30))
        if free - need < min_free:
            logger.info(
                "[MLA EKV] need %.1f GB for %d layers, free %.1f GB, min free "
                "%.1f GB; disabled",
                need / 2**30,
                n,
                free / 2**30,
                min_free / 2**30,
            )
            self.state = "off"
            return
        try:
            shape = (n, self.capacity, num_heads)
            fp8 = torch.float8_e4m3fn
            self.k_buf = torch.empty(shape + (k_dim,), dtype=fp8, device=device)
            self.v_buf = torch.empty(shape + (v_dim,), dtype=fp8, device=device)
        except torch.OutOfMemoryError:
            self.k_buf = self.v_buf = None
            logger.info("[MLA EKV] allocation OOM; disabled")
            self.state = "off"
            return
        self.slot = {lid: i for i, lid in enumerate(sorted(self.layer_shapes))}
        self.state = "ready"
        logger.info(
            "[MLA EKV] ready: %d layers x %d tokens, %.1f GB",
            n,
            self.capacity,
            need / 2**30,
        )

    def begin(self, layer_id: int, rid: str, prefix_len: int):
        """Return (k_buf, v_buf, first prefix token not yet expanded)."""
        if self.rid.get(layer_id) != rid:
            self.rid[layer_id] = rid
            self.valid[layer_id] = 0
        i = self.slot[layer_id]
        return self.k_buf[i], self.v_buf[i], min(self.valid[layer_id], prefix_len)

    def commit(self, layer_id: int, valid_len: int):
        """Mark the first ``valid_len`` prefix tokens of the layer as expanded."""
        self.valid[layer_id] = valid_len


_MLA_EXPANDED_PREFIX_KV: Optional[MLAExpandedPrefixKV] = None


def get_mla_expanded_prefix_kv() -> MLAExpandedPrefixKV:
    """Process-wide MLAExpandedPrefixKV, created on first use."""
    global _MLA_EXPANDED_PREFIX_KV
    if _MLA_EXPANDED_PREFIX_KV is None:
        _MLA_EXPANDED_PREFIX_KV = MLAExpandedPrefixKV()
    return _MLA_EXPANDED_PREFIX_KV


class ForwardBatchDeepSeekMHAMixin:
    # For MLA chunked prefix cache used in chunked prefill
    # Tell attention backend whether the kv cache needs to be attended in current pass
    attn_attend_prefix_cache: Optional[bool] = None
    # Number of prefix cache chunks
    num_prefix_chunks: Optional[int] = None
    # Index of current chunk, used by attention backend
    prefix_chunk_idx: Optional[int] = None
    # Maximum number of tokens in each chunk per sequence. Computed from maximum chunk capacity
    prefix_chunk_len: Optional[int] = None
    # Start positions of prefix cache for each chunk, (num_prefix_chunks, batch_size)
    prefix_chunk_starts: Optional[torch.Tensor] = None
    # Start positions of prefix cache for each chunk, (num_prefix_chunks, batch_size), need prefix_chunk_starts_cpu for dcp all gather kv cache
    prefix_chunk_starts_cpu: Optional[torch.Tensor] = None
    # length of prefix cache for each chunk, (num_prefix_chunks, batch_size)
    prefix_chunk_seq_lens_cpu: Optional[torch.Tensor] = None
    # Lengths of prefix cache for each chunk, (num_prefix_chunks, batch_size)
    prefix_chunk_seq_lens: Optional[torch.Tensor] = None
    # Accumulated lengths of prefix cache for each chunk, (num_prefix_chunks, batch_size + 1)
    prefix_chunk_cu_seq_lens: Optional[torch.Tensor] = None
    # Max lengths of prefix cache for each chunk, (num_prefix_chunks,)
    prefix_chunk_max_seq_lens: Optional[List[int]] = None
    # Per-chunk flag: True if any sequence has kv_len==0 in that chunk.
    # Precomputed on CPU to avoid GPU-CPU sync in the hot path.
    prefix_chunk_has_zero_kv: Optional[List[bool]] = None
    # Number of tokens in each prefix cache chunk, (num_prefix_chunks,)
    prefix_chunk_num_tokens: Optional[List[int]] = None
    # KV Indices for each chunk
    prefix_chunk_kv_indices: Optional[List[torch.Tensor]] = None
    # For MLA chunked prefix cache used in chunked prefill
    # Tell attention backend whether lse needs to be returned
    mha_return_lse: Optional[bool] = None
    # Whether to apply MHA_ONE_SHOT forward method
    mha_one_shot: Optional[bool] = None
    # KV Indices for MHA_ONE_SHOT forward method
    mha_one_shot_kv_indices: Optional[torch.Tensor] = None
    # Prefix K/V comes from MLAExpandedPrefixKV in a single segment
    mla_ekv: bool = False

    def get_max_chunk_capacity(self):
        return envs.SGLANG_MAX_KV_CHUNK_CAPACITY.get()

    def _mla_ekv_eligible(self, device: torch.device) -> bool:
        store = get_mla_expanded_prefix_kv()
        if store.state == "off" or self.batch_size != 1 or not self.rids:
            return False
        from sglang.srt.runtime_context import get_parallel

        if get_parallel().dcp_enabled:
            return False
        store.try_allocate(device)
        return store.ready and (
            self.extend_prefix_lens_cpu[0] + self.extend_seq_lens_cpu[0]
            <= store.capacity
        )

    def set_prefix_chunk_idx(self, idx: int):
        self.prefix_chunk_idx = idx

    def set_attn_attend_prefix_cache(self, attn_attend_prefix_cache: bool):
        self.attn_attend_prefix_cache = attn_attend_prefix_cache

    def prepare_chunked_kv_indices(self, device: torch.device):
        self.prefix_chunk_kv_indices = []
        req_to_token = get_req_to_token_pool().req_to_token
        for idx in range(self.num_prefix_chunks):
            chunk_starts = self.prefix_chunk_starts[idx]
            chunk_seq_lens = self.prefix_chunk_seq_lens[idx]
            chunk_cu_seq_lens = self.prefix_chunk_cu_seq_lens[idx]
            num_chunk_tokens = self.prefix_chunk_num_tokens[idx]

            chunk_kv_indices = torch.empty(
                num_chunk_tokens, dtype=torch.int32, device=device
            )

            create_chunked_prefix_cache_kv_indices[(self.batch_size,)](
                req_to_token,
                self.req_pool_indices,
                chunk_starts,
                chunk_seq_lens,
                chunk_cu_seq_lens,
                chunk_kv_indices,
                req_to_token.shape[1],
            )
            self.prefix_chunk_kv_indices.append(chunk_kv_indices)

    # Here we suppose the length of each chunk is equal
    # For example, if we have 4 sequences with prefix length [256, 512, 768, 1024], prefix_chunk_len = 256
    # num_prefix_chunks = cdiv(1024, 256) = 4
    # prefix_chunk_starts = [[0, 0, 0, 0], [256, 256, 256, 256], [512, 512, 512, 512], [768, 768, 768, 768]]
    # prefix_chunk_ends = [[256, 256, 256, 256], [256, 512, 512, 512], [256, 512, 768, 768], [256, 512, 768, 1024]]
    # prefix_chunk_seq_lens = [[256, 256, 256, 256], [0, 256, 256, 256], [0, 0, 256, 256], [0, 0, 0, 256]]
    # TODO: Implement a better way to allocate chunk lengths that uses memory spaces more efficiently.
    def get_prefix_chunk_seq_lens(
        self, prefix_lens: torch.Tensor, num_prefix_chunks: int, prefix_chunk_len: int
    ):
        device = prefix_lens.device
        prefix_chunk_starts = (
            torch.arange(num_prefix_chunks, device=device, dtype=torch.int32)
            .unsqueeze(1)
            .expand(-1, self.batch_size)
            * prefix_chunk_len
        )
        prefix_chunk_ends = torch.min(
            prefix_lens.unsqueeze(0),
            prefix_chunk_starts + prefix_chunk_len,
        ).to(torch.int32)

        prefix_chunk_seq_lens = (
            (prefix_chunk_ends - prefix_chunk_starts).clamp(min=0).to(torch.int32)
        )

        return prefix_chunk_starts, prefix_chunk_seq_lens

    # Called before each attention module if using chunked kv cache for prefill
    # Some of the codes are adapted from https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/mla/common.py
    def prepare_chunked_prefix_cache_info(self, device: torch.device):

        from sglang.srt.mem_cache.memory_pool import (
            HybridLinearKVPool,
            MLATokenToKVPool,
        )

        token_to_kv_pool = get_token_to_kv_pool()
        assert isinstance(token_to_kv_pool, MLATokenToKVPool) or (
            isinstance(token_to_kv_pool, HybridLinearKVPool)
            and isinstance(token_to_kv_pool.full_kv_pool, MLATokenToKVPool)
        ), "Currently chunked prefix cache can only be used by Deepseek models"

        if not any(self.extend_prefix_lens_cpu):
            self.num_prefix_chunks = 0
            return

        if self.prefix_chunk_len is not None:
            # Chunked kv cache info already prepared by prior modules
            return

        self.prefix_chunk_idx = -1

        # A single long request whose expanded prefix K/V is cached attends to
        # the whole prefix in one segment (see MLAExpandedPrefixKV).
        self.mla_ekv = self._mla_ekv_eligible(device)

        # chunk_capacity is the maximum number of tokens in each chunk
        chunk_capacity = (
            max(self.extend_prefix_lens_cpu)
            if self.mla_ekv
            else self.get_max_chunk_capacity()
        )
        self.prefix_chunk_len = chunk_capacity // self.batch_size

        self.num_prefix_chunks = (
            max(self.extend_prefix_lens_cpu) + self.prefix_chunk_len - 1
        ) // self.prefix_chunk_len

        # Here we compute chunk lens twice to avoid stream sync, once on gpu and once on cpu.
        prefix_chunk_starts_cuda, prefix_chunk_seq_lens_cuda = (
            self.get_prefix_chunk_seq_lens(
                self.extend_prefix_lens,
                self.num_prefix_chunks,
                self.prefix_chunk_len,
            )
        )
        prefix_chunk_starts_cpu, prefix_chunk_seq_lens_cpu = (
            self.get_prefix_chunk_seq_lens(
                torch.tensor(self.extend_prefix_lens_cpu),
                self.num_prefix_chunks,
                self.prefix_chunk_len,
            )
        )
        self.prefix_chunk_starts = prefix_chunk_starts_cuda
        self.prefix_chunk_seq_lens = prefix_chunk_seq_lens_cuda

        # set prefix_chunk_starts_cpu and prefix_chunk_seq_lens_cpu for dcp to gather chunk kv cache with arbitrary lens
        self.prefix_chunk_starts_cpu = prefix_chunk_starts_cpu
        self.prefix_chunk_seq_lens_cpu = prefix_chunk_seq_lens_cpu
        # Metadata for attention backend
        self.prefix_chunk_cu_seq_lens = torch.zeros(
            self.num_prefix_chunks,
            self.batch_size + 1,
            device=device,
            dtype=torch.int32,
        )
        self.prefix_chunk_cu_seq_lens[:, 1:] = prefix_chunk_seq_lens_cuda.cumsum(
            dim=1
        ).to(torch.int32)
        self.prefix_chunk_max_seq_lens = prefix_chunk_seq_lens_cpu.max(
            dim=1
        ).values.tolist()

        self.prefix_chunk_num_tokens = prefix_chunk_seq_lens_cpu.sum(dim=1).tolist()
        assert max(self.prefix_chunk_num_tokens) <= chunk_capacity

        # Per-chunk flag: does any sequence have kv_len == 0?
        # Pure CPU check (prefix_chunk_seq_lens_cpu is on CPU), no GPU sync.
        self.prefix_chunk_has_zero_kv = [
            bool((prefix_chunk_seq_lens_cpu[i] == 0).any())
            for i in range(self.num_prefix_chunks)
        ]

        # Precompute the kv indices for each chunk
        self.prepare_chunked_kv_indices(device)

    def fetch_mha_one_shot_kv_indices(self):
        if self.mha_one_shot_kv_indices is not None:
            return self.mha_one_shot_kv_indices
        batch_size = self.batch_size
        paged_kernel_lens_sum = sum(self.seq_lens_cpu)
        kv_indices = torch.empty(
            paged_kernel_lens_sum,
            dtype=torch.int32,
            device=self.req_pool_indices.device,
        )
        kv_indptr = torch.zeros(
            batch_size + 1,
            dtype=torch.int32,
            device=self.req_pool_indices.device,
        )
        kv_indptr[1:] = torch.cumsum(self.seq_lens, dim=0)
        req_to_token = get_req_to_token_pool().req_to_token
        create_flashinfer_kv_indices_triton[(self.batch_size,)](
            req_to_token,
            self.req_pool_indices,
            self.seq_lens,
            kv_indptr,
            None,
            kv_indices,
            req_to_token.shape[1],
        )
        self.mha_one_shot_kv_indices = kv_indices
        return kv_indices
