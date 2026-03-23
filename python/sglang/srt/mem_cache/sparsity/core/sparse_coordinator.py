import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import BaseSparseAlgorithm
from sglang.srt.mem_cache.sparsity.backend.backend_adaptor import BackendAdaptor

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)
_global_sparse_coordinator: Optional["SparseCoordinator"] = None


def _call_coordinator(method_name: str, *args, default=None, **kwargs):
    """Dispatch to the global SparseCoordinator if one is registered.

    All module-level sparse hooks use this to stay no-op when sparse attention
    is disabled (coordinator is None), avoiding if-checks at every call site.
    """
    coordinator = _global_sparse_coordinator
    if coordinator is None:
        return default
    return getattr(coordinator, method_name)(*args, **kwargs)


def register_sparse_coordinator(coordinator: Optional["SparseCoordinator"]) -> None:
    global _global_sparse_coordinator
    _global_sparse_coordinator = coordinator


def get_sparse_coordinator() -> Optional["SparseCoordinator"]:
    return _global_sparse_coordinator


def clear_sparse_coordinator() -> None:
    register_sparse_coordinator(None)


def notify_sparse_request_begin(req) -> None:
    """Called after req_pool_idx is assigned so the coordinator can set up per-request state."""
    _call_coordinator("on_request_begin", req)


def finish_sparse_request(req, tree_cache) -> tuple[bool, Optional[int]]:
    """Finalize sparse state for a completed request.

    Returns (bypass_tree_cache, released_prefix_len).  When bypass is True,
    the caller must free KV slots directly instead of going through tree cache,
    because DSA has already released the prompt prefix to host memory.
    """
    result = _call_coordinator("finish_request", req, tree_cache)
    return (False, None) if result is None else result


def handle_disagg_cache_received(req, token_to_kv_pool_allocator) -> None:
    """Finalize DSA prompt residency after the disagg transfer lands on the decode worker."""
    _call_coordinator("on_disagg_cache_received", req, token_to_kv_pool_allocator)


def get_disagg_state_indices(req_pool_idx: int, token_count: int, fallback_indices):
    """Return the KV indices that represent disagg state for this request.

    With DSA enabled, these come from the indexer slot namespace (stable slots
    that survive host offloading) instead of the main req_to_token mapping.
    """
    return _call_coordinator(
        "get_disagg_state_indices",
        req_pool_idx,
        token_count,
        fallback_indices,
        default=fallback_indices,
    )


def prepare_sparse_decode_step(forward_batch: "ForwardBatch") -> None:
    """Precompute DSA metadata BEFORE cuda graph replay.

    This is the only place where Python-level DSA logic runs per decode step;
    the graph itself replays a fixed kernel sequence that reads from the
    precomputed buffers.
    """
    _call_coordinator("prepare_decode_step", forward_batch)


def init_sparse_cuda_graph_state(max_bs: int, max_topk: int) -> None:
    """Freeze DSA workspace capacity for cuda graph capture.

    Must be called before graph capture so that all DSA buffers have their
    final size and addresses remain stable across replays.
    """
    _call_coordinator("init_cuda_graph_state", max_bs=max_bs, max_topk=max_topk)


def init_sparse_cuda_graph_state_for_model(*, max_bs: int, model_config) -> None:
    """Convenience wrapper that extracts topk from model config before freezing."""
    from sglang.srt.configs.model_config import get_nsa_index_topk, is_deepseek_nsa

    max_topk = (
        get_nsa_index_topk(model_config.hf_config)
        if is_deepseek_nsa(model_config.hf_config)
        else 0
    )
    if max_topk > 0:
        init_sparse_cuda_graph_state(max_bs=max_bs, max_topk=max_topk)


def get_sparse_reserved_token_count() -> int:
    """Token count carved out by the sparse cache allocator (not a memory leak)."""
    return _call_coordinator("get_reserved_token_count", default=0)


def get_sparse_indexer_available_token_count() -> Optional[int]:
    """Available capacity in the indexer slot namespace, or None if DSA is off."""
    return _call_coordinator("get_indexer_available_token_count", default=None)


def get_sparse_host_available_token_count() -> Optional[int]:
    """Available host-backed KV capacity for DSA offload, or None if DSA is off."""
    return _call_coordinator("get_host_available_token_count", default=None)


def get_sparse_host_kv_pool():
    """Return the DSA host KV pool, or None when DSA is off."""
    return _call_coordinator("get_host_kv_pool", default=None)


def resolve_sparse_selected_indices(selected_indices, forward_batch):
    """Remap logical topk indices to physical addresses through the sparse cache.

    Returns (remapped_indices, is_physical). When is_physical is True, the
    attention backend must skip its own page-table translation.
    """
    return _call_coordinator(
        "resolve_selected_indices",
        selected_indices,
        forward_batch,
        default=(selected_indices, False),
    )


def get_sparse_indexer_metadata(page_table_1, forward_batch):
    """Get indexer-aware page tables for NSA indexer lookup.

    The coordinator/DSARuntime decides whether to return precomputed metadata
    or compute it on the fly.
    """
    return _call_coordinator(
        "get_indexer_metadata",
        page_table_1,
        forward_batch,
        default=(False, None, None),
    )


def get_sparse_indexer_write_loc(forward_batch):
    """Get the indexer write location for the current decode step.

    Returns None when DSA is off (caller falls back to out_cache_loc).
    """
    return _call_coordinator("get_indexer_write_loc", forward_batch, default=None)


def is_sparse_cuda_graph_required() -> bool:
    """DSA offload mandates cuda graph replay for decode batches."""
    coordinator = _global_sparse_coordinator
    return bool(coordinator is not None and coordinator.config.is_dsa_offload)


class RequestTrackers:
    """State tracker for sparse attention requests."""

    def __init__(
        self,
        max_pool_size: int,
        device: torch.device,
        num_layers: int,
        min_sparse_prompt_len: int,
        max_context_len: int,
    ):
        self.device = device
        self.num_layers = num_layers

        self.repr_constructed = torch.zeros(
            max_pool_size, dtype=torch.bool, device=device
        )
        self.prompt_lens = torch.zeros(max_pool_size, dtype=torch.int64, device=device)
        self.last_constructed_page = torch.zeros(
            max_pool_size, dtype=torch.int64, device=device
        )

        # TODO: Add more trackers for hierarchical KVCache management

    def register(self, idx: int, prompt_len: int) -> None:
        self.repr_constructed[idx] = False
        self.prompt_lens[idx] = prompt_len
        self.last_constructed_page[idx] = 0

    def clear(self, idx: int) -> None:
        self.repr_constructed[idx] = False
        self.prompt_lens[idx] = 0
        self.last_constructed_page[idx] = 0


@dataclass
class SparseConfig:
    """Configuration for sparse attention."""

    top_k: int = 2048
    device_buffer_size: int = 4096
    host_to_device_ratio: int = 2
    algorithm: Optional[str] = None
    backend: Optional[str] = None
    page_size: Optional[int] = None
    min_sparse_prompt_len: Optional[int] = None
    sparse_extra_config: dict = field(
        default_factory=dict
    )  # Algorithm-specific config, parsed by each algorithm
    is_dsa_offload: bool = False  # True when DSA offload is enabled for deepseek_nsa


class SparseCoordinator:
    """
    Coordinator for sparse attention with retrievable KV cache compression.

    This coordinator framework is designed for decode-phase retrievable algorithms
    (e.g., Quest, PQCache, SnapKV) that dynamically select important KV cache entries
    based on current queries. It manages the lifecycle of sparse attention including
    representation construction, sparse retrieval, and token offloading.

    Request Lifecycle and API Calls:
        1. Request Start:
           - on_request_begin(req) -> Register request and initialize state

        2. Prefill Phase:
           - attention_end(...)    -> Construct representations

        3. Decode Phase:
           - attention_begin(...)  -> Identify important KV, load offloaded KVCache, adapt attention metadata
           - attention_end(...)    -> Construct/update representations

        4. Request End:
           - on_request_end(req) -> Clean up state and resources
    """

    def __init__(
        self,
        config: SparseConfig,
        algorithm: BaseSparseAlgorithm,
        backend_adaptor: Optional[BackendAdaptor],
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool: KVCache,
        start_layer: int,
        end_layer: int,
        device: torch.device,
        dsa_runtime=None,
    ):
        self.config = config
        self.algorithm = algorithm
        self.backend_adaptor = backend_adaptor
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.device = device
        self.page_size = config.page_size
        self._dsa_runtime = dsa_runtime

        self.states = RequestTrackers(
            req_to_token_pool.req_to_token.shape[0],
            device,
            end_layer - start_layer + 1,
            self.config.min_sparse_prompt_len,
            self.req_to_token_pool.max_context_len,
        )

        # Initialize algorithm representation pool and context
        self.algorithm.initialize_representation_pool(
            start_layer,
            end_layer,
            self.token_to_kv_pool,
            self.req_to_token_pool,
            self.states,
        )

        logger.info(
            f"SparseCoordinator initialized with sparse algorithm={type(algorithm).__name__}"
        )

    def _call_dsa_runtime(self, method_name: str, *args, default=None, **kwargs):
        """Forward a call to the DSA offload runtime, returning default when DSA is off."""
        runtime = self._dsa_runtime
        if runtime is None:
            return default
        return getattr(runtime, method_name)(*args, **kwargs)

    def _clear_request_state(self, req: "Req") -> None:
        if req.req_pool_idx is not None:
            self.states.clear(req.req_pool_idx)

    def on_request_begin(self, req: "Req") -> None:
        """Register request state for sparse attention tracking and DSA page table init."""
        if req.req_pool_idx is not None:
            self.states.register(req.req_pool_idx, len(req.origin_input_ids))
        self._call_dsa_runtime("on_request_begin", req)

    def on_request_end(self, req: "Req") -> None:
        """Release sparse state and DSA resources (sparse cache slots, host memory)."""
        if req.req_pool_idx is not None:
            self._clear_request_state(req)
            self._call_dsa_runtime("on_request_end", req)

    def finish_request(
        self, req: "Req", tree_cache
    ) -> tuple[bool, Optional[int]]:
        """Finalize a request: read DSA state then release resources.

        Merges on_request_end_finish + on_request_end into a single call to
        guarantee the required ordering (read released_prefix_len before
        freeing sparse pages and host slots).
        """
        bypass, released_prefix_len = self._call_dsa_runtime(
            "on_request_end_finish",
            req,
            tree_cache,
            default=(False, None),
        )
        self._call_dsa_runtime("on_request_end", req)
        self._clear_request_state(req)
        return bypass, released_prefix_len

    def attention_begin(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        attn_metadata: Optional[Any],
        **kwargs,
    ) -> Optional[Any]:
        """
        Handle attention begin event. Called before each attention pass starts.

        Identify important KV entries via sparse algorithm, load offloaded KVCache if needed,
        and adapt attention metadata for the attention backend.
        """
        if layer.layer_id == self.start_layer:
            self.backend_adaptor.save_original_metadata(attn_metadata)

        return self._handle_sparse_retrieve(
            query, layer, forward_batch, attn_metadata, **kwargs
        )

    def attention_end(
        self,
        output: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
    ) -> None:
        """
        Handle attention end event. Called after each attention pass completes.

        Maybe construct and update sparse representations.
        """
        layer_id = layer.layer_id

        # Maybe construct representations
        self.algorithm.construct_representations(
            layer_id=layer_id,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            k_buffer=self.token_to_kv_pool.get_key_buffer(layer_id),
            forward_batch=forward_batch,
        )

        # Maybe update representations
        self.algorithm.update_representations(
            layer_id=layer_id,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            k_buffer=self.token_to_kv_pool.get_key_buffer(layer_id),
            forward_batch=forward_batch,
        )

    def _handle_sparse_retrieve(
        self,
        query: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        attn_metadata: Optional[Any],
        **kwargs,
    ) -> Optional[torch.Tensor]:
        req_pool_indices = forward_batch.req_pool_indices
        # Compute Topk
        sparse_mask = self._compute_sparse_mask(req_pool_indices)
        selected_indices, valid_lengths = self.algorithm.retrieve_topk(
            queries=query,
            layer_id=layer.layer_id,
            req_pool_indices=req_pool_indices,
            sparse_mask=sparse_mask,
            forward_batch=forward_batch,
            attn_metadata=attn_metadata,
            **kwargs,
        )

        # Adapt Attention Metadata
        return self.backend_adaptor.adapt_for_attn_metadata(
            selected_indices=selected_indices,
            valid_lengths=valid_lengths,
            sparse_mask=sparse_mask,
            current_metadata=attn_metadata,
            forward_batch=forward_batch,
            req_to_token=self.req_to_token_pool.req_to_token,
            page_size=self.page_size,
            layer_id=layer.layer_id,
        )

    def _compute_sparse_mask(self, req_pool_indices):
        mask = (
            self.states.prompt_lens[req_pool_indices]
            >= self.config.min_sparse_prompt_len
        )

        return mask

    # ---- DSA offload hooks (no-op when _dsa_runtime is None) ----
    #
    # These thin wrappers exist so the module-level functions only talk to one
    # object (the coordinator), and the coordinator routes to DSARuntime.  The
    # dsa_precomputed fast-path check lives exclusively in DSARuntime to avoid
    # duplicated logic.

    def on_disagg_cache_received(self, req, token_to_kv_pool_allocator) -> None:
        """Canonicalize prompt prefix to host after disagg transfer completes."""
        self._call_dsa_runtime(
            "on_disagg_cache_received",
            req,
            token_to_kv_pool_allocator,
        )

    def resolve_selected_indices(self, selected_indices, forward_batch):
        """Detect misses, materialize from host, and remap topk to physical addresses."""
        remapped = self._call_dsa_runtime(
            "remap_selected_indices",
            selected_indices,
            forward_batch,
            default=None,
        )
        return (selected_indices, False) if remapped is None else (remapped, True)

    def get_indexer_metadata(self, page_table_1, forward_batch):
        """Return indexer-aware page tables, potentially from precomputed buffers."""
        return self._call_dsa_runtime(
            "get_indexer_metadata",
            page_table_1,
            forward_batch,
            default=(False, None, None),
        )

    def get_disagg_state_indices(self, req_pool_idx, token_count, fallback_indices):
        """Return stable indexer slot indices for disagg state serialization."""
        return self._call_dsa_runtime(
            "get_disagg_state_indices",
            req_pool_idx,
            token_count,
            fallback_indices,
            default=fallback_indices,
        )

    def init_cuda_graph_state(self, max_bs: int, max_topk: int) -> None:
        """Freeze DSA buffer sizes for cuda graph capture."""
        self._call_dsa_runtime(
            "init_cuda_graph_state",
            max_bs=max_bs,
            max_topk=max_topk,
        )

    def prepare_decode_step(self, forward_batch: "ForwardBatch") -> None:
        """Precompute DSA metadata outside the cuda graph replay path."""
        self._call_dsa_runtime("prepare_decode_step", forward_batch)

    def get_indexer_write_loc(self, forward_batch: "ForwardBatch"):
        """Return the stable indexer write location for the current decode token."""
        return self._call_dsa_runtime(
            "get_indexer_write_loc", forward_batch, default=None
        )

    def get_reserved_token_count(self) -> int:
        """Token count permanently carved out for the sparse cache."""
        return self._call_dsa_runtime("get_reserved_token_count", default=0)

    def get_indexer_available_token_count(self) -> Optional[int]:
        """Available indexer slot capacity, or None when DSA is off."""
        return self._call_dsa_runtime(
            "get_indexer_available_token_count", default=None
        )

    def get_host_available_token_count(self) -> Optional[int]:
        """Available host KV capacity, or None when DSA is off."""
        return self._call_dsa_runtime("get_host_available_token_count", default=None)

    def get_host_kv_pool(self):
        """Return the DSA host KV pool, or None when DSA is off."""
        return self._call_dsa_runtime("get_host_kv_pool", default=None)
