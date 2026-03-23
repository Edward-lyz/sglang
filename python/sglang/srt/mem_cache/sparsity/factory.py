import json
import logging
from typing import Optional, Union

import torch

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import BaseSparseAlgorithm
from sglang.srt.mem_cache.sparsity.algorithms.deepseek_nsa import DeepSeekNSAAlgorithm
from sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm import QuestAlgorithm
from sglang.srt.mem_cache.sparsity.backend.backend_adaptor import (
    FlashAttentionAdaptor,
    NSABackendAdaptor,
)
from sglang.srt.mem_cache.sparsity.core.sparse_coordinator import (
    SparseConfig,
    SparseCoordinator,
    clear_sparse_coordinator,
    register_sparse_coordinator,
)
from sglang.srt.server_args import parse_sparse_extra_config

logger = logging.getLogger(__name__)

_ALGORITHM_REGISTRY = {
    "quest": lambda config, device, **kw: QuestAlgorithm(config, device, **kw),
    "deepseek_nsa": lambda config, device, **kw: DeepSeekNSAAlgorithm(
        config, device, **kw
    ),
}


def _get_device_type(device: Union[str, torch.device]) -> str:
    if isinstance(device, torch.device):
        return device.type
    if isinstance(device, str):
        return device.split(":", 1)[0]
    return getattr(device, "type", str(device))


# ---------------------------------------------------------------------------
# DSA offload runtime for deepseek_nsa decode workers
# ---------------------------------------------------------------------------


def _build_dsa_runtime(model_runner, config: SparseConfig):
    """Build DSARuntime as an internal component of SparseCoordinator."""
    from sglang.srt.mem_cache.memory_pool_host import create_host_kv_pool
    from sglang.srt.mem_cache.sparsity.core.sparse_cache_allocator import (
        SparseCacheAllocator,
    )
    from sglang.srt.mem_cache.sparsity.kernel import DSAGraphWorkspace
    from sglang.srt.mem_cache.sparsity.runtime.dsa_runtime import DSARuntime

    page_size = model_runner.page_size
    device = model_runner.device
    server_args = model_runner.server_args
    assert server_args.disaggregation_mode == "decode", (
        "deepseek_nsa offload is only supported on decode disaggregation workers"
    )
    assert not server_args.disable_cuda_graph, (
        "deepseek_nsa offload requires cuda graph to remain enabled"
    )
    assert page_size > 1, "deepseek_nsa offload requires a paged KV allocator"

    cache_ratio = float(config.sparse_extra_config.get("cache_ratio", 0.4))
    assert 0.0 < cache_ratio < 1.0, f"DSA cache_ratio must be in (0, 1), got {cache_ratio}"
    sparse_cache = SparseCacheAllocator.create_from_allocator(
        allocator=model_runner.token_to_kv_pool_allocator,
        cache_ratio=cache_ratio,
        page_size=page_size,
        device=device,
    )

    host_kv_pool = create_host_kv_pool(
        model_runner.token_to_kv_pool,
        host_to_device_ratio=max(getattr(server_args, "hicache_ratio", 2.0), 1.1),
        host_size=getattr(server_args, "hicache_size", 0),
        page_size=page_size,
        layout="layer_first",
        override_kv_cache_dim=getattr(model_runner.token_to_kv_pool, "kv_cache_dim", None),
    )

    workspace = DSAGraphWorkspace.create(device=device)

    runtime = DSARuntime.create(
        req_to_token_pool=model_runner.req_to_token_pool,
        main_allocator=model_runner.token_to_kv_pool_allocator,
        sparse_cache_allocator=sparse_cache,
        host_kv_pool=host_kv_pool,
        kv_pool=model_runner.token_to_kv_pool,
        workspace=workspace,
        page_size=page_size,
        start_layer=model_runner.start_layer,
        end_layer=model_runner.end_layer,
        device=device,
    )

    logger.info(
        "DSA offload runtime created inside coordinator: "
        "cache_ratio=%.2f, reserved_pages=%d",
        cache_ratio,
        sparse_cache.capacity,
    )
    return runtime


# ---------------------------------------------------------------------------
# Sparse coordinator factory
# ---------------------------------------------------------------------------


def _create_sparse_algorithm(
    config: SparseConfig,
    device: Union[str, torch.device],
    **kwargs,
) -> BaseSparseAlgorithm:
    algorithm_name = config.algorithm.lower()
    factory = _ALGORITHM_REGISTRY.get(algorithm_name)
    if factory is None:
        raise ValueError(f"Unknown sparse algorithm: {algorithm_name}")
    return factory(config, device, **kwargs)


def _create_backend_adaptor(
    backend: str,
    device: Union[str, torch.device],
    sparse_algorithm: BaseSparseAlgorithm,
    req_to_token_pool,
):
    if isinstance(sparse_algorithm, DeepSeekNSAAlgorithm):
        return NSABackendAdaptor(device, req_to_token_pool)
    if backend in ["fa3", "flashattention"]:
        return FlashAttentionAdaptor(device)
    raise ValueError(f"Unknown attention backend: {backend}")


def _parse_sparse_config(server_args) -> SparseConfig:
    """Parse hierarchical sparse config from JSON string.

    Required fields with defaults: top_k (2048), device_buffer_size (2*top_k),
    host_to_device_ratio (2).
    Optional fields (default None): algorithm, backend, min_sparse_prompt_len,
    page_size. All remaining fields go to sparse_extra_config.
    """
    extra_config_str = server_args.hisparse_config
    if extra_config_str is not None:
        try:
            extra_config = json.loads(extra_config_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse hisparse_config: {e}") from e
    else:
        extra_config = {}

    top_k = extra_config.pop("top_k", 2048)
    device_buffer_size = extra_config.pop("device_buffer_size", 2 * top_k)
    host_to_device_ratio = extra_config.pop("host_to_device_ratio", 2)

    if device_buffer_size < top_k:
        raise ValueError(
            f"device_buffer_size ({device_buffer_size}) must be no smaller than top_k ({top_k})"
        )

    algorithm = extra_config.pop("algorithm", None)
    backend = extra_config.pop("backend", None)
    min_sparse_prompt_len = extra_config.pop("min_sparse_prompt_len", None)
    page_size = extra_config.pop("page_size", None)

    return SparseConfig(
        top_k=top_k,
        device_buffer_size=device_buffer_size,
        host_to_device_ratio=host_to_device_ratio,
        algorithm=algorithm,
        backend=backend,
        page_size=page_size,
        min_sparse_prompt_len=min_sparse_prompt_len,
        sparse_extra_config=sparse_extra_config,
    )


def parse_hisparse_config(server_args) -> SparseConfig:
    """Parse hisparse config from server_args, returning defaults if no config provided."""
    return _parse_sparse_config(server_args)


def _is_dsa_offload_config(config: SparseConfig, server_args) -> bool:
    return (
        config.algorithm.lower() == "deepseek_nsa"
        and server_args.disaggregation_mode == "decode"
    )


def create_sparse_coordinator(
    device: Union[str, torch.device],
    req_to_token_pool,
    token_to_kv_pool,
    start_layer: int,
    end_layer: int,
    server_args,
    model_runner=None,
    **kwargs,
) -> SparseCoordinator:
    config = _parse_sparse_config(server_args)

    # DSA offload is enabled by using deepseek_nsa on decode disaggregation workers.
    dsa_runtime = None
    if _is_dsa_offload_config(config, server_args):
        assert model_runner is not None, "deepseek_nsa offload requires model_runner"
        assert _get_device_type(device) == "cuda", (
            "deepseek_nsa offload only supports CUDA workers"
        )
        dsa_runtime = _build_dsa_runtime(model_runner, config)
        logger.info(
            "Enable deepseek_nsa offload on the decode worker with graph-external "
            "metadata precompute and graph-only decode replay."
        )
        config.is_dsa_offload = True

    algorithm = _create_sparse_algorithm(config, device, **kwargs)
    backend_adaptor = _create_backend_adaptor(
        config.backend, device, algorithm, req_to_token_pool
    )

    coordinator = SparseCoordinator(
        config=config,
        algorithm=algorithm,
        backend_adaptor=backend_adaptor,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool=token_to_kv_pool,
        start_layer=start_layer,
        end_layer=end_layer,
        device=device,
        dsa_runtime=dsa_runtime,
    )
    register_sparse_coordinator(coordinator)
    return coordinator


def maybe_create_sparse_coordinator(model_runner) -> Optional[SparseCoordinator]:
    server_args = model_runner.server_args
    if server_args.hierarchical_sparse_attention_extra_config is None:
        clear_sparse_coordinator()
        return None
    return create_sparse_coordinator(
        device=model_runner.device,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool=model_runner.token_to_kv_pool,
        start_layer=model_runner.start_layer,
        end_layer=model_runner.end_layer,
        server_args=server_args,
        model_runner=model_runner,
    )
