"""Checkpoint endpoint bindings for AttentionStore cache plans."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import torch

from .page_binding import (
    _create_mha_page_endpoint,
    _create_mla_page_endpoint,
    _key_buffer,
    _uniform_page_bytes,
    _value_buffer,
)
from .schema import pad_component_major_pointers, pad_pointer_buffers
from .state_layout import PackedStateLayout, StateTensorSpec
from .topology import attention_store_kda_namespaces
from .transport import (
    CheckpointedPageStore,
    _BufferPageIO,
    _once,
    _PassiveStore,
    _sdk_pod_config,
    _standard_namespace,
)
from .types import (
    AttentionStoreBinding,
    AttentionStoreEndpointBinding,
    AttentionStoreEndpointRole,
    AttentionStoreInitContext,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PackedStateSnapshot:
    """Own one immutable packed SPA checkpoint staging row."""

    tensor: torch.Tensor
    release: Callable[[], None]
    ready_event: Optional[Any] = None


class KDACheckpointStore:
    """Bind fixed KDA state buffers to the SDK pointer-array transport."""

    def __init__(
        self,
        sdk,
        binding,
        slot_group: str,
        layer_ids: Sequence[int],
    ):
        """Initialize a KDA checkpoint transport over registered state buffers."""
        self._sdk = sdk
        self.layout = binding
        self._state_ptrs = binding.state_ptrs
        self._layer_ids = tuple(map(int, layer_ids))
        if len(self._layer_ids) != binding.layer_num:
            raise ValueError("KDA pointer and schema layer counts differ")
        self._device = binding.device
        self._slot_group = slot_group
        self._pack_validator = binding.pack_validator
        self._unpack_hook = binding.unpack_hook
        # Event waits must happen before acquiring this SDK mutex.
        self._sdk_lock = threading.Lock()

    @staticmethod
    def _wait_for_event(event, timeout: Optional[float], operation: str) -> None:
        """Poll a device event with a deadline."""
        if event is None:
            return
        query = getattr(event, "query", None)
        if query is None:
            raise RuntimeError(f"{operation} event does not support bounded polling")
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while not query():
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"{operation} completion wait timed out")
            time.sleep(0.001)

    def wait_for_producer(self, producer_ready, timeout: Optional[float] = 1.0) -> None:
        """Fence registered buffers before an SDK write."""
        if producer_ready is None and self._device.type in ("cuda", "xpu"):
            raise RuntimeError("KDA checkpoint write requires a producer event")
        self._wait_for_event(producer_ready, timeout, "KDA checkpoint write")

    def _acquire_sdk_lock(self, timeout: Optional[float], operation: str) -> None:
        """Acquire the SDK mutex within the remaining operation deadline."""
        started = time.monotonic()
        if timeout is None:
            acquired = self._sdk_lock.acquire()
        elif timeout <= 0:
            acquired = self._sdk_lock.acquire(blocking=False)
        else:
            acquired = self._sdk_lock.acquire(timeout=timeout)
        if not acquired:
            raise TimeoutError(f"{operation} SDK lock wait timed out")
        waited_ms = (time.monotonic() - started) * 1000
        if waited_ms >= 10:
            logger.warning(
                "KDA checkpoint SDK lock wait: operation=%s wait_ms=%.2f",
                operation,
                waited_ms,
            )

    def _slot(self, slot: Mapping[str, int] | int) -> tuple[dict[str, int], int]:
        slot_map = slot if isinstance(slot, Mapping) else {self._slot_group: int(slot)}
        if set(slot_map) != {self._slot_group}:
            raise ValueError(
                f"KDA checkpoint requires slot group {self._slot_group!r}, "
                f"got {sorted(slot_map)}"
            )
        return dict(slot_map), int(slot_map[self._slot_group])

    def match(self, block_hash_ids: Sequence[int], timeout: float = 1.0) -> int:
        """Return the matched KDA checkpoint prefix length."""
        from attentionstore_sdk.sdk import MatchRequest

        self._acquire_sdk_lock(timeout, "match")
        try:
            return self._sdk.match(MatchRequest(block_hash_ids, timeout=timeout))
        finally:
            self._sdk_lock.release()

    def write(
        self,
        block_hash_id: int,
        slot: Mapping[str, int] | int,
        timeout: float = 1.0,
        params: Optional[dict] = None,
        producer_ready=None,
    ) -> int:
        """Write one KDA checkpoint from a logical state slot."""
        from attentionstore_sdk.sdk import BlockBatch, BufferBindings, WriteRequest

        slot_map, slot_id = self._slot(slot)
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        self.wait_for_producer(producer_ready, remaining)
        if self._pack_validator is not None:
            self._pack_validator([slot_map])
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        request = WriteRequest(
            blocks=BlockBatch(
                [block_hash_id],
                [slot_id],
                layer_ids=self._layer_ids,
            ),
            buffers=BufferBindings(state=self._state_ptrs),
            timeout=(
                timeout if deadline is None else max(0.0, deadline - time.monotonic())
            ),
            params=params or {},
        )
        self._acquire_sdk_lock(remaining, "write")
        # Snapshot before the SDK call: this is the state content the Agent gets.
        try:
            written = self._sdk.write(request)
        finally:
            self._sdk_lock.release()
        return written

    def read(
        self,
        block_hash_id: int,
        slot: Mapping[str, int] | int,
        timeout: float = 1.0,
        remote_addrs=None,
    ) -> int:
        """Read one KDA checkpoint into a logical state slot."""
        from attentionstore_sdk.sdk import BlockBatch, BufferBindings, ReadRequest

        slot_map, slot_id = self._slot(slot)
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        request = ReadRequest(
            blocks=BlockBatch(
                [block_hash_id],
                [slot_id],
                layer_ids=self._layer_ids,
            ),
            buffers=BufferBindings(state=self._state_ptrs),
            timeout=(
                timeout if deadline is None else max(0.0, deadline - time.monotonic())
            ),
            remote_addrs=remote_addrs or (),
        )
        self._acquire_sdk_lock(
            None if deadline is None else max(0.0, deadline - time.monotonic()),
            "read",
        )
        try:
            count = self._sdk.read(request)
            if count and self._unpack_hook is not None:
                self._unpack_hook([slot_map])
        finally:
            self._sdk_lock.release()
        return count

    def stop(self) -> None:
        """Stop the underlying checkpoint SDK client."""
        self._sdk.stop()


@dataclass(frozen=True)
class _KDACheckpointBinding:
    layer_num: int
    temporal_bytes: tuple[int, ...]
    conv_bytes: tuple[int, ...]
    state_ptrs: tuple[int, ...]
    tensors: tuple[torch.Tensor, ...]
    payload_fields: tuple[str, ...]
    device: torch.device
    pack_validator: Optional[Callable] = None
    unpack_hook: Optional[Callable] = None

    @property
    def bytes_per_layer(self) -> int:
        """Return the checkpoint byte size for one KDA layer."""
        return sum(self.temporal_bytes) + sum(self.conv_bytes)

    @property
    def bytes_per_block(self) -> int:
        """Return the checkpoint byte size across all KDA layers."""
        return self.layer_num * self.bytes_per_layer


def _kda_checkpoint_tensors(checkpoint_pool):
    temporal = [
        ("temporal.qdata", checkpoint_pool.temporal.qdata),
        ("temporal.scale", checkpoint_pool.temporal.scale),
    ]
    conv = [
        (f"conv.{index}", tensor) for index, tensor in enumerate(checkpoint_pool.conv)
    ]
    return temporal, conv


def _make_kda_checkpoint_binding(
    temporal_tensors,
    conv_tensors,
    *,
    pack_validator=None,
    unpack_hook=None,
) -> _KDACheckpointBinding:
    """Bind per-layer KDA components as component-major pointer arrays."""
    temporal_tensors = tuple(temporal_tensors)
    conv_tensors = tuple(conv_tensors)
    if not temporal_tensors:
        raise ValueError("KDA AttentionStore requires temporal state buffers")
    if not conv_tensors:
        raise ValueError("KDA AttentionStore requires conv state buffers")
    named_tensors = temporal_tensors + conv_tensors

    names = tuple(name for name, _ in named_tensors)
    tensors = tuple(tensor for _, tensor in named_tensors)
    if len(set(names)) != len(names):
        raise ValueError(f"KDA state buffer names must be unique, got {names}")

    first = tensors[0]
    if first.ndim < 2:
        raise ValueError("KDA state buffers must have [layer, slot, ...] layout")
    layer_num, slot_num = first.shape[:2]
    if layer_num <= 0 or slot_num <= 0:
        raise ValueError("KDA state buffers require positive layer and slot counts")

    component_bytes = []
    state_ptrs = []
    for name, tensor in named_tensors:
        if tensor.ndim < 2 or tuple(tensor.shape[:2]) != (layer_num, slot_num):
            raise ValueError(
                f"KDA state buffer {name!r} must start with "
                f"[{layer_num}, {slot_num}], got {tuple(tensor.shape)}"
            )
        if tensor.device != first.device:
            raise ValueError(
                f"KDA state buffer {name!r} is on {tensor.device}, "
                f"expected {first.device}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"KDA state buffer {name!r} must be contiguous")

        bytes_per_slot = tensor[0, 0].numel() * tensor.element_size()
        component_bytes.append(bytes_per_slot)
        for layer in range(layer_num):
            layer_buffer = tensor[layer]
            if not layer_buffer.is_contiguous():
                raise ValueError(
                    f"KDA state buffer {name!r} layer {layer} must be contiguous"
                )
            state_ptrs.append(layer_buffer.data_ptr())

    temporal_count = len(temporal_tensors)
    return _KDACheckpointBinding(
        layer_num=layer_num,
        temporal_bytes=tuple(component_bytes[:temporal_count]),
        conv_bytes=tuple(component_bytes[temporal_count:]),
        state_ptrs=tuple(state_ptrs),
        tensors=tensors,
        payload_fields=names,
        device=first.device,
        pack_validator=pack_validator,
        unpack_hook=unpack_hook,
    )


def make_kda_checkpoint_layout(checkpoint_pool) -> _KDACheckpointBinding:
    """Bind the fixed recurrent-state buffers of an int8 KDA model."""
    temporal, conv = _kda_checkpoint_tensors(checkpoint_pool)
    return _make_kda_checkpoint_binding(temporal, conv)


def _kda_active_state_parts(mamba_pool):
    """Build full-precision KDA state with ReplaySSM boundary checks."""
    cache = mamba_pool.mamba_cache
    temporal_tensors = [("temporal", cache.temporal)]
    conv_tensors = list(
        (f"conv.{index}", tensor) for index, tensor in enumerate(cache.conv)
    )
    write_pos = getattr(mamba_pool, "replayssm_write_pos", None)

    def checkpoint_indices(slots: Sequence[Mapping[str, int]]) -> torch.Tensor:
        return torch.tensor(
            [int(slot["checkpoint"]) for slot in slots],
            dtype=torch.long,
            device=write_pos.device,
        )

    def require_flushed_replayssm(slots: Sequence[Mapping[str, int]]) -> None:
        if write_pos is None or not slots:
            return
        indices = checkpoint_indices(slots)
        pending = write_pos[indices]
        if torch.any(pending != 0).item():
            raise ValueError(
                "KDA AttentionStore checkpoints require fully flushed ReplaySSM "
                f"slots; write_pos={pending.tolist()} for slots={indices.tolist()}"
            )

    def reset_replayssm_cursor(slots: Sequence[Mapping[str, int]]) -> None:
        if write_pos is None or not slots:
            return
        write_pos[checkpoint_indices(slots)] = 0

    return (
        temporal_tensors,
        conv_tensors,
        require_flushed_replayssm,
        reset_replayssm_cursor,
    )


def make_kda_active_state_layout(mamba_pool) -> _KDACheckpointBinding:
    """Bind the fixed recurrent-state buffers of a full-precision KDA model."""
    temporal_tensors, conv_tensors, validator, hook = _kda_active_state_parts(
        mamba_pool
    )
    return _make_kda_checkpoint_binding(
        temporal_tensors,
        conv_tensors,
        pack_validator=validator,
        unpack_hook=hook,
    )


def _split_kda_namespace_chunks(total_chunks: int) -> tuple[int, int]:
    """Partition Agent chunks between KDA page and checkpoint namespaces."""
    if total_chunks < 2:
        raise ValueError(
            "KDA page/state namespaces require at least two Agent memory chunks"
        )
    raw_checkpoint_chunks = os.getenv("AS_KDA_CHECKPOINT_GPU_COUNT", "1")
    try:
        checkpoint_chunks = int(raw_checkpoint_chunks)
    except ValueError:
        raise ValueError(
            "AS_KDA_CHECKPOINT_GPU_COUNT must be an integer, "
            f"got {raw_checkpoint_chunks!r}"
        ) from None
    if not 1 <= checkpoint_chunks < total_chunks:
        raise ValueError(
            "AS_KDA_CHECKPOINT_GPU_COUNT must be in "
            f"[1, {total_chunks - 1}], got {checkpoint_chunks}"
        )
    return total_chunks - checkpoint_chunks, checkpoint_chunks


def describe_kda_local_endpoint_bytes(
    kv_cache,
    req_to_token_pool,
    page_size: int,
    *,
    page_active: bool,
    state_active: bool,
) -> tuple[int, int]:
    """Return local per-layer byte sizes before any SDK registration."""
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

    if not isinstance(kv_cache, HybridLinearKVPool):
        raise ValueError("KDA endpoint sizing requires HybridLinearKVPool")

    page_bytes_per_layer = 0
    if page_active:
        pool = kv_cache.full_kv_pool
        local_layer_ids = tuple(range(int(pool.layer_num)))
        if not local_layer_ids:
            raise ValueError("active KDA page endpoint has no local layers")
        key_buffers = [_key_buffer(pool, layer_id) for layer_id in local_layer_ids]
        key_bytes = _uniform_page_bytes(
            key_buffers,
            [pool] * len(key_buffers),
            page_size,
            "KDA page K",
        )
        if kv_cache.use_mla:
            page_bytes_per_layer = key_bytes
        else:
            value_buffers = [
                _value_buffer(pool, layer_id) for layer_id in local_layer_ids
            ]
            value_bytes = _uniform_page_bytes(
                value_buffers,
                [pool] * len(value_buffers),
                page_size,
                "KDA page V",
            )
            page_bytes_per_layer = key_bytes + value_bytes

    state_bytes_per_layer = 0
    if state_active:
        checkpoint_pool = getattr(req_to_token_pool, "mamba_ckpt_pool", None)
        state_binding = (
            make_kda_checkpoint_layout(checkpoint_pool)
            if checkpoint_pool is not None
            else make_kda_active_state_layout(kv_cache.mamba_pool)
        )
        state_bytes_per_layer = state_binding.bytes_per_layer

    return page_bytes_per_layer, state_bytes_per_layer


def create_kda_binding(ctx: AttentionStoreInitContext) -> AttentionStoreBinding:
    """Create independent KDA page and recurrent-checkpoint stores."""
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

    if not isinstance(ctx.kv_cache, HybridLinearKVPool):
        raise ValueError("KDA AttentionStore requires HybridLinearKVPool")
    page_active = ctx.page_shard_num > 0
    state_active = ctx.state_shard_num > 0
    if not page_active and not state_active:
        raise ValueError("KDA AttentionStore node role has no page or state layers")

    page_namespace, checkpoint_namespace = attention_store_kda_namespaces(
        ctx.namespace_base, ctx.node_pp_start
    )
    if page_active and state_active:
        page_chunks, checkpoint_chunks = _split_kda_namespace_chunks(ctx.gpu_count)
    elif page_active:
        page_chunks, checkpoint_chunks = ctx.gpu_count, 0
    else:
        page_chunks, checkpoint_chunks = 0, ctx.gpu_count
    logger.info(
        "KDA AttentionStore Agent chunk split: total=%d page=%d checkpoint=%d",
        ctx.gpu_count,
        page_chunks,
        checkpoint_chunks,
    )
    page_factory = (
        _create_mla_page_endpoint if ctx.kv_cache.use_mla else _create_mha_page_endpoint
    )
    if ctx.page_shard_id >= 0:
        page_endpoint = page_factory(
            ctx,
            pool=ctx.kv_cache.full_kv_pool,
            namespace=page_namespace,
            gpu_count=page_chunks,
            registration_endpoint_name="kda-page",
        )
    else:
        page_endpoint = AttentionStoreEndpointBinding(
            name="mla" if ctx.kv_cache.use_mla else "mha",
            registration_name="kda-page",
            role=AttentionStoreEndpointRole.PAGE,
            store=_PassiveStore(),
            namespace=page_namespace,
            layout_id="kda-page-passive",
            bytes_per_block=max(ctx.page_layer_num, 1),
            payload_fields=("kv_latent",) if ctx.kv_cache.use_mla else ("k", "v"),
        )

    checkpoint_store = None
    try:
        if ctx.state_shard_id >= 0:
            from attentionstore_sdk.sdk import AttentionStoreSDK, KDAConfig

            checkpoint_pool = getattr(ctx.req_to_token_pool, "mamba_ckpt_pool", None)
            checkpoint_binding = (
                make_kda_checkpoint_layout(checkpoint_pool)
                if checkpoint_pool is not None
                else make_kda_active_state_layout(ctx.kv_cache.mamba_pool)
            )
            if checkpoint_binding.layer_num != len(ctx.state_layer_ids):
                raise ValueError("KDA state manifest and local buffer counts differ")
            is_kda_single_shard = "kda-state" in ctx.endpoint_registrations
            state_layer_ids = (
                tuple(ctx.state_layer_ids)
                if is_kda_single_shard
                else tuple(range(ctx.state_layer_num))
            )
            state_layer_num = len(state_layer_ids)
            # The request carries this client's compact layer IDs and must
            # carry exactly this client's component-major pointers.  Padding
            # with the last pointer would alias another PP client's layer and
            # also make the SDK's layer/pointer count inconsistent.
            checkpoint_sdk = AttentionStoreSDK(
                _sdk_pod_config(
                    ctx,
                    checkpoint_namespace,
                    endpoint_name="kda-state",
                    gpu_count=checkpoint_chunks,
                ),
                KDAConfig(
                    # The Agent namespace is shared by all PP clients on the
                    # node.  Register its complete layer geometry, while the
                    # checkpoint store retains only this client's pointers.
                    layer_num=ctx.state_layer_num if is_kda_single_shard else state_layer_num,
                    shard_id=0 if is_kda_single_shard else ctx.state_shard_id,
                    shard_num=1 if is_kda_single_shard else ctx.state_shard_num,
                    block_token_size=ctx.block_size,
                    temporal_bytes=checkpoint_binding.temporal_bytes,
                    conv_bytes=checkpoint_binding.conv_bytes,
                ),
                enable_as_kv_rw=True,
            )
            checkpoint_store = KDACheckpointStore(
                checkpoint_sdk,
                checkpoint_binding,
                slot_group="checkpoint",
                layer_ids=state_layer_ids,
            )
            checkpoint_layout_id = "kda-state"
            checkpoint_bytes = (
                checkpoint_binding.bytes_per_layer * ctx.state_layer_num
            )
            checkpoint_fields = checkpoint_binding.payload_fields
        else:
            checkpoint_store = _PassiveStore()
            checkpoint_layout_id = "kda-state-passive"
            checkpoint_bytes = max(ctx.state_layer_num, 1)
            checkpoint_fields = ("state",)
    except Exception:
        page_endpoint.store.stop()
        raise
    cache_store = CheckpointedPageStore(
        page_endpoint.store,
        checkpoint_store,
        page_match_is_joint=(
            "kda-page" in ctx.endpoint_registrations
            and "kda-state" in ctx.endpoint_registrations
        ),
    )
    try:
        checkpoint_endpoint = AttentionStoreEndpointBinding(
            name="checkpoint",
            registration_name="kda-state",
            role=AttentionStoreEndpointRole.CHECKPOINT,
            store=checkpoint_store,
            namespace=checkpoint_namespace,
            layout_id=checkpoint_layout_id,
            bytes_per_block=checkpoint_bytes,
            payload_fields=checkpoint_fields,
        )
        return AttentionStoreBinding(
            plan_name="KDA",
            cache_store=cache_store,
            endpoints=(page_endpoint, checkpoint_endpoint),
            endpoint_registrations=ctx.endpoint_registrations,
        )
    except Exception:
        cache_store.stop()
        raise


def make_deepseek_v4_checkpoint_layout(
    pool, *, include_compressed_pages: bool = True
) -> PackedStateLayout:
    """Describe DSV4 KV, SWA, C4, and C128 checkpoint groups."""
    specs: list[StateTensorSpec] = []

    def add_buffers(
        prefix: str,
        buffers: Sequence[torch.Tensor],
        *,
        group: str,
        items_per_slot: int = 1,
        base_index: int = 0,
        slot_count_limit: Optional[int] = None,
    ) -> None:
        specs.extend(
            StateTensorSpec(
                f"{prefix}.{index}",
                tensor,
                group=group,
                slot_dim=0,
                items_per_slot=items_per_slot,
                base_index=base_index,
                slot_count_limit=slot_count_limit,
            )
            for index, tensor in enumerate(buffers)
        )

    if getattr(pool, "_unified_kv", False):
        unified = pool.unified_kv_pool
        stage_ratios = pool.compression_ratios[pool._stage_start : pool._stage_end]
        if include_compressed_pages:
            for local_layer_id, ratio in enumerate(stage_ratios):
                if ratio not in (4, 128):
                    continue
                add_buffers(
                    f"unified.compressed.{local_layer_id}",
                    [unified.kv_buffer[local_layer_id]],
                    group="kv_page",
                    items_per_slot=pool.page_size // ratio,
                    base_index=unified.swa_pages,
                )
            add_buffers(
                "c4.indexer",
                pool.c4_indexer_kv_pool.index_k_with_scale_buffer,
                group="kv_page",
            )
        add_buffers(
            "unified.swa",
            unified.kv_buffer,
            group="swa_request",
            items_per_slot=unified.swa_ring_size,
            slot_count_limit=unified.num_slots,
        )
    else:
        if include_compressed_pages:
            add_buffers("c4.kv", pool.c4_kv_pool.kv_buffer, group="kv_page")
            add_buffers(
                "c4.indexer",
                pool.c4_indexer_kv_pool.index_k_with_scale_buffer,
                group="kv_page",
            )
            add_buffers("c128.kv", pool.c128_kv_pool.kv_buffer, group="kv_page")
        add_buffers("swa", pool.swa_kv_pool.kv_buffer, group="swa_page")

    for layer_id, state_pool in enumerate(pool.compress_state_pools):
        if state_pool is None:
            continue
        group = "c4_state" if state_pool.ratio == 4 else "c128_state"
        state = state_pool.kv_score_buffer.kv_score
        is_online_c128 = state_pool.ratio == 128 and state_pool.ring_size == 1
        if is_online_c128:
            slot_count = pool.online_c128_state_num_req_slots
            add_buffers(
                f"attention_state.{layer_id}.committed",
                [state],
                group=group,
                slot_count_limit=slot_count,
            )
            for bank in range(state_pool.online_mtp_max_draft_tokens):
                add_buffers(
                    f"attention_state.{layer_id}.draft.{bank}",
                    [state],
                    group=group,
                    base_index=(bank + 1) * state_pool.online_mtp_state_slot_offset,
                    slot_count_limit=slot_count,
                )
        else:
            slot_count_limit = None
            if state_pool.ratio == 4 and hasattr(pool, "c4_state_pool_size"):
                slot_count_limit = pool.c4_state_pool_size // state_pool.ring_size
            elif state_pool.ratio == 128 and hasattr(pool, "c128_state_pool_size"):
                slot_count_limit = pool.c128_state_pool_size // state_pool.ring_size
            add_buffers(
                f"attention_state.{layer_id}",
                [state],
                group=group,
                items_per_slot=state_pool.ring_size,
                slot_count_limit=slot_count_limit,
            )

    for layer_id, state_pool in enumerate(pool.indexer_compress_state_pools):
        if state_pool is None:
            continue
        add_buffers(
            f"indexer_state.{layer_id}",
            [state_pool.kv_score_buffer.kv_score],
            group="c4_state",
            items_per_slot=state_pool.ring_size,
            slot_count_limit=(
                pool.c4_state_pool_size // state_pool.ring_size
                if hasattr(pool, "c4_state_pool_size")
                else None
            ),
        )

    pending_seq_lens = getattr(pool, "online_c128_mtp_pending_seq_lens", None)
    if pending_seq_lens is not None:
        specs.append(
            StateTensorSpec(
                "c128.pending_seq_len",
                pending_seq_lens,
                group="c128_state",
                slot_dim=0,
                slot_count_limit=pool.online_c128_state_num_req_slots,
            )
        )

    return PackedStateLayout(
        specs,
        layout_name=(
            "SPA_CHECKPOINT_WITH_PAGES"
            if include_compressed_pages
            else "SPA_CHECKPOINT"
        ),
    )


class SPACheckpointStore:
    """SPA-owned packed resume state carried by the SPA SWA segment."""

    def __init__(
        self,
        sdk,
        layout: PackedStateLayout,
        storage_bytes_per_block: Optional[int] = None,
        write_snapshot_capacity: int = 1,
    ):
        """Initialize packed SPA staging and snapshot capacity."""
        if write_snapshot_capacity <= 0:
            raise ValueError("SPA write snapshot capacity must be positive")
        self._sdk = sdk
        self.layout = layout
        self.storage_bytes_per_block = int(
            storage_bytes_per_block or layout.bytes_per_block
        )
        if self.storage_bytes_per_block < layout.bytes_per_block:
            raise ValueError(
                "SPA checkpoint storage width is smaller than its packed layout: "
                f"storage={self.storage_bytes_per_block}, "
                f"layout={layout.bytes_per_block}"
            )
        # The second row matches the SDK's SPA read offset.
        self.staging = self._allocate_staging(2)
        self._lock = threading.Lock()
        self._read_lock = threading.Lock()
        self._read_ready_event = None
        self._snapshot_condition = threading.Condition()
        self._available_write_snapshots = [
            self._allocate_staging(1) for _ in range(write_snapshot_capacity)
        ]
        self._pending_write_snapshots = []
        self._stopped = False

    def _allocate_staging(self, capacity: int) -> torch.Tensor:
        return torch.empty(
            (capacity, self.storage_bytes_per_block),
            dtype=torch.uint8,
            device=self.layout.device,
        )

    def _layout_view(self, staging: torch.Tensor) -> torch.Tensor:
        return staging[:, : self.layout.bytes_per_block]

    def _record_stream_event(self):
        device = self.layout.device
        if device.type not in ("cuda", "xpu"):
            return None
        device_module = torch.get_device_module(device)
        event = device_module.Event()
        event.record(device_module.current_stream(device))
        return event

    @staticmethod
    def _acquire_lock(lock, timeout: Optional[float], operation: str) -> None:
        if timeout is None:
            acquired = lock.acquire()
        elif timeout <= 0:
            acquired = lock.acquire(blocking=False)
        else:
            acquired = lock.acquire(timeout=timeout)
        if not acquired:
            raise TimeoutError(f"{operation} lock wait timed out")

    def _reap_write_snapshots_locked(self) -> None:
        pending = []
        for tensor, event in self._pending_write_snapshots:
            query = getattr(event, "query", None)
            if query is not None and query():
                self._available_write_snapshots.append(tensor)
            else:
                pending.append((tensor, event))
        self._pending_write_snapshots = pending

    def match(self, block_hash_ids: Sequence[int], timeout: float = 1.0) -> int:
        """Return the matched SPA checkpoint prefix length."""
        from attentionstore_sdk.sdk import MatchRequest

        self._acquire_lock(self._lock, timeout, "SPA match")
        try:
            return self._sdk.match(MatchRequest(block_hash_ids, timeout=timeout))
        finally:
            self._lock.release()

    def write(
        self,
        block_hash_id: int,
        slots: Mapping[str, int],
        timeout: float = 1.0,
        params: Optional[dict] = None,
    ) -> int:
        """Snapshot and write one SPA checkpoint."""
        snapshot = self.snapshot(slots, timeout=timeout)
        try:
            return self.write_snapshot(
                block_hash_id,
                snapshot,
                timeout=timeout,
                params=params,
            )
        finally:
            snapshot.release()

    def snapshot(
        self, slots: Mapping[str, int], timeout: float = 1.0
    ) -> PackedStateSnapshot:
        """Pack state into an exclusively leased checkpoint snapshot."""
        deadline = time.monotonic() + max(0.0, timeout)
        with self._snapshot_condition:
            while True:
                if self._stopped:
                    raise RuntimeError("SPA checkpoint store is already stopped")
                self._reap_write_snapshots_locked()
                if self._available_write_snapshots:
                    tensor = self._available_write_snapshots.pop()
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("SPA snapshot lease wait timed out")
                self._snapshot_condition.wait(timeout=min(remaining, 0.01))

        ready_event = None

        def return_snapshot() -> None:
            with self._snapshot_condition:
                query = getattr(ready_event, "query", None)
                if ready_event is not None and (query is None or not query()):
                    self._pending_write_snapshots.append((tensor, ready_event))
                else:
                    self._available_write_snapshots.append(tensor)
                    self._snapshot_condition.notify()

        release = _once(return_snapshot)
        try:
            tensor.zero_()
            self.layout.pack([slots], self._layout_view(tensor))
            ready_event = self._record_stream_event()
            return PackedStateSnapshot(tensor, release, ready_event)
        except Exception:
            ready_event = self._record_stream_event()
            release()
            raise

    def write_snapshot(
        self,
        block_hash_id: int,
        snapshot: PackedStateSnapshot,
        timeout: float = 1.0,
        params: Optional[dict] = None,
    ) -> int:
        """Write a previously packed SPA checkpoint snapshot."""
        from attentionstore_sdk.sdk import (
            BlockBatch,
            BufferBindings,
            WriteRequest,
        )

        if snapshot.tensor.shape != (1, self.storage_bytes_per_block):
            raise ValueError(
                f"invalid SPA checkpoint snapshot shape: {tuple(snapshot.tensor.shape)}"
            )
        deadline = time.monotonic() + max(0.0, timeout)
        KDACheckpointStore._wait_for_event(
            snapshot.ready_event,
            max(deadline - time.monotonic(), 0.0),
            "SPA checkpoint write",
        )
        remaining = max(deadline - time.monotonic(), 0.0)
        self._acquire_lock(self._lock, remaining, "SPA write")
        try:
            return self._sdk.write(
                WriteRequest(
                    blocks=BlockBatch(hash_ids=[block_hash_id], data_ids=[0]),
                    buffers=BufferBindings(
                        swa=(snapshot.tensor.data_ptr(),), swa_ids=(0,)
                    ),
                    timeout=max(deadline - time.monotonic(), 0.0),
                    params=params or {},
                    swa_block_step=1,
                )
            )
        finally:
            self._lock.release()

    def read(
        self,
        block_hash_id: int,
        slots: Mapping[str, int],
        timeout: float = 1.0,
        remote_addrs=None,
    ) -> int:
        """Read and unpack one SPA checkpoint into logical slots."""
        from attentionstore_sdk.sdk import (
            BlockBatch,
            BufferBindings,
            ReadRequest,
        )

        deadline = time.monotonic() + max(0.0, timeout)
        self._acquire_lock(self._read_lock, timeout, "SPA read staging")
        try:
            KDACheckpointStore._wait_for_event(
                self._read_ready_event,
                max(deadline - time.monotonic(), 0.0),
                "SPA checkpoint read staging",
            )
            remaining = max(deadline - time.monotonic(), 0.0)
            self._acquire_lock(self._lock, remaining, "SPA read")
            try:
                count = self._sdk.read(
                    ReadRequest(
                        blocks=BlockBatch(hash_ids=[block_hash_id], data_ids=[0]),
                        buffers=BufferBindings(
                            swa=(self.staging.data_ptr(),), swa_ids=(0,)
                        ),
                        timeout=max(deadline - time.monotonic(), 0.0),
                        remote_addrs=remote_addrs or (),
                        swa_block_step=1,
                    )
                )
            finally:
                self._lock.release()
            if count == 1:
                try:
                    self.layout.unpack([slots], self._layout_view(self.staging[1:2]))
                finally:
                    self._read_ready_event = self._record_stream_event()
            return count
        finally:
            self._read_lock.release()

    def stop(self) -> None:
        """Wake snapshot waiters and stop the underlying SDK client."""
        with self._snapshot_condition:
            if self._stopped:
                return
            self._stopped = True
            self._snapshot_condition.notify_all()
        self._sdk.stop()


def _uniform_nonzero(values: Sequence[int], name: str) -> int:
    if not values:
        return 0
    unique = set(map(int, values))
    if len(unique) != 1 or next(iter(unique)) <= 0:
        raise ValueError(f"SPA {name} block sizes are invalid: {sorted(unique)}")
    return unique.pop()


@dataclass(frozen=True)
class _SPAPageGeometry:
    c4_count: int
    c128_count: int
    c4_kv_ptrs: tuple[int, ...]
    c4_indexer_ptrs: tuple[int, ...]
    c128_ptrs: tuple[int, ...]
    c4_kv_bytes: int
    c4_indexer_bytes: int
    c128_bytes: int
    c4_kv_storage_bytes: int
    c4_indexer_storage_bytes: int
    c128_storage_bytes: int
    device: torch.device


def _spa_page_geometry(pool) -> _SPAPageGeometry:
    ratios = pool.compression_ratios[pool._stage_start : pool._stage_end]
    c4_count = sum(ratio == 4 for ratio in ratios)
    c128_count = sum(ratio == 128 for ratio in ratios)
    ptrs, data_lens, item_lens = pool.get_contiguous_buf_infos()
    expected = c4_count * 2 + c128_count
    if (
        len(ptrs) != expected
        or len(data_lens) != expected
        or len(item_lens) != expected
    ):
        raise ValueError(
            "SPA contiguous buffer order mismatch: "
            f"expected={expected}, ptrs={len(ptrs)}, "
            f"buffers={len(data_lens)}, items={len(item_lens)}"
        )
    c4_kv_end = c4_count
    c4_indexer_end = c4_kv_end + c4_count
    return _SPAPageGeometry(
        c4_count=c4_count,
        c128_count=c128_count,
        c4_kv_ptrs=tuple(ptrs[:c4_kv_end]),
        c4_indexer_ptrs=tuple(ptrs[c4_kv_end:c4_indexer_end]),
        c128_ptrs=tuple(ptrs[c4_indexer_end:]),
        c4_kv_bytes=_uniform_nonzero(item_lens[:c4_kv_end], "c4 KV"),
        c4_indexer_bytes=_uniform_nonzero(
            item_lens[c4_kv_end:c4_indexer_end], "c4 indexer"
        ),
        c128_bytes=_uniform_nonzero(item_lens[c4_indexer_end:], "c128 KV"),
        c4_kv_storage_bytes=_uniform_nonzero(data_lens[:c4_kv_end], "c4 KV storage"),
        c4_indexer_storage_bytes=_uniform_nonzero(
            data_lens[c4_kv_end:c4_indexer_end], "c4 indexer storage"
        ),
        c128_storage_bytes=_uniform_nonzero(
            data_lens[c4_indexer_end:], "c128 KV storage"
        ),
        device=torch.device(pool.device),
    )


def describe_spa_endpoint_schema(pool) -> Mapping[str, int]:
    """Describe the local SPA payload before node-local PP schema merging."""
    geometry = _spa_page_geometry(pool)
    checkpoint_layout = make_deepseek_v4_checkpoint_layout(
        pool, include_compressed_pages=False
    )
    return {
        "c4_layer_count": geometry.c4_count,
        "c128_layer_count": geometry.c128_count,
        "c4_kv_stride": geometry.c4_kv_bytes,
        "c4_indexer_stride": geometry.c4_indexer_bytes,
        "c128_kv_stride": geometry.c128_bytes,
        "c4_kv_storage_bytes": geometry.c4_kv_storage_bytes,
        "c4_indexer_storage_bytes": geometry.c4_indexer_storage_bytes,
        "c128_kv_storage_bytes": geometry.c128_storage_bytes,
        "checkpoint_bytes": checkpoint_layout.bytes_per_block,
    }


def _split_spa_namespace_chunks(total_chunks: int) -> tuple[int, int]:
    if total_chunks < 2:
        raise ValueError(
            "SPA page/checkpoint namespaces require at least two Agent memory chunks"
        )
    raw_checkpoint_chunks = os.getenv("AS_SPA_CHECKPOINT_GPU_COUNT", "1")
    try:
        checkpoint_chunks = int(raw_checkpoint_chunks)
    except ValueError:
        raise ValueError(
            "AS_SPA_CHECKPOINT_GPU_COUNT must be an integer, "
            f"got {raw_checkpoint_chunks!r}"
        ) from None
    if not 1 <= checkpoint_chunks < total_chunks:
        raise ValueError(
            "AS_SPA_CHECKPOINT_GPU_COUNT must be in "
            f"[1, {total_chunks - 1}], got {checkpoint_chunks}"
        )
    return total_chunks - checkpoint_chunks, checkpoint_chunks


def _normalize_spa_pointers(
    pointers: Sequence[int],
    layer_count: int,
    storage_bytes: int,
    device: torch.device,
    name: str,
):
    def allocate_placeholder() -> torch.Tensor:
        if storage_bytes <= 0:
            raise ValueError(f"{name} has no backing-buffer capacity")
        return torch.zeros(storage_bytes, dtype=torch.uint8, device=device)

    return pad_pointer_buffers(
        pointers,
        layer_count,
        name,
        placeholder_factory=allocate_placeholder,
    )


def create_spa_binding(ctx: AttentionStoreInitContext) -> AttentionStoreBinding:
    """Create the DeepSeek-V4 SPA page/checkpoint binding."""
    from attentionstore_sdk.sdk import (
        AttentionStoreSDK,
        BufferBindings,
        SPAConfig,
    )

    from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
        DeepSeekV4TokenToKVPool,
    )

    pool = ctx.kv_cache
    if not isinstance(pool, DeepSeekV4TokenToKVPool):
        raise ValueError("SPA AttentionStore requires DeepSeekV4TokenToKVPool")
    if ctx.page_shard_id < 0 or ctx.page_shard_num <= 0:
        raise ValueError("SPA AttentionStore stage has no page shard")
    geometry = _spa_page_geometry(pool)
    schema = ctx.endpoint_schema
    c4_count = int(schema.get("c4_layer_count", geometry.c4_count))
    c128_count = int(schema.get("c128_layer_count", geometry.c128_count))
    c4_kv_bytes = int(schema.get("c4_kv_stride", geometry.c4_kv_bytes))
    c4_indexer_bytes = int(schema.get("c4_indexer_stride", geometry.c4_indexer_bytes))
    c128_bytes = int(schema.get("c128_kv_stride", geometry.c128_bytes))
    c4_kv_storage_bytes = int(
        schema.get("c4_kv_storage_bytes", geometry.c4_kv_storage_bytes)
    )
    c4_indexer_storage_bytes = int(
        schema.get("c4_indexer_storage_bytes", geometry.c4_indexer_storage_bytes)
    )
    c128_storage_bytes = int(
        schema.get("c128_kv_storage_bytes", geometry.c128_storage_bytes)
    )
    c4_kv = _normalize_spa_pointers(
        geometry.c4_kv_ptrs,
        c4_count,
        c4_kv_storage_bytes,
        geometry.device,
        "SPA c4 KV",
    )
    c4_indexer = _normalize_spa_pointers(
        geometry.c4_indexer_ptrs,
        c4_count,
        c4_indexer_storage_bytes,
        geometry.device,
        "SPA c4 indexer",
    )
    c128 = _normalize_spa_pointers(
        geometry.c128_ptrs,
        c128_count,
        c128_storage_bytes,
        geometry.device,
        "SPA c128 KV",
    )
    page_chunks, checkpoint_chunks = _split_spa_namespace_chunks(ctx.gpu_count)

    page_namespace = _standard_namespace(ctx, "spa-page", ctx.page_shard_num)
    page_sdk = AttentionStoreSDK(
        _sdk_pod_config(ctx, page_namespace, gpu_count=page_chunks),
        SPAConfig(
            block_token_size=ctx.page_size,
            c4_layer_count=c4_count,
            c128_layer_count=c128_count,
            c4_kv_bytes=c4_kv_bytes,
            c4_indexer_bytes=c4_indexer_bytes,
            c128_kv_bytes=c128_bytes,
            swa_read_layer_bytes=0,
            swa_write_layer_bytes=0,
            swa_layer_count=0,
            swa_step=ctx.page_size,
            shard_id=ctx.page_shard_id,
            shard_num=ctx.page_shard_num,
        ),
        enable_as_kv_rw=True,
    )
    page_buffers = BufferBindings(
        c4_kv=c4_kv.pointers,
        c4_indexer=c4_indexer.pointers,
        c128_kv=c128.pointers,
    )
    page_store = _BufferPageIO(
        page_sdk,
        page_buffers,
        layer_ids=(0,),
        buffer_owners=c4_kv.owners + c4_indexer.owners + c128.owners,
    )

    try:
        state_layout = make_deepseek_v4_checkpoint_layout(
            pool, include_compressed_pages=False
        )
        checkpoint_bytes = int(
            schema.get("checkpoint_bytes", state_layout.bytes_per_block)
        )
        state_namespace = _standard_namespace(ctx, "spa-checkpoint", ctx.page_shard_num)
        state_sdk = AttentionStoreSDK(
            _sdk_pod_config(ctx, state_namespace, gpu_count=checkpoint_chunks),
            SPAConfig(
                block_token_size=ctx.page_size,
                c4_layer_count=0,
                c128_layer_count=0,
                c4_kv_bytes=0,
                c4_indexer_bytes=0,
                c128_kv_bytes=0,
                swa_read_layer_bytes=checkpoint_bytes,
                swa_write_layer_bytes=checkpoint_bytes,
                swa_layer_count=1,
                swa_step=ctx.page_size,
                shard_id=ctx.page_shard_id,
                shard_num=ctx.page_shard_num,
            ),
            enable_as_kv_rw=True,
        )
    except Exception:
        page_store.stop()
        raise
    try:
        state_store = SPACheckpointStore(
            state_sdk,
            state_layout,
            storage_bytes_per_block=checkpoint_bytes,
        )
    except Exception:
        state_sdk.stop()
        page_store.stop()
        raise
    page_bytes_per_block = (
        c4_kv_bytes + c4_indexer_bytes
    ) * c4_count + c128_bytes * c128_count
    cache_store = CheckpointedPageStore(page_store, state_store)
    try:
        page_endpoint = AttentionStoreEndpointBinding(
            name="spa",
            role=AttentionStoreEndpointRole.PAGE,
            store=page_store,
            namespace=page_namespace,
            layout_id="spa-page",
            bytes_per_block=page_bytes_per_block,
            payload_fields=("c4_kv", "c4_indexer", "c128_kv"),
        )
        checkpoint_endpoint = AttentionStoreEndpointBinding(
            name="checkpoint",
            role=AttentionStoreEndpointRole.CHECKPOINT,
            store=state_store,
            namespace=state_namespace,
            layout_id=state_layout.layout_id,
            bytes_per_block=checkpoint_bytes,
            payload_fields=tuple(entry.spec.name for entry in state_layout.entries),
        )
        return AttentionStoreBinding(
            plan_name="SPA",
            cache_store=cache_store,
            endpoints=(page_endpoint, checkpoint_endpoint),
        )
    except Exception:
        cache_store.stop()
        raise
