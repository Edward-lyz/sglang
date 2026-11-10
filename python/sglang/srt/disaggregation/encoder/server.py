import asyncio
import concurrent.futures
import ctypes
import hashlib
import logging
import os
import pickle
import threading
import time
import traceback
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Set, Tuple

import msgspec
import numpy as np
import torch
import zmq
import zmq.asyncio

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.disaggregation.encode_collector import (
    EncoderMetricsCollector as FactoryEncoderMetricsCollector,
    StageReadyMark,
)
from sglang.srt.disaggregation.encoder.preprocessor import (
    EncoderPreprocessor,
    EncoderPreprocessResult,
    _convert,
    _mm_grid_attrs,
    embed_ready,
    media_load_seconds,
)
from sglang.srt.disaggregation.encoder.receiver import (
    EmbeddingData,
    video_meta_attrs_for,
)
from sglang.srt.distributed.parallel_state import (
    get_default_distributed_backend,
    get_mooncake_transfer_engine,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import initialize_dp_attention
from sglang.srt.managers.io_struct import (
    ProfileReq,
    ProfileReqType,
    async_sock_recv,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult, MultiModalStaticCache
from sglang.srt.model_executor.model_runner_components.load_model_utils import (
    maybe_precompile_model_kernels_after_loading,
)
from sglang.srt.model_loader import get_model as load_model
from sglang.srt.multimodal.encoder_preprocessing import (
    get_encoder_preprocessed_items,
    resolve_encoder_media_processor_config,
)
from sglang.srt.observability.metrics_collector import EncoderMetricsCollector
from sglang.srt.runtime_context import (
    assert_published,
    get_device,
    get_disagg,
    get_exec,
    get_mm,
    get_model,
    get_parallel,
    publish,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import configure_media_url_security
from sglang.srt.utils.network import (
    NetworkAddress,
    config_socket,
    get_local_ip_auto,
    get_zmq_socket,
)

logger = logging.getLogger(__name__)


def is_health_check_request(rid: Optional[str]) -> bool:
    return isinstance(rid, str) and rid.startswith(HEALTH_CHECK_RID_PREFIX)


def _flatten_encoder_embedding(embedding):
    """Normalize a get_*_feature() result to one ``[tokens, hidden]`` tensor.

    K3's ``get_image_feature`` returns a single tensor, but models like
    DeepSeek-V4.1 return a per-image list of spans (with the image delimiter
    embeddings baked in). Mirror the colocated path
    (``mm_schedule._flatten_embedding_result``) so the encoder always holds one
    contiguous embedding to cache / move to host / RDMA-transfer.
    """
    from sglang.srt.managers.mm_schedule import _flatten_embedding_result

    return _flatten_embedding_result(embedding)


def _cgroup_memory_limit_bytes() -> int:
    """This container's memory limit (bytes), or 0 when unlimited/unknown."""
    candidates = (
        "/sys/fs/cgroup/memory.max",  # cgroup v2
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # cgroup v1
    )
    for path in candidates:
        try:
            with open(path) as handle:
                value = handle.read().strip()
        except OSError:
            continue
        if value in ("", "max"):
            return 0
        try:
            limit = int(value)
        except ValueError:
            continue
        # cgroup v1 reports a huge sentinel when unlimited.
        if limit <= 0 or limit >= (1 << 60):
            return 0
        return limit
    return 0


def _split_rows(embedding: torch.Tensor, row_counts: Iterable[int]) -> List[torch.Tensor]:
    """Consecutive row slices of ``embedding`` (short slices if it is too short)."""
    parts, offset = [], 0
    for count in row_counts:
        parts.append(embedding[offset : offset + count])
        offset += count
    return parts


def _request_segments(ctx, mm_items) -> Optional[List[Tuple[int, int, int]]]:
    """``(item_start, item_end, num_tokens)`` of each request of ``ctx``.

    Item ranges index ``mm_items``. A fused batch built from a legacy aggregate
    item cannot be split per request, so it has no segments (None).
    """
    token_counts = ctx.preprocess_result.token_counts
    if len(ctx.items_per_req) == 1:
        return [(0, len(mm_items), sum(token_counts))]
    if len(mm_items) != ctx.num_items:
        return None
    segments, start = [], 0
    for count in ctx.items_per_req:
        segments.append((start, start + count, sum(token_counts[start : start + count])))
        start += count
    return segments



rid_lock = asyncio.Lock()
rid_to_receive_endpoint: Dict[str, Set[str]] = dict()
rid_to_receive_count: Dict[str, int] = dict()
cond_dict_lock = asyncio.Lock()
rid_to_cond: Dict[str, asyncio.Condition] = {}
encode_state_condition = asyncio.Condition()


async def _get_receive_condition(req_id: str) -> asyncio.Condition:
    async with cond_dict_lock:
        if req_id not in rid_to_cond:
            rid_to_cond[req_id] = asyncio.Condition()
        return rid_to_cond[req_id]


async def _notify_receive_waiters(req_id: str) -> None:
    """Wake an existing destination waiter without creating new state."""
    async with cond_dict_lock:
        cond = rid_to_cond.get(req_id)
    if cond is not None:
        async with cond:
            cond.notify_all()


ENCODER_MAX_BATCH_SIZE = envs.SGLANG_ENCODER_MAX_BATCH_SIZE.get()
ENCODER_MAX_BATCH_SIZE_EXPLICIT = envs.SGLANG_ENCODER_MAX_BATCH_SIZE.is_set()
# Watchdog: max time to wait for a batched /encode result. Bounds HTTP latency
# if the batch worker stalls (NCCL hang, dead worker proc, etc.).
ENCODER_REQ_TIMEOUT = envs.SGLANG_ENCODER_REQ_TIMEOUT.get()


async def await_task_completion_on_cancel(task: asyncio.Task, operation: str):
    """Keep task-owned resources live until cancellation reaches a safe point."""
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not task.cancelled() and task.exception() is not None:
            logger.error(
                "%s failed while draining cancellation",
                operation,
                exc_info=task.exception(),
            )
        raise


async def _await_transfer_completion(awaitable, operation: str):
    """Do not let cancellation outlive a zero-copy transfer using its buffer."""
    return await await_task_completion_on_cancel(
        asyncio.ensure_future(awaitable), operation
    )


class EncoderMetaRegistry:
    """Per-part metadata shared by every encoder request lifecycle.

    Mooncake decoder ranks consume it early to allocate landing buffers. ZMQ
    publishes the same state for a uniform pipeline but does not consume it
    before encode/send completes.
    """

    def __init__(self, *, wait_timeout: float, sweep_timeout: float):
        # How long a decoder blocks in /scheduler_receive_meta_data.
        self.wait_timeout = wait_timeout
        # Backstop for state whose /send calls never all land.
        self.sweep_timeout = sweep_timeout
        self._rid_to_meta: Dict[str, dict] = {}
        self._rid_to_send_done: Dict[str, Set[str]] = {}
        self._pending_at: Dict[str, float] = {}
        self._sweeper_task: Optional[asyncio.Task] = None
        self._stale_release_tasks: Dict[str, asyncio.Task] = {}
        # Set only where the embedding also lives; None in the DP main process.
        self.on_release: Optional[Callable[[str], Awaitable[None]]] = None

    def _touch(self, req_id: str) -> None:
        self._pending_at[req_id] = time.monotonic()
        self._ensure_sweeper()

    def _ensure_sweeper(self) -> None:
        if self._sweeper_task is not None and not self._sweeper_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._sweeper_task = loop.create_task(self._sweep_loop())

    async def _sweep_loop(self) -> None:
        # Same idiom as DPDispatcher._cleanup_stale_mappings: one eternal
        # scanner; interval re-read each pass so the MMEncoder override applies.
        while True:
            await asyncio.sleep(max(self.sweep_timeout / 4, 0.01))
            now = time.monotonic()
            stale = [
                rid
                for rid, ts in self._pending_at.items()
                if now - ts > self.sweep_timeout
                and rid not in self._stale_release_tasks
            ]
            for rid in stale:
                self._schedule_stale_release(rid)

    def _schedule_stale_release(self, req_id: str) -> asyncio.Task:
        """Release one stale request without blocking cleanup of other requests."""
        if task := self._stale_release_tasks.get(req_id):
            return task
        task = asyncio.create_task(self._release_stale(req_id))
        self._stale_release_tasks[req_id] = task
        task.add_done_callback(
            lambda done, rid=req_id: self._finish_stale_release(rid, done)
        )
        return task

    def _finish_stale_release(self, req_id: str, task: asyncio.Task) -> None:
        if self._stale_release_tasks.get(req_id) is task:
            self._stale_release_tasks.pop(req_id)

    async def _release_stale(self, req_id: str) -> None:
        try:
            await self._release(req_id)
        except Exception:
            logger.exception("Failed to release stale encoder request %s", req_id)
            # Keep the request eligible for a later sweep without retrying in a
            # tight loop. Its metadata and buffer ownership remain intact.
            async with rid_lock:
                if req_id in self._pending_at:
                    self._pending_at[req_id] = time.monotonic()

    async def publish(
        self,
        req_id: str,
        nbytes: int,
        embedding_len: int,
        embedding_dim: int,
        error: Optional[str] = None,
    ) -> None:
        """Publish per-part metadata (or an error), wake waiters, arm the sweep."""
        meta = (
            {"error": error}
            if error is not None
            else {
                "embedding_size": nbytes,
                "embedding_len": embedding_len,
                "embedding_dim": embedding_dim,
            }
        )
        async with rid_lock:
            self._rid_to_meta[req_id] = meta
            self._touch(req_id)
        cond = await _get_receive_condition(req_id)
        async with cond:
            cond.notify_all()

    async def wait(self, req_id: str) -> Optional[dict]:
        """Block until req_id's metadata is published; TimeoutError past wait_timeout.
        No _touch here: a pull-first timestamp would let the sweeper pop the very
        Condition this waiter holds, stranding it when publish notifies a new one."""
        cond = await _get_receive_condition(req_id)
        async with cond:
            await asyncio.wait_for(
                cond.wait_for(lambda: self._rid_to_meta.get(req_id) is not None),
                timeout=self.wait_timeout,
            )
        return self._rid_to_meta.get(req_id)

    async def note_send_done(
        self, req_id: str, receive_count: int, destination_endpoint: str
    ) -> None:
        """Count one destination once; release after every receiver has sent."""
        async with rid_lock:
            completed = self._rid_to_send_done.setdefault(req_id, set())
            completed.add(destination_endpoint)
            all_done = len(completed) >= receive_count
        if all_done:
            await self._release(req_id)

    async def _release(self, req_id: str) -> None:
        if self.on_release is not None:
            await self.on_release(req_id)
        await self.discard(req_id)

    async def discard(self, req_id: str) -> None:
        """Drop the meta rendezvous state for req_id. Idempotent."""
        async with rid_lock:
            self._rid_to_meta.pop(req_id, None)
            self._rid_to_send_done.pop(req_id, None)
            self._pending_at.pop(req_id, None)
        async with cond_dict_lock:
            rid_to_cond.pop(req_id, None)


meta_registry = EncoderMetaRegistry(
    wait_timeout=ENCODER_REQ_TIMEOUT,
    sweep_timeout=envs.SGLANG_ENCODER_SEND_TIMEOUT.get(),
)


class MMError(Exception):
    def __init__(self, message, code=HTTPStatus.INTERNAL_SERVER_ERROR):
        self.message = message
        self.code = code
        super().__init__(self.message)


class BadRequestError(MMError):
    def __init__(self, message):
        super().__init__(message, code=HTTPStatus.BAD_REQUEST)


class InternalError(MMError):
    def __init__(self, message):
        super().__init__(message, code=HTTPStatus.INTERNAL_SERVER_ERROR)


class EncodeContext(msgspec.Struct):
    """One flattened encode batch; a single request is the N=1 case."""

    req_id: str  # first request's id, for cache prefetch keys and logs
    modality: Modality
    preprocess_result: EncoderPreprocessResult
    get_feature_fn: Any
    mm_feature: Any
    num_items: int
    items_per_req: List[int]  # grid entries per request, in flatten order
    aux_data: dict
    str_mm_hashes: Optional[List[str]]
    use_global_cache: bool
    is_health_check: bool


def _preprocess_layout_digest(ctx: EncodeContext) -> tuple[int, int]:
    """Hash metadata that must agree before TP ranks enter model forward."""

    def normalize(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return (
                str(value.dtype),
                tuple(value.shape),
                tuple(value.reshape(-1).tolist()),
            )
        if isinstance(value, (list, tuple)):
            return tuple(normalize(item) for item in value)
        if isinstance(value, np.generic):
            return value.item()
        return value

    signature = (
        tuple(ctx.items_per_req),
        tuple(ctx.preprocess_result.token_counts),
        normalize(ctx.preprocess_result.grid_thw),
    )
    digest = hashlib.blake2b(pickle.dumps(signature), digest_size=16).digest()
    return (
        int.from_bytes(digest[:8], byteorder="little", signed=True),
        int.from_bytes(digest[8:], byteorder="little", signed=True),
    )


@dataclass
class ReqState:
    """The result and in-flight work for one encoder request."""

    req_id: str
    embedding_data: Optional[EmbeddingData] = None
    active_encodes: int = 0
    active_sends: int = 0
    release_requested: bool = False
    preserve_metadata_on_release: bool = False
    embedding_ready: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    lifecycle_condition: asyncio.Condition = field(
        default_factory=asyncio.Condition, repr=False
    )


@dataclass(frozen=True)
class SendDestination:
    """One normalized destination for exactly one transfer."""

    endpoint: str
    session_id: Optional[str] = None
    buffer_address: Optional[int] = None

    @classmethod
    def from_host_port(
        cls,
        prefill_host: str,
        embedding_port: int,
        *,
        session_id: Optional[str] = None,
        buffer_address: Optional[int] = None,
    ) -> "SendDestination":
        return cls(
            endpoint=NetworkAddress(prefill_host, embedding_port).to_host_port_str(),
            session_id=session_id,
            buffer_address=buffer_address,
        )

    @classmethod
    def from_url(cls, url: str) -> "SendDestination":
        return cls(endpoint=NetworkAddress.parse(url).to_host_port_str())


class TensorWrapper:
    """Wrapper to keep tensor alive while exposing buffer for zero-copy."""

    def __init__(self, tensor):
        # Ensure tensor is on CPU and contiguous
        if tensor.is_cuda:
            tensor = tensor.cpu()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        # Keep tensor reference
        self.tensor = tensor
        self.shape = list(tensor.shape)
        self.dtype = tensor.dtype

    def __buffer__(self):
        data_ptr = self.tensor.data_ptr()
        total_bytes = self.tensor.numel() * self.tensor.element_size()
        c_obj = (ctypes.c_char * total_bytes).from_address(data_ptr)
        c_obj._keep_alive_ref = self
        return memoryview(c_obj)


class EncoderDelivery(ABC):
    """Transfer backend boundary. Send never releases the request."""

    def __init__(self, encoder: "MMEncoder"):
        self.encoder = encoder

    @abstractmethod
    async def send(
        self,
        state: ReqState,
        destination: SendDestination,
    ) -> None: ...

    @abstractmethod
    async def release(self, state: ReqState) -> None: ...


class MooncakeDelivery(EncoderDelivery):
    async def send(
        self,
        state: ReqState,
        destination: SendDestination,
    ) -> None:
        mm_data = await self.encoder._wait_for_embedding(state)
        await self.encoder._send(
            mm_data.embedding,
            mm_data,
            session_id=destination.session_id,
            buffer_address=destination.buffer_address,
            url=destination.endpoint,
        )

    async def release(self, state: ReqState) -> None:
        mm_data = state.embedding_data
        if mm_data is not None:
            self.encoder._deregister_shared_mr(mm_data)


class ZmqDelivery(EncoderDelivery):
    def __init__(self, encoder: "MMEncoder", *, cleanup_receive_state: bool) -> None:
        super().__init__(encoder)
        self.cleanup_receive_state = cleanup_receive_state

    async def send(
        self,
        state: ReqState,
        destination: SendDestination,
    ) -> None:
        mm_data = await self.encoder._wait_for_embedding(state)
        await self.encoder._send(mm_data.embedding, mm_data, url=destination.endpoint)

    async def release(self, state: ReqState) -> None:
        if not self.cleanup_receive_state:
            return
        async with rid_lock:
            rid_to_receive_endpoint.pop(state.req_id, None)
            rid_to_receive_count.pop(state.req_id, None)
        async with cond_dict_lock:
            rid_to_cond.pop(state.req_id, None)


_mm_feature_attrs = {
    Modality.IMAGE: ["pixel_values"],
    Modality.VIDEO: ["pixel_values_videos"],
    Modality.AUDIO: ["input_features"],
}


def _get_mm_feature(mm_inputs, modality):
    for attr in _mm_feature_attrs[modality]:
        if attr in mm_inputs:
            return mm_inputs[attr]
    raise ValueError(
        f"Feature attrs ({_mm_feature_attrs[modality]}) not found in {mm_inputs}"
    )


def _normalize_aux_value(val):
    """Normalize aux values to pickle types compatible with safe_pickle_loads.

    HF multimodal processors (e.g. Qwen3-VL/Omni) emit numpy arrays for
    fields like ``video_timestamps`` / ``second_per_grid_ts``. ``numpy.*`` is
    not in SafeUnpickler's allowlist, so the receiver would refuse to load
    those payloads. Convert numpy values to torch tensors (numeric) or plain
    Python lists (object dtype) before pickling.
    """
    if val is None:
        return None
    if isinstance(val, np.ndarray):
        if val.dtype == object:
            return val.tolist()
        return torch.from_numpy(np.ascontiguousarray(val))
    if isinstance(val, np.generic):
        return val.item()
    if isinstance(val, (list, tuple)):
        return type(val)(_normalize_aux_value(v) for v in val)
    if isinstance(val, dict):
        return {k: _normalize_aux_value(v) for k, v in val.items()}
    return val


# Models whose factory TaskResult carries each image's pre-resize [w, h] (K3's
# EncoderAddr.original_image_sizes).
_MODELS_REPORTING_ORIGINAL_IMAGE_SIZES = frozenset({"kimi_k3", "deepseek_v41"})


def _build_mm_aux_data(mm_inputs, model_type=None):
    # Video aux metadata, scoped to model_type's video-meta attrs.
    aux = {
        attr: _normalize_aux_value(mm_inputs.get(attr))
        for attr in video_meta_attrs_for(model_type)
    }
    if model_type in _MODELS_REPORTING_ORIGINAL_IMAGE_SIZES:
        aux["original_image_sizes"] = _normalize_aux_value(
            mm_inputs.get("original_image_sizes")
        )
    return aux


def compute_embed_digest(embedding: "torch.Tensor") -> Dict[str, Any]:
    """Lightweight embedding fingerprint for transfer verification (as K3)."""
    flat = embedding.reshape(-1).float()
    return {
        "mean": flat.mean().item(),
        "std": flat.std().item(),
        "min": flat.min().item(),
        "max": flat.max().item(),
        "first8": flat[:8].tolist(),
        "last8": flat[-8:].tolist(),
        "num_elements": flat.numel(),
    }


@dataclass
class WorkflowTaskEmbedding:
    """One embedding held for a factory (厂内) pull, keyed by task_id.

    The New EPD encoder normally pushes embeddings to prefill; the factory
    protocol instead has prefill pull them, so a workflow-produced embedding is
    parked here (with the geometry the pull reply needs) until pulled. ``refs``
    guards against eviction while a transfer is in flight, mirroring the K3
    mm_cache acquire/release. ``created`` (monotonic) drives TTL eviction of
    holds a dead/absent prefill never pulls; ``mr_ptr`` is the mooncake MR that
    must be deregistered when the hold is dropped.
    """

    embedding: "torch.Tensor"
    image_grid_thw: Optional[Any] = None
    original_image_sizes: Optional[Any] = None
    refs: int = 0
    created: float = 0.0
    mr_ptr: Optional[int] = None
    # Address in the pre-registered hold pool, freed when the hold is dropped.
    pool_addr: Optional[int] = None
    size_bytes: int = 0
    pulled_once: bool = False
    # TaskResults handed out for this hold that prefill has not pulled yet.
    # LRU eviction needs this to be 0, so a second request served from the
    # same hold cannot lose it to eviction before its own pull.
    pending_pulls: int = 0
    modality: str = "image"
    chunk_timestamps: Optional[List[str]] = None


def _error_from_encode_result(result, what: str) -> MMError:
    """Rebuild the MMError a batch_encode result row reports."""
    _, _, _, message, code = result
    message = message or f"No embedding staged for {what}"
    if code == HTTPStatus.BAD_REQUEST:
        return BadRequestError(message)
    return MMError(message, code=code or HTTPStatus.INTERNAL_SERVER_ERROR)


class FactoryEncodeBatcher:
    """Coalesce concurrent factory tasks into fused ``batch_encode`` calls.

    Each factory task used to be its own encode (one ViT forward per image).
    Tasks submitted while a batch is forming join it: a batch closes when it
    holds ``max_batch`` tasks or ``max_wait_s`` after its first task, and at
    most ``max_inflight`` batches run at once, so the next batch loads its
    media while the current one is in the forward. A failed fused batch is
    retried task by task, so one bad image fails only its own task. ``release``
    frees the request state of a task whose caller left before its batch ended.
    """

    def __init__(
        self,
        run_batch: Callable[..., Awaitable[List[Tuple]]],
        *,
        max_batch: int,
        max_wait_s: float,
        max_inflight: int,
        release: Optional[Callable[[dict], Awaitable[None]]] = None,
    ):
        self._run_batch = run_batch
        self._release = release
        self.max_batch = max(1, max_batch)
        self.max_wait_s = max(0.0, max_wait_s)
        self.max_inflight = max(1, max_inflight)
        self._pending: Dict[Modality, List[Tuple[dict, asyncio.Future]]] = {}
        self._timers: Dict[Modality, asyncio.TimerHandle] = {}
        self._inflight = 0
        self._tasks: Set[asyncio.Task] = set()

    async def submit(self, request: dict, modality: Modality) -> EmbeddingData:
        """Encode ``request`` in some fused batch; returns its staged EmbeddingData."""
        future = asyncio.get_running_loop().create_future()
        self._pending.setdefault(modality, []).append((request, future))
        if len(self._pending[modality]) >= self.max_batch:
            self._flush(modality)
        elif modality not in self._timers:
            self._timers[modality] = asyncio.get_running_loop().call_later(
                self.max_wait_s, self._on_timer, modality
            )
        return await future

    def _on_timer(self, modality: Modality) -> None:
        self._timers.pop(modality, None)
        self._flush(modality)

    def _flush(self, modality: Modality) -> None:
        """Start a batch for ``modality`` if one is pending and a slot is free."""
        if self._inflight >= self.max_inflight or not self._pending.get(modality):
            return
        timer = self._timers.pop(modality, None)
        if timer is not None:
            timer.cancel()
        entries = self._pending[modality][: self.max_batch]
        del self._pending[modality][: self.max_batch]
        self._inflight += 1
        task = asyncio.create_task(self._run(entries, modality))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run(self, entries, modality: Modality) -> None:
        try:
            # Callers that gave up (task timeout) before the batch started.
            entries = [entry for entry in entries if not entry[1].done()]
            if not entries:
                return
            staged, failure = await self._encode(
                [request for request, _ in entries], modality
            )
            if staged is not None:
                results = staged
            elif len(entries) == 1:
                results = [_error_from_encode_result(failure[0], entries[0][0]["req_id"])]
            else:
                # The fused batch failed as a whole (one undecodable image
                # fails the shared preprocess): retry the tasks still waited
                # on one by one, so only the failing task fails. The failed
                # batch staged an error into each request's state; drop it
                # first or the retry trips the staged-metadata check.
                results = []
                for request, future in entries:
                    await self._release_request(request)
                    if future.done():
                        results.append(None)
                        continue
                    one, one_failure = await self._encode([request], modality)
                    results.append(
                        one[0]
                        if one is not None
                        else _error_from_encode_result(
                            one_failure[0], request["req_id"]
                        )
                    )
            for (request, future), result in zip(entries, results):
                if future.done():
                    # The caller left mid-batch, possibly before its release
                    # could find the request state this batch then created.
                    await self._release_request(request)
                    continue
                if isinstance(result, Exception):
                    future.set_exception(result)
                else:
                    future.set_result(result)
        except BaseException as error:
            for _, future in entries:
                if future.done():
                    continue
                if isinstance(error, Exception):
                    future.set_exception(error)
                else:
                    future.cancel()
            if not isinstance(error, Exception):
                raise
        finally:
            self._inflight -= 1
            # A finished batch frees a slot: start whatever queued behind it.
            for pending_modality in list(self._pending):
                if self._pending[pending_modality]:
                    self._flush(pending_modality)

    async def _encode(self, requests: List[dict], modality: Modality):
        """``(staged, None)`` with one EmbeddingData per request, or ``(None, rows)``.

        A batch either stages every request or none (``rows`` then carries the
        batch_encode error results).
        """
        staged: List[EmbeddingData] = []
        results = await self._run_batch(requests, modality, out_staged=staged)
        if len(staged) == len(requests):
            return staged, None
        return None, results

    async def _release_request(self, request: dict) -> None:
        if self._release is None:
            return
        try:
            await self._release(request)
        except Exception:
            logger.exception("Releasing factory task %s failed", request["req_id"])


class MMEncoder:
    # Class default so instances built without __init__ (tests) read None.
    metrics_collector: Optional[FactoryEncoderMetricsCollector] = None

    def __init__(
        self,
        server_args: ServerArgs,
        schedule_path=None,
        dist_init_method=None,
        rank: int = 0,
        gpu_id: Optional[int] = None,
    ):
        """``gpu_id`` pins this encoder to a device other than
        ``base_gpu_id + rank`` — the DP launcher's per-worker placement. It is
        this instance's value, not a config change, so it travels as an
        argument."""
        assert_published(server_args, role="encoder")
        logger.info(f"init MMEncoder {rank}/{get_parallel().tp_size}")
        self.server_args = server_args
        configure_media_url_security(
            get_mm().allowed_media_domains,
            get_mm().media_url_max_file_size_mb,
        )
        self.transfer_backend = get_disagg().encoder_transfer_backend
        self.use_mooncake = self.transfer_backend == "mooncake"
        self.rank = rank
        # DP rank for metric labels; overridden by runtime.run_dp_worker.
        # 0 in the single-instance (non-DP) path.
        self.dp_rank = 0
        # 厂内 dashboard metrics (encode_collector); set by attach_metrics when
        # --enable-metrics is on. EncodeWorkflow reads it for request counters.
        self.metrics_collector: Optional[FactoryEncoderMetricsCollector] = None
        self.profiler = EncoderProfiler(rank)

        self.model_config = ModelConfig.from_server_args(
            server_args,
        )
        self.load_config = LoadConfig(
            load_format=get_model().load_format,
            download_dir=get_model().download_dir,
            model_loader_extra_config=get_model().model_loader_extra_config,
            remote_instance_weight_loader_seed_instance_ip=get_model().remote_instance_weight_loader_seed_instance_ip,
            remote_instance_weight_loader_seed_instance_service_port=get_model().remote_instance_weight_loader_seed_instance_service_port,
            remote_instance_weight_loader_send_weights_group_ports=get_model().remote_instance_weight_loader_send_weights_group_ports,
        )
        self.model_type = getattr(
            self.model_config.hf_config, "model_type", "unknown"
        ).lower()

        self.device = get_device().device
        self.gpu_id = get_device().base_gpu_id + rank if gpu_id is None else gpu_id

        self.device_config = DeviceConfig(
            device=self.device,
            gpu_id=self.gpu_id,
        )

        torch.get_device_module(self.device).set_device(self.gpu_id)

        init_distributed_environment(
            backend=get_default_distributed_backend(self.device),
            world_size=get_parallel().tp_size,
            rank=rank,
            distributed_init_method=dist_init_method,
            local_rank=rank,
        )
        initialize_model_parallel(
            tensor_model_parallel_size=get_parallel().tp_size,
            attention_context_model_parallel_size=get_parallel().attn_cp_size,
        )
        initialize_dp_attention(server_args, self.model_config)

        self.model = load_model(
            model_config=self.model_config,
            load_config=self.load_config,
            device_config=self.device_config,
        )
        encoder_media_processor_config = resolve_encoder_media_processor_config(
            self.model
        )
        maybe_precompile_model_kernels_after_loading(self.model, self.device)

        # CPU preprocessing pipeline (Rust-replaceable)
        self.preprocessor = EncoderPreprocessor(
            server_args=server_args,
            model_config=self.model_config,
            model_preprocessor=getattr(self.model, "preprocess_mm_for_encoder", None),
            encoder_media_processor_config=encoder_media_processor_config,
        )
        # Resolve model-side preprocessing backends (a Rust extension may be
        # built here) at startup, not on the first request's worker thread.
        prepare_preprocessing = getattr(
            self.model, "prepare_encoder_preprocessing", None
        )
        if prepare_preprocessing is not None:
            prepare_preprocessing()

        self.context = zmq.asyncio.Context(2)
        self.sync_context = zmq.Context()  # Reuse sync context for thread pool
        self.scheduler_send_sockets = {}
        self.scheduler_send_locks = {}
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)

        embedding_cache_size = int(os.environ.get("SGLANG_VLM_CACHE_SIZE_MB", "4096"))
        self.mm_cache = MultiModalStaticCache(embedding_cache_size * 1024 * 1024)
        self.mm_cache_lock = asyncio.Lock()

        self.send_timeout = envs.SGLANG_ENCODER_SEND_TIMEOUT.get()

        if schedule_path is not None:
            self.schedule_socket = get_zmq_socket(
                self.context, zmq.PULL, schedule_path, True
            )
        self.background_tasks: Set[asyncio.Task] = set()

        # Embedding dtype = model param dtype. Always available (both transfer
        # backends and the global-cache pool rely on it).
        self._embedding_dtype = next(self.model.parameters()).dtype
        self._element_size = torch.tensor(
            [], dtype=self._embedding_dtype
        ).element_size()
        self._embedding_dims = self._infer_embedding_dims()

        if get_mm().enable_mm_global_cache:
            from sglang.srt.mem_cache.embedding_cache_controller import (
                EmbeddingCacheController,
            )
            from sglang.srt.mem_cache.embedding_store import EmbeddingStoreFactory

            embedding_store = EmbeddingStoreFactory.create_backend(
                get_mm().mm_global_cache_backend,
            )
            self.mm_global_cache = EmbeddingCacheController(
                rank,
                get_parallel().tp_size,
                embedding_store=embedding_store,
                hidden_dims=self._embedding_dims,
                tp_group=get_tp_group().cpu_group,
                all_rank_get=False,
                dtype=self._embedding_dtype,
            )
        else:
            self.mm_global_cache = None

        if self.rank == 0:
            logger.info(
                f"Using transfer backend: {get_disagg().encoder_transfer_backend}"
            )

            if get_disagg().encoder_transfer_backend == "mooncake":
                self.local_ip = get_local_ip_auto()

                self.engine = get_mooncake_transfer_engine()
                if self.engine is None:
                    from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
                        init_mooncake_transfer_engine,
                    )

                    self.engine = init_mooncake_transfer_engine(
                        hostname=self.local_ip,
                        gpu_id=self.gpu_id,
                        ib_device=(
                            get_disagg().disaggregation_ib_device
                            or get_exec().moe.mooncake_ib_device
                        ),
                    )

            self.req_states: Dict[str, ReqState] = {}
            # A DP caller can disappear before its encode creates ReqState.
            # Preserve that release intent until _acquire_encode_ref runs.
            self.abandoned_req_ids: Set[str] = set()
            # Need to ensure the NCCL launch order on rank0 matches the dispatch order rank>0
            self.encode_dispatch_lock = asyncio.Lock()

            # Factory (厂内) EPD producer path: when polling a vit-server, the
            # encoder holds each computed embedding here keyed by task_id so
            # prefill can pull it (via PullZmqServer / /epd/transfer), instead
            # of the New EPD push. Populated by encode_and_store_for_workflow;
            # drained on pull. Empty unless --vit-server-url is set.
            self.task_embeddings: "OrderedDict[str, WorkflowTaskEmbedding]" = (
                OrderedDict()
            )
            self.task_embeddings_lock = asyncio.Lock()
            # Content-addressed LRU bound (bytes) for the hold cache above: a
            # pulled embedding is retained (not dropped) so sibling requests
            # sharing the same task_id can pull it too; capacity is reclaimed by
            # LRU eviction of unreferenced holds (see _evict_to_fit_locked),
            # with TTL as a fallback for holds a dead prefill never pulls.
            self.task_embeddings_bytes = 0
            # Tasks served from an existing hold without re-encoding.
            self.task_embedding_hits = 0
            # task_id -> stage costs for TaskResult.time_cost_info.
            self.task_time_costs: Dict[str, Dict[str, float]] = {}
            self.task_embeddings_max_bytes = int(
                envs.SGLANG_ENCODER_TASK_EMB_CACHE_GB.get() * (1024**3)
            )
            # K3's ScatterMemPool counterpart: one pinned buffer of the hold
            # cap, registered with mooncake once (see _init_task_hold_pool).
            self.task_hold_pool = None
            # Port the factory PullZmqServer binds for prefill DEALER pulls; 0
            # disables (no factory pull path).
            self.pull_zmq_port = envs.SGLANG_ENCODER_PULL_ZMQ_PORT.get()
            # Set once EncodeWorkflow starts; free_tokens reads its pending.
            self.encode_workflow = None

            if get_disagg().encoder_transfer_backend == "mooncake":
                self.delivery = MooncakeDelivery(self)
                # Embeddings live here, so registry cleanup uses the common release.
                meta_registry.on_release = self.release_request
                meta_registry.sweep_timeout = self.send_timeout
            else:
                self.delivery = ZmqDelivery(
                    self,
                    cleanup_receive_state=(
                        get_disagg().encoder_transfer_backend == "zmq_to_scheduler"
                    ),
                )

        logger.info(f"rank {rank} init finish ")

    def _background_task_done(self, task: asyncio.Task) -> None:
        self.background_tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("MMEncoder background task failed")

    def _create_background_task(self, awaitable: Awaitable[Any]) -> asyncio.Task:
        task = asyncio.create_task(awaitable)
        self.background_tasks.add(task)
        task.add_done_callback(self._background_task_done)
        return task

    def supports_modality(self, modality: Modality) -> bool:
        return self.preprocessor.supports_modality(modality)

    def has_pending_embeddings(self) -> bool:
        return bool(getattr(self, "req_states", None))

    def _require_active_encode_state(self, req_id: str) -> ReqState:
        """Return the state holding an encode ref; never resurrect a request."""
        state = self.req_states.get(req_id)
        if state is None:
            raise InternalError(
                f"No request state exists while encoding request: {req_id}"
            )
        if state.active_encodes <= 0:
            raise InternalError(f"Request state has no active encode work: {req_id}")
        return state

    def _acquire_encode_ref(self, req_id: str) -> Optional[ReqState]:
        """Acquire a rank 0 encode ref before preprocessing can suspend."""
        if self.rank != 0:
            return None
        state = self.req_states.get(req_id)
        if state is None:
            state = ReqState(req_id)
            self.req_states[req_id] = state
        state.active_encodes += 1
        if req_id in self.abandoned_req_ids:
            state.release_requested = True
            self.abandoned_req_ids.discard(req_id)
        return state

    async def abandon_request(self, req_id: str) -> None:
        """Release now, or remember the release until encode state exists."""
        self.abandoned_req_ids.add(req_id)
        if req_id in self.req_states:
            self.abandoned_req_ids.discard(req_id)
            await self.release_request(req_id)

    def clear_abandoned_request(self, req_id: str) -> None:
        """Drop an unused release marker after the worker task exits."""
        self.abandoned_req_ids.discard(req_id)

    async def _release_encode_ref(self, state: Optional[ReqState]) -> None:
        if state is None:
            return
        async with state.lifecycle_condition:
            state.active_encodes -= 1
            assert state.active_encodes >= 0
            should_release = state.release_requested and state.active_encodes == 0
            state.lifecycle_condition.notify_all()
        if should_release:
            await self.release_request(state.req_id)

    def _embedding_state_for_stage(self, mm_data: EmbeddingData) -> ReqState:
        state = self._require_active_encode_state(mm_data.req_id)
        metadata = state.embedding_data
        if (
            metadata is not None
            and metadata.embedding is None
            and mm_data.embedding is not None
            and (metadata.shape != mm_data.shape or metadata.dtype != mm_data.dtype)
        ):
            raise InternalError(
                f"Embedding metadata mismatch for {mm_data.req_id}: "
                f"expected={metadata.shape}/{metadata.dtype}, "
                f"actual={mm_data.shape}/{mm_data.dtype}"
            )
        return state

    def _stage_embedding(self, mm_data: EmbeddingData) -> None:
        state = self._embedding_state_for_stage(mm_data)
        state.embedding_data = mm_data
        state.embedding_ready.set()

    def _stage_embedding_batch(self, embeddings: List[EmbeddingData]) -> None:
        """Validate the whole fused batch before publishing any result."""
        states = [self._embedding_state_for_stage(mm_data) for mm_data in embeddings]
        for state, mm_data in zip(states, embeddings):
            state.embedding_data = mm_data
        for state in states:
            state.embedding_ready.set()

    async def _wait_for_embedding(self, state: ReqState) -> EmbeddingData:
        await state.embedding_ready.wait()
        if state.embedding_data is None:
            raise InternalError(f"No embedding available for request: {state.req_id}")
        return state.embedding_data

    # ------------------------------------------------------------------
    # Factory (厂内) EPD producer helpers: fetch/encode/hold-for-pull.
    # Only exercised when --vit-server-url is set; otherwise inert.
    # ------------------------------------------------------------------

    @property
    def free_tokens(self) -> int:
        """Fetch-admission budget: token budget minus in-flight pipeline tokens.

        The factory poll loop only fetches new tasks while this stays above its
        admit threshold, which is what bounds how much work is in flight. With
        no workflow attached the full budget is free.
        """
        budget = envs.SGLANG_ENCODER_WORKFLOW_TOKEN_BUDGET.get()
        workflow = self.encode_workflow
        pending = workflow.pending_num_tokens if workflow is not None else 0
        return budget - pending

    async def store_task_embedding(
        self,
        task_id: str,
        embedding: "torch.Tensor",
        image_grid_thw: Optional[Any] = None,
        original_image_sizes: Optional[Any] = None,
        modality: str = "image",
        chunk_timestamps: Optional[List[str]] = None,
    ) -> None:
        """Hold one workflow-produced embedding for prefill to pull by task_id.

        Kept on CPU and contiguous so the pull reply is a plain buffer view and
        the transfer never blocks on a device copy. On the mooncake backend the
        buffer is registered as an MR here (once) so ``transfer_task_embedding``
        can RDMA it without a per-pull register; the MR is torn down when the
        hold is dropped.
        """
        pool_addr = None
        mr_ptr = None
        stored = False
        try:
            pool = getattr(self, "task_hold_pool", None)
            if pool is not None and int(embedding.nbytes) > pool.total_bytes:
                # Can never fit, so the pool-full retry would never succeed:
                # fall back to one MR per hold (as when no pool is configured).
                logger.warning(
                    "hold of %d bytes exceeds the %d byte factory hold pool; "
                    "registering it per hold",
                    int(embedding.nbytes),
                    pool.total_bytes,
                )
                pool = None
            if pool is not None:
                async with self.task_embeddings_lock:
                    # Reap TTL-expired holds first, so a pool full of dead
                    # (never-pulled) holds does not refuse a new task.
                    self._evict_stale_task_embeddings_locked()
                    pool_addr, held_view = self._alloc_from_hold_pool_locked(
                        embedding
                    )
                if held_view is not None:
                    # One copy straight into pre-registered pinned memory: no
                    # pageable staging, no per-hold MR register (K3 ScatterMemPool).
                    held_view.copy_(embedding.reshape(held_view.shape))
                    embedding = held_view
                else:
                    # The pool is full of in-flight or not-yet-pulled holds.
                    # Falling back to per-hold pin+register here would be
                    # unbounded in host memory and reintroduce the "overlapped
                    # memory region" failures the pool removes. Fail the task
                    # retryably so vit-server places it on another encoder;
                    # free_tokens admission normally keeps the pool from filling.
                    raise MMError(
                        f"factory hold pool full for task_id={task_id}; retry",
                        code=HTTPStatus.SERVICE_UNAVAILABLE,
                    )
            if pool_addr is None and embedding.is_cuda:
                embedding = embedding.cpu()
            if pool_addr is None and (
                embedding.untyped_storage().nbytes() > embedding.nbytes
            ):
                # Per-hold fallback of a row slice out of a fused batch would
                # keep the whole batch alive; give it its own storage.
                embedding = embedding.clone()
            if not embedding.is_contiguous():
                embedding = embedding.contiguous()
            if (
                pool_addr is None
                and self.use_mooncake
                and getattr(self, "engine", None) is not None
            ):
                # Pin the host buffer before registering: an RDMA MR must sit on
                # memory the OS cannot move, and the prefill landing pool is
                # pinned too. Mirrors K3's pinned mem_pool (a plain .cpu() tensor
                # is pageable and unsafe to RDMA from).
                try:
                    embedding = embedding.pin_memory()
                except Exception as pin_err:
                    logger.warning(
                        "pin_memory failed for task_id=%s (%s); registering pageable",
                        task_id,
                        pin_err,
                    )
                try:
                    self.engine.register(embedding.data_ptr(), embedding.nbytes)
                    mr_ptr = embedding.data_ptr()
                except Exception as reg_err:
                    # A hold prefill cannot RDMA from must not be advertised:
                    # fail the task so vit-server can place it elsewhere.
                    raise InternalError(
                        f"MR register failed for task_id={task_id}: {reg_err}"
                    ) from reg_err
            async with self.task_embeddings_lock:
                self._evict_stale_task_embeddings_locked()
                # task_id == md5(model_id + media_url) == identical content, so
                # an existing hold carries the same embedding. Keep it (a sibling
                # request that shares this image may be mid-pull, refs>0) and
                # just refresh its TTL, instead of tearing down the live slot and
                # racing a concurrent pull into a 404. The freshly-encoded
                # duplicate is retired in the finally below.
                old = self.task_embeddings.get(task_id)
                if old is not None:
                    self._mark_hold_reserved_locked(task_id, old)
                    return
                size_bytes = int(embedding.nbytes)
                if pool_addr is None:
                    self._evict_to_fit_locked(size_bytes)
                self.task_embeddings[task_id] = WorkflowTaskEmbedding(
                    embedding=embedding,
                    image_grid_thw=image_grid_thw,
                    original_image_sizes=original_image_sizes,
                    created=time.monotonic(),
                    mr_ptr=mr_ptr,
                    pool_addr=pool_addr,
                    size_bytes=size_bytes,
                    modality=modality,
                    chunk_timestamps=chunk_timestamps,
                    pending_pulls=1,
                )
                self.task_embeddings_bytes += size_bytes
                self.task_embeddings.move_to_end(task_id)
                stored = True
        finally:
            if not stored:
                # Duplicate, error or cancellation: give back the pool region
                # and MR this store took, or they leak for the process lifetime.
                if pool_addr is not None:
                    self.task_hold_pool.free(pool_addr)
                if mr_ptr is not None and getattr(self, "engine", None) is not None:
                    try:
                        self.engine.deregister(mr_ptr)
                    except Exception as e:
                        logger.debug(
                            "MR deregister (unstored hold) task_id=%s: %s", task_id, e
                        )

    def _mark_hold_reserved_locked(self, task_id, entry) -> None:
        """A TaskResult for this hold goes out: keep it until that pull lands.

        Refreshes the TTL and counts one outstanding result; LRU eviction needs
        ``pending_pulls == 0``, so neither this nor an earlier unpulled sibling
        can lose the hold first. TTL stays the reaper for a dead prefill.
        Caller holds task_embeddings_lock.
        """
        entry.created = time.monotonic()
        entry.pending_pulls += 1
        self.task_embeddings.move_to_end(task_id)

    def _drop_task_embedding_locked(self, task_id, entry) -> None:
        """Pop a hold and tear down its MR. Caller holds task_embeddings_lock."""
        if self.task_embeddings.pop(task_id, None) is not None:
            self.task_embeddings_bytes -= entry.size_bytes
        if entry.pool_addr is not None:
            # refs == 0 here (callers only drop unreferenced holds), so no
            # transfer can still be reading this region.
            entry.embedding = None
            self.task_hold_pool.free(entry.pool_addr)
            entry.pool_addr = None
        if entry.mr_ptr is not None and getattr(self, "engine", None) is not None:
            try:
                self.engine.deregister(entry.mr_ptr)
            except Exception as e:
                logger.debug("MR deregister for task_id=%s returned %s", task_id, e)
            entry.mr_ptr = None

    def init_task_hold_pool(self) -> None:
        """Allocate and register the factory hold pool once, at producer start.

        Mirrors K3's ScatterMemPool: a single pinned buffer sized to the hold
        cap is registered with mooncake up front, so holds never pin/register
        per embedding (the per-hold register is what produced "overlapped
        memory region" errors). Without mooncake or a cap, holds keep the old
        per-hold path.
        """
        if (
            not self.use_mooncake
            or getattr(self, "engine", None) is None
            or self.task_embeddings_max_bytes <= 0
            or self.task_hold_pool is not None
        ):
            return
        from sglang.srt.disaggregation.gpu_buffer_pool import GpuBufferPool

        page = envs.SGLANG_ENCODER_TASK_HOLD_PAGE_KB.get() * 1024
        if page <= 0 or self.task_embeddings_max_bytes < page:
            # Too small for even one page: keep the per-hold path instead of
            # crashing the encoder at startup (GpuBufferPool would raise).
            logger.warning(
                "Factory hold pool disabled: cap %d B < page %d B; holds use "
                "the per-embedding register path",
                self.task_embeddings_max_bytes,
                page,
            )
            return
        self._check_hold_pool_budget()
        try:
            # zero=False: every region is fully written before it is handed
            # out, so the memset only costs startup time (tens of seconds at
            # this size) and first-touch memory.
            pool = GpuBufferPool(
                self.task_embeddings_max_bytes,
                page,
                device="cpu",
                pin_memory=True,
                zero=False,
            )
            self.engine.register(pool.data_ptr, pool.total_bytes)
        except Exception as error:
            # Fail fast and leave no half-initialized pool: running without it
            # would fall back to per-hold pin + register (unbounded host memory,
            # "overlapped memory region"). A transient failure recovers on the
            # container restart; a misconfiguration is reported by
            # _check_hold_pool_budget before we get here.
            self.task_hold_pool = None
            raise RuntimeError(
                "Failed to allocate/register the factory hold pool of "
                f"{self.task_embeddings_max_bytes / 1024**3:.1f} GB pinned "
                f"(SGLANG_ENCODER_TASK_EMB_CACHE_GB={envs.SGLANG_ENCODER_TASK_EMB_CACHE_GB.get()}); "
                f"check the container memory limit: {error}"
            ) from error
        self.task_hold_pool = pool
        logger.info(
            "Factory hold pool: registered %.1f GB pinned (%d pages x %d KB)",
            pool.total_bytes / 1024**3,
            pool.num_pages,
            page // 1024,
        )

    def _check_hold_pool_budget(self) -> None:
        """Refuse a hold pool that cannot fit the container's memory limit.

        The failure would otherwise surface as a memset of tens of seconds
        followed by an OOM kill / crash loop, with no hint about which env var
        is too large. Restarting cannot fix a misconfiguration, so say what is
        wrong instead; a genuinely transient shortage still recovers on restart.
        """
        limit = _cgroup_memory_limit_bytes()
        if limit <= 0:
            return
        vlm_cache = int(os.environ.get("SGLANG_VLM_CACHE_SIZE_MB", "0")) * 1024 * 1024
        # Leave room for the model loader peaks and the Python runtime; the
        # pool is resident from startup, the cache grows on demand.
        needed = self.task_embeddings_max_bytes + vlm_cache
        if needed <= limit * 0.9:
            return
        raise RuntimeError(
            "Factory hold pool + content cache do not fit the container memory "
            f"limit: pool {self.task_embeddings_max_bytes / 1024**3:.1f} GB "
            f"(SGLANG_ENCODER_TASK_EMB_CACHE_GB={envs.SGLANG_ENCODER_TASK_EMB_CACHE_GB.get()}) "
            f"+ SGLANG_VLM_CACHE_SIZE_MB={vlm_cache // 1024**2} MB vs "
            f"{limit / 1024**3:.1f} GB limit. Lower the pool and/or the cache, "
            "or raise the container memory limit."
        )

    def _alloc_from_hold_pool_locked(self, embedding):
        """Reserve pool space for ``embedding``: ``(addr, view)`` or ``(None, None)``.

        LRU-evicts pulled, unreferenced holds until the region fits; when only
        in-flight or never-pulled holds remain, the caller falls back to a
        per-hold buffer. Caller holds task_embeddings_lock.
        """
        pool = self.task_hold_pool
        nbytes = int(embedding.nbytes)
        if nbytes <= 0 or nbytes > pool.total_bytes:
            return None, None
        addr = pool.try_allocate(nbytes)
        while addr is None:
            victim = next(
                (
                    (tid, e)
                    for tid, e in self.task_embeddings.items()
                    if (
                        e.refs <= 0
                        and e.pulled_once
                        and e.pending_pulls <= 0
                        and e.pool_addr is not None
                    )
                ),
                None,
            )
            if victim is None:
                logger.warning(
                    "hold pool full (%d bytes needed, %d holds in flight or "
                    "unpulled); failing this task for retry",
                    nbytes,
                    len(self.task_embeddings),
                )
                return None, None
            self._drop_task_embedding_locked(*victim)
            addr = pool.try_allocate(nbytes)
        offset = addr - pool.data_ptr
        view = (
            pool._buf[offset : offset + nbytes]
            .view(embedding.dtype)
            .view(embedding.shape)
        )
        return addr, view

    def _evict_to_fit_locked(self, needed_bytes: int) -> None:
        """LRU-evict unreferenced holds until ``needed_bytes`` fits under the cap.

        Mirrors K3's ``MultiModalStaticCache._evict_to_fit``: walk the holds in
        LRU order (oldest first) and drop those with no in-flight pull
        (``refs <= 0``) until there is room. Referenced holds are never evicted
        so an in-flight RDMA transfer cannot have its MR torn down; if only
        referenced holds remain, the insert is allowed to run slightly over the
        cap rather than block, and the next store reclaims once they release.

        Only holds that have already served a pull (``pulled_once``) are
        eligible: unlike K3's cache we cannot recompute at pull time, so a
        never-pulled hold must survive to its first pull (TTL is its only
        reaper), otherwise eviction would reintroduce the 404 this cache fixes.
        """
        cap = self.task_embeddings_max_bytes
        if cap <= 0 or self.task_embeddings_bytes + needed_bytes <= cap:
            return
        target = self.task_embeddings_bytes + needed_bytes - cap
        freed = 0
        victims = []
        for tid, entry in self.task_embeddings.items():
            if entry.refs <= 0 and entry.pulled_once and entry.pending_pulls <= 0:
                victims.append((tid, entry))
                freed += entry.size_bytes
                if freed >= target:
                    break
        for tid, entry in victims:
            logger.debug(
                "LRU-evicting factory embedding task_id=%s size=%d", tid, entry.size_bytes
            )
            self._drop_task_embedding_locked(tid, entry)
        if freed < target:
            logger.warning(
                "task embedding cache over budget: used=%d need=%d cap=%d "
                "(remaining holds are in-flight or awaiting first pull); "
                "admitting anyway",
                self.task_embeddings_bytes,
                needed_bytes,
                cap,
            )

    def _evict_stale_task_embeddings_locked(self) -> None:
        """Drop refs==0 holds older than the TTL. Caller holds the lock."""
        ttl = envs.SGLANG_ENCODER_TASK_EMB_TTL_S.get()
        if ttl <= 0:
            return
        wait = envs.SGLANG_ENCODER_RECV_TIMEOUT.get()
        if ttl < 1.5 * wait and not getattr(self, "_ttl_warned", False):
            # prefill only takes a ref once its pull starts, so a hold reaped
            # between "TaskResult sent" and "pull started" becomes a 404.
            self._ttl_warned = True
            logger.warning(
                "SGLANG_ENCODER_TASK_EMB_TTL_S=%.0fs is close to the prefill "
                "wait timeout %.0fs; unpulled holds may be reclaimed before a "
                "slow prefill pulls them",
                ttl,
                wait,
            )
        now = time.monotonic()
        stale = [
            tid
            for tid, e in self.task_embeddings.items()
            if e.refs <= 0 and (now - e.created) > ttl
        ]
        for tid in stale:
            logger.warning("Evicting un-pulled factory embedding task_id=%s", tid)
            self._drop_task_embedding_locked(tid, self.task_embeddings[tid])

    def attach_metrics(self, collector: FactoryEncoderMetricsCollector) -> None:
        """Enable 厂内 dashboard metrics, incl. the loader / process queues."""
        self.metrics_collector = collector
        self.preprocessor.io_executor.stage = collector.loader_stage
        self.preprocessor.preproc_executor.stage = collector.process_stage

    async def acquire_task_embedding(
        self, task_id: str
    ) -> Optional[WorkflowTaskEmbedding]:
        """Take a ref on a held embedding so it survives an in-flight pull."""
        async with self.task_embeddings_lock:
            entry = self.task_embeddings.get(task_id)
            if entry is not None:
                entry.refs += 1
                entry.pending_pulls = max(0, entry.pending_pulls - 1)
                # This hold has now served at least one pull, so it is safe to
                # LRU-evict later (a future sibling re-submits an encode and
                # re-stores). A never-pulled hold is left for TTL only, so
                # eviction can never race a first pull into a 404.
                entry.pulled_once = True
                # Most-recently-used: keep hot (shared) images from being the
                # first LRU eviction victims.
                self.task_embeddings.move_to_end(task_id)
            return entry

    async def release_task_embedding(self, task_id: str) -> None:
        """Drop a pull ref, but retain the hold for reuse.

        The hold is content-addressed (task_id == md5(model_id + media_url)),
        so several requests that reference the same image pull the same slot.
        Dropping it on the first release let a sibling's pull hit a 404; instead
        keep it and let TTL eviction (``_evict_stale_task_embeddings_locked``)
        reclaim it once it has sat idle long enough.
        """
        async with self.task_embeddings_lock:
            entry = self.task_embeddings.get(task_id)
            if entry is None:
                return
            entry.refs -= 1

    def _advertised_http_port(self) -> int:
        """Port prefill dials for this encoder; must match EncoderAddr.http_port.

        mooncake pulls go to ``/epd/transfer`` on the HTTP port, zmq pulls to
        the PullZmqServer port (see ``TaskProcessor``).
        """
        if self.use_mooncake:
            return int(self.server_args.port)
        return int(getattr(self, "pull_zmq_port", 0) or 0)

    async def list_held_embeddings(self) -> Dict[str, Any]:
        """Snapshot the factory hold cache in K3's ``/list_vit_caches`` format.

        The vit cache collector parses this verbatim, so the top-level
        ``model_id`` and the per-entry EncoderAddr-shaped fields (and an
        ``http_port`` that prefill can actually dial) must match K3's
        ``VitCacheHandler.list_vit_caches``. Hold bookkeeping is appended as
        extra keys for operators.
        """
        from sglang.srt.disaggregation.encode_utils import normalize_transfer_type

        model_id = os.environ.get("MODEL_ID", "")
        transfer_type = normalize_transfer_type(
            get_disagg().encoder_transfer_backend
        )
        ip = os.environ.get("POD_IP") or getattr(self, "local_ip", None)
        http_port = self._advertised_http_port()
        now = time.monotonic()
        caches = []
        async with self.task_embeddings_lock:
            for task_id, entry in self.task_embeddings.items():
                emb = entry.embedding
                shape = list(emb.shape) if emb is not None else []
                caches.append(
                    {
                        "ip": ip,
                        "http_port": http_port,
                        "transfer_type": transfer_type,
                        "task_id": task_id,
                        "model_id": model_id,
                        "embedding_size": entry.size_bytes,
                        "embedding_len": shape[0] if shape else 0,
                        "embedding_dim": shape[1] if len(shape) > 1 else 0,
                        "embedding_dtype": str(getattr(emb, "dtype", "")).replace(
                            "torch.", ""
                        ),
                        "img_grid_thw": entry.image_grid_thw,
                        "original_image_sizes": entry.original_image_sizes,
                        "rdma_port": 0,
                        "item_addr": 0,
                        "cache_size": 0,
                        "header_size": 0,
                        "mooncake_session_id": "",
                        "modality": entry.modality,
                        "chunk_timestamps": entry.chunk_timestamps,
                        "refs": entry.refs,
                        "pulled_once": entry.pulled_once,
                        "pending_pulls": entry.pending_pulls,
                        "age_s": round(now - entry.created, 3),
                    }
                )
            total_bytes = self.task_embeddings_bytes
            cap_bytes = self.task_embeddings_max_bytes
        return {
            "model_id": model_id,
            "caches": caches,
            "count": len(caches),
            "total_bytes": total_bytes,
            "cap_bytes": cap_bytes,
            "cache_hits": getattr(self, "task_embedding_hits", 0),
        }

    async def flush_task_embeddings(self) -> Dict[str, int]:
        """Drop every hold no pull is using; in-flight ones are kept."""
        async with self.task_embeddings_lock:
            before_bytes = self.task_embeddings_bytes
            before_entries = len(self.task_embeddings)
            for tid, entry in list(self.task_embeddings.items()):
                # A hold whose TaskResult is already on its way to vit-server
                # must survive, exactly as it does for LRU eviction.
                if entry.refs <= 0 and entry.pending_pulls <= 0:
                    self._drop_task_embedding_locked(tid, entry)
            after_bytes = self.task_embeddings_bytes
            after_entries = len(self.task_embeddings)
        logger.info(
            "[flush_cache] flushed %d entries, %d bytes; remaining %d entries",
            before_entries - after_entries,
            before_bytes - after_bytes,
            after_entries,
        )
        return {
            "flushed_bytes": before_bytes - after_bytes,
            "flushed_entries": before_entries - after_entries,
            "remaining_bytes": after_bytes,
            "remaining_entries": after_entries,
        }

    async def transfer_task_embedding(
        self, task_id: str, session_id: str, buffer_address: int, byte_size: int
    ) -> None:
        """RDMA-write a held embedding into prefill's buffer (mooncake pull).

        The factory ``/epd/transfer`` is pull-triggered-push: prefill hands its
        own ``session_id``/``buffer_address`` and the encoder writes the rows it
        is holding for ``task_id`` there, reusing the same mooncake engine and
        transfer primitive as the New EPD push path. The source MR was
        registered at ``store_task_embedding`` time.
        """
        entry = await self.acquire_task_embedding(task_id)
        metrics = self.metrics_collector
        if entry is None:
            if metrics is not None:
                metrics.record_cache_miss(int(byte_size), source="transfer")
            raise MMError(
                f"embedding not held for task_id={task_id}",
                code=HTTPStatus.NOT_FOUND,
            )
        try:
            embedding = entry.embedding
            size = int(byte_size) or embedding.nbytes
            if size != embedding.nbytes:
                # The pool is registered as one MR, so an oversized read would
                # silently ship a neighbouring hold's rows instead of failing.
                raise BadRequestError(
                    f"byte_size {size} != held embedding {embedding.nbytes} bytes "
                    f"for task_id={task_id}"
                )
            if metrics is not None:
                metrics.record_cache_hit(size, source="transfer")
            transfer_start = time.perf_counter()
            await self._run_mooncake_transfer(
                session_id, embedding.data_ptr(), int(buffer_address), size
            )
            transfer_s = time.perf_counter() - transfer_start
            if metrics is not None:
                metrics.transfer_stage.observe_batch(transfer_s, 1, size)
            if encoder_metrics_collector is not None:
                encoder_metrics_collector.observe_transfer(
                    transfer_s, backend="mooncake"
                )
            digest = (
                compute_embed_digest(embedding)
                if envs.SGLANG_EMBED_DIGEST.get()
                else None
            )
            return transfer_s * 1000.0, digest
        finally:
            await self.release_task_embedding(task_id)

    def _record_workflow_cache(
        self, hit: bool, nbytes: int, entry: Optional[WorkflowTaskEmbedding] = None
    ) -> None:
        """Count a task served from its hold as a workflow cache hit.

        Only hits: a miss falls through to ``encode``, whose prefix mm_cache
        lookup already records the hit/miss for that request.
        """
        collector = getattr(self, "metrics_collector", None)
        if collector is None:
            return
        if hit:
            collector.record_cache_hit(nbytes, source="workflow")
        else:
            collector.record_cache_miss(nbytes, source="workflow")
        if encoder_metrics_collector is not None:
            # Same hit/miss on the New EPD series (encoder_cache_hit_tokens/
            # files_total), counted in rows and in images.
            rows = nbytes // max(1, self._task_row_bytes())
            # Same labels as the prefix path, where video frames are images;
            # a held video counts one file per frame (one grid row each).
            grids = entry.image_grid_thw if entry is not None else None
            files = len(grids) if isinstance(grids, (list, tuple)) and grids else 1
            encoder_metrics_collector.record_cache_tokens(
                rows if hit else 0, rows, modality="image"
            )
            encoder_metrics_collector.record_cache_files(
                files if hit else 0, files, modality="image"
            )
            encoder_metrics_collector.set_cache_state(
                self.task_embeddings_bytes, len(self.task_embeddings)
            )

    def _task_row_bytes(self) -> int:
        """Bytes of one embedding row, from any held entry (0 if none)."""
        for entry in self.task_embeddings.values():
            emb = entry.embedding
            if emb is not None and emb.dim() == 2 and emb.shape[0] > 0:
                return emb.shape[1] * emb.element_size()
        return 0

    async def _reuse_held_task_embedding(
        self, task_id: str
    ) -> Optional[WorkflowTaskEmbedding]:
        """Return the hold for ``task_id`` (refreshing its TTL), or None.

        This is the --enable-prefix-mm-cache lookup of the factory path, keyed
        by task_id like K3's ``mm_cache.get(task_id)``: the New EPD prefix cache
        is bypassed for mooncake transfers, and task_id is content-addressed
        (md5(model_id + media_url)), so a held entry is the same embedding.
        """
        if not get_mm().enable_prefix_mm_cache:
            return None
        async with self.task_embeddings_lock:
            self._evict_stale_task_embeddings_locked()
            entry = self.task_embeddings.get(task_id)
            if entry is None:
                return None
            self._mark_hold_reserved_locked(task_id, entry)
            self.task_embedding_hits += 1
            return entry

    async def encode_video_and_store_for_workflow(
        self, video, req_id, task_id, sample_fps: Optional[float]
    ):
        """Encode one factory video task as sampled frames (K3 video protocol).

        For vision towers that only take images, each sampled frame is a chunk:
        the held embedding concatenates the frame spans in time order, the grid
        has one row per frame and ``chunk_timestamps`` one ``MM:SS`` per frame.
        Returns the image tuple of ``encode_and_store_for_workflow`` plus the
        timestamps.
        """
        from sglang.srt.disaggregation.encoder.video_frames import (
            decode_video_frames,
            resolve_sample_fps,
        )

        held = await self._reuse_held_task_embedding(task_id)
        if held is not None and held.chunk_timestamps:
            logger.info(
                "[encode] Cache HIT for video task_id=%s frames=%d",
                task_id,
                len(held.chunk_timestamps),
            )
            self._record_workflow_cache(hit=True, nbytes=held.size_bytes, entry=held)
            return (
                held.embedding.shape[0],
                held.embedding.shape[1],
                held.embedding,
                held.image_grid_thw,
                held.original_image_sizes,
                held.chunk_timestamps,
            )
        try:
            fps = resolve_sample_fps(
                sample_fps, envs.SGLANG_ENCODER_VIDEO_DEFAULT_FPS.get()
            )
        except (TypeError, ValueError) as e:
            raise BadRequestError(f"Invalid sample_fps {sample_fps!r}: {e}")
        decode_start = time.perf_counter()
        try:
            frames, timestamps = await asyncio.get_running_loop().run_in_executor(
                self.executor,
                decode_video_frames,
                video,
                fps,
                envs.SGLANG_ENCODER_VIDEO_MAX_FRAMES.get(),
                envs.SGLANG_ENCODER_VIDEO_FRAME_MAX_PIXELS.get(),
                envs.SGLANG_ENCODER_VIDEO_GPU_DECODE.get(),
                self.gpu_id,
            )
        except Exception as e:
            raise BadRequestError(f"Failed to decode video for task {task_id}: {e}")
        self.task_time_costs.setdefault(task_id, {})["decode_time_cost"] = round(
            time.perf_counter() - decode_start, 4
        )
        result = await self.encode_and_store_for_workflow(
            frames,
            req_id,
            task_id,
            Modality.IMAGE,
            store_modality="video",
            chunk_timestamps=timestamps,
        )
        return (*result, timestamps)

    async def encode_and_store_for_workflow(
        self,
        mm_items,
        req_id,
        task_id,
        modality: Modality,
        store_modality: Optional[str] = None,
        chunk_timestamps: Optional[List[str]] = None,
    ):
        """Encode one factory task and hold its embedding for pull by task_id.

        Reuses the New EPD ``encode`` compute path, then moves the staged
        embedding out of the push-oriented ReqState into the task-keyed hold so
        the request state can be released immediately. Returns
        ``(embedding_len, embedding_dim, embedding, grid_thw, original_image_sizes)``
        for the caller to build the ``EncoderAddr`` reply.
        """
        held = (
            None
            if chunk_timestamps is not None
            else await self._reuse_held_task_embedding(task_id)
        )
        if held is not None:
            logger.info(
                "[encode] Cache HIT for task_id=%s rows=%d",
                task_id,
                held.embedding.shape[0],
            )
            self._record_workflow_cache(hit=True, nbytes=held.size_bytes, entry=held)
            return (
                held.embedding.shape[0],
                held.embedding.shape[1],
                held.embedding,
                held.image_grid_thw,
                held.original_image_sizes,
            )
        # The same image twice in one request is the same task_id twice; the
        # sequence number keeps each copy's encode state separate.
        self._workflow_encode_seq = getattr(self, "_workflow_encode_seq", 0) + 1
        encode_req_id = f"{req_id}#{task_id}#{self._workflow_encode_seq}"
        try:
            # Align with K3: read the embedding from the encode return (this
            # call's own staged EmbeddingData), NOT from self.req_states[req_id].
            # A multi-image request fans out into several tasks that share one
            # req_id, so each task encodes under its own internal request id.
            # A video task is already a batch of frames; batching it with
            # images would only hold the images behind its frames.
            data = await self._encode_workflow_task(
                mm_items,
                encode_req_id,
                task_id,
                modality,
                batchable=chunk_timestamps is None,
            )
            if data is None or data.embedding is None:
                raise InternalError(f"No embedding staged for task {task_id}")
            embedding = data.embedding
            grid_thw = (
                data.grid_dim.tolist()
                if getattr(data.grid_dim, "tolist", None)
                else data.grid_dim
            )
            original_image_sizes = getattr(data, "original_image_sizes", None)
            await self.store_task_embedding(
                task_id,
                embedding,
                grid_thw,
                original_image_sizes,
                modality=store_modality or modality.name.lower(),
                chunk_timestamps=chunk_timestamps,
            )
            # store_task_embedding may have moved the tensor to CPU/contiguous;
            # read geometry back from the held copy (reserved, so it cannot be
            # evicted before this reply). No ref is taken here.
            held = self.task_embeddings.get(task_id)
            held_embedding = held.embedding if held is not None else embedding
            return (
                held_embedding.shape[0],
                held_embedding.shape[1],
                held_embedding,
                grid_thw,
                original_image_sizes,
            )
        finally:
            # The embedding now lives in task_embeddings; free the push-side
            # ReqState so it does not leak while awaiting the pull.
            await self.release_request(encode_req_id)

    def _factory_batcher(self) -> Optional[FactoryEncodeBatcher]:
        """The factory task batcher, or None with SGLANG_ENCODER_FACTORY_BATCH_SIZE <= 1."""
        max_batch = envs.SGLANG_ENCODER_FACTORY_BATCH_SIZE.get()
        if max_batch <= 1:
            return None
        batcher = getattr(self, "_factory_encode_batcher", None)
        if batcher is None:

            async def release(request: dict) -> None:
                await self.release_request(request["req_id"])

            batcher = FactoryEncodeBatcher(
                self.batch_encode,
                max_batch=max_batch,
                max_wait_s=envs.SGLANG_ENCODER_FACTORY_BATCH_WAIT_MS.get() / 1000.0,
                max_inflight=envs.SGLANG_ENCODER_FACTORY_BATCH_INFLIGHT.get(),
                release=release,
            )
            self._factory_encode_batcher = batcher
        return batcher

    async def _encode_workflow_task(
        self, mm_items, req_id: str, task_id, modality: Modality, *, batchable=True
    ) -> Optional[EmbeddingData]:
        """Encode one factory task; returns its staged EmbeddingData."""
        batcher = self._factory_batcher() if batchable else None
        if batcher is not None:
            return await batcher.submit(
                {
                    "req_id": req_id,
                    "num_parts": 1,
                    "part_idx": 0,
                    "mm_items": mm_items,
                    "hashes": None,
                    "task_id": task_id,
                },
                modality,
            )
        staged: List[EmbeddingData] = []
        result = await self.encode(
            mm_items=mm_items,
            modality=modality,
            req_id=req_id,
            num_parts=1,
            part_idx=0,
            out_staged=staged,
            task_id=task_id,
        )
        if not staged and result is not None:
            # Keep the encoder's own error and code (400 for a bad image), as
            # K3 does, instead of a generic 500 "No embedding staged".
            raise _error_from_encode_result(result, f"task {task_id}")
        return staged[0] if staged else None

    async def send_to_destination(
        self, state: ReqState, destination: SendDestination
    ) -> None:
        async with state.lifecycle_condition:
            if (
                self.req_states.get(state.req_id) is not state
                or state.release_requested
            ):
                raise InternalError(f"Encoder request was released: {state.req_id}")
            state.active_sends += 1
        try:
            await self.delivery.send(state, destination)
        finally:
            async with state.lifecycle_condition:
                state.active_sends -= 1
                state.lifecycle_condition.notify_all()

    async def release_request(
        self, req_id: str, *, preserve_metadata: bool = False
    ) -> None:
        """Release backend resources, then the embedding, through one path."""
        state = self.req_states.get(req_id)
        if state is None:
            if not preserve_metadata:
                await meta_registry.discard(req_id)
            return
        async with state.lifecycle_condition:
            state.release_requested = True
            state.preserve_metadata_on_release |= preserve_metadata
            encode_is_active = state.active_encodes > 0

        # ``send_with_url`` may be waiting for a destination that will never
        # arrive after its HTTP caller disappears. Wake it so the worker slot
        # is retired together with the staged embedding.
        await _notify_receive_waiters(req_id)
        if encode_is_active:
            return

        async with state.lifecycle_condition:
            await state.lifecycle_condition.wait_for(lambda: state.active_sends == 0)
            if self.req_states.get(req_id) is not state:
                return
            self.req_states.pop(req_id, None)
        await self.delivery.release(state)
        state.embedding_data = None
        if not state.preserve_metadata_on_release:
            await meta_registry.discard(req_id)

    async def register_embedding_destinations(
        self,
        req_id: str,
        expected_destination_count: int,
        destination_urls: Iterable[str],
    ) -> None:
        state = self.req_states.get(req_id)
        if state is None:
            # registration can beat /encode or its queued batch; only encode creates state
            try:
                async with encode_state_condition:
                    await asyncio.wait_for(
                        encode_state_condition.wait_for(
                            lambda: req_id in self.req_states
                        ),
                        timeout=ENCODER_REQ_TIMEOUT,
                    )
            except asyncio.TimeoutError as exc:
                raise MMError(
                    f"Timed out waiting for encoder request to start: {req_id}",
                    code=HTTPStatus.GATEWAY_TIMEOUT,
                ) from exc
            state = self.req_states.get(req_id)
        if state is None:
            raise BadRequestError(f"Encoder request is not active: {req_id}")

        async with state.lifecycle_condition:
            if self.req_states.get(req_id) is not state or state.release_requested:
                raise BadRequestError(f"Encoder request is not active: {req_id}")
            async with rid_lock:
                if req_id not in rid_to_receive_endpoint:
                    rid_to_receive_endpoint[req_id] = set()
                    rid_to_receive_count[req_id] = expected_destination_count
                registered_count = rid_to_receive_count[req_id]
                if registered_count != expected_destination_count:
                    raise BadRequestError(
                        f"Inconsistent receive_count for req_id={req_id}: "
                        f"registered {registered_count}, got {expected_destination_count}"
                    )
                rid_to_receive_endpoint[req_id].update(destination_urls)
            cond = await _get_receive_condition(req_id)
            async with cond:
                cond.notify_all()

    def _infer_embedding_dims(self) -> dict:
        """Infer per-modality embedding dimensions from hf_config at init time."""
        default = self.model_config.hidden_size
        hf_cfg = self.model_config.hf_config
        thinker_cfg = getattr(hf_cfg, "thinker_config", None)
        dims = {
            Modality.IMAGE: default,
            Modality.VIDEO: default,
            Modality.AUDIO: default,
        }

        vision_cfg = getattr(thinker_cfg, "vision_config", None) or getattr(
            hf_cfg, "vision_config", None
        )
        if vision_cfg is not None:
            out_hs = getattr(vision_cfg, "out_hidden_size", None)
            if out_hs is not None:
                ds = getattr(vision_cfg, "deepstack_visual_indexes", None)
                vis_dim = (
                    out_hs * (1 + len(ds))
                    if isinstance(ds, (list, tuple)) and ds
                    else out_hs
                )
                dims[Modality.IMAGE] = vis_dim
                dims[Modality.VIDEO] = vis_dim

        audio_cfg = getattr(thinker_cfg, "audio_config", None) or getattr(
            hf_cfg, "audio_config", None
        )
        if audio_cfg is not None:
            for attr in ("output_dim", "d_model"):
                val = getattr(audio_cfg, attr, None)
                if val and int(val) > 0:
                    dims[Modality.AUDIO] = int(val)
                    break

        logger.info(f"Global cache embedding dims: {dims}")
        return dims

    def slice_embedding(
        self,
        mm_embedding: torch.Tensor,
        token_counts: Iterable[int],
    ) -> List[torch.Tensor]:
        """Slice embeddings using preprocessing-owned token counts."""
        slices, offset = [], 0
        for count in token_counts:
            slices.append(mm_embedding[offset : offset + count])
            offset += count
        if mm_embedding.shape[0] != offset:
            raise InternalError(
                f"Encoder produced {mm_embedding.shape[0]} tokens, but "
                f"preprocessor metadata expected {offset}"
            )
        return slices

    def _calculate_hashes_from_features(
        self, mm_feature, grid_thw: List, modality: Modality, mm_inputs=None
    ) -> List[int]:
        """CPU Task: Compute hashes based on processed feature patches."""
        preprocessed_items = (
            get_encoder_preprocessed_items(mm_inputs) if mm_inputs is not None else None
        )
        if preprocessed_items is not None:
            if len(preprocessed_items) != len(grid_thw):
                raise ValueError(
                    "Encoder preprocess item/grid mismatch: "
                    f"{len(preprocessed_items)} items != {len(grid_thw)} grids"
                )
            hashes = []
            for item in preprocessed_items:
                item.set_pad_value()
                hashes.append(item.hash)
            return hashes

        hashes = []
        if modality == Modality.AUDIO and isinstance(mm_feature, list):
            for feature in mm_feature:
                tmp_item = MultimodalDataItem(modality=modality, feature=feature)
                tmp_item.set_pad_value()
                hashes.append(tmp_item.hash)
            return hashes

        offset = 0
        logger.info(f"{mm_feature.shape=} with {modality=}")
        for grid in grid_thw:
            num_patches = self.preprocessor.get_num_patches(grid, modality)
            feature_slice = mm_feature[offset : offset + num_patches]
            tmp_item = MultimodalDataItem(modality=modality, feature=feature_slice)
            tmp_item.set_pad_value()
            hashes.append(tmp_item.hash)
            offset += num_patches
        return hashes

    def _encode_missing(
        self,
        mm_feature,
        preprocess_result: EncoderPreprocessResult,
        indices: List[int],
        modality: Modality = Modality.IMAGE,
        get_feature_fn=None,
    ) -> List[torch.Tensor]:
        """
        GPU Task: Run ViT inference ONLY on the subset of mm items missing from the cache.
        """
        token_counts = preprocess_result.token_counts
        mm_items = self._build_model_mm_items(
            mm_feature, preprocess_result, indices, modality
        )

        forward_start = time.perf_counter()
        with torch.inference_mode():
            new_embeddings = get_feature_fn(mm_items)
            new_embeddings = _flatten_encoder_embedding(new_embeddings)
            if new_embeddings.ndim != 2:
                new_embeddings = new_embeddings.reshape(-1, new_embeddings.shape[-1])
        if encoder_metrics_collector is not None:
            encoder_metrics_collector.observe_model_forward(
                time.perf_counter() - forward_start, modality=modality.name.lower()
            )

        return self.slice_embedding(new_embeddings, (token_counts[i] for i in indices))

    def _build_model_mm_items(
        self,
        mm_feature,
        preprocess_result: EncoderPreprocessResult,
        indices: List[int],
        modality: Modality,
    ) -> List[MultimodalDataItem]:
        """Build the model-facing items selected for one encoder forward.

        Model preprocessors can preserve an item-wise representation with
        ``EncoderPreprocessOutput``. This avoids concatenating and re-slicing
        features before encoder-DP knows which rank owns each item. Legacy
        processor outputs retain their aggregate tensor behavior.
        """
        mm_inputs = preprocess_result.mm_inputs
        grid_thw = preprocess_result.grid_thw
        preprocessed_items = get_encoder_preprocessed_items(mm_inputs)
        if preprocessed_items is not None:
            if len(preprocessed_items) != len(grid_thw):
                raise ValueError(
                    "Encoder preprocess item/grid mismatch: "
                    f"{len(preprocessed_items)} items != {len(grid_thw)} grids"
                )
            selected = [preprocessed_items[index] for index in indices]
            if any(item.modality != modality for item in selected):
                raise ValueError("Encoder preprocess output contains wrong modality")
            return selected

        split_kimi_k3_images = (
            self.model_type == "kimi_k3" and modality == Modality.IMAGE
        )

        if modality == Modality.AUDIO:
            if isinstance(mm_feature, list):
                sub_feature = [mm_feature[i] for i in indices]
            else:
                sub_feature = mm_feature[list(indices)]
        else:
            feature_slices = []
            offsets = [0]
            curr = 0
            for grid in grid_thw:
                curr += self.preprocessor.get_num_patches(grid, modality)
                offsets.append(curr)
            for idx in indices:
                feature_slices.append(mm_feature[offsets[idx] : offsets[idx + 1]])
            if not split_kimi_k3_images:
                sub_feature = torch.cat(feature_slices, dim=0)

        if split_kimi_k3_images:
            mm_items = [
                MultimodalDataItem.from_dict(
                    {"modality": modality, "feature": _convert(feature)}
                )
                for feature in feature_slices
            ]
        else:
            mm_items = [
                MultimodalDataItem.from_dict(
                    {
                        "modality": modality,
                        "feature": (
                            sub_feature
                            if isinstance(sub_feature, list)
                            else _convert(sub_feature)
                        ),
                    }
                )
            ]

        for key, value in mm_inputs.items():
            if key in _mm_feature_attrs.get(modality, []):
                continue
            value = _convert(value)
            if key in _mm_grid_attrs.get(modality, []):
                if split_kimi_k3_images:
                    for mm_item, idx in zip(mm_items, indices):
                        mm_item.set(key, value[idx : idx + 1])
                else:
                    mm_items[0].set(key, value[indices])
            else:
                for mm_item in mm_items:
                    mm_item.set(key, value)
        return mm_items

    async def _prepare_encode_context(
        self,
        requests: List[dict],
        modality: Modality,
        *,
        use_global_cache: bool,
        is_health_check: bool = False,
    ) -> EncodeContext:
        """Flatten a batch of requests into one EncodeContext (single = N of 1)."""
        modality_str = modality.name.lower()
        preprocess_start = time.perf_counter()
        media_load_seconds.set(0.0)
        try:
            (
                preprocess_result,
                items_per_req,
            ) = await self.preprocessor.process_batch_mm_items(requests, modality)
        except MMError:
            raise
        except NotImplementedError as e:
            raise InternalError(f"Not implemented error: {str(e)}")
        except (TypeError, ValueError) as e:
            raise BadRequestError(f"Failed to process mm items: {str(e)}")
        except Exception as e:
            raise InternalError(f"Failed to process mm items: {str(e)}")

        if len(items_per_req) != len(requests) or any(n <= 0 for n in items_per_req):
            raise InternalError(
                f"Invalid batch layout {items_per_req} for {len(requests)} requests"
            )

        if encoder_metrics_collector is not None and not is_health_check:
            encoder_metrics_collector.observe_preprocess(
                time.perf_counter() - preprocess_start,
                modality=modality_str,
            )
            for item_count in items_per_req:
                encoder_metrics_collector.observe_mm_items_per_request(
                    item_count, modality=modality_str
                )
            encoder_metrics_collector.observe_mm_items_per_batch(
                sum(items_per_req), modality=modality_str
            )
        if self.metrics_collector is not None and not is_health_check:
            load_seconds = media_load_seconds.get()
            num_tokens = sum(preprocess_result.token_counts)
            self.metrics_collector.loader_stage.observe_batch(
                load_seconds, len(requests), num_tokens
            )
            self.metrics_collector.process_stage.observe_batch(
                time.perf_counter() - preprocess_start - load_seconds,
                len(requests),
                num_tokens,
            )
        target = self.model.thinker if hasattr(self.model, "thinker") else self.model
        get_feature_fn = getattr(target, f"get_{modality_str}_feature")

        mm_inputs = preprocess_result.mm_inputs
        grid_thw = preprocess_result.grid_thw
        token_counts = preprocess_result.token_counts
        mm_feature = _convert(_get_mm_feature(mm_inputs, modality))
        num_items = len(grid_thw)
        if num_items != sum(items_per_req):
            raise InternalError(
                f"Batch layout {items_per_req} expects {sum(items_per_req)} "
                f"grids, but the processor produced {num_items}"
            )
        if len(token_counts) != num_items:
            raise InternalError(
                f"Preprocessor returned {len(token_counts)} token counts for "
                f"{num_items} {modality_str} grid entries"
            )

        str_mm_hashes = None
        if use_global_cache:
            # Hashes must be grid-space per request (a leaf-space list would
            # size-mismatch rank>0's mask and deadlock TP); validate on every
            # rank so a bad request fails symmetrically before any collective.
            per_req_hashes = [req.get("hashes") for req in requests]
            mm_hashes = None
            if all(h is not None for h in per_req_hashes):
                for req, hashes, n in zip(requests, per_req_hashes, items_per_req):
                    if len(hashes) != n:
                        raise BadRequestError(
                            f"User-supplied hashes length {len(hashes)} != grid "
                            f"count {n} for req {req['req_id']}; hashes must be "
                            f"grid-space (1 per encoder grid entry)."
                        )
                mm_hashes = [h for hashes in per_req_hashes for h in hashes]
            if self.rank == 0:
                if mm_hashes is None:
                    mm_hashes = self._calculate_hashes_from_features(
                        mm_feature, grid_thw, modality, mm_inputs
                    )
                # Embedding stores use string cache keys.
                str_mm_hashes = [str(h) for h in mm_hashes]

        return EncodeContext(
            req_id=requests[0]["req_id"],
            modality=modality,
            preprocess_result=preprocess_result,
            get_feature_fn=get_feature_fn,
            mm_feature=mm_feature,
            num_items=num_items,
            items_per_req=items_per_req,
            aux_data=_build_mm_aux_data(mm_inputs, self.model_type),
            str_mm_hashes=str_mm_hashes,
            use_global_cache=use_global_cache,
            is_health_check=is_health_check,
        )

    async def _prepare_encode_context_on_all_ranks(
        self,
        requests: List[dict],
        modality: Modality,
        *,
        use_global_cache: bool,
        is_health_check: bool = False,
    ) -> EncodeContext:
        """Prepare one context consistently before TP ranks enter model forward."""
        ctx = None
        local_error = None
        error_phase = 0
        try:
            ctx = await self._prepare_encode_context(
                requests,
                modality,
                use_global_cache=use_global_cache,
                is_health_check=is_health_check,
            )
        except Exception as e:
            local_error = e
            error_phase = 1

        if local_error is None:
            try:
                assert ctx is not None
                await self._publish_preprocess_metadata(ctx, requests)
            except Exception as e:
                local_error = e
                error_phase = 2

        if local_error is None:
            assert ctx is not None
            layout_digest = _preprocess_layout_digest(ctx)
        else:
            layout_digest = (0, 0)
        statuses = self._sync_tp_prepare_status(
            local_error,
            error_phase=error_phase,
            layout_digest=layout_digest,
        )

        expected_layout = tuple(statuses[0][2:].tolist())
        mismatch_rank = next(
            (
                rank
                for rank, rank_status in enumerate(statuses[1:], start=1)
                if tuple(rank_status[2:].tolist()) != expected_layout
            ),
            None,
        )
        if mismatch_rank is not None:
            raise InternalError(
                "Encoder preprocessing produced inconsistent layouts across TP "
                f"ranks 0 and {mismatch_rank}"
            )

        assert ctx is not None
        return ctx

    def _sync_tp_prepare_status(
        self,
        local_error: Optional[Exception],
        *,
        error_phase: int,
        layout_digest: tuple[int, int],
    ) -> List[torch.Tensor]:
        """Raise the same preparation error on every TP rank."""
        tp_group = get_tp_group()
        error_code = (
            int(
                local_error.code
                if isinstance(local_error, MMError)
                else HTTPStatus.INTERNAL_SERVER_ERROR
            )
            if local_error is not None
            else 0
        )
        local_status = torch.tensor(
            [error_code, error_phase, *layout_digest], dtype=torch.int64
        )
        statuses = [torch.empty_like(local_status) for _ in range(tp_group.world_size)]
        if tp_group.world_size > 1:
            torch.distributed.all_gather(
                statuses,
                local_status,
                group=tp_group.cpu_group,
            )
        else:
            statuses[0].copy_(local_status)

        failures = [
            (
                rank,
                int(rank_status[0].item()),
                int(rank_status[1].item()),
            )
            for rank, rank_status in enumerate(statuses)
            if rank_status[0].item() != 0
        ]
        if not failures:
            return statuses

        errors = (
            tp_group.all_gather_object(
                str(local_error) if local_error is not None else None
            )
            if tp_group.world_size > 1
            else [str(local_error)]
        )
        rank, failure_code, failure_phase = next(
            (
                (rank, rank_error_code, rank_error_phase)
                for rank, rank_error_code, rank_error_phase in failures
                if rank_error_code != HTTPStatus.BAD_REQUEST
            ),
            failures[0],
        )
        phase = (
            "Encoder metadata publication"
            if failure_phase == 2
            else "Encoder preprocessing"
        )
        message = f"{phase} failed on TP rank {rank}: {errors[rank]}"
        if failure_code == HTTPStatus.BAD_REQUEST:
            raise BadRequestError(message)
        raise InternalError(message)

    def _broadcast_global_cache_mask(self, mask_tensor: torch.Tensor):
        if get_parallel().tp_size > 1:
            torch.distributed.broadcast(
                mask_tensor,
                src=0,
                group=self.mm_global_cache.prefetch_tp_group,
            )

    async def _lookup_global_cache(
        self,
        ctx: EncodeContext,
    ) -> Tuple[List[int], List[int]]:
        if self.rank == 0:
            try:
                exist_mask = await self.mm_global_cache.batch_is_exist(
                    ctx.str_mm_hashes
                )
            except Exception:
                logger.exception(
                    "Global multimodal cache lookup failed for req %s; "
                    "falling back to ViT",
                    ctx.req_id,
                )
                exist_mask = [False] * ctx.num_items
            mask_tensor = torch.tensor(
                [1 if e else 0 for e in exist_mask], dtype=torch.int32
            )
        else:
            mask_tensor = torch.zeros(ctx.num_items, dtype=torch.int32)

        self._broadcast_global_cache_mask(mask_tensor)

        exist_mask = [m.item() == 1 for m in mask_tensor]
        missing_indices = [i for i, e in enumerate(exist_mask) if not e]
        hit_indices = [i for i, e in enumerate(exist_mask) if e]
        return missing_indices, hit_indices

    def _prefetch_global_cache_hits(
        self,
        ctx: EncodeContext,
        hit_indices: List[int],
    ) -> Tuple[List[str], bool]:
        if self.rank != 0 or not hit_indices:
            return [], False

        hit_hashes = [ctx.str_mm_hashes[i] for i in hit_indices]
        hit_tokens = [ctx.preprocess_result.token_counts[i] for i in hit_indices]
        try:
            self.mm_global_cache.prefetch(
                ctx.req_id, hit_hashes, hit_tokens, ctx.modality
            )
        except Exception:
            logger.exception(
                "Global multimodal cache prefetch failed for req %s; "
                "falling back to ViT",
                ctx.req_id,
            )
            return [], True
        return hit_hashes, False

    async def _wait_global_cache_prefetch(
        self,
        ctx: EncodeContext,
        hit_indices: List[int],
        hit_hashes: List[str],
        prefetch_failed: bool,
    ) -> List[int]:
        fallback_mask = torch.zeros(ctx.num_items, dtype=torch.int32)
        if self.rank == 0 and hit_indices:
            if prefetch_failed:
                for idx in hit_indices:
                    fallback_mask[idx] = 1
            else:
                try:

                    async def _wait_prefetch():
                        while not self.mm_global_cache.check_prefetch_progress(
                            ctx.req_id
                        ):
                            await asyncio.sleep(0.005)

                    await asyncio.wait_for(_wait_prefetch(), timeout=60.0)

                    for i, idx in enumerate(hit_indices):
                        if not self.mm_global_cache.has_local_embedding(hit_hashes[i]):
                            fallback_mask[idx] = 1
                    num_partial_fail = int(fallback_mask.sum().item())
                    if num_partial_fail > 0:
                        logger.warning(
                            f"Req {ctx.req_id}: {num_partial_fail}/{len(hit_indices)} "
                            f"cache-hit items failed to load, falling back to ViT"
                        )
                except Exception as e:
                    logger.error(
                        f"Prefetch failed for req {ctx.req_id}: {e}. "
                        f"Falling back to ViT for {len(hit_indices)} hit items."
                    )
                    for idx in hit_indices:
                        fallback_mask[idx] = 1

        self._broadcast_global_cache_mask(fallback_mask)
        fallback_indices = [
            i for i in range(ctx.num_items) if fallback_mask[i].item() == 1
        ]
        return fallback_indices

    def _stage_global_cache_slices(
        self,
        ctx: EncodeContext,
        indices: List[int],
        slices: List[torch.Tensor],
    ) -> Tuple[List[str], List[Any]]:
        """Stage cache insert data without making cache failure fatal."""
        if not slices:
            return [], []
        hashes = [ctx.str_mm_hashes[i] for i in indices]
        try:
            handles = self.mm_global_cache.store_to_pool_async(
                hashes, slices, ctx.modality
            )
        except Exception:
            logger.exception(
                "Global multimodal cache staging failed for req %s; skipping insert",
                ctx.req_id,
            )
            return [], []
        return hashes, handles

    def _launch_global_cache_insert(
        self,
        ctx: EncodeContext,
        hashes: List[str],
        d2h_handles: List[Any],
    ):
        if not hashes:
            return

        async def _background_insert():
            try:
                await asyncio.to_thread(
                    self.mm_global_cache.wait_store_to_pool,
                    d2h_handles,
                )
                await asyncio.to_thread(
                    self.mm_global_cache.insert_batch,
                    hashes,
                    ctx.modality,
                )
            except Exception:
                logger.exception(
                    "Global multimodal cache insert failed for req %s", ctx.req_id
                )

        self._create_background_task(_background_insert())

    @staticmethod
    def _as_2d_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 2:
            tensor = tensor.reshape(-1, tensor.shape[-1])
        return tensor

    def _assemble_global_cache_cpu(
        self,
        ctx: EncodeContext,
        hit_indices: List[int],
        missing_indices: List[int],
        fallback_indices: List[int],
        new_slices: List[torch.Tensor],
        fallback_slices: List[torch.Tensor],
    ) -> torch.Tensor:
        miss_slice_pos = {idx: pos for pos, idx in enumerate(missing_indices)}
        fallback_slice_pos = {idx: pos for pos, idx in enumerate(fallback_indices)}
        fallback_index_set = set(fallback_indices)
        token_counts = ctx.preprocess_result.token_counts
        dim = self.mm_global_cache.get_embedding_dim(ctx.modality)

        mm_embedding = torch.empty(
            (sum(token_counts), dim),
            dtype=self._embedding_dtype,
            pin_memory=True,
        )

        hit_view_hashes = [
            ctx.str_mm_hashes[idx]
            for idx in hit_indices
            if idx not in fallback_index_set
        ]
        hit_views = {}
        try:
            if hit_view_hashes:
                cached_slice_lists = self.mm_global_cache.get_pool_views(
                    hit_view_hashes
                )
                for h, slices in zip(hit_view_hashes, cached_slice_lists):
                    if slices is None:
                        raise InternalError(
                            f"Cached embedding {h} not available for req {ctx.req_id}"
                        )
                    hit_views[h] = slices

            offset = 0
            for idx, num_tokens in enumerate(token_counts):
                if idx in miss_slice_pos:
                    src = self._as_2d_tensor(new_slices[miss_slice_pos[idx]])
                    mm_embedding[offset : offset + num_tokens].copy_(
                        src, non_blocking=True
                    )
                elif idx in fallback_slice_pos:
                    src = self._as_2d_tensor(fallback_slices[fallback_slice_pos[idx]])
                    mm_embedding[offset : offset + num_tokens].copy_(
                        src, non_blocking=True
                    )
                else:
                    copied = 0
                    for view in hit_views[ctx.str_mm_hashes[idx]]:
                        n = view.shape[0]
                        mm_embedding[offset + copied : offset + copied + n].copy_(view)
                        copied += n
                offset += num_tokens

            torch.cuda.current_stream(self.device).synchronize()
            return mm_embedding
        finally:
            if hit_view_hashes:
                self.mm_global_cache.release_pool_views(hit_view_hashes)

    def _assemble_global_cache_gpu(
        self,
        ctx: EncodeContext,
        missing_indices: List[int],
        fallback_indices: List[int],
        new_slices: List[torch.Tensor],
        fallback_slices: List[torch.Tensor],
    ) -> torch.Tensor:
        miss_slice_pos = {idx: pos for pos, idx in enumerate(missing_indices)}
        fallback_slice_pos = {idx: pos for pos, idx in enumerate(fallback_indices)}
        token_counts = ctx.preprocess_result.token_counts
        embedding_dim = self.mm_global_cache.get_embedding_dim(ctx.modality)
        mm_embedding = torch.empty(
            (sum(token_counts), embedding_dim),
            dtype=self._embedding_dtype,
            device=self.device,
        )

        offset = 0
        copy_handles = []
        for idx, num_tokens in enumerate(token_counts):
            if idx in miss_slice_pos:
                mm_embedding[offset : offset + num_tokens].copy_(
                    new_slices[miss_slice_pos[idx]],
                    non_blocking=True,
                )
            elif idx in fallback_slice_pos:
                mm_embedding[offset : offset + num_tokens].copy_(
                    fallback_slices[fallback_slice_pos[idx]],
                    non_blocking=True,
                )
            else:
                handle = self.mm_global_cache.load_to_device_async(
                    ctx.str_mm_hashes[idx], mm_embedding, offset
                )
                if handle is None:
                    raise InternalError(
                        f"Cached embedding {ctx.str_mm_hashes[idx]} disappeared "
                        f"during assembly for req {ctx.req_id}"
                    )
                copy_handles.append(handle)
            offset += num_tokens

        self.mm_global_cache.wait_load_to_device(copy_handles)
        torch.cuda.current_stream(mm_embedding.device).synchronize()
        return mm_embedding

    async def _compute_global_cache_embedding(
        self,
        ctx: EncodeContext,
        *,
        keep_on_gpu: bool,
    ) -> Optional[torch.Tensor]:
        """Resolve cache hits, compute misses, assemble output, and insert misses."""
        missing_indices, hit_indices = await self._lookup_global_cache(ctx)
        hit_hashes, prefetch_failed = self._prefetch_global_cache_hits(ctx, hit_indices)

        new_slices = []
        if missing_indices:
            new_slices = self._encode_missing(
                ctx.mm_feature,
                ctx.preprocess_result,
                missing_indices,
                ctx.modality,
                ctx.get_feature_fn,
            )

        miss_hashes = []
        miss_d2h_handles = []
        # The CPU output path starts D2H staging before waiting for cache-hit loads.
        if self.rank == 0 and new_slices and not keep_on_gpu:
            miss_hashes, miss_d2h_handles = self._stage_global_cache_slices(
                ctx, missing_indices, new_slices
            )

        fallback_indices = await self._wait_global_cache_prefetch(
            ctx, hit_indices, hit_hashes, prefetch_failed
        )

        fallback_slices = []
        fallback_hashes = []
        fallback_d2h_handles = []
        if fallback_indices:
            logger.info(
                f"Req {ctx.req_id}: All ranks running ViT fallback "
                f"for {len(fallback_indices)} items."
            )
            fallback_slices = self._encode_missing(
                ctx.mm_feature,
                ctx.preprocess_result,
                fallback_indices,
                ctx.modality,
                ctx.get_feature_fn,
            )
            if self.rank == 0 and not keep_on_gpu:
                fallback_hashes, fallback_d2h_handles = self._stage_global_cache_slices(
                    ctx, fallback_indices, fallback_slices
                )

        if self.rank == 0:
            if keep_on_gpu:
                # Start staging newly computed GPU slices into the CPU cache
                # pool asynchronously before assembling the GPU output.
                if new_slices:
                    miss_hashes, miss_d2h_handles = self._stage_global_cache_slices(
                        ctx, missing_indices, new_slices
                    )
                if fallback_slices:
                    fallback_hashes, fallback_d2h_handles = (
                        self._stage_global_cache_slices(
                            ctx, fallback_indices, fallback_slices
                        )
                    )
                mm_embedding = self._assemble_global_cache_gpu(
                    ctx,
                    missing_indices,
                    fallback_indices,
                    new_slices,
                    fallback_slices,
                )
            else:
                mm_embedding = self._assemble_global_cache_cpu(
                    ctx,
                    hit_indices,
                    missing_indices,
                    fallback_indices,
                    new_slices,
                    fallback_slices,
                )

            self._launch_global_cache_insert(
                ctx,
                miss_hashes + fallback_hashes,
                miss_d2h_handles + fallback_d2h_handles,
            )
            return mm_embedding

        return None

    def _forward_executor(self) -> Optional[concurrent.futures.ThreadPoolExecutor]:
        """The vision-forward thread, or None to run the forward on the loop.

        Enabled by SGLANG_ENCODER_FORWARD_IN_THREAD with tp_size == 1 only:
        TP ranks issue their collectives in dispatch order on the loop. One
        worker keeps forwards serialized as they were on the loop.
        """
        executor = getattr(self, "_forward_thread_pool", None)
        if executor is not None:
            return executor
        if not envs.SGLANG_ENCODER_FORWARD_IN_THREAD.get():
            return None
        if get_parallel().tp_size != 1:
            return None
        initializer, initargs = None, ()
        if str(self.device) != "cpu":
            initializer = torch.get_device_module(self.device).set_device
            initargs = (self.gpu_id,)
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="encoder-forward",
            initializer=initializer,
            initargs=initargs,
        )
        self._forward_thread_pool = executor
        return executor

    async def _feature_forward(
        self, get_feature_fn, mm_items, *, to_cpu: bool
    ) -> torch.Tensor:
        """Run the vision forward; returns one ``[tokens, hidden]`` tensor."""

        def forward() -> torch.Tensor:
            with torch.inference_mode():
                embedding = _flatten_encoder_embedding(get_feature_fn(mm_items))
                if to_cpu:
                    embedding = embedding.cpu()
            if embedding.ndim != 2:
                embedding = embedding.reshape(-1, embedding.shape[-1])
            return embedding

        executor = self._forward_executor()
        if executor is None:
            return forward()
        # A cancelled caller still waits: the forward reads the batch's inputs.
        return await await_task_completion_on_cancel(
            asyncio.ensure_future(
                asyncio.get_running_loop().run_in_executor(executor, forward)
            ),
            "Encoder vision forward",
        )

    def _assemble_cached_embedding(
        self, ctx, segments, cached, missing, fresh, *, keep_on_gpu: bool
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Merge cache hits and the fresh forward in request order.

        Returns the flattened embedding and its per-request parts (empty when
        the context has no request segments).
        """
        if len(missing) == len(cached):
            parts = (
                _split_rows(fresh, [segment[2] for segment in segments])
                if segments is not None
                else []
            )
            return fresh, parts
        fresh_parts = iter(())
        if fresh is not None:
            expected = sum(segments[r][2] for r in missing)
            if fresh.shape[0] != expected:
                raise InternalError(
                    f"Encoder produced {fresh.shape[0]} tokens, but "
                    f"preprocessor metadata expected {expected}"
                )
            fresh_parts = iter(_split_rows(fresh, [segments[r][2] for r in missing]))
        parts = [
            next(fresh_parts)
            if hit is None
            # H2D outside the cache lock so other requests are not blocked.
            else (hit.to(self.device) if keep_on_gpu else hit)
            for hit in cached
        ]
        return (torch.cat(parts) if len(parts) > 1 else parts[0]), parts

    async def _compute_direct_embedding(
        self,
        ctx: EncodeContext,
        *,
        keep_on_gpu: bool,
    ) -> torch.Tensor:
        """Compute without global cache, optionally using the prefix MM cache."""
        modality = ctx.modality
        modality_str = modality.name.lower()
        try:
            mm_items = self._build_model_mm_items(
                ctx.mm_feature,
                ctx.preprocess_result,
                list(range(ctx.num_items)),
                modality,
            )

            # mm_cache holds CPU tensors, one entry per request keyed by the
            # combined hash of its items. Mooncake pushes from device memory,
            # so with keep_on_gpu a hit is copied back to the device and a
            # miss stores a CPU copy (bounded by the mm_cache size). A fused
            # batch needs item-wise model inputs to split per request; a
            # legacy aggregate item only has a key for an N=1 context.
            segments = _request_segments(ctx, mm_items)
            use_mm_cache = (
                get_mm().enable_prefix_mm_cache
                and not ctx.is_health_check
                and segments is not None
            )
            cached = [None] * len(ctx.items_per_req)
            req_hashes = [None] * len(ctx.items_per_req)
            if use_mm_cache:
                for mm_item in mm_items:
                    mm_item.set_pad_value()
                async with self.mm_cache_lock:
                    for r, (item_start, item_end, _) in enumerate(segments):
                        mm_hashes = [it.hash for it in mm_items[item_start:item_end]]
                        req_hashes[r] = MultiModalStaticCache.combine_hashes(mm_hashes)
                        entry = self.mm_cache.get(mm_hashes)
                        if entry is not None:
                            cached[r] = entry.embedding
            missing = [r for r, hit in enumerate(cached) if hit is None]

            fresh = None
            if missing:
                forward_items = mm_items
                if len(missing) < len(cached):
                    forward_items = [
                        mm_item
                        for r in missing
                        for mm_item in mm_items[segments[r][0] : segments[r][1]]
                    ]
                forward_start = time.perf_counter()
                fresh = await self._feature_forward(
                    ctx.get_feature_fn, forward_items, to_cpu=not keep_on_gpu
                )
                if encoder_metrics_collector is not None and not ctx.is_health_check:
                    encoder_metrics_collector.observe_model_forward(
                        time.perf_counter() - forward_start, modality=modality_str
                    )
            mm_embedding, parts = self._assemble_cached_embedding(
                ctx, segments, cached, missing, fresh, keep_on_gpu=keep_on_gpu
            )

            try:
                self._validate_embedding_token_count(ctx, mm_embedding)
            except InternalError:
                # Old releases could cache a malformed result before the
                # outer validation ran. Do not make that entry permanently
                # poison every request for the same media.
                hit_hashes = [h for h, c in zip(req_hashes, cached) if c is not None]
                if hit_hashes:
                    async with self.mm_cache_lock:
                        for mm_hash in hit_hashes:
                            self.mm_cache.free(mm_hash, None)
                raise

            if use_mm_cache:
                for (item_start, item_end, _), part, hit in zip(
                    segments, parts, cached
                ):
                    # Per-request cache hit metrics: tokens = embedding rows.
                    num_files = item_end - item_start
                    if encoder_metrics_collector is not None:
                        total_tokens = int(part.shape[0])
                        encoder_metrics_collector.record_cache_tokens(
                            total_tokens if hit is not None else 0,
                            total_tokens,
                            modality=modality_str,
                        )
                        encoder_metrics_collector.record_cache_files(
                            num_files if hit is not None else 0,
                            num_files,
                            modality=modality_str,
                        )
                    if self.metrics_collector is not None:
                        if hit is not None:
                            self.metrics_collector.record_cache_hit(part.nbytes)
                        else:
                            self.metrics_collector.record_cache_miss(part.nbytes)

            if use_mm_cache and missing:
                # Host copies of the fresh parts only (one copy when the whole
                # batch missed); the cache clones the slices it admits.
                if len(missing) == len(parts):
                    host = mm_embedding.cpu()
                    host_parts = _split_rows(host, [part.shape[0] for part in parts])
                else:
                    host_parts = [
                        parts[r].cpu() if r in missing else None
                        for r in range(len(parts))
                    ]
                async with self.mm_cache_lock:
                    entries_before = len(self.mm_cache)
                    added = 0
                    for r in missing:
                        already_present = self.mm_cache.has(req_hashes[r])
                        inserted = self.mm_cache.set(
                            req_hashes[r], EmbeddingResult(embedding=host_parts[r])
                        )
                        added += 0 if already_present else (1 if inserted else 0)
                    entries_after = len(self.mm_cache)
                    if encoder_metrics_collector is not None:
                        evictions = max(0, added - (entries_after - entries_before))
                        if evictions > 0:
                            encoder_metrics_collector.inc_cache_evictions(
                                modality=modality_str, count=evictions
                            )
                        encoder_metrics_collector.set_cache_state(
                            self.mm_cache.current_size, entries_after
                        )

            if (
                not keep_on_gpu
                and modality == Modality.VIDEO
                and ctx.preprocess_result.mm_inputs.get("video_audio_features")
            ):
                target = (
                    self.model.thinker if hasattr(self.model, "thinker") else self.model
                )
                encode_video_audio_fn = getattr(target, "encode_video_audio", None)
                if encode_video_audio_fn is not None:
                    audio_forward_start = time.perf_counter()
                    audio_embedding = encode_video_audio_fn(
                        ctx.preprocess_result.mm_inputs
                    )
                    if (
                        encoder_metrics_collector is not None
                        and not ctx.is_health_check
                    ):
                        encoder_metrics_collector.observe_model_forward(
                            time.perf_counter() - audio_forward_start, modality="audio"
                        )
                    if audio_embedding is not None:
                        ctx.aux_data["video_audio_embedding"] = audio_embedding
                else:
                    logger.warning(
                        "Videos carry audio tracks but model has no "
                        "encode_video_audio; dropping audio for EPD encoding."
                    )

            return mm_embedding
        except BadRequestError as e:
            raise BadRequestError(f"Bad request error: {str(e)}")
        except Exception as e:
            raise InternalError(f"Internal encoding error: {str(e)}")

    async def _compute_embedding(
        self,
        ctx: EncodeContext,
        *,
        keep_on_gpu: bool,
    ) -> Optional[torch.Tensor]:
        """Compute one flattened request with global cache as an optional stage."""
        if ctx.use_global_cache:
            mm_embedding = await self._compute_global_cache_embedding(
                ctx, keep_on_gpu=keep_on_gpu
            )
            if mm_embedding is not None:
                self._validate_embedding_token_count(ctx, mm_embedding)
            return mm_embedding
        return await self._compute_direct_embedding(ctx, keep_on_gpu=keep_on_gpu)

    @staticmethod
    def _validate_embedding_token_count(
        ctx: EncodeContext, mm_embedding: torch.Tensor
    ) -> None:
        expected_tokens = sum(ctx.preprocess_result.token_counts)
        if mm_embedding.shape[0] != expected_tokens:
            raise InternalError(
                f"Encoder produced {mm_embedding.shape[0]} tokens, but "
                f"preprocessor metadata expected {expected_tokens}"
            )

    async def _publish_preprocess_metadata(
        self, ctx: EncodeContext, requests: List[dict]
    ) -> None:
        """Publish each request's size after preprocessing, before model forward."""
        if self.rank != 0:
            return
        embedding_dim = self._embedding_dims[ctx.modality]
        item_offset = 0
        for request, item_count in zip(requests, ctx.items_per_req):
            item_end = item_offset + item_count
            token_count = sum(ctx.preprocess_result.token_counts[item_offset:item_end])
            req_id = request["req_id"]
            state = self._require_active_encode_state(req_id)
            state.embedding_data = EmbeddingData(
                req_id,
                request["num_parts"],
                request["part_idx"],
                ctx.preprocess_result.grid_thw[item_offset:item_end],
                ctx.modality,
                embedding_shape=[token_count, embedding_dim],
                dtype=self._embedding_dtype,
            )
            await meta_registry.publish(
                req_id,
                token_count * embedding_dim * self._element_size,
                token_count,
                embedding_dim,
            )
            item_offset = item_end

    async def _send(
        self,
        embedding: torch.Tensor,
        mm_data: EmbeddingData,
        session_id=None,
        buffer_address=None,
        prefill_host=None,
        embedding_port=None,
        url=None,
    ):
        if get_disagg().encoder_transfer_backend == "mooncake":
            # Encode is synchronous, so mm_data was staged before /encode returned.
            req_id = mm_data.req_id
            if embedding is None:
                raise InternalError(
                    f"No embedding available for Mooncake GPU-direct transfer: {req_id}"
                )

            expected_nbytes = mm_data.shape[0] * mm_data.shape[1] * self._element_size
            assert embedding.nbytes == expected_nbytes, (
                f"Embedding size mismatch for {req_id}: "
                f"actual={embedding.nbytes}, expected={expected_nbytes} "
                f"(shape={mm_data.shape}, element_size={self._element_size})"
            )

            # Fall back to a per-send registration only if the shared one failed.
            mr_already_registered = mm_data._mr_ptr == embedding.data_ptr()
            if not mr_already_registered:
                self.engine.register(embedding.data_ptr(), embedding.nbytes)
            transfer_error = None
            try:
                _t_xfer_start = time.monotonic()
                xfer_ret = await self._run_mooncake_transfer(
                    session_id,
                    embedding.data_ptr(),
                    buffer_address,
                    embedding.nbytes,
                )
            except BaseException as error:
                transfer_error = error
                raise
            finally:
                if not mr_already_registered:
                    try:
                        self.engine.deregister(embedding.data_ptr())
                    except Exception:
                        if transfer_error is None:
                            raise
                        logger.exception(
                            "Per-send MR deregistration also failed for %s; "
                            "preserving the transfer error",
                            req_id,
                        )
            xfer_ms = (time.monotonic() - _t_xfer_start) * 1000.0
            if encoder_metrics_collector is not None:
                encoder_metrics_collector.observe_transfer(
                    xfer_ms / 1000.0, backend="mooncake"
                )
            if self.metrics_collector is not None:
                self.metrics_collector.transfer_stage.observe_batch(
                    xfer_ms / 1000.0, 1, embedding.nbytes
                )
            if xfer_ret < 0:
                raise InternalError(
                    f"Mooncake transfer_sync failed for {req_id} "
                    f"(session={session_id}, nbytes={embedding.nbytes}, "
                    f"ret={xfer_ret})"
                )
            # Emit at INFO for slow transfers or per-send registrations.
            if xfer_ms > 200.0 or not mr_already_registered:
                logger.info(
                    f"[{req_id}] mooncake transfer_sync={xfer_ms:.1f}ms "
                    f"nbytes={embedding.nbytes} shared_mr={mr_already_registered}"
                )

            # Sibling ranks re-read mm_data here; meta_registry owns the release.

        # Send ack/data
        if url is not None:
            endpoint = NetworkAddress.parse(url).to_tcp()
        else:
            endpoint = NetworkAddress(prefill_host, embedding_port).to_tcp()
        logger.info(f"{endpoint = }")

        # Serialize data
        if get_disagg().encoder_transfer_backend == "mooncake":
            # Mooncake already pushed the embedding via RDMA;
            new_mm_data = mm_data.copy_without_embedding()
            serialized_data = pickle.dumps(new_mm_data)
            buffer = None
        else:
            new_mm_data = mm_data.copy_without_embedding()
            if new_mm_data.error_msg is not None:
                buffer = None
                serialized_data = pickle.dumps(new_mm_data)
            else:
                embedding_tensor = TensorWrapper(mm_data.embedding)
                serialized_data = pickle.dumps(new_mm_data)
                buffer = embedding_tensor.__buffer__()

        transfer_start = time.perf_counter()
        if self.transfer_backend == "zmq_to_scheduler" and url is not None:
            lock = self.scheduler_send_locks.get(endpoint)
            if lock is None:
                lock = asyncio.Lock()
                self.scheduler_send_locks[endpoint] = lock

            async with lock:
                sock = self.scheduler_send_sockets.get(endpoint)
                if sock is None:
                    sock = self.context.socket(zmq.PUSH)
                    config_socket(sock, zmq.PUSH)
                    sock.setsockopt(zmq.IMMEDIATE, 1)
                    sock.setsockopt(zmq.SNDTIMEO, int(self.send_timeout * 1000))
                    sock.connect(endpoint)
                    self.scheduler_send_sockets[endpoint] = sock
                try:
                    frames = (
                        [serialized_data, buffer]
                        if buffer is not None
                        else [serialized_data]
                    )
                    tracker = await sock.send_multipart(frames, copy=False, track=True)
                except Exception:
                    if self.scheduler_send_sockets.get(endpoint) is sock:
                        self.scheduler_send_sockets.pop(endpoint, None)
                    sock.close(linger=0)
                    raise

            # MessageTracker.wait() protects the zero-copy source buffer; it
            # is not a receiver acknowledgement. Waiting under the per-peer
            # lock serialized every large embedding on that TCP connection.
            # Queue sends in order under the lock, then wait for buffer
            # ownership independently so libzmq can pipeline the connection.
            try:
                await _await_transfer_completion(
                    asyncio.to_thread(tracker.wait, self.send_timeout),
                    f"ZMQ transfer for req_id={mm_data.req_id}",
                )
            except Exception:
                if self.scheduler_send_sockets.get(endpoint) is sock:
                    self.scheduler_send_sockets.pop(endpoint, None)
                    sock.close(linger=0)
                raise

            if encoder_metrics_collector is not None:
                encoder_metrics_collector.observe_transfer(
                    time.perf_counter() - transfer_start,
                    backend=self.transfer_backend,
                )
            return

        # Per-request sockets remain for zmq_to_tokenizer and legacy direct
        # scheduler sends. Scheduler URL sends use persistent sockets above.
        def send_with_socket():
            sock = self.sync_context.socket(zmq.PUSH)
            config_socket(sock, zmq.PUSH)
            sock.setsockopt(zmq.IMMEDIATE, 1)
            sock.setsockopt(zmq.SNDTIMEO, int(self.send_timeout * 1000))
            try:
                sock.connect(endpoint)
                if buffer is not None:
                    tracker = sock.send_multipart(
                        [serialized_data, buffer], copy=False, track=True
                    )
                else:
                    tracker = sock.send_multipart(
                        [serialized_data], copy=False, track=True
                    )
                tracker.wait(timeout=self.send_timeout)
            finally:
                sock.close(linger=5000)

        await _await_transfer_completion(
            asyncio.get_running_loop().run_in_executor(self.executor, send_with_socket),
            f"ZMQ transfer for req_id={mm_data.req_id}",
        )
        if (
            encoder_metrics_collector is not None
            and get_disagg().encoder_transfer_backend != "mooncake"
        ):
            encoder_metrics_collector.observe_transfer(
                time.perf_counter() - transfer_start,
                backend=get_disagg().encoder_transfer_backend,
            )

    async def _run_mooncake_transfer(
        self,
        session_id,
        source_address: int,
        destination_address: int,
        size: int,
    ) -> int:
        """Keep the send active until its blocking transfer stops using the MR."""
        stage = (
            self.metrics_collector.transfer_stage
            if self.metrics_collector is not None
            else None
        )
        enqueued = time.perf_counter()
        # Whoever takes this first (worker start or cancel) leaves the queue.
        queue_claim = threading.Lock()
        if stage is not None:
            stage.queue_enter()

        def transfer() -> int:
            if stage is not None and queue_claim.acquire(blocking=False):
                stage.queue_leave(wait_seconds=time.perf_counter() - enqueued)
            return self.engine.transfer_sync(
                session_id, source_address, destination_address, size
            )

        try:
            return await _await_transfer_completion(
                asyncio.to_thread(transfer),
                f"Mooncake transfer to session={session_id}",
            )
        finally:
            # Cancelled before a worker thread picked it up.
            if stage is not None and queue_claim.acquire(blocking=False):
                stage.queue_leave()

    def _register_shared_mr(self, mm_data: EmbeddingData, embedding: torch.Tensor):
        """Register one MR shared by every rank's /send; _send re-registers on failure."""
        try:
            self.engine.register(embedding.data_ptr(), embedding.nbytes)
            mm_data._mr_ptr = embedding.data_ptr()
        except Exception as reg_err:
            logger.warning(
                f"Shared-MR register failed for {mm_data.req_id}, "
                f"falling back to per-/send register: {reg_err}"
            )

    def _deregister_shared_mr(self, mm_data: EmbeddingData) -> None:
        if mm_data._mr_ptr is None:
            return
        try:
            self.engine.deregister(mm_data._mr_ptr)
        except Exception as dereg_err:
            logger.warning(
                f"Shared-MR deregister failed for {mm_data.req_id}: {dereg_err}"
            )
        finally:
            mm_data._mr_ptr = None

    def _stage_embeddings(
        self,
        ctx: EncodeContext,
        requests: List[dict],
        mm_embedding: Optional[torch.Tensor],
        *,
        keep_on_gpu: bool,
        out_staged: Optional[List] = None,
    ) -> List[Tuple[int, int, int, Optional[str], Optional[int]]]:
        """Split the fused embedding per request and stage one EmbeddingData each.

        Per-request token ranges are contiguous in flatten order, so each
        staged embedding is a slice of the batch tensor.
        """
        if self.rank != 0:
            return [(0, 0, 0, None, None)] * len(requests)
        if mm_embedding is None:
            raise InternalError(f"Rank 0 produced no embedding for {ctx.req_id}")

        results = []
        staged_embeddings = []
        try:
            item_offset = 0
            token_offset = 0
            for req, num_items in zip(requests, ctx.items_per_req):
                item_end = item_offset + num_items
                num_tokens = sum(
                    ctx.preprocess_result.token_counts[item_offset:item_end]
                )
                embedding = mm_embedding[token_offset : token_offset + num_tokens]
                if keep_on_gpu and len(requests) > 1:
                    # A view would pin the whole batch tensor until the last transfer.
                    embedding = embedding.clone()
                req_aux_data = dict(ctx.aux_data)
                if ctx.aux_data.get("original_image_sizes") is not None:
                    req_aux_data["original_image_sizes"] = ctx.aux_data[
                        "original_image_sizes"
                    ][item_offset:item_end]
                mm_data = EmbeddingData(
                    req["req_id"],
                    req["num_parts"],
                    req["part_idx"],
                    ctx.preprocess_result.grid_thw[item_offset:item_end],
                    ctx.modality,
                    embedding,
                    **req_aux_data,
                )
                # Global-cache embeddings keep registering per /send instead.
                # Factory tasks (task_id) are copied into the pre-registered
                # hold pool, so registering the GPU slice here would be wasted.
                if keep_on_gpu and not ctx.use_global_cache and not req.get("task_id"):
                    self._register_shared_mr(mm_data, embedding)
                staged_embeddings.append(mm_data)
                results.append(
                    (
                        embedding.nbytes,
                        embedding.shape[0],
                        embedding.shape[1],
                        None,
                        None,
                    )
                )
                item_offset = item_end
                token_offset += num_tokens

            # transfer_sync bypasses CUDA streams, so GPU writes (forward and the
            # per-request clones) must land before /send reads the buffers.
            if keep_on_gpu and mm_embedding.is_cuda:
                torch.cuda.current_stream(mm_embedding.device).synchronize()
            self._stage_embedding_batch(staged_embeddings)
            if out_staged is not None:
                # K3-style: hand the per-request EmbeddingData back to the caller
                # so the factory workflow reads the embedding from the return
                # value instead of the shared req_id-keyed ReqState.
                out_staged.extend(staged_embeddings)
            return results
        except BaseException:
            for mm_data in staged_embeddings:
                self._deregister_shared_mr(mm_data)
            raise

    def _stage_errors(
        self, requests: List[dict], modality: Modality, exc: Exception
    ) -> List[Tuple[int, int, int, Optional[str], Optional[int]]]:
        """Stage one error EmbeddingData per request so /send reports the failure."""
        code = (
            exc.code if isinstance(exc, MMError) else HTTPStatus.INTERNAL_SERVER_ERROR
        )
        msg = str(exc)
        logger.error(f"Rank {self.rank} encode failed: {msg} {code = }", exc_info=True)
        if self.rank == 0:
            for req in requests:
                self._stage_embedding(
                    EmbeddingData(
                        req["req_id"],
                        req["num_parts"],
                        req["part_idx"],
                        None,
                        modality,
                        error_msg=msg,
                        error_code=code,
                    )
                )
        return [(0, 0, 0, msg, code)] * len(requests)

    @staticmethod
    def _embeddings_go_to_host(requests: List[dict]) -> bool:
        """True when no device copy of this batch's embedding is ever read.

        Factory tasks (task_id) are held in host memory for prefill to pull,
        and the prefix mm_cache keeps host copies too. Producing the batch on
        the host spares a cache hit its H2D + D2H round trip (a miss trades
        its second D2H for a host copy); without the cache, the device output
        goes straight into the pinned hold pool, so it stays on the device.
        """
        return all(req.get("task_id") for req in requests) and bool(
            get_mm().enable_prefix_mm_cache
        )

    async def batch_encode(
        self, requests: List[dict], modality: Modality, out_staged: Optional[List] = None
    ) -> List[Tuple[int, int, int, Optional[str], Optional[int]]]:
        """Encode requests through one fused pipeline; encode() is the N=1 case.

        Fuse-or-not is EncoderScheduler policy, not an API fork. Health probes
        bypass caches and stage on CPU so completion confirms a model forward.
        """
        states = [self._acquire_encode_ref(req["req_id"]) for req in requests]
        is_health_check = all(
            is_health_check_request(req["req_id"]) for req in requests
        )
        keep_on_gpu = (
            self.use_mooncake
            and not is_health_check
            and not self._embeddings_go_to_host(requests)
        )
        use_global_cache = self.mm_global_cache is not None and not is_health_check
        ready_mark = None
        if self.metrics_collector is not None and not is_health_check:
            ready_mark = StageReadyMark(
                self.metrics_collector.embed_stage, len(requests)
            )
        ready_token = embed_ready.set(ready_mark)
        started_at = time.perf_counter()
        try:
            if self.rank == 0:
                async with encode_state_condition:
                    encode_state_condition.notify_all()
            try:
                ctx = await self._prepare_encode_context_on_all_ranks(
                    requests,
                    modality,
                    use_global_cache=use_global_cache,
                    is_health_check=is_health_check,
                )
            finally:
                embed_ready.reset(ready_token)
            prepared_at = time.perf_counter()
            self._correct_workflow_task_tokens(ctx, requests)
            if ready_mark is not None:
                ready_mark.start()
            embed_start = time.perf_counter()
            mm_embedding = await self._compute_embedding(ctx, keep_on_gpu=keep_on_gpu)
            self._record_workflow_task_timings(
                requests, prepared_at - started_at, time.perf_counter() - prepared_at
            )
            if (
                self.metrics_collector is not None
                and not is_health_check
                and mm_embedding is not None
            ):
                self.metrics_collector.embed_stage.observe_batch(
                    time.perf_counter() - embed_start,
                    len(requests),
                    int(mm_embedding.shape[0]),
                )

            if self.profiler is not None:
                for _ in requests:
                    self.profiler.step()

            return self._stage_embeddings(
                ctx, requests, mm_embedding, keep_on_gpu=keep_on_gpu, out_staged=out_staged
            )
        except Exception as e:
            return self._stage_errors(requests, modality, e)
        finally:
            if ready_mark is not None:
                ready_mark.discard()
            for state in states:
                await self._release_encode_ref(state)

    def _record_workflow_task_timings(self, requests, process_s, embed_s) -> None:
        """Stage costs for the factory TaskResult.time_cost_info (K3 fields).

        process covers media loading + preprocessing, embed the vision forward;
        a fused batch reports the batch's costs for each of its tasks.
        """
        if getattr(self, "encode_workflow", None) is None or self.rank != 0:
            return
        for req in requests:
            task_id = req.get("task_id")
            if task_id:
                costs = self.task_time_costs.setdefault(task_id, {})
                costs["process_time_cost"] = round(process_s, 4)
                costs["embed_time_cost"] = round(embed_s, 4)

    def pop_task_time_costs(self, task_id: str) -> Dict[str, float]:
        """Take the recorded stage costs of ``task_id`` (empty if none)."""
        return self.task_time_costs.pop(task_id, {})

    def _correct_workflow_task_tokens(self, ctx, requests) -> None:
        """Replace vit-server's token estimate with the preprocessed count.

        Mirrors K3: once preprocessing knows each task's real token count, the
        fetch budget (free_tokens) is corrected before the vision forward runs,
        so large images and videos do not leave admission working off a guess.
        """
        workflow = getattr(self, "encode_workflow", None)
        if workflow is None or self.rank != 0:
            return
        token_counts = ctx.preprocess_result.token_counts
        offset = 0
        for req, num_items in zip(requests, ctx.items_per_req):
            task_id = req.get("task_id")
            actual = sum(token_counts[offset : offset + num_items])
            offset += num_items
            if task_id:
                workflow.update_task_tokens(task_id, int(actual))

    async def encode(
        self, mm_items, modality: Modality, req_id, num_parts, part_idx, hashes=None,
        out_staged=None, task_id=None,
    ):
        """Encode one request: the batch-of-1 case of batch_encode.

        ``task_id`` (factory workflow tasks) lets preprocessing correct the
        task's fetch-budget estimate.
        """
        results = await self.batch_encode(
            [
                {
                    "req_id": req_id,
                    "num_parts": num_parts,
                    "part_idx": part_idx,
                    "mm_items": mm_items,
                    "hashes": hashes,
                    "task_id": task_id,
                }
            ],
            modality,
            out_staged=out_staged,
        )
        return results[0]

    async def encode_request(self, req: dict, modality: Modality):
        """Adapt a request dictionary to the single-request encode interface."""
        return await self.encode(
            mm_items=req["mm_items"],
            modality=modality,
            req_id=req["req_id"],
            num_parts=req["num_parts"],
            part_idx=req["part_idx"],
            hashes=req.get("hashes"),
        )

    # For zmq_to_tokenizer zmq_to_scheduler and mooncake
    async def send(
        self, req_id, prefill_host, embedding_port, session_id=None, buffer_address=None
    ):
        state = self.req_states.get(req_id)
        if state is None:
            # False = nothing transferred: callers must not count this send
            # nor report success, or the decoder waits on an ack never coming.
            logger.warning(
                f"MMEncoder.send: no embedding for req_id={req_id} "
                f"(already released or unknown)"
            )
            return False
        await self.send_to_destination(
            state,
            SendDestination.from_host_port(
                prefill_host,
                embedding_port,
                session_id=session_id,
                buffer_address=buffer_address,
            ),
        )
        return True

    # For zmq_to_scheduler
    async def send_with_url(
        self,
        req_id,
    ):
        state = self.req_states.get(req_id)
        if state is None:
            return
        sent_urls: Set[str] = set()
        all_tasks: List[Tuple[asyncio.Task, str]] = []
        start_time = asyncio.get_running_loop().time()
        timeout = self.send_timeout
        cond = await _get_receive_condition(req_id)

        try:
            while True:
                if state.release_requested:
                    break

                async with rid_lock:
                    current_targets = rid_to_receive_endpoint.get(req_id, set()).copy()
                    expected_count = rid_to_receive_count.get(req_id)

                new_targets = current_targets - sent_urls

                if new_targets:
                    logger.info(
                        f"Found {len(new_targets)} new endpoints for {req_id}. Starting tasks..."
                    )
                    for url in new_targets:
                        task = asyncio.create_task(
                            self.send_to_destination(
                                state,
                                SendDestination.from_url(url),
                            )
                        )
                        all_tasks.append((task, url))
                        sent_urls.add(url)  # Mark as handled immediately
                if expected_count is not None and len(sent_urls) >= expected_count:
                    logger.info(
                        f"All {expected_count} endpoints initiated for {req_id}. Breaking loop."
                    )
                    break
                remaining = timeout - (asyncio.get_running_loop().time() - start_time)
                if remaining <= 0:
                    logger.error(
                        f"[{req_id}] Timeout! Sent {len(sent_urls)}/{expected_count}"
                    )
                    break

                async with cond:
                    try:
                        await asyncio.wait_for(cond.wait(), timeout=remaining)
                    except asyncio.TimeoutError:
                        continue

            if all_tasks:
                logger.info(
                    f"Loop finished. Awaiting completion of {len(all_tasks)} sending tasks..."
                )
                tasks_only = [t[0] for t in all_tasks]
                results = await asyncio.gather(*tasks_only, return_exceptions=True)

                # Process results and log errors
                for i, result in enumerate(results):
                    url = all_tasks[i][1]  # Retrieve URL associated with the task
                    if isinstance(result, Exception):
                        logger.error(f"Failed to send to {url}: {result}")
                    else:
                        logger.debug(f"Successfully sent to {url}")

            logger.info(f"All tasks completed for req_id: {req_id}")

        finally:
            logger.info(f"Cleaning up resources for req_id {req_id}")
            await self.release_request(req_id)


class EncoderProfiler:
    def __init__(self, rank: int):
        self.rank = rank
        self.profiler = None
        self.steps_left = None
        self.output_dir = None
        self.prefix = None
        self.profile_id = None

    def start(self, obj: ProfileReq):
        if self.profiler is not None:
            return False, "profiling already running"

        output_dir = obj.output_dir or os.getenv("SGLANG_TORCH_PROFILER_DIR", "/tmp")
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.prefix = obj.profile_prefix or "encoder"
        self.profile_id = str(time.time())

        activities = obj.activities or ["CPU", "GPU"]
        torch_activities = []
        if "CPU" in activities:
            torch_activities.append(torch.profiler.ProfilerActivity.CPU)
        if "GPU" in activities:
            torch_activities.append(torch.profiler.ProfilerActivity.CUDA)

        profile_memory = "MEM" in activities
        if not torch_activities and not profile_memory:
            return False, "no supported activities"

        self.profiler = torch.profiler.profile(
            activities=torch_activities,
            with_stack=True if obj.with_stack is None else obj.with_stack,
            record_shapes=False if obj.record_shapes is None else obj.record_shapes,
            profile_memory=profile_memory,
        )
        self.profiler.start()
        self.steps_left = obj.num_steps
        logger.info(
            f"Encoder profiling started. output_dir={self.output_dir} profile_id={self.profile_id}"
        )
        return True, None

    def step(self):
        if self.profiler is None:
            return
        self.profiler.step()
        if self.steps_left is not None:
            self.steps_left -= 1
            if self.steps_left <= 0:
                self.stop()

    def stop(self):
        if self.profiler is None:
            return False, "profiling not running"
        self.profiler.stop()
        filename = f"{self.prefix}-rank{self.rank}-{self.profile_id}.trace.json"
        trace_path = os.path.join(self.output_dir, filename)
        self.profiler.export_chrome_trace(trace_path)
        logger.info("Encoder profiling saved to: %s", trace_path)
        self.profiler = None
        self.steps_left = None
        return True, None


async def run_encoder(
    server_args: ServerArgs, schedule_path, dist_init_method, rank: int
):
    encoder = MMEncoder(server_args, schedule_path, dist_init_method, rank)
    while True:
        request = await async_sock_recv(encoder.schedule_socket)
        await _handle_encoder_worker_request(encoder, request)


async def _handle_encoder_worker_request(encoder: MMEncoder, request):
    if isinstance(request, ProfileReq):
        if request.req_type == ProfileReqType.START_PROFILE:
            if encoder.profiler is None:
                encoder.profiler = EncoderProfiler(encoder.rank)
            encoder.profiler.start(request)
        else:
            encoder.profiler.stop()
    elif isinstance(request, dict) and request.get("type") == "batch_encode":
        await encoder.batch_encode(
            request["requests"],
            Modality.from_str(request["modality"]),
        )
    else:
        # Health-check rids need no special routing: batch_encode derives
        # health semantics from the rid prefix itself.
        await encoder.encode_request(request, Modality.from_str(request["modality"]))


def launch_encoder(server_args, schedule_path, dist_init_method, rank):
    publish(server_args, role="encoder")
    try:
        asyncio.run(run_encoder(server_args, schedule_path, dist_init_method, rank))
    except KeyboardInterrupt:
        logger.info(f"Exit rank {rank}")
    except Exception:
        traceback.print_exc()


# Per-process encoder metrics collector. Set by
# runtime.launch_local_runtime (non-DP) and
# runtime.run_dp_worker (DP mode). None when metrics disabled. Kept
# here because MMEncoder GPU methods reference it directly.
encoder_metrics_collector: Optional[EncoderMetricsCollector] = None
