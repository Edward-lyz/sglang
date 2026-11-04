"""
Encode workflow: fetch tasks from vit-server, encode with MMEncoder, submit results.

Architecture (stream mode via StreamExecutor):
  - Models: Pydantic data contracts (FetchTaskRequest, TaskResult, etc.)
  - VitServerClient: HTTP communication with vit-server
  - EncodeWorkflow: fetch → StreamExecutor → submit lifecycle orchestrator

The VitTaskImpl adapts the async encode pipeline into a StreamingImplProto
that StreamExecutor can drive.
"""

import asyncio
import json
import logging
import os
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any, List, Optional

import aiohttp
from pydantic import BaseModel

from sglang.srt.disaggregation.encode_collector import (
    BatchExecutorCollector,
    VitFetchCollector,
)
from sglang.srt.disaggregation.encode_health import update_encoder_output_timestamp
from sglang.srt.disaggregation.encode_utils import (
    StreamExecutor,
    normalize_transfer_type,
)

logger = logging.getLogger(__name__)


# ==================== Models ====================


class FetchTaskRequest(BaseModel):
    """Request model for fetching encoding tasks from vit-server."""

    model_id: Optional[str] = None
    max_task_num: Optional[int] = None
    running_task_num: Optional[int] = None
    free_tokens: Optional[int] = None
    encoder_name: Optional[str] = None
    transfer_type: str = "zmq"

    # Fields sent by direct clients (not used in polling)
    req_id: Optional[str] = None
    mm_items: Optional[List[str]] = None
    num_parts: Optional[int] = None
    part_idx: Optional[int] = None
    prefill_host: Optional[str] = None
    embedding_port: Optional[int] = None


class MediaItem(BaseModel):
    """Single media item with type and metadata."""

    media_type: str  # "image" or "video"
    media_url: str
    sample_fps: Optional[float] = None
    tokens: int = 0


class MediaInfo(BaseModel):
    """Media information containing URLs and/or media items."""

    media_urls: List[str] = []
    media_items: Optional[List[MediaItem]] = None


class MMInfo(BaseModel):
    """Multimodal information wrapper."""

    media_info: MediaInfo


class TaskItem(BaseModel):
    """Single task item for encoding."""

    task_id: str
    request_id: str
    index: int
    type: str
    tokens: int
    mm_info: MMInfo


class FetchTaskResponse(BaseModel):
    """Response model for fetch_tasks API."""

    status_code: int = 200
    message: str = ""
    tasks: List[TaskItem] = []


class TimeCostInfo(BaseModel):
    """Time cost information for encoding task."""

    total_time: float = 0.0
    queue_time_cost: float = 0.0
    bos_time_cost: float = 0.0
    infer_time_cost: float = 0.0
    load_time_cost: float = 0.0
    decode_time_cost: float = 0.0
    process_time_cost: float = 0.0
    embed_time_cost: float = 0.0


class EncoderAddr(BaseModel):
    """Encoder address and metadata information."""

    ip: str
    http_port: int
    transfer_type: str
    task_id: str
    model_id: str
    embedding_size: int
    embedding_len: int
    embedding_dim: int
    embedding_dtype: str
    img_grid_thw: Optional[List[List[int]]] = None
    # Original [width, height] per media item, before any resize. Prefill and
    # decode both need it to rebuild Kimi-K3's resolution-carrying media
    # framing; other models leave it unset.
    original_image_sizes: Optional[List[List[int]]] = None
    rdma_port: int = 0
    item_addr: int = 0
    cache_size: int = 0
    header_size: int = 0
    mooncake_session_id: str = ""
    modality: str = "image"
    chunk_timestamps: Optional[List[str]] = None


class TaskResult(BaseModel):
    """Result of an encoding task."""

    model_id: str
    task_id: str
    request_id: str
    image_url: str = ""
    status_code: int
    message: str = ""
    token_num: int
    encoder_addr: EncoderAddr
    time_cost_info: Optional[TimeCostInfo] = None


class SubmitResultRequest(BaseModel):
    """Request model for submitting encoding results."""

    model_id: str
    task_results: List[TaskResult]


# ==================== VitServerClient ====================



def _new_epd_metric(method: str, **kwargs) -> None:
    """Also report a task on the New EPD collector (encoder_requests_*), if on.

    The factory path bypasses the /encode handler that normally counts these,
    so the per-encoder QPS and error panels would otherwise stay empty.
    """
    from sglang.srt.disaggregation.encoder import server as server_module

    collector = server_module.encoder_metrics_collector
    if collector is None:
        return
    try:
        getattr(collector, method)(**kwargs)
    except Exception:
        # Metrics must never decide whether a task produces its TaskResult.
        logger.warning("encoder metric %s failed", method, exc_info=True)

@dataclass
class FetchResult:
    """Result of a fetch_tasks call."""

    success: bool
    tasks: list[dict[str, Any]] = field(default_factory=list)
    message: str = ""


@dataclass
class PendingTaskTokenStat:
    """Per-copy reserved token amounts for one task_id (one list entry per
    in-flight fetch/copy).

    Accounting is strictly per copy — each copy is added on fetch, corrected once
    when its actual size is known, and subtracted on its own completion — so the
    invariant ``_pipeline_pending_tokens == sum(all entries across all tasks)``
    always holds. Duplicates and any interleaving of fetch/correct/done stay
    balanced without scaling a shared price by an outstanding count.
    """

    reserved: list = field(default_factory=list)

    @property
    def count(self) -> int:
        """Number of in-flight copies (reserved entries) for this task_id."""
        return len(self.reserved)


class VitServerClient:
    """HTTP client for vit-server /fetch_task and /submit_result endpoints.

    Reuses a single aiohttp.ClientSession for connection pooling.
    """

    def __init__(self, vit_server_url: str, timeout: float = 30.0):
        """Bind the client to one vit-server base URL and request timeout."""
        self._base_url = vit_server_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None
        # Body-level status of the latest /fetch_task reply; vit-server can answer
        # HTTP 200 with a non-200 ``status_code`` in the body.
        self.last_body_status: Optional[int] = None
        self.last_body_message: Optional[str] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Return the shared aiohttp session, recreating it if closed."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def close(self):
        """Close the underlying HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def fetch_tasks(self, request: FetchTaskRequest) -> FetchResult:
        """POST /fetch_task."""
        url = f"{self._base_url}/fetch_task"
        payload = {
            "model_id": request.model_id,
            "max_task_num": request.max_task_num,
            "running_task_num": request.running_task_num,
            "free_tokens": request.free_tokens,
            "encoder_name": request.encoder_name,
            "transfer_type": request.transfer_type,
        }
        try:
            session = await self._get_session()
            logger.debug(f"Fetching tasks from {url}, payload: {payload}")
            async with session.post(url, json=payload) as resp:
                text = await resp.text()
                try:
                    data = json.loads(text)
                except (json.JSONDecodeError, ValueError):
                    data = {"raw_response": text}

                if resp.status == HTTPStatus.OK:
                    self.last_body_status = data.get("status_code")
                    self.last_body_message = data.get("message")
                    return FetchResult(
                        success=True,
                        tasks=data.get("tasks", []),
                    )

                logger.error(
                    f"Fetch failed: status={resp.status}, " f"response={text[:500]}"
                )
                return FetchResult(
                    success=False,
                    message=data.get("message", f"vit-server error: {text[:200]}"),
                )
        except aiohttp.ClientError as e:
            logger.error(f"Failed to connect to vit-server: {e}")
            return FetchResult(success=False, message=str(e))
        except Exception as e:
            logger.error(f"Unexpected error fetching tasks: {e}")
            return FetchResult(success=False, message=str(e))

    async def submit_results(self, request: SubmitResultRequest) -> tuple[int, str]:
        """POST /submit_result. Returns (status_code, message)."""
        url = f"{self._base_url}/submit_result"
        try:
            session = await self._get_session()
            async with session.post(url, json=request.model_dump()) as resp:
                response_text = await resp.text()
                try:
                    body = json.loads(response_text)
                    message = (
                        body.get("message")
                        or body.get("error")
                        or body.get("detail")
                        or str(body)
                    )
                except Exception:
                    message = response_text or "<empty response>"
                return resp.status, message
        except Exception as e:
            logger.error(f"Failed to submit results to vit-server: {e}")
            return HTTPStatus.INTERNAL_SERVER_ERROR, str(e)


# ==================== TaskProcessor ====================


class TaskProcessor:
    """Processes a single vit-server task using an MMEncoder."""

    def __init__(self, encoder):
        """Bind the processor to the MMEncoder it drives."""
        self._encoder = encoder
        self._pod_ip = os.environ["POD_IP"]
        self._transfer_type = normalize_transfer_type(
            encoder.server_args.encoder_transfer_backend
        )
        # Port prefill dials, per transport: mooncake POSTs /epd/transfer to the
        # encoder's HTTP port; zmq DEALER-connects to the PullZmqServer ROUTER
        # port. Advertised verbatim in every EncoderAddr this processor emits.
        self._advertised_port = (
            encoder.server_args.port
            if self._transfer_type == "mooncake"
            else encoder.pull_zmq_port
        )

    def _pop_time_costs(self, task_id: str) -> dict:
        """Stage costs the encoder recorded for ``task_id`` (K3 time_cost_info)."""
        pop = getattr(self._encoder, "pop_task_time_costs", None)
        return pop(task_id) if pop is not None else {}

    def _known_time_costs(self, task_id: str) -> dict:
        """``_pop_time_costs`` limited to TimeCostInfo fields.

        An extra key would otherwise raise inside the success path and turn a
        finished task into an error result.
        """
        return {
            key: value
            for key, value in self._pop_time_costs(task_id).items()
            if key in TimeCostInfo.model_fields
        }

    async def process(self, task: dict[str, Any]) -> TaskResult:
        """Encode a task and return a TaskResult."""
        start = time.monotonic()
        task_id = task.get("task_id", "")
        request_id = task.get("request_id", "")
        model_id = task.get("model_id", "")
        modality = task.get("type", "image")

        media_info = task.get("mm_info", {}).get("media_info", {})
        mm_items = media_info.get("media_urls", []) or []
        # sample_fps resolution order:
        #   1. media_items[0]["sample_fps"]  (per-item, from vit-server payload)
        #   2. task["sample_fps"]            (top-level, legacy / direct callers)
        #   3. None  → encoder will fall back to model default
        sample_fps: Optional[float] = task.get("sample_fps")
        if not mm_items:
            # Fallback to media_items
            media_items_list = media_info.get("media_items", [])
            if media_items_list:
                mm_items = [item["media_url"] for item in media_items_list]
                modality = media_items_list[0].get("media_type", modality)
                item_fps = media_items_list[0].get("sample_fps")
                if item_fps is not None:
                    sample_fps = item_fps
        image_url = mm_items[0] if mm_items else ""

        logger.info(
            f"[Task] fetched task: req_id={request_id} task_id={task_id} "
            f"type={modality} budget_tokens={task.get('tokens', 0)} "
            f"sample_fps={sample_fps} "
        )
        _new_epd_metric("inc_requests_received", modality=modality)

        session_id = task.get("session_id", "")
        buffer_address = task.get("buffer_address", 0)
        rdma_port = task.get("rdma_port", 0)
        header_size = task.get("header_size", 0)

        try:
            from sglang.srt.disaggregation.encoder.server import BadRequestError
            from sglang.srt.managers.schedule_batch import Modality

            chunk_timestamps = None
            if modality == "video":
                if len(mm_items) != 1:
                    raise BadRequestError(
                        f"video task {task_id} carries {len(mm_items)} media urls; "
                        "expected exactly one"
                    )
                (
                    embedding_len,
                    embedding_dim,
                    mm_embedding,
                    img_grid_thw,
                    orig_sizes,
                    chunk_timestamps,
                ) = await self._encoder.encode_video_and_store_for_workflow(
                    mm_items[0], request_id, task_id, sample_fps
                )
            else:
                (
                    embedding_len,
                    embedding_dim,
                    mm_embedding,
                    img_grid_thw,
                    orig_sizes,
                ) = await self._encoder.encode_and_store_for_workflow(
                    mm_items=mm_items,
                    req_id=request_id,
                    task_id=task_id,
                    modality=Modality.from_str(modality),
                )
            update_encoder_output_timestamp()

            embedding_dtype = str(mm_embedding.dtype).replace("torch.", "")

            elapsed = time.monotonic() - start
            logger.info(
                f"[Task] done req_id={request_id} task_id={task_id} "
                f"type={modality} out_tokens={embedding_len} "
                f"shape=[{embedding_len},{embedding_dim}] compute_e2e={elapsed:.3f}s"
            )
            metrics = getattr(self._encoder, "metrics_collector", None)
            if metrics is not None:
                metrics.observe_one_finished_request()
            _new_epd_metric("inc_requests_total", modality=modality, status="success")

            return TaskResult(
                model_id=model_id,
                task_id=task_id,
                request_id=request_id,
                image_url=image_url,
                status_code=HTTPStatus.OK,
                message="success",
                token_num=embedding_len,
                encoder_addr=EncoderAddr(
                    ip=self._pod_ip,
                    http_port=self._advertised_port,
                    transfer_type=self._transfer_type,
                    task_id=task_id,
                    model_id=model_id,
                    embedding_size=mm_embedding.nbytes,
                    embedding_len=embedding_len,
                    embedding_dim=embedding_dim,
                    embedding_dtype=embedding_dtype,
                    img_grid_thw=img_grid_thw,
                    original_image_sizes=orig_sizes,
                    rdma_port=rdma_port,
                    item_addr=buffer_address,
                    header_size=header_size,
                    mooncake_session_id=session_id,
                    modality=modality,
                    chunk_timestamps=chunk_timestamps,
                ),
                time_cost_info=TimeCostInfo(
                    infer_time_cost=round(elapsed, 4),
                    **self._known_time_costs(task_id),
                ),
            )

        except asyncio.CancelledError:
            # Task timeout: drop the stage costs this task recorded.
            self._pop_time_costs(task_id)
            raise
        except Exception as e:
            self._pop_time_costs(task_id)
            elapsed = time.monotonic() - start
            error_code = getattr(e, "code", HTTPStatus.INTERNAL_SERVER_ERROR)
            logger.error(
                f"[Task] error req_id={request_id} task_id={task_id} "
                f"type={modality} url={str(image_url).split('?')[0]} "
                f"code={error_code} error={e} time={elapsed:.3f}s"
            )
            metrics = getattr(self._encoder, "metrics_collector", None)
            if metrics is not None:
                metrics.observe_one_failed_request()
            _new_epd_metric("inc_requests_total", modality=modality, status="error")
            return TaskResult(
                model_id=model_id,
                task_id=task_id,
                request_id=request_id,
                image_url=image_url,
                status_code=error_code or HTTPStatus.INTERNAL_SERVER_ERROR,
                message=str(e),
                token_num=0,
                encoder_addr=EncoderAddr(
                    ip=self._pod_ip,
                    http_port=self._advertised_port,
                    transfer_type=self._transfer_type,
                    task_id=task_id,
                    model_id=model_id,
                    embedding_size=0,
                    embedding_len=0,
                    embedding_dim=0,
                    embedding_dtype="bfloat16",
                    rdma_port=rdma_port,
                    item_addr=buffer_address,
                    header_size=header_size,
                    mooncake_session_id=session_id,
                    modality=modality,
                ),
                time_cost_info=TimeCostInfo(infer_time_cost=round(elapsed, 4)),
            )


# ==================== VitTask StreamingImplProto ====================


@dataclass
class VitTaskReq:
    """Request wrapper for StreamExecutor — carries the raw task dict from vit-server."""

    rid: str
    task: dict

    def num_tokens(self) -> int:
        """Return the number of tokens for this task."""
        return self.task["tokens"]


@dataclass
class VitTaskResult:
    """Result wrapper for StreamExecutor — carries the TaskResult from encoding."""

    task_result: Any  # TaskResult from encode_workflow

    def num_tokens(self) -> int:
        """Return the number of tokens for this result."""
        return self.task_result.token_num or 1

    def batch_size(self) -> int:
        """Return the batch size for this result."""
        return 1


class VitTaskImpl:
    """StreamingImplProto implementation for encoding vit-server tasks.

    Wraps the async encoder.encode() pipeline into a synchronous submit() -> Future
    by running coroutines on a dedicated asyncio event loop in a background thread.
    """

    # Hard per-task timeout. A hung task (never completes, never errors — e.g. a
    # decode/embed stalled on a saturated GPU) would otherwise hold its pending
    # tokens and pipeline slot forever, so a few of them would permanently wedge
    # the encoder (free_tokens never recovers -> fetch loop stops). On timeout the
    # future completes with TimeoutError, which fires _on_task_done (releasing the
    # pending tokens) and frees the slot. Aligned to the 30s upstream link timeout:
    # a task that can't finish within it produces a result the link has already
    # abandoned, so waiting longer is pointless — fail fast and reclaim resources.
    # Tune via SGLANG_ENCODER_TASK_TIMEOUT_S.
    TASK_TIMEOUT_S: float = float(os.environ.get("SGLANG_ENCODER_TASK_TIMEOUT_S", "30"))

    def __init__(self, encoder, process_fn, collector: BatchExecutorCollector = None):
        """Initialize VitTaskImpl.

        Args:
            encoder: The encoder instance (used for name/labels).
            process_fn: An async callable (task_dict) -> TaskResult that performs
                        the actual encoding work.
            collector: Optional metrics collector. If None, a default one is created.
        """
        self.name = "vit_task"
        self._encoder = encoder
        if collector is not None:
            self.collector = collector
        else:
            labels = (
                {"model_name": encoder.server_args.served_model_name}
                if encoder.server_args.served_model_name
                else {}
            )
            self.collector = BatchExecutorCollector(self.name, labels=labels)
        self._process_fn = process_fn

        # Coroutines run on the encoder's MAIN event loop (bound in poll_forever),
        # NOT a private loop: the New EPD MMEncoder.encode() and the task-embedding
        # locks/state it touches are bound to the serving loop, so driving encode
        # from a different loop would raise "attached to a different loop".
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind to the encoder's serving loop; submit() schedules coros there."""
        self._loop = loop

    def submit(self, req: VitTaskReq) -> Future:
        """Submit a single task and return a concurrent.futures.Future."""
        assert self._loop is not None, "VitTaskImpl.bind_loop must be called first"
        return asyncio.run_coroutine_threadsafe(self._do_process(req), self._loop)

    async def _do_process(self, req: VitTaskReq) -> VitTaskResult:
        """Run process_fn (bounded by a hard timeout) and wrap result."""
        try:
            task_result = await asyncio.wait_for(
                self._process_fn(req.task), timeout=self.TASK_TIMEOUT_S
            )
        except asyncio.TimeoutError:
            tid = req.task.get("task_id") if isinstance(req.task, dict) else "?"
            logger.error(
                "[vit_task] task timed out after %.0fs (task_id=%s); treated as "
                "failed so its pending tokens/slot are released (avoids wedging "
                "the encoder). Likely a stalled decode/embed on a saturated GPU.",
                self.TASK_TIMEOUT_S,
                tid,
            )
            metrics = getattr(self._encoder, "metrics_collector", None)
            if metrics is not None:
                metrics.observe_one_aborted_request()
            raise
        return VitTaskResult(task_result=task_result)

    def shutdown(self):
        """No private loop to stop; coroutines run on the encoder's main loop."""


# ==================== EncodeWorkflow ====================


class EncodeWorkflow:
    """Fetch-encode-submit lifecycle orchestrator (stream mode via StreamExecutor).

    Architecture:
      - poll_forever() runs in the main asyncio loop, polling vit-server for tasks
        whenever free_tokens >= FETCH_ADMIT_THRESHOLD, and feeding them into a StreamExecutor.
      - The StreamExecutor runs in a dedicated thread, dispatching tasks to
        VitTaskImpl which processes them on its own async event loop.
      - Completed results are collected via done-callbacks and flushed back to
        vit-server each polling iteration.
    """

    # 留给 prefill 拉取 embedding 的最小缓冲时间
    GRACE_BUFFER: float = float(os.environ.get("SGLANG_ENCODER_GRACE_BUFFER", "30"))
    # stop 后最大等待时间（硬性截止）
    MAX_WAIT: float = float(os.environ.get("SGLANG_ENCODER_MAX_WAIT", "120"))
    # 当 last_complete 距 stop_time 超过 GRACE_BUFFER 时，仍需等待的最小时间
    MIN_WAIT_AFTER_STOP: float = float(
        os.environ.get("SGLANG_ENCODER_MIN_WAIT_AFTER_STOP", "10")
    )
    # 拉取新任务的最小 free_tokens 准入阈值。
    # 只有 free_tokens >= FETCH_ADMIT_THRESHOLD 时才向 vit-server 发起拉取请求。
    # free_tokens 的统计口径是 LLM tokens × video_compute_weight（视频为 4），
    # 因此所有阈值均需按该权重折算：
    #   单视频 256K LLM tokens × 4 = 1M compute-weighted tokens
    # 配合 embed_max_tokens 默认 2.5M，可确保 pipeline 已有任务不超过 2 个大视频时才继续拉取。
    # 可通过环境变量 SGLANG_ENCODER_FETCH_ADMIT_TOKENS 调整。
    FETCH_ADMIT_THRESHOLD: int = int(
        os.environ.get("SGLANG_ENCODER_FETCH_ADMIT_TOKENS", 1048576)
    )

    # 单次 fetch 请求允许拉取的最大任务数。
    # 防止 vit-server 一次性返回大量小任务（如图片）导致 GPU 突发过载。
    # 默认 16：每轮最多拉 16 个任务，配合 50ms 轮询间隔，最高吞吐约 320 tasks/s，
    # 足够满足正常负载，同时避免单轮突发。
    # 可通过环境变量 SGLANG_ENCODER_FETCH_MAX_TASKS 调整。
    FETCH_MAX_TASKS_PER_BATCH: int = int(
        os.environ.get("SGLANG_ENCODER_FETCH_MAX_TASKS", 16)
    )

    # 轮询心跳日志间隔（秒）：空轮询不打日志，靠它确认 encoder 确实在向 vit-server 拉取。
    POLL_HEARTBEAT_SECS: float = 30.0

    # pipeline 中允许的最大并发任务数。
    # 即使 free_tokens 充足，当 pipeline 中正在处理的任务数达到此上限时也暂停拉取。
    # 防止大量小 token 任务（如图片）同时涌入导致 GPU 显存/计算瞬间打满。
    # 默认 32：按 StreamExecutor 吞吐能力，32 个并发任务可保持 GPU 高利用率而不过载。
    # 可通过环境变量 SGLANG_ENCODER_MAX_PIPELINE_TASKS 调整。
    MAX_PIPELINE_TASKS: int = int(
        os.environ.get("SGLANG_ENCODER_MAX_PIPELINE_TASKS", 32)
    )

    def __init__(self, encoder):
        """Wire the workflow to its encoder, vit-server client and executor."""
        self._encoder = encoder
        self._client = VitServerClient(encoder.server_args.vit_server_url)
        self._processor = TaskProcessor(encoder)
        self._model_id = os.environ["MODEL_ID"]
        self._fetch_collector = VitFetchCollector(
            labels=(
                {"model_name": encoder.server_args.served_model_name}
                if encoder.server_args.served_model_name
                else None
            )
        )
        self.is_alive: bool = False
        self._stopped: bool = False
        self._stop_time: float | None = None
        self._last_batch_complete_time: float = 0.0

        # StreamExecutor: VitTaskImpl wraps async process into sync Future
        self._impl = VitTaskImpl(encoder, process_fn=self._processor.process)
        self._executor = StreamExecutor(self._impl)

        # Submit buffer: done-callback appends here, flush drains it
        self._lock = threading.Lock()
        self._submit_buffer: list[TaskResult] = []
        self._pipeline_pending_tokens: int = 0
        self._task_token_map: dict[str, PendingTaskTokenStat] = {}
        self._fetch_skip_count: int = 0
        # Poll heartbeat: an idle poll returns no tasks and logs nothing, so
        # without this an encoder that polls fine looks identical to one that
        # never polls.
        self._poll_count: int = 0
        self._poll_nonempty_count: int = 0
        self._poll_fail_count: int = 0
        self._last_heartbeat: float = time.monotonic()

    def _log_poll_heartbeat(self) -> None:
        """Log poll counters every POLL_HEARTBEAT_SECS."""
        now = time.monotonic()
        if now - self._last_heartbeat < self.POLL_HEARTBEAT_SECS:
            return
        self._last_heartbeat = now
        logger.info(
            f"[Poll] model_id={self._model_id} fetches={self._poll_count} "
            f"non_empty={self._poll_nonempty_count} failed={self._poll_fail_count} "
            f"free_tokens={self._encoder.free_tokens} "
            f"active_tasks={self.active_task_count} "
            f"last_reply_status={self._client.last_body_status} "
            f"last_reply_message={self._client.last_body_message!r}"
        )

    @property
    def pending_num_tokens(self) -> int:
        """整个 pipeline 中正在处理的 token 总数（从 fetch 到 encode 完成）。"""
        return self._pipeline_pending_tokens

    def update_task_tokens(self, task_id: str, actual_tokens: int) -> None:
        """用 encode 阶段计算出的精确 token 数修正 vit-server 估算值。

        vit-server 在 fetch 阶段用估算的 token 数（task["tokens"]）初始化
        _pipeline_pending_tokens，但实际 token 数只有在 preprocess 完成后才能精确
        得知。两者若有偏差，偏差会一直累积直到任务结束才消除，导致 free_tokens 长期
        偏低或偏高。调用本方法可提前用精确值覆盖估算值，消除偏差。

        线程安全：使用与 _on_task_done 相同的 _lock。
        """
        with self._lock:
            stat = self._task_token_map.get(task_id)
            if stat is None:
                return  # task already completed or not known
            # Correct exactly ONE not-yet-corrected copy (a reserved entry still at
            # its estimate) to the precise value, compensating pending by only that
            # copy's diff. One diff per call (this method is invoked once per
            # processed copy) — never scaled by an outstanding count — so a
            # duplicate that arrives later is just another independent entry,
            # corrected by its own later call. This keeps the invariant
            # pending == sum(reserved) balanced under any fetch/correct/done order.
            for i, r in enumerate(stat.reserved):
                if r != actual_tokens:
                    delta = actual_tokens - r
                    stat.reserved[i] = actual_tokens
                    self._pipeline_pending_tokens += delta
                    logger.debug(
                        f"[workflow] task_id={task_id} token estimate corrected: "
                        f"{r} -> {actual_tokens} (delta={delta:+d}, one copy)"
                    )
                    return
            # every in-flight copy is already at the actual value: nothing to do

    @property
    def active_task_count(self) -> int:
        """当前 pipeline 中正在处理的任务数。"""
        with self._lock:
            return len(self._task_token_map)

    def _build_fetch_request(self) -> FetchTaskRequest:
        """Build a /fetch_task request from the encoder's current free budget."""
        return FetchTaskRequest(
            model_id=self._model_id,
            max_task_num=self.FETCH_MAX_TASKS_PER_BATCH,
            running_task_num=None,
            free_tokens=self._encoder.free_tokens,
            encoder_name=f"sglang-encoder-{self._encoder.rank}",
            transfer_type=normalize_transfer_type(
                self._encoder.server_args.encoder_transfer_backend
            ),
        )

    def _on_task_done(self, future: Future, task_id: str):
        """Callback invoked when a task future completes (from StreamExecutor thread)."""
        with self._lock:
            stat = self._task_token_map.get(task_id)
            before = self._pipeline_pending_tokens
            if stat and stat.reserved:
                tokens = stat.reserved.pop()  # subtract exactly this copy's reserve
                count_after = stat.count
                if not stat.reserved:
                    self._task_token_map.pop(task_id, None)
            else:
                tokens = 0
                count_after = 0
            self._pipeline_pending_tokens -= tokens
            after = self._pipeline_pending_tokens
        try:
            vit_result: VitTaskResult = future.result()
            self._submit_buffer.append(vit_result.task_result)
            self._last_batch_complete_time = time.time()
        except Exception as e:
            logger.error(f"Task future failed: {e}", exc_info=True)

    async def _flush_submit(self) -> None:
        """Submit all buffered results to vit-server."""
        if not self._submit_buffer:
            return

        buffer, self._submit_buffer = self._submit_buffer, []

        model_id = buffer[0].model_id
        submit_request = SubmitResultRequest(
            model_id=model_id,
            task_results=buffer,
        )
        task_ids = [r.task_id for r in buffer]
        status, message = await self._client.submit_results(submit_request)
        if status == HTTPStatus.OK:
            logger.info(
                f"Submit tasks={len(buffer)} "
                f"free_tokens={self._encoder.free_tokens} task_ids={task_ids}"
            )
        else:
            logger.error(
                f"Submit tasks={len(buffer)} failed with "
                f"free_tokens={self._encoder.free_tokens} task_ids={task_ids} "
                f"message={message[:200]}"
            )

    def graceful_stop(self):
        """优雅停止：停止接收新任务，等待逻辑在 poll_forever 退出后执行。"""
        if self._stopped:
            return
        self._stopped = True
        self._stop_time = time.time()
        logger.info(f"Graceful stop initiated at {self._stop_time:.1f}")

    def stop(self):
        """Signal the polling loop to stop (non-graceful, immediate)."""
        if not self._stopped:
            logger.info("Stopping EncodeWorkflow")
            self._stopped = True

    @property
    def stopped(self) -> bool:
        """Return whether the polling loop has been signalled to stop."""
        return self._stopped

    async def poll_forever(self):
        """Stream mode polling loop using StreamExecutor.

        Continuously fetches tasks from vit-server when free_tokens reaches
        the admission threshold, feeds them into StreamExecutor, and flushes
        completed results back to vit-server each iteration.
        """
        vit_server_url = self._encoder.server_args.vit_server_url
        if not vit_server_url:
            logger.info("vit_server_url not configured, skipping task polling")
            return

        logger.info(
            f"Starting stream-mode task polling from vit-server: {vit_server_url}"
            f" | fetch_admit_threshold={self.FETCH_ADMIT_THRESHOLD}"
            f" | fetch_max_tasks_per_batch={self.FETCH_MAX_TASKS_PER_BATCH}"
            f" | max_pipeline_tasks={self.MAX_PIPELINE_TASKS}"
        )

        # Bind task execution to THIS (the encoder's serving) loop before the
        # StreamExecutor thread starts submitting, so encode() runs where its
        # state/locks live.
        self._impl.bind_loop(asyncio.get_running_loop())

        # Start the StreamExecutor serve loop in a dedicated thread
        executor_thread = threading.Thread(
            target=self._executor.serve, daemon=True, name="vit-task-executor"
        )
        executor_thread.start()

        self.is_alive = True

        try:
            while not self._stopped:
                free_tokens = self._encoder.free_tokens
                # ---- Fetch only when free_tokens >= FETCH_ADMIT_THRESHOLD ----
                # free_tokens 口径为 LLM tokens × video_compute_weight（视频为 4）。
                # 阈值 1M = 256K LLM tokens × 4，确保 pipeline 剩余容量够容纳一个完整大视频
                # 才继续拉取，避免 vit-server "至少返回一个 task" 逻辑把第三个大视频塞入。
                # ---- 同时检查并发任务数不超过 MAX_PIPELINE_TASKS ----
                # 防止大量小 token 任务（如图片）在 token 预算充足时突发涌入打爆 GPU。
                can_fetch = (
                    free_tokens >= self.FETCH_ADMIT_THRESHOLD
                    and self.active_task_count < self.MAX_PIPELINE_TASKS
                )
                if can_fetch:
                    self._fetch_skip_count = 0
                    fetch_start = time.monotonic()
                    try:
                        result = await self._client.fetch_tasks(
                            self._build_fetch_request()
                        )
                        self._poll_count += 1
                        if not result.success:
                            self._poll_fail_count += 1
                        elif result.tasks:
                            self._poll_nonempty_count += 1
                        if result.success and result.tasks:
                            self._fetch_collector.observe_task_count(len(result.tasks))
                            for task in result.tasks:
                                task_id = task.get("task_id", "")
                                req = VitTaskReq(rid=task_id, task=task)
                                tokens = req.num_tokens()
                                with self._lock:
                                    before = self._pipeline_pending_tokens
                                    stat = self._task_token_map.get(task_id)
                                    if stat is None:
                                        stat = PendingTaskTokenStat()
                                        self._task_token_map[task_id] = stat
                                    # One reserved entry per copy (the estimate);
                                    # corrected/subtracted independently later.
                                    stat.reserved.append(tokens)
                                    count_after = stat.count
                                    self._pipeline_pending_tokens += tokens
                                    after = self._pipeline_pending_tokens
                                    active_after = len(self._task_token_map)
                                logger.info(
                                    f"[Quota] fetched task_id={task_id} "
                                    f"task_tokens={tokens} "
                                    f"pending_tokens={before}->{after} "
                                    f"free_tokens={self._encoder.free_tokens} "
                                    f"active_tasks={active_after} dup_count={count_after} "
                                    f"(admit_threshold={self.FETCH_ADMIT_THRESHOLD}, "
                                    f"max_pipeline={self.MAX_PIPELINE_TASKS})"
                                )
                                fut = self._executor(req)
                                fut.add_done_callback(
                                    lambda f, tid=task_id: self._on_task_done(f, tid)
                                )
                    except Exception as e:
                        logger.error(f"Error fetching tasks: {e}")
                    self._fetch_collector.observe_duration(
                        time.monotonic() - fetch_start
                    )
                else:
                    self._fetch_skip_count += 1
                    if self._fetch_skip_count % 200 == 1 and self.active_task_count > 0:
                        logger.info(
                            f"Fetch skipped (x{self._fetch_skip_count}): "
                            f"free_tokens={free_tokens} "
                            f"active_tasks={self.active_task_count} "
                            f"(admit_threshold={self.FETCH_ADMIT_THRESHOLD}, "
                            f"max_pipeline={self.MAX_PIPELINE_TASKS})"
                        )

                # Always update free_tokens gauge so monitoring reflects real-time usage
                self._fetch_collector.set_free_tokens(self._encoder.free_tokens)
                self._log_poll_heartbeat()

                # ---- Flush completed results ----
                await self._flush_submit()

                # Brief sleep to avoid busy-waiting
                await asyncio.sleep(0.05)

        finally:
            # Stop the StreamExecutor
            self._executor.stop()
            executor_thread.join(timeout=30.0)

            # Flush remaining results
            await self._flush_submit()
            await self._client.close()
            self._impl.shutdown()
            self.is_alive = False
            logger.info("EncodeWorkflow polling loop exited")

        # 优雅退出：等待 prefill 拉取最后一个 embedding
        if self._stop_time is not None:
            await self._graceful_shutdown_wait()

    async def _graceful_shutdown_wait(self):
        """优雅退出等待逻辑（不含 shutdown）。

        核心公式：target = min(max(last_complete + GRACE_BUFFER, stop_time), deadline)
        确保 prefill 有 GRACE_BUFFER 时间拉取，同时不超过 MAX_WAIT 硬性截止。
        shutdown 由调用方在 await 本方法完成后执行。
        """
        stop_time = self._stop_time
        assert stop_time is not None
        deadline = stop_time + self.MAX_WAIT
        last_complete = self._last_batch_complete_time

        target = max(
            last_complete + self.GRACE_BUFFER,
            stop_time + self.MIN_WAIT_AFTER_STOP,
        )
        target = min(target, deadline)

        wait_time = target - time.time()
        if wait_time > 0:
            logger.info(
                f"Graceful shutdown: last_batch_complete={last_complete:.1f}, "
                f"stop_time={stop_time:.1f}, deadline={deadline:.1f}, "
                f"target={target:.1f}, waiting {wait_time:.1f}s"
            )
            await asyncio.sleep(wait_time)
        else:
            logger.info(
                f"Graceful shutdown: target={target:.1f} already passed, "
                f"stopping now"
            )
