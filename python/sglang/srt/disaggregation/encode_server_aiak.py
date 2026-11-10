"""AIAK encode server implementation.

This module provides the main encode server for processing multimodal
embeddings in disaggregated mode.
"""
import asyncio
import concurrent.futures
import ctypes
import logging
import multiprocessing as mp
import os
import queue
import signal
import threading
import time
import traceback
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Set, Tuple, Iterable

from pydantic import BaseModel, Field
from sglang.srt.disaggregation.encode_workflow import EncodeWorkflow
from sglang.srt.disaggregation.encode_health import (
    PROBE_IMAGE_BASE64,
    PROBE_REQ_ID_PREFIX,
    PROBE_TRIGGER_THRESHOLD,
    HEALTH_CHECK_TIMEOUT,
    encoder_health_state,
    get_encoder_output_timestamp,
    record_transfer_attempt,
    record_transfer_success,
    send_probe_request,
    transfer_health_state,
)

import numpy as np
import torch
import uvicorn
import zmq
import zmq.asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from transformers import AutoImageProcessor, AutoProcessor

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.disaggregation.encode_receiver_aiak import EmbeddingData
from sglang.srt.disaggregation.pull_zmq_server import PullZmqServer
from sglang.srt.disaggregation.vit_cache_handler import VitCacheHandler
from sglang.srt.disaggregation.encode_utils import (
    BatchExecutor,
    StreamExecutor,
    TokenLimitSched,
    compute_num_tokens_per_item,
    get_closest_rdma_nic,
)
from sglang.srt.disaggregation.encode_stage import (
    EmbeddingImpl, MMProcessImpl, MMGPUProcessImpl, MMDecodeImpl, MMLoaderImpl,
    TransferImpl, TransferReq, TransferResult,
    LoadResult, DecodeReq, DecodeResult, ProcessResult, EmbedResult,
    LoadReq, ProcessReq, EmbedReq,
    VideoLoadReq, VideoDecodeImpl, VideoDecodeResult,
)

from sglang.srt.disaggregation.gpu_processor_registry import get_gpu_processor_cls
from sglang.srt.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import initialize_dp_attention
from sglang.srt.managers.io_struct import ProfileReq, ProfileReqType
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.disaggregation.encode_collector import EncoderMetricsCollector, TransferEndpointCollector
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult, MultiModalStaticCache
from sglang.srt.mem_cache.scatter_mm_cache import ScatterMultiModalCache, ScatterMemPool, ScatteredTensor
from sglang.srt.model_loader import get_architecture_class_name, get_model
from sglang.srt.server_args import (
    PortArgs,
    ServerArgs,
    set_global_server_args_for_scheduler,
)


from sglang.srt.utils.network import (
    get_local_ip_auto,
    get_zmq_socket,
    get_zmq_socket_on_host,
    
)
from sglang.srt.utils import random_uuid
from sglang.srt.utils import configure_logger

logger = logging.getLogger(__name__)

use_image_processor_gpu = (
    int(os.getenv("SGLANG_ENCODER_IMAGE_PROCESSOR_USE_GPU", "0")) == 1
)


class MMError(Exception):
    """Base exception for multimodal errors."""

    def __init__(self, message, code=HTTPStatus.INTERNAL_SERVER_ERROR):
        self.message = message
        self.code = code
        super().__init__(self.message)


class BadRequestError(MMError):
    """Exception for bad requests."""

    def __init__(self, message):
        super().__init__(message, code=HTTPStatus.BAD_REQUEST)


class InternalError(MMError):
    """Exception for internal errors."""

    def __init__(self, message):
        super().__init__(message, code=HTTPStatus.INTERNAL_SERVER_ERROR)


class TensorWrapper:
    """Wrapper to keep tensor alive while exposing buffer for zero-copy."""

    def __init__(self, tensor):
        """Initialize tensor wrapper."""
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


class MMRequestInfo:
    """Request-level metadata that maps to mm_cache entries."""

    def __init__(
        self,
        req_id,
        mm_hash,
        num_parts,
        part_idx,
        image_grid_dim,
        image_url=None,
        error_msg=None,
        error_code=None,
    ):
        self.req_id = req_id
        self.mm_hash = mm_hash  # Key to find embedding in mm_cache
        self.num_parts = num_parts
        self.part_idx = part_idx
        self.image_grid_dim = image_grid_dim
        self.image_url = image_url  # Source image URL for reference
        self.error_msg = error_msg
        self.error_code = error_code

    def __repr__(self):
        return f"MMRequestInfo(req_id={self.req_id}, mm_hash={self.mm_hash}, part_idx={self.part_idx})"


class EmbeddingDataMeta:
    """Metadata for embedding data returned via ZMQ multipart to Prefill."""

    def __init__(
        self,
        req_id,
        num_parts,
        part_idx,
        image_grid_dim,
        dtype,
        shape,
        error_msg=None,
        error_code=None,
    ):
        self.req_id = req_id
        self.num_parts = num_parts
        self.part_idx = part_idx
        self.image_grid_dim = image_grid_dim
        self.dtype = dtype
        self.shape = shape
        self.error_msg = error_msg
        self.error_code = error_code


def _convert(data):
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.tensor(data)
    elif isinstance(data, list) and isinstance(data[0], np.ndarray):
        return torch.tensor(np.array(data))
    elif isinstance(data, list) and isinstance(data[0], (int, float)):
        return torch.tensor(data)
    else:
        return data




class MMEncoder:
    """Multimodal encoder class."""

    def __init__(
        self,
        server_args: ServerArgs,
        schedule_path=None,
        dist_init_method=None,
        rank: int = 0,
    ):
        logger.info(f"init MMEncoder {rank}/{server_args.tp_size}")
        self.server_args = server_args
        set_global_server_args_for_scheduler(server_args)
        self.rank = rank
        self.profiler = EncoderProfiler(rank)

        self.model_config = ModelConfig.from_server_args(
            server_args,
        )

        self.model_arch = get_architecture_class_name(self.model_config)
        self.requires_dict_image_input = (
            self.model_arch == "KimiK25ForConditionalGeneration"
        )

        self.load_config = LoadConfig(
            load_format=server_args.load_format,
            download_dir=server_args.download_dir,
            model_loader_extra_config=server_args.model_loader_extra_config,
            remote_instance_weight_loader_seed_instance_ip=(
                server_args.remote_instance_weight_loader_seed_instance_ip
            ),
            remote_instance_weight_loader_seed_instance_service_port=(
                server_args.remote_instance_weight_loader_seed_instance_service_port
            ),
            remote_instance_weight_loader_send_weights_group_ports=(
                server_args.remote_instance_weight_loader_send_weights_group_ports
            ),
        )

        self.device = server_args.device
        self.gpu_id = server_args.base_gpu_id + rank

        self.device_config = DeviceConfig(
            device=self.device,
            gpu_id=self.gpu_id,
        )

        torch.get_device_module(self.device).set_device(self.gpu_id)

        self.use_image_processor_gpu = use_image_processor_gpu

        init_distributed_environment(
            world_size=server_args.tp_size,
            rank=rank,
            distributed_init_method=dist_init_method,
            local_rank=rank,
        )
        initialize_model_parallel(tensor_model_parallel_size=server_args.tp_size)
        initialize_dp_attention(server_args, self.model_config)

        self.model = get_model(
            model_config=self.model_config,
            load_config=self.load_config,
            device_config=self.device_config,
        )

        self.context = zmq.asyncio.Context(2)
        self.sync_context = zmq.Context()  # Reuse sync context for thread pool
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)

        self.send_timeout = envs.SGLANG_ENCODER_SEND_TIMEOUT.get()

        self.mm_cache = None
        self._shutdown_called: bool = False

        # BatchExecutors for load (stage 1), processor (stage 2), embed (stage 3)
        load_batch_size = int(os.environ.get("SGLANG_ENCODER_LOAD_BATCH_SIZE", 8))
        proc_batch_size = int(os.environ.get("SGLANG_ENCODER_PROC_BATCH_SIZE", 8))
        embed_max_tokens = int(os.environ.get("SGLANG_ENCODER_EMBED_MAX_TOKENS", 2621440))
        self.embed_max_tokens = embed_max_tokens
        self._workflow_embed_tokens: int = 0
        self._gpu_processor_cls = get_gpu_processor_cls(server_args.model_path)
        shm_buf_size = int(os.environ.get("SGLANG_ENCODER_PROC_SHM_SIZE_MB", 32)) << 20
        use_shm_ipc = os.environ.get("SGLANG_ENCODE_PROC_USE_SHM_IPC", "false").lower() == "true"

        embedding_cache_size = int(os.environ.get("SGLANG_VLM_CACHE_SIZE_MB", "4096"))
        cache_page_size = int(os.environ.get("SGLANG_VLM_CACHE_PAGE_SIZE", "64"))

        if server_args.encoder_transfer_backend == "mooncake":
            self.mem_pool = ScatterMemPool(
                max_bytes=embedding_cache_size * 1024 * 1024,
                embed_dim=self.model_config.hidden_size,
                page_size=cache_page_size,
                dtype=self.model_config.dtype,
                device="cpu",
            )
            logger.info(f"create ScatterMemPool for mooncke transfer with page_size={cache_page_size}, "
                        f"size={embedding_cache_size} MB, embed_dim={self.model_config.hidden_size}, "
                        f"dtype={self.model_config.dtype}, device={self.device}")
        else:
            self.mem_pool = None

        if self.server_args.enable_prefix_mm_cache:
            if server_args.encoder_transfer_backend == "mooncake":
                self.mm_cache = ScatterMultiModalCache(self.mem_pool)
                logger.info(f"create ScatterMultiModalCache by reference ScatterMemPool")
            else:
                self.mm_cache = MultiModalStaticCache(embedding_cache_size * 1024 * 1024)
                logger.info(f"create MultiModalStaticCache size={embedding_cache_size} MB")

        self.metrics_collector = EncoderMetricsCollector(
            labels={"model_name": server_args.served_model_name}
            if server_args.served_model_name else None
        )

        self.embed_executor = BatchExecutor(
            EmbeddingImpl(
                self.model,
                server_args,
                gpu_processor_cls=self._gpu_processor_cls,
            ),
            sched=TokenLimitSched(embed_max_tokens)
        )
        # 4-stage pipeline: Load -> Decode -> Process -> Embed
        self.load_executor = StreamExecutor(
            MMLoaderImpl(server_args, load_batch_size, return_bytes=True),
        )
        self.decode_executor = StreamExecutor(
            MMDecodeImpl(server_args, device=torch.device("cuda", self.gpu_id),
                         use_gpu=self.use_image_processor_gpu),
        )
        if self.use_image_processor_gpu:
            self.processor_executor = StreamExecutor(
                MMGPUProcessImpl(
                    server_args,
                    device=torch.device("cuda", self.gpu_id),
                    gpu_processor_cls=self._gpu_processor_cls,
                ),
            )
        else:
            self.processor_executor = StreamExecutor(
                MMProcessImpl(
                    server_args,
                    proc_batch_size,
                    use_shm_ipc=use_shm_ipc,
                    shm_buf_size=shm_buf_size,
                    device=torch.device(self.device, self.gpu_id),
                ),
            )

        # Start serve threads (downstream first: embed -> processor -> decode -> load)
        self._embed_thread = threading.Thread(
            target=self.embed_executor.serve,
            daemon=True, name="embed-executor"
        )
        self._embed_thread.start()

        self._processor_thread = threading.Thread(
            target=self.processor_executor.serve,
            daemon=True, name="processor-executor"
        )
        self._processor_thread.start()

        self._decode_thread = threading.Thread(
            target=self.decode_executor.serve,
            daemon=True, name="decode-executor"
        )
        self._decode_thread.start()

        self._load_thread = threading.Thread(
            target=self.load_executor.serve,
            daemon=True, name="load-executor"
        )
        self._load_thread.start()

        # GPU video pipeline executor
        video_load_batch_size = int(os.environ.get(
            "SGLANG_ENCODER_VIDEO_LOAD_BATCH_SIZE", "4"
        ))
        self.video_decode_impl = VideoDecodeImpl(server_args, video_load_batch_size)
        self.video_decode_executor = StreamExecutor(self.video_decode_impl)
        self._video_decode_thread = threading.Thread(
            target=self.video_decode_executor.serve, daemon=True, name="video-decode-executor")
        self._video_decode_thread.start()
        self._video_preprocess_lock = threading.Lock()
        self._video_max_inflight_chunks = max(
            1,
            int(os.environ.get("SGLANG_ENCODER_VIDEO_MAX_INFLIGHT_CHUNKS", "8")),
        )
        self._video_empty_cache_tokens = max(
            0,
            int(os.environ.get("SGLANG_ENCODER_VIDEO_EMPTY_CACHE_TOKENS", "10000")),
        )
        # Streaming pipeline: decode each chunk immediately and pipeline into
        # preprocess+embed, instead of decoding the whole video first.
        self._video_streaming_pipeline = (
            os.environ.get("SGLANG_ENCODER_VIDEO_STREAMING_PIPELINE", "true").lower() == "true"
        )
        self._default_video_sample_fps = self._compute_default_video_sample_fps()

        if schedule_path is not None:
            self.schedule_socket = get_zmq_socket(
                self.context, zmq.PULL, schedule_path, True
            )

        if self.rank == 0:
            logger.info(
                f"Using transfer backend: {self.server_args.encoder_transfer_backend}"
            )

            # local_ip is needed for zmq_to_scheduler backend too
            self.local_ip = get_local_ip_auto()

            if self.server_args.encoder_transfer_backend == "mooncake":
                ib_device = (
                    self.server_args.disaggregation_ib_device
                    or self.server_args.mooncake_ib_device
                ) or get_closest_rdma_nic()
                logger.info(f"embedding transfer use RDMA nic: {ib_device}")
                self.transfer_impl = TransferImpl(
                    server_args=self.server_args,
                    hostname=self.local_ip,
                    gpu_id=self.gpu_id,
                    ib_device=ib_device,
                )
                assert self.mem_pool is not None, "mem_pool must be initialized for mooncake transfer"
                # Register buffer with RDMA engine for zero-copy transfer
                data_ptr, nbytes = self.mem_pool.get_buffer_info()
                self.transfer_impl.register_memory(data_ptr, nbytes)
                self.transfer_executor = BatchExecutor(self.transfer_impl)
                logger.info("TransferImpl (mooncake) initialized")

                self._transfer_thread = threading.Thread(
                    target=self.transfer_executor.serve,
                    daemon=True, name="transfer-executor"
                )
                self._transfer_thread.start()
                self.pull_zmq_server = None
                self.pull_zmq_port = server_args.port
            else:
                # Use get_zmq_socket_on_host to automatically assign a ZMQ port
                zmq_bind_host = get_local_ip_auto()
                self.pull_zmq_port, _ = get_zmq_socket_on_host(
                    self.sync_context, zmq.REP, host=zmq_bind_host, port=(30000, 32768)
                )
                self.background_tasks: Set[asyncio.Task] = set()
                # Initialize PullZmqServer for handling embedding pull requests
                self.pull_zmq_server = PullZmqServer(
                    self, self.context, self.pull_zmq_port
                )

                logger.info(f"pull_zmq_port: {self.pull_zmq_port}")

        logger.info(f"rank {rank} init finish ")

    @property
    def free_tokens(self) -> int:
        """返回可用于负载均衡的剩余 token 额度。

        计算方式: embed_max_tokens - pipeline 中全部待处理 token 数
        精确公式: workflow_pending + direct_embed_pending（无重复计算）

        允许返回负值：负值表示 pipeline 已超过阈值，绝对值为超出的 token 数。
        供 fetcher 做负载均衡及背压判断用。
        """
        embed_pending = self.embed_executor.pending_num_tokens
        workflow_pending = 0
        if hasattr(self, 'encode_workflow') and self.encode_workflow is not None:
            workflow_pending = self.encode_workflow.pending_num_tokens
        direct_embed_pending = max(0, embed_pending - self._workflow_embed_tokens)
        pipeline_pending = workflow_pending + direct_embed_pending
        return self.embed_max_tokens - pipeline_pending

    def _compute_default_video_sample_fps(self) -> float:
        """Compute default video sample fps from model config at init time."""
        cfg = getattr(self.video_decode_impl, "_video_config", None)
        if cfg is not None:
            cfg_fps = getattr(cfg, "sample_fps", None)
            if cfg_fps is not None:
                return float(cfg_fps)
        return 2.0

    @property
    def default_video_sample_fps(self) -> float:
        """模型配置中的默认视频抽帧 fps，供请求未指定时使用。"""
        return self._default_video_sample_fps

    async def _encode_video(self, mm_items, req_id, task_id, sample_fps: Optional[float] = None,
                            from_workflow: bool = False):
        """4-stage video pipeline: load → decode → process → embed.

        When self._video_streaming_pipeline is True, decode, preprocess, and embed
        are pipelined per chunk (streaming mode). Otherwise falls back to the
        original batch-decode-then-preprocess-embed mode.
        """
        if self._video_streaming_pipeline:
            return await self._encode_video_streaming(mm_items, req_id, task_id, sample_fps, from_workflow)
        return await self._encode_video_batch(mm_items, req_id, task_id, sample_fps, from_workflow)

    async def _encode_video_batch(self, mm_items, req_id, task_id, sample_fps: Optional[float] = None,
                                  from_workflow: bool = False):
        _t_start = time.monotonic()
        if sample_fps is None:
            sample_fps = self.default_video_sample_fps
            logger.info(
                f"[encode] task_id={task_id} sample_fps not provided, "
                f"falling back to model default {sample_fps}"
            )
        # Stage 1: Load (download video to local temp file)
        _t_load_start = time.monotonic()
        load_result: LoadResult = await asyncio.wrap_future(
            self.load_executor(LoadReq(rid=task_id, mm_items=mm_items, modalities=["video"] * len(mm_items)))
        )
        _t_load_cost = time.monotonic() - _t_load_start
        local_paths = load_result.images

        # Stage 2: Decode (NVDEC/decord → frame chunks)
        _t_decode_start = time.monotonic()
        try:
            decode_result: VideoDecodeResult = await asyncio.wrap_future(
                self.video_decode_executor(VideoLoadReq(rid=task_id, video_items=local_paths, sample_fps=sample_fps))
            )
        finally:
            for p in local_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass
        _t_decode_cost = time.monotonic() - _t_decode_start

        # Stage 3+4: Per-chunk GPU process → embed (sliding-window inflight)
        # Instead of preprocessing all chunks and queuing all embed futures before
        # waiting, we keep at most `max_inflight` chunks in-flight at any time.
        # This bounds GPU memory: only `max_inflight` pixel_values tensors + embed
        # futures are alive simultaneously instead of all 600+ for a long video.
        video_decode_impl = self.video_decode_impl
        preprocess_cost = 0.0
        _t_preprocess_total = 0.0
        _t_embed_drain_total = 0.0
        max_inflight = self._video_max_inflight_chunks

        # Ordered result list — pre-allocate so we can scatter results by index.
        total_chunks = len(decode_result.chunks)
        embed_results: List[Any] = [None] * total_chunks

        # Track inflight (chunk_idx, embed_req, future) tuples.
        inflight: List[Tuple[int, Any, Any]] = []
        actual_total_tokens = 0
        video_compute_weight = video_decode_impl._gpu_processor_cls.get_video_compute_weight(
            video_decode_impl._video_config
        )

        async def _drain_one() -> None:
            """Await the oldest inflight future and store its result."""
            nonlocal actual_total_tokens, _t_embed_drain_total
            if not inflight:
                return
            _td = time.monotonic()
            idx, embed_req, future = inflight.pop(0)
            result = await asyncio.wrap_future(future)
            _t_embed_drain_total += time.monotonic() - _td
            embed_results[idx] = result
            # Now the pixel_values tensor inside embed_req.mm_item.feature can be
            # released because the future is done.
            embed_req.mm_item.feature = None

        logger.info(
            "[_encode_video] task_id=%s total_chunks=%d max_inflight=%d",
            task_id, total_chunks, max_inflight,
        )

        try:
            for chunk_idx, chunk_tensor in enumerate(decode_result.chunks):
                chunk_rid = f"{task_id}_c{chunk_idx}"

                # Drain oldest inflight chunk when window is full.
                while len(inflight) >= max_inflight:
                    await _drain_one()

                # Stage 3: GPU preprocess (serialised to avoid CUDA stream conflicts)
                t0 = time.monotonic()
                with self._video_preprocess_lock:
                    with torch.cuda.stream(video_decode_impl._preprocess_stream):
                        pixel_values, grid_thw = video_decode_impl._gpu_processor_cls.gpu_preprocess_video_chunk(
                            frames=chunk_tensor, config=video_decode_impl._video_config)
                    video_decode_impl._preprocess_stream.synchronize()
                preprocess_cost += time.monotonic() - t0
                _t_preprocess_total += time.monotonic() - t0

                # Explicitly release the raw decoded frame tensor now that pixel_values is ready.
                # This is especially important when chunks are views of a large all_float storage.
                del chunk_tensor
                decode_result.chunks[chunk_idx] = None

                mm_item = MultimodalDataItem.from_dict({"modality": Modality.VIDEO, "feature": pixel_values})
                mm_item.set("image_grid_thw", grid_thw.unsqueeze(0).clone())

                # Stage 4: Embed
                chunk_output_tokens = compute_num_tokens_per_item(
                    self.model, mm_item, video_decode_impl._gpu_processor_cls
                )
                chunk_embed_req = EmbedReq(
                    rid=chunk_rid, mm_item=mm_item,
                    image_url=mm_items[0], task_id=chunk_rid,
                    num_output_tokens=chunk_output_tokens,
                    num_load_tokens=chunk_output_tokens * video_compute_weight,
                )
                actual_total_tokens += chunk_embed_req.num_tokens()
                embed_future = self.embed_executor(chunk_embed_req)
                inflight.append((chunk_idx, chunk_embed_req, embed_future))

                if from_workflow:
                    chunk_tokens = chunk_embed_req.num_tokens()
                    self._workflow_embed_tokens += chunk_tokens

            # Drain all remaining inflight futures.
            while inflight:
                await _drain_one()

            # All chunks preprocessed — actual token count is now known.
            # Correct the vit-server estimate in workflow so free_tokens reflects reality.
            if from_workflow and hasattr(self, 'encode_workflow') and self.encode_workflow is not None:
                self.encode_workflow.update_task_tokens(task_id, actual_total_tokens)

        finally:
            # Release any inflight embed_reqs that never completed (e.g. exception path).
            if from_workflow:
                # actual_total_tokens was incremented once per submitted chunk (via
                # chunk_embed_req.num_tokens()), which mirrors what we added to
                # _workflow_embed_tokens. Subtract everything we added regardless of
                # whether we exited normally or via an exception mid-loop.
                self._workflow_embed_tokens -= actual_total_tokens
        # Unify all chunk embeddings to the same device before cat.
        # Different chunks may arrive from different paths (normal forward, cache
        # retrieval, TensorWrapper) and end up on different devices, causing
        # "Expected all tensors to be on the same device" errors.
        target_device = embed_results[0].mm_embedding.device
        embeddings = []
        for r in embed_results:
            emb = r.mm_embedding
            if emb.device != target_device:
                logger.warning(
                    "chunk %s embedding on %s, expected %s, moving to target device",
                    r.rid, emb.device, target_device,
                )
                emb = emb.to(target_device, non_blocking=True)
            embeddings.append(emb)
        _t_cat_start = time.monotonic()
        all_embeddings = torch.cat(embeddings, dim=0)
        _t_cat_cost = time.monotonic() - _t_cat_start
        # Concatenate per-chunk grid dims for correct img_grid_thw.
        # Keep grid dims on CPU since they are metadata, not model tensors.
        grid_dims = [
            r.image_grid_dim.cpu()
            for r in embed_results
            if r.image_grid_dim is not None
        ]
        combined_grid_dim = torch.cat(grid_dims, dim=0) if grid_dims else None

        final_embed = EmbedResult(
            rid=task_id, time_cost=max(r.time_cost for r in embed_results),
            image_grid_dim=combined_grid_dim, mm_embedding=all_embeddings,
            num_load_tokens=sum(r.num_tokens() for r in embed_results))
        proc_res = ProcessResult(rid=task_id, time_cost=preprocess_cost, mm_item=None)
        _t_total = time.monotonic() - _t_start
        logger.info(
            "[TIMING] task_id=%s total_chunks=%d total_tokens=%d "
            "load=%.3fs decode=%.3fs preprocess=%.3fs embed_drain=%.3fs cat=%.3fs total=%.3fs",
            task_id, total_chunks, actual_total_tokens,
            _t_load_cost, _t_decode_cost, _t_preprocess_total, _t_embed_drain_total,
            _t_cat_cost, _t_total,
        )
        if (
            self._video_empty_cache_tokens > 0
            and actual_total_tokens >= self._video_empty_cache_tokens
            and torch.cuda.is_available()
        ):
            torch.cuda.empty_cache()
        return load_result, decode_result, proc_res, final_embed

    async def _encode_video_streaming(self, mm_items, req_id, task_id, sample_fps: Optional[float] = None,
                                      from_workflow: bool = False):
        """Streaming video pipeline: load → per-chunk decode → preprocess → embed."""
        _t_start = time.monotonic()
        if sample_fps is None:
            sample_fps = self.default_video_sample_fps
            logger.info(
                f"[encode] task_id={task_id} sample_fps not provided, "
                f"falling back to model default {sample_fps}"
            )

        _t_load_start = time.monotonic()
        load_result: LoadResult = await asyncio.wrap_future(
            self.load_executor(LoadReq(rid=task_id, mm_items=mm_items, modalities=["video"] * len(mm_items)))
        )
        _t_load_cost = time.monotonic() - _t_load_start
        local_paths = load_result.images

        video_decode_impl = self.video_decode_impl
        preprocess_cost = 0.0
        _t_preprocess_total = 0.0
        _t_embed_drain_total = 0.0
        _t_decode_cost = 0.0
        max_inflight = self._video_max_inflight_chunks
        embed_results: List[Any] = []
        inflight: List[Tuple[int, Any, Any]] = []
        actual_total_tokens = 0
        total_chunks = 0
        chunk_timestamps: List[str] = []
        video_h = video_w = 0
        video_compute_weight = video_decode_impl._gpu_processor_cls.get_video_compute_weight(
            video_decode_impl._video_config
        )
        decode_queue: queue.Queue = queue.Queue(maxsize=max_inflight + 2)
        decode_future = video_decode_impl.submit_stream(
            VideoLoadReq(rid=task_id, video_items=local_paths, sample_fps=sample_fps),
            decode_queue,
        )

        async def _drain_one() -> None:
            nonlocal _t_embed_drain_total
            if not inflight:
                return
            _td = time.monotonic()
            idx, embed_req, future = inflight.pop(0)
            result = await asyncio.wrap_future(future)
            _t_embed_drain_total += time.monotonic() - _td
            embed_results[idx] = result
            embed_req.mm_item.feature = None

        try:
            logger.info(
                "[_encode_video_streaming] task_id=%s max_inflight=%d",
                task_id, max_inflight,
            )
            while True:
                item = await asyncio.to_thread(decode_queue.get)
                kind = item[0]
                if kind == "error":
                    raise item[1]
                if kind == "done":
                    _, total_chunks, decode_time, frame_cost, video_h, video_w, chunk_timestamps = item
                    _t_decode_cost = decode_time
                    break

                _, chunk_idx, chunk_tensor, timestamp, height, width, frame_cost = item
                while len(embed_results) <= chunk_idx:
                    embed_results.append(None)
                video_h, video_w = height, width

                while len(inflight) >= max_inflight:
                    await _drain_one()

                chunk_rid = f"{task_id}_c{chunk_idx}"
                t0 = time.monotonic()
                with self._video_preprocess_lock:
                    with torch.cuda.stream(video_decode_impl._preprocess_stream):
                        pixel_values, grid_thw = video_decode_impl._gpu_processor_cls.gpu_preprocess_video_chunk(
                            frames=chunk_tensor, config=video_decode_impl._video_config)
                    video_decode_impl._preprocess_stream.synchronize()
                preprocess_cost += time.monotonic() - t0
                _t_preprocess_total += time.monotonic() - t0
                del chunk_tensor

                mm_item = MultimodalDataItem.from_dict({"modality": Modality.VIDEO, "feature": pixel_values})
                mm_item.set("image_grid_thw", grid_thw.unsqueeze(0).clone())
                chunk_output_tokens = compute_num_tokens_per_item(
                    self.model, mm_item, video_decode_impl._gpu_processor_cls
                )
                chunk_embed_req = EmbedReq(
                    rid=chunk_rid, mm_item=mm_item,
                    image_url=mm_items[0], task_id=chunk_rid,
                    num_output_tokens=chunk_output_tokens,
                    num_load_tokens=chunk_output_tokens * video_compute_weight,
                )
                actual_total_tokens += chunk_embed_req.num_tokens()
                embed_future = self.embed_executor(chunk_embed_req)
                inflight.append((chunk_idx, chunk_embed_req, embed_future))

                if from_workflow:
                    chunk_tokens = chunk_embed_req.num_tokens()
                    self._workflow_embed_tokens += chunk_tokens

            while inflight:
                await _drain_one()

            if from_workflow and hasattr(self, 'encode_workflow') and self.encode_workflow is not None:
                self.encode_workflow.update_task_tokens(task_id, actual_total_tokens)
        finally:
            for p in local_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass
            if from_workflow:
                self._workflow_embed_tokens -= actual_total_tokens

        decode_future.result()
        if not embed_results or any(r is None for r in embed_results):
            raise RuntimeError(f"streaming video encode produced incomplete results for task_id={task_id}")

        target_device = embed_results[0].mm_embedding.device
        embeddings = []
        for r in embed_results:
            emb = r.mm_embedding
            if emb.device != target_device:
                logger.warning(
                    "chunk %s embedding on %s, expected %s, moving to target device",
                    r.rid, emb.device, target_device,
                )
                emb = emb.to(target_device, non_blocking=True)
            embeddings.append(emb)

        _t_cat_start = time.monotonic()
        all_embeddings = torch.cat(embeddings, dim=0)
        _t_cat_cost = time.monotonic() - _t_cat_start
        grid_dims = [
            r.image_grid_dim.cpu()
            for r in embed_results
            if r.image_grid_dim is not None
        ]
        combined_grid_dim = torch.cat(grid_dims, dim=0) if grid_dims else None
        decode_result = VideoDecodeResult(
            rid=task_id,
            time_cost=_t_decode_cost,
            download_cost=0.0,
            frame_extraction_cost=_t_decode_cost,
            chunks=None,
            video_height=video_h,
            video_width=video_w,
            chunk_timestamps=chunk_timestamps,
        )
        final_embed = EmbedResult(
            rid=task_id, time_cost=max(r.time_cost for r in embed_results),
            image_grid_dim=combined_grid_dim, mm_embedding=all_embeddings,
            num_load_tokens=sum(r.num_tokens() for r in embed_results))
        proc_res = ProcessResult(rid=task_id, time_cost=preprocess_cost, mm_item=None)
        _t_total = time.monotonic() - _t_start
        logger.info(
            "[TIMING] task_id=%s total_chunks=%d total_tokens=%d "
            "load=%.3fs decode=%.3fs preprocess=%.3fs embed_drain=%.3fs cat=%.3fs total=%.3fs",
            task_id, total_chunks, actual_total_tokens,
            _t_load_cost, _t_decode_cost, _t_preprocess_total, _t_embed_drain_total,
            _t_cat_cost, _t_total,
        )
        if (
            self._video_empty_cache_tokens > 0
            and actual_total_tokens >= self._video_empty_cache_tokens
            and torch.cuda.is_available()
        ):
            torch.cuda.empty_cache()
        return load_result, decode_result, proc_res, final_embed

    async def _encode(self, mm_items, req_id, task_id, from_workflow=False) \
            -> Tuple[LoadResult, "DecodeResult", ProcessResult, EmbedResult]:
        """Pure 4-stage async pipeline: load -> decode -> process -> embed.

        No cache logic. Multiple concurrent calls naturally pipeline across
        stages: while request A is in stage 4, request B can be in stage 3, etc.
        """
        # Stage 1: load images (thread pool)
        _t_start = time.monotonic()
        _t_load_start = time.monotonic()
        load_result: LoadResult = await asyncio.wrap_future(
            self.load_executor(LoadReq(rid=task_id, mm_items=mm_items))
        )
        _t_load_cost = time.monotonic() - _t_load_start

        # Stage 2: decode (bytes -> GPU tensors or PIL Images)
        _t_decode_start = time.monotonic()
        decode_result: DecodeResult = await asyncio.wrap_future(
            self.decode_executor(DecodeReq(rid=task_id, images=load_result.images, modalities=load_result.modalities))
        )
        _t_decode_cost = time.monotonic() - _t_decode_start
        proc_images = decode_result.images

        # Stage 3: preprocess
        _t_preprocess_start = time.monotonic()
        proc_result: ProcessResult = await asyncio.wrap_future(
            self.processor_executor(ProcessReq(rid=task_id, images=proc_images))
        )
        _t_preprocess_cost = time.monotonic() - _t_preprocess_start

        # Stage 3: embedding (GPU batch)
        embed_req = EmbedReq(
            rid=task_id,
            mm_item=proc_result.mm_item,
            image_url=mm_items[0],
            task_id=task_id,
            num_output_tokens=compute_num_tokens_per_item(
                self.model, proc_result.mm_item, self._gpu_processor_cls
            ),
        )
        if from_workflow:
            _n = embed_req.num_tokens()
            self._workflow_embed_tokens += _n
            # Correct vit-server token estimate with the encoder-computed exact value
            if hasattr(self, 'encode_workflow') and self.encode_workflow is not None:
                self.encode_workflow.update_task_tokens(task_id, _n)
        try:
            _t_embed_start = time.monotonic()
            embed_result: EmbedResult = await asyncio.wrap_future(
                self.embed_executor(embed_req)
            )
            _t_embed_cost = time.monotonic() - _t_embed_start
        finally:
            if from_workflow:
                self._workflow_embed_tokens -= _n

        _t_total = time.monotonic() - _t_start
        logger.info(
            "[TIMING] task_id=%s modality=image total_tokens=%d "
            "load=%.3fs decode=%.3fs preprocess=%.3fs embed=%.3fs total=%.3fs",
            task_id, embed_req.num_tokens(), _t_load_cost, _t_decode_cost,
            _t_preprocess_cost, _t_embed_cost, _t_total,
        )
        return load_result, decode_result, proc_result, embed_result

    async def encode(
        self, mm_items, req_id, num_parts, part_idx, task_id=None, skip_cache=False,
        from_workflow=False,
        modality="image", sample_fps: Optional[float] = None,
    ) -> Tuple[
        Optional[LoadResult], Optional["DecodeResult"],
        Optional[ProcessResult], EmbedResult
    ]:
        """Run encode pipeline with optional cache.

        Cache logic is handled here; _encode is pure stages only.
        Supports both image and video via modality parameter.
        """
        if task_id is None:
            task_id = req_id
        _t_encode_start = time.monotonic()
        try:
            # 1. Cache check — skip all stages on hit
            if not skip_cache and self.server_args.enable_prefix_mm_cache:
                cached = self.mm_cache.get(task_id)
                if cached is not None:
                    emb_bytes = cached.nbytes
                    self.metrics_collector.record_cache_hit(emb_bytes)
                    logger.info(f"[encode] Cache HIT for task_id={task_id}, bytes={emb_bytes}")
                    embed_result = EmbedResult(task_id, 0.0, cached.image_grid_thw, cached.embedding)
                    decode_res_stub = None
                    if cached.chunk_timestamps is not None:
                        decode_res_stub = VideoDecodeResult(
                            rid=task_id,
                            time_cost=0.0,
                            download_cost=0.0,
                            frame_extraction_cost=0.0,
                            chunks=None,
                            chunk_timestamps=cached.chunk_timestamps,
                        )
                    logger.info(
                        "[TIMING] task_id=%s req_id=%s modality=%s cache_hit=true "
                        "cache_lookup_total=%.3fs total=%.3fs",
                        task_id, req_id, modality,
                        time.monotonic() - _t_encode_start,
                        time.monotonic() - _t_encode_start,
                    )
                    return None, decode_res_stub, None, embed_result

            # 2. Cache miss — run full pipeline
            if modality == "video":
                load_res, decode_res, proc_res, embed_res = await self._encode_video(
                    mm_items, req_id, task_id, sample_fps, from_workflow=from_workflow)
            else:
                load_res, decode_res, proc_res, embed_res = await self._encode(
                mm_items, req_id, task_id, from_workflow=from_workflow)

            # 3. Store to cache
            if not skip_cache and self.server_args.enable_prefix_mm_cache:
                image_url = mm_items[0] if mm_items else None
                embedding_result = EmbeddingResult(
                    embedding=embed_res.mm_embedding,
                    image_grid_thw=embed_res.image_grid_dim,
                    image_url=image_url,
                    modality=modality,
                    chunk_timestamps=getattr(decode_res, "chunk_timestamps", None) if decode_res else None,
                )
                self.metrics_collector.record_cache_miss(embedding_result.nbytes)
                success = self.mm_cache.set(task_id, embedding_result)
                if success:
                    embed_res.mm_embedding = embedding_result.embedding
                    logger.info(f"[encode] Cache STORED for task_id={task_id}, "
                                f"image_url={image_url}, cache_size={len(self.mm_cache)}")
                else:
                    logger.warning(
                        f"[encode] Cache STORE FAILED for task_id={task_id}, "
                        f"image_url={image_url}. "
                        f"Current cache usage: {self.mm_cache.current_size}/{self.mm_cache.max_size} "
                        f"({len(self.mm_cache)} entries). "
                        f"Consider increasing SGLANG_VLM_CACHE_SIZE_MB"
                    )

            return load_res, decode_res, proc_res, embed_res
        except Exception as e:
            error_code = getattr(e, "code", HTTPStatus.INTERNAL_SERVER_ERROR)
            error_msg = str(e)
            logger.error(f"Rank {self.rank} encode failed: {error_msg} {error_code = }")
            raise e from e

    def shutdown(self):
        """Gracefully stop all executors and process pools."""
        if self._shutdown_called:
            return
        self._shutdown_called = True
        logger.info("MMEncoder shutting down...")

        # Stop stage executors (drain queued requests then exit serve loop)
        for name, exc in [
            ("load", self.load_executor),
            ("processor", self.processor_executor),
            ("embed", self.embed_executor),
            ("video_decode", self.video_decode_executor),
        ]:
            try:
                exc.stop()
                logger.info(f"  {name} executor stopped")
            except Exception as e:
                logger.warning(f"  {name} executor stop failed: {e}")

        if self.rank == 0 and hasattr(self, "transfer_executor"):
            try:
                self.transfer_executor.stop()
                logger.info("  transfer executor stopped")
            except Exception as e:
                logger.warning(f"  transfer executor stop failed: {e}")

        # Shutdown the process pool inside MMProcessImpl
        proc_impl = self.processor_executor._impl
        if hasattr(proc_impl, "_pool"):
            try:
                proc_impl._pool.shutdown(wait=False, cancel_futures=True)
                logger.info("  process pool shut down")
            except Exception as e:
                logger.warning(f"  process pool shutdown failed: {e}")
        if hasattr(proc_impl, "close"):
            try:
                proc_impl.close()
                logger.info("  shm buffers released")
            except Exception as e:
                logger.warning(f"  shm release failed: {e}")

        # Shutdown the thread pool executor
        try:
            self.executor.shutdown(wait=False, cancel_futures=True)
            logger.info("  thread pool shut down")
        except Exception as e:
            logger.warning(f"  thread pool shutdown failed: {e}")

        logger.info("MMEncoder shutdown complete")


class EncoderProfiler:
    """Profiling utility for encoder."""

    def __init__(self, rank: int):
        """Initialize profiler with rank."""
        self.rank = rank
        self.profiler = None
        self.steps_left = None
        self.output_dir = None
        self.prefix = None
        self.profile_id = None

    def start(self, obj: ProfileReq):
        """Start profiler."""
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
        """Step the profiler."""
        if self.profiler is None:
            return
        self.profiler.step()
        if self.steps_left is not None:
            self.steps_left -= 1
            if self.steps_left <= 0:
                self.stop()

    def stop(self):
        """Stop the profiler."""
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


app = FastAPI()
encoder: MMEncoder = None
encode_workflow: Optional[EncodeWorkflow] = None
_encode_poll_task: asyncio.Task[None] | None = None
vit_cache_handler: Optional[VitCacheHandler] = None
send_sockets: List[zmq.Socket] = []
ENCODER_WARMUP_COMPLETE = False
transfer_endpoint_collector: Optional[TransferEndpointCollector] = None


@app.on_event("startup")
async def _startup_pull_zmq_server():
    """Start the ZMQ REP pull server as a background task when uvicorn is ready.

    Only starts if encoder_transfer_backend is not mooncake (zmq_to_scheduler, zmq, etc. need it).
    """
    if encoder is None or encoder.pull_zmq_server is None:
        return
    task = asyncio.create_task(encoder.pull_zmq_server.run())
    encoder.background_tasks.add(task)
    task.add_done_callback(encoder.background_tasks.discard)
    logger.info(
        f"ZMQ REP pull server task started (port {encoder.pull_zmq_port})"
    )


@app.on_event("startup")
async def _startup_warmup():
    """Send a probe image through the full encode pipeline on startup.

    Triggers the complete load -> process -> embed pipeline with a minimal
    14x14 image to warm up all components (FA4 JIT compilation, CUDA kernels,
    memory allocators, etc.) before real requests arrive.
    Broadcasts to all TP ranks so every GPU is warmed up.
    """

    global ENCODER_WARMUP_COMPLETE

    if encoder is None:
        return

    req_id = f"{PROBE_REQ_ID_PREFIX}warmup_{time.time()}"
    request = {
        "mm_items": [PROBE_IMAGE_BASE64],
        "req_id": req_id,
        "num_parts": 1,
        "part_idx": 0,
        "task_id": req_id,
        "enter_time": time.time(),
    }

    logger.info(f"Running startup warmup via probe request: req_id={req_id}")
    try:
        # Broadcast to other TP ranks (same pattern as handle_encode_request)
        for socket in send_sockets:
            socket.send_pyobj(request)

        load_res, decode_res, proc_res, embed_res = await asyncio.wait_for(
            encoder.encode(
                mm_items=request["mm_items"],
                req_id=request["req_id"],
                num_parts=request["num_parts"],
                part_idx=request["part_idx"],
                task_id=request["task_id"],
            ),
            timeout=600,  # FA4 JIT can take several minutes
        )
        logger.info(
            f"Startup warmup completed: req_id={req_id}, "
            f"load={load_res.time_cost:.2f}s, proc={proc_res.time_cost:.2f}s, "
            f"embed={embed_res.time_cost:.2f}s"
        )
        ENCODER_WARMUP_COMPLETE = True
    except asyncio.TimeoutError:
        logger.error(f"Startup warmup timed out (600s): req_id={req_id}")
    except Exception as e:
        logger.error(f"Startup warmup failed: req_id={req_id}, error={e}", exc_info=True)


async def run_encoder(
    server_args: ServerArgs, schedule_path, dist_init_method, rank: int
):
    """Run encoder loop."""
    encoder = MMEncoder(server_args, schedule_path, dist_init_method, rank)
    while True:
        request = await encoder.schedule_socket.recv_pyobj()
        if isinstance(request, ProfileReq):
            if request.type == ProfileReqType.START_PROFILE:
                if encoder.profiler is None:
                    encoder.profiler = EncoderProfiler(encoder.rank)
                encoder.profiler.start(request)
            else:
                encoder.profiler.stop()
        else:
            task_id = request.get("task_id", request["req_id"])
            await encoder.encode(
                mm_items=request["mm_items"],
                req_id=request["req_id"],
                num_parts=request["num_parts"],
                part_idx=request["part_idx"],
                task_id=task_id,
                modality=request.get("modality", request.get("media_type", "image")),
                sample_fps=request.get("sample_fps"),
            )


def launch_encoder(server_args, schedule_path, dist_init_method, rank):
    """Launch encoder process."""
    try:
        asyncio.run(run_encoder(server_args, schedule_path, dist_init_method, rank))
    except KeyboardInterrupt:
        logger.info(f"Exit rank {rank}")
    except Exception:
        traceback.print_exc()


def launch_server(server_args: ServerArgs):
    """Launch the encode server."""
    global encoder, vit_cache_handler, transfer_endpoint_collector
    if server_args.served_model_name:
        transfer_endpoint_collector = TransferEndpointCollector(
            labels={"model_name": server_args.served_model_name}
        )
    ctx = mp.get_context("spawn")
    zmq_ctx = zmq.Context(10)
    ipc_path_prefix = random_uuid()
    port_args = PortArgs.init_new(server_args)
    if server_args.dist_init_addr:
        dist_init_method = f"tcp://{server_args.dist_init_addr}"
    else:
        dist_init_method = f"tcp://127.0.0.1:{port_args.nccl_port}"
    for rank in range(1, server_args.tp_size):
        schedule_path = f"ipc:///tmp/{ipc_path_prefix}_schedule_{rank}"
        send_sockets.append(
            get_zmq_socket(zmq_ctx, zmq.PUSH, schedule_path, bind=False)
        )
        ctx.Process(
            target=launch_encoder,
            args=(server_args, schedule_path, dist_init_method, rank),
            daemon=True,
        ).start()
    encoder = MMEncoder(server_args, dist_init_method=dist_init_method)
    vit_cache_handler = VitCacheHandler(encoder)
    configure_logger(server_args)
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"]["fmt"] = \
        '%(asctime)s %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
    uvicorn.run(app, host=server_args.host, port=server_args.port, log_config=log_config)



@app.post("/encode")
async def handle_encode_request(request: dict):
    """Handle encode request."""
    req_id = request["req_id"]
    task_id = request.get("task_id", req_id)
    try:

        # broadcast request
        request.update({"enter_time": time.time()})
        for socket in send_sockets:
            socket.send_pyobj(request)

        load_res, decode_res, proc_res, embed_res = (
            await encoder.encode(
                mm_items=request["mm_items"],
                req_id=request["req_id"],
                num_parts=request["num_parts"],
                part_idx=request["part_idx"],
                task_id=task_id,
                modality=request.get("modality", request.get("media_type", "image")),
                sample_fps=request.get("sample_fps"),
            )
        )

        image_grid_dim, mm_embedding = embed_res.image_grid_dim, embed_res.mm_embedding
        return JSONResponse(
            content={
                "embedding_size": mm_embedding.nbytes,
                "embedding_len": mm_embedding.shape[0],
                "embedding_dim": mm_embedding.shape[1],
                "load_cost": load_res.time_cost if load_res else 0.0,
                "decode_cost": decode_res.time_cost if decode_res else 0.0,
                "process_cost": proc_res.time_cost if proc_res else 0.0,
                "embed_cost": embed_res.time_cost

            }
        )
        
    except Exception as e:
        logger.error(e, exc_info=True)
        error_msg = str(e)
        logger.error(f"Unexpected error in encoder logic for {req_id}: {error_msg}")
        return JSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": error_msg,
                "req_id": req_id,
            },
        )



_ENABLE_EMBED_DIGEST = os.environ.get("SGLANG_EMBED_DIGEST", "0") == "1"


def compute_embed_digest(embed) -> Optional[dict]:
    """Compute a lightweight digest of embedding data for transfer verification.

    Works with both torch.Tensor and ScatteredTensor.
    Returns a dict with mean/std/min/max/first8/last8/num_elements.
    """
    from sglang.srt.mem_cache.scatter_mm_cache import ScatteredTensor
    if isinstance(embed, ScatteredTensor):
        t = embed.as_tensor()
    else:
        t = embed

    flat = t.reshape(-1).float()
    n = flat.numel()
    return {
        "mean": flat.mean().item(),
        "std": flat.std().item(),
        "min": flat.min().item(),
        "max": flat.max().item(),
        "first8": flat[:8].tolist(),
        "last8": flat[-8:].tolist(),
        "num_elements": n,
    }


class TransferRequest(BaseModel):
    """Transfer request model."""

    req_id: str
    task_id: str
    part_idx: int = 0
    num_parts: int = 1
    session_id: str
    buffer_address: int
    byte_size: int = Field(..., gt=0)
    image_url: Optional[str] = None
    modality: str = "image"
    sample_fps: Optional[float] = None

    def make_ack(self, status_code: int, message: str, **kwargs) -> "TransferAck":
        """Create a transfer acknowledgement."""
        return TransferAck(
            req_id=self.req_id, task_id=self.task_id, part_idx=self.part_idx,
            status_code=status_code, message=message, **kwargs,
        )


class TransferAck(BaseModel):
    """Transfer acknowledgement model."""

    req_id: str
    task_id: str
    part_idx: int
    status_code: int
    message: str
    transfer_cost_ms: float = 0.0
    total_cost_ms: float = 0.0
    embed_digest: Optional[dict] = None

    def to_response(self) -> JSONResponse:
        """Convert to JSONResponse."""
        http_status = HTTPStatus.OK if self.status_code == 200 else (
            HTTPStatus.BAD_REQUEST if self.status_code == 400
            else HTTPStatus.INTERNAL_SERVER_ERROR
        )
        return JSONResponse(status_code=http_status, content=self.model_dump())


@app.post("/encode_video_gpu_chunked")
async def handle_encode_video_gpu_chunked_request(request: dict):
    """Chunk-pipelined GPU video encode with decode/embed overlap."""
    req_id = request["req_id"]
    task_id = request.get("task_id", req_id)
    try:
        video_items = request["video_items"]
        sample_fps = request.get("sample_fps")
        load_res, decode_res, proc_res, embed_res = await encoder._encode_video(
            video_items, req_id, task_id, sample_fps
        )
        return JSONResponse(content={
            "embedding_size": embed_res.mm_embedding.nbytes,
            "embedding_len": embed_res.mm_embedding.shape[0],
            "embedding_dim": embed_res.mm_embedding.shape[1],
            "load_cost": load_res.time_cost if load_res else 0.0,
            "frame_extraction_cost": decode_res.frame_extraction_cost,
            "download_cost": 0.0,
            "gpu_preprocess_cost": proc_res.time_cost,
            "embed_cost": embed_res.time_cost,
            "num_chunks": len(decode_res.chunks),
            "chunk_timestamps": decode_res.chunk_timestamps,
        })
    except Exception as e:
        logger.error(e, exc_info=True)
        return JSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            content={"status": "error", "message": str(e), "req_id": req_id},
        )


@app.post("/epd/transfer")
async def handle_epd_transfer_request(req: TransferRequest):
    """
    Handle mooncake transfer request from Prefill.

    Flow: encode (handles cache hit/miss) → transfer via executor → ack
    RAII on ScatteredTensor guarantees memory safety during transfer.
    """
    tc = transfer_endpoint_collector
    start_time = time.perf_counter()

    try:
        # Step 1: encode (内部处理 cache hit/miss)
        if not req.image_url:
            tc.on_request_end("400", time.perf_counter() - start_time)
            return req.make_ack(
                400, f"No image_url provided for task_id={req.task_id}",
            ).to_response()

        logger.info(f"[epd_transfer] req_id={req.req_id}, task_id={req.task_id}, "
                    f"modality={req.modality}, sample_fps={req.sample_fps}, "
                    f"image_url={req.image_url[:120]}")

        _, _, _, embed_res = await encoder.encode(
            mm_items=[req.image_url],
            req_id=req.req_id,
            num_parts=req.num_parts,
            part_idx=req.part_idx,
            task_id=req.task_id,
            modality=req.modality,
            sample_fps=req.sample_fps,
        )
        # 将 Tensor 拷贝到 pool 中，返回 handler(注意此方法线程不安全)
        embedding = encoder.mem_pool.place(embed_res.mm_embedding)

        grid_shape = (
            embed_res.image_grid_dim.shape
            if embed_res.image_grid_dim is not None else None
        )
        logger.info(
            f"[epd_transfer] encode done task_id={req.task_id}, "
            f"embedding_shape={embed_res.mm_embedding.shape}, "
            f"grid_dim_shape={grid_shape}"
        )

        # Step 2: Compute embed digest (optional)
        embed_digest = None
        if _ENABLE_EMBED_DIGEST:
            try:
                embed_digest = compute_embed_digest(embedding)
            except Exception as e:
                logger.warning(f"compute_embed_digest failed for task_id={req.task_id}: {e}")

        # Step 3: Transfer via executor
        record_transfer_attempt(req.session_id)
        tr: TransferResult = await asyncio.wrap_future(
            encoder.transfer_executor(TransferReq(
                rid=req.req_id,
                task_id=req.task_id,
                session_id=req.session_id,
                part_idx=req.part_idx,
                buffer_address=req.buffer_address,
                byte_size=req.byte_size,
                start_time=start_time,
                embed=embedding,
            ))
        )

        del embedding, embed_res   # 归还内存到pool

        if tr.status_code == 200:
            record_transfer_success(req.session_id)
            logger.info(f"[epd_transfer] transfer OK task_id={req.task_id}, "
                        f"transfer_cost_ms={tr.transfer_cost_ms:.1f}, "
                        f"total_cost_ms={tr.total_cost_ms:.1f}")
        else:
            logger.warning(f"[epd_transfer] transfer FAILED task_id={req.task_id}, "
                           f"status={tr.status_code}, msg={tr.message}")

        # Step 4: Build ack
        tc.on_request_end(str(tr.status_code), time.perf_counter() - start_time)
        return TransferAck(
            req_id=tr.rid,
            task_id=tr.task_id,
            part_idx=tr.part_idx,
            status_code=tr.status_code,
            message=tr.message,
            transfer_cost_ms=tr.transfer_cost_ms,
            total_cost_ms=tr.total_cost_ms,
            embed_digest=embed_digest,
        ).to_response()

    except Exception as e:
        logger.error(f"Transfer failed for req_id={req.req_id}, task_id={req.task_id}: {e}", exc_info=True)
        tc.on_request_end("500", time.perf_counter() - start_time)
        return req.make_ack(500, f"Transfer failed: {e}").to_response()




@app.post("/flush_cache")
async def flush_cache():
    """Flush the mm_cache and return how many bytes / entries were cleared."""
    if encoder is None:
        return JSONResponse(status_code=503, content={"error": "encoder not ready"})
    if encoder.mm_cache is None:
        return JSONResponse(
            status_code=400,
            content={"error": "mm_cache is not enabled (enable_prefix_mm_cache=False)"},
        )

    before_bytes = encoder.mm_cache.current_size
    before_entries = len(encoder.mm_cache)

    result = encoder.mm_cache.clear()

    after_bytes = encoder.mm_cache.current_size
    after_entries = len(encoder.mm_cache)

    flushed_bytes = before_bytes - after_bytes
    flushed_entries = before_entries - after_entries

    logger.info(
        f"[flush_cache] flushed {flushed_entries} entries, {flushed_bytes} bytes. "
        f"remaining: {after_entries} entries, {after_bytes} bytes"
    )

    return JSONResponse(content={
        "flushed_bytes": flushed_bytes,
        "flushed_entries": flushed_entries,
        "remaining_bytes": after_bytes,
        "remaining_entries": after_entries,
    })


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)



def _stopped_by_external():
    """
    Check if the stop signal is triggered by reading a signal file.
    """
    signal_path = os.environ.get("AIAK_STOP_SIGNAL_DIR", "/run/sglang/stop")
    if not signal_path:
        return False

    try:
        if os.path.exists(signal_path):
            with open(signal_path, 'r') as file:
                return file.read().strip().lower() == "stop"
    except Exception as e:
        logger.info(f"Error reading stop signal file: {e}")

    return False


@app.get("/ready")
@app.get("/health")
@app.get("/health_forward")
@app.get("/health_generate")
async def health_generate(request: Request):
    """
    Encoder 健康检查端点（时间窗模式 + 探活机制 + transfer 健康）。

    检查流程：
    1. encoder 未初始化 → 503
    2. transfer 持续失败（有请求但一直不成功超过阈值）→ 503
    3. 最近有产出（< 1分钟）→ 直接返回 200
    4. 长时间无产出（> 1分钟）→ 发送探活请求
    5. 在时间窗内等待响应
    6. 时间窗内收到响应 → 200，否则 → 503

    Returns:
        200: Encoder 健康
        503: Encoder 未初始化或不健康
    """
    if encoder is None:
        return Response(status_code=503, content="Encoder is not initialized")

    if not ENCODER_WARMUP_COMPLETE:
        return Response(status_code=503, content="Encoder warmup is in progress")
    
    if request.url.path.rstrip("/") == "/ready" and _stopped_by_external():
        logger.warn("Server is shutting down, graceful stop initiated")
        if encode_workflow is not None and not encode_workflow.stopped:
            asyncio.create_task(_graceful_shutdown())
        return Response(content="Encoder is shutting down", status_code=200)

    # Transfer 健康检查（所有 session 都持续失败 → 大概率本地 RDMA 问题）
    if not transfer_health_state.is_healthy():
        bad = transfer_health_state.unhealthy_sessions()
        logger.error(
            f"Health check failed: transfer unhealthy "
            f"(unhealthy_sessions={bad}, "
            f"last_attempt={transfer_health_state.last_transfer_attempt:.1f}, "
            f"last_success={transfer_health_state.last_transfer_success:.1f})"
        )
        return Response(status_code=503)

    # Polling loop 存活检查
    if encode_workflow is not None and not encode_workflow.is_alive:
        logger.error("Health check failed: vit-fetch polling loop is not alive")
        return Response(status_code=503, content="vit-fetch polling loop is dead")

    # 记录健康检查开始时间
    tic = time.time()

    # 检查 encoder 输出（使用时间窗）
    # 如果最近 PROBE_TRIGGER_THRESHOLD 秒内有产出，直接返回健康
    if encoder_health_state.has_output_since(tic - PROBE_TRIGGER_THRESHOLD):
        return Response(status_code=200)

    # 超过 PROBE_TRIGGER_THRESHOLD 秒无产出，需要探活
    time_since_output = time.time() - get_encoder_output_timestamp()
    logger.info(
        f"Encoder no output for {time_since_output:.1f}s (threshold: {PROBE_TRIGGER_THRESHOLD}s), "
        "sending probe..."
    )

    # 发送探活请求（异步）
    probe_task = asyncio.create_task(send_probe_request(encoder))

    # 在时间窗内等待响应
    while time.time() < tic + HEALTH_CHECK_TIMEOUT:
        await asyncio.sleep(1)

        # 检查是否有新的产出（时间窗模式）
        if encoder_health_state.has_output_since(tic):
            probe_task.cancel()
            logger.info("Health check passed: encoder responded in time window")
            return Response(status_code=200)

    # 时间窗内无响应
    probe_task.cancel()
    last_output_time = time.strftime(
        "%H:%M:%S", time.localtime(get_encoder_output_timestamp())
    )
    logger.error(
        f"Health check failed: no response within {HEALTH_CHECK_TIMEOUT}s. "
        f"Last output time: {last_output_time}"
    )
    return Response(status_code=503)


@app.get("/health/simple")
async def health_simple():
    """
    Simple health check endpoint for basic liveness probe.
    Returns 200 if encoder is initialized, 503 otherwise.
    """
    if encoder is None:
        return Response(status_code=503)
    return Response(status_code=200)


@app.get("/health/transfer")
async def health_transfer():
    """Return per-session transfer health details."""
    return JSONResponse(
        status_code=200,
        content={
            "healthy": transfer_health_state.is_healthy(),
            "unhealthy_sessions": transfer_health_state.unhealthy_sessions(),
            "sessions": transfer_health_state.snapshot(),
        },
    )


@app.api_route("/start_profile", methods=["GET", "POST"])
async def start_profile_async(obj: Optional[ProfileReq] = None):
    """Start profiling."""
    if encoder is None:
        return Response(content="encoder not ready\n", status_code=503)
    req = None
    if obj is None:
        req = ProfileReq(ProfileReqType.START_PROFILE)
    else:
        req = ProfileReq(
            type=ProfileReqType.START_PROFILE,
            output_dir=obj.output_dir,
            start_step=obj.start_step,
            num_steps=obj.num_steps,
            activities=obj.activities,
            with_stack=obj.with_stack,
            record_shapes=obj.record_shapes,
            profile_by_stage=obj.profile_by_stage,
            profile_id=str(time.time()),
            merge_profiles=obj.merge_profiles,
            profile_prefix=obj.profile_prefix,
            profile_stages=obj.profile_stages,
        )
    for socket in send_sockets:
        socket.send_pyobj(req)
    if encoder.profiler is None:
        encoder.profiler = EncoderProfiler(encoder.rank)
    ok, msg = encoder.profiler.start(req)
    if ok:
        detail = (
            f"Start profiling. output_dir={encoder.profiler.output_dir} "
            f"profile_id={encoder.profiler.profile_id}\n"
        )
        return Response(content=detail, status_code=200)
    return Response(
        content=(msg or "Start profiling failed.\n"), status_code=HTTPStatus.BAD_REQUEST
    )


@app.api_route("/list_vit_caches", methods=["GET", "POST"])
async def handle_list_vit_caches():
    """
    Interface 3: GET/POST /list_vit_caches

    Called periodically by vit-cache-collector to pull the current embedding
    cache inventory from this Encoder instance.
    """
    return await vit_cache_handler.list_vit_caches()


@app.api_route("/stop_profile", methods=["GET", "POST"])
async def stop_profile_async():
    """Stop profiling and return results."""
    if encoder is None:
        return Response(content="encoder not ready\n", status_code=503)
    if encoder.profiler is None:
        return Response(
            content="profiling not initialized\n", status_code=HTTPStatus.BAD_REQUEST
        )
    req = ProfileReq(ProfileReqType.STOP_PROFILE)
    for socket in send_sockets:
        socket.send_pyobj(req)
    ok, msg = encoder.profiler.stop()
    if ok:
        return Response(content="Stop profiling.\n", status_code=200)
    return Response(
        content=(msg or "Stop profiling failed.\n"), status_code=HTTPStatus.BAD_REQUEST
    )



@app.on_event("startup")
async def start_polling_task():
    """Start the background polling task on server startup."""
    global encode_workflow, _encode_poll_task
    assert encoder is not None

    if not encoder.server_args.vit_server_url:
        logger.info("vit_server_url not configured, skip EncodeWorkflow polling")
        return

    encode_workflow = EncodeWorkflow(encoder)
    encoder.encode_workflow = encode_workflow
    _encode_poll_task = asyncio.create_task(encode_workflow.poll_forever())


async def _graceful_shutdown():
    """封装优雅停止流程：graceful_stop → 等待 prefill 拉取 → encoder.shutdown()

    1. graceful_stop() 停止 polling 循环
    2. poll_forever() 退出后执行等待逻辑（确保 prefill 有时间拉取 embedding）
    3. 等待完成后关闭 encoder
    """
    # Step 1: 停止 polling
    encode_workflow.graceful_stop()

    # Step 2: 等待 poll_forever 完成（包含等待逻辑）
    if _encode_poll_task is not None:
        await _encode_poll_task
        _encode_poll_task = None

    # Step 3: 关闭 encoder
    if encoder is not None:
        encoder.shutdown()



@app.on_event("shutdown")
async def shutdown_event():
    """非 K8s 场景（如 Ctrl+C）的关闭处理。

    注意：
        K8s 部署下（preStop hook 使用 ``pkill -9``）此事件不会被触发，
        因为 SIGKILL 无法被捕获。优雅退出走 /ready → _graceful_shutdown() 路径。

        此 handler 仅在进程收到 SIGTERM/SIGINT 时触发，例如本地开发时按 Ctrl+C。
        此处做的是尽力清理，不包含完整的优雅退出等待逻辑。
    """
    logger.info("FastAPI shutdown event triggered")
    if encode_workflow is not None:
        encode_workflow.stop()
    if encoder is not None:
        encoder.shutdown()
