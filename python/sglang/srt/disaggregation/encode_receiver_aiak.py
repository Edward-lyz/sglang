"""
AIAK MM receiver for processing multimodal embeddings.

This module provides classes for receiving and processing multimodal
embeddings from encoders in disaggregated mode.
"""
import asyncio
import logging
import pickle
import random
import uuid
from abc import ABC, abstractmethod
from http import HTTPStatus
from typing import TYPE_CHECKING, List, Optional

import aiohttp
import torch
import zmq
import zmq.asyncio
from transformers import PretrainedConfig

# Re-exported for pickle back-compat: EmbeddingData payloads produced by an
# older encode server name this module.  Safe to delete once E and P are both
# past this commit.
from sglang.srt.disaggregation.embedding_data import EmbeddingData  # noqa: F401
from sglang.srt.disaggregation.embedding_transfer.mooncake_transport import (
    _MooncakeBufferManager,
)
from sglang.srt.disaggregation.embedding_transfer.request import WaitingImageRequest
from sglang.srt.distributed.parallel_state import (
    GroupCoordinator,
    get_mooncake_transfer_engine,
)
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from sglang.srt.managers.multimodal_processor import get_mm_processor, import_processors
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.metrics.encode_collector import MMReceiverMetricsCollector
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.network import get_local_ip_auto, get_zmq_socket_on_host
from sglang.srt.utils.hf_transformers_utils import get_processor

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


def _determine_tensor_transport_mode(server_args):
    is_cross_node = server_args.dist_init_addr

    if is_cross_node:
        # Fallback to default CPU transport for multi-node
        return "default"
    else:
        return "cuda_ipc"


class MMReceiverBase(ABC):
    """Base class for multimodal receiver."""

    def __init__(
        self,
        server_args: ServerArgs,
        dtype: Optional[torch.dtype] = None,
        hf_config: Optional[PretrainedConfig] = None,
        pp_rank: Optional[int] = None,
        tp_rank: Optional[int] = None,
        tp_group: Optional[GroupCoordinator] = None,
        scheduler: Optional["Scheduler"] = None,
    ):
        pass

    @abstractmethod
    def process_waiting_requests(self, recv_reqs):
        """Split recv_reqs into (non_mm_recv_reqs, mm_waiting_reqs, abort_reqs).

        Non-blocking: no transport poll and no collective happens here.
        """
        pass

    @abstractmethod
    async def recv_mm_data(self, img_data, mm_processor, prompt):
        """Receive multimodal data."""
        pass

    @abstractmethod
    def send_encode_request(self, obj, time_stats_json=None):
        """Send encode request."""
        pass

    def abort_waiting_request(self, recv_req) -> None:
        """Abort waiting requests matching recv_req. Default no-op."""
        pass


class MMReceiverHTTP(MMReceiverBase):
    """HTTP-based multimodal receiver."""

    def __init__(
        self,
        server_args: ServerArgs,
        dtype: Optional[torch.dtype] = None,
        hf_config: Optional[PretrainedConfig] = None,
        pp_rank: Optional[int] = None,
        tp_rank: Optional[int] = None,
        tp_group: Optional[GroupCoordinator] = None,
        scheduler: Optional["Scheduler"] = None,
        encode_urls: Optional[List[str]] = None,
    ):
        self.context = zmq.asyncio.Context(20)
        self.encoder_transfer_backend = server_args.encoder_transfer_backend
        self.enable_mooncake_in_zmq_mode = server_args.enable_mooncake_in_zmq_mode
        self.encode_urls = (
            encode_urls if encode_urls is not None else list(server_args.encoder_urls)
        )
        self.encode_idx = list(range(len(self.encode_urls)))
        self.host = get_local_ip_auto(server_args.host)
        if self.encoder_transfer_backend == "mooncake":
            self.dtype = dtype
            self.embeddings_engine = get_mooncake_transfer_engine()
            self.embeddings_buffer = dict()
        elif self.encoder_transfer_backend == "zmq_to_scheduler":
            # Initialize dtype for mooncake transfer (used when enable_mooncake_in_zmq_mode=True)
            self.dtype = dtype
            self.pp_rank = pp_rank
            self.tp_rank = tp_rank
            self.enable_dp_attention = server_args.enable_dp_attention
            self.tp_group = tp_group
            self.nnodes = server_args.nnodes
            self.scheduler = scheduler
            self.wait_timeout = envs.SGLANG_ENCODER_RECV_TIMEOUT.get()
            self.mm_receiver_metrics_collector = None
            if (
                self.scheduler is not None
                and self.scheduler.enable_metrics
                and hasattr(self.scheduler, "metrics_collector")
            ):
                labels = dict(self.scheduler.metrics_collector.labels)
                labels["mm_transfer_mode"] = "zmq_to_scheduler"
                self.mm_receiver_metrics_collector = MMReceiverMetricsCollector(
                    labels=labels
                )
                self.mm_receiver_metrics_collector.set_waiting_requests(0)
            if hf_config is not None:
                transport_mode = _determine_tensor_transport_mode(server_args)
                import_processors("sglang.srt.multimodal.processors")
                _processor = None
                try:
                    _processor = get_processor(
                        server_args.tokenizer_path,
                        tokenizer_mode=server_args.tokenizer_mode,
                        trust_remote_code=server_args.trust_remote_code,
                        revision=server_args.revision,
                        use_fast=not server_args.disable_fast_image_processor,
                    )
                except ValueError as e:
                    error_message = str(e)
                    if "does not have a slow version" in error_message:
                        logger.info(
                            "Processor %s does not have a slow version. "
                            "Automatically use fast version",
                            server_args.tokenizer_path,
                        )
                        _processor = get_processor(
                            server_args.tokenizer_path,
                            tokenizer_mode=server_args.tokenizer_mode,
                            trust_remote_code=server_args.trust_remote_code,
                            revision=server_args.revision,
                            use_fast=True,
                        )
                    else:
                        raise e
                self.mm_processor = get_mm_processor(
                    hf_config,
                    server_args,
                    _processor,
                    transport_mode,
                    skip_mm_pool=True,
                )

    def create_req(self, recv_req):
        """Create a request from receive request."""
        req = Req(
            recv_req.rid,
            recv_req.input_text,
            recv_req.input_ids,
            recv_req.sampling_params,
            return_logprob=recv_req.return_logprob,
            top_logprobs_num=recv_req.top_logprobs_num,
            token_ids_logprob=recv_req.token_ids_logprob,
            stream=recv_req.stream,
            lora_id=recv_req.lora_id,
            input_embeds=recv_req.input_embeds,
            custom_logit_processor=recv_req.custom_logit_processor,
            require_reasoning=recv_req.require_reasoning,
            return_hidden_states=recv_req.return_hidden_states,
            return_routed_experts=recv_req.return_routed_experts,
            eos_token_ids=self.scheduler.model_config.hf_eos_token_id,
            bootstrap_host=recv_req.bootstrap_host,
            bootstrap_port=recv_req.bootstrap_port,
            bootstrap_room=recv_req.bootstrap_room,
            disagg_mode=self.scheduler.disaggregation_mode,
            routed_dp_rank=recv_req.routed_dp_rank,
            disagg_prefill_dp_rank=recv_req.disagg_prefill_dp_rank,
            vocab_size=self.scheduler.model_config.vocab_size,
            priority=recv_req.priority,
            metrics_collector=(
                self.scheduler.metrics_collector
                if self.scheduler.enable_metrics
                else None
            ),
            http_worker_ipc=recv_req.http_worker_ipc,
            dllm_config=self.scheduler.dllm_config,
        )
        req.tokenizer = self.scheduler.tokenizer
        req.tokenize_only = getattr(recv_req, "tokenize_only", False)
        return req

    def abort_waiting_request(self, recv_req) -> None:
        """Abort requests matching the AbortReq across both staged queues."""
        all_waiting: List[WaitingImageRequest] = []
        if self.scheduler is not None:
            prealloc_queue = self.scheduler.disagg_prefill_mm_prealloc_queue
            transfer_queue = self.scheduler.disagg_prefill_mm_transfer_queue
            if prealloc_queue is not None:
                all_waiting.extend(prealloc_queue.queue)
            if transfer_queue is not None:
                all_waiting.extend(transfer_queue.queue)
        for waiting_req in all_waiting:
            rid = waiting_req.recv_req.rid
            if recv_req.abort_all or rid.startswith(recv_req.rid):
                waiting_req.mark_failed(
                    "Aborted by upstream disconnect",
                    499,
                )
                logger.info(f"[trace] mm_receiver req {rid} abort_by_upstream")

    def _get_sync_group(self) -> GroupCoordinator:
        """Resolve the attn-tp sync group for status / embedding broadcast."""
        if self.enable_dp_attention and self.scheduler is not None:
            return self.scheduler.attn_tp_group
        return self.tp_group

    def _is_first_rank(self) -> bool:
        """Whether the current rank is rank 0 of the attn-tp sync group."""
        sync_group = self._get_sync_group()
        if sync_group is None or sync_group.world_size <= 1:
            # Single-rank: every rank is its own first rank.
            return True
        return sync_group.rank_in_group == 0

    def sync_ready_flags(self, ready_flags: List[int]) -> List[int]:
        """Agree on a per-request readiness bit list across the sync group.

        Mirrors ``poll_and_all_reduce`` on the PD path: a staged queue may only
        advance on a condition that just one rank observes (here: rank 0 got its
        pull requests out).  A MIN reduction makes every rank wait for the
        slowest, so the per-rank queues cannot drift apart and the next
        iteration's collectives run against identical request lists.
        """
        sync_group = self._get_sync_group()
        if sync_group is None or sync_group.world_size <= 1 or not ready_flags:
            return ready_flags
        packed = torch.tensor(ready_flags, dtype=torch.uint8, device="cpu")
        torch.distributed.all_reduce(
            packed,
            op=torch.distributed.ReduceOp.MIN,
            group=sync_group.cpu_group,
        )
        return packed.tolist()

    def _report_buffer_pool_metrics(self) -> None:
        """Report pinned embedding pool utilization if available.

        The metric names say "gpu" for historical reasons; the pool is CPU
        pinned memory (see _MooncakeBufferManager).
        """
        if self.mm_receiver_metrics_collector is None or not self.enable_mooncake_in_zmq_mode:
            return
        # Followers do not register a Mooncake buffer pool; touching the
        # singleton would force the multi-GB pinned buffer to be allocated
        # (SGLANG_MC_POOL_GB, default 2).
        if not self._is_first_rank():
            return
        try:
            buf_mgr = _MooncakeBufferManager.get_instance()
            pool = buf_mgr._pool
            self.mm_receiver_metrics_collector.set_gpu_buffer_pool_used_bytes(
                pool.used_bytes()
            )
            self.mm_receiver_metrics_collector.set_gpu_buffer_pool_total_bytes(
                pool.total_bytes
            )
        except Exception:
            pass

    # For zmq_to_scheduler
    def process_waiting_requests(self, recv_reqs):
        """Non-blocking dispatch: separate MM requests from normal requests.

        MM requests are returned as a separate list of WaitingImageRequest
        objects.  The caller (scheduler) is responsible for adding them to
        MMPreallocQueue.  No transport poll and no collective happens here.

        Returns:
            (non_mm_recv_reqs, mm_waiting_reqs, abort_reqs)
        """
        self._report_buffer_pool_metrics()
        new_recv_reqs = []
        mm_waiting_reqs: List[WaitingImageRequest] = []
        abort_reqs = []
        for recv_req in recv_reqs:
            if (
                isinstance(recv_req, TokenizedGenerateReqInput)
                and recv_req.need_wait_for_mm_inputs is True
            ):
                waiting_req = self._build_waiting_request(recv_req, abort_reqs)
                if waiting_req is not None:
                    mm_waiting_reqs.append(waiting_req)
            else:
                new_recv_reqs.append(recv_req)
        return new_recv_reqs, mm_waiting_reqs, abort_reqs

    def _build_waiting_request(
        self, recv_req, abort_reqs
    ) -> Optional[WaitingImageRequest]:
        """Construct a WaitingImageRequest from a tokenized request.

        Returns:
            WaitingImageRequest on success, None if the request is invalid and
            should be aborted immediately.
        """
        logger.info(f"[trace] mm_receiver req {recv_req.rid} received_by_scheduler")
        if not recv_req.task_results:
            logger.info(
                f"[trace] mm_receiver req {recv_req.rid} missing_task_results"
            )
            abort_reqs.append(
                (
                    self.create_req(recv_req),
                    "Missing task_results for zmq_to_scheduler pull mode.",
                    HTTPStatus.BAD_REQUEST,
                )
            )
            return None

        is_first_rank = self._is_first_rank()
        waiting_req = WaitingImageRequest(
            recv_req=recv_req,
            mm_processor=self.mm_processor,
            wait_timeout=self.wait_timeout,
            task_results=recv_req.task_results,
            is_first_rank=is_first_rank,
        )
        logger.info(
            f"[trace] mm_receiver req {recv_req.rid} added_to_mm_prealloc_queue: "
            f"task_results={len(recv_req.task_results)} is_first_rank={is_first_rank}"
        )
        return waiting_req

    async def encode(
        self,
        req_id,
        img_data,
        embedding_port,
        endpoint_encode,
        endpoint_send,
        num_items_assigned=None,
    ):
        """Encode image data."""
        if len(img_data) == 0:
            return

        # Split mm_items
        encode_requests = []
        if num_items_assigned is None:
            random.shuffle(self.encode_idx)
            num_items_assigned = [
                (idx + len(img_data)) // len(self.encode_urls)
                for idx in self.encode_idx
            ]
        num_parts = sum(1 for x in num_items_assigned if x != 0)
        cum_num_items = 0
        cum_idx = 0
        for idx, assigned_num in enumerate(num_items_assigned):
            if assigned_num == 0:
                continue
            encode_requests.append(
                {
                    "encoder_idx": idx,
                    "mm_items": img_data[cum_num_items : cum_num_items + assigned_num],
                    "num_parts": num_parts,
                    "part_idx": cum_idx,
                    "req_id": req_id,
                    "prefill_host": self.host,
                    "embedding_port": embedding_port,
                }
            )
            cum_idx += 1
            cum_num_items += assigned_num

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=1800
            )  # Add timeout for request reliability
        ) as session:
            # Send encode requests

            tasks = [
                session.post(
                    f"{self.encode_urls[encode_request['encoder_idx']]}/{endpoint_encode}",
                    json=encode_request,
                )
                for encode_request in encode_requests
            ]

            responses = await asyncio.gather(*tasks)
            for response in responses:
                if response.status != 200:
                    try:
                        err_data = await response.json()
                        msg = err_data.get("message", "Unknown encoder error")
                    except:
                        msg = await response.text()

                    logger.error(f"Encoder returned error {response.status}: {msg}")
                    return
            response_json_list_unsort = [
                await response.json() for response in responses
            ]

            # zmq backend: return is None
            if None in response_json_list_unsort:
                return

            # mooncake backend: send bootstrap info

            embedding_size_list_sort = [None for _ in range(num_parts)]
            embedding_length_tot = 0
            response_json_list_sort = [None for _ in range(num_parts)]
            for response_json in response_json_list_unsort:
                idx = response_json["part_idx"]
                embedding_size_list_sort[idx] = response_json["embedding_size"]
                embedding_length_tot += response_json["embedding_len"]
                response_json_list_sort[idx] = response_json

            offset = 0
            metadata_tasks = []
            buffer_address = await self.allocate_embedding_buffer(
                req_id,
                embedding_length_tot,
                response_json_list_sort[0]["embedding_dim"],
            )
            for idx in range(len(tasks)):
                response_json = response_json_list_sort[idx]
                buffer_address_adjust = offset + buffer_address
                response_json.update(
                    {
                        "session_id": self.embeddings_engine.session_id,
                        "buffer_address": buffer_address_adjust,
                    }
                )
                metadata_tasks.append(
                    session.post(
                        f"{self.encode_urls[response_json['encoder_idx']]}/{endpoint_send}",
                        json=response_json,
                    )
                )
                offset += embedding_size_list_sort[idx]
            await asyncio.gather(*metadata_tasks)

    # For mooncake
    async def allocate_embedding_buffer(self, req_id, embedding_length, embedding_dim):
        """Allocate embedding buffer for mooncake."""
        embeddings = torch.zeros(
            (embedding_length, embedding_dim),
            dtype=self.dtype,
        )
        self.embeddings_engine.register(
            embeddings.data_ptr(),
            embeddings.nbytes,
        )
        self.embeddings_buffer[req_id] = embeddings
        return embeddings.data_ptr()

    # For zmq_to_scheduler
    def send_encode_request(self, obj, time_stats_json=None):
        if self.encoder_transfer_backend == "zmq_to_scheduler":
            raise ValueError(
                "Legacy zmq_to_scheduler push flow was removed. "
                "Please provide task_results and use pull mode."
            )

    # For zmq_to_tokenizer and mooncake
    async def recv_mm_data(self, img_data, mm_processor, prompt):
        try:
            if len(self.encode_urls) == 0:
                return None
            req_id = uuid.uuid4().hex
            embedding_port, recv_socket = get_zmq_socket_on_host(self.context, zmq.PULL)
            if type(img_data) != list:
                img_data = [img_data.url]
            else:
                img_data = [img.url for img in img_data]
            asyncio.create_task(
                self.encode(req_id, img_data, embedding_port, "encode", "send")
            )
            return await asyncio.wait_for(
                self._recv_mm_data(req_id, recv_socket, mm_processor, prompt),
                timeout=20,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Embedding recv timeout for request {req_id}")
            if hasattr(self, "embeddings_buffer") and req_id in self.embeddings_buffer:
                del self.embeddings_buffer[req_id]
            return None

    # For zmq_to_tokenizer and mooncake
    async def _recv_mm_data(self, req_id, recv_socket, mm_processor, prompt):
        # Bypass MMReceiverHTTP
        if req_id is None:
            return None

        recv_embedding = None

        recv_embedding_data: EmbeddingData = None

        while recv_embedding_data is None or not recv_embedding_data.ready:
            parts = await recv_socket.recv_multipart(copy=False)

            recv_obj: EmbeddingData = pickle.loads(parts[0])
            logger.info(f"{recv_obj = }")
            if self.encoder_transfer_backend == "zmq_to_tokenizer":
                buffer = parts[1].buffer if hasattr(parts[1], "buffer") else parts[1]
                recv_obj.embedding = torch.frombuffer(
                    buffer, dtype=recv_obj.dtype
                ).reshape(recv_obj.shape)
            if recv_embedding_data is None:
                recv_obj.embedding_list[recv_obj.part_idx] = recv_obj.embedding
                recv_embedding_data = recv_obj
            else:
                recv_embedding_data.add(recv_obj)

        if self.encoder_transfer_backend == "mooncake":
            recv_embedding = self.embeddings_buffer[req_id]
            del self.embeddings_buffer[req_id]
            self.embeddings_engine.deregister(recv_embedding.data_ptr())
        elif self.encoder_transfer_backend == "zmq_to_tokenizer":
            recv_embedding = recv_embedding_data.get_embedding(is_concat=True)

        recv_socket.close()

        img_grid_thw = recv_embedding_data.get_img_grid()

        # get_mm_data() takes prompt/embeddings positionally and the rest as
        # kwargs (see BaseMultimodalProcessor.get_mm_data).
        mm_inputs = mm_processor.get_mm_data(
            prompt,
            recv_embedding,
            img_grid_thw=img_grid_thw,
            original_image_sizes=getattr(
                recv_embedding_data, "original_image_sizes", None
            ),
        )
        return mm_inputs


def _validate_transport_mode(transport_mode: str, encoder_urls):
    if transport_mode != "http":
        raise ValueError(
            "AIAK MMReceiver only supports http transport mode. "
            "Set SGLANG_ENCODER_MM_RECEIVER_MODE=http."
        )
    if any(url.startswith("grpc://") for url in encoder_urls):
        raise ValueError(
            "AIAK MMReceiver http mode requires http:// encoder URLs. "
            "Set SGLANG_ENCODER_MM_RECEIVER_MODE=grpc only when using the non-AIAK receiver."
        )


_MM_RECEIVER_BY_MODE = {
    "http": MMReceiverHTTP,
}


def create_mm_receiver(
    server_args: ServerArgs,
    dtype: Optional[torch.dtype] = None,
    hf_config: Optional[PretrainedConfig] = None,
    pp_rank: Optional[int] = None,
    tp_rank: Optional[int] = None,
    tp_group: Optional[GroupCoordinator] = None,
    scheduler: Optional["Scheduler"] = None,
    transport_mode: Optional[str] = None,
    encode_urls: Optional[List[str]] = None,
):
    """Create multimodal receiver instance.

    Args:
        server_args: Server configuration arguments.
        dtype: Optional data type for embeddings.
        hf_config: Optional HuggingFace configuration.
        pp_rank: Optional pipeline parallel rank.
        tp_rank: Optional tensor parallel rank.
        tp_group: Optional tensor parallel group coordinator.
        scheduler: Optional scheduler instance.
        transport_mode: Optional transport mode (e.g., "http").

    Returns:
        MMReceiver instance.
    """
    if transport_mode is None:
        transport_mode = envs.SGLANG_ENCODER_MM_RECEIVER_MODE.get()
        logger.debug(f"AIAK MMReceiver transport_mode from env: {transport_mode}")

    _validate_transport_mode(transport_mode, encode_urls or server_args.encoder_urls)
    logger.info(f"AIAK MMReceiver: using transport_mode={transport_mode}")

    receiver_cls = _MM_RECEIVER_BY_MODE.get(transport_mode)
    if receiver_cls is None:
        raise ValueError(f"Unsupported AIAK MMReceiver transport_mode: {transport_mode}")
    return receiver_cls(
        server_args,
        dtype=dtype,
        hf_config=hf_config,
        pp_rank=pp_rank,
        tp_rank=tp_rank,
        tp_group=tp_group,
        scheduler=scheduler,
        encode_urls=encode_urls,
    )
