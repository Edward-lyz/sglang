"""
Pull ZMQ server module for handling embedding pull requests from Prefill.

This module provides ZMQ ROUTER server functionality that allows Prefill
instances to pull embedding data from Encoder via ZMQ DEALER/ROUTER pattern.
ROUTER supports concurrent request handling, unlike the old REP which was
strictly serial.
"""

import asyncio
import json
import logging
import pickle
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from typing import Set

import zmq.asyncio

from sglang.srt.disaggregation.embedding_data import EmbeddingData
from sglang.srt.disaggregation.encode_health import (
    record_transfer_attempt,
    record_transfer_success,
)
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult

logger = logging.getLogger(__name__)


class PullZmqServer:
    """
    ZMQ ROUTER server for pull_embedding requests from Prefill.

    Binds a ROUTER socket on pull_zmq_port.
    Prefill connects with a DEALER socket and sends:
      [empty_frame, json_payload]
    ROUTER receives:
      [identity, empty_frame, json_payload]
    ROUTER replies:
      [identity, empty_frame, task_id_frame, req_id_frame, meta_frame, (opt)buffer_frame]
    """

    MAX_CONCURRENT_HANDLES = 256

    def __init__(self, encoder, context, pull_zmq_port: int):
        """
        Args:
            encoder: The MMEncoder instance that holds the embedding cache.
            context: The ZMQ async context for creating sockets.
            pull_zmq_port: The port number for the ZMQ ROUTER server.
        """
        self.encoder = encoder
        self.context = context
        self.pull_zmq_port = pull_zmq_port
        self._tasks: Set[asyncio.Task] = set()
        self._semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_HANDLES)
        self._transfer_executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="zmq-cpu-xfer"
        )

    async def build_pull_reply(
        self,
        task_id: str,
        req_id: str = None,
        image_url: str = None,
        num_parts: int = 1,
        part_idx: int = 0,
    ):
        """
        Build the ZMQ multipart reply for a pull_embedding request.

        Args:
            task_id: The task ID to retrieve the embedding for.
            req_id: The request ID.
            image_url: The image URL for generating embedding if cache miss.
            num_parts: Number of parts for the image.
            part_idx: Part index for the image.

        Returns:
          [pickle(EmbeddingData)]               — on error / not found
          [pickle(EmbeddingData), buffer]        — on success
        """
        from sglang.srt.disaggregation.encode_server_aiak import TensorWrapper

        # Get embedding from mm_cache using task_id directly
        # Use acquire() to prevent eviction during transfer
        cache_entry = None
        acquired = False

        # Access mm_cache from encoder directly
        cache_entry = (
            None
            if self.encoder.mm_cache is None
            else self.encoder.mm_cache.acquire(task_id)
        )
        if cache_entry is not None:
            acquired = True
            if not self.encoder.validate_cached_video(task_id, cache_entry):
                self.encoder.mm_cache.release(task_id)
                acquired = False
                error_data = EmbeddingData(
                    req_id=req_id if req_id else task_id,
                    num_parts=num_parts,
                    part_idx=part_idx,
                    image_grid_dim=None,
                    embedding=None,
                    error_msg=f"Video metadata expired for task_id={task_id}; refresh task_results",
                    error_code=HTTPStatus.CONFLICT,
                )
                return [pickle.dumps(error_data)], False

        if cache_entry is None:
            # Try to generate embedding if image_url is provided
            if image_url:
                logger.info(
                    f"Generating embedding for task_id={task_id} using image_url={image_url}"
                )
                try:
                    # Call encoder.encode to generate embedding
                    load_res, decode_res, proc_res, embed_res = (
                        await self.encoder.encode(
                            mm_items=[image_url],
                            req_id=req_id if req_id else task_id,
                            num_parts=num_parts,
                            part_idx=part_idx,
                            task_id=task_id,
                            source="zmq_pull",
                        )
                    )
                    # Retry to acquire embedding from cache after encoding
                    cache_entry = EmbeddingResult(
                        embedding=embed_res.mm_embedding,
                        image_grid_thw=embed_res.image_grid_dim,
                        image_url=image_url,
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to generate embedding for task_id={task_id}: {e}",
                        exc_info=True,
                    )
                    error_data = EmbeddingData(
                        req_id=req_id if req_id else task_id,
                        num_parts=num_parts,
                        part_idx=part_idx,
                        image_grid_dim=None,
                        embedding=None,
                        error_msg=f"Failed to generate embedding: {str(e)}",
                        error_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    )
                    return [pickle.dumps(error_data)], False

            # If still no cache entry or no image_url provided, return error
            if cache_entry is None:
                error_data = EmbeddingData(
                    req_id=req_id if req_id else task_id,
                    num_parts=num_parts,
                    part_idx=part_idx,
                    image_grid_dim=None,
                    embedding=None,
                    error_msg=f"Embedding not found in mm_cache for task_id={task_id}"
                    + (
                        "" if image_url else " and no image_url provided to generate it"
                    ),
                    error_code=(
                        HTTPStatus.NOT_FOUND
                        if not image_url
                        else HTTPStatus.INTERNAL_SERVER_ERROR
                    ),
                )
                return [pickle.dumps(error_data)], False

        embedding = cache_entry.embedding

        # Move embedding to CPU in thread pool to avoid blocking event loop
        # This frees GPU memory and ensures contiguous for transfer
        if embedding.is_cuda:
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                self._transfer_executor, embedding.cpu
            )
        if not embedding.is_contiguous():
            embedding = embedding.contiguous()

        # Note: We intentionally DO NOT release the reference here.
        # The reference is held until the transfer is confirmed complete.
        # For ZMQ with copy=False, we need to ensure the tensor outlives the send.

        # Success: create EmbeddingData with embedding=None but set dtype/shape
        # to avoid serializing the embedding twice
        meta = EmbeddingData(
            req_id=req_id if req_id else task_id,
            num_parts=num_parts,
            part_idx=part_idx,
            image_grid_dim=cache_entry.image_grid_thw,
            embedding=None,
            error_msg=None,
            error_code=None,
        )
        # Manually set dtype and shape from the embedding
        meta.dtype = embedding.dtype
        meta.shape = list(embedding.shape)

        # Send [meta, buffer] where buffer contains the embedding
        return [pickle.dumps(meta), TensorWrapper(embedding).__buffer__()], acquired

    async def run(self):
        """
        Run the ZMQ ROUTER server loop.

        ROUTER allows concurrent request handling: each incoming request is
        dispatched to an asyncio task so that slow build_pull_reply calls
        (e.g. cache-miss encoding) do not block other requests.
        """
        sock = self.context.socket(zmq.ROUTER)
        sock.bind(f"tcp://*:{self.pull_zmq_port}")
        logger.info("ZMQ ROUTER pull server bound on port %d", self.pull_zmq_port)
        try:
            while True:
                try:
                    # ROUTER recv: [identity, empty_frame, json_payload]
                    frames = await sock.recv_multipart()
                except Exception as e:
                    logger.error("ZMQ ROUTER recv error: %s", e, exc_info=True)
                    continue

                if len(frames) < 3:
                    logger.warning(
                        "Malformed DEALER message: expected >=3 frames, got %d",
                        len(frames),
                    )
                    continue

                identity = frames[0]
                # frames[1] is the empty delimiter
                try:
                    msg = json.loads(frames[2])
                except Exception as e:
                    logger.error("Failed to parse JSON payload: %s", e)
                    continue

                task = asyncio.create_task(self._handle(sock, identity, msg))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
        finally:
            sock.close()
            logger.info("ZMQ ROUTER pull server closed")

    async def _handle(self, sock, identity: bytes, msg: dict):
        """Handle a single pull request from a Prefill DEALER client."""
        record_transfer_attempt()
        task_id = msg.get("task_id")
        req_id = msg.get("req_id")
        acquired = False
        try:
            async with self._semaphore:
                image_url = msg.get("image_url")
                num_parts = msg.get("num_parts", 1)
                part_idx = msg.get("part_idx", 0)
                logger.info("ZMQ pull req req_id=%s task_id=%s", req_id, task_id)

                parts, acquired = await self.build_pull_reply(
                    task_id, req_id, image_url, num_parts, part_idx
                )

                # ROUTER send: [identity, empty, task_id_frame, req_id_frame, ...reply_parts]
                task_id_frame = (task_id or "").encode()
                req_id_frame = (req_id or "").encode()
                await sock.send_multipart(
                    [identity, b"", task_id_frame, req_id_frame] + parts, copy=True
                )
                record_transfer_success()
                logger.info(
                    "ZMQ pull reply sent task_id=%s has_embedding=%s",
                    task_id,
                    len(parts) == 2,
                )
        except Exception as e:
            logger.error(
                "ZMQ _handle error task_id=%s req_id=%s: %s",
                task_id,
                req_id,
                e,
                exc_info=True,
            )
        finally:
            if acquired and task_id:
                self.encoder.mm_cache.release(task_id)
                logger.debug("[zmq] Released reference to task_id=%s", task_id)
