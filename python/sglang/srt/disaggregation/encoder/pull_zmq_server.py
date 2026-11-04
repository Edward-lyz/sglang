"""ZMQ ROUTER pull server for the factory (厂内) EPD path.

When an ``--encoder-only`` server polls a vit-server (``--vit-server-url``), it
holds each computed embedding by ``task_id`` (see
``MMEncoder.store_task_embedding``). Prefill then pulls it with a DEALER socket,
exactly as in Kimi-K3: the wire framing here is identical, only the embedding
lookup is adapted to the New EPD encoder's task-keyed hold instead of K3's
``mm_cache.acquire``.

Framing (matches ``factory_transport.FactoryZmqTransport``):
  Prefill DEALER sends: ``[empty, json_payload]``
  ROUTER receives:      ``[identity, empty, json_payload]``
  ROUTER replies:       ``[identity, empty, task_id, req_id, pickle(meta)(, buffer)]``
"""

import asyncio
import json
import logging
import pickle
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from typing import Set

import zmq.asyncio

from sglang.srt.disaggregation.encode_health import (
    record_transfer_attempt,
    record_transfer_success,
)

logger = logging.getLogger(__name__)


class PullZmqServer:
    """ROUTER server answering prefill's pull_embedding requests by task_id."""

    MAX_CONCURRENT_HANDLES = 256

    def __init__(self, encoder, context, pull_zmq_port: int):
        """Bind to ``pull_zmq_port``; serve embeddings held on ``encoder``."""
        self.encoder = encoder
        self.context = context
        self.pull_zmq_port = pull_zmq_port
        self._tasks: Set[asyncio.Task] = set()
        self._semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_HANDLES)
        self._transfer_executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="zmq-cpu-xfer"
        )

    async def build_pull_reply(self, task_id, req_id=None):
        """Return ``([pickle(meta)], acquired)`` or ``([pickle(meta), buf], True)``.

        ``meta`` carries ``shape``/``dtype``/``image_grid_thw`` on success, or
        ``error_msg``/``error_code`` when the task is not held. The buffer frame
        is a zero-copy view of the held (CPU, contiguous) embedding; the ref is
        released by the caller once the send returns.
        """
        from sglang.srt.disaggregation.encoder.server import TensorWrapper

        entry = await self.encoder.acquire_task_embedding(task_id)
        if entry is None:
            meta = {
                "error_msg": f"embedding not held for task_id={task_id}",
                "error_code": int(HTTPStatus.NOT_FOUND),
            }
            return [pickle.dumps(meta)], False

        embedding = entry.embedding
        meta = {
            "shape": list(embedding.shape),
            "dtype": embedding.dtype,
            "image_grid_thw": entry.image_grid_thw,
            "original_image_sizes": entry.original_image_sizes,
        }
        return [pickle.dumps(meta), TensorWrapper(embedding).__buffer__()], True

    async def run(self):
        """Serve pull requests; each is handled in its own task (ROUTER)."""
        sock = self.context.socket(zmq.ROUTER)
        sock.bind(f"tcp://*:{self.pull_zmq_port}")
        logger.info("ZMQ ROUTER pull server bound on port %d", self.pull_zmq_port)
        try:
            while True:
                try:
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

    async def _handle(self, sock, identity, msg):
        """Answer one pull request, releasing the task ref afterwards."""
        record_transfer_attempt()
        task_id = msg.get("task_id")
        req_id = msg.get("req_id")
        acquired = False
        try:
            async with self._semaphore:
                logger.info("ZMQ pull req req_id=%s task_id=%s", req_id, task_id)
                parts, acquired = await self.build_pull_reply(task_id, req_id)
                await sock.send_multipart(
                    [identity, b"", (task_id or "").encode(), (req_id or "").encode()]
                    + parts,
                    copy=True,
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
                await self.encoder.release_task_embedding(task_id)
