"""
VIT cache handler module for managing embedding cache inventory.

This module provides endpoints for listing and managing VIT caches
used by the vit-cache-collector.
"""

import logging
import os

import torch
from fastapi.responses import JSONResponse

from sglang.srt.disaggregation.encode_utils import normalize_transfer_type

logger = logging.getLogger(__name__)


class VitCacheHandler:
    """Handler for VIT cache inventory management."""

    def __init__(self, encoder):
        """
        Initialize VIT cache handler.

        Args:
            encoder: The MMEncoder instance that holds the embedding cache.
        """
        self.encoder = encoder

    async def list_vit_caches(self):
        """
        Get the current embedding cache inventory.

        Design: mm_cache stores EmbeddingResult which contains embedding and image_grid_thw.

        Response fields:
          model_id (str)    — model path / identifier
          caches   (list)   — one entry per cached embedding in mm_cache
        """
        if self.encoder is None:
            return JSONResponse(
                status_code=503,
                content={"status": "error", "message": "encoder not ready"},
            )

        if self.encoder._shutdown_called:
            return JSONResponse(
                status_code=503,
                content={"status": "error", "message": "encoder is shutting down"},
            )

        model_id = os.getenv("MODEL_ID")
        if not model_id:
            logger.error(
                "MODEL_ID environment variable is not set. Please set it before starting the server."
            )
            raise ValueError("MODEL_ID environment variable is required")
        http_port = self.encoder.pull_zmq_port
        local_ip = self.encoder.local_ip
        transfer_type = normalize_transfer_type(
            self.encoder.server_args.encoder_transfer_backend
        )

        caches = []

        # Access mm_cache from encoder directly
        if self.encoder.mm_cache is None:
            return JSONResponse(content=[])
        for task_id, embedding_result in self.encoder.mm_cache.snapshot().items():
            if not self.encoder.validate_cached_video(task_id, embedding_result):
                continue
            embedding = embedding_result.embedding
            embedding_len = embedding.shape[0]
            embedding_dim = embedding.shape[1] if len(embedding.shape) > 1 else 1
            element_size = torch.zeros(1, dtype=embedding.dtype).element_size()
            embedding_size = embedding_len * embedding_dim * element_size
            embedding_dtype = str(embedding.dtype).replace("torch.", "")

            # Get image_grid_thw and image_url from EmbeddingResult
            img_grid_thw = embedding_result.image_grid_thw
            if img_grid_thw is not None and hasattr(img_grid_thw, "tolist"):
                img_grid_thw = img_grid_thw.tolist()

            cache_entry = {
                "ip": local_ip,
                "http_port": http_port,
                "transfer_type": transfer_type,
                "task_id": task_id,
                "model_id": model_id,
                "embedding_size": embedding_size,
                "embedding_len": embedding_len,
                "embedding_dim": embedding_dim,
                "embedding_dtype": embedding_dtype,
                "img_grid_thw": img_grid_thw,
                "original_image_sizes": embedding_result.original_image_sizes,
                "rdma_port": 0,
                "item_addr": 0,
                "cache_size": 0,
                "header_size": 0,
                "mooncake_session_id": "",
                "modality": embedding_result.modality,
                "chunk_timestamps": embedding_result.chunk_timestamps,
            }

            caches.append(cache_entry)

        return JSONResponse(content={"model_id": model_id, "caches": caches})
