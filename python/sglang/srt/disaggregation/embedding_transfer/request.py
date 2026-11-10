"""
Per-request state machine for pulling image embedding data in EPD mode.

``WaitingImageRequest`` drives one multimodal request through the transport:
send the pull targets, poll until every part lands, assemble the embedding.
The staged queues that call into it live in ``queues.py``.
"""

import dataclasses
import logging
import time
from http import HTTPStatus
from typing import Any, Dict, List, Optional

import msgspec
import torch

from sglang.srt.disaggregation.embedding_transfer.base import (
    EmbeddingTransport,
    PullTarget,
    TransportResult,
    WaitingImageRequestStatus,
    make_endpoint,
)
from sglang.srt.disaggregation.embedding_transfer.mooncake_transport import (
    MooncakeEmbeddingTransport,
    _MooncakeBufferManager,
)
from sglang.srt.disaggregation.embedding_transfer.zmq_transport import (
    ZmqEmbeddingTransport,
)
from sglang.srt.disaggregation.encode_receiver import _select_mm_processor_prompt
from sglang.srt.disaggregation.encode_utils import (
    original_image_sizes_from_task_results,
)
from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Transport factory
# ---------------------------------------------------------------------------

_TRANSPORT_REGISTRY: Dict[str, type] = {
    "zmq": ZmqEmbeddingTransport,
    "mooncake": MooncakeEmbeddingTransport,
}


# ---------------------------------------------------------------------------
# WaitingImageRequest — the state machine
# ---------------------------------------------------------------------------


class WaitingImageRequest:
    """State machine for pulling image embedding data for one scheduler request.

    Uses EmbeddingTransport implementations for actual data transfer.
    """

    def __init__(
        self,
        recv_req: TokenizedGenerateReqInput,
        mm_processor,
        wait_timeout: float,
        task_results: Optional[List[Dict[str, Any]]] = None,
        is_first_rank: bool = True,
    ):
        self.recv_req = recv_req
        self.mm_processor = mm_processor
        self.wait_timeout = wait_timeout
        self.task_results = task_results or []
        self.status = WaitingImageRequestStatus.PENDING
        self.error_msg = None
        self.error_code = None
        self.start_time = time.time()
        self.is_first_rank = is_first_rank

        # One PullTarget per task; the item lists are indexed by task_results
        # position, and task_id_to_item_indices maps between them (one task can
        # feed several items when the same media repeats in a prompt).
        self.pull_targets: List[PullTarget] = []
        self.task_id_to_item_indices: Dict[str, List[int]] = {}
        self.item_ready: List[bool] = []
        self.item_embeddings: List[Optional[torch.Tensor]] = []
        self.item_grid_thw: List[Optional[torch.Tensor]] = []
        self.task_send_time: Dict[str, float] = {}

        # task_id -> pinned-pool buf_addr owned by this request until handed
        # off to the Req on SUCCESS or freed on FAIL/TIMEOUT. Keyed (not a list)
        # so a re-delivered result overwrites rather than double-frees. Empty on
        # zmq path.
        self._pull_buf_addrs: Dict[str, int] = {}

        # Transport instances (keyed by transfer_type)
        self._transports: Dict[str, EmbeddingTransport] = {}

        self._init_pull_targets()

    @property
    def all_pulls_sent(self) -> bool:
        """True once every pull target is on the wire (always so off rank 0)."""
        return not self.is_first_rank or all(t.sent for t in self.pull_targets)

    # ------------------------------------------------------------------
    # Transport management
    # ------------------------------------------------------------------

    def _get_or_create_transport(self, transfer_type: str) -> EmbeddingTransport:
        """Lazy-create one transport instance per transfer_type."""
        transport = self._transports.get(transfer_type)
        if transport is None:
            cls = _TRANSPORT_REGISTRY.get(transfer_type)
            if cls is None:
                raise ValueError(f"Unsupported transfer_type: {transfer_type}")
            transport = cls()
            self._transports[transfer_type] = transport
        return transport

    def _cleanup_all_transports(self) -> None:
        for transport in self._transports.values():
            try:
                transport.cleanup()
            except Exception as e:
                logger.warning("Transport cleanup error: %s", e)
        self._transports.clear()

    # ------------------------------------------------------------------
    # Pull target initialization
    # ------------------------------------------------------------------

    @staticmethod
    def _build_embedding_meta(encoder_addr: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build embedding_meta from encoder_addr fields."""
        embedding_len = encoder_addr.get("embedding_len")
        embedding_dim = encoder_addr.get("embedding_dim")
        if embedding_len is None or embedding_dim is None:
            return None

        return {
            "shape": [embedding_len, embedding_dim],
            "dtype": encoder_addr.get("embedding_dtype", "bfloat16"),
            "img_grid_thw": encoder_addr.get("img_grid_thw"),
            "byte_size": encoder_addr.get("embedding_size"),
        }

    def _build_pull_target(self, task_result: Dict[str, Any]) -> Optional[PullTarget]:
        task_id = task_result.get("task_id")
        if not task_id:
            self.mark_failed("task_result missing task_id", HTTPStatus.BAD_REQUEST)
            return None

        request_id = task_result.get("request_id") or self.recv_req.rid
        encoder_addr = task_result.get("encoder_addr") or {}
        transfer_type = encoder_addr.get("transfer_type", "mooncake")

        if transfer_type == "zmq":
            endpoint = make_endpoint(encoder_addr, scheme="tcp")
        elif transfer_type == "mooncake":
            endpoint = make_endpoint(encoder_addr, scheme="http")
        else:
            self.mark_failed(
                f"Unsupported transfer_type={transfer_type} for task {task_id}",
                HTTPStatus.NOT_IMPLEMENTED,
            )
            return None

        return PullTarget(
            task_id=task_id,
            request_id=request_id,
            transfer_type=transfer_type,
            endpoint=endpoint,
            embedding_meta=self._build_embedding_meta(encoder_addr),
            image_url=task_result.get("image_url"),
            modality=encoder_addr.get("modality", "image"),
            chunk_timestamps=encoder_addr.get("chunk_timestamps"),
        )

    def _init_pull_targets(self) -> None:
        task_id_to_target: Dict[str, PullTarget] = {}

        for item_idx, task_result in enumerate(self.task_results):
            target = self._build_pull_target(task_result)
            if target is None:
                return

            task_id = target.task_id
            if task_id in self.task_id_to_item_indices:
                old = task_id_to_target[task_id]
                if old.endpoint != target.endpoint or old.transfer_type != target.transfer_type:
                    self.mark_failed(
                        f"Inconsistent pull target for {task_id}",
                        HTTPStatus.BAD_REQUEST,
                    )
                    return
                self.task_id_to_item_indices[task_id].append(item_idx)
                continue

            self.pull_targets.append(target)
            task_id_to_target[task_id] = target
            self.task_id_to_item_indices[task_id] = [item_idx]

        num_items = len(self.task_results)
        if num_items == 0:
            self.mark_failed("task_results is empty", HTTPStatus.BAD_REQUEST)
            return

        self.item_ready = [False] * num_items
        self.item_embeddings = [None] * num_items
        self.item_grid_thw = [None] * num_items

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def mark_failed(self, error_msg: str, error_code: int) -> None:
        """Mark FAIL and release transports and pool buffers."""
        self.error_msg = error_msg
        self.error_code = int(error_code)
        self.status = WaitingImageRequestStatus.FAIL
        self._cleanup_all_transports()
        self._release_owned_pool_buffers()

    def mark_timeout(self, error_msg: str) -> None:
        """Mark TIMEOUT and release transports and pool buffers."""
        self.error_msg = error_msg
        self.error_code = HTTPStatus.REQUEST_TIMEOUT
        self.status = WaitingImageRequestStatus.TIMEOUT
        self._cleanup_all_transports()
        self._release_owned_pool_buffers()

    def _release_owned_pool_buffers(self) -> None:
        """Free pool buffers of already-pulled parts on FAIL/TIMEOUT (in-flight
        ones are freed by transport.cleanup()). No-op on the zmq path."""
        if not self._pull_buf_addrs:
            return
        addrs = list(self._pull_buf_addrs.values())
        self._pull_buf_addrs = {}
        buf_mgr = _MooncakeBufferManager.get_instance_if_exists()
        if buf_mgr is None:
            return
        for addr in addrs:
            try:
                buf_mgr.release_buffer(addr)
            except Exception as e:
                logger.warning("Failed to release owned pool buffer %s: %s", addr, e)

    # ------------------------------------------------------------------
    # Pull workflow: send → poll → apply
    # ------------------------------------------------------------------

    def _send_pull_requests(self) -> None:
        """Send (or retry) pull requests for all unsent targets."""
        if self.status != WaitingImageRequestStatus.PENDING:
            return

        for pull_target in self.pull_targets:
            if pull_target.sent:
                continue
            try:
                transport = self._get_or_create_transport(pull_target.transfer_type)
                sent = transport.send(pull_target)
                if sent:
                    pull_target.sent = True
                    self.task_send_time[pull_target.task_id] = time.time()
            except Exception as e:
                self.mark_failed(
                    f"Failed to send pull request for {pull_target.task_id}: {e}",
                    HTTPStatus.BAD_GATEWAY,
                )
                return

    def _poll_all_responses(self) -> None:
        """Poll all active transports and apply results."""
        for transport in self._transports.values():
            if self.status != WaitingImageRequestStatus.PENDING:
                return
            for result in transport.poll():
                self._apply_transport_result(result)
                if self.status != WaitingImageRequestStatus.PENDING:
                    return

    def _apply_transport_result(self, result: TransportResult) -> None:
        """Unified result handling — works identically for zmq and mooncake."""
        if not result.success:
            self.mark_failed(result.error_msg, result.error_code)
            return

        item_indices = self.task_id_to_item_indices.get(result.task_id)
        if item_indices is None:
            logger.warning("Received result for unknown task_id: %s", result.task_id)
            # Nobody will own this buffer; free it now to avoid a pool leak.
            if result.buf_addr is not None:
                buf_mgr = _MooncakeBufferManager.get_instance_if_exists()
                if buf_mgr is not None:
                    try:
                        buf_mgr.release_buffer(result.buf_addr)
                    except Exception as e:
                        logger.warning("Failed to free orphan buffer: %s", e)
            return

        # Take ownership of the pinned-pool buffer (None on the zmq path).
        if result.buf_addr is not None:
            self._pull_buf_addrs[result.task_id] = result.buf_addr

        for idx in item_indices:
            self.item_ready[idx] = True
            self.item_embeddings[idx] = result.embedding
            self.item_grid_thw[idx] = result.grid_thw

    def _check_pull_task_timeouts(self, now: float) -> None:
        if self.status != WaitingImageRequestStatus.PENDING:
            return

        for pull_target in self.pull_targets:
            task_id = pull_target.task_id
            item_indices = self.task_id_to_item_indices[task_id]
            if all(self.item_ready[idx] for idx in item_indices):
                continue

            sent_at = self.task_send_time.get(task_id, self.start_time)
            if now - sent_at > self.wait_timeout:
                self.mark_timeout(
                    f"Timeout waiting pull response for task_id={task_id} "
                    f"after {self.wait_timeout}s"
                )
                return

    # ------------------------------------------------------------------
    # Public API (also called from queues.py / encode_receiver_aiak.py)
    # ------------------------------------------------------------------

    def check_timeout(self, now: float) -> None:
        """Check pull task timeouts and mark as timeout if exceeded.

        Args:
            now: Current timestamp in seconds.
        """
        self._check_pull_task_timeouts(now)

    def advance(self) -> None:
        """Drive one cycle of the pull state machine."""
        if self.status != WaitingImageRequestStatus.PENDING:
            return

        # Non-first ranks never pull from the encoder — nothing to send.
        if not self.is_first_rank:
            return

        self._send_pull_requests()
        if self.status != WaitingImageRequestStatus.PENDING:
            return

        self._poll_all_responses()
        self._check_pull_task_timeouts(time.time())

    def assemble_if_ready(self) -> None:
        """Assemble mm_inputs once every item's embedding has landed."""
        if self.status != WaitingImageRequestStatus.PENDING:
            return
        if not all(self.item_ready):
            return

        try:
            grid_thw = torch.concatenate(self.item_grid_thw)
            item_hashes = self._item_hashes(self.item_grid_thw)
        except Exception as e:
            logger.exception(
                "Failed to assemble embedding/grid for %s: %s", self.recv_req.rid, e
            )
            self.mark_failed(str(e), HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self._assemble_mm_inputs(self.item_embeddings, grid_thw, item_hashes)

    def _item_hashes(self, grids: List[torch.Tensor]) -> List[int]:
        """Seed one pad_value hash per mm_item, i.e. per grid row, from task_id."""
        item_hashes: List[int] = []
        for task_result, grid in zip(self.task_results, grids):
            task_hash = int(task_result["task_id"], 16)
            num_items = grid.shape[0]
            item_hashes.extend([task_hash] * num_items)
        return item_hashes

    def _grids_from_task_results(self) -> List[torch.Tensor]:
        """Rebuild the per-task grids locally, mooncake path only."""
        grids: List[torch.Tensor] = []
        for task_result in self.task_results:
            encoder_addr = task_result.get("encoder_addr") or {}
            grid = encoder_addr.get("img_grid_thw")
            if grid is None:
                raise ValueError(
                    f"task {task_result.get('task_id')} has no "
                    "encoder_addr[img_grid_thw]; non-first ranks can only "
                    "assemble metadata on the mooncake path"
                )
            grid = torch.as_tensor(grid, dtype=torch.long)
            grids.append(grid.reshape(-1, grid.shape[-1]))
        return grids

    def assemble_metadata_only(self) -> None:
        """Assemble metadata-only mm_inputs; rows arrive via the attn-tp broadcast."""
        if self.status != WaitingImageRequestStatus.PENDING:
            return
        try:
            grids = self._grids_from_task_results()
            grid_thw = torch.concatenate(grids)
            item_hashes = self._item_hashes(grids)
        except Exception as e:
            logger.exception(
                "Failed to derive grid_thw for %s: %s", self.recv_req.rid, e
            )
            self.mark_failed(str(e), HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._assemble_mm_inputs(None, grid_thw, item_hashes)

    def _assemble_mm_inputs(
        self, recv_embedding, grid_thw, item_hashes: List[int]
    ) -> None:
        # recv_embedding is one tensor per pulled task; get_mm_data slices
        # per-item zero-copy views across them (no concat).  None requests
        # metadata-only assembly (non-first rank): offsets / hash / pad_value
        # only, the rows arrive later via the attn-tp broadcast.
        embeddings = (
            None if recv_embedding is None else {Modality.IMAGE: recv_embedding}
        )
        # Deliberately outside the try below: a processor that cannot do
        # metadata-only assembly is a deployment error, not a per-request
        # failure, so it must crash the worker instead of aborting one request
        # with a misleading "get_mm_data failed" log.
        if (
            embeddings is None
            and type(self.mm_processor).get_mm_data
            is not BaseMultimodalProcessor.get_mm_data
            and not self.mm_processor.supports_metadata_only_mm_data
        ):
            raise TypeError(
                f"{type(self.mm_processor).__name__} overrides get_mm_data and "
                "does not support the metadata-only assembly that the EPD "
                "in-forward broadcast needs on non-first ranks"
            )

        try:
            prompt = _select_mm_processor_prompt(self.recv_req, self.mm_processor)
            mm_inputs = self.mm_processor.get_mm_data(
                prompt,
                embeddings,
                img_grid_thw=grid_thw,
                original_image_sizes=original_image_sizes_from_task_results(
                    self.task_results
                ),
            )
        except Exception as e:
            logger.exception("get_mm_data failed for %s: %s", self.recv_req.rid, e)
            self.mark_failed(str(e), HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        # Pre-seed hashes so set_pad_value() skips hashing the embedding.
        for mm_item, item_hash in zip(mm_inputs.mm_items or [], item_hashes):
            mm_item.hash = item_hash

        self.recv_req.mm_inputs = mm_inputs
        self.recv_req.input_ids = mm_inputs.input_ids
        # Hand pool buffer ownership to the Req (freed after prefill consumes
        # the embedding); clear ours so FAIL/TIMEOUT can't double-free.
        if self._pull_buf_addrs:
            mm_inputs.pool_buffer_addrs = list(self._pull_buf_addrs.values())
            self._pull_buf_addrs = {}
        self.status = WaitingImageRequestStatus.SUCCESS
        self._cleanup_all_transports()


def metadata_only_recv_req(
    recv_req: TokenizedGenerateReqInput,
) -> TokenizedGenerateReqInput:
    """Copy an admitted request with the embedding payload removed, keeping the
    mm_items offsets and hashes downstream PP stages need."""
    mm_inputs = recv_req.mm_inputs
    if mm_inputs is None:
        raise ValueError(f"{recv_req.rid} has no mm_inputs to forward")

    items = []
    for item in mm_inputs.mm_items:
        if item.hash is None:
            raise ValueError(
                f"{recv_req.rid} mm_item has no hash; downstream PP stages "
                "cannot derive the same pad_value"
            )
        if any(
            isinstance(value, torch.Tensor)
            for value in item.model_specific_data.values()
        ):
            raise ValueError(
                f"{recv_req.rid} mm_item keeps a tensor in model_specific_data; "
                "it would be relayed to every PP stage"
            )
        items.append(
            dataclasses.replace(item, feature=None, precomputed_embeddings=None)
        )

    return msgspec.structs.replace(
        recv_req,
        mm_inputs=dataclasses.replace(
            mm_inputs, mm_items=items, pool_buffer_addrs=None
        ),
        need_wait_for_mm_inputs=False,
        task_results=None,
        mm_data_mooncake=None,
        num_items_assigned=None,
        encoder_urls=None,
    )
