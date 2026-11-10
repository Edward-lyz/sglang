from __future__ import annotations

import concurrent.futures
import dataclasses
import logging
import os
import struct
import threading
import time
from collections import defaultdict
from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import zmq
from prometheus_client import Counter

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll, StateType
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
    KVTransferError,
)
from sglang.srt.disaggregation.common.staging_handler import (
    STAGING_WATERMARK_WAIT_S,
    DecodeStagingContext,
    PrefillStagingContext,
    StagingRegisterInfo,
    StagingTransferInfo,
)
from sglang.srt.disaggregation.common.utils import (
    AuxDataCodec,
    FastQueue,
    TransferKVChunk,
    build_dcp_replicated_token_transfer_plan,
    build_dcp_token_transfer_plan,
    build_paged_transfer_descriptors,
    DCPTokenTransferPlan,
    contiguous_run_starts_and_counts,
    pack_int_lists,
    unpack_int_lists,
)
from sglang.srt.disaggregation.mooncake.target_parallel_transfer import (
    PerTargetKVChunk,
    maybe_enqueue_target_parallel_chunks,
    process_target_parallel_chunk,
)
from sglang.srt.disaggregation.mooncake.utils import (
    check_mooncake_custom_mem_pool_enabled,
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    build_transfer_entry_pairs,
    compute_mamba_state_slice_byte_blocks,
    resolve_dcp_dst_entry_indices,
)
from sglang.srt.distributed.parallel_state import get_mooncake_transfer_engine
from sglang.srt.environ import envs
from sglang.srt.observability.mooncake_trace import (
    MooncakeRequestStage,
    mooncake_trace_func,
    mooncake_trace_slice,
)
from sglang.srt.observability.trace import (
    TraceNullContext,
    TraceReqContext,
    trace_set_thread_info,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.common import is_cuda
from sglang.srt.utils.network import NetworkAddress

logger = logging.getLogger(__name__)

FAILED_SESSION_RECOVERIES = Counter(
    "sglang:failed_session_recoveries_total",
    "Number of mooncake_session_ids un-blacklisted via probe.",
)


# decode
@dataclasses.dataclass
class TransferInfo:
    room: int
    endpoint: str
    dst_port: int
    mooncake_session_id: str
    dst_kv_indices: npt.NDArray[np.int32]
    dst_aux_index: int
    dst_state_indices: List[List[int]]  # parallel to receiver's state_types
    required_dst_info_num: int
    is_dummy: bool
    decode_prefix_len: Optional[int] = None
    # Note: always put the optional staging field at the final (it will be set through 'STAGING_RSP' pkg when needed)
    staging: Optional[StagingTransferInfo] = None
    draft_dst_kv_indices: Optional[npt.NDArray[np.int32]] = None
    draft_window_start: Optional[int] = None

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        if msg[4] == b"" and msg[5] == b"":
            is_dummy = True
            dst_kv_indices = np.array([], dtype=np.int32)
            dst_aux_index = None
            dst_state_indices = []
        else:
            dst_kv_indices = np.frombuffer(msg[4], dtype=np.int32)
            dst_aux_index = int(msg[5].decode("ascii"))
            dst_state_indices = unpack_int_lists(msg[6], "i")
            is_dummy = False
        return cls(
            room=int(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            dst_kv_indices=dst_kv_indices,
            dst_aux_index=dst_aux_index,
            dst_state_indices=dst_state_indices,
            required_dst_info_num=int(msg[7].decode("ascii")),
            is_dummy=is_dummy,
            decode_prefix_len=(
                int(msg[8].decode("ascii")) if len(msg) > 8 and msg[8] != b"" else None
            ),
            draft_dst_kv_indices=(
                np.frombuffer(msg[9], dtype=np.int32)
                if len(msg) > 9 and msg[9] != b""
                else None
            ),
            draft_window_start=(
                int(msg[10].decode("ascii"))
                if len(msg) > 10 and msg[10] != b""
                else None
            ),
        )


# decode
@dataclasses.dataclass
class KVArgsRegisterInfo:
    room: str
    endpoint: str
    dst_port: int
    mooncake_session_id: str
    dst_kv_ptrs: list[int]
    dst_aux_ptrs: list[int]
    dst_state_data_ptrs: List[List[int]]  # parallel to state_types (same below)
    dst_tp_rank: int
    dst_attn_tp_size: int
    dst_kv_item_len: int
    # for mamba state different tp slice transfer
    dst_state_item_lens: List[List[int]]
    dst_state_dim_per_tensor: List[List[int]]
    dst_kv_layer_ids: List[int]
    dst_state_layer_ids: List[List[int]]
    dst_dcp_size: int = 1
    dst_dcp_rank: int = 0
    requires_dcp_relayout: bool = False
    dcp_token_item_lens: Optional[List[int]] = None
    # Full per-entry KV item_lens list from the decode side. Needed for DCP
    # relayout when the KV geometry is heterogeneous across entries (e.g. the
    # speculative draft (MHA) pool ships a head-sharded per-token byte size that
    # differs from the main MLA KV), where the single dst_kv_item_len scalar
    # would lose the tail entries.
    dst_kv_item_lens: List[int] = dataclasses.field(default_factory=list)
    # Count of leading MLA (main) entries on the decode side, i.e. the boundary
    # before its speculative draft pool entries. Under DCP relayout the MLA
    # prefix takes token relayout and the draft tail takes a replicated
    # head-slice transfer. 0 when unknown/no draft pool.
    dst_num_mla_kv_entries: int = 0
    # Note: always put the staging field at the final (since the staging field is optional and contains multiple inputs)
    staging: Optional[StagingRegisterInfo] = None

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        return cls(
            room=str(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            dst_kv_ptrs=list(struct.unpack(f"{len(msg[4])//8}Q", msg[4])),
            dst_aux_ptrs=list(struct.unpack(f"{len(msg[5])//8}Q", msg[5])),
            dst_state_data_ptrs=unpack_int_lists(msg[6], "Q"),
            dst_tp_rank=int(msg[7].decode("ascii")),
            dst_attn_tp_size=int(msg[8].decode("ascii")),
            dst_kv_item_len=int(msg[9].decode("ascii")),
            dst_state_item_lens=(
                unpack_int_lists(msg[10], "I") if len(msg) > 10 else []
            ),
            dst_state_dim_per_tensor=(
                unpack_int_lists(msg[11], "I") if len(msg) > 11 else []
            ),
            dst_kv_layer_ids=(
                list(struct.unpack(f"{len(msg[12]) // 4}I", msg[12]))
                if len(msg) > 12 and msg[12] != b""
                else []
            ),
            dst_state_layer_ids=(
                unpack_int_lists(msg[13], "I")
                if len(msg) > 13 and msg[13] != b""
                else []
            ),
            # msg[14:16] belong to the staging field below; DCP trails it.
            dst_dcp_size=(
                int(msg[16].decode("ascii")) if len(msg) > 16 and msg[16] != b"" else 1
            ),
            dst_dcp_rank=(
                int(msg[17].decode("ascii")) if len(msg) > 17 and msg[17] != b"" else 0
            ),
            # msg[18] carries the full per-entry KV item_lens list (packed as a
            # single int list) so DCP relayout keeps heterogeneous geometry.
            dst_kv_item_lens=(
                unpack_int_lists(msg[18], "I")[0]
                if len(msg) > 18 and msg[18] != b""
                else []
            ),
            # msg[19] carries the count of leading MLA (main) entries; the draft
            # tail beyond it takes the replicated head-slice DCP transfer.
            dst_num_mla_kv_entries=(
                int(msg[19].decode("ascii"))
                if len(msg) > 19 and msg[19] != b""
                else 0
            ),
            # Note: always put the staging field at the final
            staging=StagingRegisterInfo.from_zmq_fields(msg, 14),
        )


class MooncakeKVManager(CommonKVManager):
    AUX_DATA_HEADER = b"AUX_DATA"

    transfer_infos: dict[int, dict[str, TransferInfo]]

    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        self.kv_buffer_tensors = None
        self.init_engine()
        self.register_buffer_to_engine()
        self.enable_staging = envs.SGLANG_DISAGG_STAGING_BUFFER.get()
        self.enable_trace = server_args.enable_trace
        # PP+PD early-last-chunk keeps the aux buffer on PP0 only, so other
        # ranks legitimately send a last chunk with aux_index=None.
        self.allow_last_chunk_without_aux = (
            envs.SGLANG_PP_PD_PREFILL_EARLY_LAST_CHUNK.get()
        )
        # Speculative draft KV gather staging. These stay unset until
        # set_draft_kv_entry_tensors() is called with the draft pool this rank
        # registered for transfer (last PP stage only, prefill side only).
        self._draft_kv_entry_tensors: Optional[list] = None
        self._draft_kv_entry_ptrs_gpu = None
        self._draft_gather_max_ctas = 0
        self._draft_gather_pool = None
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.start_prefill_thread()
            self.session_failures = defaultdict(int)
            self.failed_sessions = set()
            self.session_cooldowns: dict[str, float] = {}  # 【新增】专门用于记录失败时间戳的字典
            self.session_cooldown_seconds = float(os.getenv("SGLANG_MOONCAKE_COOLDOWN", "120.0"))
            self.session_lock = threading.Lock()
            # Two independent knobs, both off by default; see
            # mooncake/target_parallel_transfer.py.
            #   - room sharding picks a chunk's queue by room, so different rooms of
            #     this rank run on different worker threads;
            #   - target parallel additionally splits one room's multi-destination
            #     fan-out into one work unit per destination.
            # The fan-out shards by room too, so it implies room sharding.
            self.enable_target_parallel_transfer = (
                envs.SGLANG_PP_PD_KV_TRANSFER_TARGET_PARALLEL.get()
            )
            self.enable_room_queue_sharding = (
                envs.SGLANG_PP_PD_KV_TRANSFER_ROOM_SHARDING.get()
                or self.enable_target_parallel_transfer
            )
            # Determine the number of threads to use for kv sender
            cpu_count = os.cpu_count()
            transfer_thread_pool_size = (
                envs.SGLANG_DISAGGREGATION_THREAD_POOL_SIZE.get()
            )
            if transfer_thread_pool_size is None:
                transfer_thread_pool_size = min(max(4, int(0.5 * cpu_count) // 8), 12)
            transfer_queue_size = envs.SGLANG_DISAGGREGATION_QUEUE_SIZE.get()
            if self.enable_room_queue_sharding and transfer_queue_size <= 1:
                # Both knobs only buy parallelism by spreading over queues, and each
                # queue owns exactly one worker thread.
                logger.warning(
                    f"SGLANG_DISAGGREGATION_QUEUE_SIZE={transfer_queue_size} leaves "
                    "nothing to spread over; room sharding / per-target fan-out are "
                    "requested but every transfer still runs on one worker thread."
                )
            self.transfer_queues: List[FastQueue] = [
                FastQueue() for _ in range(transfer_queue_size)
            ]
            if transfer_thread_pool_size < transfer_queue_size:
                # Only a soft constraint: the executors are per-layer helpers, while
                # the queues each own a worker thread. Under-provisioning the pool
                # just means one helper thread per queue, so warn instead of
                # refusing to boot -- the per-target fan-out wants many queues.
                logger.warning(
                    f"SGLANG_DISAGGREGATION_THREAD_POOL_SIZE={transfer_thread_pool_size} is "
                    f"smaller than SGLANG_DISAGGREGATION_QUEUE_SIZE={transfer_queue_size}; "
                    "each transfer queue falls back to a single-thread executor."
                )
            self.executors = [
                concurrent.futures.ThreadPoolExecutor(
                    max(1, transfer_thread_pool_size // transfer_queue_size)
                )
                for _ in range(transfer_queue_size)
            ]
            self.enable_custom_mem_pool, self.custom_mem_pool_type = (
                check_mooncake_custom_mem_pool_enabled()
            )
            self._staging_ctx = PrefillStagingContext() if self.enable_staging else None
            if self.enable_staging:
                self._init_staging_buffers(len(self.transfer_queues))
            for i, (queue, executor) in enumerate(
                zip(self.transfer_queues, self.executors)
            ):
                threading.Thread(
                    target=self.transfer_worker,
                    args=(
                        queue,
                        executor,
                        (
                            self._staging_ctx.buffers[i]
                            if self.enable_staging and self._staging_ctx.buffers
                            else None
                        ),
                        i,
                    ),
                    daemon=True,
                ).start()
            self.enable_failed_session_probe = (
                envs.SGLANG_ENABLE_FAILED_SESSION_PROBE.get()
            )
            if self.enable_failed_session_probe:
                self.failed_session_probe_interval = (
                    envs.SGLANG_FAILED_SESSION_PROBE_INTERVAL_S.get()
                )
                self._failed_session_probe_shutdown = threading.Event()
                threading.Thread(
                    target=self._failed_session_probe_loop,
                    name="MooncakeFailedSessionProbe",
                    daemon=True,
                ).start()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self._staging_ctx = DecodeStagingContext() if self.enable_staging else None
            if self.enable_staging:
                self._init_staging_allocator()
                self._staging_handler = None
                self._chunk_writer_counts: dict = defaultdict(lambda: defaultdict(list))
            self.start_decode_thread()

    def init_engine(self):
        self.engine = get_mooncake_transfer_engine()

    def register_buffer_to_engine(self):
        # Batch register KV data buffers
        if self.kv_args.kv_data_ptrs and self.kv_args.kv_data_lens:
            self.engine.batch_register(
                self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens
            )

        # Batch register auxiliary data buffers
        if self.kv_args.aux_data_ptrs and self.kv_args.aux_data_lens:
            self.engine.batch_register(
                self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens
            )

        for ptrs, lens in zip(
            self.kv_args.state_data_ptrs, self.kv_args.state_data_lens
        ):
            if ptrs and lens:
                self.engine.batch_register(ptrs, lens)

    def deregister_buffer_to_engine(self):
        if self.kv_args.kv_data_ptrs:
            self.engine.batch_deregister(self.kv_args.kv_data_ptrs)

        if self.kv_args.aux_data_ptrs:
            self.engine.batch_deregister(self.kv_args.aux_data_ptrs)

        for ptrs in self.kv_args.state_data_ptrs or []:
            if ptrs:
                self.engine.batch_deregister(ptrs)

        if hasattr(self, "connection_pool"):
            with self.connection_lock:
                self.connection_pool.clear()

    # ------------------------------------------------------------------
    # Staging buffer methods (all delegate to staging_handler.py)
    # ------------------------------------------------------------------

    def register_staging_room_bootstrap(self, room, bootstrap_infos, receiver):
        self._staging_ctx.room_bootstrap[room] = bootstrap_infos
        self._staging_ctx.room_receivers[room] = receiver

    def set_kv_buffer_tensors(self, k_buffers: list, v_buffers: list, page_size: int):
        self.kv_buffer_tensors = {
            "k_buffers": k_buffers,
            "v_buffers": v_buffers,
            "page_size": page_size,
        }

    def set_draft_kv_entry_tensors(self, entry_tensors: list) -> None:
        """Hand over the draft KV pool tensors this rank registered for transfer.

        Enables the gather-staging path in ``send_kvcache_dcp_draft``: the head
        slice one decode rank wants is strided in the token-major draft pool, so
        without a gather every token needs its own RDMA descriptor. ``entry_tensors``
        must be in registration order (``kv_data_ptrs[num_mla_kv_entries:]``);
        the geometry is re-validated against the registered pointers per transfer.
        """
        # CUDA-only: the gather is a Triton kernel writing a torch cuda buffer, so
        # the Ascend/NPU subclass keeps addressing the pool directly.
        if (
            self.disaggregation_mode != DisaggregationMode.PREFILL
            or not entry_tensors
            or not envs.SGLANG_ENABLE_DISAGG_DRAFT_GATHER.get()
            or not is_cuda()
        ):
            return

        from sglang.srt.disaggregation.common.draft_kv_gather import (
            DraftKVGatherPool,
            build_entry_ptr_tensor,
        )
        from sglang.srt.disaggregation.common.staging_handler import (
            _get_custom_mem_pool,
        )

        custom_mem_pool, _ = _get_custom_mem_pool(f"cuda:{self.kv_args.gpu_id}")
        self._draft_kv_entry_tensors = entry_tensors
        self._draft_kv_entry_ptrs_gpu = build_entry_ptr_tensor(
            entry_tensors=entry_tensors, gpu_id=self.kv_args.gpu_id
        )
        self._draft_gather_max_ctas = envs.SGLANG_DISAGG_DRAFT_GATHER_MAX_CTAS.get()
        self._draft_gather_pool = DraftKVGatherPool(
            gpu_id=self.kv_args.gpu_id,
            num_slots=max(1, envs.SGLANG_DISAGG_DRAFT_GATHER_BUFFERS.get()),
            register=lambda ptr, size: self.engine.batch_register([ptr], [size]),
            deregister=lambda ptr: self.engine.batch_deregister([ptr]),
            custom_mem_pool=custom_mem_pool,
        )
        logger.info(
            f"PD draft KV gather staging enabled for {len(entry_tensors)} draft "
            f"entries on gpu {self.kv_args.gpu_id}"
        )

    def _init_staging_buffers(self, count: int):
        from sglang.srt.disaggregation.common.staging_handler import (
            init_staging_buffers,
        )

        self._staging_ctx.buffers = init_staging_buffers(
            lambda ptr, size: self.engine.batch_register([ptr], [size]),
            self.kv_args,
            count,
            self.server_args.chunked_prefill_size,
        )
        self.kv_buffer_tensors = None

    def _init_staging_allocator(self):
        from sglang.srt.disaggregation.common.staging_handler import (
            init_staging_allocator,
        )

        self._staging_ctx.allocator = init_staging_allocator(
            lambda ptr, size: self.engine.batch_register([ptr], [size]),
            self.kv_args,
        )
        self.kv_buffer_tensors = None

    def _handle_staging_req(self, msg):
        from sglang.srt.disaggregation.common.staging_handler import (
            handle_staging_req,
        )

        room = int(msg[1].decode("ascii"))
        session_id = msg[4].decode("ascii")
        handler = self._staging_handler
        assert (
            handler is not None
        ), "STAGING_REQ received before staging handler initialized"
        decode_req = handler._room_to_decode_req.get(room)
        if decode_req is None:
            logger.warning(
                "STAGING_REQ received for unregistered room=%s, skipping",
                room,
            )
            return
        prefill_tp = decode_req.kv_receiver.prefill_info.attn_tp_size
        handle_staging_req(
            msg,
            self._staging_ctx.allocator,
            self.kv_args,
            self.attn_tp_size,
            prefill_tp,
            getattr(self, "kv_buffer_tensors", None),
            self._staging_ctx.room_receivers,
            self._staging_ctx.room_bootstrap,
        )

        receiver = self._staging_ctx.room_receivers.get(room)
        if receiver is not None:
            handler.register_wm_subscriber(receiver, session_id)

    def _is_watermark_ready(
        self, session_id: str, alloc_round: int, alloc_end: int
    ) -> bool:
        from sglang.srt.disaggregation.common.staging_handler import (
            is_watermark_ready,
        )

        return is_watermark_ready(self._staging_ctx, session_id, alloc_round, alloc_end)

    def _try_create_staging_strategy(self, staging_buffer):
        if not self.enable_staging or self.kv_buffer_tensors is None:
            return None
        from sglang.srt.disaggregation.common.staging_handler import (
            PrefillStagingStrategy,
        )

        return PrefillStagingStrategy(self, staging_buffer)

    def _send_chunk_ready(self, req, chunk_idx, kv_chunk, prefill_unique_rank):
        """Notify decode that a non-last staging chunk RDMA is complete."""
        na = NetworkAddress(req.endpoint, req.dst_port)
        self._send_multipart_locked(
            na.to_tcp(),
            [
                b"CHUNK_READY",
                str(req.room).encode("ascii"),
                str(chunk_idx).encode("ascii"),
                str(kv_chunk.index_slice.start).encode("ascii"),
                str(len(kv_chunk.prefill_kv_indices)).encode("ascii"),
                req.mooncake_session_id.encode("ascii"),
                str(prefill_unique_rank).encode("ascii"),
            ],
            is_ipv6=na.is_ipv6,
        )

    def _do_staging_transfer(
        self,
        staging_strategy,
        kv_chunk,
        req,
        target_info,
        chunked_dst_kv_indice,
        executor,
        queue,
        prefill_unique_rank,
    ):
        """Execute staging transfer for one chunk. Returns (ret, deferred).

        Handles readiness check, transfer, and CHUNK_READY notification; a chunk
        that cannot fit returns -1 (the caller fails only this room) instead of
        falling back to the slice path, which would leak the decode-side
        allocation. deferred=True means caller should re-enqueue and break.
        """
        ready, chunk_idx, c_offset, _, _ = staging_strategy.check_ready(
            req,
            kv_chunk.index_slice.start,
            len(kv_chunk.prefill_kv_indices),
        )
        if not ready:
            from sglang.srt.disaggregation.common.staging_buffer import StagingAllocator

            if c_offset == StagingAllocator.ALLOC_OVERSIZED:
                # Fail this room, not the worker thread: the same prefill still
                # serves other (same-TP, non-staging) decode instances.
                logger.warning_once(
                    "[Staging] a chunk exceeds the staging ring; failing affected "
                    "requests. Increase SGLANG_DISAGG_STAGING_POOL_SIZE_MB or "
                    "reduce chunked_prefill_size."
                )
                return (-1, False)
            # Not ready yet: wait (bounded) for a watermark advance, then
            # re-enqueue to retry. A plain block-until-ready would head-of-line
            # block other rooms on this single worker thread.
            with self._staging_ctx.watermark_cv:
                self._staging_ctx.watermark_cv.wait(STAGING_WATERMARK_WAIT_S)
            queue.put(kv_chunk)
            return (-1, True)

        ret = staging_strategy.transfer(
            req.mooncake_session_id,
            kv_chunk.prefill_kv_indices,
            target_info.staging.base_ptr + c_offset,
            target_info.staging.total_size - c_offset,
            target_info,
        )
        if ret == -1:
            # Doesn't fit the ring: fail this room (caller's ret != 0 path), do
            # not fall back to the slice path (leaks the decode-side allocation).
            logger.warning_once(
                "[Staging] a chunk does not fit the staging ring; failing affected "
                "requests. Increase SGLANG_DISAGG_STAGING_POOL_SIZE_MB or "
                "reduce chunked_prefill_size."
            )
            return (-1, False)
        if ret == 0 and not kv_chunk.is_last_chunk:
            self._send_chunk_ready(req, chunk_idx, kv_chunk, prefill_unique_rank)
        return (ret, False)

    def _prefetch_staging_reqs(self, room: int):
        if not self.enable_staging or self.kv_buffer_tensors is None:
            return

        room_infos = self.transfer_infos.get(room, {})
        needs_staging = any(
            not tinfo.is_dummy
            and self.decode_kv_args_table.get(tinfo.mooncake_session_id) is not None
            and self.decode_kv_args_table[tinfo.mooncake_session_id].dst_attn_tp_size
            != self.attn_tp_size
            for tinfo in room_infos.values()
        )
        if not needs_staging:
            return

        from sglang.srt.disaggregation.common.staging_handler import (
            prefetch_staging_reqs,
        )

        prefetch_staging_reqs(
            room,
            self.transfer_infos,
            self.kv_buffer_tensors,
            self.server_args.chunked_prefill_size,
            self._staging_ctx.prefetch_requested,
            self._staging_ctx.prefetch_sockets,
        )

    def send_kvcache_staged(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_staging_ptr: int,
        dst_staging_size: int,
        dst_tp_rank: int,
        dst_attn_tp_size: int,
        dst_kv_item_len: int,
        staging_buffer=None,
    ) -> int:
        """Transfer the TARGET KV cache via staging buffers.

        Target only: the speculative draft pool is never staged on this backend;
        draft KV goes through ``send_kvcache_draft``.
        """
        from sglang.srt.disaggregation.common.staging_buffer import (
            compute_head_slice_params,
            compute_staging_layout,
            resolve_total_kv_heads,
        )

        if self.kv_buffer_tensors is None or staging_buffer is None:
            return -1

        k_buffers = self.kv_buffer_tensors["k_buffers"]
        v_buffers = self.kv_buffer_tensors["v_buffers"]
        page_size = self.kv_buffer_tensors["page_size"]
        num_layers = len(k_buffers)
        head_dim = k_buffers[0].shape[-1]
        dtype_size = k_buffers[0].element_size()

        total_kv_heads = resolve_total_kv_heads(self.kv_args, self.attn_tp_size)

        local_tp_rank = self.kv_args.engine_rank % self.attn_tp_size
        src_head_start, num_heads_to_send, _, _ = compute_head_slice_params(
            self.attn_tp_size,
            dst_attn_tp_size,
            local_tp_rank,
            dst_tp_rank,
            total_kv_heads,
        )

        num_tokens = len(prefill_kv_indices) * page_size
        per_layer_bytes = num_tokens * num_heads_to_send * head_dim * dtype_size
        per_rank_bytes = per_layer_bytes * num_layers * 2

        num_writers, writer_rank_bytes, total_staging_needed = compute_staging_layout(
            self.attn_tp_size,
            dst_attn_tp_size,
            dst_tp_rank,
            total_kv_heads,
            num_tokens,
            head_dim * dtype_size,
            num_layers,
        )
        writer_idx = local_tp_rank % num_writers if num_writers > 1 else 0
        rank_offset = sum(writer_rank_bytes[:writer_idx])

        if not staging_buffer.fits(per_rank_bytes):
            logger.warning(
                f"Prefill staging too small for {per_rank_bytes} bytes, falling back"
            )
            return -1
        if dst_staging_size < total_staging_needed:
            logger.warning(
                f"Decode staging too small: need {total_staging_needed} bytes "
                f"({num_writers if self.attn_tp_size > dst_attn_tp_size else 1} writers "
                f"x {per_rank_bytes} bytes/rank), have {dst_staging_size}, falling back"
            )
            return -1

        from sglang.srt.disaggregation.common.staging_buffer import (
            gather_all_layers_to_staging,
        )

        gather_all_layers_to_staging(
            k_buffers,
            v_buffers,
            prefill_kv_indices,
            staging_buffer,
            src_head_start,
            num_heads_to_send,
            page_size,
            self.kv_args.gpu_id,
        )

        dst_write_ptr = dst_staging_ptr + rank_offset
        ret = self._transfer_data(
            mooncake_session_id,
            [(staging_buffer.get_ptr(), dst_write_ptr, per_rank_bytes)],
        )
        if ret != 0:
            raise RuntimeError(
                f"[Staging] Bulk RDMA transfer failed with ret={ret}. "
                f"src_ptr=0x{staging_buffer.get_ptr():x}, "
                f"dst_ptr=0x{dst_write_ptr:x}, size={per_rank_bytes}. "
                f"The decode staging buffer may not be properly registered."
            )
        return ret

    def _transfer_data(self, mooncake_session_id, transfer_blocks):
        if not transfer_blocks:
            return 0

        src_addrs, dst_addrs, lengths = zip(*transfer_blocks)
        lengths = list(lengths)
        start_ts = time.perf_counter()
        ret = self.engine.batch_transfer_sync(
            mooncake_session_id, list(src_addrs), list(dst_addrs), lengths
        )
        if ret != 0:
            self._log_batch_transfer_failure(
                mooncake_session_id=mooncake_session_id,
                num_blocks=len(lengths),
                total_bytes=sum(lengths),
                elapsed=time.perf_counter() - start_ts,
                ret=ret,
            )
        return ret

    def _log_batch_transfer_failure(
        self,
        *,
        mooncake_session_id: str,
        num_blocks: int,
        total_bytes: int,
        elapsed: float,
        ret: int,
    ) -> None:
        """Report the shape of the RDMA batch that failed.

        Block count, byte size and how long the call ran before failing are the
        things the engine's return code does not carry, and they decide whether
        the failure tracks transfer size or is independent of it -- i.e. whether
        capping the chunk size can help at all.
        """
        logger.error(
            "batch_transfer_sync failed: ret=%s session=%s blocks=%d "
            "bytes=%.1fMB elapsed=%.3fs",
            ret,
            mooncake_session_id,
            num_blocks,
            total_bytes / 1024**2,
            elapsed,
        )

    def _log_send_chunk_failure(
        self,
        *,
        kv_chunk: TransferKVChunk,
        req: TransferInfo,
        ret: int,
        elapsed: float,
        is_dcp_transfer: bool,
    ) -> None:
        """Report the shape of the chunk whose transfer failed.

        The failure record kept for the peer only names the room and the
        endpoint, so the chunk size, the transfer path taken and how long the
        call ran before failing are otherwise unrecoverable from the logs -- and
        those are exactly what tells us whether a size cap changes anything.
        """
        num_pages = len(kv_chunk.prefill_kv_indices)
        logger.error(
            "Failed to send kv chunk: room=%s ret=%s dst=%s session=%s "
            "pages=%d tokens=%s bytes=%.1fMB index_slice=[%s, %s) "
            "last_chunk=%s dcp=%s staging=%s elapsed=%.3fs",
            kv_chunk.room,
            ret,
            NetworkAddress(req.endpoint, req.dst_port).to_host_port_str(),
            req.mooncake_session_id,
            num_pages,
            kv_chunk.num_kv_tokens,
            num_pages * self.kv_item_lens_sum / 1024**2,
            kv_chunk.index_slice.start,
            kv_chunk.index_slice.stop,
            kv_chunk.is_last_chunk,
            is_dcp_transfer,
            self.enable_staging,
            elapsed,
        )

    def _transfer_data_arrays(
        self,
        mooncake_session_id,
        src_addrs: npt.NDArray[np.int64],
        dst_addrs: npt.NDArray[np.int64],
        lengths: npt.NDArray[np.int64],
    ):
        """Same contract as :meth:`_transfer_data`, for callers that already hold
        the addresses as NumPy arrays.

        Avoids building one Python tuple per block only for ``zip(*blocks)`` to
        take them apart again -- that round trip dominates when the block count
        is token-granular (DCP relayout).
        """
        if src_addrs.size == 0:
            return 0

        start_ts = time.perf_counter()
        ret = self.engine.batch_transfer_sync(
            mooncake_session_id,
            src_addrs.tolist(),
            dst_addrs.tolist(),
            lengths.tolist(),
        )
        if ret != 0:
            self._log_batch_transfer_failure(
                mooncake_session_id=mooncake_session_id,
                num_blocks=int(lengths.size),
                total_bytes=int(lengths.sum()),
                elapsed=time.perf_counter() - start_ts,
                ret=ret,
            )
        return ret

    def _send_kvcache_generic(
        self,
        mooncake_session_id: str,
        src_data_ptrs: list[int],
        dst_data_ptrs: list[int],
        item_lens: list[int],
        prefill_data_indices: npt.NDArray[np.int32],
        dst_data_indices: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
        state_type: Optional[StateType] = None,
        force_flat: bool = False,
        src_layer_ids: Optional[List[int]] = None,
        dst_layer_ids: Optional[List[int]] = None,
    ) -> int:
        """
        Generic KV cache transfer supporting both MHA and MLA architectures.
        This method is used by both send_kvcache (full pool) and maybe_send_extra.

        ``force_flat`` uses the MLA-style flat (single-buffer-per-layer) layout
        even on a non-MLA backend, for K-only state buffers (e.g. MiniMax sparse
        index) whose per-layer list must not be half-split into K/V.
        """
        layers_params = None

        # Decode pp size should be equal to prefill pp size or 1
        if self.is_mla_backend or self.is_hybrid_mla_backend or force_flat:
            # Layer IDs map PP-local buffers to global decode entries.
            # Registrations without them retain the existing PP mapping.
            if src_layer_ids or dst_layer_ids:
                pairs = build_transfer_entry_pairs(
                    src_layer_ids,
                    dst_layer_ids,
                    len(src_data_ptrs),
                    len(dst_data_ptrs),
                    allow_positional_fallback=self.pp_size == 1,
                )
                layers_params = [
                    (src_data_ptrs[i], dst_data_ptrs[j], item_lens[i]) for i, j in pairs
                ]
            else:
                src_kv_ptrs, dst_kv_ptrs, layers_current_pp_stage = (
                    self.get_mla_kv_ptrs_with_pp(
                        src_data_ptrs, dst_data_ptrs, state_type
                    )
                )
                layers_params = [
                    (
                        src_kv_ptrs[layer_id],
                        dst_kv_ptrs[layer_id],
                        item_lens[layer_id],
                    )
                    for layer_id in range(layers_current_pp_stage)
                ]
        else:
            src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_current_pp_stage = (
                self.get_mha_kv_ptrs_with_pp(src_data_ptrs, dst_data_ptrs)
            )
            # item_lens structure: [k_layer0, k_layer1, ..., k_layerN, v_layer0, v_layer1, ..., v_layerN]
            # Use correct item lengths for K and V separately
            if layers_current_pp_stage > len(dst_k_ptrs):
                logger.error(
                    "Prefill transfer kvcache error, layers_current_pp_stage is out of range: "
                    f"layers_current_pp_stage={layers_current_pp_stage}, len(dst_k_ptrs)={len(dst_k_ptrs)}"
                )
                return -1
            layers_params = [
                (
                    src_k_ptrs[layer_id],
                    dst_k_ptrs[layer_id],
                    item_lens[layer_id],  # K item length
                )
                for layer_id in range(layers_current_pp_stage)
            ] + [
                (
                    src_v_ptrs[layer_id],
                    dst_v_ptrs[layer_id],
                    item_lens[layers_current_pp_stage + layer_id],  # V item length
                )
                for layer_id in range(layers_current_pp_stage)
            ]
        assert layers_params is not None
        if not layers_params:
            return 0

        src_ptrs, dst_ptrs, entry_item_lens = zip(*layers_params)
        src_addrs, dst_addrs, lengths = build_paged_transfer_descriptors(
            src_ptrs=src_ptrs,
            dst_ptrs=dst_ptrs,
            item_lens=entry_item_lens,
            src_indices=prefill_data_indices,
            dst_indices=dst_data_indices,
        )
        if lengths.size == 0:
            return 0

        if self.enable_custom_mem_pool:
            # One transfer per entry so the custom (NVLINK) memory pool can
            # overlap them. The descriptors are entry-major, so entry i owns the
            # i-th slice of runs -- a view, no copy.
            runs_per_entry = lengths.size // len(layers_params)
            futures = [
                executor.submit(
                    self._transfer_data_arrays,
                    mooncake_session_id,
                    src_addrs[start : start + runs_per_entry],
                    dst_addrs[start : start + runs_per_entry],
                    lengths[start : start + runs_per_entry],
                )
                for start in range(0, lengths.size, runs_per_entry)
            ]
            for future in concurrent.futures.as_completed(futures):
                status = future.result()
                if status != 0:
                    for f in futures:
                        f.cancel()
                    return status
            return 0

        # Combining all layers' params in one batch transfer is more efficient
        # compared to using multiple threads
        return self._transfer_data_arrays(
            mooncake_session_id, src_addrs, dst_addrs, lengths
        )

    def send_kvcache(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
        dst_layer_ids: Optional[List[int]] = None,
        src_entry_count: Optional[int] = None,
        dst_entry_count: Optional[int] = None,
    ):
        """Whole-entry flat transfer of the registered KV buffers.

        ``src_entry_count`` / ``dst_entry_count`` cap how many leading entries
        take this path. The caller passes the MLA (main) entry counts when the
        registration also carries speculative-draft (MHA) entries: those are
        head-sharded per side and, with a window-compact decode draft pool, live
        at different slot ids, so they take ``send_kvcache_draft`` instead. The
        two counts differ per side -- a PP prefill stage registers only its own
        layers' entries while the decode peer registers the whole model's.
        """
        src_data_ptrs = self.kv_args.kv_data_ptrs
        item_lens = self.kv_args.kv_item_lens
        src_layer_ids = self.kv_args.kv_layer_ids
        if src_entry_count is not None:
            src_data_ptrs = src_data_ptrs[:src_entry_count]
            item_lens = item_lens[:src_entry_count]
            src_layer_ids = src_layer_ids[:src_entry_count] if src_layer_ids else []
        if dst_entry_count is not None:
            dst_kv_ptrs = dst_kv_ptrs[:dst_entry_count]
            dst_layer_ids = dst_layer_ids[:dst_entry_count] if dst_layer_ids else []
        return self._send_kvcache_generic(
            mooncake_session_id=mooncake_session_id,
            src_data_ptrs=src_data_ptrs,
            dst_data_ptrs=dst_kv_ptrs,
            item_lens=item_lens,
            prefill_data_indices=prefill_kv_indices,
            dst_data_indices=dst_kv_indices,
            executor=executor,
            src_layer_ids=src_layer_ids,
            dst_layer_ids=dst_layer_ids,
        )

    def send_kvcache_dcp(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        *,
        dcp_token_item_lens: List[int],
        dst_dcp_size: int,
        dst_dcp_rank: int,
        src_page_offset: int,
        decode_prefix_len: int,
        num_kv_tokens: int,
        executor: concurrent.futures.ThreadPoolExecutor,
        dst_layer_ids: List[int],
        src_mla_entry_count: int,
        dst_mla_entry_count: int,
    ) -> int:
        if num_kv_tokens is None:
            raise ValueError("PD DCP transfer requires num_kv_tokens")
        # Restrict to the leading MLA (main) entries; the head-sharded draft
        # tail is transferred separately via send_kvcache_draft, since it is
        # DCP-replicated (whole token set at every rank) and head-sliced.
        # The boundary differs per side: a PP prefill stage registers only the
        # MLA entries of its own layers while the decode peer registers all of
        # them, so the two prefixes are paired by model layer id below.
        src_kv_ptrs_all = self.kv_args.kv_data_ptrs[:src_mla_entry_count]
        src_layer_ids = self.kv_args.kv_layer_ids[:src_mla_entry_count]
        dcp_token_item_lens = dcp_token_item_lens[:src_mla_entry_count]
        dst_kv_ptrs = dst_kv_ptrs[:dst_mla_entry_count]
        dst_layer_ids = dst_layer_ids[:dst_mla_entry_count]
        physical_page_size = self.kv_args.page_size
        plan = build_dcp_token_transfer_plan(
            prefill_kv_indices,
            dst_kv_indices,
            physical_page_size=physical_page_size,
            dcp_size=dst_dcp_size,
            dcp_rank=dst_dcp_rank,
            src_page_offset=src_page_offset,
            decode_prefix_len=decode_prefix_len,
            num_kv_tokens=num_kv_tokens,
        )
        if plan.src_token_indices.size == 0:
            return 0

        if src_layer_ids or dst_layer_ids:
            dst_indices = resolve_dcp_dst_entry_indices(
                src_layer_ids,
                dst_layer_ids,
                len(src_kv_ptrs_all),
                len(dst_kv_ptrs),
            )
            src_kv_ptrs = src_kv_ptrs_all
            dst_kv_ptrs = [dst_kv_ptrs[j] for j in dst_indices]
        else:
            src_kv_ptrs, dst_kv_ptrs, _ = self.get_mla_kv_ptrs_with_pp(
                src_kv_ptrs_all,
                dst_kv_ptrs,
            )
        layers_current_pp_stage = len(src_kv_ptrs)
        # Run-length form, not materialised groups: a DCP relayout strides the
        # source by dcp_size, so every run is a single token and the per-group
        # Python lists would be as long as the token count -- per layer.
        src_starts, dst_starts, run_counts = contiguous_run_starts_and_counts(
            plan.src_token_indices,
            plan.dst_token_indices,
        )

        layers_params = [
            (
                src_kv_ptrs[layer_id],
                dst_kv_ptrs[layer_id],
                dcp_token_item_lens[layer_id],
            )
            for layer_id in range(layers_current_pp_stage)
        ]

        def layer_addrs(
            src_ptr: int, dst_ptr: int, token_item_len: int
        ) -> Tuple[
            npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]
        ]:
            return (
                src_ptr + src_starts * token_item_len,
                dst_ptr + dst_starts * token_item_len,
                run_counts * token_item_len,
            )

        def process_layer(src_ptr: int, dst_ptr: int, token_item_len: int) -> int:
            return self._transfer_data_arrays(
                mooncake_session_id,
                *layer_addrs(src_ptr, dst_ptr, token_item_len),
            )

        if self.enable_custom_mem_pool:
            futures = [
                executor.submit(process_layer, src_ptr, dst_ptr, token_item_len)
                for src_ptr, dst_ptr, token_item_len in layers_params
            ]
            for future in concurrent.futures.as_completed(futures):
                status = future.result()
                if status != 0:
                    for pending in futures:
                        pending.cancel()
                    return status
            return 0

        per_layer = [layer_addrs(*params) for params in layers_params]
        if not per_layer:
            return 0
        return self._transfer_data_arrays(
            mooncake_session_id,
            np.concatenate([addrs[0] for addrs in per_layer]),
            np.concatenate([addrs[1] for addrs in per_layer]),
            np.concatenate([addrs[2] for addrs in per_layer]),
        )

    def _send_draft_slice_via_gather(
        self,
        mooncake_session_id: str,
        *,
        src_token_indices: npt.NDArray[np.int64],
        dst_token_indices: npt.NDArray[np.int64],
        draft_dst_kv_ptrs: list[int],
        draft_src_kv_ptrs: list[int],
        src_head_start: int,
        num_heads_to_send: int,
        src_token_item_len: int,
        src_head_slice_offset: int,
        dst_token_item_len: int,
        dst_head_slice_offset: int,
        heads_bytes_per_token: int,
    ) -> Optional[int]:
        """Gather the draft head slice into staging, then send it in one batch.

        Returns the transfer status, or ``None`` when this configuration can't use
        the gather (the caller then falls back to per-token descriptors).
        """
        from sglang.srt.disaggregation.common import draft_kv_gather

        if dst_head_slice_offset != 0 or heads_bytes_per_token != dst_token_item_len:
            logger.info_once(
                "PD DCP draft gather disabled: destination slot is not fully "
                f"covered (dst_off={dst_head_slice_offset} "
                f"slice={heads_bytes_per_token} dst_item={dst_token_item_len})"
            )
            return None

        layout = draft_kv_gather.build_draft_gather_layout(
            entry_tensors=self._draft_kv_entry_tensors,
            entry_ptrs=draft_src_kv_ptrs,
            head_start=src_head_start,
            num_heads=num_heads_to_send,
            src_token_item_len=src_token_item_len,
            src_head_slice_offset=src_head_slice_offset,
            heads_bytes_per_token=heads_bytes_per_token,
        )
        if layout is None:
            return None

        entry_tensors = self._draft_kv_entry_tensors
        if len(draft_dst_kv_ptrs) != len(entry_tensors):
            # Staging is sized and laid out per LOCAL entry; a peer that split the
            # MLA/draft boundary differently would read past the gathered region.
            logger.info_once(
                "PD DCP draft gather disabled: peer has "
                f"{len(draft_dst_kv_ptrs)} draft entries, this rank has "
                f"{len(entry_tensors)}"
            )
            return None
        required_bytes = draft_kv_gather.staging_bytes_needed(
            layout=layout,
            num_entries=len(entry_tensors),
            num_tokens=src_token_indices.size,
        )
        logger.info_once(
            "PD DCP draft transfer gathers the head slice into staging: "
            f"slice={heads_bytes_per_token} src_item={src_token_item_len} "
            f"entries={len(entry_tensors)} staging={required_bytes} B"
        )
        with self._draft_gather_pool.borrow(required_bytes) as staging_buffer:
            draft_kv_gather.gather_token_slices(
                entry_ptrs_gpu=self._draft_kv_entry_ptrs_gpu,
                layout=layout,
                src_token_indices=src_token_indices,
                staging_buffer=staging_buffer,
                gpu_id=self.kv_args.gpu_id,
                max_ctas=self._draft_gather_max_ctas,
            )
            src_addrs, dst_addrs, lengths = draft_kv_gather.build_gathered_descriptors(
                staging_ptr=staging_buffer.get_ptr(),
                dst_entry_ptrs=draft_dst_kv_ptrs,
                dst_token_indices=dst_token_indices,
                layout=layout,
                dst_token_item_len=dst_token_item_len,
                dst_head_slice_offset=dst_head_slice_offset,
            )
            return self._transfer_data_arrays(
                mooncake_session_id, src_addrs, dst_addrs, lengths
            )

    def _build_draft_window_plan(
        self,
        *,
        src_pages: npt.NDArray[np.int64],
        dst_compact_slots: npt.NDArray[np.int64],
        src_page_offset: int,
        num_kv_tokens: int,
        window_start_tokens: int,
    ) -> Optional[Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]:
        """Intersect this chunk with the draft window, in TOKEN space.

        ``src_pages``/``src_page_offset`` are PAGE units (built by
        ``kv_to_page_indices``); the window args are TOKEN units. The page ->
        token expansion happens here, once. Returns the overlapping
        ``(src_token_slots, dst_compact_slots)`` or ``None``.
        """
        page_size = max(1, int(self.kv_args.page_size))
        capacity_tokens = int(src_pages.size) * page_size
        chunk_tokens = (
            capacity_tokens
            if num_kv_tokens is None
            else min(int(num_kv_tokens), capacity_tokens)
        )
        if chunk_tokens <= 0 or dst_compact_slots.size == 0:
            return None

        # chunk_end must be the chunk's real span: min(len(dst), len(src)) collapses
        # it to the window and the draft KV is silently never sent.
        chunk_start = int(src_page_offset) * page_size
        chunk_end = chunk_start + chunk_tokens
        window_end = int(window_start_tokens) + int(dst_compact_slots.size)
        lo = max(chunk_start, int(window_start_tokens))
        hi = min(chunk_end, window_end)
        if hi <= lo:
            return None

        offsets = np.arange(lo - chunk_start, hi - chunk_start, dtype=np.int64)
        src_token_slots = (
            src_pages[offsets // page_size] * page_size + offsets % page_size
        )
        dst_slice = dst_compact_slots[lo - int(window_start_tokens) : hi - int(window_start_tokens)]
        return src_token_slots, dst_slice

    def _mla_entry_counts(
        self, target_rank_registration_info: "KVArgsRegisterInfo"
    ) -> Tuple[int, int]:
        """(src, dst) counts of leading MLA (main) KV entries.
        Entries beyond these are the speculative-draft (MHA) tail. The boundary
        is per-side: a PP prefill stage registers only its own layers' MLA
        entries while the decode peer registers the whole model's. A zero count
        means the registration carries no draft tail at all, so the whole list is
        MLA.
        """
        src = self.kv_args.num_mla_kv_entries or len(self.kv_args.kv_item_lens)
        dst = (
            target_rank_registration_info.dst_num_mla_kv_entries
            or len(target_rank_registration_info.dst_kv_ptrs)
        )
        return src, dst

    def send_kvcache_draft(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        *,
        src_mla_entry_count: int,
        dst_mla_entry_count: int,
        dst_attn_tp_size: int,
        dst_tp_rank: int,
        draft_total_kv_heads: int,
        dst_dcp_size: int,
        src_page_offset: int,
        decode_prefix_len: int,
        num_kv_tokens: int,
        executor: concurrent.futures.ThreadPoolExecutor,
        draft_window_start: Optional[int] = None,
    ) -> int:
        """Transfer the speculative draft (MHA) KV entries.

        Serves both PD topologies, because the draft pool is never context-split:

        * ``draft_window_start`` set -- the decode peer runs a window-compact
          draft pool (``--speculative-draft-window-size``) and sends the compact
          destination slots of its window in ``dst_kv_indices``. This chunk is
          intersected with that window in TOKEN space
          (``_build_draft_window_plan``); ``dst_dcp_size`` plays no part, so the
          same path is correct for decode DCP and for plain decode DP.
        * ``draft_window_start`` unset -- the decode draft pool is full-size and
          shares the target's (virtual) page ids, so every token of the chunk is
          sent at its full loc. Under decode DCP that pool is *replicated* across
          the DCP ranks; ``dst_dcp_size=1`` reduces the same plan to the plain
          page -> token expansion.

        The source is always the prefill's own full-size draft pool over the
        target's global slot space: ``--speculative-draft-window-size`` is
        rejected on a prefill instance (see ``_handle_dspark``), so no source-side
        translation is needed.

        Draft KV is head-sharded per side, so heads are sliced M-to-N exactly like
        ``send_kvcache_slice`` -- that is what makes an ``attn_tp`` mismatch (e.g.
        prefill attn_tp=8 feeding a decode DP16 rank with attn_tp=1) correct here
        while the whole-entry ``send_kvcache`` path cannot express it.

        The MLA/draft boundary is per-side: a PP prefill stage registers only
        the MLA entries of its own layers while the decode peer registers the
        whole model's. So the draft entries are
        ``kv_data_ptrs[src_mla_entry_count:]`` on this side and
        ``dst_kv_ptrs[dst_mla_entry_count:]`` on the peer (layout
        ``[draft_K..., draft_V...]``, matching the main pool's K-then-V split).
        Only the last PP stage registers draft entries at all
        (``transfer_draft_cache`` in prefill.py), so earlier stages return here
        immediately.
        """
        draft_src_kv_ptrs = self.kv_args.kv_data_ptrs[src_mla_entry_count:]
        draft_dst_kv_ptrs = dst_kv_ptrs[dst_mla_entry_count:]
        if not draft_src_kv_ptrs:
            return 0

        physical_page_size = max(1, int(self.kv_args.page_size))
        src_pages = np.asarray(prefill_kv_indices, dtype=np.int64)
        dst_indices = np.asarray(dst_kv_indices, dtype=np.int64)

        if draft_window_start is None:
            # Full-size decode draft pool sharing the target's page ids.
            plan = build_dcp_replicated_token_transfer_plan(
                prefill_kv_indices,
                dst_kv_indices,
                physical_page_size=physical_page_size,
                dcp_size=dst_dcp_size,
                src_page_offset=src_page_offset,
                decode_prefix_len=decode_prefix_len,
                num_kv_tokens=num_kv_tokens,
            )
        else:
            window = self._build_draft_window_plan(
                src_pages=src_pages,
                dst_compact_slots=dst_indices,
                src_page_offset=src_page_offset,
                num_kv_tokens=num_kv_tokens,
                window_start_tokens=draft_window_start,
            )
            if window is None:
                return 0
            src_token_slots, dst_window_slots = window
            # Prefill runs a full-size draft pool over the target's global slot
            # space (--speculative-draft-window-size is decode-only; see
            # _handle_dspark in arg_groups/speculative_hook.py), so these source
            # slots are already the addresses this side owns and need no
            # translation. A compact prefill pool would silently send the wrong
            # source KV, so refuse that configuration outright.
            if self.server_args.speculative_draft_window_size is not None:
                raise RuntimeError(
                    "--speculative-draft-window-size is not supported on a "
                    "prefill instance: its draft KV would live at compact SWA "
                    "slots, which the PD draft transfer cannot address."
                )
            plan = DCPTokenTransferPlan(src_token_slots, dst_window_slots)
        if plan.src_token_indices.size == 0:
            return 0

        # kv_item_lens is per-PAGE bytes (item_bytes * page_size). The transfer
        # plan addresses tokens individually (src_token_indices are token-
        # granular), so reduce to a per-token stride before computing addresses;
        # otherwise every token index is inflated by page_size and the RDMA
        # addresses land outside the registered memory region.
        src_page_item_len = self.kv_args.kv_item_lens[src_mla_entry_count]
        src_token_item_len = src_page_item_len // physical_page_size
        # Head geometry: draft KV is head-sharded by attn_tp_size on each side.
        dst_heads_per_rank = max(1, draft_total_kv_heads // dst_attn_tp_size)
        src_heads_per_rank = max(1, draft_total_kv_heads // self.attn_tp_size)
        bytes_per_head = src_token_item_len // src_heads_per_rank
        dst_token_item_len = dst_heads_per_rank * bytes_per_head

        local_tp_rank_in_group = self.kv_args.engine_rank % self.attn_tp_size
        dst_tp_rank_in_group = dst_tp_rank % dst_attn_tp_size

        # Head-slice offsets: mirror send_kvcache_slice's M-to-N mapping.
        src_replication = max(1, self.attn_tp_size // draft_total_kv_heads)
        if self.attn_tp_size > dst_attn_tp_size:
            # Multiple prefill ranks feed one decode rank.
            src_head_start = 0
            num_heads_to_send = src_heads_per_rank
            unique_head_idx = local_tp_rank_in_group // src_replication
            dst_head_start = (
                unique_head_idx * src_heads_per_rank
            ) % dst_heads_per_rank
        else:
            # One prefill rank feeds multiple decode ranks.
            dst_replication = max(1, dst_attn_tp_size // draft_total_kv_heads)
            unique_dst_head_idx = dst_tp_rank_in_group // dst_replication
            src_head_start = (
                unique_dst_head_idx * dst_heads_per_rank
            ) % src_heads_per_rank
            num_heads_to_send = dst_heads_per_rank
            dst_head_start = 0

        src_head_slice_offset = src_head_start * bytes_per_head
        dst_head_slice_offset = dst_head_start * bytes_per_head
        heads_bytes_per_token = num_heads_to_send * bytes_per_head
        if heads_bytes_per_token > dst_token_item_len:
            logger.error(
                f"[{mooncake_session_id}] draft slice size "
                f"({heads_bytes_per_token}) exceeds target token slot size "
                f"({dst_token_item_len})"
            )
            return -1

        # The replicated draft plan selects EVERY source token (no dcp stride), so
        # consecutive tokens are usually adjacent on both sides. Merging them into
        # one descriptor per run is only address-equivalent when a token's slice
        # covers its whole slot on both sides -- with a head slice the per-token
        # blocks are strided and must stay separate.
        whole_token_slice = (
            src_head_slice_offset == 0
            and dst_head_slice_offset == 0
            and heads_bytes_per_token == src_token_item_len
            and heads_bytes_per_token == dst_token_item_len
        )
        if whole_token_slice:
            run_src_starts, run_dst_starts, run_counts = (
                contiguous_run_starts_and_counts(
                    plan.src_token_indices,
                    plan.dst_token_indices,
                )
            )
            src_token_offsets = run_src_starts * src_token_item_len
            dst_token_offsets = run_dst_starts * dst_token_item_len
            run_lengths = (run_counts * heads_bytes_per_token).tolist()
            logger.info_once(
                "PD DCP draft transfer coalesces whole-token runs "
                f"(src_item={src_token_item_len} dst_item={dst_token_item_len})"
            )
        else:
            # A head slice is strided in the token-major source pool, so descriptors
            # can only coalesce once the slice has been made contiguous. Gathering
            # into staging first turns ~one descriptor per token into ~one per
            # destination page; without it, building the list holds the GIL long
            # enough to stall this rank's scheduler (and the PP ring behind it).
            if self._draft_gather_pool is not None:
                status = self._send_draft_slice_via_gather(
                    mooncake_session_id,
                    src_token_indices=plan.src_token_indices,
                    dst_token_indices=plan.dst_token_indices,
                    draft_dst_kv_ptrs=draft_dst_kv_ptrs,
                    draft_src_kv_ptrs=draft_src_kv_ptrs,
                    src_head_start=src_head_start,
                    num_heads_to_send=num_heads_to_send,
                    src_token_item_len=src_token_item_len,
                    src_head_slice_offset=src_head_slice_offset,
                    dst_token_item_len=dst_token_item_len,
                    dst_head_slice_offset=dst_head_slice_offset,
                    heads_bytes_per_token=heads_bytes_per_token,
                )
                if status is not None:
                    return status
            src_token_offsets = (
                plan.src_token_indices * src_token_item_len + src_head_slice_offset
            )
            dst_token_offsets = (
                plan.dst_token_indices * dst_token_item_len + dst_head_slice_offset
            )
            run_lengths = [heads_bytes_per_token] * src_token_offsets.size
            logger.info_once(
                "PD DCP draft transfer stays per-token: head slice "
                f"src_off={src_head_slice_offset} dst_off={dst_head_slice_offset} "
                f"slice={heads_bytes_per_token} src_item={src_token_item_len} "
                f"dst_item={dst_token_item_len}"
            )

        def process_draft_layer(src_layer_ptr: int, dst_layer_ptr: int) -> int:
            if not run_lengths:
                return 0
            src_addr_list = (src_layer_ptr + src_token_offsets).tolist()
            dst_addr_list = (dst_layer_ptr + dst_token_offsets).tolist()
            return self.engine.batch_transfer_sync(
                mooncake_session_id, src_addr_list, dst_addr_list, run_lengths
            )

        futures = [
            executor.submit(process_draft_layer, src_ptr, dst_ptr)
            for src_ptr, dst_ptr in zip(draft_src_kv_ptrs, draft_dst_kv_ptrs)
        ]
        for future in concurrent.futures.as_completed(futures):
            status = future.result()
            if status != 0:
                logger.error(
                    "[PD DCP draft] batch transfer failed: session=%s status=%s "
                    "src_tokens=%d dst_tokens=%d layers=%d",
                    mooncake_session_id,
                    status,
                    len(plan.src_token_indices),
                    len(plan.dst_token_indices),
                    len(futures),
                )
                for f in futures:
                    f.cancel()
                return status
        return 0

    def send_kvcache_slice(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        dst_tp_rank: int,
        dst_attn_tp_size: int,
        dst_kv_item_len: int,
        executor: concurrent.futures.ThreadPoolExecutor,
    ):
        """
        Sends KV cache slices from this Prefill rank to a target Decode rank,
        supporting generic M-to-N TP size configurations.

        NOTE: This implementation calls the transfer engine for each token slot within
        each page to ensure correctness for any page_size and head-slicing configuration.
        This may introduce performance overhead (increased TTFT) for long sequences.
        """
        # Extract configuration
        local_tp_rank_in_group = self.kv_args.engine_rank % self.attn_tp_size
        src_kv_item_len = self.kv_args.kv_item_lens[0]
        dst_tp_rank_in_group = dst_tp_rank % dst_attn_tp_size
        page_size = self.kv_args.page_size

        # Use total KV head count (not per-rank) for correct head distribution.
        # Per-rank kv_head_num is max(1, total//tp) which loses info when total < tp.
        total_kv_heads = getattr(self.kv_args, "total_kv_head_num", 0)
        if total_kv_heads <= 0:
            total_kv_heads = self.kv_args.kv_head_num * self.attn_tp_size

        src_heads_per_rank = max(1, total_kv_heads // self.attn_tp_size)
        dst_heads_per_rank = max(1, total_kv_heads // dst_attn_tp_size)
        bytes_per_head_slice_to_send = (
            dst_kv_item_len // page_size // dst_heads_per_rank
        )

        # GQA replication: how many prefill ranks share the same KV head
        src_replication = max(1, self.attn_tp_size // total_kv_heads)

        # Determine slicing parameters based on TP configuration
        if self.attn_tp_size > dst_attn_tp_size:
            # Send KVCache from multiple prefill instances to 1 decode instance
            src_head_start_offset = 0
            num_heads_to_send = src_heads_per_rank
            unique_head_idx = local_tp_rank_in_group // src_replication
            dst_head_start_offset = (
                unique_head_idx * src_heads_per_rank
            ) % dst_heads_per_rank
        else:
            # Send KVCache from 1 prefill instance to multiple decode instances
            # GQA replication (total_kv_heads < dst_attn_tp_size): consecutive decode
            # ranks share one KV head (QKVParallelLinear: tp_rank // num_kv_head_replicas),
            # so map by integer division NOT modulo or ranks 1..r-1 fetch the wrong head.
            dst_replication = max(1, dst_attn_tp_size // total_kv_heads)
            unique_dst_head_idx = dst_tp_rank_in_group // dst_replication
            src_head_start_offset = (
                unique_dst_head_idx * dst_heads_per_rank
            ) % src_heads_per_rank
            num_heads_to_send = dst_heads_per_rank
            dst_head_start_offset = 0

        src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_current_pp_stage = (
            self.get_mha_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, dst_kv_ptrs)
        )

        # Calculate precise byte offset and length for the sub-slice within the token
        src_head_slice_offset = src_head_start_offset * bytes_per_head_slice_to_send
        dst_head_slice_offset = dst_head_start_offset * bytes_per_head_slice_to_send
        heads_bytes_per_token_to_send = num_heads_to_send * bytes_per_head_slice_to_send

        # Sanity check: The data sub-slice to be sent should fit into the dst buffer.
        # This means heads_bytes_per_token_to_send <= (dst_kv_item_len // page_size)
        if heads_bytes_per_token_to_send > (dst_kv_item_len // page_size):
            logger.error(
                f"[{mooncake_session_id}] slice size ({heads_bytes_per_token_to_send}) exceeds "
                f"target token slot size ({dst_kv_item_len // page_size})"
            )
            return -1

        prefill_page_indices = prefill_kv_indices.reshape(-1, 1).astype(np.int64)
        decode_page_indices = dst_kv_indices.reshape(-1, 1).astype(np.int64)
        tokens_per_page = np.arange(page_size, dtype=np.int64).reshape(1, -1)
        bytes_per_token_on_prefill = src_kv_item_len // page_size
        bytes_per_token_on_decode = dst_kv_item_len // page_size
        src_token_slot_offsets = (
            tokens_per_page * bytes_per_token_on_prefill + src_head_slice_offset
        )
        dst_token_slot_offsets = (
            tokens_per_page * bytes_per_token_on_decode + dst_head_slice_offset
        )

        def process_layer_tp_aware(src_layer_ptr, dst_layer_ptr):
            src_page_base_addrs = src_layer_ptr + prefill_page_indices * src_kv_item_len
            dst_page_base_addrs = dst_layer_ptr + decode_page_indices * dst_kv_item_len
            src_slice_addrs = src_page_base_addrs + src_token_slot_offsets
            dst_slice_addrs = dst_page_base_addrs + dst_token_slot_offsets

            src_addr_list = src_slice_addrs.reshape(-1).tolist()
            if not src_addr_list:
                # Nothing to transfer for this layer.
                return 0
            dst_addr_list = dst_slice_addrs.reshape(-1).tolist()
            total_slices = len(src_addr_list)
            length_list = [heads_bytes_per_token_to_send] * total_slices
            return self.engine.batch_transfer_sync(
                mooncake_session_id, src_addr_list, dst_addr_list, length_list
            )

        futures = []
        for i in range(layers_current_pp_stage):
            futures.append(
                executor.submit(process_layer_tp_aware, src_k_ptrs[i], dst_k_ptrs[i])
            )
        for i in range(layers_current_pp_stage):
            futures.append(
                executor.submit(process_layer_tp_aware, src_v_ptrs[i], dst_v_ptrs[i])
            )

        for future in concurrent.futures.as_completed(futures):
            status = future.result()
            if status != 0:
                for f in futures:
                    f.cancel()
                return status

        return 0

    def send_aux(
        self,
        req: TransferInfo,
        prefill_aux_index: int,
        dst_aux_ptrs: list[int],
    ):
        # TODO(shangming): Fix me when nvlink_transport of Mooncake is bug-free
        if (
            self.enable_custom_mem_pool and self.custom_mem_pool_type == "NVLINK"
        ) or envs.SGLANG_MOONCAKE_SEND_AUX_TCP.get():
            return self.send_aux_tcp(req, prefill_aux_index, dst_aux_ptrs)

        transfer_blocks = []
        prefill_aux_ptrs = self.kv_args.aux_data_ptrs
        prefill_aux_item_lens = self.kv_args.aux_item_lens

        for i, dst_aux_ptr in enumerate(dst_aux_ptrs):
            length = prefill_aux_item_lens[i]
            src_addr = prefill_aux_ptrs[i] + length * prefill_aux_index
            dst_addr = dst_aux_ptrs[i] + length * req.dst_aux_index
            transfer_blocks.append((src_addr, dst_addr, length))

        return self._transfer_data(req.mooncake_session_id, transfer_blocks)

    def send_aux_tcp(
        self,
        req: TransferInfo,
        prefill_aux_index: int,
        dst_aux_ptrs: list[int],
    ):
        prefill_aux_ptrs = self.kv_args.aux_data_ptrs
        prefill_aux_item_lens = self.kv_args.aux_item_lens

        for i in range(len(prefill_aux_ptrs)):
            length = prefill_aux_item_lens[i]
            src_addr = prefill_aux_ptrs[i] + length * prefill_aux_index
            data = AuxDataCodec.serialize_data_from_buffer(src_addr, length)

            self.send_aux_data_to_endpoint(
                remote=req.endpoint,
                dst_port=req.dst_port,
                room=req.room,
                buffer_index=i,
                aux_index=req.dst_aux_index,
                data=data,
            )

        return 0

    def send_aux_data_to_endpoint(
        self,
        remote: str,
        dst_port: int,
        room: int,
        buffer_index: int,
        aux_index: int,
        data: bytes,
    ):
        na = NetworkAddress(remote, dst_port)
        self._send_multipart_locked(
            na.to_tcp(),
            [
                MooncakeKVManager.AUX_DATA_HEADER,
                str(room).encode("ascii"),
                str(buffer_index).encode("ascii"),
                str(aux_index).encode("ascii"),
                struct.pack(">I", len(data)),
                data,
            ],
            is_ipv6=na.is_ipv6,
        )

    def _handle_aux_data(self, msg: List[bytes]):
        """Handle AUX_DATA messages received by the decode thread."""
        room = int(msg[1].decode("ascii"))
        buffer_index = int(msg[2].decode("ascii"))
        aux_index = int(msg[3].decode("ascii"))
        data_length = struct.unpack(">I", msg[4])[0]
        data = msg[5]

        if len(data) != data_length:
            logger.error(f"AUX_DATA length mismatch for bootstrap_room {room}")
            return

        AuxDataCodec.deserialize_data_to_buffer(
            self.kv_args, buffer_index, aux_index, data
        )

        logger.debug(
            f"Received AUX_DATA for bootstrap_room {room} with length:{len(data)}"
        )

    def _get_dsa_cache_transfer_skip_flags(
        self, info: Optional[KVArgsRegisterInfo]
    ) -> Tuple[bool, bool]:
        skip_kv = False
        skip_state = False
        if not self.is_hybrid_mla_backend:
            return skip_kv, skip_state

        if info is not None and self.attn_tp_size > info.dst_attn_tp_size:
            sub_rank = (self.kv_args.engine_rank % self.attn_tp_size) % (
                self.attn_tp_size // info.dst_attn_tp_size
            )
            if sub_rank != 0:
                skip_kv = True
                # Hybrid-MLA KV is replicated across these source ranks, but
                # TP-sharded state needs every rank for the aggregation path.

        if (
            self.attn_cp_size > 1
            and self.attn_cp_rank != 0
            and not self.server_args.enable_dsa_cache_layer_split
        ):
            skip_state = True

        return skip_kv, skip_state

    def maybe_send_extra(
        self,
        req: TransferInfo,
        prefill_state_indices: List,
        executor: concurrent.futures.ThreadPoolExecutor,
        target_rank_registration_info: Optional[KVArgsRegisterInfo] = None,
    ):
        rc = 0
        state_types = getattr(self.kv_args, "state_types", [])
        for i, st in enumerate(state_types):
            indices = (
                prefill_state_indices[i] if i < len(prefill_state_indices) else None
            )
            if indices is None:
                continue
            src_data_ptrs = self.kv_args.state_data_ptrs[i]
            src_item_lens = self.kv_args.state_item_lens[i]
            src_dim_per_tensor = (
                self.kv_args.state_dim_per_tensor[i]
                if i < len(self.kv_args.state_dim_per_tensor)
                else []
            )
            src_conv_shard_groups = getattr(self.kv_args, "state_conv_shard_groups", [])
            src_conv_shard_groups = (
                src_conv_shard_groups[i] if i < len(src_conv_shard_groups) else []
            )
            src_slice_outer_counts = getattr(
                self.kv_args, "state_slice_outer_counts", []
            )
            src_slice_outer_counts = (
                src_slice_outer_counts[i] if i < len(src_slice_outer_counts) else []
            )
            src_state_layer_ids = self.kv_args.state_layer_ids
            src_state_layer_ids = (
                src_state_layer_ids[i] if i < len(src_state_layer_ids) else []
            )
            if target_rank_registration_info is not None:
                dst_data_ptrs = (
                    target_rank_registration_info.dst_state_data_ptrs[i]
                    if i < len(target_rank_registration_info.dst_state_data_ptrs)
                    else []
                )
                dst_item_lens = (
                    target_rank_registration_info.dst_state_item_lens[i]
                    if i < len(target_rank_registration_info.dst_state_item_lens)
                    else []
                )
                dst_dim_per_tensor = (
                    target_rank_registration_info.dst_state_dim_per_tensor[i]
                    if i < len(target_rank_registration_info.dst_state_dim_per_tensor)
                    else []
                )
                dst_state_layer_ids = (
                    target_rank_registration_info.dst_state_layer_ids[i]
                    if i < len(target_rank_registration_info.dst_state_layer_ids)
                    else []
                )
            else:
                dst_data_ptrs, dst_item_lens, dst_dim_per_tensor = [], [], []
                dst_state_layer_ids = []
            dst_indices = (
                req.dst_state_indices[i] if i < len(req.dst_state_indices) else []
            )

            if st == StateType.MAMBA:
                if (
                    target_rank_registration_info is not None
                    and self.attn_tp_size
                    != target_rank_registration_info.dst_attn_tp_size
                ):
                    rc = (
                        self._send_mamba_state_slice(
                            req,
                            indices,
                            src_data_ptrs,
                            src_item_lens,
                            src_dim_per_tensor,
                            dst_data_ptrs,
                            dst_indices,
                            dst_item_lens,
                            dst_dim_per_tensor,
                            target_rank_registration_info.dst_tp_rank,
                            target_rank_registration_info.dst_attn_tp_size,
                            src_conv_shard_groups,
                            src_slice_outer_counts,
                            src_state_layer_ids,
                            dst_state_layer_ids,
                        )
                        or rc
                    )
                else:
                    rc = (
                        self._send_mamba_state(
                            req,
                            indices,
                            src_data_ptrs,
                            src_item_lens,
                            dst_data_ptrs,
                            dst_indices,
                            src_state_layer_ids,
                            dst_state_layer_ids,
                        )
                        or rc
                    )
            elif st in (
                StateType.SWA,
                StateType.DSA,
                StateType.SWA_RING,
                StateType.C128_STATE,
            ):
                if (
                    target_rank_registration_info is not None
                    and not self.is_mla_backend
                    and self.attn_tp_size
                    != target_rank_registration_info.dst_attn_tp_size
                ):
                    raise RuntimeError(
                        f"PD Disaggregation does NOT support PD different TP sizes for non-MLA {st.upper()} hybrid models yet."
                    )
                src_indices = list(indices)
                dst_indices_local = list(dst_indices)
                if (
                    st == StateType.C128_STATE
                    and len(src_indices) == 0
                    and len(dst_indices_local) == 0
                ):
                    continue
                if len(src_indices) != len(dst_indices_local):
                    # These components are position- or request-indexed:
                    # truncating silently misaligns rows and corrupts KV.
                    # Paged SWA/DSA tolerate a 1-page drift -> keep the
                    # lenient truncation below.
                    if st in (StateType.SWA_RING, StateType.C128_STATE):
                        raise RuntimeError(
                            f"{st.upper()} state index length mismatch: "
                            f"prefill={len(src_indices)}, dst={len(dst_indices_local)}"
                        )
                    logger.warning(
                        f"len(prefill_state_indices) = {len(src_indices)}, len(dst_state_indices) = {len(dst_indices_local)}"
                    )
                    if len(src_indices) > len(dst_indices_local):
                        src_indices = src_indices[: len(dst_indices_local)]
                    else:
                        dst_indices_local = dst_indices_local[: len(src_indices)]
                rc = (
                    self._send_kvcache_generic(
                        mooncake_session_id=req.mooncake_session_id,
                        src_data_ptrs=src_data_ptrs,
                        dst_data_ptrs=dst_data_ptrs,
                        item_lens=src_item_lens,
                        prefill_data_indices=np.array(src_indices, dtype=np.int32),
                        dst_data_indices=np.array(dst_indices_local, dtype=np.int32),
                        executor=executor,
                        state_type=st,
                    )
                    or rc
                )
            elif st == StateType.MINIMAX_INDEX_K:
                # Equal-TP / PP=1 only. Sub-pools are compacted sparse-layer
                # lists, so PP>1 mis-slices and heterogeneous TP is unsupported.
                if self.pp_size is not None and self.pp_size > 1:
                    raise RuntimeError(
                        "PD disagg: PP>1 not supported for MiniMax sparse index yet."
                    )
                if (
                    target_rank_registration_info is not None
                    and self.attn_tp_size
                    != target_rank_registration_info.dst_attn_tp_size
                ):
                    raise RuntimeError(
                        "PD disagg: heterogeneous TP not supported for MiniMax "
                        "sparse index yet."
                    )
                src_indices = list(indices)
                dst_indices_local = list(dst_indices)
                if len(src_indices) > len(dst_indices_local):
                    src_indices = src_indices[: len(dst_indices_local)]
                elif len(src_indices) < len(dst_indices_local):
                    dst_indices_local = dst_indices_local[: len(src_indices)]
                rc = (
                    self._send_kvcache_generic(
                        mooncake_session_id=req.mooncake_session_id,
                        src_data_ptrs=src_data_ptrs,
                        dst_data_ptrs=dst_data_ptrs,
                        item_lens=src_item_lens,
                        prefill_data_indices=np.array(src_indices, dtype=np.int32),
                        dst_data_indices=np.array(dst_indices_local, dtype=np.int32),
                        executor=executor,
                        force_flat=True,
                    )
                    or rc
                )
        return rc

    def _send_mamba_state(
        self,
        req: TransferInfo,
        prefill_mamba_index: list,
        src_state_data_ptrs: list[int],
        src_state_item_lens: list[int],
        dst_state_data_ptrs: list[int],
        dst_mamba_index: list,
        src_layer_ids: Optional[List[int]] = None,
        dst_layer_ids: Optional[List[int]] = None,
    ):
        assert len(prefill_mamba_index) == 1, "Mamba should have single state index"

        transfer_blocks = []
        pairs = build_transfer_entry_pairs(
            src_layer_ids or [],
            dst_layer_ids or [],
            len(src_state_data_ptrs),
            len(dst_state_data_ptrs),
            allow_positional_fallback=self.pp_size == 1,
        )
        for i, j in pairs:
            dst_state_ptr = dst_state_data_ptrs[j]
            length = src_state_item_lens[i]
            src_addr = src_state_data_ptrs[i] + length * int(prefill_mamba_index[0])
            dst_addr = dst_state_ptr + length * int(dst_mamba_index[0])
            transfer_blocks.append((src_addr, dst_addr, length))

        return self._transfer_data(req.mooncake_session_id, transfer_blocks)

    def _send_mamba_state_slice(
        self,
        req: TransferInfo,
        prefill_mamba_index: list,
        src_state_data_ptrs: list[int],
        src_state_item_lens: list[int],
        src_state_dim_per_tensor: list[int],
        dst_state_data_ptrs: list[int],
        dst_mamba_index: list,
        dst_state_item_lens: list[int],
        dst_state_dim_per_tensor: list[int],
        dst_tp_rank: int,
        dst_attn_tp_size: int,
        src_state_conv_shard_groups: list = None,
        src_state_slice_outer_counts: list[int] = None,
        src_layer_ids: Optional[List[int]] = None,
        dst_layer_ids: Optional[List[int]] = None,
    ):
        """Transfer Mamba states with TP slice support.

        Mamba state layout:
        - conv_state: [num_layers, size+1, conv_dim/tp, conv_kernel-1]
        - temporal_state: [num_layers, size+1, num_heads/tp, head_dim, state_size]

        The 3rd dimension is sliced by TP. When prefill and decode have different
        attn_tp_size, we slice the state accordingly. GDN conv_state is the
        concatenation [query | key | value] with each sub-block head-sharded
        independently, so on the scatter path it is sliced per sub-block via
        ``src_state_conv_shard_groups`` (see compute_mamba_state_slice_blocks).
        """
        logger.warning_once(
            "Using Mamba state slice transfer for different TP sizes between prefill and decode. "
            f"Prefill attn_tp_size={self.attn_tp_size}, Decode attn_tp_size={dst_attn_tp_size}. "
            "Performance may be affected."
        )
        assert len(prefill_mamba_index) == 1, "Mamba should have single state index"

        # If no dimension info available, fall back to regular transfer
        if not src_state_dim_per_tensor or not dst_state_dim_per_tensor:
            return self._send_mamba_state(
                req,
                prefill_mamba_index,
                src_state_data_ptrs,
                src_state_item_lens,
                dst_state_data_ptrs,
                dst_mamba_index,
                src_layer_ids,
                dst_layer_ids,
            )

        local_tp_rank_in_group = self.kv_args.engine_rank % self.attn_tp_size
        dst_tp_rank_in_group = dst_tp_rank % dst_attn_tp_size

        transfer_blocks = []
        pairs = build_transfer_entry_pairs(
            src_layer_ids or [],
            dst_layer_ids or [],
            len(src_state_data_ptrs),
            len(dst_state_data_ptrs),
            allow_positional_fallback=self.pp_size == 1,
        )
        for i, j in pairs:
            dst_state_ptr = dst_state_data_ptrs[j]
            src_item_len = src_state_item_lens[i]
            dst_item_len = dst_state_item_lens[j]
            src_dim = src_state_dim_per_tensor[i]
            dst_dim = dst_state_dim_per_tensor[j]

            conv_shard_groups = (
                src_state_conv_shard_groups[i]
                if src_state_conv_shard_groups and i < len(src_state_conv_shard_groups)
                else None
            )
            outer_count = (
                src_state_slice_outer_counts[i]
                if src_state_slice_outer_counts
                and i < len(src_state_slice_outer_counts)
                else 1
            )
            for (
                src_offset,
                dst_offset,
                bytes_to_send,
            ) in compute_mamba_state_slice_byte_blocks(
                src_item_len=src_item_len,
                dst_item_len=dst_item_len,
                src_dim=src_dim,
                dst_dim=dst_dim,
                outer_count=outer_count,
                src_attn_tp_size=self.attn_tp_size,
                dst_attn_tp_size=dst_attn_tp_size,
                dst_tp_rank_in_group=dst_tp_rank_in_group,
                local_tp_rank_in_group=local_tp_rank_in_group,
                conv_shard_groups=conv_shard_groups,
            ):
                src_addr = (
                    src_state_data_ptrs[i]
                    + src_item_len * int(prefill_mamba_index[0])
                    + src_offset
                )
                dst_addr = (
                    dst_state_ptr + dst_item_len * int(dst_mamba_index[0]) + dst_offset
                )
                transfer_blocks.append((src_addr, dst_addr, bytes_to_send))

        return self._transfer_data(req.mooncake_session_id, transfer_blocks)

    def sync_status_to_decode_endpoint(
        self, remote: str, dst_port: int, room: int, status: int, prefill_rank: int
    ):
        na = NetworkAddress(remote, dst_port)
        self._send_multipart_locked(
            na.to_tcp(),
            [
                str(room).encode("ascii"),
                str(status).encode("ascii"),
                str(prefill_rank).encode("ascii"),
            ],
            is_ipv6=na.is_ipv6,
        )

    def transfer_worker(
        self,
        queue: FastQueue,
        executor: concurrent.futures.ThreadPoolExecutor,
        staging_buffer=None,
        worker_index=0,
    ):
        staging_strategy = None
        if self.enable_trace:
            trace_set_thread_info(
                f"mooncake transfer worker {worker_index}",
                tp_rank=self.attn_tp_rank,
                dp_rank=self.attn_dp_rank,
            )

        while True:
            try:
                kv_chunk: TransferKVChunk = queue.get()
                if isinstance(kv_chunk, PerTargetKVChunk):
                    # Per-destination work unit: handled by the parallel path in
                    # target_parallel_transfer.py. Everything below is the untouched
                    # legacy whole-room path, reached whenever the fan-out flag is
                    # off. Kept inside the same `try` so a unit still gets the
                    # legacy exception handling (fail the room, drop its infos).
                    if (
                        self.enable_staging
                        and staging_strategy is None
                        and staging_buffer is not None
                    ):
                        staging_strategy = self._try_create_staging_strategy(
                            staging_buffer
                        )
                    process_target_parallel_chunk(
                        self,
                        kv_chunk=kv_chunk,
                        executor=executor,
                        queue=queue,
                        staging_strategy=staging_strategy,
                    )
                    continue

                if self.enable_trace:
                    kv_chunk.trace_ctx.rebuild_thread_context()
                    kv_chunk.trace_ctx.trace_slice_start(
                        MooncakeRequestStage.MOONCAKE_WORKER_SEND.stage_name,
                        MooncakeRequestStage.MOONCAKE_WORKER_SEND.level,
                    )

                if (
                    kv_chunk.room not in self.request_status
                    or self.check_status(kv_chunk.room) == KVPoll.Failed
                ):
                    logger.debug(
                        f"Skipping chunk for room {kv_chunk.room} because it has already failed or been aborted"
                    )
                    if self.enable_trace:
                        kv_chunk.trace_ctx.trace_slice_end(
                            MooncakeRequestStage.MOONCAKE_WORKER_SEND.stage_name,
                            MooncakeRequestStage.MOONCAKE_WORKER_SEND.level,
                            thread_finish_flag=True,
                        )
                    continue

                if (
                    self.enable_staging
                    and staging_strategy is None
                    and staging_buffer is not None
                ):
                    staging_strategy = self._try_create_staging_strategy(staging_buffer)
                reqs_to_be_processed = (
                    self.transfer_infos[kv_chunk.room].values()
                    if kv_chunk.room in self.transfer_infos
                    else []
                )
                polls = []
                dst_ranks_infos = []
                # Unique id per prefill sender so decode's response set size matches expected_response_num.
                prefill_unique_rank = (
                    self.attn_tp_rank * (self.pp_size * self.attn_cp_size)
                    + self.pp_rank * self.attn_cp_size
                    + self.attn_cp_rank
                )
                # When staging transfer is not yet ready (watermark/allocation pending),
                # the chunk is re-enqueued and we break out of the req loop to retry later.
                staging_deferred = False
                for req in reqs_to_be_processed:
                    start_ts = time.perf_counter()
                    if not req.is_dummy:
                        # Early exit if the request has failed
                        with self.session_lock:
                            if req.mooncake_session_id in self.failed_sessions:
                                # Check cooldown: if the failed session's cooldown has
                                # expired, release the session and continue transferring;
                                # otherwise keep treating it as failed.
                                continue_transfer = False
                                if req.mooncake_session_id in self.session_cooldowns:
                                    time_since_fail = (
                                        time.time()
                                        - self.session_cooldowns[req.mooncake_session_id]
                                    )
                                    if time_since_fail >= self.session_cooldown_seconds:
                                        self.failed_sessions.remove(
                                            req.mooncake_session_id
                                        )
                                        del self.session_cooldowns[
                                            req.mooncake_session_id
                                        ]
                                        self.session_failures[
                                            req.mooncake_session_id
                                        ] = 0
                                        logger.info(
                                            f"Session {req.mooncake_session_id} "
                                            f"cooldown expired, released."
                                        )
                                        continue_transfer = True
                                    else:
                                        logger.info(
                                            f"Session {req.mooncake_session_id} is "
                                            f"still cooling down, will check again later."
                                        )
                                if not continue_transfer:
                                    self.record_failure(
                                        kv_chunk.room,
                                        f"Decode instance could be dead, remote mooncake "
                                        f"session {req.mooncake_session_id} is not alive",
                                    )
                                    self.update_status(kv_chunk.room, KVPoll.Failed)
                                    self.sync_status_to_decode_endpoint(
                                        req.endpoint,
                                        req.dst_port,
                                        req.room,
                                        KVPoll.Failed,
                                        prefill_unique_rank,
                                    )
                                    break

                        target_rank_registration_info: KVArgsRegisterInfo = (
                            self.decode_kv_args_table[req.mooncake_session_id]
                        )
                        is_dcp_transfer = (
                            target_rank_registration_info.requires_dcp_relayout
                        )
                        if is_dcp_transfer:
                            chunked_dst_kv_indice = req.dst_kv_indices
                        else:
                            chunked_dst_kv_indice = req.dst_kv_indices[
                                kv_chunk.index_slice
                            ]

                            # NOTE: This is temporarily a workaround to deal with the case where the prefill_kv_indices
                            # is mismatched with the dst_kv_indices when page size > 1, this should never happen.
                            if len(chunked_dst_kv_indice) < len(
                                kv_chunk.prefill_kv_indices
                            ):
                                logger.warning(
                                    f"len(chunked_dst_kv_indice) = {len(chunked_dst_kv_indice)}, len(kv_chunk.prefill_kv_indices) = {len(kv_chunk.prefill_kv_indices)}"
                                )
                                kv_chunk.prefill_kv_indices = (
                                    kv_chunk.prefill_kv_indices[
                                        : len(chunked_dst_kv_indice)
                                    ]
                                )

                        skip_kv, skip_state = self._get_dsa_cache_transfer_skip_flags(
                            target_rank_registration_info
                        )
                        if (
                            len(kv_chunk.prefill_kv_indices) == 0
                            or not self.kv_args.kv_data_ptrs
                            or skip_kv
                        ):
                            ret = 0
                        elif is_dcp_transfer:
                            dcp_token_item_lens = (
                                target_rank_registration_info.dcp_token_item_lens
                            )
                            assert dcp_token_item_lens is not None
                            # Split entries at the MLA/draft boundary: the MLA
                            # prefix takes DCP token relayout; the head-sharded
                            # draft tail (DCP-replicated) takes a head-slice
                            # whole-token transfer.
                            src_mla_entry_count, dst_mla_entry_count = (
                                self._mla_entry_counts(target_rank_registration_info)
                            )
                            ret = self.send_kvcache_dcp(
                                req.mooncake_session_id,
                                kv_chunk.prefill_kv_indices,
                                target_rank_registration_info.dst_kv_ptrs,
                                chunked_dst_kv_indice,
                                dcp_token_item_lens=dcp_token_item_lens,
                                dst_dcp_size=target_rank_registration_info.dst_dcp_size,
                                dst_dcp_rank=target_rank_registration_info.dst_dcp_rank,
                                src_page_offset=kv_chunk.index_slice.start or 0,
                                decode_prefix_len=req.decode_prefix_len or 0,
                                num_kv_tokens=kv_chunk.num_kv_tokens,
                                executor=executor,
                                dst_layer_ids=(
                                    target_rank_registration_info.dst_kv_layer_ids
                                ),
                                src_mla_entry_count=src_mla_entry_count,
                                dst_mla_entry_count=dst_mla_entry_count,
                            )
                            if ret == 0 and src_mla_entry_count < len(
                                self.kv_args.kv_item_lens
                            ):
                                draft_total_kv_heads = (
                                    self.kv_args.draft_kv_head_num * self.attn_tp_size
                                )
                                ret = self.send_kvcache_draft(
                                    req.mooncake_session_id,
                                    kv_chunk.prefill_kv_indices,
                                    target_rank_registration_info.dst_kv_ptrs,
                                    (
                                        req.draft_dst_kv_indices
                                        if req.draft_dst_kv_indices is not None
                                        else chunked_dst_kv_indice
                                    ),
                                    src_mla_entry_count=src_mla_entry_count,
                                    dst_mla_entry_count=dst_mla_entry_count,
                                    dst_attn_tp_size=(
                                        target_rank_registration_info.dst_attn_tp_size
                                    ),
                                    dst_tp_rank=(
                                        target_rank_registration_info.dst_tp_rank
                                    ),
                                    draft_total_kv_heads=draft_total_kv_heads,
                                    dst_dcp_size=(
                                        target_rank_registration_info.dst_dcp_size
                                    ),
                                    src_page_offset=kv_chunk.index_slice.start or 0,
                                    decode_prefix_len=req.decode_prefix_len or 0,
                                    num_kv_tokens=kv_chunk.num_kv_tokens,
                                    draft_window_start=req.draft_window_start,
                                    executor=executor,
                                )
                        elif (
                            self.is_mla_backend
                            or self.is_hybrid_mla_backend
                            or self.attn_tp_size
                            == target_rank_registration_info.dst_attn_tp_size
                        ):
                            # Same MLA/draft split as the DCP branch above, for
                            # the same reason: the whole-entry flat transfer can
                            # express neither the head slicing that a draft
                            # attn_tp mismatch needs (prefill attn_tp=8 -> decode
                            # DP16 attn_tp=1) nor a window-compact decode draft
                            # pool, whose slots are not the target's page ids.
                            src_mla_entry_count, dst_mla_entry_count = (
                                self._mla_entry_counts(target_rank_registration_info)
                            )
                            has_draft_entries = src_mla_entry_count < len(
                                self.kv_args.kv_item_lens
                            )
                            ret = self.send_kvcache(
                                req.mooncake_session_id,
                                kv_chunk.prefill_kv_indices,
                                target_rank_registration_info.dst_kv_ptrs,
                                chunked_dst_kv_indice,
                                executor,
                                target_rank_registration_info.dst_kv_layer_ids,
                                src_entry_count=(
                                    src_mla_entry_count if has_draft_entries else None
                                ),
                                dst_entry_count=(
                                    dst_mla_entry_count if has_draft_entries else None
                                ),
                            )
                            if ret == 0 and has_draft_entries:
                                draft_total_kv_heads = (
                                    self.kv_args.draft_kv_head_num * self.attn_tp_size
                                )
                                ret = self.send_kvcache_draft(
                                    req.mooncake_session_id,
                                    kv_chunk.prefill_kv_indices,
                                    target_rank_registration_info.dst_kv_ptrs,
                                    # The compact window destinations are the
                                    # WHOLE window, not a per-chunk slice:
                                    # send_kvcache_draft intersects them with this
                                    # chunk itself from src_page_offset.
                                    (
                                        req.draft_dst_kv_indices
                                        if req.draft_dst_kv_indices is not None
                                        else chunked_dst_kv_indice
                                    ),
                                    src_mla_entry_count=src_mla_entry_count,
                                    dst_mla_entry_count=dst_mla_entry_count,
                                    dst_attn_tp_size=(
                                        target_rank_registration_info.dst_attn_tp_size
                                    ),
                                    dst_tp_rank=(
                                        target_rank_registration_info.dst_tp_rank
                                    ),
                                    draft_total_kv_heads=draft_total_kv_heads,
                                    # No DCP relayout on this branch, so the
                                    # no-window fallback plan must not stride.
                                    dst_dcp_size=1,
                                    src_page_offset=kv_chunk.index_slice.start or 0,
                                    decode_prefix_len=req.decode_prefix_len or 0,
                                    num_kv_tokens=kv_chunk.num_kv_tokens,
                                    draft_window_start=req.draft_window_start,
                                    executor=executor,
                                )
                        elif (
                            self.enable_staging
                            and staging_strategy is not None
                            and target_rank_registration_info.staging is not None
                        ):
                            ret, deferred = self._do_staging_transfer(
                                staging_strategy,
                                kv_chunk,
                                req,
                                target_rank_registration_info,
                                chunked_dst_kv_indice,
                                executor,
                                queue,
                                prefill_unique_rank,
                            )
                            if deferred:
                                staging_deferred = True
                                # Chunk re-enqueued; stop processing remaining reqs for this chunk
                                break
                        else:
                            ret = self.send_kvcache_slice(
                                req.mooncake_session_id,
                                kv_chunk.prefill_kv_indices,
                                target_rank_registration_info.dst_kv_ptrs,
                                chunked_dst_kv_indice,
                                target_rank_registration_info.dst_tp_rank,
                                target_rank_registration_info.dst_attn_tp_size,
                                target_rank_registration_info.dst_kv_item_len,
                                executor,
                            )
                        if ret != 0:
                            self._log_send_chunk_failure(
                                kv_chunk=kv_chunk,
                                req=req,
                                ret=ret,
                                elapsed=time.perf_counter() - start_ts,
                                is_dcp_transfer=is_dcp_transfer,
                            )
                            with self.session_lock:
                                self.session_failures[req.mooncake_session_id] += 1
                                # Failures should never happen if the session is not dead, if the session fails once, mark it as failed
                                if self.session_failures[req.mooncake_session_id] >= 1:
                                    self.failed_sessions.add(req.mooncake_session_id)
                                    # 【新增】记录第一次失败的时间戳
                                    if req.mooncake_session_id not in self.session_cooldowns:
                                        self.session_cooldowns[req.mooncake_session_id] = time.time()
                                        logger.error(
                                            f"Session {req.mooncake_session_id} failed, cooldown started."
                                        )
                                    logger.error(
                                        f"Session {req.mooncake_session_id} failed."
                                    )
                            self.record_failure(
                                kv_chunk.room,
                                f"Failed to send kv chunk of {kv_chunk.room} to "
                                f"{NetworkAddress(req.endpoint, req.dst_port).to_host_port_str()}",
                            )
                            self.update_status(kv_chunk.room, KVPoll.Failed)
                            self.sync_status_to_decode_endpoint(
                                req.endpoint,
                                req.dst_port,
                                req.room,
                                KVPoll.Failed,
                                prefill_unique_rank,
                            )
                            break

                        if kv_chunk.is_last_chunk:
                            if kv_chunk.state_indices and not skip_state:
                                state_rc = self.maybe_send_extra(
                                    req,
                                    kv_chunk.state_indices,
                                    executor,
                                    target_rank_registration_info,
                                )
                                if state_rc != 0:
                                    with self.session_lock:
                                        self.session_failures[
                                            req.mooncake_session_id
                                        ] += 1
                                        self.failed_sessions.add(
                                            req.mooncake_session_id
                                        )
                                    self.record_failure(
                                        kv_chunk.room,
                                        f"Failed to send state components of {kv_chunk.room} to "
                                        f"{NetworkAddress(req.endpoint, req.dst_port).to_host_port_str()}",
                                    )
                                    self.update_status(kv_chunk.room, KVPoll.Failed)
                                    self.sync_status_to_decode_endpoint(
                                        req.endpoint,
                                        req.dst_port,
                                        req.room,
                                        KVPoll.Failed,
                                        prefill_unique_rank,
                                    )
                                    break

                            # Only the last chunk we need to send the aux data.
                            # prefill_aux_index is None when this rank is not the
                            # aux owner (PP+PD early-last-chunk); the room still
                            # concludes here so decode's per-rank response
                            # tracker still collects this rank.
                            if kv_chunk.prefill_aux_index is None:
                                ret = 0
                            else:
                                ret = self.send_aux(
                                    req,
                                    kv_chunk.prefill_aux_index,
                                    target_rank_registration_info.dst_aux_ptrs,
                                )
                            polls.append(True if ret == 0 else False)
                            dst_ranks_infos.append(
                                (req.endpoint, req.dst_port, req.room)
                            )

                            # Only sync status when all the dst ranks have received the kvcache
                            if len(polls) == req.required_dst_info_num:
                                status = KVPoll.Success if all(polls) else KVPoll.Failed
                                self.update_status(req.room, status)
                                for endpoint, dst_port, room in dst_ranks_infos:
                                    self.sync_status_to_decode_endpoint(
                                        endpoint,
                                        dst_port,
                                        room,
                                        status,
                                        prefill_unique_rank,
                                    )
                    else:
                        # Dummy request means the decode instance is not used, so its status can be marked as success directly
                        # Dummy request does not need to sync status to decode endpoint
                        if kv_chunk.is_last_chunk and req.room in self.request_status:
                            self.update_status(req.room, KVPoll.Success)

                    if self.enable_trace:
                        mooncake_trace_slice(
                            kv_chunk.trace_ctx,
                            MooncakeRequestStage.MOONCAKE_WORKER_SEND_SESSION,
                            start_ts,
                        )

                if self.enable_trace:
                    kv_chunk.trace_ctx.trace_slice_end(
                        MooncakeRequestStage.MOONCAKE_WORKER_SEND.stage_name,
                        MooncakeRequestStage.MOONCAKE_WORKER_SEND.level,
                        thread_finish_flag=True,
                    )

                if staging_deferred:
                    continue

                if (
                    kv_chunk.room not in self.request_status
                    or self.check_status(kv_chunk.room) == KVPoll.Success
                ):
                    if kv_chunk.room in self.transfer_infos:
                        self.transfer_infos.pop(kv_chunk.room)
                    self.req_to_decode_prefix_len.pop(kv_chunk.room, None)
                    if self.enable_staging:
                        # Purge prefetch bookkeeping for the finished room.
                        # Snapshot first: the scheduler thread adds concurrently.
                        for key in list(self._staging_ctx.prefetch_requested):
                            if key[0] == kv_chunk.room:
                                self._staging_ctx.prefetch_requested.discard(key)
                        self._staging_ctx.prefetched_rooms.discard(kv_chunk.room)

            except Exception as e:
                _targets = [
                    f"{r.endpoint}:{r.dst_port}"
                    for r in self.transfer_infos.get(kv_chunk.room, {}).values()
                    if not r.is_dummy
                ]
                logger.error(
                    f"Transfer worker encountered an error processing "
                    f"room {kv_chunk.room}: {e}. "
                    f"is_last={kv_chunk.is_last_chunk}, "
                    f"index_slice={kv_chunk.index_slice}, "
                    f"targets={_targets}. "
                    f"bootstrap_port={self.bootstrap_port}.",
                    exc_info=True,
                )
                # Mark the current request as failed and notify decode endpoint
                try:
                    self.record_failure(
                        kv_chunk.room,
                        f"Transfer worker exception: {e}",
                    )
                    self.update_status(kv_chunk.room, KVPoll.Failed)

                    # Best-effort notify decode endpoint
                    local_rank = self.attn_tp_rank * self.pp_size + self.pp_rank
                    reqs = self.transfer_infos.get(kv_chunk.room, {}).values()
                    for req in reqs:
                        if not req.is_dummy:
                            try:
                                self.sync_status_to_decode_endpoint(
                                    req.endpoint,
                                    req.dst_port,
                                    req.room,
                                    KVPoll.Failed,
                                    local_rank,
                                )
                            except Exception:
                                pass
                except Exception:
                    logger.error(
                        f"Failed to clean up after transfer worker error for room {kv_chunk.room}",
                        exc_info=True,
                    )
                finally:
                    # Clean up transfer_infos to prevent memory leak
                    self.transfer_infos.pop(kv_chunk.room, None)

    def start_prefill_thread(self):
        def bootstrap_thread():
            """This thread recvs pre-alloc notification from the decode engine"""
            # KVPoll.Bootstrapping -> KVPoll.WaitingForInput
            # Fail-fast circuit breaker (aligned with the glm/community
            # branches, luno-3962): a single malformed frame -- a wrong frame
            # count, a non-numeric int() field, a from_zmq unpack failure --
            # only costs this one message and the loop continues. But ten
            # consecutive failures means the peer is emitting systematically
            # bad frames (e.g. P/D protocol mismatch during a rolling
            # upgrade), the handshake can never recover, and every PD request
            # would silently abort -- so kill the process and let the
            # deployment restart cleanly. Before this guard, a single bad
            # packet killed the thread permanently and the whole rank started
            # aborting every PD request (incident: qianfan-vl-v1
            # bootstrap_thread struct.error).
            consecutive_failures = 0
            MAX_CONSECUTIVE_FAILURES = 10
            try:
                while True:
                    waiting_req_bytes = self.server_socket.recv_multipart()
                    try:
                        room = waiting_req_bytes[0].decode("ascii")
                        # Staging: decode reports consumption watermark back to
                        # prefill. Recognize the frame type unconditionally and
                        # gate only the handler, so a WATERMARK frame with
                        # staging disabled is dropped instead of being
                        # mis-parsed as a TransferInfo.
                        if room == "WATERMARK":
                            if self.enable_staging:
                                from sglang.srt.disaggregation.common.staging_handler import (
                                    handle_watermark_msg,
                                )

                                handle_watermark_msg(self._staging_ctx, waiting_req_bytes)
                        # Staging: decode replies with allocated staging offset
                        elif room == "STAGING_RSP":
                            from sglang.srt.disaggregation.common.staging_handler import (
                                handle_staging_rsp,
                            )

                            handle_staging_rsp(waiting_req_bytes, self.transfer_infos)
                        # PP + PD prefill sideband consensus: a peer prefill PP
                        # rank reports that it finished this room's local
                        # handshake.
                        elif room == "PP_READY":
                            self.handle_pp_ready(waiting_req_bytes)
                        # Decode-side abort notification: mark room as failed and ACK
                        elif room == "ABORT":
                            room_to_be_aborted = int(waiting_req_bytes[1].decode("ascii"))
                            decode_ip = waiting_req_bytes[2].decode("ascii")
                            decode_port = int(waiting_req_bytes[3].decode("ascii"))
                            self.clear_pp_ready(room_to_be_aborted)
                            # No need to abort the room if it has already succeeded
                            if (
                                room_to_be_aborted in self.request_status
                                and self.check_status(room_to_be_aborted) != KVPoll.Success
                            ):
                                self.update_status(room_to_be_aborted, KVPoll.Failed)
                                logger.debug(
                                    f"Received abort notification for room {room_to_be_aborted}, "
                                    f"marked as Failed"
                                )
                            else:
                                logger.debug(
                                    f"Received abort notification for room {room_to_be_aborted}, "
                                    f"ignoring (already completed or unknown)"
                                )
                            # Send ACK back to decode endpoint
                            try:
                                na = NetworkAddress(decode_ip, decode_port)
                                self._send_multipart_locked(
                                    na.to_tcp(),
                                    [
                                        b"ABORT_ACK",
                                        str(room_to_be_aborted).encode("ascii"),
                                    ],
                                    is_ipv6=na.is_ipv6,
                                )
                                logger.debug(
                                    f"Sent ABORT_ACK for room {room_to_be_aborted} to "
                                    f"{decode_ip}:{decode_port}"
                                )
                            except Exception as e:
                                logger.debug(
                                    f"Failed to send ABORT_ACK for room {room_to_be_aborted}: {e}"
                                )
                        else:
                            mooncake_session_id = waiting_req_bytes[3].decode("ascii")
                            if room == "None":
                                decode_kv_args = KVArgsRegisterInfo.from_zmq(waiting_req_bytes)
                                decode_kv_args.requires_dcp_relayout = self.requires_dcp_relayout(
                                    decode_kv_args.dst_dcp_size,
                                    decode_kv_args.dst_dcp_rank,
                                )
                                if decode_kv_args.requires_dcp_relayout:
                                    # Prefer the full per-entry list from the decode side so
                                    # heterogeneous geometry (e.g. the head-sharded draft
                                    # pool) survives; fall back to broadcasting the scalar
                                    # for older peers that only sent dst_kv_item_len.
                                    if decode_kv_args.dst_kv_item_lens:
                                        dst_page_item_lens = decode_kv_args.dst_kv_item_lens
                                    else:
                                        dst_page_item_lens = [
                                            decode_kv_args.dst_kv_item_len
                                        ] * len(decode_kv_args.dst_kv_ptrs)
                                    # Boundary before the head-sharded draft tail: validate
                                    # only the MLA prefix for exact geometry match.
                                    src_num_mla_kv_entries, dst_num_mla_kv_entries = (
                                        self._resolve_mla_entry_boundaries(
                                            decode_kv_args.dst_num_mla_kv_entries
                                        )
                                    )
                                    # Persist the reconciled boundary for the transfer
                                    # dispatch to split MLA vs draft entries.
                                    decode_kv_args.dst_num_mla_kv_entries = dst_num_mla_kv_entries
                                    # A PP prefill stage registers only its own layers, so
                                    # pair the decode entries by model layer id before
                                    # comparing geometry.
                                    paired_dst_page_item_lens = [
                                        dst_page_item_lens[j]
                                        for j in resolve_dcp_dst_entry_indices(
                                            self.kv_args.kv_layer_ids,
                                            decode_kv_args.dst_kv_layer_ids,
                                            len(self.kv_args.kv_item_lens),
                                            len(dst_page_item_lens),
                                        )
                                    ]
                                    decode_kv_args.dcp_token_item_lens = (
                                        self.prepare_dcp_token_item_lens(
                                            paired_dst_page_item_lens,
                                            num_mla_kv_entries=src_num_mla_kv_entries,
                                        )
                                    )
                                self.decode_kv_args_table[mooncake_session_id] = decode_kv_args
                                with self.session_lock:
                                    if mooncake_session_id in self.failed_sessions:
                                        self.failed_sessions.remove(mooncake_session_id)
                                    if mooncake_session_id in self.session_cooldowns:
                                        del self.session_cooldowns[mooncake_session_id]
                                    if mooncake_session_id in self.session_failures:
                                        del self.session_failures[mooncake_session_id]
                                logger.debug(
                                    f"Register KVArgs from {mooncake_session_id} successfully"
                                )
                            else:
                                required_dst_info_num = int(waiting_req_bytes[7].decode("ascii"))
                                room = int(room)
                                if room not in self.transfer_infos:
                                    self.transfer_infos[room] = {}

                                self.transfer_infos[room][mooncake_session_id] = (
                                    TransferInfo.from_zmq(waiting_req_bytes)
                                )
                                # NOTE: after bootstrapping we can mark the req as waiting for input
                                if len(self.transfer_infos[room]) == required_dst_info_num:
                                    self.resolve_kv_replica_factor(self.transfer_infos[room])
                                    self.req_to_decode_prefix_len[room] = next(
                                        (
                                            info.decode_prefix_len
                                            for info in self.transfer_infos[room].values()
                                            if info.decode_prefix_len is not None
                                        ),
                                        0,
                                    )
                                    if self._pp_pd_sideband_bootstrap:
                                        # The "all PP ranks are ready" AND moves off the PP
                                        # ring and onto this socket; see
                                        # CommonKVManager.on_local_pp_bootstrap_ready.
                                        self.on_local_pp_bootstrap_ready(room)
                                    else:
                                        self.update_status(room, KVPoll.WaitingForInput)
                        consecutive_failures = 0
                    except Exception as e:
                        consecutive_failures += 1
                        logger.error(
                            f"Error processing bootstrap message: {e}",
                            exc_info=True,
                        )
                        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                            logger.critical(
                                f"bootstrap_thread: {consecutive_failures} consecutive "
                                f"failures, terminating process"
                            )
                            kill_process_tree(os.getpid())
                        continue
            except Exception as e:
                logger.critical(
                    f"bootstrap_thread crashed with unrecoverable error: {e}",
                    exc_info=True,
                )
                kill_process_tree(os.getpid())

        threading.Thread(
            target=bootstrap_thread, name="MooncakeBootstrapThread", daemon=True
        ).start()

    def start_decode_thread(self):
        def decode_thread():
            # Fail-fast circuit breaker (aligned with the glm/community
            # branches, luno-3962): see bootstrap_thread for the rationale.
            consecutive_failures = 0
            MAX_CONSECUTIVE_FAILURES = 10
            try:
                while True:
                    msg = self.server_socket.recv_multipart()
                    try:
                        if msg[0] == MooncakeKVManager.AUX_DATA_HEADER:
                            self._handle_aux_data(msg)
                        # Staging: prefill notifies a chunk written to staging buffer
                        elif msg[0] == b"CHUNK_READY":
                            room = int(msg[1].decode("ascii"))
                            chunk_idx = int(msg[2].decode("ascii"))
                            page_start = int(msg[3].decode("ascii"))
                            num_pages = int(msg[4].decode("ascii"))
                            session_id = msg[5].decode("ascii")
                            handler = self._staging_handler
                            assert (
                                handler is not None
                            ), "CHUNK_READY received before staging handler initialized"
                            handler.handle_chunk_arrived(
                                room,
                                chunk_idx,
                                page_start,
                                num_pages,
                                session_id,
                                self._chunk_writer_counts,
                            )
                        # Staging: prefill pre-requests staging allocation before forward
                        elif msg[0] == b"STAGING_REQ":
                            self._handle_staging_req(msg)
                        # Prefill acknowledges abort notification
                        elif msg[0] == b"ABORT_ACK":
                            # TODO(shangming): use this info to implement the deferred release mechanism if needed
                            ack_aborted_room = int(msg[1].decode("ascii"))
                            logger.debug(f"Received ABORT_ACK for room {ack_aborted_room}")
                        else:
                            bootstrap_room, status, prefill_rank = msg
                            status = int(status.decode("ascii"))
                            bootstrap_room = int(bootstrap_room.decode("ascii"))
                            prefill_rank = int(prefill_rank.decode("ascii"))

                            if status == KVPoll.Success:
                                if bootstrap_room in self.request_status:
                                    self.prefill_response_tracker[bootstrap_room].add(prefill_rank)
                                    expected_response_num = (
                                        self.required_prefill_response_num_table[bootstrap_room]
                                    )
                                    arrived_response_num = len(
                                        self.prefill_response_tracker[bootstrap_room]
                                    )
                                    if arrived_response_num == expected_response_num:
                                        if self.enable_staging:
                                            handler = self._staging_handler
                                            if handler.is_staging_room(bootstrap_room):
                                                handler.submit_last_scatter_async(bootstrap_room)
                                            self._chunk_writer_counts.pop(bootstrap_room, None)
                                        self.update_status(bootstrap_room, KVPoll.Success)
                            elif status == KVPoll.Failed:
                                self.record_failure(
                                    bootstrap_room,
                                    "Failed to get kvcache from prefill instance, it might be dead",
                                )
                                self.update_status(bootstrap_room, status)
                        consecutive_failures = 0
                    except Exception as e:
                        consecutive_failures += 1
                        logger.error(
                            f"Error processing decode message: {e}",
                            exc_info=True,
                        )
                        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                            logger.critical(
                                f"decode_thread: {consecutive_failures} consecutive "
                                f"failures, terminating process"
                            )
                            kill_process_tree(os.getpid())
                        continue
            except Exception as e:
                logger.critical(
                    f"decode_thread crashed with unrecoverable error: {e}",
                    exc_info=True,
                )
                kill_process_tree(os.getpid())

        def heartbeat_checker():
            while True:
                time.sleep(self.heartbeat_interval)
                with self.connection_lock:
                    addresses = list(self.prefill_info_table.keys())

                for bootstrap_addr in addresses:
                    session = None
                    try:
                        with self.session_pool_lock:
                            session = self.session_pool[bootstrap_addr]
                        response = session.get(
                            f"http://{bootstrap_addr}/health",
                            timeout=(2, 3),
                            headers={"Connection": "keep-alive"},
                        )
                        if response.status_code == 200:
                            self.heartbeat_failures[bootstrap_addr] = 0

                            current_rooms = self.addr_to_rooms_tracker[
                                bootstrap_addr
                            ].copy()

                            for bootstrap_room in current_rooms:
                                # Remove KVPoll.Success requests from the tracker
                                if bootstrap_room not in self.request_status:
                                    self.addr_to_rooms_tracker[bootstrap_addr].discard(
                                        bootstrap_room
                                    )
                        else:
                            logger.info(
                                f"Attempting to reconnect to {bootstrap_addr}..."
                            )
                            self.heartbeat_failures[bootstrap_addr] = (
                                self.heartbeat_failures.get(bootstrap_addr, 0) + 1
                            )
                            with self.session_pool_lock:
                                if bootstrap_addr in self.session_pool:
                                    del self.session_pool[bootstrap_addr]
                    except Exception as e:
                        logger.info(f"Attempting to reconnect to {bootstrap_addr}...")
                        self.heartbeat_failures[bootstrap_addr] = (
                            self.heartbeat_failures.get(bootstrap_addr, 0) + 1
                        )

                    if (
                        self.heartbeat_failures.get(bootstrap_addr, 0)
                        >= self.max_failures
                    ):
                        self._handle_node_failure(bootstrap_addr)
                        with self.session_pool_lock:
                            if bootstrap_addr in self.session_pool:
                                del self.session_pool[bootstrap_addr]

        threading.Thread(target=decode_thread).start()
        self._start_heartbeat_checker_thread()

    def add_transfer_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int32],
        index_slice: slice,
        is_last_chunk: bool,
        aux_index: Optional[int] = None,
        state_indices: Optional[List] = None,
        num_kv_tokens: Optional[int] = None,
        trace_ctx: Optional[Union[TraceReqContext, TraceNullContext]] = None,
    ):
        assert self.disaggregation_mode == DisaggregationMode.PREFILL
        assert (
            not is_last_chunk
            or aux_index is not None
            or self.allow_last_chunk_without_aux
        ), "the last chunk must carry aux unless this rank is not the aux owner"

        if (
            bootstrap_room not in self.request_status
            or self.check_status(bootstrap_room) == KVPoll.Failed
        ):
            logger.debug(
                "Request with bootstrap_room=%s already failed", bootstrap_room
            )
            return

        if bootstrap_room not in self.transfer_infos:
            # This means that the current rank is a dummy rank for this request,
            # and it has already been marked as success, so there is no need to
            # add further chunks into the transfer queue.
            return

        if trace_ctx is None:
            trace_ctx = TraceNullContext()

        if self.enable_target_parallel_transfer:
            handled = maybe_enqueue_target_parallel_chunks(
                self,
                bootstrap_room=bootstrap_room,
                kv_indices=kv_indices,
                index_slice=index_slice,
                is_last_chunk=is_last_chunk,
                aux_index=aux_index,
                state_indices=state_indices,
                num_kv_tokens=num_kv_tokens,
                trace_ctx=trace_ctx,
            )
            # Declined when the room has fewer than two real destinations: there is
            # no fan-out to parallelize, so it takes the whole-room path below.
            if handled:
                return

        self.transfer_queues[self._transfer_queue_shard_idx(bootstrap_room)].put(
            TransferKVChunk(
                room=bootstrap_room,
                prefill_kv_indices=kv_indices,
                index_slice=index_slice,
                is_last_chunk=is_last_chunk,
                prefill_aux_index=aux_index,
                state_indices=state_indices,
                num_kv_tokens=num_kv_tokens,
                trace_ctx=trace_ctx,
            )
        )

    def _transfer_queue_shard_idx(self, bootstrap_room: int) -> int:
        """Pick the transfer queue -- hence the worker thread -- for a whole-room chunk.

        Both keys depend only on values that are constant across a room's chunks,
        which the transfer path requires: only the last chunk concludes the room, so
        the earlier chunks must stay ordered ahead of it on one queue.

        Room sharding spreads this rank's concurrent rooms over the workers.
        ``hash(int)`` is ``mod (2**61 - 1)`` and ignores PYTHONHASHSEED, so the
        spread is uniform and reproducible across processes. The legacy key instead
        sums the destination rpc ports, which is identical for every room whenever
        the decode fleet is fixed, collapsing all rooms onto one worker. Its stated
        purpose -- early abort on a failed session -- does not depend on co-location:
        ``failed_sessions`` is shared under ``session_lock``, so any worker sees it.
        """
        if self.enable_room_queue_sharding:
            return hash(bootstrap_room) % len(self.transfer_queues)

        # NOTE(shangming): sharding according to the dst_infos to make sure
        # requests with the same dst_sessions will be added into the same
        # queue, which enables early abort with failed sessions.
        dst_infos = self.transfer_infos[bootstrap_room].keys()
        session_port_sum = sum(int(session.rsplit(":", 1)[1]) for session in dst_infos)
        return session_port_sum % len(self.transfer_queues)

    def get_session_id(self):
        return self.engine.get_session_id()

    def _on_heartbeat_success(self, bootstrap_addr: str):
        current_rooms = self.addr_to_rooms_tracker[bootstrap_addr].copy()
        for bootstrap_room in current_rooms:
            # Remove KVPoll.Success requests from the tracker
            if bootstrap_room not in self.request_status:
                self.addr_to_rooms_tracker[bootstrap_addr].discard(bootstrap_room)

    def _run_one_probe_pass(self) -> None:
        with self.session_lock:
            snapshot = list(self.failed_sessions)
        for session_id in snapshot:
            send_probe = getattr(self.engine, "send_probe", None)
            if send_probe is None:
                rc = -1
            else:
                try:
                    rc = send_probe(session_id)
                except Exception as e:
                    logger.warning("send_probe(%s) raised: %s", session_id, e)
                    continue
            if rc == 0:
                with self.session_lock:
                    was_blacklisted = session_id in self.failed_sessions
                    self.failed_sessions.discard(session_id)
                    self.session_failures.pop(session_id, None)
                if was_blacklisted:
                    logger.info(
                        "Session %s recovered via probe; un-blacklisted",
                        session_id,
                    )
                    FAILED_SESSION_RECOVERIES.inc()
            else:
                logger.debug("Probe still failing for %s (rc=%d)", session_id, rc)

    def _failed_session_probe_loop(self) -> None:
        logger.info(
            "Starting failed-session probe loop (interval=%.1fs)",
            self.failed_session_probe_interval,
        )
        while not self._failed_session_probe_shutdown.wait(
            self.failed_session_probe_interval
        ):
            self._run_one_probe_pass()


class MooncakeKVSender(CommonKVSender):

    def __init__(
        self,
        mgr: MooncakeKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
        req_has_disagg_prefill_dp_rank: bool = False,
    ):
        super().__init__(
            mgr,
            bootstrap_addr,
            bootstrap_room,
            dest_tp_ranks,
            pp_rank,
            req_has_disagg_prefill_dp_rank,
        )
        self.conclude_state = None
        self.init_time = time.time()
        self._init_trace_ctx()

    @mooncake_trace_func(MooncakeRequestStage.MOONCAKE_SEND)
    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List] = None,
        num_kv_tokens: Optional[int] = None,
        send_aux: bool = True,
    ):
        kv_indices, index_slice, is_last_chunk, should_skip = (
            self._prepare_send_indices(kv_indices, state_indices)
        )
        if should_skip:
            return

        if not is_last_chunk:
            self.kv_mgr.add_transfer_request(
                self.bootstrap_room,
                kv_indices,
                index_slice,
                False,
                num_kv_tokens=num_kv_tokens,
                trace_ctx=self.trace_ctx.copy_for_thread(),
            )
        else:
            self.kv_mgr.add_transfer_request(
                self.bootstrap_room,
                kv_indices,
                index_slice,
                True,
                # None means "this rank does not own the aux buffer"; the chunk
                # still carries KV/state and still concludes the room.
                aux_index=self.aux_index if send_aux else None,
                state_indices=state_indices,
                num_kv_tokens=num_kv_tokens,
                trace_ctx=self.trace_ctx.copy_for_thread(),
            )
        self._record_transfer_indices(kv_indices, state_indices)

    def poll(self) -> KVPoll:
        if self.conclude_state is None:
            status = self.kv_mgr.check_status(self.bootstrap_room)
            if status in (KVPoll.Success, KVPoll.Failed):
                self.conclude_state = status
                self.trace_ctx.trace_req_finish()
            elif status == KVPoll.Bootstrapping:
                timeout_result = self._check_bootstrap_timeout()
                if timeout_result is not None:
                    return timeout_result

            return status
        else:
            return self.conclude_state

    def clear(self) -> None:
        """Remove the bootstrap room entry from the KV manager's request status."""
        # PP0's PP_READY ledger is keyed by room too; a concluded room never
        # needs it again, and a room that stalled below pp_size reports would
        # otherwise leak its entry.
        self.kv_mgr.clear_pp_ready(self.bootstrap_room)
        if self.bootstrap_room in self.kv_mgr.request_status:
            self.kv_mgr.request_status.pop(self.bootstrap_room)

            # self.kv_mgr.transfer_finish_times.pop(self.bootstrap_room, None)

    def failure_exception(self):
        # Explicitly set the status to failure since this request has failed in another rank
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        self.clear()

        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(self.bootstrap_room, None)
        is_propagated = failure_reason is None
        if is_propagated:
            failure_reason = "Failed due to an unknown reason from another rank"
        raise KVTransferError(
            self.bootstrap_room, failure_reason, is_from_another_rank=is_propagated
        )

    def _init_trace_ctx(self):
        if self.kv_mgr.enable_trace:
            self.trace_ctx = TraceReqContext(
                rid=str(hex(self.bootstrap_room)),
                bootstrap_room=self.bootstrap_room,
                role="Sender",
                module_name="mooncake",
            )
            if not self.trace_ctx.tracing_enable:
                self.trace_ctx = TraceNullContext()
        else:
            self.trace_ctx = TraceNullContext()

        self.trace_ctx.trace_req_start()

    def abort(self):
        super().abort()
        self.trace_ctx.abort(abort_info={"reason": "Aborted"})
        self.trace_ctx.trace_req_finish()


class MooncakeKVReceiver(CommonKVReceiver):
    def __init__(
        self,
        mgr: MooncakeKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ):
        self.session_id = mgr.get_session_id()
        self.init_time = None
        super().__init__(mgr, bootstrap_addr, bootstrap_room)

    def _register_kv_args(self) -> bool:
        for bootstrap_info in self.bootstrap_infos:
            packed_kv_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.kv_data_ptrs
            )
            packed_aux_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.aux_data_ptrs
            )
            packed_state_data_ptrs = pack_int_lists(
                self.kv_mgr.kv_args.state_data_ptrs, "Q"
            )
            packed_state_item_lens = pack_int_lists(
                self.kv_mgr.kv_args.state_item_lens, "I"
            )
            packed_state_dim_per_tensor = pack_int_lists(
                getattr(self.kv_mgr.kv_args, "state_dim_per_tensor", []) or [], "I"
            )
            packed_state_layer_ids = pack_int_lists(
                self.kv_mgr.kv_args.state_layer_ids, "I"
            )
            packed_kv_layer_ids = b"".join(
                struct.pack("I", layer_id)
                for layer_id in self.kv_mgr.kv_args.kv_layer_ids
            )
            # Note(shangming): No need to add pp rank here since decode pp size should be equal to prefill pp size or 1
            tp_rank = self.kv_mgr.kv_args.engine_rank
            kv_item_len = (
                self.kv_mgr.kv_args.kv_item_lens[0]
                if self.kv_mgr.kv_args.kv_item_lens
                else 0
            )
            dst_tp_rank = str(tp_rank).encode("ascii")
            dst_attn_tp_size = str(self.kv_mgr.attn_tp_size).encode("ascii")
            dst_kv_item_len = str(kv_item_len).encode("ascii")
            dst_dcp_size = str(self.kv_mgr.dcp_size).encode("ascii")
            dst_dcp_rank = str(self.kv_mgr.dcp_rank).encode("ascii")
            # Full per-entry KV item_lens list; used by DCP relayout to preserve
            # heterogeneous geometry (e.g. the head-sharded draft pool) that the
            # single dst_kv_item_len scalar cannot represent.
            packed_kv_item_lens = pack_int_lists(
                [list(self.kv_mgr.kv_args.kv_item_lens)], "I"
            )
            # Count of leading MLA (main) entries; the draft tail beyond it takes
            # the replicated head-slice DCP transfer.
            dst_num_mla_kv_entries = str(
                self.kv_mgr.kv_args.num_mla_kv_entries
            ).encode("ascii")
            if (
                self.kv_mgr.enable_staging
                and self.kv_mgr._staging_ctx.allocator is not None
            ):
                _alloc = self.kv_mgr._staging_ctx.allocator
                packed_staging_base_ptr = struct.pack("Q", _alloc.get_base_ptr())
                staging_total_size_str = str(_alloc.get_total_size()).encode("ascii")
            else:
                packed_staging_base_ptr = b""
                staging_total_size_str = b""

            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            try:
                with lock:
                    sock.send_multipart(
                        [
                            "None".encode("ascii"),
                            self.kv_mgr.local_ip.encode("ascii"),
                            str(self.kv_mgr.rank_port).encode("ascii"),
                            self.session_id.encode("ascii"),
                            packed_kv_data_ptrs,
                            packed_aux_data_ptrs,
                            packed_state_data_ptrs,
                            dst_tp_rank,
                            dst_attn_tp_size,
                            dst_kv_item_len,
                            packed_state_item_lens,
                            packed_state_dim_per_tensor,
                            packed_kv_layer_ids,
                            packed_state_layer_ids,
                            packed_staging_base_ptr,
                            staging_total_size_str,
                            dst_dcp_size,
                            dst_dcp_rank,
                            packed_kv_item_lens,
                            dst_num_mla_kv_entries,
                        ]
                    )
            except zmq.ZMQError:
                self.kv_mgr.record_failure(
                    self.bootstrap_room,
                    f"_register_kv_args to prefill {bootstrap_info.get('rank_ip')}:{bootstrap_info.get('rank_port')} failed",
                )
                self.conclude_state = KVPoll.Failed
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                return False
        return True

    def send_metadata(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List] = None,
        decode_prefix_len: Optional[int] = None,
        draft_dst_kv_indices: Optional[npt.NDArray[np.int32]] = None,
        draft_window_start: Optional[int] = None,
    ):
        self.draft_dst_kv_indices = draft_dst_kv_indices
        self.draft_window_start = draft_window_start
        if self.bootstrap_infos is None:
            self.kv_mgr.record_failure(
                self.bootstrap_room,
                f"Could not fetch prefill parallel info from bootstrap_addr: {self.bootstrap_addr}",
            )
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            return

        self.chunk_staging_infos = []
        if (
            self.kv_mgr.enable_staging
            and self.kv_mgr._staging_ctx.allocator is not None
        ):
            self.kv_mgr.register_staging_room_bootstrap(
                self.bootstrap_room, self.bootstrap_infos, self
            )

        for bootstrap_info in self.bootstrap_infos:
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            is_dummy = bootstrap_info["is_dummy"]
            try:
                with lock:
                    sock.send_multipart(
                        [
                            str(self.bootstrap_room).encode("ascii"),
                            self.kv_mgr.local_ip.encode("ascii"),
                            str(self.kv_mgr.rank_port).encode("ascii"),
                            self.session_id.encode("ascii"),
                            kv_indices.tobytes() if not is_dummy else b"",
                            str(aux_index).encode("ascii") if not is_dummy else b"",
                            (
                                pack_int_lists(state_indices, "i")
                                if not is_dummy and state_indices
                                else b""
                            ),
                            str(self.required_dst_info_num).encode("ascii"),
                            str(decode_prefix_len or 0).encode("ascii"),
                            (
                                draft_dst_kv_indices.tobytes()
                                if draft_dst_kv_indices is not None and not is_dummy
                                else b""
                            ),
                            str(draft_window_start).encode("ascii")
                            if draft_window_start is not None and not is_dummy
                            else b"",
                        ]
                    )
            except zmq.ZMQError:
                self.kv_mgr.record_failure(
                    self.bootstrap_room,
                    f"send_metadata to prefill {bootstrap_info.get('rank_ip')}:{bootstrap_info.get('rank_port')} failed",
                )
                self.conclude_state = KVPoll.Failed
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                return
        self.init_time = time.time()

    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state

        status = self.kv_mgr.check_status(self.bootstrap_room)
        if status in (KVPoll.Success, KVPoll.Failed):
            self.conclude_state = status
        elif status == KVPoll.WaitingForInput:
            timeout_result = self._check_waiting_timeout()
            if timeout_result is not None:
                return timeout_result

        return status

    def failure_exception(self):
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        self.clear()

        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(self.bootstrap_room, None)
        is_propagated = failure_reason is None
        if is_propagated:
            failure_reason = "Failed due to an unknown reason from another rank"
        raise KVTransferError(
            self.bootstrap_room, failure_reason, is_from_another_rank=is_propagated
        )


class MooncakeKVBootstrapServer(CommonKVBootstrapServer):
    pass
