"""Trace manager for dumping KV cache tensors and request metadata."""

from __future__ import annotations

import json
import logging
import os
import time

import torch
from typing import TYPE_CHECKING, Any, Dict, Optional

from sglang.srt.managers.io_struct import TraceType

if TYPE_CHECKING:
    from sglang.srt.managers.io_struct import TraceReq
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.memory_pool import KVCache
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

logger = logging.getLogger(__name__)


class TraceManager:
    """Manages tracing of KV cache and request metadata for debugging."""

    def _serialize_payload(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: self._serialize_payload(item)
                for key, item in value.items()
                if key != "__req__"
            }
        if isinstance(value, list):
            return [self._serialize_payload(item) for item in value]
        if isinstance(value, set):
            return list(value)
        if hasattr(value, "__struct_fields__"):
            return {
                field: self._serialize_payload(getattr(value, field))
                for field in value.__struct_fields__
            }
        if hasattr(value, "model_dump"):
            return self._serialize_payload(value.model_dump(mode="json"))
        return value

    def __init__(
        self,
        tp_rank: int,
        tp_size: int,
        pp_rank: int,
        pp_size: int,
        num_hidden_layers: int,
        dp_rank: int = 0,
        dp_size: int = 1,
        req_to_token_pool: "ReqToTokenPool" = None,
        token_to_kv_pool_allocator: Any = None,
        tokenizer: Any = None,
    ) -> None:
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.num_hidden_layers = num_hidden_layers
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.tokenizer = tokenizer

    def _rank_metadata(self) -> Dict[str, int]:
        return {
            "tp_rank": self.tp_rank,
            "tp_size": self.tp_size,
            "pp_rank": self.pp_rank,
            "pp_size": self.pp_size,
            "dp_rank": self.dp_rank,
            "dp_size": self.dp_size,
        }

    def _trace_stem(self, rid: str, suffix: Optional[str] = None) -> str:
        stem = f"{rid}.tp{self.tp_rank}.pp{self.pp_rank}.dp{self.dp_rank}"
        return f"{stem}.{suffix}" if suffix else stem

    def _dump_json(self, dump_dir: str, stem: str, data: Dict[str, Any]) -> str:
        """Write *data* as JSON to ``{dump_dir}/{stem}.json``.

        Returns the full file path on success.
        """
        path = os.path.join(dump_dir, f"{stem}.json")
        with open(path, "w") as f:
            json.dump(data, f, ensure_ascii=False, default=str)
        return path

    def _build_base_metadata(
        self,
        recv_req: "TraceReq",
        req: "Req",
        token_indices: Optional[torch.Tensor],
        total_len: int,
    ) -> Dict[str, Any]:
        """Build base metadata dict shared across trace types."""
        meta: Dict[str, Any] = {
            "rid": recv_req.rid,
            "timestamp": time.time(),
            **self._rank_metadata(),
            "req_pool_idx": req.req_pool_idx,
            "prefix_len": int(req.prefix_indices.numel()),
            "origin_input_len": len(req.origin_input_ids),
            "origin_input_ids": list(req.origin_input_ids),
            "output_len": len(req.output_ids),
            "output_ids": list(req.output_ids),
            "total_len": total_len,
        }
        if token_indices is not None:
            meta["kv_token_indices"] = token_indices.tolist()
        return meta

    def _trace_kv(
        self,
        recv_req: "TraceReq",
        req: "Req",
    ) -> None:
        """Dump raw K/V tensors (safetensors) + JSON metadata for *recv_req.rid*."""
        rid = recv_req.rid
        if not rid:
            logger.warning("TraceManager.trace_kv: empty rid, skipped")
            return

        token_indices = self._get_token_indices(rid, req, self.req_to_token_pool)
        if token_indices is None:
            return

        if recv_req.return_online:
            meta = self._build_base_metadata(recv_req, req, token_indices, len(token_indices))
            meta["trace_type"] = "kv"
            req.trace_info = meta
            return

        tensors = self._collect_kv(
            rid,
            token_indices,
            self.token_to_kv_pool_allocator.get_kvcache(),
        )
        if tensors is None:
            return

        self._persist(recv_req, req, token_indices, tensors, len(token_indices))

    def _trace_metadata(
        self,
        recv_req: "TraceReq",
        req: "Req",
    ) -> None:
        """Dump request metadata only (no tensor copy) as a JSON file.

        Lightweight alternative to trace_kv when only token indices and
        request stats are needed.
        """
        rid = recv_req.rid
        if not rid:
            logger.warning("TraceManager.trace_metadata: empty rid, skipped")
            return

        token_indices = self._get_token_indices(rid, req, self.req_to_token_pool)
        total_len = 0 if token_indices is None else len(token_indices)

        meta = self._build_base_metadata(recv_req, req, token_indices, total_len)
        meta["trace_type"] = "metadata"
        if token_indices is None:
            meta["kv_token_indices"] = []

        if recv_req.return_online:
            req.trace_info = meta
            return

        dump_dir = recv_req.dump_dir
        try:
            os.makedirs(dump_dir, exist_ok=True)
            stem = self._trace_stem(rid, "meta")
            self._dump_json(dump_dir, stem, meta)
            logger.info(
                f"TraceManager: metadata dumped rid={rid} "
                f"tp={self.tp_rank}/{self.tp_size} "
                f"pp={self.pp_rank}/{self.pp_size} "
                f"dp={self.dp_rank}/{self.dp_size} → {dump_dir}/{stem}.json"
            )
        except Exception:
            logger.exception(f"TraceManager: metadata persist failed rid={rid}")

    def _trace_payload(
        self,
        recv_req: "TraceReq",
        req: "Req",
    ) -> None:
        """Dump trace payload data as a JSON file for the requested rid."""
        rid = recv_req.rid
        if not rid:
            logger.warning("TraceManager.trace_payload: empty rid, skipped")
            return

        trace_type = recv_req.trace_type
        dump_dir = recv_req.dump_dir
        payload = self._serialize_payload(recv_req.payload)
        if trace_type == "sample":
            payload = {"sampling_params": self._serialize_payload(req.sampling_params)}
        meta = {
            "rid": rid,
            "trace_type": trace_type,
            "timestamp": time.time(),
            **self._rank_metadata(),
            "payload": payload,
        }
        if recv_req.return_online:
            req.trace_info = meta
            return

        try:
            os.makedirs(dump_dir, exist_ok=True)
            stem = self._trace_stem(rid, trace_type)
            self._dump_json(dump_dir, stem, meta)
            logger.info(
                f"TraceManager: payload dumped rid={rid} "
                f"tp={self.tp_rank}/{self.tp_size} "
                f"pp={self.pp_rank}/{self.pp_size} "
                f"dp={self.dp_rank}/{self.dp_size} → {dump_dir}/{stem}.json"
            )
        except Exception:
            logger.exception(f"TraceManager: payload persist failed rid={rid}")

    def _trace_all(
        self,
        recv_req: "TraceReq",
        req: "Req",
    ) -> None:
        """Dump all trace information into a single JSON + safetensors.

        Combines KV cache tensors, request metadata, logprobs, and payload
        into one comprehensive dump.
        """
        rid = recv_req.rid
        if not rid:
            logger.warning("TraceManager.trace_all: empty rid, skipped")
            return

        token_indices = self._get_token_indices(rid, req, self.req_to_token_pool)
        total_len = 0 if token_indices is None else len(token_indices)

        # 1. KV tensors (skip in online mode)
        tensors = None
        if token_indices is not None and not recv_req.return_online:
            tensors = self._collect_kv(
                rid,
                token_indices,
                self.token_to_kv_pool_allocator.get_kvcache(),
            )

        # 2. Logprobs
        logprobs_data = self._collect_logprobs(req)

        # 3. Payload & sampling params
        payload = self._serialize_payload(recv_req.payload)
        sampling_params = self._serialize_payload(req.sampling_params)

        # 4. Comprehensive metadata
        meta = self._build_base_metadata(recv_req, req, token_indices, total_len)
        meta["trace_type"] = "all"

        if tensors is not None:
            layer_ids = sorted(
                int(k.split(".")[0].split("_")[1])
                for k in tensors
                if k.endswith(".k")
            )
            meta["start_layer"] = layer_ids[0] if layer_ids else 0
            meta["layer_num"] = len(layer_ids)

        if logprobs_data:
            meta["logprobs"] = logprobs_data
        if payload:
            meta["payload"] = payload
        meta["sampling_params"] = sampling_params

        # Set online trace_info (without large KV tensors)
        if recv_req.return_online:
            req.trace_info = meta
            return

        # 5. Persist to a single JSON + safetensors
        dump_dir = recv_req.dump_dir
        try:
            os.makedirs(dump_dir, exist_ok=True)
        except Exception:
            logger.exception(f"TraceManager: mkdir failed {dump_dir}")
            return

        stem = self._trace_stem(rid, recv_req.trace_type)
        try:
            self._dump_json(dump_dir, stem, meta)
            if tensors:
                from safetensors.torch import save_file

                save_file(tensors, os.path.join(dump_dir, f"{stem}.safetensors"))
            logger.info(
                f"TraceManager: all dumped rid={rid} "
                f"tp={self.tp_rank}/{self.tp_size} "
                f"pp={self.pp_rank}/{self.pp_size} "
                f"dp={self.dp_rank}/{self.dp_size} "
                f"tokens={total_len} → {dump_dir}/{stem}.*"
            )
        except Exception:
            logger.exception(f"TraceManager: all persist failed rid={rid}")

    def _get_token_indices(
        self,
        rid: str,
        target: "Req",
        req_to_token_pool: "ReqToTokenPool",
    ) -> Optional[torch.Tensor]:
        total_len = len(target.origin_input_ids) + len(target.output_ids)
        if target.req_pool_idx is None or total_len <= 0:
            logger.warning(
                f"TraceManager: rid={rid} has no pool slot "
                f"(idx={target.req_pool_idx}, total_len={total_len})"
            )
            return None
        try:
            return req_to_token_pool.req_to_token[
                target.req_pool_idx, :total_len
            ].clone()
        except Exception:
            logger.exception(f"TraceManager: read token indices failed rid={rid}")
            return None

    def _collect_kv(
        self,
        rid: str,
        token_indices: torch.Tensor,
        kv_cache: "KVCache",
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Return {layer_N.k: Tensor, layer_N.v: Tensor} slices on CPU."""
        start_layer = getattr(kv_cache, "start_layer", 0)
        layer_num = getattr(kv_cache, "layer_num", self.num_hidden_layers)

        tensors: Dict[str, torch.Tensor] = {}
        try:
            for layer_id in range(start_layer, start_layer + layer_num):
                k, v = kv_cache.get_kv_buffer(layer_id)
                tensors[f"layer_{layer_id}.k"] = (
                    k[token_indices].detach().to("cpu", copy=True)
                )
                if v is not None:
                    tensors[f"layer_{layer_id}.v"] = (
                        v[token_indices].detach().to("cpu", copy=True)
                    )
        except Exception:
            logger.exception(f"TraceManager: gather kv failed rid={rid}")
            return None

        return tensors

    def _batch_decode_tokens(self, token_ids):
        """Batch decode token IDs to text using the tokenizer."""
        if not token_ids:
            return {}
        if self.tokenizer is None:
            return {tid: str(tid) for tid in token_ids}
        texts = self.tokenizer.batch_decode([[tid] for tid in token_ids])
        return dict(zip(token_ids, texts))

    def _collect_logprobs(self, req: "Req") -> Dict[str, Any]:
        """Collect logprobs data from a Req object into a JSON-serializable dict."""
        data = {}
        id_to_text = {}
        logprob = req.logprob

        if logprob.output_token_logprobs_val:
            ids = [int(idx) for idx in logprob.output_token_logprobs_idx]
            id_to_text.update(self._batch_decode_tokens(ids))
            data["output_token_logprobs"] = [
                {
                    "logprob": float(v),
                    "token_id": tid,
                    "text": id_to_text.get(tid, str(tid)),
                }
                for v, tid in zip(
                    logprob.output_token_logprobs_val, ids
                )
            ]

        if logprob.output_top_logprobs_val:
            all_ids = set()
            for idx_list in logprob.output_top_logprobs_idx:
                if idx_list is not None:
                    all_ids.update(int(i) for i in idx_list)
            id_to_text.update(self._batch_decode_tokens(list(all_ids)))
            top_k_list = []
            for val_list, idx_list in zip(
                logprob.output_top_logprobs_val, logprob.output_top_logprobs_idx
            ):
                if idx_list is None:
                    continue
                top_k = [
                    {
                        "logprob": float(v),
                        "token_id": tid,
                        "text": id_to_text.get(tid, str(tid)),
                    }
                    for v, tid in zip(val_list, [int(i) for i in idx_list])
                ]
                top_k_list.append(top_k)
            data["output_top_logprobs"] = top_k_list

        if logprob.input_token_logprobs_val:
            ids = [int(idx) for idx in logprob.input_token_logprobs_idx]
            id_to_text.update(self._batch_decode_tokens(ids))
            data["input_token_logprobs"] = [
                {
                    "logprob": float(v),
                    "token_id": tid,
                    "text": id_to_text.get(tid, str(tid)),
                }
                for v, tid in zip(
                    logprob.input_token_logprobs_val, ids
                )
                if v is not None
            ]

        if logprob.input_top_logprobs_val:
            all_ids = set()
            for idx_list in logprob.input_top_logprobs_idx:
                if idx_list is not None:
                    all_ids.update(int(i) for i in idx_list)
            id_to_text.update(self._batch_decode_tokens(list(all_ids)))
            top_k_list = []
            for val_list, idx_list in zip(
                logprob.input_top_logprobs_val, logprob.input_top_logprobs_idx
            ):
                if idx_list is None:
                    continue
                top_k = [
                    {
                        "logprob": float(v),
                        "token_id": tid,
                        "text": id_to_text.get(tid, str(tid)),
                    }
                    for v, tid in zip(val_list, [int(i) for i in idx_list])
                    if v is not None
                ]
                top_k_list.append(top_k)
            data["input_top_logprobs"] = top_k_list

        return data

    def _trace_logprobs(
        self,
        recv_req: "TraceReq",
        req: "Req",
    ) -> None:
        """Dump logprobs data as a JSON file.

        Lightweight trace that captures output/input token logprobs and
        top-k logprobs.
        """
        rid = recv_req.rid
        if not rid:
            logger.warning("TraceManager.trace_logprobs: empty rid, skipped")
            return

        dump_dir = recv_req.dump_dir
        logprobs_data = self._collect_logprobs(req)

        meta = {
            "rid": rid,
            "trace_type": "logprobs",
            "timestamp": time.time(),
            **self._rank_metadata(),
        }
        if logprobs_data:
            meta["logprobs"] = logprobs_data
        if recv_req.return_online:
            req.trace_info = meta
            return

        try:
            os.makedirs(dump_dir, exist_ok=True)
            stem = self._trace_stem(rid, "logprobs")
            self._dump_json(dump_dir, stem, meta)
            logger.info(
                f"TraceManager: logprobs dumped rid={rid} "
                f"tp={self.tp_rank}/{self.tp_size} "
                f"pp={self.pp_rank}/{self.pp_size} "
                f"dp={self.dp_rank}/{self.dp_size} → {dump_dir}/{stem}.json"
            )
        except Exception:
            logger.exception(f"TraceManager: logprobs persist failed rid={rid}")

    def _persist(
        self,
        recv_req: "TraceReq",
        target: "Req",
        token_indices: torch.Tensor,
        tensors: Dict[str, torch.Tensor],
        total_len: int,
    ) -> None:
        from safetensors.torch import save_file

        rid = recv_req.rid
        dump_dir = recv_req.dump_dir

        try:
            os.makedirs(dump_dir, exist_ok=True)
        except Exception:
            logger.exception(f"TraceManager: mkdir failed {dump_dir}")
            return

        layer_ids = sorted(
            int(k.split(".")[0].split("_")[1]) for k in tensors if k.endswith(".k")
        )
        start_layer = layer_ids[0] if layer_ids else 0
        layer_num = len(layer_ids)

        meta = self._build_base_metadata(recv_req, target, token_indices, total_len)
        meta["start_layer"] = start_layer
        meta["layer_num"] = layer_num

        if recv_req.return_online:
            target.trace_info = meta
            return

        stem = self._trace_stem(rid)
        try:
            self._dump_json(dump_dir, stem, meta)
            save_file(tensors, os.path.join(dump_dir, f"{stem}.safetensors"))
            logger.info(
                f"TraceManager: dumped rid={rid} "
                f"tp={self.tp_rank}/{self.tp_size} "
                f"pp={self.pp_rank}/{self.pp_size} "
                f"dp={self.dp_rank}/{self.dp_size} "
                f"layers=[{start_layer},{start_layer + layer_num}) "
                f"tokens={total_len} → {dump_dir}/{stem}.*"
            )
        except Exception:
            logger.exception(f"TraceManager: persist failed rid={rid}")

    def run_trace(
        self,
        recv_req: "TraceReq",
        req: "Req",
    ):
        """Dispatch the trace request to the corresponding trace handler based on trace type."""
        if recv_req.trace_type == TraceType.KV:
            self._trace_kv(recv_req=recv_req, req=req)
        elif recv_req.trace_type == TraceType.METADATA:
            self._trace_metadata(recv_req=recv_req, req=req)
        elif recv_req.trace_type == TraceType.LOGPROBS:
            self._trace_logprobs(recv_req=recv_req, req=req)
        elif recv_req.trace_type in (TraceType.PROMPT, TraceType.SAMPLE):
            self._trace_payload(recv_req=recv_req, req=req)
        elif recv_req.trace_type == TraceType.ALL:
            self._trace_all(recv_req=recv_req, req=req)
        else:
            logger.warning(
                f"trace_request: unsupported trace_type={recv_req.trace_type} rid={recv_req.rid}"
            )