"""Trace manager for collecting KV page digests and request metadata."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import torch

from sglang.srt.managers.io_struct import TraceType
from sglang.srt.managers.kv_trace_collector import KVTraceCollectorRegistry

if TYPE_CHECKING:
    from sglang.srt.managers.io_struct import TraceReq
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.memory_pool import KVCache
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

logger = logging.getLogger(__name__)

# Upper bound on how long an online AS trace holds back the request's finish
# output while waiting for its AS writes to retire.
_DEFAULT_ONLINE_AS_TRACE_TIMEOUT_S = 10.0


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
        disaggregation_mode: str = "null",
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
        self.disaggregation_mode = disaggregation_mode
        # Streams a finish output that was held back for an online AS trace.
        self.on_online_trace_ready: Optional[Callable[["Req"], None]] = None
        # rid -> (req, AS trace context, monotonic deadline)
        self._deferred_online: Dict[str, Tuple["Req", Any, float]] = {}

    def _rank_metadata(self) -> Dict[str, Any]:
        return {
            "tp_rank": self.tp_rank,
            "tp_size": self.tp_size,
            "pp_rank": self.pp_rank,
            "pp_size": self.pp_size,
            "dp_rank": self.dp_rank,
            "dp_size": self.dp_size,
            "disaggregation_mode": self.disaggregation_mode,
        }

    def _trace_stem(self, rid: str, suffix: Optional[str] = None) -> str:
        stem = f"{rid}.tp{self.tp_rank}.pp{self.pp_rank}.dp{self.dp_rank}"
        return f"{stem}.{suffix}" if suffix else stem

    def log_trace_request(self, req: "Req") -> None:
        """Log that the engine received a request carrying trace parameters."""
        trace_req = getattr(req, "to_trace_req", None)
        if trace_req is None:
            return
        logger.info(
            "TraceManager: trace requested rid=%s trace_type=%s "
            "online=%s",
            trace_req.rid,
            trace_req.trace_type,
            trace_req.return_online,
        )

    def _set_online_trace(
        self, recv_req: "TraceReq", req: "Req", payload: Dict[str, Any]
    ) -> None:
        """Attach the final trace payload for online return and log it."""
        req.trace_info = payload
        logger.info(
            "TraceManager: trace ready for online return rid=%s "
            "trace_type=%s",
            recv_req.rid,
            recv_req.trace_type,
        )

    def _should_trace(self, req: "Req") -> bool:
        """Check whether the trace config matches this P/D instance."""
        config = getattr(req, "trace_config", None) or {}
        target = config.get("pd", "all")
        if target not in ("p", "d", "all"):
            logger.warning(
                "TraceManager: invalid trace_config.pd=%r; using all", target
            )
            target = "all"
        if target == "all" or self.disaggregation_mode == "null":
            return True
        role = {"prefill": "p", "decode": "d"}.get(self.disaggregation_mode)
        return role == target

    @staticmethod
    def _request_cache_metadata(req: "Req") -> Dict[str, int]:
        return {
            "prefix_len": int(req.prefix_indices.numel()),
            "origin_input_len": len(req.origin_input_ids),
            "cached_tokens": int(getattr(req, "cached_tokens", 0) or 0),
            "cached_tokens_device": int(
                getattr(req, "cached_tokens_device", 0) or 0
            ),
            "cached_tokens_host": int(getattr(req, "cached_tokens_host", 0) or 0),
            "cached_tokens_storage": int(
                getattr(req, "cached_tokens_storage", 0) or 0
            ),
        }

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
            "unique_trace_id": recv_req.unique_trace_id,
            "timestamp": time.time(),
            **self._rank_metadata(),
            "req_pool_idx": req.kv.req_pool_idx,
            **self._request_cache_metadata(req),
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
        """Collect page-level KV MD5s and persist JSON metadata for ``rid``."""
        rid = recv_req.rid
        if not rid:
            logger.warning("TraceManager.trace_kv: empty rid, skipped")
            return

        token_indices = self._get_token_indices(rid, req, self.req_to_token_pool)
        total_len = 0 if token_indices is None else len(token_indices)
        kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
        layer_ids = self._get_kv_trace_layers(req, kv_cache)
        kv_match_md5 = self._collect_kv(
            rid,
            token_indices,
            kv_cache,
            self._get_model_identity(req, recv_req),
            layer_ids,
        )
        self._persist(recv_req, req, token_indices, kv_match_md5, total_len)

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
            self._set_online_trace(recv_req, req, meta)
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
            "unique_trace_id": recv_req.unique_trace_id,
            "trace_type": trace_type,
            "timestamp": time.time(),
            **self._rank_metadata(),
            "payload": payload,
        }
        if recv_req.return_online:
            self._set_online_trace(recv_req, req, meta)
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
        """Dump all trace information into one JSON document."""
        rid = recv_req.rid
        if not rid:
            logger.warning("TraceManager.trace_all: empty rid, skipped")
            return

        token_indices = self._get_token_indices(rid, req, self.req_to_token_pool)
        total_len = 0 if token_indices is None else len(token_indices)
        kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
        layer_ids = self._get_kv_trace_layers(req, kv_cache)
        kv_match_md5 = self._collect_kv(
            rid,
            token_indices,
            kv_cache,
            self._get_model_identity(req, recv_req),
            layer_ids,
        )

        meta = self._build_base_metadata(recv_req, req, token_indices, total_len)
        meta.update(
            {
                "trace_type": "all",
                "kv_match_md5": kv_match_md5,
                "sampling_params": self._serialize_payload(req.sampling_params),
            }
        )
        self._set_kv_layer_metadata(meta, kv_match_md5)

        logprobs_data = self._collect_logprobs(req)
        if logprobs_data:
            meta["logprobs"] = logprobs_data
        payload = self._serialize_payload(recv_req.payload)
        if payload:
            meta["payload"] = payload

        def persist(events, pending_writes=0):
            final_meta = {
                **meta,
                "as_cache": self._as_cache_payload(events, pending_writes),
            }
            if recv_req.return_online:
                self._set_online_trace(recv_req, req, final_meta)
                self._release_online(req)
                return
            dump_dir = recv_req.dump_dir
            try:
                os.makedirs(dump_dir, exist_ok=True)
                stem = self._trace_stem(rid, recv_req.trace_type)
                self._dump_json(dump_dir, stem, final_meta)
                logger.info(
                    f"TraceManager: all dumped rid={rid} "
                    f"tp={self.tp_rank}/{self.tp_size} "
                    f"pp={self.pp_rank}/{self.pp_size} "
                    f"dp={self.dp_rank}/{self.dp_size} "
                    f"tokens={total_len} → {dump_dir}/{stem}.json"
                )
            except Exception:
                logger.exception(f"TraceManager: all persist failed rid={rid}")

        ctx = getattr(req, "as_trace_context", None)
        if ctx is None:
            persist([])
        else:
            self._register_as_finalizer(recv_req, req, ctx, persist)

    @staticmethod
    def _as_cache_payload(events: list, pending_writes: int) -> Dict[str, Any]:
        return {
            "events": events,
            "as_write_status": "complete" if pending_writes == 0 else "timeout",
            "pending_as_writes": pending_writes,
        }

    @staticmethod
    def _online_as_trace_timeout(req: "Req") -> float:
        config = (getattr(req, "trace_config", None) or {}).get("as_cache")
        if isinstance(config, dict):
            try:
                return float(
                    config.get("online_timeout_s", _DEFAULT_ONLINE_AS_TRACE_TIMEOUT_S)
                )
            except (TypeError, ValueError):
                pass
        return _DEFAULT_ONLINE_AS_TRACE_TIMEOUT_S

    def _register_as_finalizer(
        self, recv_req: "TraceReq", req: "Req", ctx: Any, persist: Callable
    ) -> None:
        """Hold the online finish output until the AS trace finalizes."""
        if recv_req.return_online:
            req.defer_finish_output = True
            self._deferred_online[req.rid] = (
                req,
                ctx,
                time.monotonic() + self._online_as_trace_timeout(req),
            )
        # May finalize immediately when nothing is pending.
        ctx.set_finalizer(persist)

    def _release_online(self, req: "Req") -> None:
        """Let a held-back finish output go out now that trace_info is ready."""
        self._deferred_online.pop(req.rid, None)
        req.defer_finish_output = False
        if req.finish_output_deferred:
            # output_streamer already skipped this request; send it now.
            req.finish_output_deferred = False
            if self.on_online_trace_ready is not None:
                self.on_online_trace_ready(req)

    def check_deferred_online_traces(self) -> None:
        """Force-finalize online AS traces whose writes outlived the timeout."""
        if not self._deferred_online:
            return
        now = time.monotonic()
        for rid, (req, ctx, deadline) in list(self._deferred_online.items()):
            if now < deadline:
                continue
            logger.warning(
                "TraceManager: online AS trace timed out waiting for AS writes "
                "rid=%s; returning partial events",
                rid,
            )
            if not ctx.force_finalize():
                # Finalizer never registered or already ran; never hold output.
                self._release_online(req)

    def _get_token_indices(
        self,
        rid: str,
        target: "Req",
        req_to_token_pool: "ReqToTokenPool",
    ) -> Optional[torch.Tensor]:
        total_len = len(target.origin_input_ids)
        if target.kv.req_pool_idx is None or total_len <= 0:
            logger.warning(
                f"TraceManager: rid={rid} has no pool slot "
                f"(idx={target.kv.req_pool_idx}, total_len={total_len})"
            )
            return None
        try:
            return req_to_token_pool.req_to_token[
                target.kv.req_pool_idx, :total_len
            ].clone()
        except Exception:
            logger.exception(f"TraceManager: read token indices failed rid={rid}")
            return None

    def _get_model_identity(self, req: "Req", recv_req: Any = None) -> str:
        config = getattr(req, "trace_config", None) or {}
        if "model_identity" not in config:
            config = getattr(recv_req, "trace_config", None) or config
        return str(config.get("model_identity", ""))

    def _get_kv_trace_layers(
        self,
        req: "Req",
        kv_cache: "KVCache",
    ) -> list[int]:
        """Select KV-trace layers; default to global layer zero.

        The collector receives this resolved list and must not apply another
        layout-specific default. Explicit ``trace_config.kv.layers`` entries are
        retained only when they belong to the local PP cache range.
        """
        start_layer = int(getattr(kv_cache, "start_layer", 0))
        layer_num = int(getattr(kv_cache, "layer_num", self.num_hidden_layers))
        local_layers = range(start_layer, start_layer + layer_num)
        kv_config = (getattr(req, "trace_config", None) or {}).get("kv") or {}
        layers = kv_config.get("layers", [0]) if isinstance(kv_config, dict) else [0]
        if not isinstance(layers, list):
            layers = [0]
        return list(dict.fromkeys(layer for layer in layers if layer in local_layers))

    def _get_kv_layout(self, kv_cache: "KVCache") -> str:
        return KVTraceCollectorRegistry.detect_layout(kv_cache)

    def _empty_kv_match_md5(
        self, kv_cache: Optional["KVCache"], model_identity: str
    ) -> Dict[str, Any]:
        warnings: list[Dict[str, Any]] = []
        if kv_cache is None:
            warnings.append(
                {
                    "code": "missing_kv_cache",
                    "message": "KV cache is unavailable",
                }
            )
            page_size = 1
            layout = "unknown"
        else:
            page_size = KVTraceCollectorRegistry.get("mha_gqa").page_size(kv_cache, warnings)
            layout = self._get_kv_layout(kv_cache)
        return {
            "schema_version": 1,
            "algorithm": "md5",
            "coverage": "all_logical_token_pages",
            "hash_scope": "physical_page_row",
            "system_page_size": page_size,
            "layout": layout,
            "model_identity": model_identity,
            "total_tokens": 0,
            "logical_page_count": 0,
            "collection_status": "error" if kv_cache is None else "ok",
            "layers": [],
            "warnings": warnings,
        }

    @staticmethod
    def _page_md5(page: torch.Tensor) -> str:
        """Backward-compatible wrapper for the shared physical-byte digest."""
        return KVTraceCollectorRegistry.get("mha_gqa").page_md5(page)

    def _collect_kv(
        self,
        rid: str,
        token_indices: Optional[torch.Tensor],
        kv_cache: Optional["KVCache"],
        model_identity: str = "",
        layer_ids: Optional[list[int]] = None,
    ) -> Dict[str, Any]:
        """Return JSON-safe page MD5 metadata for the request KV cache."""
        result = self._empty_kv_match_md5(kv_cache, model_identity)
        if kv_cache is None:
            logger.warning("TraceManager: KV cache unavailable rid=%s", rid)
            return result
        if token_indices is None:
            result["collection_status"] = "error"
            result["warnings"].append(
                {
                    "code": "missing_token_indices",
                    "message": "request has no readable KV token slots",
                }
            )
            return result

        total_len = len(token_indices)
        page_size = result["system_page_size"]
        result["total_tokens"] = total_len
        result["logical_page_count"] = (total_len + page_size - 1) // page_size
        if total_len == 0:
            return result

        collector = KVTraceCollectorRegistry.get(result["layout"])
        if collector is None:
            result["warnings"].append(
                {
                    "code": "unsupported_kv_layout",
                    "layout": result["layout"],
                    "message": "no KV trace collector is registered for this layout",
                }
            )
        else:
            collector.collect(
                kv_cache,
                token_indices,
                result,
                layer_ids,
            )
        if result["warnings"]:
            result["collection_status"] = "partial"
        return result

    @staticmethod
    def _set_kv_layer_metadata(
        meta: Dict[str, Any], kv_match_md5: Dict[str, Any]
    ) -> None:
        layer_ids = [layer["layer_id"] for layer in kv_match_md5["layers"]]
        if layer_ids:
            meta["start_layer"] = min(layer_ids)
            meta["layer_num"] = len(layer_ids)

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
            "unique_trace_id": recv_req.unique_trace_id,
            "trace_type": "logprobs",
            "timestamp": time.time(),
            **self._rank_metadata(),
        }
        if logprobs_data:
            meta["logprobs"] = logprobs_data
        if recv_req.return_online:
            self._set_online_trace(recv_req, req, meta)
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
        token_indices: Optional[torch.Tensor],
        kv_match_md5: Dict[str, Any],
        total_len: int,
    ) -> None:
        """Persist the KV trace as JSON; no tensor sidecar is produced."""
        rid = recv_req.rid
        meta = self._build_base_metadata(recv_req, target, token_indices, total_len)
        meta.update(
            {
                "trace_type": "kv",
                "kv_match_md5": kv_match_md5,
            }
        )
        self._set_kv_layer_metadata(meta, kv_match_md5)

        if recv_req.return_online:
            self._set_online_trace(recv_req, target, meta)
            return

        dump_dir = recv_req.dump_dir
        try:
            os.makedirs(dump_dir, exist_ok=True)
            stem = self._trace_stem(rid, recv_req.trace_type)
            self._dump_json(dump_dir, stem, meta)
            logger.info(
                f"TraceManager: KV trace dumped rid={rid} "
                f"tp={self.tp_rank}/{self.tp_size} "
                f"pp={self.pp_rank}/{self.pp_size} "
                f"dp={self.dp_rank}/{self.dp_size} "
                f"tokens={total_len} → {dump_dir}/{stem}.json"
            )
        except Exception:
            logger.exception(f"TraceManager: KV trace persist failed rid={rid}")

    def _finalize_as_trace(
        self, recv_req: "TraceReq", req: "Req"
    ) -> Optional[Dict[str, Any]]:
        """Close the AS trace session and return its event payload.
        """
        ctx = getattr(req, "as_trace_context", None)
        if ctx is None:
            return None
        ctx.close()
        return {"events": ctx.snapshot()}

    def _trace_as_cache(self, recv_req: "TraceReq", req: "Req") -> None:
        """Finalize an AS trace session as one regular JSON trace file."""
        ctx = getattr(req, "as_trace_context", None)
        if ctx is None:
            return

        static_payload = {
            "trace_type": "as_cache",
            "unique_trace_id": recv_req.unique_trace_id,
            "rid": recv_req.rid,
            "timestamp": time.time(),
            **self._rank_metadata(),
            **self._request_cache_metadata(req),
        }

        def persist(events, pending_writes=0):
            payload = {
                **static_payload,
                **self._as_cache_payload(events, pending_writes),
            }
            if recv_req.return_online:
                self._set_online_trace(recv_req, req, payload)
                self._release_online(req)
                return
            dump_dir = recv_req.dump_dir
            rid = recv_req.rid
            try:
                os.makedirs(dump_dir, exist_ok=True)
                stem = self._trace_stem(rid, recv_req.trace_type)
                self._dump_json(dump_dir, stem, payload)
                logger.info(
                    f"TraceManager: as_cache dumped rid={rid} "
                    f"unique={payload.get('unique_trace_id')} "
                    f"events={len(events)} → {dump_dir}/{stem}.json"
                )
            except Exception:
                logger.exception(f"TraceManager: as_cache persist failed rid={rid}")

        self._register_as_finalizer(recv_req, req, ctx, persist)

    def run_trace(
        self,
        recv_req: "TraceReq",
        req: "Req",
    ):
        """Dispatch the trace request to the corresponding trace handler based on trace type."""
        if not self._should_trace(req):
            logger.info(
                "TraceManager: trace skipped by pd filter=%s mode=%s rid=%s",
                (getattr(req, "trace_config", None) or {}).get("pd", "all"),
                self.disaggregation_mode,
                recv_req.rid,
            )
            return
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
        elif recv_req.trace_type == TraceType.AS_CACHE:
            self._trace_as_cache(recv_req=recv_req, req=req)
        else:
            logger.warning(
                f"trace_request: unsupported trace_type={recv_req.trace_type} rid={recv_req.rid}"
            )
