"""Layout-specific collectors for page-level KV trace digests.

Collectors translate a runtime KV-cache layout into the stable ``layers`` and
``sources`` portion of TraceManager's KV trace schema.  TraceManager owns
request lifecycle, result metadata, and persistence; this module owns buffer
layout and slot-mapping details.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import torch

from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    DSATokenToKVPool,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool


class KVTraceCollector:
    """Base collector and shared physical-page digest helpers."""

    layout = "unknown"

    @staticmethod
    def page_md5(page: torch.Tensor) -> str:
        """Return an MD5 of the physical bytes stored by one cache page."""
        raw = page.detach().contiguous().view(torch.uint8).to("cpu").numpy().tobytes()
        return hashlib.md5(raw).hexdigest()

    @staticmethod
    def page_size(pool: Any, warnings: list[Dict[str, Any]]) -> int:
        """Return a pool page size, recording a fallback warning if unavailable."""
        try:
            size = int(getattr(pool, "page_size"))
            if size > 0:
                return size
        except (AttributeError, TypeError, ValueError):
            pass
        warnings.append(
            {
                "code": "invalid_page_size",
                "pool": type(pool).__name__,
                "message": "pool.page_size is missing or not positive; using 1",
            }
        )
        return 1

    @staticmethod
    def group_slots_by_page(
        slots: torch.Tensor, page_size: int, min_valid_slot: int = 0
    ) -> Tuple["OrderedDict[int, list[int]]", int]:
        """Group slots by physical page in request order.

        SWA slot zero is reserved.  A translated value of zero therefore means
        the token has no SWA backing page rather than that it belongs to page 0.
        """
        pages: "OrderedDict[int, list[int]]" = OrderedDict()
        skipped = 0
        for token_pos, slot in enumerate(
            int(slot) for slot in slots.detach().to("cpu").reshape(-1).tolist()
        ):
            if slot < min_valid_slot:
                skipped += 1
                continue
            pages.setdefault(slot // page_size, []).append(token_pos)
        return pages, skipped

    def _collect_source_pages(
        self,
        buffer: torch.Tensor,
        slots: torch.Tensor,
        page_size: int,
        *,
        raw_page_rows: bool,
        warnings: list[Dict[str, Any]],
        source_name: str,
        layer_id: int,
        pool: str,
        min_valid_slot: int = 0,
    ) -> list[Dict[str, Any]]:
        pages, skipped = self.group_slots_by_page(slots, page_size, min_valid_slot)
        if skipped:
            warnings.append(
                {
                    "code": "unmapped_token_slot",
                    "layer_id": layer_id,
                    "pool": pool,
                    "source": source_name,
                    "skipped_tokens": skipped,
                    "min_valid_slot": min_valid_slot,
                    "message": "tokens without a valid physical slot were not hashed",
                }
            )

        result: list[Dict[str, Any]] = []
        for logical_page, (physical_page_id, positions) in enumerate(pages.items()):
            entry: Dict[str, Any] = {
                "logical_page": logical_page,
                "token_range": [min(positions), max(positions) + 1],
                "physical_page_ids": [physical_page_id],
                "source_token_count": len(positions),
                "md5": None,
            }
            try:
                if raw_page_rows:
                    if physical_page_id >= buffer.shape[0]:
                        raise IndexError(
                            f"page {physical_page_id} >= buffer rows {buffer.shape[0]}"
                        )
                    page = buffer[physical_page_id]
                else:
                    start = physical_page_id * page_size
                    end = min(start + page_size, buffer.shape[0])
                    if start >= end:
                        raise IndexError(
                            f"token page [{start}, {start + page_size}) exceeds "
                            f"buffer rows {buffer.shape[0]}"
                        )
                    page = buffer[start:end]
                    if end - start != page_size:
                        entry["truncated"] = True
                entry["md5"] = self.page_md5(page)
            except Exception as exc:
                warnings.append(
                    {
                        "code": "page_hash_failed",
                        "layer_id": layer_id,
                        "pool": pool,
                        "source": source_name,
                        "physical_page_id": physical_page_id,
                        "message": str(exc),
                    }
                )
            result.append(entry)
        return result

    def append_source(
        self,
        layer: Dict[str, Any],
        *,
        name: str,
        buffer: Optional[torch.Tensor],
        slots: torch.Tensor,
        page_size: int,
        layout: str,
        raw_page_rows: bool,
        warnings: list[Dict[str, Any]],
        min_valid_slot: int = 0,
    ) -> None:
        """Append one source and its page digests to a layer trace record."""
        if buffer is None:
            warnings.append(
                {
                    "code": "missing_buffer",
                    "layer_id": layer["layer_id"],
                    "pool": layer["pool"],
                    "source": name,
                    "message": "source buffer is unavailable",
                }
            )
            return
        layer["sources"].append(
            {
                "name": name,
                "layout": layout,
                "page_size": page_size,
                "pages": self._collect_source_pages(
                    buffer,
                    slots,
                    page_size,
                    raw_page_rows=raw_page_rows,
                    warnings=warnings,
                    source_name=name,
                    layer_id=layer["layer_id"],
                    pool=layer["pool"],
                    min_valid_slot=min_valid_slot,
                ),
            }
        )

    def collect(
        self,
        kv_cache: Any,
        token_indices: torch.Tensor,
        result: Dict[str, Any],
        layer_ids: list[int],
    ) -> None:
        """Collect layout-specific source pages for the requested layer IDs."""
        raise NotImplementedError


class MHAKVTraceCollector(KVTraceCollector):
    """Collect separate token-indexed K and V pages for MHA/GQA."""

    layout = "mha_gqa"

    def collect(self, kv_cache, token_indices, result, layer_ids):
        warnings = result["warnings"]
        for layer_id in layer_ids:
            try:
                page_size = self.page_size(kv_cache, warnings)
                key, value = kv_cache.get_kv_buffer(layer_id)
                layer = {
                    "layer_id": layer_id,
                    "pool": "full",
                    "local_layer_id": layer_id,
                    "sources": [],
                }
                self.append_source(
                    layer,
                    name="k",
                    buffer=key,
                    slots=token_indices,
                    page_size=page_size,
                    layout="token_indexed",
                    raw_page_rows=False,
                    warnings=warnings,
                )
                self.append_source(
                    layer,
                    name="v",
                    buffer=value,
                    slots=token_indices,
                    page_size=page_size,
                    layout="token_indexed",
                    raw_page_rows=False,
                    warnings=warnings,
                )
                result["layers"].append(layer)
            except Exception as exc:
                warnings.append(
                    {
                        "code": "layer_collect_failed",
                        "layer_id": layer_id,
                        "message": str(exc),
                    }
                )


class MLAKVTraceCollector(KVTraceCollector):
    """Collect the one token-indexed compressed KV buffer used by MLA."""

    layout = "mla"

    def collect(self, kv_cache, token_indices, result, layer_ids):
        warnings = result["warnings"]
        for layer_id in layer_ids:
            try:
                page_size = self.page_size(kv_cache, warnings)
                key, _ = kv_cache.get_kv_buffer(layer_id)
                layer = {
                    "layer_id": layer_id,
                    "pool": "full",
                    "local_layer_id": layer_id,
                    "sources": [],
                }
                self.append_source(
                    layer,
                    name="k",
                    buffer=key,
                    slots=token_indices,
                    page_size=page_size,
                    layout="token_indexed",
                    raw_page_rows=False,
                    warnings=warnings,
                )
                result["layers"].append(layer)
            except Exception as exc:
                warnings.append(
                    {
                        "code": "layer_collect_failed",
                        "layer_id": layer_id,
                        "message": str(exc),
                    }
                )


class DSAKVTraceCollector(MLAKVTraceCollector):
    """Collect MLA KV plus DSA's independent page-indexed indexer cache."""

    layout = "dsa"

    def collect(self, kv_cache, token_indices, result, layer_ids):
        warnings = result["warnings"]
        for layer_id in layer_ids:
            try:
                page_size = self.page_size(kv_cache, warnings)
                key, _ = kv_cache.get_kv_buffer(layer_id)
                layer = {
                    "layer_id": layer_id,
                    "pool": "full",
                    "local_layer_id": layer_id,
                    "sources": [],
                }
                self.append_source(
                    layer,
                    name="k",
                    buffer=key,
                    slots=token_indices,
                    page_size=page_size,
                    layout="token_indexed",
                    raw_page_rows=False,
                    warnings=warnings,
                )
                self.append_source(
                    layer,
                    name="indexer_k",
                    buffer=kv_cache.get_index_k_with_scale_buffer(layer_id),
                    slots=token_indices,
                    page_size=page_size,
                    layout="raw_paged",
                    raw_page_rows=True,
                    warnings=warnings,
                )
                result["layers"].append(layer)
            except Exception as exc:
                warnings.append(
                    {
                        "code": "layer_collect_failed",
                        "layer_id": layer_id,
                        "message": str(exc),
                    }
                )


class SWAKVTraceCollector(KVTraceCollector):
    """Collect MHA K/V from either full-attention or SWA sub-pools."""

    layout = "swa"

    def collect(self, kv_cache, token_indices, result, layer_ids):
        warnings = result["warnings"]
        for layer_id in layer_ids:
            try:
                local_layer_id, is_swa_layer = kv_cache.layers_mapping[layer_id]
                pool = kv_cache.swa_kv_pool if is_swa_layer else kv_cache.full_kv_pool
                slots = (
                    kv_cache.translate_loc_from_full_to_swa(token_indices)
                    if is_swa_layer
                    else token_indices
                )
                page_size = self.page_size(pool, warnings)
                key, value = pool.get_kv_buffer(local_layer_id)
                layer = {
                    "layer_id": layer_id,
                    "pool": "swa" if is_swa_layer else "full",
                    "local_layer_id": local_layer_id,
                    "sources": [],
                }
                for name, buffer in (("k", key), ("v", value)):
                    self.append_source(
                        layer,
                        name=name,
                        buffer=buffer,
                        slots=slots,
                        page_size=page_size,
                        layout="token_indexed",
                        raw_page_rows=False,
                        warnings=warnings,
                        min_valid_slot=1 if is_swa_layer else 0,
                    )
                result["layers"].append(layer)
            except Exception as exc:
                warnings.append(
                    {
                        "code": "layer_collect_failed",
                        "layer_id": layer_id,
                        "message": str(exc),
                    }
                )


class UnsupportedDeepSeekV4KVTraceCollector(KVTraceCollector):
    """Fail explicitly until DeepSeekV4 publishes a stable layer-map API."""

    layout = "deepseek_v4"

    def collect(self, kv_cache, token_indices, result, layer_ids):
        result["warnings"].append(
            {
                "code": "unsupported_kv_layout",
                "layout": self.layout,
                "message": (
                    "DeepSeekV4 KV trace requires a public global-layer to "
                    "SWA/compressed-pool mapping and is not collected yet"
                ),
            }
        )


class KVTraceCollectorRegistry:
    """Resolve collectors without exposing layout-specific branches to TraceManager."""

    _collectors = {
        "mha_gqa": MHAKVTraceCollector,
        "mla": MLAKVTraceCollector,
        "dsa": DSAKVTraceCollector,
        "swa": SWAKVTraceCollector,
        "deepseek_v4": UnsupportedDeepSeekV4KVTraceCollector,
    }

    @classmethod
    def detect_layout(cls, kv_cache: Any) -> str:
        """Resolve production cache layouts by their concrete pool type.
        """
        if isinstance(kv_cache, SWAKVPool):
            return "swa"
        if isinstance(kv_cache, DSATokenToKVPool):
            return "dsa"
        if isinstance(kv_cache, MLATokenToKVPool):
            return "mla"
        if isinstance(kv_cache, MHATokenToKVPool):
            return "mha_gqa"

        declared_layout = getattr(kv_cache, "kv_trace_layout", None)
        if declared_layout in cls._collectors:
            return declared_layout

        # DeepSeek V4 has no stable public cache class in this repository yet.
        if hasattr(kv_cache, "c4_kv_pool") and hasattr(kv_cache, "layer_mapping"):
            return "deepseek_v4"
        return "unknown"

    @classmethod
    def get(cls, layout: str) -> Optional[KVTraceCollector]:
        """Create the registered collector for a resolved layout, if supported."""
        collector_type = cls._collectors.get(layout)
        return collector_type() if collector_type is not None else None
