"""KV-balanced MLA decode for the tokenspeed_mla backend under DCP.

One ``KvBalancedMla`` per decode shape (gathered heads, q tokens per request):
the target verify and the regular decode. The split-KV work list is planned by
the first MLA layer of every forward and reused by the later layers. A forward
starts when the layer id does not increase, so eager runs and CUDA graph
captures both plan exactly once, and a replay re-plans from the captured
``seq_lens`` buffer.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from . import core

logger = logging.getLogger(__name__)


class KvBalancedMlaDecoder:
    """KV-balanced MLA decode for one rank's DCP decode and verify launches."""
    def __init__(self, device: torch.device, page_size: int, cp_world: int, cp_rank: int):
        self.device = device
        self.page_size = page_size
        # sglang's DCP owner rule is pos % cp_world == cp_rank (interleave 1).
        self.cp = (cp_world, cp_rank, 1) if cp_world > 1 else (1, 0, 1)
        self.arch = torch.cuda.get_device_capability(device)
        self.clusters = core.max_active_clusters(device)
        self.max_requests = 0
        self.shapes: set[tuple[int, int]] = set()
        # (heads, q) -> (KvBalancedMla, counts, work)
        self.cores: dict[tuple[int, int], tuple] = {}
        self.workspace: Optional[torch.Tensor] = None
        self._last: Optional[tuple[tuple[int, int], int]] = None
        self._warned: set[str] = set()

    def reserve(self, max_requests: int, shapes) -> None:
        """Size the buffers for ``max_requests`` and compile ``shapes``. Must run
        outside CUDA graph capture."""
        shapes = {
            (h, q) for h, q in shapes if core.supports(h, q, self.device)
        } | self.shapes
        if max_requests <= self.max_requests and shapes <= set(self.cores):
            return
        self.max_requests = max(self.max_requests, max_requests)
        self.shapes = shapes
        self.cores = {}
        self._last = None
        ws_bytes = 0
        for heads, q in sorted(shapes):
            limit = core.max_planned_requests(heads, q)
            if self.max_requests > limit:
                logger.warning(
                    "KV-balanced MLA: %d requests exceed the plan limit %d at "
                    "heads=%d q=%d; this shape stays on tokenspeed_mla",
                    self.max_requests, limit, heads, q,
                )
                continue
            mla = core.KvBalancedMla(heads, q, self.max_requests, True, self.device)
            counts = torch.zeros(self.max_requests, dtype=torch.int32, device=self.device)
            work = torch.full((mla.list_len,), -1, dtype=torch.int32, device=self.device)
            core._compile(
                heads, q, self.page_size, self.arch, self.clusters, True, True, self.cp
            )
            self.cores[(heads, q)] = (mla, counts, work)
            ws_bytes = max(ws_bytes, mla.workspace_bytes)
        if self.workspace is None or self.workspace.numel() < ws_bytes:
            self.workspace = torch.empty(ws_bytes, dtype=torch.int8, device=self.device)
        logger.info(
            "KV-balanced MLA decode: shapes(heads, q)=%s max_requests=%d "
            "workspace=%.0f MiB cp=%s",
            sorted(self.cores), self.max_requests, ws_bytes / 2**20, self.cp,
        )

    def _skip(self, reason: str) -> None:
        if reason not in self._warned:
            self._warned.add(reason)
            logger.warning("KV-balanced MLA decode falls back to tokenspeed_mla: %s", reason)

    def decode(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        causal_seqs: Optional[torch.Tensor],
        softmax_scale: float,
        output_scale: float,
        layer_id: int,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """(out [T, H, 512], log2 LSE [B, q, H]) or None to fall back."""
        if query.dim() != 4 or query.dtype != torch.float8_e4m3fn:
            self._skip(f"query {tuple(query.shape)} {query.dtype}")
            return None
        requests, q, heads = query.shape[0], query.shape[1], query.shape[2]
        key = (heads, q)
        if key not in self.cores:
            if torch.cuda.is_current_stream_capturing():
                self._skip(f"shape heads={heads} q={q} not reserved before capture")
                return None
            self.reserve(self.max_requests, [key])
            if key not in self.cores:
                self._skip(f"shape heads={heads} q={q} unsupported")
                return None
        if requests != seq_lens.numel() or requests > self.max_requests:
            self._skip(f"{requests} requests (max {self.max_requests})")
            return None
        mla, counts, work = self.cores[key]
        seq_lens = seq_lens.to(torch.int32).contiguous()
        if self._last is None or self._last[0] != key or layer_id <= self._last[1]:
            mla.plan(seq_lens, counts, work)
        self._last = (key, layer_id)

        lse = torch.empty((requests, q, heads), dtype=torch.float32, device=query.device)
        cache = kv_cache.view(-1, self.page_size, kv_cache.shape[-1])
        out = mla.decode(
            query.reshape(requests * q, heads, query.shape[-1]),
            cache,
            block_tables.contiguous(),
            seq_lens,
            softmax_scale,
            output_scale,
            counts,
            work,
            self.workspace,
            lse=lse,
            causal_lens=(
                causal_seqs.to(torch.int32).contiguous() if self.cp[0] > 1 else None
            ),
            cp=self.cp,
        )
        return out, lse
