# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

"""Attention backend for the tokenspeed-mla CuTe DSL kernels on Blackwell.

Subclasses :class:`TRTLLMMLABackend` to share its MLA data preparation and
prefill plumbing. Decode-context parallelism is implemented here because the
TokenSpeed decode kernel natively accepts CP rank/world metadata and returns
the partial log-sum-exp needed by the cross-rank merge.
"""

import logging
import os
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.kernels.ops.attention.dcp_kernels import (
    create_mla_kv_page_table_for_dcp,
)
from sglang.kernels.ops.attention.fixup_zero_kv import fixup_zero_kv_rows
from sglang.kernels.ops.attention.mla_kv_pack_quantize_fp8 import (
    mla_kv_pack_quantize_fp8,
)
from sglang.kernels.ops.attention.utils import (
    mla_quantize_and_rope_for_fp8,
    mla_quantize_without_rope_for_fp8,
)
from sglang.kernels.ops.kvcache.kv_indices import (
    get_num_kv_index_blocks_flashmla,
    get_num_page_per_block_flashmla,
)
from sglang.kernels.ops.quantization.fp8_quantize import fp8_quantize
from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
    TRTLLMMLAMultiStepDraftBackend,
)
from sglang.srt.layers.dcp.layout import get_dcp_lens
from sglang.srt.runtime_context import get_parallel
from sglang.srt.environ import envs
from sglang.srt.utils import is_flashinfer_available, is_tokenspeed_mla_available

if is_flashinfer_available():
    import flashinfer.rope as _flashinfer_rope

if is_tokenspeed_mla_available():
    import tokenspeed_mla

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA

logger = logging.getLogger(__name__)


# Workspace upper bound for tokenspeed_mla_decode:
#   num_sms * num_heads * max_q_len * (kv_lora_rank + 1) * sizeof(float32)
# MAX_Q_LEN=8 covers EAGLE3 num_draft_tokens=4 plus headroom. Larger
# speculative widths grow the bound at backend initialization.
_TOKENSPEED_MAX_Q_LEN = 8


def _get_tokenspeed_workspace(
    device: torch.device,
    num_heads: int,
    kv_lora_rank: int,
    max_q_len: int = _TOKENSPEED_MAX_Q_LEN,
) -> torch.Tensor:
    from sglang.srt.runtime_context import get_resources

    # DCP target verification gathers/project-replicates Q to the full head
    # count before launching TokenSpeed. Size for that launch shape rather than
    # the rank-local head count used by ordinary decode.
    num_heads *= get_parallel().attn_dcp_size
    max_q_len = max(max_q_len, _TOKENSPEED_MAX_Q_LEN)
    needed = (
        tokenspeed_mla.get_num_sm(device)
        * num_heads
        * max_q_len
        * (kv_lora_rank + 1)
        * 4
    )
    buffers = get_resources().buffers
    key = f"tokenspeed_mla_workspace:{device}"
    existing = buffers.get(key)
    if existing is None or existing.numel() < needed:
        buffers[key] = torch.empty(needed, dtype=torch.int8, device=device)
    return buffers[key]


# TODO(Qiaolin-Yu): Merge this attention backend into trtllm_mla_backend.py
# once the same CuteDSL kernels in flashinfer_trtllm are stable
# and there is no performance gap compared to this backend.
class TokenspeedMLABackend(TRTLLMMLABackend):
    """tokenspeed-mla CuTe DSL attention backend (Blackwell SM100, FP8 KV)."""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        q_indptr_decode_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            model_runner,
            skip_prefill,
            kv_indptr_buf,
            q_indptr_decode_buf,
        )

        if self.data_type != torch.float8_e4m3fn:
            raise ValueError(
                "tokenspeed_mla backend requires --kv-cache-dtype fp8_e4m3, "
                f"got data_type={self.data_type}."
            )
        if self.page_size not in (32, 64):
            raise ValueError(
                "tokenspeed_mla backend requires page_size in {32, 64}, "
                f"got page_size={self.page_size}."
            )

        self._tokenspeed_workspace: Optional[torch.Tensor] = None
        # KV-balanced decode for the DCP decode / target-verify launches of the
        # target model (draft backends never pass DCP arguments).
        self._kvb_decoder = None
        if (
            envs.SGLANG_K3_KV_BALANCED_MLA_DECODE.get()
            and get_parallel().dcp_enabled
            and not skip_prefill
        ):
            try:
                from sglang.kernels.ops.attention.kv_balanced_mla.decoder import (
                    KvBalancedMlaDecoder,
                )
            except (ImportError, AttributeError) as e:
                raise RuntimeError(
                    "SGLANG_K3_KV_BALANCED_MLA_DECODE needs nvidia-cutlass-dsl "
                    f">= 4.8 and apache-tvm-ffi: {e}"
                ) from e
            parallel = get_parallel()
            self._kvb_decoder = KvBalancedMlaDecoder(
                torch.device("cuda", torch.cuda.current_device()),
                self.page_size,
                parallel.dcp_size,
                parallel.dcp_rank,
            )
            heads = self.num_q_heads * parallel.attn_dcp_size
            self._kvb_shapes = [(heads, 1)] + (
                [(heads, self.num_draft_tokens)] if self.num_draft_tokens else []
            )
            self._kvb_decoder.reserve(model_runner.max_running_requests, self._kvb_shapes)
        if is_tokenspeed_mla_available():
            self._tokenspeed_workspace = _get_tokenspeed_workspace(
                self.device,
                self.num_q_heads,
                self.kv_lora_rank,
                max_q_len=(
                    model_runner.server_args.max_speculative_num_draft_tokens or 1
                ),
            )

            # Pre-JIT the prefill kernel variants. Each cute.compile takes 1-2
            # min; without warm-up the first request trips the 300 s scheduler
            # watchdog.
            _compile_prefill_kernel = tokenspeed_mla.mla_prefill._compile_prefill_kernel
            _compiled_kernels = tokenspeed_mla.mla_prefill._compiled_kernels
            head_dim_qk = self.qk_nope_head_dim + self.qk_rope_head_dim
            enable_ex2_emulation = tokenspeed_mla.mla_prefill._enable_ex2_emulation()
            use_pdl = is_arch_support_pdl()
            for is_causal in (True, False):
                for return_lse in (True, False):
                    # Non-causal is only entered from the chunked-prefix
                    # branch, which always asks for the LSE.
                    if is_causal is False and return_lse is False:
                        continue
                    # Runtime feeds fp8_e4m3fn q/k/v
                    config = (
                        torch.float8_e4m3fn,
                        head_dim_qk,
                        self.v_head_dim,
                        is_causal,
                        return_lse,
                        use_pdl,
                        enable_ex2_emulation,
                    )
                    if config in _compiled_kernels:
                        continue
                    _compiled_kernels[config] = _compile_prefill_kernel(
                        torch.float8_e4m3fn,
                        head_dim_qk,
                        self.v_head_dim,
                        is_causal,
                        return_lse,
                        use_pdl=use_pdl,
                        enable_ex2_emulation=enable_ex2_emulation,
                    )

    def _fused_rope_fp8_quantize(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
        is_neox: bool,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused RoPE + FP8 quantize that also packs nope+pe along the last
        dim, so FMHA consumes contig FP8 Q/K without an extra concat or cast.
        """
        num_heads = q_nope.shape[1]
        seq_len = q_nope.shape[0]
        q_fp8 = torch.empty(
            (seq_len, num_heads, qk_nope_head_dim + qk_rope_head_dim),
            dtype=torch.float8_e4m3fn,
            device=q_nope.device,
        )
        k_fp8 = torch.empty(
            (seq_len, num_heads, qk_nope_head_dim + qk_rope_head_dim),
            dtype=torch.float8_e4m3fn,
            device=k_nope.device,
        )
        if seq_len == 0:
            return q_fp8, k_fp8

        # Broadcast the shared latent k_pe across heads — RoPE is position-only
        # so per-head outputs are identical, and the cache write below reuses
        # head 0.
        if k_pe.dim() == 3 and k_pe.shape[1] == 1:
            k_pe_expanded = k_pe.expand(-1, num_heads, -1)
        else:
            k_pe_expanded = k_pe

        _flashinfer_rope.mla_rope_quantize_fp8(
            q_rope=q_pe,
            k_rope=k_pe_expanded,
            q_nope=q_nope,
            k_nope=k_nope,
            cos_sin_cache=cos_sin_cache,
            pos_ids=positions,
            is_neox=is_neox,
            quantize_dtype=torch.float8_e4m3fn,
            q_rope_out=q_fp8[..., qk_nope_head_dim:],
            k_rope_out=k_fp8[..., qk_nope_head_dim:],
            q_nope_out=q_fp8[..., :qk_nope_head_dim],
            k_nope_out=k_fp8[..., :qk_nope_head_dim],
            quant_scale_q=1.0,
            quant_scale_kv=1.0,
            enable_pdl=is_arch_support_pdl(),
        )
        return q_fp8, k_fp8

    def prepare_prefill_qkv(
        self,
        *,
        q: torch.Tensor,
        q_pe: torch.Tensor,
        kv_a: torch.Tensor,
        k_pe: torch.Tensor,
        positions: torch.Tensor,
        layer: DeepseekV2AttentionMLA,
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build FP8 (Q, K, V) for the FMHA kernel and write FP8 KV cache."""
        kv = layer.kv_b_proj(kv_a)[0]
        kv = kv.view(
            -1, layer.num_local_heads, layer.qk_nope_head_dim + layer.v_head_dim
        )
        k_nope = kv[..., : layer.qk_nope_head_dim]
        v_bf16 = kv[..., layer.qk_nope_head_dim :]
        q_nope = q[..., : layer.qk_nope_head_dim]

        q_fp8, k_fp8 = self._fused_rope_fp8_quantize(
            q_nope=q_nope,
            q_pe=q_pe,
            k_nope=k_nope,
            k_pe=k_pe,
            cos_sin_cache=layer.rotary_emb.cos_sin_cache,
            positions=positions,
            is_neox=getattr(layer.rotary_emb, "is_neox_style", True),
            qk_nope_head_dim=layer.qk_nope_head_dim,
            qk_rope_head_dim=layer.qk_rope_head_dim,
        )
        v_fp8 = fp8_quantize(v_bf16, enable_pdl=is_arch_support_pdl())

        # k_pe is shared across heads (RoPE is position-only), so head 0
        # reproduces the original [tokens, 1, qk_rope] latent layout.
        kv_a_fp8 = fp8_quantize(kv_a, enable_pdl=is_arch_support_pdl())
        k_pe_fp8 = k_fp8[:, 0:1, layer.qk_nope_head_dim :]
        self.token_to_kv_pool.set_mla_kv_buffer(
            layer.attn_mha,
            forward_batch.out_cache_loc,
            kv_a_fp8.unsqueeze(1),
            k_pe_fp8,
        )
        return q_fp8, k_fp8, v_fp8

    def pack_prefix_chunk_kv(
        self,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack strided ``k_nope``+``k_pe`` into contig FP8 K and quantize
        strided ``v`` into contig FP8 V in a single kernel.
        """
        return mla_kv_pack_quantize_fp8(
            k_nope, k_pe, v, enable_pdl=is_arch_support_pdl()
        )

    def _get_dcp_local_seq_lens(self, seq_lens: torch.Tensor) -> torch.Tensor:
        parallel = get_parallel()
        if not parallel.dcp_enabled:
            return seq_lens
        return get_dcp_lens(seq_lens, parallel.dcp_size, parallel.dcp_rank).to(
            torch.int32
        )

    def _get_dcp_local_max_seq_len(self, max_seq_len: int) -> int:
        parallel = get_parallel()
        if not parallel.dcp_enabled:
            return max_seq_len
        local_max = max_seq_len // parallel.dcp_size + int(
            parallel.dcp_rank < max_seq_len % parallel.dcp_size
        )
        # TokenSpeed requires a positive scheduling bound even when every
        # sequence in a padded graph row is empty on this rank.
        return max(local_max, 1)

    def _fill_dcp_block_kv_indices(
        self,
        block_kv_indices: torch.Tensor,
        req_pool_indices: torch.Tensor,
        local_seq_lens: torch.Tensor,
    ) -> None:
        parallel = get_parallel()
        pages_per_block = get_num_page_per_block_flashmla(self.page_size)
        create_mla_kv_page_table_for_dcp[
            (
                block_kv_indices.shape[0],
                get_num_kv_index_blocks_flashmla(
                    block_kv_indices.shape[1], self.page_size
                ),
            )
        ](
            self.req_to_token,
            req_pool_indices,
            local_seq_lens,
            block_kv_indices,
            self.req_to_token.stride(0),
            block_kv_indices.stride(0),
            PHYSICAL_PAGE_SIZE=self.page_size,
            DCP_SIZE=parallel.dcp_size,
            DCP_RANK=parallel.dcp_rank,
            PAGES_PER_BLOCK=pages_per_block,
        )

    def _create_block_kv_indices(
        self,
        batch_size: int,
        max_blocks: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if not get_parallel().dcp_enabled:
            return super()._create_block_kv_indices(
                batch_size,
                max_blocks,
                req_pool_indices,
                seq_lens,
                device,
            )

        block_kv_indices = torch.full(
            (batch_size, max_blocks), -1, dtype=torch.int32, device=device
        )
        self._fill_dcp_block_kv_indices(
            block_kv_indices,
            req_pool_indices,
            self._get_dcp_local_seq_lens(seq_lens),
        )
        return block_kv_indices

    def init_cuda_graph_state(self, max_bs: int, *args, **kwargs):
        """Also size the KV-balanced decode buffers for ``max_bs`` before capture."""
        super().init_cuda_graph_state(max_bs, *args, **kwargs)
        if self._kvb_decoder is not None:
            # Graph capture can't allocate or compile: size for the padded bs.
            self._kvb_decoder.reserve(max_bs, self._kvb_shapes)

    def _init_cuda_graph_metadata(
        self,
        bs: int,
        num_tokens: int,
        forward_mode,
        seq_lens: torch.Tensor,
        device: torch.device,
    ):
        super()._init_cuda_graph_metadata(
            bs, num_tokens, forward_mode, seq_lens, device
        )
        if get_parallel().dcp_enabled:
            self.forward_decode_metadata.max_seq_len_k = (
                self._get_dcp_local_max_seq_len(
                    self.max_context_len
                    + (self.num_draft_tokens if forward_mode.is_target_verify() else 0)
                )
            )

    def _apply_cuda_graph_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode,
    ):
        if not get_parallel().dcp_enabled:
            return super()._apply_cuda_graph_metadata(
                bs, req_pool_indices, seq_lens, forward_mode
            )

        metadata = self.decode_cuda_graph_metadata[bs]
        if forward_mode.is_target_verify():
            torch.add(
                seq_lens[:bs],
                self.num_draft_tokens,
                out=metadata.global_seq_lens_k,
            )
            metadata.seq_lens_k.copy_(
                self._get_dcp_local_seq_lens(metadata.global_seq_lens_k)
            )
            local_seq_lens = metadata.seq_lens_k
        elif forward_mode.is_draft_extend_v2():
            num_tokens_per_req = self.num_draft_tokens
            metadata.max_seq_len_q = num_tokens_per_req
            metadata.sum_seq_lens_q = num_tokens_per_req * bs
            seq_lens = seq_lens[:bs]
            metadata.seq_lens_k.copy_(seq_lens)
            local_seq_lens = self._get_dcp_local_seq_lens(seq_lens)
        else:
            seq_lens = seq_lens[:bs]
            local_seq_lens = self._get_dcp_local_seq_lens(seq_lens)

        self._fill_dcp_block_kv_indices(
            metadata.block_kv_indices,
            req_pool_indices[:bs],
            local_seq_lens,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        super().init_forward_metadata(forward_batch)
        if (
            get_parallel().dcp_enabled
            and self.forward_decode_metadata is not None
            and (
                forward_batch.forward_mode.is_decode_or_idle()
                or forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend_v2()
            )
        ):
            if forward_batch.forward_mode.is_target_verify():
                metadata = self.forward_decode_metadata
                metadata.global_seq_lens_k = metadata.seq_lens_k
                metadata.seq_lens_k = self._get_dcp_local_seq_lens(
                    metadata.global_seq_lens_k
                )
            self.forward_decode_metadata.max_seq_len_k = (
                self._get_dcp_local_max_seq_len(
                    self.forward_decode_metadata.max_seq_len_k
                )
            )

    def _flashmla_metadata_for_step(self, flash_mla, seq_lens, s_q, h_q):
        """``get_mla_metadata`` for the current decode step, computed once and
        shared by every MLA attention layer in the step.

        The split schedule only depends on ``seq_lens`` (identical for all MLA
        layers in one forward) plus ``s_q`` / ``h_q`` (fixed), so recomputing it
        per layer is pure redundancy. Cache ``(tile_scheduler_metadata,
        num_splits)`` on the per-forward decode-metadata object -- a fresh
        ``TRTLLMMLADecodeMetadata`` each eager step, and a stable per-bs object
        under CUDA graph. Under CUDA graph this Python cache is consulted only
        during capture, so the captured graph keeps a single get_mla_metadata
        kernel: its ``num_sm_parts`` shape is a hardware constant, and that one
        kernel re-reads the persistent ``seq_lens`` buffer on replay (verified
        bitwise-identical to eager in test_dense_fp8_cudagraph.py). If no
        metadata object is available, fall back to a per-call compute.
        """
        md = getattr(self, "forward_decode_metadata", None)
        key = (s_q, h_q)
        cached = getattr(md, "_fm_meta_cache", None) if md is not None else None
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        meta, num_splits = flash_mla.get_mla_metadata(
            seq_lens,
            s_q * h_q,  # num_q_tokens_per_head_k
            1,
            h_q,
            is_fp8_kvcache=True,
            topk=None,
            is_fp8_query=True,
        )
        if md is not None:
            md._fm_meta_cache = (key, meta, num_splits)
        return meta, num_splits

    def _run_decode_kernel_flashmla_dense(
        self,
        *,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        layer: RadixAttention,
        softmax_scale: float,
        output_scale: float,
        return_lse: bool,
    ):
        """Decode / target-verify path via FlashMLA's dense FP8 kernel.

        Handles s_q==1 (regular decode, non-causal) and s_q>1 (target verify,
        intra-window causal). Returns the same contract as
        ``tokenspeed_mla_decode`` -- ``out`` (or ``(out, lse)`` when
        ``return_lse``) -- or ``None`` to fall back.

        Layout parity: the tokenspeed FP8 KV cache stores each token as
        ``[kv_lora_rank(512) fp8 | qk_rope(64) fp8]`` == 576 bytes, and the
        absorbed MLA query is the same 576-wide FP8 layout, which is exactly
        what the FlashMLA dense kernel expects.
        """
        try:
            import flash_mla
        except ImportError:
            return None

        h_q = layer.tp_q_head_num
        # query is [b, s_q, h_q, kv_lora_rank + qk_rope] fp8, which is exactly
        # FlashMLA's expected q layout (s_q==1 regular decode, s_q>1 target
        # verify). No reshape/copy needed.
        s_q = int(query.shape[1])

        # One-time activation log per distinct q_len, so the serving log confirms
        # the FlashMLA-dense op is actually replacing tokenspeed_mla_decode
        # (s_q==1 regular decode, s_q>1 target-verify).
        _seen = self.__dict__.setdefault("_fm_logged_sq", set())
        if s_q not in _seen:
            _seen.add(s_q)
            logger.info(
                "[FlashMLA-dense-fp8] ACTIVE: tokenspeed_mla_decode -> "
                "flash_mla_dense_fp8_with_kvcache (s_q=%d, causal=%s, h_q=%d, "
                "page_size=%d, kv_lora_rank=%d)",
                s_q, s_q > 1, h_q, self.page_size, self.kv_lora_rank,
            )
            # Opt-in process-independent activation marker (logger output from a
            # scheduler subprocess is not reliably merged into the launcher log).
            # Only writes when SGLANG_FLASHMLA_DENSE_MARKER points at a file.
            _marker = os.environ.get("SGLANG_FLASHMLA_DENSE_MARKER")
            if _marker:
                try:
                    with open(_marker, "a") as _f:
                        _f.write(
                            f"pid={os.getpid()} s_q={s_q} causal={s_q > 1} "
                            f"h_q={h_q} page_size={self.page_size}\n"
                        )
                except Exception:
                    pass

        # Normalise the paged FP8 KV cache to FlashMLA's expected 4D layout
        # [num_blocks, page_size, num_heads_k=1, 576]. The trtllm decode path
        # hands us [num_blocks, num_heads_k=1, page_size, 576] (head axis at
        # dim 1, see trtllm_mla_backend: k_cache.view(-1, page, dim).unsqueeze(1)),
        # so drop that head axis and re-insert it at dim 2. Because num_heads_k
        # is 1 the underlying [num_blocks, page_size, 576] buffer is untouched
        # (squeeze/unsqueeze of size-1 dims stay views -- no copy).
        if kv_cache.dim() == 4:
            # [num_blocks, 1, page_size, 576] -> [num_blocks, page_size, 1, 576]
            kv4d = kv_cache.squeeze(1).unsqueeze(2)
        elif kv_cache.dim() == 3:
            # [num_blocks, page_size, 576] -> [num_blocks, page_size, 1, 576]
            kv4d = kv_cache.unsqueeze(2)
        else:
            kv4d = kv_cache

        # Split schedule for this decode step. Computed once per forward and
        # reused by every MLA layer (see _flashmla_metadata_for_step); seq_lens
        # is identical across layers within a step.
        meta, num_splits = self._flashmla_metadata_for_step(
            flash_mla, seq_lens, s_q, h_q
        )
        out, lse = flash_mla.flash_mla_dense_fp8_with_kvcache(
            query,
            kv4d,
            block_tables,
            seq_lens,
            self.kv_lora_rank,  # head_dim_v
            meta,
            num_splits,
            softmax_scale=softmax_scale,
            causal=(s_q > 1),  # target-verify: query row j attends [0, seqlen_k - s_q + 1 + j)
        )
        # FlashMLA out: [b, s_q, h_q, kv_lora_rank] bf16 -> tokenspeed raw_out
        # [tokens == b*s_q, h_q, v]. flatten(0,1) is a view (contiguous), no copy.
        out = out.flatten(0, 1)
        if output_scale != 1.0:
            out = out * output_scale
        if return_lse:
            # FlashMLA lse: [b, h_q, s_q] -> [tokens == b*s_q, h_q]
            return out, lse.transpose(1, 2).reshape(-1, h_q)
        return out

    def _run_decode_kernel(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        layer: RadixAttention,
        *,
        causal_seqs: Optional[torch.Tensor] = None,
        cp_world: int = 1,
        cp_rank: int = 0,
        return_lse: bool = False,
    ):
        k_scale = getattr(layer, "k_scale_float", None)
        if k_scale is None:
            k_scale = 1.0
        softmax_scale = float(layer.scaling) * float(k_scale)
        output_scale = float(k_scale)

        seq_lens_i32 = (
            seq_lens if seq_lens.dtype == torch.int32 else seq_lens.to(torch.int32)
        )

        # Env-gated equivalent replacement: route decode and target-verify to the
        # FlashMLA dense FP8 kernel. Only the cases FlashMLA dense covers 1:1 are
        # routed; everything else stays on tokenspeed_mla_decode.
        #   * no decode-context-parallel (cp_world == 1, causal_seqs is None)
        #   * page_size == 64 (FlashMLA requires page_block_size % 64 == 0)
        #   * query is dense [b, s_q, h_q, head_dim] with a uniform per-request
        #     q_len. s_q==1 (regular decode, non-causal) is always safe; s_q>1
        #     (target-verify, intra-window causal) is only routed when
        #     forward_extend marked it uniform+non-DCP (self._fm_route_ok).
        #     Ragged verify / draft-extend keep a padded q_len and stay on
        #     tokenspeed.
        if (
            envs.SGLANG_ENABLE_TOKENSPEED_MLA_DECODE_BALANCED.get()
            and cp_world == 1
            and causal_seqs is None
            and self.page_size == 64
            and query.dim() == 4
            and query.shape[0] == seq_lens_i32.shape[0]
            and (query.shape[1] == 1 or getattr(self, "_fm_route_ok", False))
        ):
            out = self._run_decode_kernel_flashmla_dense(
                query=query,
                kv_cache=kv_cache,
                block_tables=block_tables,
                seq_lens=seq_lens_i32,
                layer=layer,
                softmax_scale=softmax_scale,
                output_scale=output_scale,
                return_lse=return_lse,
            )
            if out is not None:
                return out

        if self._kvb_decoder is not None and cp_world > 1:
            res = self._kvb_decoder.decode(
                query,
                kv_cache,
                block_tables,
                seq_lens_i32,
                causal_seqs,
                softmax_scale,
                output_scale,
                layer.layer_id,
            )
            if res is not None:
                return res if return_lse else res[0]

        return tokenspeed_mla.tokenspeed_mla_decode(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=self._tokenspeed_workspace,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens_i32,
            max_seq_len=int(max_seq_len),
            softmax_scale=softmax_scale,
            output_scale=output_scale,
            enable_pdl=is_arch_support_pdl(),
            return_lse=return_lse,
            causal_seqs=causal_seqs,
            cp_world=cp_world,
            cp_rank=cp_rank,
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        is_neox: Optional[bool] = False,
        llama_4_scaling: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        # Mark FlashMLA-dense eligibility for the (multi-token) decode kernel used
        # inside super().forward_extend -> _run_decode_kernel. Only *uniform*,
        # non-DCP target-verify has a dense [bs, draft_token_num, h_q, d] query
        # that maps 1:1 to FlashMLA's causal kernel. Ragged verify (ragged_verify_
        # layout set) and draft-extend keep a padded/varying q_len that FlashMLA
        # dense cannot express, so they must stay on tokenspeed_mla_decode.
        """
        spec = getattr(forward_batch, "spec_info", None)
        self._fm_route_ok = bool(
            forward_batch.forward_mode.is_target_verify()
            and getattr(spec, "ragged_verify_layout", None) is None
            and not get_parallel().dcp_enabled
        )
        try:
            return super().forward_extend(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                q_rope,
                k_rope,
                cos_sin_cache,
                is_neox,
                llama_4_scaling,
            )
        finally:
            self._fm_route_ok = False

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        is_neox: Optional[bool] = False,
        llama_4_scaling: Optional[torch.Tensor] = None,
    ):
        parallel = get_parallel()
        if not parallel.dcp_enabled:
            return super().forward_decode(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                q_rope,
                k_rope,
                cos_sin_cache,
                is_neox,
                llama_4_scaling,
            )

        assert q_rope is not None and k_rope is not None
        if cos_sin_cache is None:
            q, k, k_rope = mla_quantize_without_rope_for_fp8(
                q, q_rope, k.squeeze(1), k_rope.squeeze(1)
            )
        else:
            q, k, k_rope = mla_quantize_and_rope_for_fp8(
                q,
                q_rope,
                k.squeeze(1),
                k_rope.squeeze(1),
                forward_batch.positions,
                cos_sin_cache,
                is_neox,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
            )

        if save_kv_cache:
            self.token_to_kv_pool.set_mla_kv_buffer(
                layer, forward_batch.out_cache_loc, k, k_rope
            )

        query = q.view(-1, layer.tp_q_head_num, layer.head_dim)
        if llama_4_scaling is not None:
            query = (query.to(self.q_data_type) * llama_4_scaling).to(self.data_type)
        if query.dim() == 3:
            query = query.unsqueeze(1)

        k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        kv_cache = k_cache.view(-1, self.page_size, self.kv_cache_dim).unsqueeze(1)
        metadata = (
            getattr(forward_batch, "decode_trtllm_mla_metadata", None)
            or self.forward_decode_metadata
        )
        metadata_batch_size = getattr(metadata, "batch_size", None)
        if (
            metadata_batch_size is not None
            and metadata_batch_size < forward_batch.batch_size
        ):
            self.init_forward_metadata(forward_batch)
            metadata = forward_batch.decode_trtllm_mla_metadata

        global_seq_lens = forward_batch.seq_lens[: forward_batch.batch_size]
        local_seq_lens = self._get_dcp_local_seq_lens(global_seq_lens)
        raw_out, lse = self._run_decode_kernel(
            query=query,
            kv_cache=kv_cache,
            block_tables=metadata.block_kv_indices,
            seq_lens=local_seq_lens,
            max_seq_len=metadata.max_seq_len_k,
            layer=layer,
            causal_seqs=global_seq_lens,
            cp_world=parallel.dcp_size,
            cp_rank=parallel.dcp_rank,
            return_lse=True,
        )

        output = raw_out.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        lse = lse.view(-1, layer.tp_q_head_num)
        fixup_zero_kv_rows(
            output,
            lse,
            local_seq_lens,
            self.q_indptr_decode[: forward_batch.batch_size + 1],
            1,
        )
        return output.flatten(1), lse

    def _run_prefill_kernel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        batch_size: int,
        cum_seq_lens_q: torch.Tensor,
        max_q_len: int,
        seq_lens_kv: torch.Tensor,
        cum_seq_lens_kv: torch.Tensor,
        max_kv_len: int,
        is_causal: bool,
        return_lse: bool,
        out_buffer: torch.Tensor,
        o_sf_scale: float = 1.0,
    ):  # Q/K/V arrive already in FP8 via the model-side fused path
        # (prepare_prefill_qkv / pack_prefix_chunk_kv); no quantize here.
        # Hybrid MLA models resolve the model-side hook through the outer
        # HybridLinearAttnBackend, so their fallback MHA path can pass V as a
        # last-dimension slice of kv_b_proj (stride(-2) > size(-1)). The
        # TokenSpeed prefill kernel requires dense Q/K/V layouts even though
        # the public wrapper accepts arbitrary torch tensors.
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        valid_q_tokens = int(cum_seq_lens_q[-1].detach().item())
        has_tail_padding = valid_q_tokens < q.shape[0]

        if has_tail_padding:
            # SGLang may pass padded Q/O buffers while cum_seq_lens_q describes
            # the real ragged token count. TokenSpeed only writes real rows, so
            # keep padded rows deterministic instead of returning uninitialized
            # torch.empty() contents into the residual stream.
            out_buffer.zero_()

        result = tokenspeed_mla.tokenspeed_mla_prefill(
            query=q,
            key=k,
            value=v,
            seq_lens=seq_lens_kv,
            cum_seq_lens=cum_seq_lens_kv,
            max_seq_len=int(max_kv_len),
            batch_size=int(batch_size),
            softmax_scale=float(layer.scaling),
            is_causal=is_causal,
            return_lse=return_lse,
            cum_seq_lens_q=cum_seq_lens_q,
            max_seq_len_q=int(max_q_len),
            enable_pdl=is_arch_support_pdl(),
        )
        if has_tail_padding:
            if isinstance(result, tuple):
                out, lse = result
                if out.shape[0] > valid_q_tokens:
                    out[valid_q_tokens:].zero_()
                if lse is not None and lse.shape[0] > valid_q_tokens:
                    lse[valid_q_tokens:].fill_(float("-inf"))
            elif result.shape[0] > valid_q_tokens:
                result[valid_q_tokens:].zero_()
        return result


class TokenspeedMLAMultiStepDraftBackend(TRTLLMMLAMultiStepDraftBackend):
    """Multi-step draft backend for tokenspeed_mla used by EAGLE."""

    def __init__(
        self, model_runner: ModelRunner, topk: int, speculative_num_steps: int
    ):
        super().__init__(model_runner, topk, speculative_num_steps)
        # Parent populates self.attn_backends with TRT-LLM instances; replace
        # them with tokenspeed instances sharing the parent's index buffers.
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i] = TokenspeedMLABackend(
                model_runner,
                skip_prefill=True,
                kv_indptr_buf=self.kv_indptr[i],
                q_indptr_decode_buf=self.q_indptr_decode,
            )
