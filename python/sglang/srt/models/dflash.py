# Adapted from the DFlash reference implementation (HF) but implemented with
# SGLang primitives (RadixAttention + SGLang KV cache). This model intentionally
# does not include token embeddings or an LM head; DFlash uses the target model's
# embedding/lm_head.

from __future__ import annotations

import logging
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.configs.laguna import normalize_gating
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.runtime_context import get_parallel
from sglang.srt.sharding import (
    WeightShardForwardPlanner,
    finalize_model_weight_shard,
    get_weight_sharding_config,
    get_weight_sharding_side_stream,
    prepare_layer_weight_shard_forward,
    resolve_weight_module,
    setup_layer_weight_shard_flat,
)
from sglang.srt.speculative.dflash_utils import (
    can_dflash_slice_qkv_weight,
    get_dflash_attention_sliding_window_size,
    get_dflash_layer_types,
    parse_dflash_draft_config,
)
from sglang.srt.utils import is_npu
from sglang.srt.utils.hf_transformers_utils import get_rope_config

_is_npu = is_npu()
if _is_npu:
    from sgl_kernel_npu.norm.split_qkv_rmsnorm_rope import split_qkv_rmsnorm_rope
logger = logging.getLogger(__name__)


def _get_dflash_layer_attention_params(
    config, layer_id: int
) -> Tuple[int, AttentionType]:
    layer_types = get_dflash_layer_types(config)
    if layer_types is None:
        return -1, AttentionType.ENCODER_ONLY
    if layer_id >= len(layer_types):
        raise ValueError(
            "DFLASH config.layer_types must contain one entry per draft layer. "
            f"Got {len(layer_types)} entries, layer_id={layer_id}."
        )

    layer_type = layer_types[layer_id]
    if layer_type == "full_attention":
        return -1, AttentionType.ENCODER_ONLY
    if layer_type == "sliding_attention":
        sliding_window_size = get_dflash_attention_sliding_window_size(config)
        assert sliding_window_size is not None
        return sliding_window_size, AttentionType.DECODER
    raise ValueError(
        "Unsupported DFLASH draft layer type. "
        f"layer_types[{layer_id}]={layer_type!r}."
    )


class DFlashAttention(nn.Module):
    def __init__(self, config, layer_id: int, quant_config=None) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        tp_size = int(get_parallel().tp_size)
        total_num_heads = int(config.num_attention_heads)
        total_num_kv_heads = int(
            getattr(config, "num_key_value_heads", total_num_heads)
        )
        head_dim = int(getattr(config, "head_dim", hidden_size // total_num_heads))

        self.hidden_size = hidden_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        assert self.total_num_heads % tp_size == 0, (
            f"DFlashAttention requires total_num_heads divisible by tp_size. "
            f"total_num_heads={self.total_num_heads}, tp_size={tp_size}."
        )
        self.num_heads = self.total_num_heads // tp_size
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0, (
                f"DFlashAttention requires total_num_kv_heads divisible by tp_size when >= tp_size. "
                f"total_num_kv_heads={self.total_num_kv_heads}, tp_size={tp_size}."
            )
        else:
            assert tp_size % self.total_num_kv_heads == 0, (
                f"DFlashAttention requires tp_size divisible by total_num_kv_heads when total_num_kv_heads < tp_size. "
                f"total_num_kv_heads={self.total_num_kv_heads}, tp_size={tp_size}."
            )
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * head_dim
        self.kv_size = self.num_kv_heads * head_dim

        attention_bias = bool(getattr(config, "attention_bias", False))
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            prefix="qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            prefix="o_proj",
        )

        # Per-head Q/K RMSNorm, matching HF Qwen3.
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

        rope_theta, rope_scaling = get_rope_config(config)
        rope_is_neox_style = bool(
            getattr(
                config, "rope_is_neox_style", getattr(config, "is_neox_style", True)
            )
        )
        max_position_embeddings = int(getattr(config, "max_position_embeddings", 32768))
        self.rotary_emb = get_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=rope_is_neox_style,
        )

        self.scaling = head_dim**-0.5
        self.sliding_window_size, self.attn_type = _get_dflash_layer_attention_params(
            config, layer_id
        )
        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=self.sliding_window_size,
            attn_type=self.attn_type,
        )

    def forward_prepare_npu(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)

        if self.attn.layer_id == 0:
            self.rotary_emb.get_cos_sin_with_position(positions)
        q, k, v = split_qkv_rmsnorm_rope(
            qkv,
            self.rotary_emb.position_sin,
            self.rotary_emb.position_cos,
            self.q_size,
            self.kv_size,
            self.head_dim,
            eps=self.q_norm.variance_epsilon,
            q_weight=self.q_norm.weight,
            k_weight=self.k_norm.weight,
            q_bias=getattr(self.q_norm, "bias", None),
            k_bias=getattr(self.k_norm, "bias", None),
        )
        return q, k, v

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        if _is_npu:
            q, k, v = self.forward_prepare_npu(positions, hidden_states)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q, k = apply_qk_norm(q, k, self.q_norm, self.k_norm, self.head_dim)
            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        attn_output = self.apply_attention_output(attn_output, hidden_states)
        output, _ = self.o_proj(attn_output)
        return output

    def apply_attention_output(
        self, attn_output: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        return attn_output

    def kv_proj_only(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project hidden_states to K/V only (skip Q).

        This is used by DFlash to materialize ctx tokens into the draft KV cache:
        we only need K/V for the cached tokens; Q is never consumed.
        """
        # Fast path for unquantized weights: slice the fused QKV weight and run one GEMM.
        can_slice_qkv_weight, _ = can_dflash_slice_qkv_weight(self.qkv_proj)
        if can_slice_qkv_weight:
            kv_slice = slice(self.q_size, self.q_size + 2 * self.kv_size)
            weight = self.qkv_proj.weight[kv_slice]
            bias = (
                self.qkv_proj.bias[kv_slice] if self.qkv_proj.bias is not None else None
            )
            kv = F.linear(hidden_states, weight, bias)
            k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
            return k, v

        # Fallback: compute full QKV and discard Q (keeps compatibility with quantized weights).
        qkv, _ = self.qkv_proj(hidden_states)
        _, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return k, v

    def apply_k_norm(self, k: torch.Tensor) -> torch.Tensor:
        k_by_head = k.reshape(-1, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        return k_by_head.view_as(k)

    def apply_k_rope(self, positions: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        # Match K shape so RoPE kernel head-count check passes on all backends.
        dummy_q = k.new_empty(k.shape)
        _, k = self.rotary_emb(positions, dummy_q, k)
        return k


class DFlashMLP(nn.Module):
    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        intermediate_size = int(getattr(config, "intermediate_size", 0))
        if intermediate_size <= 0:
            raise ValueError(
                f"Invalid intermediate_size={intermediate_size} for DFlash MLP."
            )

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix="gate_up_proj" if not prefix else f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix="down_proj" if not prefix else f"{prefix}.down_proj",
        )
        hidden_act = getattr(config, "hidden_act", "silu")
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported DFlash activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class DFlashDecoderLayer(nn.Module):
    attention_cls = DFlashAttention

    def __init__(self, config, layer_id: int, quant_config=None) -> None:
        super().__init__()
        # Weight sharding resolves the layer id off the module (`_layer_id`) to
        # apply SGLANG_K3_WEIGHT_SHARD_LAYERS, so it must be recorded here.
        self.layer_id = layer_id
        # DFlash layers are draft (MTP) layers: weight sharding applies its
        # draft-specific layer selection (default "all", i.e. every draft layer
        # including layer 0) instead of the K3 target's 1-92 range.
        self._weight_shard_is_draft = True
        hidden_size = int(config.hidden_size)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = self.attention_cls(
            config=config, layer_id=layer_id, quant_config=quant_config
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = DFlashMLP(config=config, quant_config=quant_config)

        # Must stay last in __init__: it shrinks `weight.data` to this rank's
        # row slice and replaces the weight_loader, so every init-time shape
        # decision above must have been made against the full weight.
        self._setup_weight_shard()

    def weight_shard_targets(self) -> list:
        """Dotted paths of draft weights that runtime weight sharding may split.

        The DSPARK dense draft is built under the attn-TP group (draft_tp_context,
        attn_tp==1 under full DP attention), so every linear is fully replicated
        across ranks and qualifies as a shard target. `fc` / `markov_w2` are plain
        ``nn.Linear`` without ``quant_method`` and are deliberately excluded (the
        sharding layer only supports SGLang linear modules).

        `self_attn.qkv_proj` is also excluded on purpose: the context-KV
        injection path (`DSparkDraftMixin.write_context_hidden_kv`) reads that
        weight directly, outside this layer's AllGather window. Reconstructed
        weights live in a 2-slot ping-pong staging pool, so by then the rows
        would already have been overwritten by another layer's AllGather. QKV
        is the smallest of the four linears, so keeping it replicated costs
        little.
        """
        return [
            "self_attn.o_proj",
            "mlp.gate_up_proj",
            "mlp.down_proj",
        ]

    def _setup_weight_shard(self) -> None:
        """Host one flat AllGather covering this layer's replicated weights.

        Mirrors KimiK3DecoderLayer._setup_weight_shard: the draft runs one
        AllGather per layer, launched one layer ahead by WeightShardForwardPlanner
        and consumed at the top of `forward`.
        """
        cfg = get_weight_sharding_config()
        if not cfg.enabled:
            return
        targets: list = []
        for target_path in self.weight_shard_targets():
            target_mod = resolve_weight_module(self, target_path)
            mod_tp = (
                int(getattr(target_mod, "tp_size", 1))
                if target_mod is not None
                else None
            )
            # Only fully replicated weights may be sharded; a draft built under
            # a real TP narrow must be left untouched.
            if target_mod is not None and mod_tp == 1:
                targets.append(target_path)
        if not targets:
            return
        sched = setup_layer_weight_shard_flat(
            self,
            targets=[(p, p) for p in targets],
            side_stream=get_weight_sharding_side_stream(),
        )
        logger.debug(
            "[K3WeightShard draft-layer=%s] targets=%s sched=%s",
            self.layer_id,
            targets,
            sched,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # First statement: attention reads sharded weights before the MLP, so
        # the layer's AllGather must be joined and `weight.data` rebound here.
        prepare_layer_weight_shard_forward(self)
        if hidden_states.numel() == 0:
            # Keep return types consistent for upstream callers.
            if residual is None:
                residual = hidden_states
            return hidden_states, residual

        # Pre-norm attention with fused residual+norm when possible (Qwen3-style).
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        attn_out = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states, residual = self.post_attention_layernorm(attn_out, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class DFlashDraftModel(nn.Module):
    """SGLang DFlash draft model (no embedding / lm_head weights).

    The checkpoint provides:
      - transformer weights for `layers.*`
      - `fc.weight`, `hidden_norm.weight` for projecting target context features
      - `norm.weight` for final normalization
    """

    decoder_layer_cls = DFlashDecoderLayer
    supports_fused_context_kv = True

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        self.config = config

        hidden_size = int(config.hidden_size)
        num_layers = int(config.num_hidden_layers)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.layers = nn.ModuleList(
            [
                self.decoder_layer_cls(
                    config=config, layer_id=i, quant_config=quant_config
                )
                for i in range(num_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # Project per-token target context features:
        # concat(K * hidden_size) -> hidden_size, where K is the number of target-layer
        # feature tensors concatenated per token (not necessarily equal to num_layers).
        draft_config = parse_dflash_draft_config(draft_hf_config=config)
        target_num_layers = (
            int(draft_config.num_target_layers)
            if draft_config.num_target_layers is not None
            else num_layers
        )
        target_layer_ids = draft_config.resolve_target_layer_ids(
            target_num_layers=target_num_layers, draft_num_layers=num_layers
        )
        num_context_features = len(target_layer_ids)

        self.num_context_features = int(num_context_features)
        self.fc = nn.Linear(
            self.num_context_features * hidden_size, hidden_size, bias=False
        )
        self.hidden_norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        self.block_size = draft_config.resolve_block_size(default=16)
        # feature_indices tuple -> column index into fc.weight. The PP-local
        # feature set is fixed at startup, so the index is built once (see
        # `_partial_fc_column_index`).
        self._partial_fc_index_cache: dict[Tuple[int, ...], torch.Tensor] = {}

        # Runtime weight sharding (K3 DSPARK draft under full DP attention):
        # every draft layer is fully replicated, so its four big linears are
        # split 1/N and rebuilt per forward by one flat AllGather each.
        self._weight_shard_layers = [
            (i, self.layers[i])
            for i in range(num_layers)
            if getattr(self.layers[i], "_weight_shard_sched", None) is not None
        ]
        self._weight_shard_planner = WeightShardForwardPlanner(
            self._weight_shard_layers
        )
        self._weight_shard_pool = None
        self._weight_shard_finalized = False
        logger.info(
            "[K3WeightShard] %s: %d weight-sharded layers",
            type(self).__name__,
            len(self._weight_shard_layers),
        )

    def get_attention_sliding_window_size(self) -> Optional[int]:
        return get_dflash_attention_sliding_window_size(self.config)

    def prepare_context_hidden_for_kv(
        self, layer: DFlashDecoderLayer, ctx_hidden: torch.Tensor
    ) -> torch.Tensor:
        return ctx_hidden

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        """Project concatenated target-layer hidden states into draft hidden_size."""
        expected = int(self.fc.in_features)
        if target_hidden.ndim != 2 or int(target_hidden.shape[-1]) != expected:
            raise ValueError(
                "DFLASH target_hidden feature dim mismatch. "
                f"Expected shape [N, {expected}] "
                f"(num_context_features={self.num_context_features}, hidden_size={int(self.config.hidden_size)}), "
                f"but got shape={tuple(target_hidden.shape)}. "
                "This usually means the target model is capturing a different number of layer features than "
                "the draft checkpoint/config expects."
            )
        return self.hidden_norm(self.fc(target_hidden))

    def project_target_hidden_partial(
        self, target_hidden: torch.Tensor, feature_indices: list[int]
    ) -> torch.Tensor:
        """Project PP-local target features into an additive pre-norm context."""
        if not feature_indices:
            raise ValueError("feature_indices must be non-empty.")
        feature_indices = [int(i) for i in feature_indices]
        if min(feature_indices) < 0 or max(feature_indices) >= self.num_context_features:
            raise ValueError(
                "feature_indices out of range for DFLASH context projection: "
                f"{feature_indices=} {self.num_context_features=}."
            )
        hidden_size = int(self.config.hidden_size)
        expected = len(feature_indices) * hidden_size
        if target_hidden.ndim != 2 or int(target_hidden.shape[-1]) != expected:
            raise ValueError(
                "DFLASH partial target_hidden feature dim mismatch. "
                f"Expected shape [N, {expected}] for {feature_indices=}, "
                f"but got shape={tuple(target_hidden.shape)}."
            )

        index = self._partial_fc_column_index(tuple(feature_indices), hidden_size)
        weight = self.fc.weight.index_select(1, index)
        if target_hidden.dtype != weight.dtype:
            target_hidden = target_hidden.to(weight.dtype)
        return F.linear(target_hidden, weight)

    def _partial_fc_column_index(
        self, feature_indices: Tuple[int, ...], hidden_size: int
    ) -> torch.Tensor:
        """fc.weight column index for `feature_indices`, built once per feature set.

        Building it per forward from a host list (`torch.tensor(cols, device=...)`)
        takes the pageable-H2D path, which appends a cudaStreamSynchronize: the
        host would park until the whole target forward drained, delaying every PP
        ring send that follows and idling the downstream stages' GPUs.
        """
        index = self._partial_fc_index_cache.get(feature_indices)
        if index is None:
            device = self.fc.weight.device
            index = torch.cat(
                [
                    torch.arange(
                        idx * hidden_size,
                        (idx + 1) * hidden_size,
                        dtype=torch.long,
                        device=device,
                    )
                    for idx in feature_indices
                ]
            )
            self._partial_fc_index_cache[feature_indices] = index
        return index

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        if input_embeds is None:
            if hasattr(self, "forward_embed"):
                input_embeds = self.forward_embed(input_ids)
            else:
                raise ValueError(
                    "DFlashDraftModel requires `input_embeds` (use the target "
                    "embedding)."
                )
        hidden_states = input_embeds
        residual: Optional[torch.Tensor] = None

        self._weight_shard_planner.reset()
        self._weight_shard_planner.pre_dispatch(0)
        for i, layer in enumerate(self.layers):
            self._weight_shard_planner.maybe_dispatch(i + 1)
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual
            )
            self._weight_shard_planner.update_done_event(i, layer)

        if hidden_states.numel() != 0:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)

        return LogitsProcessorOutput(
            next_token_logits=None,
            hidden_states=hidden_states,
        )

    def force_weight_shard_finalize(self) -> None:
        """Finalize the draft's replicated weight shards before graph capture.

        Invoked by ModelRunner.initialize() through the same hasattr hook as the
        Kimi K3 target model; runs inside the draft worker after all weights are
        loaded and before the draft decode graphs are captured.
        """
        if self._weight_shard_finalized:
            return
        modules = [module for _, module in self._weight_shard_layers]
        self._weight_shard_pool = finalize_model_weight_shard(modules, mode="force")
        self._weight_shard_finalized = True
        if modules:
            torch.cuda.empty_cache()
        self._enable_first_weight_shard_prefetch()

    def _num_external_weight_shards(self) -> int:
        # Exactly AG 0 is hoisted (host-launched to overlap the ReplaySSM commit
        # fold); AG 1..N-1 stay in-graph under the planner's depth-1 lookahead
        # (layer i dispatches AG i+1), which keeps the in-model pipeline intact.
        return min(1, len(self._weight_shard_layers))

    def _enable_first_weight_shard_prefetch(self) -> None:
        """Move ONLY layer 0's AllGather off the forward path.

        Layers 1..N-1 keep their in-graph AGs dispatched one layer ahead, so the
        draft forward itself carries no extra bubble beyond layer 0's slot. AG 0
        has nothing inside the forward to overlap with: it is launched by the
        host on a dedicated stream right where this step's ReplaySSM commit fold
        begins, so it overlaps that ~4ms pure-compute window instead of sitting
        exposed at the head of the next draft forward.

        The dedicated stream matters: draft and target share one side stream,
        and the target's own flat AGs queue ahead of an eager AG there, pushing
        its start past the fold (v5 measured ~25us overlap). On its own stream
        the AG's start time is determined solely by the gate event.
        """
        self._external_weight_shard_scheds = []
        self._weight_shard_prefetched = False
        self._weight_shard_gate = None
        n_external = self._num_external_weight_shards()
        if n_external <= 0:
            return
        sched = getattr(self._weight_shard_layers[0][1], "_weight_shard_sched", None)
        if sched is None:
            return
        sched.set_external_prefetch(True)
        # One dedicated prefetch stream shared by all hoisted layers; today that
        # is exactly one (AG 0), so there is never concurrent eager traffic on it.
        self._ag_prefetch_stream = torch.cuda.Stream()
        sched.set_prefetch_stream(self._ag_prefetch_stream)
        self._external_weight_shard_scheds.append(sched)
        self._weight_shard_gate = torch.cuda.Event()
        logger.info(
            "[K3WeightShard] draft hoisted %d/%d layer AG(s) out of the graph "
            "(dedicated prefetch stream)",
            len(self._external_weight_shard_scheds),
            len(self._weight_shard_layers),
        )

    def prefetch_first_weight_shard(self) -> None:
        """Launch AG 0 for the *next* forward on its own stream.

        Called by the worker immediately before the ReplaySSM commit fold. The
        gate event is recorded here on the compute stream just before the fold
        enters the stream, so on the device the AG starts together with the fold
        and runs inside its ~4ms window.
        """
        scheds = getattr(self, "_external_weight_shard_scheds", None)
        if not scheds or torch.cuda.is_current_stream_capturing():
            return
        if self._weight_shard_prefetched:
            # One AG per step, no matter how many gate modes/call sites run:
            # a duplicate submission would desynchronize the shard-group PG.
            return
        gate = self._weight_shard_gate
        if gate is not None:
            gate.record(torch.cuda.current_stream())
        for sched in scheds:
            sched.external_launch(gate_event=gate)
        self._weight_shard_prefetched = True

    def wait_first_weight_shard(self) -> None:
        """Block the compute stream on the prefetched AG before the forward."""
        scheds = getattr(self, "_external_weight_shard_scheds", None)
        if not scheds or torch.cuda.is_current_stream_capturing():
            return
        if not self._weight_shard_prefetched:
            # Cold start (and any step that skipped the prefetch): fall back to
            # launching inline so the forward never reads a stale buffer.
            for sched in scheds:
                sched.external_launch()
        for sched in scheds:
            sched.external_wait()
        # Later layers reuse the staging slots inside this same forward, so the
        # prefetched contents do not survive it: every step must re-launch.
        self._weight_shard_prefetched = False

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, weight_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())

        def resolve_param_name(name: str) -> Optional[str]:
            if name in params_dict:
                return name
            if name.startswith("model."):
                stripped_name = name[len("model.") :]
                if stripped_name in params_dict:
                    return stripped_name
            else:
                prefixed_name = f"model.{name}"
                if prefixed_name in params_dict:
                    return prefixed_name
            return None

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if f".{weight_name}." not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                resolved_name = resolve_param_name(mapped_name)
                if resolved_name is None:
                    continue
                param = params_dict[resolved_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                resolved_name = resolve_param_name(name)
                if resolved_name is None:
                    # Ignore unexpected weights (e.g., HF rotary caches).
                    continue
                param = params_dict[resolved_name]
                if resolved_name.endswith("fc.weight") and tuple(
                    loaded_weight.shape
                ) != tuple(param.shape):
                    raise ValueError(
                        "DFLASH fc.weight shape mismatch. This usually means the draft checkpoint's "
                        "number of context features (K) does not match this config. "
                        f"Expected fc.weight.shape={tuple(param.shape)} "
                        f"(num_context_features={self.num_context_features}, hidden_size={int(self.config.hidden_size)}), "
                        f"but got {tuple(loaded_weight.shape)} for weight '{name}'."
                    )
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


class DFlashLagunaAttention(DFlashAttention):
    """Laguna DFlash attention with the trained Laguna softplus gate."""

    def __init__(self, config, layer_id: int, quant_config=None) -> None:
        super().__init__(config=config, layer_id=layer_id, quant_config=quant_config)
        hidden_size = int(config.hidden_size)
        total_num_heads = self.total_num_heads
        gating = normalize_gating(getattr(config, "gating", True))
        self.gating = gating
        self.gate_per_head = gating == "per-head"
        if self.gating == "disabled":
            self.g_proj = None
        else:
            g_out = (
                total_num_heads
                if self.gate_per_head
                else total_num_heads * self.head_dim
            )
            self.g_proj = ColumnParallelLinear(
                hidden_size,
                g_out,
                bias=False,
                quant_config=quant_config,
                prefix="g_proj",
            )

    def apply_attention_output(
        self, attn_output: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if self.g_proj is None:
            return attn_output

        gate, _ = self.g_proj(hidden_states)
        gate = F.softplus(gate.float()).to(attn_output.dtype)
        if self.gate_per_head:
            attn_shape = attn_output.shape
            return (
                attn_output.view(*attn_shape[:-1], self.num_heads, self.head_dim)
                * gate.unsqueeze(-1)
            ).view(attn_shape)
        else:
            return attn_output * gate


class DFlashLagunaDecoderLayer(DFlashDecoderLayer):
    attention_cls = DFlashLagunaAttention


class DFlashLagunaForCausalLM(DFlashDraftModel):
    """Laguna DFlash draft model matching the exported Speculators checkpoint."""

    decoder_layer_cls = DFlashLagunaDecoderLayer
    supports_fused_context_kv = False

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))
        hidden_size = int(config.hidden_size)
        self.aux_hidden_norms = nn.ModuleList(
            [
                RMSNorm(hidden_size, eps=rms_norm_eps)
                for _ in range(self.num_context_features)
            ]
        )

    def prepare_context_hidden_for_kv(
        self, layer: DFlashLagunaDecoderLayer, ctx_hidden: torch.Tensor
    ) -> torch.Tensor:
        return layer.input_layernorm(ctx_hidden)

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        expected = int(self.fc.in_features)
        if target_hidden.ndim != 2 or int(target_hidden.shape[-1]) != expected:
            raise ValueError(
                "Laguna DFLASH target_hidden feature dim mismatch. "
                f"Expected shape [N, {expected}] "
                f"(num_context_features={self.num_context_features}, hidden_size={int(self.config.hidden_size)}), "
                f"but got shape={tuple(target_hidden.shape)}."
            )

        num_slices = int(self.num_context_features)
        slice_size = int(target_hidden.shape[-1]) // num_slices
        slices = target_hidden.view(target_hidden.shape[0], num_slices, slice_size)
        compute_dtype = self.fc.weight.dtype
        if slices.dtype != compute_dtype:
            slices = slices.to(compute_dtype)
        normed = torch.empty_like(slices)
        for i, norm in enumerate(self.aux_hidden_norms):
            normed[:, i, :] = norm(slices[:, i, :])
        fused = normed.reshape(target_hidden.shape[0], -1)
        return self.hidden_norm(self.fc(fused))

    def project_target_hidden_partial(
        self, target_hidden: torch.Tensor, feature_indices: list[int]
    ) -> torch.Tensor:
        if not feature_indices:
            raise ValueError("feature_indices must be non-empty.")
        feature_indices = [int(i) for i in feature_indices]
        hidden_size = int(self.config.hidden_size)
        expected = len(feature_indices) * hidden_size
        if target_hidden.ndim != 2 or int(target_hidden.shape[-1]) != expected:
            raise ValueError(
                "Laguna DFLASH partial target_hidden feature dim mismatch. "
                f"Expected shape [N, {expected}] for {feature_indices=}, "
                f"but got shape={tuple(target_hidden.shape)}."
            )
        slices = target_hidden.view(
            target_hidden.shape[0], len(feature_indices), hidden_size
        )
        compute_dtype = self.fc.weight.dtype
        if slices.dtype != compute_dtype:
            slices = slices.to(compute_dtype)
        normed = torch.empty_like(slices)
        for out_idx, feature_idx in enumerate(feature_indices):
            normed[:, out_idx, :] = self.aux_hidden_norms[feature_idx](
                slices[:, out_idx, :]
            )
        return super().project_target_hidden_partial(
            normed.reshape(target_hidden.shape[0], -1), feature_indices
        )


EntryClass = [DFlashDraftModel, DFlashLagunaForCausalLM]
