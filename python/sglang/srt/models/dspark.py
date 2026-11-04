from __future__ import annotations

import logging
from typing import Callable, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.distributed.communication_op import tensor_model_parallel_all_gather
from sglang.srt.environ import envs
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dflash import DFlashDraftModel
from sglang.srt.runtime_context import get_parallel
from sglang.srt.sharding import WeightShardForwardPlanner
from sglang.srt.speculative.dflash_utils import can_dflash_slice_qkv_weight
from sglang.srt.speculative.dspark_components.dspark_config import (
    parse_dspark_draft_config,
)
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyMode,
    read_ragged_verify_mode,
)

logger = logging.getLogger(__name__)

StepSampler = Callable[[torch.Tensor, int], torch.Tensor]


def gather_and_crop_vocab(
    local_logits: torch.Tensor, lm_head: nn.Module, group=None
) -> torch.Tensor:
    if group is None:
        full_logits = tensor_model_parallel_all_gather(local_logits, dim=-1)
    else:
        full_logits = group.all_gather(local_logits, dim=-1)
    return full_logits[..., : int(lm_head.org_vocab_size)]


def shared_vocab_group(module: Optional[nn.Module]):
    """TP group whose ranks own the vocab shards of a shared target module.

    Read it off the module (``vocab_shard_group``, set when the layer built its
    shard geometry) rather than re-deriving it: the draft forward runs with the
    ambient TP group patched to the attention-TP group, and the scope may also
    be an intra-node block instead of the whole TP group
    (SGLANG_VOCAB_TP_SIZE). Returns None when the module is not sharded, in
    which case no vocab collective is needed.
    """
    if module is None or int(getattr(module, "tp_size", 1)) <= 1:
        return None
    group = getattr(module, "vocab_shard_group", None)
    if group is not None:
        return group
    # Layers that predate vocab_shard_group (or non-VocabParallelEmbedding
    # modules) keep the historical scope.
    if getattr(module, "use_attn_tp_group", False):
        return get_parallel().attn_tp_group
    return get_parallel().tp_group


def vocab_group_spans_dp(group) -> bool:
    """True when the vocab shard group spans more than one attention-DP group.

    Then every rank of the group drafts a *different* token set while holding
    only 1/N of the vocab, so the rows have to be gathered over the shard
    group before the vocab collective -- the same thing the target's
    LogitsProcessor does in this layout (``do_tensor_parallel_all_gather_dp_attn``).
    """
    return group is not None and group.world_size > get_parallel().attn_tp_size


def lm_head_dp_gather_enabled(group) -> bool:
    """Whether the shared lm_head needs (and is allowed to use) the DP-gathered
    base-logits path. Gated by SGLANG_ENABLE_DSPARK_LM_HEAD_DP_SHARD so the
    default DSpark layout stays --enable-dp-lm-head; raise rather than silently
    computing only the local vocab slice when the layout demands the gather but
    the switch is off (the arg hook normally catches this at startup)."""
    if not vocab_group_spans_dp(group):
        return False
    if not envs.SGLANG_ENABLE_DSPARK_LM_HEAD_DP_SHARD.get():
        raise ValueError(
            "DSpark's shared lm_head shards its vocab over the global TP group "
            "while dp attention is on, which requires the DP-gathered "
            "base-logits path. Set SGLANG_ENABLE_DSPARK_LM_HEAD_DP_SHARD=1 "
            "(pure DP only) or run with --enable-dp-lm-head."
        )
    return True


def _redistribute_vocab_all_to_all(
    local_logits: torch.Tensor, *, group, cap: int, world: int
) -> torch.Tensor:
    """Transpose the (row block, vocab shard) grid with one all-to-all.

    Chunk j holds the rows of rank j; after the all-to-all chunk j holds rank
    j's vocab shard for our own rows, so concatenating in rank order rebuilds
    the padded vocab dim. all_to_all_single splits by numel over the flattened
    leading dim, so hand it flat views of the chunked buffers.

    Traffic-optimal (one row block in, one out) but it needs send/recv
    connections to all ``world - 1`` peers; NCCL builds those lazily on the
    first call, and when that call is a draft cuda graph capture the one-off
    cost is billed to the graph (see the all-gather variant).
    """
    send = local_logits.view(world, cap, -1).contiguous()
    recv = torch.empty_like(send)
    group.all_to_all_single(recv.view(-1), send.view(-1))
    return recv.permute(1, 0, 2).reshape(cap, -1)


def _redistribute_vocab_all_gather(
    local_logits: torch.Tensor, *, group, cap: int, world: int
) -> torch.Tensor:
    """Same transpose over a ring all-gather (SGLANG_DSPARK_LM_HEAD_VOCAB_ALL_GATHER).

    An all-gather is symmetric, so every rank has to contribute its shard for
    EVERY row block: the gather lands the whole grid and we keep only our own
    row block, i.e. ``world`` times the traffic and a ``world``-sized buffer.
    In exchange it rides the ring connections the vocab collectives already
    established, so it never triggers the per-peer P2P setup the all-to-all
    pays for.

    Slice our row block BEFORE the permute so the contiguous copy is one row
    block (cap x vocab) instead of the whole gathered grid.
    """
    shard_width = local_logits.shape[-1]
    grid = local_logits.new_empty((world * local_logits.shape[0], shard_width))
    group.all_gather_into_tensor(grid, local_logits)
    # grid[r] is rank r's vocab shard for every row block, so grid[r] row block
    # j is the tile (r, j). Ours is j == our rank in the group.
    tiles = grid.view(world, world * cap, shard_width)
    start = group.rank_in_group * cap
    return tiles[:, start : start + cap, :].permute(1, 0, 2).reshape(cap, -1)


def shared_lm_head_logits(
    hidden: torch.Tensor,
    lm_head: nn.Module,
    compute_shard: Callable[[torch.Tensor], torch.Tensor],
    *,
    group=None,
    dp_gather: bool = False,
    rows_capacity: Optional[int] = None,
) -> torch.Tensor:
    """Full-vocab logits from the shared (vocab-parallel) target lm_head.

    Plain case: every rank of ``group`` holds the same rows, so the per-rank
    vocab shards are all-gathered along the vocab dim.

    ``dp_gather`` case (pure DP: the lm_head shards over the global TP group
    while each rank drafts its own tokens): pad the rows to ``rows_capacity``
    so every rank contributes the same amount, all-gather the rows over the
    shard group, run the local vocab shard over all rows, then redistribute so
    each rank keeps the full vocab for *its own* rows only. The redistribution
    is a transpose of the (row block, vocab shard) grid; both an all-to-all
    (default) and a ring all-gather implement it, see the two helpers above for
    the tradeoff. This path is captured in the draft cuda graph, so their
    buffers are held for the lifetime of the graph.
    """
    if not dp_gather:
        return gather_and_crop_vocab(compute_shard(hidden), lm_head, group)

    rows = int(hidden.shape[0])
    cap = rows if rows_capacity is None else int(rows_capacity)
    if rows > cap:
        raise ValueError(
            f"DSpark DP-gathered lm head got {rows} rows > per-rank capacity {cap}."
        )
    # Pin the gathered dtype to the lm_head dtype so an idle rank joining with
    # zero rows issues byte-identical collectives.
    hidden = hidden.to(lm_head.weight.dtype)
    if rows != cap:
        padded = hidden.new_zeros((cap,) + tuple(hidden.shape[1:]))
        padded[:rows] = hidden
    else:
        padded = hidden.contiguous()

    gathered = group.all_gather(padded, dim=0)
    local_logits = compute_shard(gathered)
    redistribute = (
        _redistribute_vocab_all_gather
        if envs.SGLANG_DSPARK_LM_HEAD_VOCAB_ALL_GATHER.get()
        else _redistribute_vocab_all_to_all
    )
    full = redistribute(local_logits, group=group, cap=cap, world=group.world_size)
    return full[:rows, : int(lm_head.org_vocab_size)]


def shared_embed_lookup(
    embed_tokens: nn.Module,
    input_ids: torch.Tensor,
    *,
    group=None,
    dp_gather: bool = False,
    rows_capacity: Optional[int] = None,
) -> torch.Tensor:
    """Embed with the shared (vocab-parallel) target embedding.

    Plain case: the layer's own reduce already stays inside a group that holds
    the same tokens as the draft, so just call it.

    ``dp_gather`` case (the table shards over the global TP group while each
    rank drafts its own tokens): pad the ids to ``rows_capacity``, gather them
    over the shard group so every rank looks up the same ids, reduce the shards
    on that group EXPLICITLY (the draft's ambient TP group is patched to
    attn-TP, so the layer's own reduce would go to the wrong group), then keep
    this rank's rows.
    """
    if not dp_gather:
        return embed_tokens(input_ids)

    rows = int(input_ids.shape[0])
    if rows_capacity is None:
        raise RuntimeError(
            "DSpark's DP-gathered embed needs a per-rank row capacity agreed "
            "across attention-DP ranks; the draft worker must publish "
            "dp_embed_rows_capacity before the forward (without it the ranks "
            "would enter the id gather with different shapes and deadlock)."
        )
    cap = int(rows_capacity)
    if rows > cap:
        raise ValueError(
            f"DSpark DP-gathered embed got {rows} ids > per-rank capacity {cap}."
        )
    if rows != cap:
        padded = input_ids.new_zeros((cap,))
        padded[:rows] = input_ids
    else:
        padded = input_ids.contiguous()

    gathered = group.all_gather(padded, dim=0)
    local = embed_tokens.embed_local_shard(gathered)
    full = group.all_reduce(local)
    start = group.rank_in_group * cap
    return full[start : start + rows]


def run_markov_block(
    head: nn.Module,
    base_logits: torch.Tensor,
    *,
    first_prev_tokens: torch.Tensor,
    hidden_states: Optional[torch.Tensor],
    sampler: StepSampler,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, proposal_len = base_logits.shape[:2]
    if proposal_len == 0:
        empty = torch.empty(batch_size, 0, dtype=torch.long, device=base_logits.device)
        return empty, base_logits

    sampled_tokens = []
    corrected_logits = []
    prev_tokens = first_prev_tokens.long()
    for step_idx in range(proposal_len):
        step_hidden = None if hidden_states is None else hidden_states[:, step_idx, ...]
        step_logits = head.apply_step_logits(
            base_logits[:, step_idx, :],
            token_ids=prev_tokens,
            hidden_states=step_hidden,
        )
        next_tokens = sampler(step_logits, step_idx)
        sampled_tokens.append(next_tokens)
        corrected_logits.append(step_logits.unsqueeze(1))
        prev_tokens = next_tokens
    return (
        torch.stack(sampled_tokens, dim=1),
        torch.cat(corrected_logits, dim=1),
    )


class VanillaMarkov(nn.Module):

    markov_head_type = "vanilla"

    def __init__(self, *, vocab_size: int, markov_rank: int) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        if self.markov_rank <= 0:
            raise ValueError(
                f"VanillaMarkov requires markov_rank > 0, got {self.markov_rank}."
            )
        self.markov_w1 = nn.Embedding(self.vocab_size, self.markov_rank)
        self.markov_w2 = nn.Linear(self.markov_rank, self.vocab_size, bias=False)

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids.long())

    def project_bias(self, latent_states: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(latent_states)

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        del hidden_states
        return self.project_bias(self.get_prev_embeddings(token_ids))

    def apply_step_logits(
        self,
        logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return logits + self.compute_step_bias(token_ids, hidden_states)

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if base_logits.size(-2) == 0:
            return base_logits
        return base_logits + self.compute_step_bias(token_ids, hidden_states)

    def sample_block(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_tokens: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        sampler: StepSampler,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return run_markov_block(
            self,
            base_logits,
            first_prev_tokens=first_prev_tokens,
            hidden_states=hidden_states,
            sampler=sampler,
        )


class GatedMarkovHead(VanillaMarkov):

    markov_head_type = "gated"

    def __init__(self, *, vocab_size: int, markov_rank: int, hidden_size: int) -> None:
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.gate_proj = nn.Linear(int(hidden_size) + markov_rank, markov_rank)

    def compute_gate(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if hidden_states is None:
            raise ValueError("GatedMarkovHead requires hidden_states.")
        prev_embeddings = self.get_prev_embeddings(token_ids)
        gate_inputs = torch.cat([hidden_states, prev_embeddings], dim=-1)
        return torch.sigmoid(self.gate_proj(gate_inputs))

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        prev_embeddings = self.get_prev_embeddings(token_ids)
        gate = self.compute_gate(token_ids, hidden_states).to(
            dtype=prev_embeddings.dtype
        )
        return self.project_bias(gate * prev_embeddings)


class RNNHead(VanillaMarkov):

    markov_head_type = "rnn"

    def __init__(self, *, vocab_size: int, markov_rank: int, hidden_size: int) -> None:
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.hidden_size = int(hidden_size)
        self.state_size = markov_rank
        self.joint_proj = nn.Linear(2 * markov_rank + self.hidden_size, 3 * markov_rank)

    def _rnn_step(
        self,
        state: torch.Tensor,
        prev_embeddings: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat([state, prev_embeddings, hidden_states], dim=-1)
        gate_raw, candidate_raw, output_raw = self.joint_proj(z).chunk(3, dim=-1)
        gate = torch.sigmoid(gate_raw)
        candidate = torch.tanh(candidate_raw)
        new_state = gate * state + (1.0 - gate) * candidate
        bias = self.project_bias(torch.tanh(output_raw))
        return new_state, bias

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if hidden_states is None:
            raise ValueError("RNNHead requires hidden_states.")
        prev_embeddings = self.get_prev_embeddings(token_ids)
        state = torch.zeros_like(prev_embeddings)
        _, bias = self._rnn_step(state, prev_embeddings, hidden_states)
        return bias

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if hidden_states is None:
            raise ValueError("RNNHead requires hidden_states.")
        block_size = base_logits.size(-2)
        if block_size == 0:
            return base_logits
        leading_shape = base_logits.shape[:-2]
        state = torch.zeros(
            *leading_shape,
            self.markov_rank,
            device=base_logits.device,
            dtype=hidden_states.dtype,
        )
        output_logits = []
        for k in range(block_size):
            prev_emb = self.get_prev_embeddings(token_ids[..., k])
            state, bias = self._rnn_step(state, prev_emb, hidden_states[..., k, :])
            output_logits.append(base_logits[..., k, :] + bias)
        return torch.stack(output_logits, dim=-2)

    def sample_block(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_tokens: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        sampler: StepSampler,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden_states is None:
            raise ValueError("RNNHead requires hidden_states.")
        batch_size, proposal_len = base_logits.shape[:2]
        if proposal_len == 0:
            empty = torch.empty(
                batch_size, 0, dtype=torch.long, device=base_logits.device
            )
            return empty, base_logits

        state = torch.zeros(
            batch_size,
            self.markov_rank,
            device=base_logits.device,
            dtype=hidden_states.dtype,
        )
        sampled_tokens = []
        corrected_logits = []
        prev_tokens = first_prev_tokens.long()
        for step_idx in range(proposal_len):
            prev_emb = self.get_prev_embeddings(prev_tokens)
            state, bias = self._rnn_step(state, prev_emb, hidden_states[:, step_idx, :])
            step_logits = base_logits[:, step_idx, :] + bias
            next_tokens = sampler(step_logits, step_idx)
            sampled_tokens.append(next_tokens)
            corrected_logits.append(step_logits.unsqueeze(1))
            prev_tokens = next_tokens
        return (
            torch.stack(sampled_tokens, dim=1),
            torch.cat(corrected_logits, dim=1),
        )


def build_markov_head(config) -> Optional[nn.Module]:
    markov_rank = int(getattr(config, "markov_rank", 0))
    if markov_rank <= 0:
        raise ValueError(
            "DSpark requires markov_rank > 0 (the Markov head is the core of the "
            f"semi-AR draft); got markov_rank={markov_rank}."
        )
    markov_head_type = str(getattr(config, "markov_head_type", "vanilla")).lower()
    vocab_size = int(config.vocab_size)
    hidden_size = int(config.hidden_size)
    if markov_head_type == "vanilla":
        return VanillaMarkov(vocab_size=vocab_size, markov_rank=markov_rank)
    if markov_head_type == "gated":
        return GatedMarkovHead(
            vocab_size=vocab_size, markov_rank=markov_rank, hidden_size=hidden_size
        )
    if markov_head_type == "rnn":
        return RNNHead(
            vocab_size=vocab_size, markov_rank=markov_rank, hidden_size=hidden_size
        )
    raise ValueError(f"Unsupported DSpark markov_head_type={markov_head_type!r}.")


class DSparkConfidenceHead(nn.Module):

    def __init__(
        self,
        *,
        hidden_size: int,
        markov_rank: int,
        with_markov: bool = True,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.with_markov = bool(with_markov)
        input_dim = int(hidden_size) + (int(markov_rank) if self.with_markov else 0)
        self.proj = nn.Linear(input_dim, 1, bias=bias, dtype=dtype)
        self.register_buffer(
            "sts_temperatures", torch.ones((), dtype=torch.float32), persistent=False
        )
        self._last_confidence_raw: Optional[torch.Tensor] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        markov_embed_stack: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.with_markov:
            if markov_embed_stack is None:
                raise ValueError(
                    "DSparkConfidenceHead(with_markov=True) requires markov_embed_stack."
                )
            features = torch.cat(
                [hidden_states, markov_embed_stack.to(dtype=hidden_states.dtype)],
                dim=-1,
            )
        else:
            features = hidden_states
        features = features.to(dtype=self.proj.weight.dtype)
        return self.proj(features).squeeze(-1)

    def apply_sts(self, confidence_raw: torch.Tensor) -> torch.Tensor:
        self._last_confidence_raw = confidence_raw
        return torch.sigmoid(confidence_raw.float() / self.sts_temperatures)


def build_confidence_head(config) -> Optional[nn.Module]:
    if read_ragged_verify_mode() is RaggedVerifyMode.STATIC:
        return None
    if not hasattr(config, "enable_confidence_head"):
        logger.warning(
            "DSpark draft config has no enable_confidence_head field; treating the "
            "confidence head as enabled."
        )
    hidden_size = int(config.hidden_size)
    markov_rank = int(getattr(config, "markov_rank", 0))
    with_markov = bool(getattr(config, "confidence_head_with_markov", markov_rank > 0))
    if with_markov and markov_rank <= 0:
        raise ValueError(
            "DSpark confidence_head_with_markov requires markov_rank > 0, "
            f"got markov_rank={markov_rank}."
        )
    return DSparkConfidenceHead(
        hidden_size=hidden_size,
        markov_rank=markov_rank,
        with_markov=with_markov,
    )


_DSPARK_SKIPPED_WEIGHT_PREFIXES = (
    "embed_tokens.",
    "lm_head.",
    "rotary_emb.",
)


class DSparkDraftMixin:

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        dspark_config = parse_dspark_draft_config(draft_hf_config=config)
        if not dspark_config.require_markov():
            raise ValueError(
                "DSpark draft requires markov_rank > 0, "
                f"got markov_rank={dspark_config.markov_rank}."
            )
        self.gamma = int(dspark_config.resolve_gamma(default=self.block_size))
        self.markov_head = build_markov_head(config)
        self.confidence_head = build_confidence_head(config)
        self.lm_head: Optional[nn.Module] = None
        self.lm_head_group = None
        self.lm_head_dp_gather = False
        self.embed_group = None
        self.embed_dp_gather = False
        # Row capacity of the DP-gathered embed, published by the draft worker
        # (fixed at graph-capture time, refreshed per step for the eager path).
        self.dp_embed_rows_capacity: Optional[int] = None

    def attach_shared_modules(
        self, *, embed_tokens: nn.Module, lm_head: nn.Module
    ) -> None:
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head
        self.embed_group = shared_vocab_group(embed_tokens)
        self.embed_dp_gather = vocab_group_spans_dp(self.embed_group)
        self.lm_head_group = shared_vocab_group(lm_head)
        self.lm_head_dp_gather = lm_head_dp_gather_enabled(self.lm_head_group)

    def forward_embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Embeds with the shared target embedding INSIDE the draft graph
        # (the runner skips the eager input_embeds staging when the draft
        # model exposes forward_embed).
        return shared_embed_lookup(
            self.embed_tokens,
            input_ids,
            group=self.embed_group,
            dp_gather=self.embed_dp_gather,
            rows_capacity=self.dp_embed_rows_capacity,
        )

    def compute_base_logits(
        self, hidden: torch.Tensor, *, dp_rows_capacity: Optional[int] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.lm_head is None:
            raise ValueError(
                "DSpark dense draft requires the target lm_head "
                "(call attach_shared_modules first)."
            )
        weight = self.lm_head.weight
        base_logits = shared_lm_head_logits(
            hidden,
            self.lm_head,
            lambda x: torch.matmul(x.to(weight.dtype), weight.T),
            group=self.lm_head_group,
            dp_gather=self.lm_head_dp_gather,
            rows_capacity=dp_rows_capacity,
        )
        return base_logits, None

    def join_dp_vocab_idle(self, *, rows_capacity: int) -> None:
        """Idle-rank counterpart of the DP-gathered lm-head collectives: run
        the same sequence with zero local rows so the ranks that do have draft
        work are not left waiting. Only needed on the eager path -- a graph
        replay already brings every DP rank through the folded sampler."""
        if not self.lm_head_dp_gather:
            return
        hidden = torch.zeros(
            (0, int(self.lm_head.weight.shape[1])),
            dtype=self.lm_head.weight.dtype,
            device=self.lm_head.weight.device,
        )
        self.compute_base_logits(hidden, dp_rows_capacity=rows_capacity)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        markov_weights = []
        confidence_weights = []
        backbone_weights = []
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if any(name.startswith(p) for p in _DSPARK_SKIPPED_WEIGHT_PREFIXES):
                continue
            if name.startswith("confidence_head."):
                if self.confidence_head is None:
                    continue
                confidence_weights.append((name, loaded_weight))
            elif name.startswith("markov_head."):
                markov_weights.append((name, loaded_weight))
            else:
                backbone_weights.append((name, loaded_weight))

        super().load_weights(backbone_weights)

        for name, loaded_weight in markov_weights:
            if name not in params_dict:
                raise ValueError(
                    f"DSpark unexpected markov weight {name!r} not found in model "
                    f"parameters (known markov params require a {type(self.markov_head).__name__} head)."
                )
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)

        self._load_confidence_weights(
            confidence_weights=confidence_weights, params_dict=params_dict
        )

    def _load_confidence_weights(
        self,
        *,
        confidence_weights: list,
        params_dict: dict,
    ) -> None:
        if self.confidence_head is None:
            return
        loaded_names = set()
        for name, loaded_weight in confidence_weights:
            if name not in params_dict:
                raise ValueError(
                    f"DSpark unexpected confidence weight {name!r} not found in "
                    "model parameters."
                )
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_names.add(name)

        confidence_param_names = {
            name for name in params_dict if name.startswith("confidence_head.")
        }
        missing = confidence_param_names - loaded_names
        if missing:
            raise ValueError(
                f"DSpark confidence head is enabled but the checkpoint is missing "
                f"{sorted(missing)}. Provide a checkpoint with trained confidence weights, "
                f"or disable the confidence head (enable_confidence_head=False)."
            )

    def _stacked_ctx_kv_params(self) -> Optional[dict]:
        """Stack every layer's KV projection into one weight (exact: the input
        hidden is shared, so concatenating output columns is equivalent).
        Cached; None (per-layer fallback) when a QKV weight cannot be sliced
        (quantized) or layers disagree on norm epsilon / bias presence.
        """
        # NOTE: safe under runtime weight sharding because `qkv_proj` is never
        # a shard target (see DFlashDecoderLayer.weight_shard_targets), so the
        # weight read here is always the full, stable tensor.
        cached = getattr(self, "_stacked_ctx_kv_cache", False)
        if cached is not False:
            return cached
        weights, biases, k_norm_weights = [], [], []
        eps = None
        for layer in self.layers:
            attn = layer.self_attn
            can_slice, _ = can_dflash_slice_qkv_weight(attn.qkv_proj)
            if not can_slice or eps not in (None, attn.k_norm.variance_epsilon):
                self._stacked_ctx_kv_cache = None
                return None
            eps = attn.k_norm.variance_epsilon
            kv_slice = slice(attn.q_size, attn.q_size + 2 * attn.kv_size)
            weights.append(attn.qkv_proj.weight[kv_slice])
            biases.append(
                attn.qkv_proj.bias[kv_slice] if attn.qkv_proj.bias is not None else None
            )
            k_norm_weights.append(attn.k_norm.weight)
        has_bias = [b is not None for b in biases]
        if any(has_bias) and not all(has_bias):
            self._stacked_ctx_kv_cache = None
            return None
        self._stacked_ctx_kv_cache = {
            "weight": torch.cat(weights, dim=0),
            "bias": torch.cat(biases, dim=0) if all(has_bias) else None,
            "k_norm_weight": torch.stack(k_norm_weights, dim=0).float(),
            "eps": eps,
        }
        return self._stacked_ctx_kv_cache

    def write_target_hidden_kv(
        self,
        *,
        target_hidden: torch.Tensor,
        pool,
        positions: torch.Tensor,
        cache_loc: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor] = None,
        commit_lens: Optional[torch.Tensor] = None,
    ) -> None:
        ctx_hidden = self.project_target_hidden(target_hidden)
        stacked = self._stacked_ctx_kv_params()
        if stacked is not None:
            k_all, v_all = self._project_ctx_kv_stacked(
                ctx_hidden=ctx_hidden, positions=positions, stacked=stacked
            )
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            if stacked is not None:
                k = k_all[i]
                v = v_all[i]
            else:
                k, v = attn.kv_proj_only(ctx_hidden)
                k = attn.apply_k_norm(k)
                k = attn.apply_k_rope(positions, k)
                k = k.view(-1, attn.num_kv_heads, attn.head_dim)
                v = v.view(-1, attn.num_kv_heads, attn.head_dim)
            if cache_loc_2d is not None and commit_lens is not None:
                pool.set_kv_buffer_prefix_valid(
                    attn.attn,
                    cache_loc_2d,
                    commit_lens,
                    k,
                    v,
                    attn.attn.k_scale,
                    attn.attn.v_scale,
                )
            else:
                pool.set_kv_buffer(
                    attn.attn,
                    cache_loc,
                    k,
                    v,
                    attn.attn.k_scale,
                    attn.attn.v_scale,
                )

    def _project_ctx_kv_stacked(
        self,
        *,
        ctx_hidden: torch.Tensor,
        positions: torch.Tensor,
        stacked: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn0 = self.layers[0].self_attn
        num_layers = len(self.layers)
        kv_size = attn0.kv_size
        head_dim = attn0.head_dim
        num_kv_heads = attn0.num_kv_heads
        tokens = ctx_hidden.shape[0]

        kv_all = F.linear(ctx_hidden, stacked["weight"], stacked["bias"])
        kv_all = kv_all.view(tokens, num_layers, 2, kv_size)
        # Batched per-head k-norm across layers (fp32 variance + weight, cast back).
        k32 = (
            kv_all[:, :, 0, :]
            .reshape(tokens, num_layers, num_kv_heads, head_dim)
            .to(torch.float32)
        )
        variance = k32.pow(2).mean(dim=-1, keepdim=True)
        k32 = k32 * torch.rsqrt(variance + stacked["eps"])
        k32 = k32 * stacked["k_norm_weight"].view(1, num_layers, 1, head_dim)
        k_all = k32.to(ctx_hidden.dtype)
        # One RoPE over all layers' heads (shared rotary params + positions).
        k_flat = k_all.reshape(tokens, num_layers * kv_size)
        dummy_q = k_flat.new_empty(k_flat.shape)
        _, k_flat = attn0.rotary_emb(positions, dummy_q, k_flat)
        # [layers, tokens, heads, dim]: per-layer slices are contiguous views.
        k_all = (
            k_flat.view(tokens, num_layers, num_kv_heads, head_dim)
            .permute(1, 0, 2, 3)
            .contiguous()
        )
        v_all = (
            kv_all[:, :, 1, :]
            .view(tokens, num_layers, num_kv_heads, head_dim)
            .permute(1, 0, 2, 3)
            .contiguous()
        )
        return k_all, v_all


class DSparkDraftModel(DSparkDraftMixin, DFlashDraftModel):

    def prune_to_ctx_kv_injection(self) -> None:
        self.markov_head = None
        self.confidence_head = None
        for layer in self.layers:
            layer.mlp = None
            layer.self_attn.o_proj = None
        if self._weight_shard_layers:
            # Only the pruned modules were shard targets (qkv stays replicated
            # for the KV-injection path), and the pruned draft never runs a
            # forward, so drop the shard bookkeeping to release the per-layer
            # shards and the staging pool.
            for layer in self.layers:
                for _role, sharded_attr, _path in getattr(
                    layer, "_weight_shard_targets", []
                ):
                    if hasattr(layer, sharded_attr):
                        delattr(layer, sharded_attr)
                for attr in ("_weight_shard_sched", "_weight_shard_targets"):
                    if hasattr(layer, attr):
                        delattr(layer, attr)
            self._weight_shard_layers = []
            self._weight_shard_planner = WeightShardForwardPlanner([])
            self._weight_shard_pool = None
            # Drop the host-side AG-0 prefetch too; its scheduler was just
            # deleted along with the layer bookkeeping.
            self._external_weight_shard_scheds = []
            self._weight_shard_prefetched = False
            self._weight_shard_gate = None
        torch.cuda.empty_cache()


class Qwen3DSparkModel(DSparkDraftModel):
    pass


EntryClass = [Qwen3DSparkModel, DSparkDraftModel]
