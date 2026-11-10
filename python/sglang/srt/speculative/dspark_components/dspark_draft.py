from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Optional

import msgspec
import torch

from sglang.kernels.ops.speculative.dspark.dspark_draft_model import (
    SampleStepTokens,
)
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import vocab_dp_static_rows
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    enable_num_token_non_padded,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.draft_window import (
    compact_draft_seq_lens,
    fill_compact_seq_lens_cpu_bound,
    rebuild_compact_draft_req_to_token,
)
from sglang.srt.speculative.draft_worker_common import make_draft_input_v2
from sglang.srt.speculative.dspark_components.dspark_planner import VerifyWindow
from sglang.srt.speculative.spec_info import (
    SpeculativeAlgorithm,
    spec_scale_global_num_tokens,
)
from sglang.srt.speculative.spec_utils import draft_tp_context
from sglang.srt.utils.invariants import Bucket, Invariant, NotNaN, expect

logger = logging.getLogger(__name__)


def _one_hot_token0(probs: torch.Tensor) -> torch.Tensor:
    degenerate = torch.isnan(probs[:, :1])
    one_hot = torch.zeros_like(probs)
    one_hot[:, 0] = 1.0
    return torch.where(degenerate, one_hot, probs)


# Draft step logits: NaN is a bug, but -inf is legitimate (masking). The data
# layer lives downstream (probs one-hot below, or the fast kernel's clamp).
_DRAFT_STEP_LOGITS = Invariant("dspark.draft.step_logits", Bucket.GUARD, NotNaN())
# Draft sampling probs: SOFTEN (tolerate + count), matching the original
# unconditional clamp; an all-NaN row would otherwise make multinomial raise.
_DRAFT_PROBS = Invariant(
    "dspark.draft.probs", Bucket.SOFTEN, NotNaN(), recover=_one_hot_token0
)


class DraftBlockResult(msgspec.Struct, frozen=True):
    draft_tokens: torch.Tensor
    corrected_logits: Optional[torch.Tensor]
    greedy_mask: torch.Tensor
    temperatures: torch.Tensor


class DraftForwardResult(msgspec.Struct, frozen=True):
    draft_block_ids: torch.Tensor
    raw_hidden: torch.Tensor
    draft_hidden_3d: torch.Tensor
    can_run_graph: bool


class DraftProposal(msgspec.Struct, frozen=True):
    draft_block_ids: torch.Tensor
    draft_block: DraftBlockResult
    draft_hidden: Optional[torch.Tensor]
    confidence: Optional[torch.Tensor] = None
    confidence_tap: Optional[torch.Tensor] = None
    folded: bool = False


class DsparkDraftSampler:
    """Graph-folded draft proposal head (base logits + markov chain sampling).

    Captured as a draft-graph tail hook. Handles greedy and sampling rows in
    one pass: the per-step fused kernel (SampleStepTokens) argmaxes greedy
    rows and Gumbel-samples the rest with in-graph philox noise (CUDAGraph
    advances the generator per replay). Per-step sampling params live in
    static buffers refreshed by stage_sampling_params before each replay;
    corrected block logits are exported for the sampling accept path.
    """

    def __init__(self, *, model, gamma, max_bs, device, confidence_fn=None, out=None):
        self.model = model
        self.markov_head = model.markov_head
        self.gamma = int(gamma)
        max_bs = int(max_bs)
        if out is not None:
            assert out.shape == (max_bs * self.gamma,) and out.dtype == torch.int64
            self.out = out
        else:
            self.out = torch.empty(
                (max_bs * self.gamma,), dtype=torch.int64, device=device
            )
        self.confidence_fn = confidence_fn
        self.confidence_out = (
            torch.empty((max_bs, self.gamma), dtype=torch.float32, device=device)
            if confidence_fn is not None
            else None
        )
        # DP-gathered lm head: pad to the LARGEST captured row count instead of
        # this graph's bucket, so every DP rank enters the row gather with the
        # same shape whether it replays a graph, runs eager, or joins idle. A
        # per-bucket capacity would desync the moment two ranks take different
        # paths in the same step (which is what an uneven request burst does).
        self.dp_rows_capacity = (
            max_bs * self.gamma
            if getattr(model, "lm_head_dp_gather", False)
            else None
        )
        vocab = int(model.lm_head.org_vocab_size)
        self.temperatures = torch.ones((max_bs,), dtype=torch.float32, device=device)
        self.greedy_mask = torch.ones((max_bs,), dtype=torch.bool, device=device)
        self.exp_noise = torch.empty(
            (max_bs, vocab), dtype=torch.float32, device=device
        )
        self.corrected_out = torch.empty(
            (max_bs * self.gamma, vocab),
            dtype=model.lm_head.weight.dtype,
            device=device,
        )

    def stage_sampling_params(self, *, bs: int, sampling_info) -> None:
        """Host-side refresh of the static sampling params; must run before
        the draft graph replay that consumes them."""
        if sampling_info is None:
            self.temperatures[:bs].fill_(1.0)
            self.greedy_mask[:bs].fill_(True)
            return
        torch.clamp(
            sampling_info.temperatures.view(-1)[:bs].to(torch.float32),
            min=1e-5,
            out=self.temperatures[:bs],
        )
        self.greedy_mask[:bs].copy_((sampling_info.top_ks <= 1).view(-1)[:bs])

    def __call__(self, hidden_states, input_ids):
        bs = hidden_states.shape[0] // self.gamma
        base_logits, confidence_tap = self.model.compute_base_logits(
            hidden_states, dp_rows_capacity=self.dp_rows_capacity
        )
        base_logits = base_logits.view(bs, self.gamma, -1)
        anchor = input_ids.view(bs, self.gamma)[:, 0]

        def _step_sampler(step_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
            del step_idx
            noise = self.exp_noise[:bs].exponential_()
            return SampleStepTokens.execute(
                step_logits=step_logits,
                temperatures=self.temperatures[:bs],
                greedy_mask=self.greedy_mask[:bs],
                exp_noise=noise,
            )

        draft_tokens, corrected_logits = self.markov_head.sample_block(
            base_logits,
            first_prev_tokens=anchor,
            hidden_states=hidden_states.view(bs, self.gamma, -1),
            sampler=_step_sampler,
        )
        self.out[: draft_tokens.numel()].copy_(draft_tokens.reshape(-1))
        self.corrected_out[: bs * self.gamma].copy_(
            corrected_logits.reshape(bs * self.gamma, -1)
        )
        if self.confidence_out is not None:
            confidence = self.confidence_fn(
                draft_hidden=hidden_states.view(bs, self.gamma, -1),
                anchor_tokens=anchor,
                draft_tokens=draft_tokens,
                confidence_tap=confidence_tap,
            )
            self.confidence_out[:bs].copy_(confidence)


def maybe_build_draft_sampler(
    *,
    draft_model,
    gamma: int,
    max_bs: int,
    device,
    tp_rank: int,
    confidence_fn=None,
    out=None,
) -> Optional[DsparkDraftSampler]:
    """Build the graph-folded draft sampler (greedy + sampling rows), or
    return None (with the reason logged) when the draft model cannot support
    folding and the proposal must stay eager."""

    def _eager(reason):
        if tp_rank == 0:
            logger.info("DSpark draft proposal kept eager (reason=%s).", reason)
        return None

    if gamma <= 0:
        return _eager("gamma<=0")
    if not hasattr(draft_model, "compute_base_logits"):
        return _eager("no compute_base_logits")
    if getattr(draft_model, "markov_head", None) is None:
        return _eager("no markov head")
    if tp_rank == 0:
        logger.info(
            "DSpark draft proposal (greedy + sampling) folded into the draft "
            "cuda graph."
        )
    return DsparkDraftSampler(
        model=draft_model,
        gamma=gamma,
        max_bs=max_bs,
        device=device,
        confidence_fn=confidence_fn,
        out=out,
    )


def make_next_draft_input(
    *,
    bonus_tokens: torch.Tensor,
    new_seq_lens: torch.Tensor,
) -> DFlashDraftInputV2:
    return make_draft_input_v2(bonus_tokens=bonus_tokens, new_seq_lens=new_seq_lens)


def resolve_greedy_mask(
    *,
    bs: int,
    sampling_info,
    device: torch.device,
) -> torch.Tensor:
    if sampling_info is None:
        return torch.ones(bs, dtype=torch.bool, device=device)
    return (sampling_info.top_ks <= 1).view(-1)


def sample_draft_block(
    *,
    base_logits: torch.Tensor,
    anchor_tokens: torch.Tensor,
    draft_hidden: torch.Tensor,
    sampling_info,
    markov_head,
    device: torch.device,
) -> DraftBlockResult:
    bs = base_logits.shape[0]
    greedy_mask = resolve_greedy_mask(bs=bs, sampling_info=sampling_info, device=device)
    any_sampling = sampling_info is not None and not sampling_info.is_all_greedy
    fast_sampling = envs.SGLANG_DSPARK_FAST_SAMPLING.get()

    if sampling_info is None:
        temperatures = torch.ones(bs, dtype=torch.float32, device=device)
    else:
        temperatures = (
            sampling_info.temperatures.view(-1).to(torch.float32).clamp_min(1e-5)
        )

    if not any_sampling:

        def sampler(step_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
            expect(_DRAFT_STEP_LOGITS, step_logits, msg=f"step {step_idx}")
            return torch.argmax(step_logits, dim=-1)

    else:

        def sampler(step_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
            expect(_DRAFT_STEP_LOGITS, step_logits, msg=f"step {step_idx}")
            if fast_sampling:
                exp_noise = torch.empty(
                    step_logits.shape, dtype=torch.float32, device=step_logits.device
                ).exponential_(1)
                return SampleStepTokens.execute(
                    step_logits=step_logits,
                    temperatures=temperatures,
                    greedy_mask=greedy_mask,
                    exp_noise=exp_noise,
                )
            else:
                probs = torch.softmax(
                    step_logits.float() / temperatures[:, None], dim=-1
                )
                probs = expect(_DRAFT_PROBS, probs)
                argmax_tokens = torch.argmax(step_logits, dim=-1)
                sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                return torch.where(greedy_mask, argmax_tokens, sampled_tokens)

    draft_tokens, corrected_logits = markov_head.sample_block(
        base_logits,
        first_prev_tokens=anchor_tokens,
        hidden_states=draft_hidden,
        sampler=sampler,
    )
    return DraftBlockResult(
        draft_tokens=draft_tokens,
        corrected_logits=corrected_logits,
        greedy_mask=greedy_mask,
        temperatures=temperatures,
    )


class DraftBlockProposer:
    def __init__(
        self,
        *,
        draft_model,
        draft_model_runner,
        gamma: int,
        mask_token_id: int,
        draft_block_spec_info,
        dp_moe_sync: bool = False,
        dp_vocab_sync: bool = False,
        dp_graph_rows: Optional[int] = None,
        draft_window_size: Optional[int] = None,
        page_size: int = 1,
        resolve_target_req_to_token=None,
        resolve_slot_translator=None,
    ) -> None:
        self.draft_model = draft_model
        self.draft_model_runner = draft_model_runner
        self.gamma = gamma
        self._mask_token_id = mask_token_id
        self._draft_block_spec_info = draft_block_spec_info
        self._draft_sampler = None
        self._dp_moe_sync = dp_moe_sync
        # The shared target lm_head shards its vocab over the global TP group
        # (dp attention without --enable-dp-lm-head): the base-logits path then
        # runs cross-DP collectives, so every DP rank has to join and the draft
        # batch needs the DP sync metadata.
        self._dp_vocab_sync = dp_vocab_sync
        self._dp_sync = dp_moe_sync or dp_vocab_sync
        # Largest captured row count: the capacity every DP-gathered vocab
        # collective pads to whenever the batch fits it, so a graph replay (which
        # bakes it) and an eager/idle join always agree.
        self._dp_graph_rows = dp_graph_rows
        # Last observed graph-vs-eager mode of the draft forward on each path;
        # logged on change so busy/idle divergence is visible in the log rather
        # than inferred.
        self._idle_participation_mode: Optional[bool] = None
        self._busy_forward_mode: Optional[bool] = None
        # Draft-side SWA. The req_to_token tensors do not exist yet at
        # construction (pools are allocated later), so the target table is
        # resolved lazily -- same pattern as DsparkVerifyEpilogue's commit ctx.
        self._draft_window_size = draft_window_size
        self._page_size = int(page_size)
        self._resolve_target_req_to_token = resolve_target_req_to_token
        # Resolved per call, not captured: the slot map is built later, in
        # alloc_memory_pool. Returns None when windowing is off, which keeps the
        # compact row (and the KV writes) in the target-global space.
        self._resolve_slot_translator = resolve_slot_translator
        self._compact = draft_window_size is not None
        if self._compact and resolve_target_req_to_token is None:
            raise ValueError(
                "DSpark draft windowing requires resolve_target_req_to_token."
            )
        # Host-side planning bound for the compact seq_lens; grown on demand.
        self._compact_seq_lens_cpu_buf: Optional[torch.Tensor] = None

    def attach_draft_sampler(self, draft_sampler) -> None:
        self._draft_sampler = draft_sampler

    def _compact_draft_lengths(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        verify_cache_loc_2d: torch.Tensor,
        bs: int,
        device: str,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Rebuild the draft-local compact window view and return
        ``(draft_seq_lens, draft_seq_lens_cpu, draft_seq_lens_sum)``.

        The compact row is ``[recent window slots][this step's block slots]``, so
        draft attention structurally cannot reach keys outside the window. The
        row is fully rewritten every step before the backend reads it, which is
        why the private table needs no reclamation.
        """
        assert self._draft_window_size is not None
        prefix_lens = batch.seq_lens
        draft_prefix_lens = compact_draft_seq_lens(
            seq_lens=prefix_lens,
            window_size=self._draft_window_size,
            page_size=self._page_size,
            device=device,
        )
        if (
            self._compact_seq_lens_cpu_buf is None
            or self._compact_seq_lens_cpu_buf.numel() < bs
        ):
            self._compact_seq_lens_cpu_buf = torch.empty(
                (bs,), dtype=torch.int32, device="cpu"
            )
        seq_lens_cpu = self._compact_seq_lens_cpu_buf[:bs]
        fill_compact_seq_lens_cpu_bound(
            batch_seq_lens_cpu=batch.seq_lens_cpu,
            reserved_seq_lens_cpu=draft_input.reserved_seq_lens_cpu,
            draft_prefix_lens=draft_prefix_lens,
            window_size=self._draft_window_size,
            page_size=self._page_size,
            out=seq_lens_cpu,
        )
        # Host-known bound: compact_draft_seq_lens never exceeds
        # window + page (page alignment keeps up to page_size-1 extra tokens),
        # and the live row adds the block. Avoids the exact device-side
        # ``.max()`` readback, which is a D2H sync on the decode hot path.
        span_bound = (
            self._draft_window_size
            + (self._page_size if self._page_size > 1 else 0)
            + self.gamma
        )
        rebuild_compact_draft_req_to_token(
            draft_req_to_token=self.draft_model_runner.req_to_token_pool.req_to_token,
            target_req_to_token=self._resolve_target_req_to_token(),
            req_pool_indices=batch.req_pool_indices,
            prefix_lens=prefix_lens,
            draft_prefix_lens=draft_prefix_lens,
            verify_cache_loc_2d=verify_cache_loc_2d,
            bs=bs,
            block_size=self.gamma,
            slot_translator=(
                None
                if self._resolve_slot_translator is None
                else self._resolve_slot_translator()
            ),
            max_span=span_bound,
        )
        # Match the non-compact convention: seq_lens_cpu is the committed length
        # plus this step's block, as a safe upper bound for backend planning.
        seq_lens_cpu.add_(self.gamma)
        return draft_prefix_lens, seq_lens_cpu, int(seq_lens_cpu.sum())

    def _base_logits_context(self):
        if self._dp_moe_sync:
            return draft_tp_context(get_parallel().attn_tp_group)
        return nullcontext()

    def propose(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        verify_window: VerifyWindow,
        bs: int,
        device: str,
        target_model,
        sampling_info,
    ) -> DraftProposal:
        embed_module = target_model.get_input_embeddings()
        draft_sampler = self._draft_sampler
        all_greedy = sampling_info is None or sampling_info.is_all_greedy
        fwd = self._run_forward(
            batch=batch,
            draft_input=draft_input,
            verify_window=verify_window,
            bs=bs,
            device=device,
            embed_module=embed_module,
            draft_sampler=draft_sampler,
            sampling_info=sampling_info,
        )
        draft_block_ids = fwd.draft_block_ids

        folded_confidence = None
        confidence_tap = None
        folded = False
        if draft_sampler is not None and fwd.can_run_graph:
            folded = True
            draft_block = DraftBlockResult(
                draft_tokens=draft_sampler.out[: bs * self.gamma].view(bs, self.gamma),
                # The sampling accept path needs the markov-corrected block
                # logits; greedy accept only compares tokens.
                corrected_logits=(
                    None
                    if all_greedy
                    else draft_sampler.corrected_out[: bs * self.gamma].view(
                        bs, self.gamma, -1
                    )
                ),
                greedy_mask=draft_sampler.greedy_mask[:bs],
                temperatures=draft_sampler.temperatures[:bs],
            )
            if draft_sampler.confidence_out is not None:
                folded_confidence = draft_sampler.confidence_out[:bs]
        else:
            rows_capacity = self._dp_rows_capacity(batch)
            if (
                getattr(self.draft_model, "lm_head_dp_gather", False)
                and rows_capacity is None
            ):
                raise RuntimeError(
                    "DSpark's DP-gathered lm head needs batch.global_num_tokens to "
                    "agree on a per-rank row capacity; without it the ranks would "
                    "enter the row gather with different shapes and deadlock."
                )
            with self._base_logits_context():
                base_logits, confidence_tap = self.draft_model.compute_base_logits(
                    fwd.raw_hidden,
                    dp_rows_capacity=rows_capacity,
                )
                base_logits = base_logits.view(bs, self.gamma, -1)
            draft_block = sample_draft_block(
                base_logits=base_logits,
                anchor_tokens=draft_block_ids[:, 0],
                draft_hidden=fwd.draft_hidden_3d,
                sampling_info=sampling_info,
                markov_head=self.draft_model.markov_head,
                device=device,
            )
        return DraftProposal(
            draft_block_ids=draft_block_ids,
            draft_block=draft_block,
            draft_hidden=fwd.draft_hidden_3d,
            confidence=folded_confidence,
            confidence_tap=confidence_tap,
            folded=folded,
        )

    def run_idle_participation(self, batch: ScheduleBatch) -> None:
        # A weight-sharded dense draft also needs idle participation: its
        # per-layer flat AllGather runs on the node-local shard group, so every
        # rank must enter the draft forward. It MUST take the same graph-vs-
        # eager branch as busy ranks, so it goes through the bs=1 dummy-tier
        # path below -- the generic bs=0 IDLE forward underneath cannot replay
        # and would be the only eager AG producer while every busy rank replays.
        draft_weight_sharded = bool(
            getattr(self.draft_model, "_weight_shard_layers", None)
        )
        if draft_weight_sharded:
            self._run_weight_shard_idle_participation(batch)
            return
        if not self._dp_sync or batch.global_num_tokens is None:
            return
        device = self.draft_model_runner.device
        empty_long = torch.empty((0,), dtype=torch.int64, device=device)
        idle_batch = ForwardBatch(
            forward_mode=ForwardMode.IDLE,
            batch_size=0,
            input_ids=empty_long,
            req_pool_indices=empty_long,
            seq_lens=empty_long,
            out_cache_loc=empty_long,
            seq_lens_sum=0,
            seq_lens_cpu=torch.empty((0,), dtype=torch.int64),
            positions=empty_long,
            spec_algorithm=SpeculativeAlgorithm.DSPARK,
            spec_info=self._draft_block_spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        self._fill_dp_moe_sync_metadata(idle_batch, batch)
        self._publish_dp_embed_rows_capacity(batch)
        with torch.inference_mode():
            draft_out = self.draft_model_runner.forward(idle_batch)
        if self._dp_vocab_sync and not draft_out.can_run_graph:
            # A graph replay already ran the folded sampler (hence the
            # base-logits collectives) on this rank; the eager path did not.
            with torch.inference_mode(), self._base_logits_context():
                self.draft_model.join_dp_vocab_idle(
                    rows_capacity=self._dp_rows_capacity(batch)
                )

    def _publish_dp_embed_rows_capacity(self, batch: ScheduleBatch) -> None:
        """The draft's embed runs INSIDE the model forward (and inside the draft
        graph), so it reads its row capacity off the model. A graph replay baked
        the capture-time value; refresh it for the eager path from the same
        DP-synced rule the lm-head path uses."""
        if not getattr(self.draft_model, "embed_dp_gather", False):
            return
        capacity = self._dp_rows_capacity(batch)
        if capacity is not None:
            self.draft_model.dp_embed_rows_capacity = capacity

    def _dp_rows_capacity(self, batch: ScheduleBatch) -> Optional[int]:
        """Per-rank row capacity of the DP-gathered vocab collectives (embed and
        lm head).

        Must be identical on every attention-DP rank in a step, including the
        idle ones, or the ranks enter the row gather with different shapes and
        NCCL deadlocks. So it is derived only from the DP-synced sync values and
        prefers the largest captured row count -- that is what a graph replay
        bakes, so eager and idle joins have to match it. Above that capacity no
        rank can be on the graph path, and the raw DP max is agreed on by
        everyone.

        A static capacity short-circuits all of that: it is a constant, so it
        needs no DP-synced values at all, which is what makes this path work
        under SGLANG_SCHEDULER_SKIP_ALL_GATHER (where ``global_num_tokens`` holds
        the LOCAL count and ``max(gnt)`` would differ per rank).
        """
        if not self._dp_vocab_sync:
            return None
        static_rows = vocab_dp_static_rows()
        if static_rows:
            return static_rows
        if batch.global_num_tokens is None:
            return None
        gnt, _ = spec_scale_global_num_tokens(
            self._draft_block_spec_info,
            batch.global_num_tokens,
            batch.global_num_tokens_for_logprob,
        )
        rows = max(gnt)
        graph_rows = self._dp_graph_rows
        if graph_rows is None:
            graph_rows = getattr(self._draft_sampler, "dp_rows_capacity", None)
        if graph_rows is not None and rows <= graph_rows:
            return int(graph_rows)
        return rows

    def _run_weight_shard_idle_participation(self, batch: ScheduleBatch) -> None:
        """Idle participation for a weight-sharded draft, shaped like a real
        decode step so it takes the same graph-vs-eager branch as busy ranks.

        Mirrors `TargetVerifyExecutor.run_idle_participation`, which sizes its
        dummy verify batch to a captured tier for exactly this reason. A bs=0
        IDLE batch cannot replay (`ForwardMode.IDLE.is_cuda_graph()` is False
        and no bs=0 tier is captured), so it would be the only eager producer
        of the shard-group AllGather while every busy rank replays a graph.
        """
        gamma = self.gamma
        device = self.draft_model_runner.device
        if not hasattr(self.draft_model, "forward_embed"):
            # Without a draft-owned embedding we would need the target's embed
            # module to build input_embeds, which this call site does not have.
            raise RuntimeError(
                "Weight-sharded draft idle participation requires a draft model "
                "that owns its embedding (forward_embed)."
            )
        # bs=1 is the smallest captured draft decode tier; req/cache indices
        # point at slot 0, the same dummy slot the target's idle verify uses.
        bs = 1
        num_tokens = bs * gamma
        idle_forward_batch = self._build_draft_forward_batch(
            batch=batch,
            bs=bs,
            device=device,
            input_ids=torch.full(
                (num_tokens,), int(self._mask_token_id), dtype=torch.long, device=device
            ),
            req_pool_indices=torch.zeros((bs,), dtype=torch.int64, device=device),
            seq_lens=torch.ones((bs,), dtype=torch.int64, device=device),
            out_cache_loc=torch.zeros((num_tokens,), dtype=torch.int64, device=device),
            positions=torch.arange(1, num_tokens + 1, dtype=torch.int64, device=device),
            seq_lens_cpu=torch.full((bs,), 1 + gamma, dtype=torch.int64),
            seq_lens_sum=bs * (1 + gamma),
            input_embeds=None,
        )
        draft_out = self._forward_draft_model(idle_forward_batch)
        self._log_idle_participation_mode(bool(draft_out.can_run_graph))

    def _forward_draft_model(self, forward_batch: ForwardBatch):
        """Run the draft forward, joining the prefetched AG 0.

        AG 0 is launched by the worker on a dedicated stream late in the
        previous step (`prefetch_weight_shard`, right before the ReplaySSM
        commit fold), so here we only wait for it. Busy decode and idle
        participation both go through this, which keeps the per-step AG
        sequence identical on every rank of the shard group.
        """
        wait = getattr(self.draft_model, "wait_first_weight_shard", None)
        if wait is not None:
            wait()
        with torch.inference_mode():
            draft_out = self.draft_model_runner.forward(forward_batch)
        return draft_out

    def prefetch_weight_shard(self) -> None:
        """Launch AG 0 for the next forward (called by the worker).

        Kept on the proposer so the draft model stays the only owner of the
        shard bookkeeping; the worker decides *when* in the step to call it.
        """
        prefetch = getattr(self.draft_model, "prefetch_first_weight_shard", None)
        if prefetch is not None:
            prefetch()

    def _log_idle_participation_mode(self, can_run_graph: bool) -> None:
        """Report the mode idle participation actually took, once per state.

        The previous fix attempt inferred the mode instead of measuring it and
        was wrong (the batch still replayed), so record the observed value.
        """
        if self._idle_participation_mode == can_run_graph:
            return
        self._idle_participation_mode = can_run_graph
        logger.info(
            "[K3WeightShard] draft idle participation can_run_graph=%s "
            "(must match the busy ranks' draft forward)",
            can_run_graph,
        )

    def _log_busy_forward_mode(self, can_run_graph: bool) -> None:
        """Same for the busy path, so the two can be compared in one log."""
        if self._busy_forward_mode == can_run_graph:
            return
        self._busy_forward_mode = can_run_graph
        logger.info(
            "[K3WeightShard] draft busy forward can_run_graph=%s "
            "(must match idle participation)",
            can_run_graph,
        )

    def _run_forward(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        verify_window: VerifyWindow,
        bs: int,
        device: str,
        embed_module,
        draft_sampler=None,
        sampling_info=None,
    ) -> DraftForwardResult:
        gamma = self.gamma
        prefix_lens = batch.seq_lens
        positions_2d = verify_window.positions_2d
        verify_cache_loc_2d = verify_window.verify_cache_loc_2d

        draft_block_ids = torch.full(
            (bs, gamma), int(self._mask_token_id), dtype=torch.long, device=device
        )
        draft_block_ids[:, 0].copy_(draft_input.bonus_tokens.view(-1))
        draft_positions = positions_2d[:, :gamma].reshape(-1)
        draft_cache_loc = verify_cache_loc_2d[:, :gamma].reshape(-1)
        # SWA writes must land at DRAFT-swa ids, matching the compact-row page
        # table the SAME forward reads back. Without this, draft self-attention
        # writes K/V at target-global slot G but reads at swa slot S=translate(G)
        # -- the read is garbage, block-internal predictions collapse, and
        # accept_len drops to ~1 (only the pre-committed bonus token survives).
        if self._compact and self._resolve_slot_translator is not None:
            translator = self._resolve_slot_translator()
            if translator is not None:
                draft_cache_loc = translator(draft_cache_loc).to(
                    dtype=draft_cache_loc.dtype
                )

        draft_owns_embed = hasattr(self.draft_model, "forward_embed")
        draft_input_embeds: Optional[torch.Tensor] = None
        if not draft_owns_embed:
            noise_embedding = embed_module(draft_block_ids)
            draft_input_embeds = noise_embedding.view(-1, noise_embedding.shape[-1])

        if self._compact:
            # Windowed: seq_lens is the compact row length, not the true prefix.
            # draft_positions above stays global-real so RoPE is unaffected.
            draft_seq_lens, draft_seq_lens_cpu, draft_seq_lens_sum = (
                self._compact_draft_lengths(
                    batch=batch,
                    draft_input=draft_input,
                    verify_cache_loc_2d=verify_cache_loc_2d,
                    bs=bs,
                    device=device,
                )
            )
        elif batch.seq_lens_cpu is not None:
            draft_seq_lens = prefix_lens
            draft_seq_lens_cpu = batch.seq_lens_cpu + gamma
            draft_seq_lens_sum = int(draft_seq_lens_cpu.sum())
        elif draft_input.reserved_seq_lens_cpu is not None:
            draft_seq_lens = prefix_lens
            draft_seq_lens_cpu = draft_input.reserved_seq_lens_cpu
            draft_seq_lens_sum = int(draft_input.reserved_seq_lens_sum)
        else:
            raise RuntimeError("DSpark decode expected batch.seq_lens_cpu, got None")

        draft_forward_batch = self._build_draft_forward_batch(
            batch=batch,
            bs=bs,
            device=device,
            input_ids=draft_block_ids.flatten(),
            req_pool_indices=batch.req_pool_indices,
            seq_lens=draft_seq_lens,
            out_cache_loc=draft_cache_loc,
            positions=draft_positions,
            seq_lens_cpu=draft_seq_lens_cpu,
            seq_lens_sum=draft_seq_lens_sum,
            input_embeds=draft_input_embeds,
        )
        graph_runner = self.draft_model_runner.decode_cuda_graph_runner
        if (
            draft_sampler is not None
            and graph_runner is not None
            and graph_runner.can_run_graph(draft_forward_batch)
        ):
            draft_sampler.stage_sampling_params(bs=bs, sampling_info=sampling_info)
        draft_out = self._forward_draft_model(draft_forward_batch)
        logits_output = draft_out.logits_output
        raw_hidden = logits_output.hidden_states
        if raw_hidden is None:
            raise RuntimeError("DSpark draft model returned no hidden states.")
        draft_hidden_3d = raw_hidden.view(bs, gamma, -1)
        if getattr(self.draft_model, "_weight_shard_layers", None):
            self._log_busy_forward_mode(bool(draft_out.can_run_graph))
        return DraftForwardResult(
            draft_block_ids=draft_block_ids,
            raw_hidden=raw_hidden,
            draft_hidden_3d=draft_hidden_3d,
            can_run_graph=draft_out.can_run_graph,
        )

    def _build_draft_forward_batch(
        self,
        *,
        batch: ScheduleBatch,
        bs: int,
        device: str,
        input_ids: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        positions: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        seq_lens_sum: int,
        input_embeds: Optional[torch.Tensor],
    ) -> ForwardBatch:
        """Build the draft ForwardBatch shared by the busy and idle paths.

        Busy decode and idle participation MUST produce structurally identical
        batches: the draft's per-layer flat AllGather is launched from inside
        `DFlashDraftModel.forward`, so if one rank replays a cuda graph while
        another runs eager, the two disagree on both the AG submission order
        and on the ownership of `FlatAGScheduler._shared_event` (a graph event
        node vs. an eager record on the same cudaEvent). That mismatch stalls
        the shard group. Routing both paths through this builder keeps replay
        admission (`can_run_graph`) reading the same fields on both sides.
        """
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens_sum,
            seq_lens_cpu=seq_lens_cpu,
            positions=positions,
            input_embeds=input_embeds,
            spec_algorithm=SpeculativeAlgorithm.DSPARK,
            spec_info=self._draft_block_spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            out_cache_loc_is_physical=self._compact,
        )
        self._fill_dp_moe_sync_metadata(forward_batch, batch)
        self._publish_dp_embed_rows_capacity(batch)
        if not self._dp_moe_sync:
            # Dense (dp-local) draft: under --enable-dp-attention the draft's
            # graph runner keeps require_mlp_sync, so replay admission reads
            # can_run_dp_cuda_graph off this hand-built batch; without it the
            # captured draft graphs never replay. The dense draft's only
            # cross-DP collective is the DP-gathered lm head (dp_vocab_sync),
            # which pads to the same captured bucket on every rank, so the
            # scheduler's global decode-sync flag is a safe gate.
            # num_token_non_padded mirrors ForwardBatch.init_new; the replay
            # hook localizes it per attn-TP rank when the gathered-buffer path
            # is active.
            forward_batch.can_run_dp_cuda_graph = batch.can_run_dp_cuda_graph
            if enable_num_token_non_padded():
                num_tokens = int(forward_batch.input_ids.numel())
                forward_batch.num_token_non_padded = torch.tensor(
                    num_tokens, dtype=torch.int32
                ).to(device, non_blocking=True)
                forward_batch.num_token_non_padded_cpu = num_tokens
        return forward_batch

    def _fill_dp_moe_sync_metadata(
        self, forward_batch: ForwardBatch, batch: ScheduleBatch
    ) -> None:
        if not self._dp_sync or batch.global_num_tokens is None:
            return
        gnt, gnt_logprob = spec_scale_global_num_tokens(
            self._draft_block_spec_info,
            batch.global_num_tokens,
            batch.global_num_tokens_for_logprob,
        )
        device = self.draft_model_runner.device
        forward_batch.global_num_tokens_cpu = gnt
        forward_batch.global_num_tokens_for_logprob_cpu = gnt_logprob
        forward_batch.global_num_tokens_gpu = torch.tensor(gnt, dtype=torch.int64).to(
            device, non_blocking=True
        )
        forward_batch.global_num_tokens_for_logprob_gpu = torch.tensor(
            gnt_logprob, dtype=torch.int64
        ).to(device, non_blocking=True)
        forward_batch.can_run_dp_cuda_graph = batch.can_run_dp_cuda_graph
