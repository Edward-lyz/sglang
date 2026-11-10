"""Zero-config per-token FLOPs estimation for the dynamic-chunking cost model.

Instead of asking users to pre-compute per-token FLOPs constants (the "A" in
the per_token_flops formulas of aiak_ds_tool simulator_v2), walk the weights
that are actually loaded on this rank. This automatically reflects:

- PP/TP sharding: the rank only holds its own stage/shard, so the walk yields
  the per-stage share directly, no manual "/ pp_size" bookkeeping;
- quantization: packed 4-bit weights are unpacked to element counts, so
  switching e.g. routed experts to mxfp4 needs no user action.

Counting convention matches the simulator_v2 MFU formulas: 2 FLOPs per
activated weight element per token. Sequence-length dependent work (MHA core,
MLA kv_b_proj re-expansion) is priced separately by ChunkCostModel's k_a/k_e
terms, so kv_b_proj weights are excluded here to avoid double counting.

FLOPs are counted per dtype-agnostic element: chunk admission compares stages
and operating points against a budget derived from the same counts, so a
per-dtype throughput correction (fp8 GEMMs running ~2x bf16) would only matter
if it differed *between* the k_g/k_e/k_a terms. Pricing it would need measured
per-dtype rates, not a hardware name table, so it is deliberately not modeled.
"""

import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Substrings of parameter names that are not per-token GEMM work:
# embeddings are lookups; lm_head only runs on the last token of each
# request during prefill; kv_b_proj is covered by the k_e re-expansion term.
_SKIP_NAME_PARTS = ("embed", "lm_head", "kv_b_proj", "rotary")

# Keys of the per-stage constants produced by the weight walk.
STAGE_KEYS = ("flops", "mla_layers")


def _moe_topk_and_num_experts(hf_config) -> Tuple[int, int]:
    topk = 0
    for attr in ("num_experts_per_token", "num_experts_per_tok", "moe_topk", "topk"):
        val = getattr(hf_config, attr, None)
        if isinstance(val, int) and val > 0:
            topk = val
            break
    num_experts = 0
    for attr in ("n_routed_experts", "num_experts", "expert_num"):
        val = getattr(hf_config, attr, None)
        if isinstance(val, int) and val > 0:
            num_experts = val
            break
    return topk, num_experts


def resolve_llm_config(hf_config):
    """Multimodal configs (e.g. Kimi-K3) nest all LLM fields under
    text_config; resolve to the inner config so attribute reads work."""
    return getattr(hf_config, "text_config", None) or hf_config


def _logical_elems_per_stored_elem(name: str, param) -> float:
    """How many logical weight values one stored element holds.

    A byte-typed tensor is ambiguous on its own: `uint8` carries two 4-bit
    values under mxfp4/int4, but exactly one under an 8-bit scheme that stores
    its bytes untyped. Guessing from the dtype alone silently doubles that
    stage's FLOPs, and because stages can carry different quantization the
    error does not cancel in the max-over-stages normalization.

    So the packing factor is taken from the parameter when it declares one, and
    only falls back to the 4-bit assumption for `uint8` -- which is announced by
    the caller, never silent.
    """
    packed_factor = getattr(param, "packed_factor", None)
    if packed_factor is not None:
        try:
            factor = float(packed_factor)
        except (TypeError, ValueError):
            factor = 0.0
        if factor >= 1.0:
            return factor
    if param.dtype == torch.uint8:
        return 2.0
    # int8 / float8 / bf16 / fp16 / fp32 all store one value per element.
    return 1.0


def estimate_stage_flops_per_token(model, hf_config) -> Dict[str, float]:
    """Per-new-token weight-GEMM FLOPs of this rank's loaded stage, plus the
    stage's full-attention (MLA) layer count."""
    hf_config = resolve_llm_config(hf_config)
    topk, num_experts = _moe_topk_and_num_experts(hf_config)
    buckets = {k: 0.0 for k in STAGE_KEYS}
    # Layer prefixes owning a kv_b_proj, i.e. this stage's actual MLA layers.
    # Keyed by prefix rather than by counting ".weight" tensors so quantized
    # namings (.qweight, .weight_packed, plus their scale tensors) neither miss
    # a layer nor count one twice. PP partitions are uneven, so "total layers /
    # pp_size" cannot give this, and the count has to come from the same stage
    # as the FLOPs it scales.
    mla_layer_prefixes = set()
    assumed_4bit: List[str] = []
    for name, param in model.named_parameters():
        if "kv_b_proj" in name:
            mla_layer_prefixes.add(name.split("kv_b_proj")[0])
        if param.dim() < 2 or param.numel() == 0:
            continue
        lname = name.lower()
        if any(part in lname for part in _SKIP_NAME_PARTS):
            continue
        elems = float(param.numel())
        factor = _logical_elems_per_stored_elem(name, param)
        if factor != 1.0:
            elems *= factor
            if getattr(param, "packed_factor", None) is None:
                assumed_4bit.append(name)
        if param.dim() == 3 and "expert" in lname and num_experts > 0 and topk > 0:
            # Stacked routed-expert weights [E_local, n, k]: a token activates
            # topk of the global experts, and with EP each rank executes its
            # share of the dispatched tokens, so the per-rank per-token work
            # averages to topk / num_experts_global of the local stack.
            elems *= topk / float(num_experts)
        buckets["flops"] += 2.0 * elems
    buckets["mla_layers"] = float(len(mla_layer_prefixes))
    if assumed_4bit:
        # Printed so a checkpoint that stores 8-bit weights in uint8 shows up as
        # a doubled count here instead of quietly shrinking every chunk.
        logger.info(
            "[ChunkFlopsEstimator] Assumed 4-bit packing (2 values/byte) for %d "
            "uint8 tensors with no declared packed_factor, e.g. %s. If this "
            "checkpoint is 8-bit, its FLOPs are counted 2x.",
            len(assumed_4bit),
            assumed_4bit[0],
        )
    logger.info(
        "[ChunkFlopsEstimator] Stage per-token GEMM FLOPs from loaded weights: "
        "%.3e, mla_layers=%d (topk=%d, num_experts=%d)",
        buckets["flops"],
        int(buckets["mla_layers"]),
        topk,
        num_experts,
    )
    return buckets



def aggregate_stage_flops(
    stage_buckets: Sequence[Dict[str, float]]
) -> Dict[str, np.ndarray]:
    """Stack the per-stage walked constants without collapsing them.

    A microbatch costs ``max_s cost_s(L, x)`` and the argmax moves with
    (L, x), so no single stage can stand in for the rest and a per-term max
    would describe a stage heavier than any real one. ChunkCostModel keeps all
    stages and takes the max/min online instead. The gathered list is identical
    on every rank, so schedulers stay in lockstep.
    """
    if not stage_buckets:
        raise ValueError("no per-stage FLOPs buckets to aggregate")
    return {
        key: np.array([b.get(key, 0.0) for b in stage_buckets], dtype=np.float64)
        for key in STAGE_KEYS
    }

