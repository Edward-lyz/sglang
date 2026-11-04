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
from typing import Dict, Sequence, Tuple

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
    for name, param in model.named_parameters():
        if "kv_b_proj" in name:
            mla_layer_prefixes.add(name.split("kv_b_proj")[0])
        if param.dim() < 2 or param.numel() == 0:
            continue
        lname = name.lower()
        if any(part in lname for part in _SKIP_NAME_PARTS):
            continue
        elems = float(param.numel())
        if param.dtype == torch.uint8:
            # mxfp4 / int4 weights are stored two elements per byte.
            elems *= 2.0
        if param.dim() == 3 and "expert" in lname and num_experts > 0 and topk > 0:
            # Stacked routed-expert weights [E_local, n, k]: a token activates
            # topk of the global experts, and with EP each rank executes its
            # share of the dispatched tokens, so the per-rank per-token work
            # averages to topk / num_experts_global of the local stack.
            elems *= topk / float(num_experts)
        buckets["flops"] += 2.0 * elems
    buckets["mla_layers"] = float(len(mla_layer_prefixes))
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

