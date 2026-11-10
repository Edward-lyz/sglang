"""Per-stage timing calibration for the measured dynamic-chunking cost model.

Every PP stage times its own forward on a grid of (cached prefix L, new chunk
x) microbatches and fits ``t = e + g*x + r*(L+x) + a*x*(L+0.5x)`` in ms (see
``chunk_cost_model.fit_stage_coefficients``). The forwards take PP proxy
tensors, so no PP send/recv happens and stages calibrate in parallel; callers
must only use this when a stage's forward has no intra-stage collectives
(tp = attn_cp = ep = attn_dp = 1), otherwise ranks of one stage could diverge
on a failure and hang.

The cached prefix is faked: L KV slots are allocated and handed to the request
as ``prefix_indices``, so attention reads L real cache entries without running
L tokens first. Their contents only need to be finite; the pool is fresh at
startup and earlier calibration forwards write real activations into it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from array import array
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from sglang.srt.layers.dp_attention import (
    get_attention_dp_rank,
    get_attention_dp_size,
    is_dp_attention_enabled,
    set_is_extend_in_batch,
)
from sglang.srt.managers.chunk_cost_model import (
    CalibrationSample,
    fit_stage_coefficients,
)
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils.common import get_device_module

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)

# Bump whenever the grid changes: a saved profile fitted on another grid must
# not be reused (the grid is not otherwise part of the fingerprint).
PROFILE_VERSION = 2
# Cached-history lengths to time. The tail reaches the production P99/max
# prompt (~545K / ~705K): chunking a fresh very long request, or a long-context
# prefix hit, runs at those L, and without points there the fit extrapolates.
# Points with L + x beyond the context or the free KV pool are skipped.
_HISTORY_GRID = (0, 8192, 32768, 65536, 131072, 262144, 393216, 524288, 720896)
_CHUNK_FRACTIONS = (16, 8, 4, 2, 1)  # base_chunk_size / f


def _align(n: int, page_size: int) -> int:
    return max(page_size, n // page_size * page_size)


def build_grid(
    base_chunk_size: int, page_size: int, max_seq_len: int
) -> List[List[Tuple[int, int]]]:
    """Microbatches to time, as lists of (history_len, chunk_len).

    Single-request points cover the (L, x) plane up to ``max_seq_len``; a few
    multi-request points check that request costs add up with the constant
    charged once per microbatch, which is how the scheduler uses the model.
    """
    xs = sorted({_align(base_chunk_size // f, page_size) for f in _CHUNK_FRACTIONS})
    grid: List[List[Tuple[int, int]]] = []
    for L in _HISTORY_GRID:
        for x in xs:
            if L + x <= max_seq_len:
                grid.append([(L, x)])
    quarter = _align(base_chunk_size // 4, page_size)
    eighth = _align(base_chunk_size // 8, page_size)
    grid.append([(0, quarter)] * 4)
    grid.append([(0, eighth)] * 8)
    if 32768 + quarter <= max_seq_len:
        grid.append([(32768, quarter), (0, quarter)])
    return grid


def _free_req(scheduler: "Scheduler", req: Req) -> None:
    if req.req_pool_idx is not None:
        kv_indices = scheduler.req_to_token_pool.req_to_token[
            req.req_pool_idx, : req.extend_range.end
        ]
        scheduler.token_to_kv_pool_allocator.free(kv_indices)
        if req.mamba_pool_idx is not None:
            scheduler.req_to_token_pool.free_mamba_cache(req)
        scheduler.req_to_token_pool.free(req)
        req.kv = None
    elif len(req.prefix_indices) > 0:
        # prepare_for_extend never ran: the prefix is not in req_to_token yet.
        scheduler.token_to_kv_pool_allocator.free(req.prefix_indices)
    req.prefix_indices = torch.empty((0,), dtype=torch.int64)


def _make_req(
    scheduler: "Scheduler", rid: str, history_len: int, chunk_len: int, rng
) -> Req:
    total = history_len + chunk_len
    input_ids = array(
        "q", rng.randint(0, 10000, size=total, dtype=np.int64).tobytes()
    )
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=input_ids,
        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
    )
    req.full_untruncated_fill_ids = req.origin_input_ids
    req.logprob_start_len = -1
    if history_len > 0:
        prefix = scheduler.token_to_kv_pool_allocator.alloc(history_len)
        if prefix is None:
            raise RuntimeError(f"no KV slots for a {history_len}-token fake prefix")
        req.prefix_indices = prefix.to(torch.int64)
    req.set_extend_range(history_len, total)
    return req


def _proxy_residual_shape(model_runner, num_tokens: int, model_config) -> Tuple[int, ...]:
    """Shape of the ``residual`` a stage receives from the previous PP rank.

    Usually [T, H]. Models with an attention-residual bank (K3,
    ``attn_res_block_size``) ship the bank instead: [T, NB, H] with one row
    per block the earlier layers already wrote, i.e. cdiv(start_layer, bs).
    """
    hidden = model_config.hidden_size
    for module in model_runner.model.modules():
        start = getattr(module, "start_layer", None)
        cfg = getattr(module, "config", None)
        if start is None or cfg is None:
            continue
        cfg = getattr(cfg, "text_config", None) or cfg
        bs = getattr(cfg, "attn_res_block_size", None)
        if bs:
            return (num_tokens, -(-int(start) // int(bs)), hidden)
        break
    return (num_tokens, hidden)


def _time_once(
    scheduler: "Scheduler", items: Sequence[Tuple[int, int]], rng, tag: str
) -> float:
    model_runner = scheduler.tp_worker.model_runner
    model_config = model_runner.model_config
    reqs: List[Req] = []
    try:
        for i, (history_len, chunk_len) in enumerate(items):
            reqs.append(
                _make_req(scheduler, f"chunk-calib-{tag}-{i}", history_len, chunk_len, rng)
            )
        batch = ScheduleBatch.init_new(
            reqs,
            scheduler.req_to_token_pool,
            scheduler.token_to_kv_pool_allocator,
            scheduler.tree_cache,
            scheduler.model_config,
            False,
            scheduler.spec_algorithm,
        )
        num_tokens = sum(chunk_len for _, chunk_len in items)
        if is_dp_attention_enabled():
            global_num_tokens = [0] * get_attention_dp_size()
            global_num_tokens[get_attention_dp_rank()] = num_tokens
            batch.global_num_tokens = global_num_tokens
            batch.global_num_tokens_for_logprob = global_num_tokens

        batch.prepare_for_extend()
        if batch.input_ids is None and batch.prefill_input_ids_cpu is not None:
            batch.input_ids = batch.prefill_input_ids_cpu.to(
                scheduler.device, non_blocking=True
            )
            batch.prefill_input_ids_cpu = None

        # Random, not zero, activations: a zero hidden state gives every
        # expert the same gate logit and routes all tokens to the same top-k,
        # which times a different MoE than real traffic does.
        hs = getattr(model_config, "hc_hidden_size", None) or model_config.hidden_size
        residual_shape = _proxy_residual_shape(model_runner, num_tokens, model_config)
        proxy_tensors = {
            "hidden_states": torch.randn(
                (num_tokens, hs), dtype=model_config.dtype, device=scheduler.device
            ),
            "residual": torch.randn(
                residual_shape, dtype=model_config.dtype, device=scheduler.device
            ),
        }
        topk_size = model_runner.get_pp_proxy_topk_size()
        if topk_size is not None:
            proxy_tensors["topk_indices"] = torch.zeros(
                (num_tokens, topk_size), dtype=torch.int32, device=scheduler.device
            )

        forward_batch = ForwardBatch.init_new(
            batch, model_runner, return_hidden_states_before_norm=False
        )
        set_is_extend_in_batch(batch.forward_mode.is_extend())
        device_module = get_device_module()
        device_module.synchronize()
        start = time.perf_counter()
        model_runner.forward(
            forward_batch=forward_batch, pp_proxy_tensors=PPProxyTensors(proxy_tensors)
        )
        device_module.synchronize()
        return (time.perf_counter() - start) * 1e3
    finally:
        for req in reqs:
            _free_req(scheduler, req)


def calibrate_stage(
    scheduler: "Scheduler",
    base_chunk_size: int,
    warmup: int = 1,
    repeats: int = 5,
) -> List[CalibrationSample]:
    """Time this rank's stage on the calibration grid; min of ``repeats``.

    A forward is fixed GPU work; anything on top comes from interference --
    every PP rank calibrates at once, eight to a host, so CPU contention
    delays launches -- and only ever adds time. Per-point spread on K3.1 PP16
    is p50 7% / p90 17% / max 100% of the fastest run, so a median of three
    still lets outliers through and pushes stages past the 10% LOO fit gate;
    the minimum of five estimates the uncontended cost (p90 LOO error over 16
    stages: max 0.101 -> 0.069, mean 0.075 -> 0.051).
    """
    page_size = scheduler.page_size
    max_seq_len = min(
        scheduler.model_config.context_len,
        scheduler.token_to_kv_pool_allocator.available_size() - 4 * page_size,
    )
    grid = build_grid(base_chunk_size, page_size, max_seq_len)
    skipped = [L for L in _HISTORY_GRID if L + page_size > max_seq_len]
    if skipped:
        logger.info(
            "[PP Dynamic Chunk] [PP%d] Calibration skips history lengths %s "
            "(max calibratable sequence %d); the fit extrapolates there.",
            scheduler.ps.pp_rank,
            skipped,
            max_seq_len,
        )
    rng = np.random.RandomState(1234)
    samples: List[CalibrationSample] = []
    for gi, items in enumerate(grid):
        for w in range(warmup):
            _time_once(scheduler, items, rng, f"{gi}w{w}")
        times = [_time_once(scheduler, items, rng, f"{gi}r{r}") for r in range(repeats)]
        samples.append((list(items), float(np.min(times))))
    logger.info(
        "[PP Dynamic Chunk] [PP%d] Calibrated %d microbatches: %s",
        scheduler.ps.pp_rank,
        len(samples),
        [(items[0], len(items), round(ms, 3)) for items, ms in samples],
    )
    return samples


def calibrate_and_fit(
    scheduler: "Scheduler", base_chunk_size: int
) -> Tuple[bool, Optional[List[float]], float, str]:
    """Calibrate and fit this stage. Never raises: a failure is returned as
    data so it can ride the PP all-gather instead of stranding other ranks."""
    try:
        samples = calibrate_stage(scheduler, base_chunk_size)
        coef, err = fit_stage_coefficients(samples)
        return True, [float(c) for c in coef], err, ""
    except Exception as e:  # noqa: BLE001 -- reported to every rank below
        logger.warning(
            "[PP Dynamic Chunk] [PP%d] Stage calibration failed: %s",
            scheduler.ps.pp_rank,
            e,
        )
        return False, None, float("inf"), repr(e)
    finally:
        # Long-history grid points leave tens of GB in the caching allocator
        # (indexer logits); NCCL's lazy P2P communicators allocate outside it
        # and fail with "Failed to CUDA calloc" on the first PP send.
        torch.cuda.empty_cache()


def config_fingerprint(scheduler: "Scheduler") -> str:
    """Identity of everything that moves the coefficients. A stale profile is
    worse than none, so anything plausibly affecting kernel speed is in it."""
    sa = scheduler.server_args
    device_name = (
        torch.cuda.get_device_name() if torch.cuda.is_available() else "unknown"
    )
    try:
        import sglang

        version = getattr(sglang, "__version__", "")
    except Exception:  # noqa: BLE001
        version = ""
    payload = {
        "version": PROFILE_VERSION,
        "sglang": version,
        "model_path": sa.model_path,
        "device": device_name,
        "pp_size": sa.pp_size,
        "tp_size": sa.tp_size,
        "ep_size": getattr(sa, "ep_size", 1),
        "dp_size": sa.dp_size,
        "kv_cache_dtype": sa.kv_cache_dtype,
        "attention_backend": sa.attention_backend,
        "quantization": sa.quantization,
        "page_size": sa.page_size,
        "chunked_prefill_size": sa.chunked_prefill_size,
        "pp_partition": os.environ.get("SGLANG_PP_LAYER_PARTITION", ""),
    }
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def resolve_profile_path(env_value: str) -> Tuple[str, bool]:
    """Profile to load and whether a fresh calibration may be saved to it.

    An explicit path is read and rewritten. Without one there is no profile:
    every startup calibrates and keeps the coefficients in memory.
    """
    env_value = env_value.strip()
    if env_value:
        return env_value, True
    return "", False


def load_profile(path: str, fingerprint: str, pp_size: int) -> Optional[List[List[float]]]:
    """Per-stage coefficients from ``path``, or None if missing or stale."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        logger.warning("[PP Dynamic Chunk] Unreadable profile %s: %s", path, e)
        return None
    if data.get("fingerprint") != fingerprint:
        logger.warning(
            "[PP Dynamic Chunk] Profile %s fingerprint %s != current %s; "
            "recalibrating.",
            path,
            data.get("fingerprint"),
            fingerprint,
        )
        return None
    coefs = data.get("coefs")
    if not isinstance(coefs, list) or len(coefs) != pp_size:
        return None
    return coefs


def save_profile(
    path: str, fingerprint: str, coefs: List[List[float]], errs: List[float]
) -> None:
    """Write the fitted coefficients to ``path`` atomically; errors only warn."""
    if not path:
        return
    data: Dict = {
        "fingerprint": fingerprint,
        "coefs": coefs,
        "fit_p90_rel_err": errs,
        "columns": ["e_ms", "g_ms_per_token", "r_ms_per_token", "a_ms_per_pair"],
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
        logger.info("[PP Dynamic Chunk] Saved measured profile to %s", path)
    except OSError as e:
        logger.warning("[PP Dynamic Chunk] Could not save profile %s: %s", path, e)
