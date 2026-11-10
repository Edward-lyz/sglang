"""Extended startup warmup: sweep real prefill shapes before serving traffic.

The built-in startup warmup sends one 3-4 token request per DP rank, which is
enough to prove the engine is alive but leaves DeepGEMM JIT, attention-backend
plan buffers, MoE all-to-all buffers and the Triton autotuner cold for the
shapes real traffic uses. The first client requests then pay that one-time cost
and appear to stall right after the server reports ready.

This module sweeps a configurable list of prefill lengths across every DP rank
before the server is marked ``Up``. It is opt-in via
``SGLANG_WARMUP_PREFILL_LENS`` (``auto`` or an explicit comma-separated list)
and ``SGLANG_WARMUP_BATCH_SIZE``.
"""

from __future__ import annotations

import asyncio
import errno
import logging
import os
import ssl
import tempfile
import time
from typing import Dict, List, Optional, Union

import aiohttp
import numpy as np

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Lengths used by the "auto" sweep before the prefill budget itself is added.
# They cover the small / medium / large brackets that DeepGEMM, the attention
# backend and the MoE dispatch pick different configs for.
_AUTO_BASE_LENS = (512, 2048, 8192)

# Hard ceiling for a single auto-derived warmup request. Beyond this the extra
# shape coverage is not worth the startup time.
_AUTO_MAX_LEN = 64 * 1024

# Written into the election file once the elected process finishes sweeping.
_SWEEP_DONE_MARKER = b"done"


def sweep_is_enabled() -> bool:
    return bool(_raw_lens_config())


def _raw_lens_config() -> str:
    raw = envs.SGLANG_WARMUP_PREFILL_LENS.get()
    return raw.strip() if raw else ""


def resolve_sweep_lens(
    *,
    chunked_prefill_size: Optional[int],
    max_prefill_tokens: int,
    page_size: int,
    max_req_input_len: Optional[int],
) -> List[int]:
    """Return the prefill lengths to sweep, shortest first.

    ``auto`` derives a small / medium / large ladder from the prefill budget and
    adds one length above the chunk size so the chunked-prefill resume path is
    warmed too. An explicit list is used as given, only clamped to the model's
    input limit.
    """

    raw = _raw_lens_config()
    if not raw:
        return []

    limit = _sweep_upper_bound(
        chunked_prefill_size=chunked_prefill_size,
        max_prefill_tokens=max_prefill_tokens,
        max_req_input_len=max_req_input_len,
    )

    if raw == "auto":
        chunk = chunked_prefill_size if _is_positive(chunked_prefill_size) else limit
        candidates = list(_AUTO_BASE_LENS) + [chunk, 2 * chunk]
    else:
        candidates = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                candidates.append(int(token))
            except ValueError:
                logger.warning(
                    "Ignoring non-integer entry %r in SGLANG_WARMUP_PREFILL_LENS.",
                    token,
                )

    lens = set()
    for length in candidates:
        if length <= 0:
            continue
        lens.add(_align_len(min(length, limit), page_size))
    return sorted(lens)


def _sweep_upper_bound(
    *,
    chunked_prefill_size: Optional[int],
    max_prefill_tokens: int,
    max_req_input_len: Optional[int],
) -> int:
    # A request longer than the chunk size is legal (it just gets chunked), so
    # the ceiling comes from the request-length limit, not the per-forward one.
    limit = _AUTO_MAX_LEN
    if _is_positive(max_req_input_len):
        limit = min(limit, max_req_input_len - 1)
    elif _is_positive(chunked_prefill_size):
        limit = min(limit, 2 * chunked_prefill_size)
    else:
        limit = min(limit, max(max_prefill_tokens, 1))
    return max(limit, 1)


def _is_positive(value: Optional[int]) -> bool:
    return value is not None and value > 0


def _align_len(length: int, page_size: int) -> int:
    if page_size <= 1:
        return max(length, 1)
    return max(length // page_size * page_size, page_size if length >= page_size else 1)


def claim_sweep_election(key: str) -> bool:
    """Return True for the one process that should run the sweep.

    With ``--tokenizer-worker-num N`` every HTTP worker runs its own warmup
    thread, so without this the sweep would be sent N times. The first process
    to create the marker file wins; the others skip the sweep and wait for the
    winner to finish so no worker reports ready early.
    """

    path = _election_path(key)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    except OSError as e:
        # Unwritable temp dir: fall back to letting every worker sweep rather
        # than skipping warmup entirely.
        logger.warning(
            "Extended warmup: sweep election unavailable (%s); running unelected.",
            os.strerror(e.errno) if e.errno else e,
        )
        return True
    os.close(fd)
    return True


def reset_sweep_election(key: str) -> None:
    """Drop a marker left behind by a previous launch that reused this key."""

    try:
        os.unlink(_election_path(key))
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.warning(
            "Extended warmup: could not reset sweep election (%s).",
            os.strerror(e.errno) if e.errno else e,
        )


def mark_sweep_done(key: str) -> None:
    try:
        with open(_election_path(key), "wb") as f:
            f.write(_SWEEP_DONE_MARKER)
    except OSError as e:
        logger.warning(
            "Extended warmup: could not mark sweep done (%s).",
            os.strerror(e.errno) if e.errno else e,
        )


def wait_for_sweep_done(key: str, timeout: float) -> None:
    """Block until the elected process marks the sweep done, or timeout."""

    path = _election_path(key)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with open(path, "rb") as f:
                if f.read() == _SWEEP_DONE_MARKER:
                    return
        except OSError as e:
            if e.errno != errno.ENOENT:
                return
        time.sleep(2)
    logger.warning(
        "Extended warmup: timed out after %.0fs waiting for the elected worker's "
        "prefill sweep; proceeding.",
        timeout,
    )


def _election_path(key: str) -> str:
    return os.path.join(tempfile.gettempdir(), f"sglang_warmup_sweep_{key}")


def run_prefill_sweep(
    *,
    url: str,
    headers: Dict[str, str],
    ssl_verify: Union[bool, str],
    timeout: float,
    dp_size: int,
    disaggregation_mode: str,
    lens: List[int],
    batch_size: int,
    vocab_size: int,
) -> None:
    """Send the prefill-shape sweep, one length at a time, shortest first.

    Best effort: a failed length is logged and the sweep continues, because the
    server is already known to work from the minimal warmup request that ran
    before this.
    """

    if not lens:
        return

    logger.info(
        "Extended warmup: sweeping prefill lens %s x %s req/DP-rank across %s DP rank(s).",
        lens,
        batch_size,
        dp_size,
    )
    t_all = time.perf_counter()
    for length in lens:
        t_len = time.perf_counter()
        try:
            statuses = asyncio.run(
                _send_sweep_requests(
                    url=url,
                    headers=headers,
                    ssl_verify=ssl_verify,
                    timeout=timeout,
                    dp_size=dp_size,
                    disaggregation_mode=disaggregation_mode,
                    length=length,
                    batch_size=batch_size,
                    vocab_size=vocab_size,
                )
            )
        except Exception as e:
            logger.warning(
                "Extended warmup: prefill len=%s failed (%s: %s); continuing.",
                length,
                type(e).__name__,
                e,
            )
            continue

        failed = [code for code in statuses if code != 200]
        if failed:
            logger.warning(
                "Extended warmup: prefill len=%s returned non-200 status codes %s.",
                length,
                failed,
            )
        else:
            logger.info(
                "Extended warmup: prefill len=%s done in %.2fs.",
                length,
                time.perf_counter() - t_len,
            )

    logger.info(
        "Extended warmup: prefill sweep finished in %.2fs.", time.perf_counter() - t_all
    )


async def _send_sweep_requests(
    *,
    url: str,
    headers: Dict[str, str],
    ssl_verify: Union[bool, str],
    timeout: float,
    dp_size: int,
    disaggregation_mode: str,
    length: int,
    batch_size: int,
    vocab_size: int,
) -> List[int]:
    ssl_context = (
        ssl_verify
        if isinstance(ssl_verify, bool)
        else ssl.create_default_context(cafile=ssl_verify)
    )
    high = max(min(vocab_size, 1 << 16), 2)

    async def send(session: aiohttp.ClientSession, dp_rank: int, index: int) -> int:
        json_data = {
            # Fresh ids per request so a prefix cache cannot shorten the
            # prefill we are trying to warm.
            "input_ids": np.random.randint(1, high, size=[length]).tolist(),
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 8,
                "ignore_eos": True,
            },
            "log_metrics": False,
            "skip_radix_cache_insert": True,
        }
        if disaggregation_mode != "null":
            json_data["bootstrap_host"] = FAKE_BOOTSTRAP_HOST
            # Keep room % dp_size == dp_rank so follow_bootstrap_room routing
            # and the explicit routed_dp_rank agree on the target rank.
            json_data["bootstrap_room"] = dp_rank + index * dp_size
            json_data["routed_dp_rank"] = dp_rank
        async with session.post(
            url + "/generate", json=json_data, ssl=ssl_context
        ) as response:
            await response.read()
            return response.status

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout),
        headers=headers,
    ) as session:
        return await asyncio.gather(
            *(
                send(session, dp_rank, index)
                for index in range(batch_size)
                for dp_rank in range(dp_size)
            )
        )
