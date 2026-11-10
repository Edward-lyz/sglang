"""通过两波请求压测内存 cache 命中场景下的 prefill 吞吐。

每个目标命中率都执行以下流程：先分别清理 GPU cache 和所有节点的内存
cache；再发送按 page 对齐的短 prompt 填充内存 cache；随后只清 GPU cache；
最后发送对应的完整 prompt 并统计这一波的性能。填充波与测量波严格复用请求
编号、prompt 前缀、到达时间表以及 DP rank，确保测到的是内存 cache 命中，
而不是测量前遗留的 GPU cache 命中。

Usage
-----
Batch mode（默认，用高并发闭环请求测饱和吞吐）:
    python3 benchmark/bench_ascache.py --host 127.0.0.1 --port 30000 \\
        --total-len 1024 --num-rounds 10 --batch-size 16 \\
        --target-hit-rates "0.0,0.25,0.5,0.75,0.9"

QPS mode（显式指定泊松到达率，建议传多个 QPS 找饱和点）:
    python3 benchmark/bench_ascache.py --host 127.0.0.1 --port 30000 \\
        --total-len 1024 --qps 10 --duration 60 \\
        --target-hit-rates "0.0,0.5,0.9"

For non-zero cache-hit targets, the server must be started with
--enable-cache-report to get accurate cached_tokens from
usage.prompt_tokens_details.cached_tokens. A zero-hit run can infer that value
after clearing both cache layers and skipping the cache-fill wave.
"""

import argparse
import asyncio
import json
import os
import random
import time
from typing import List, Tuple

import aiohttp
import numpy as np
import requests

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=20 * 60 * 60)

# marker = FIRST_TOKEN_BASE + req_index ; body = arange(ARANGE_OFFSET + i)
# Both values stay fixed across the cache-fill and measured waves so prompts
# with the same request index share a deterministic prefix.
FIRST_TOKEN_BASE = 1000
ARANGE_OFFSET = 10000

# ---------------------------------------------------------------------------
# Server / tokenizer helpers
# ---------------------------------------------------------------------------


def get_vocab_size(tokenizer_path: str) -> int:
    """Read vocab size without loading a model-specific Transformers config."""
    config_path = os.path.join(tokenizer_path, "config.json")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            vocab_size = json.load(f).get("vocab_size")
        if isinstance(vocab_size, int) and vocab_size > 0:
            return vocab_size

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    return tokenizer.vocab_size


def get_model_from_server(base_url: str) -> str:
    """从服务端查询当前加载的模型名称。"""
    resp = requests.get(f"{base_url}/v1/models", timeout=10)
    resp.raise_for_status()
    return resp.json()["data"][0]["id"]


def get_page_size(base_url: str) -> int:
    """从服务端读取 prefix cache 的 page size，查询失败时返回 1。"""
    try:
        resp = requests.get(f"{base_url}/get_server_info", timeout=10)
        resp.raise_for_status()
        return resp.json().get("page_size", 1)
    except Exception:
        return 1


def get_dp_size(base_url: str) -> int:
    """Detect server data-parallel size. Each DP rank has its own prefix cache,
    so both waves must pin each req_id to the same deterministic rank."""
    try:
        resp = requests.get(f"{base_url}/get_server_info", timeout=10)
        resp.raise_for_status()
        info = resp.json()
        for k in ("data_parallel_size", "dp_size", "data_parallel_world_size"):
            v = info.get(k)
            if v:
                return int(v)
    except Exception:
        pass
    return 1


def flush_cache(base_url: str):
    """只清理 GPU/device KV cache，保留各节点的内存 cache。"""
    resp = requests.post(
        f"{base_url}/flush_cache", params={"evict_evictable": "true"}, timeout=60
    )
    resp.raise_for_status()


def clear_memory_cache(base_url: str):
    """通过统一接口清理该实例所有节点 8080 服务中的内存 cache。"""
    resp = requests.post(
        f"{base_url}/clear_all_cache",
        params={"memory_only": "true"},
        timeout=5 * 60,
    )
    # 接口以 207 表示部分节点失败，而 requests.raise_for_status() 会把所有
    # 2xx（包括 207）视为成功；这里必须严格要求 200，避免带着残留 cache
    # 继续压测并得到虚高的命中率。
    if resp.status_code != 200:
        raise RuntimeError(
            f"memory cache clear returned HTTP {resp.status_code}: {resp.text}"
        )


def flush_blocks_to_ssd(ssd_base_url: str, namespace: str, count: int):
    """把 count 个 host 内存 cache block 换出到 SSD，用于构造 SSD 命中场景。

    对应接口：POST /op/flushBlocksToSSD?namespace=<ns>&count=<n>
    """
    resp = requests.post(
        f"{ssd_base_url}/op/flushBlocksToSSD",
        params={"namespace": namespace, "count": count},
        timeout=5 * 60,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"flushBlocksToSSD returned HTTP {resp.status_code}: {resp.text}"
        )


def maybe_flush_to_ssd(
    ssd_base_url: str,
    ssd_namespace: str,
    ssd_hit_rate: float,
    total_chunks: int,
    hit_chunks: int,
    num_fill_requests: int,
) -> dict:
    """按 SSD 命中率把对应数量的 page 从 host 内存换出到 SSD。

    语义约定：
    - ``ssd_hit_rate`` 是"占整个 prompt 总 page 数"的比例，因此每个请求需要
      换出的 page 数 = ``round(ssd_hit_rate * total_chunks)``。
    - 只能换出填充波实际写入的 page，因此会 clamp 到 ``hit_chunks``
      （由 ``target_hit_rate`` 决定）。若 ssd_hit_rate > target_hit_rate 会打印提示。
    - count 以填充波实际写入的 block 总数为口径：每个填充请求的 marker 不同、
      独占自己的 page，所以 ``count = 每请求换出 page 数 × 填充请求数``。

    返回本次换出的统计信息（dict）；无需换出时返回 ``None``。
    """
    if ssd_hit_rate <= 0.0 or num_fill_requests <= 0 or hit_chunks <= 0:
        return None

    ssd_pages_per_req = round(ssd_hit_rate * total_chunks)
    if ssd_pages_per_req <= 0:
        return None
    if ssd_pages_per_req > hit_chunks:
        print(
            f"  [ssd] ssd_hit_rate={ssd_hit_rate:.4f} 需要每请求 "
            f"{ssd_pages_per_req} 个 page，但填充波每请求只写入 {hit_chunks} 个 "
            f"page（受 target_hit_rate 限制），已 clamp 到 {hit_chunks}。"
        )
        ssd_pages_per_req = hit_chunks

    count = ssd_pages_per_req * num_fill_requests
    print(
        f"  [ssd] Flushing {count} blocks to SSD "
        f"(namespace={ssd_namespace}, {ssd_pages_per_req} pages/req × "
        f"{num_fill_requests} reqs) ..."
    )
    flush_blocks_to_ssd(ssd_base_url, ssd_namespace, count)
    return {
        "ssd_hit_rate": ssd_hit_rate,
        "ssd_pages_per_request": ssd_pages_per_req,
        "ssd_flush_count": count,
    }


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def sample_hit_chunks(
    total_chunks: int, target_hit_rate: float, rng: random.Random
) -> int:
    """Return a fixed hit_chunks so every request has exactly target_hit_rate cache hits.

    Unlike bench_v2 which sampled from a uniform distribution (mean = target),
    this always returns round(target_hit_rate * total_chunks), guaranteeing
    the same page-aligned target prefix length for every request.
    """
    if total_chunks == 0 or target_hit_rate <= 0.0:
        return 0
    if target_hit_rate >= 1.0:
        return total_chunks
    return round(target_hit_rate * total_chunks)


def build_request_prompt(
    req_index: int,
    total_len: int,
    vocab_size: int,
    first_token_base: int = FIRST_TOKEN_BASE,
    arange_offset: int = ARANGE_OFFSET,
) -> List[int]:
    """构造总长度恰好为 ``total_len`` 的确定性 prompt。

    第一个 token 是每个请求独有的 marker，避免并发请求互相命中；剩余 token
    使用固定 arange。填充波传入裁剪后的长度，测量波传入完整长度，因此同一
    req_index 的短 prompt 一定是长 prompt 的严格前缀。
    """
    if total_len <= 0:
        return []

    marker = first_token_base + req_index
    if marker >= vocab_size:
        marker = first_token_base + (req_index % max(1, vocab_size - first_token_base))
    # total_len 包含 marker；不能再生成 total_len 个 body token，否则会产生
    # 1 token 的长度与 page 对齐偏差，导致实际命中率偏离目标值。
    body = [(arange_offset + i) % vocab_size for i in range(total_len - 1)]
    return [marker] + body


def build_batch(
    vocab_size: int,
    total_len: int,
    page_size: int,
    target_hit_rate: float,
    batch_size: int,
    req_id_offset: int,
    rng: random.Random,
    first_token_base: int = FIRST_TOKEN_BASE,
    arange_offset: int = ARANGE_OFFSET,
) -> Tuple[List[List[int]], List[int]]:
    """Build one batch of ``batch_size`` prompts.

    Each request's marker = first_token_base + (req_id_offset + i), so markers
    are unique within the batch.  The body is a shared positional arange.

    Returns
    -------
    prompts           : list of batch_size prompt-id-lists
    per_req_hit_chunks: expected hit_chunks per request (label for logging)
    """
    total_chunks = total_len // page_size
    prompts: List[List[int]] = []
    per_req_hit_chunks: List[int] = []
    hc = sample_hit_chunks(total_chunks, target_hit_rate, rng)

    for i in range(batch_size):
        prompt = build_request_prompt(
            req_id_offset + i, total_len, vocab_size, first_token_base, arange_offset
        )
        prompts.append(prompt)
        per_req_hit_chunks.append(hc)

    return prompts, per_req_hit_chunks


# ---------------------------------------------------------------------------
# Async request
# ---------------------------------------------------------------------------


def _strip_data_prefix(s: str) -> str:
    """移除 Server-Sent Events 数据行开头的 ``data: `` 前缀。"""
    return s[len("data: ") :] if s.startswith("data: ") else s


async def send_request(
    session: aiohttp.ClientSession,
    api_url: str,
    prompt_ids: List[int],
    output_len: int,
    model: str,
    req_id: int = 0,
    bootstrap_host: str = "2.2.2.2",
    bootstrap_port: int = 10000,
    dp_size: int = 1,
) -> dict:
    """Send a /v1/completions streaming request and return per-request metrics."""
    # Pin to a deterministic DP rank: each rank has its own prefix cache, so
    # the SAME req_id in a later invocation must land on the SAME rank to hit.
    dp_rank = req_id % dp_size if dp_size > 1 else 0
    payload = {
        "model": model,
        "rid": f"test-{random.randint(0, 0xFFFFFFFF):08x}",
        "prompt": prompt_ids,
        "max_tokens": output_len,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
        "bootstrap_host": bootstrap_host,
        "bootstrap_port": bootstrap_port,
        "bootstrap_room": req_id,
        "data_parallel_rank": dp_rank,
    }

    result = {
        "success": False,
        "prompt_tokens": 0,
        "cached_tokens": 0,
        "completion_tokens": 0,
        "ttft": 0.0,
        "latency": 0.0,
        "error": "",
    }
    st = time.perf_counter()
    first_token = False

    try:
        async with session.post(url=api_url, json=payload) as response:
            if response.status != 200:
                result["error"] = f"HTTP {response.status}: {response.reason}"
                return result

            async for raw in response.content:
                raw = raw.strip()
                if not raw:
                    continue
                chunk_str = _strip_data_prefix(raw.decode("utf-8"))
                if chunk_str == "[DONE]":
                    continue
                try:
                    data = json.loads(chunk_str)
                except json.JSONDecodeError:
                    continue

                now = time.perf_counter()

                choices = data.get("choices", [])
                if choices and not first_token and choices[0].get("text"):
                    result["ttft"] = now - st
                    first_token = True

                usage = data.get("usage")
                if usage:
                    result["prompt_tokens"] = usage.get("prompt_tokens", 0)
                    result["completion_tokens"] = usage.get("completion_tokens", 0)
                    details = usage.get("prompt_tokens_details")
                    if details is not None:
                        result["cached_tokens"] = details.get("cached_tokens", 0)
                    else:
                        result["_no_cache_report"] = True

            result["latency"] = time.perf_counter() - st
            result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Batch / warmup execution
# ---------------------------------------------------------------------------


async def send_batch(
    api_url: str,
    model: str,
    output_len: int,
    prompt_list: List[List[int]],
    semaphore: asyncio.Semaphore,
    req_id_offset: int = 0,
    bootstrap_host: str = "2.2.2.2",
    bootstrap_port: int = 10000,
    dp_size: int = 1,
) -> List[dict]:
    """并发发送一批请求，并返回每个请求的性能指标。"""
    # 一个测量波复用同一个连接池。若每个请求都新建 ClientSession，TLS/TCP
    # 建连和连接池初始化会被计入墙钟时间，从而低估服务端的 cache 命中吞吐。
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:

        async def _one(i, ids):
            """在并发度限制内发送批次中的单个请求。"""
            async with semaphore:
                return await send_request(
                    session,
                    api_url,
                    ids,
                    output_len,
                    model,
                    req_id=req_id_offset + i,
                    bootstrap_host=bootstrap_host,
                    bootstrap_port=bootstrap_port,
                    dp_size=dp_size,
                )

        return await asyncio.gather(
            *[asyncio.create_task(_one(i, ids)) for i, ids in enumerate(prompt_list)]
        )


def build_qps_schedule(
    qps: float, duration: float, np_rng: np.random.Generator
) -> List[float]:
    """Build one Poisson arrival schedule as offsets from wave start."""
    if qps <= 0:
        raise ValueError(f"qps must be positive, got {qps}")

    schedule: List[float] = []
    offset = 0.0
    while offset < duration:
        schedule.append(offset)
        offset += float(np_rng.exponential(1.0 / qps))
    return schedule


async def send_requests_qps(
    api_url: str,
    model: str,
    output_len: int,
    vocab_size: int,
    total_len: int,
    page_size: int,
    target_hit_rate: float,
    first_token_base: int = FIRST_TOKEN_BASE,
    arange_offset: int = ARANGE_OFFSET,
    qps: float = 10.0,
    duration: float = 60.0,
    semaphore: asyncio.Semaphore = None,
    rng: random.Random = None,
    np_rng: np.random.Generator = None,
    schedule: List[float] = None,
    bootstrap_host: str = "2.2.2.2",
    bootstrap_port: int = 10000,
    dp_size: int = 1,
) -> List[dict]:
    """Send requests at a target QPS using Poisson inter-arrival times.

    Requests are fired according to a Poisson process (exponential inter-arrival
    gaps) for ``duration`` seconds.  Each request is dispatched as a background
    task so they overlap naturally; ``semaphore`` caps max in-flight requests.

    Returns a flat list of per-request result dicts (same schema as send_request).
    """
    results: List[dict] = []
    tasks: List[asyncio.Task] = []

    loop_start = time.perf_counter()

    if schedule is None:
        schedule = build_qps_schedule(qps, duration, np_rng)

    # 整个波次共享连接池，避免客户端建连开销污染吞吐数据。
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:

        async def _one(rid: int, ids: List[int]) -> dict:
            """在并发度限制内发送 QPS 波次中的单个请求。"""
            async with semaphore:
                return await send_request(
                    session,
                    api_url,
                    ids,
                    output_len,
                    model,
                    req_id=rid,
                    bootstrap_host=bootstrap_host,
                    bootstrap_port=bootstrap_port,
                    dp_size=dp_size,
                )

        for req_id, offset in enumerate(schedule):
            sleep_time = loop_start + offset - time.perf_counter()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

            # 同一个 req_id 在两波中构造出前缀一致的 prompt。
            prompt = build_request_prompt(
                req_id, total_len, vocab_size, first_token_base, arange_offset
            )
            tasks.append(asyncio.create_task(_one(req_id, prompt)))

        if tasks:
            results = list(await asyncio.gather(*tasks))

    return results


# ---------------------------------------------------------------------------
# Single hit-rate benchmark
# ---------------------------------------------------------------------------


async def run_one_hit_rate(
    base_url: str,
    api_url: str,
    model: str,
    vocab_size: int,
    target_hit_rate: float,
    total_len: int,
    page_size: int,
    output_len: int,
    num_rounds: int,
    batch_size: int,
    max_concurrency: int,
    rng: random.Random,
    bootstrap_host: str = "2.2.2.2",
    bootstrap_port: int = 10000,
    first_token_base: int = FIRST_TOKEN_BASE,
    arange_offset: int = ARANGE_OFFSET,
    dp_size: int = 1,
    qps: float = None,
    duration: float = 60.0,
    ssd_base_url: str = None,
    ssd_namespace: str = None,
    ssd_hit_rate: float = 0.0,
) -> dict:
    """针对一个目标命中率执行一次完整、彼此隔离的 cache 压测。

    先清空两层 cache，以裁剪后的 prompt 填充内存层，再只清 GPU 层并发送
    完整 prompt。统计窗口只包含最后的完整 prompt 波次，填充耗时不会混入
    吞吐结果。
    """
    total_chunks = total_len // page_size
    hit_chunks = sample_hit_chunks(total_chunks, target_hit_rate, rng)
    if target_hit_rate >= 1.0:
        warmup_len = total_len
    elif target_hit_rate <= 0.0:
        warmup_len = 0
    else:
        warmup_len = hit_chunks * page_size

    # 两层 cache 分开清理：内存层必须走 clear_all_cache 的 memory_only
    # 模式，确保接口会遍历该实例的所有节点，而非只清理入口所在节点。
    print("  Clearing GPU cache and memory cache on all nodes ...")
    try:
        flush_cache(base_url)
        clear_memory_cache(base_url)
    except Exception as e:
        return {
            "target_hit_rate": target_hit_rate,
            "error": f"initial cache clear failed: {e}",
        }
    await asyncio.sleep(0.5)

    print(
        f"  Cache-fill wave: prompt_len={warmup_len}, hit_chunks={hit_chunks}/"
        f"{total_chunks}, target_hit_rate={target_hit_rate:.4f}"
    )

    semaphore = asyncio.Semaphore(max_concurrency)
    all_results: List[dict] = []
    warmup_results: List[dict] = []
    warmup_elapsed = 0.0

    # ------------------------------------------------------------------
    # QPS 模式：开放式泊松到达，用于观测指定 offered load 下的吞吐和延迟。
    # 若要估算容量上限，应传多个递增 QPS，直到 actual_qps 不再增长。
    # ------------------------------------------------------------------
    if qps is not None:
        np_rng = np.random.default_rng(rng.randint(0, 2**31))
        schedule = build_qps_schedule(qps, duration, np_rng)
        print(
            f"  QPS mode: target_qps={qps}, duration={duration}s, "
            f"requests_per_wave={len(schedule)}"
        )

        warmup_start = time.perf_counter()
        # 目标命中率为 0 时不需要填充；直接跳过，避免向 completions 接口发送
        # 空 prompt，也避免 marker-only 请求意外制造一个可命中的 cache page。
        if warmup_len > 0:
            warmup_results = await send_requests_qps(
                api_url=api_url,
                model=model,
                output_len=output_len,
                vocab_size=vocab_size,
                total_len=warmup_len,
                page_size=page_size,
                target_hit_rate=target_hit_rate,
                first_token_base=first_token_base,
                arange_offset=arange_offset,
                qps=qps,
                duration=duration,
                semaphore=semaphore,
                rng=rng,
                np_rng=np_rng,
                schedule=schedule,
                bootstrap_host=bootstrap_host,
                bootstrap_port=bootstrap_port,
                dp_size=dp_size,
            )
        warmup_elapsed = time.perf_counter() - warmup_start
        warmup_success = sum(r["success"] for r in warmup_results)
        if warmup_success != len(warmup_results):
            return {
                "target_hit_rate": target_hit_rate,
                "error": (
                    f"cache-fill wave failed: {warmup_success}/"
                    f"{len(warmup_results)} requests succeeded"
                ),
            }
        print(
            f"  Cache-fill complete: {warmup_success}/{len(warmup_results)} "
            f"requests in {warmup_elapsed:.2f}s"
        )

        print("  Flushing GPU cache (host cache retained) ...")
        try:
            flush_cache(base_url)
        except Exception as e:
            return {
                "target_hit_rate": target_hit_rate,
                "error": f"flush_cache failed: {e}",
            }
        await asyncio.sleep(0.5)

        # 清完 GPU cache 后，把对应数量的 host 内存 page 换出到 SSD，
        # 从而在随后的完整波中构造指定的 SSD 命中率。
        ssd_info = None
        if ssd_hit_rate > 0.0 and warmup_len > 0:
            try:
                ssd_info = maybe_flush_to_ssd(
                    ssd_base_url=ssd_base_url,
                    ssd_namespace=ssd_namespace,
                    ssd_hit_rate=ssd_hit_rate,
                    total_chunks=total_chunks,
                    hit_chunks=hit_chunks,
                    num_fill_requests=len(warmup_results),
                )
            except Exception as e:
                return {
                    "target_hit_rate": target_hit_rate,
                    "error": f"flushBlocksToSSD failed: {e}",
                }
            await asyncio.sleep(0.5)

        print(f"  Full-length wave: prompt_len={total_len} ...")
        bench_start = time.perf_counter()
        all_results = await send_requests_qps(
            api_url=api_url,
            model=model,
            output_len=output_len,
            vocab_size=vocab_size,
            total_len=total_len,
            page_size=page_size,
            target_hit_rate=target_hit_rate,
            first_token_base=first_token_base,
            arange_offset=arange_offset,
            qps=qps,
            duration=duration,
            semaphore=semaphore,
            rng=rng,
            np_rng=np_rng,
            schedule=schedule,
            bootstrap_host=bootstrap_host,
            bootstrap_port=bootstrap_port,
            dp_size=dp_size,
        )
        bench_elapsed = time.perf_counter() - bench_start

        successful_all = [r for r in all_results if r["success"]]
        failed = len(all_results) - len(successful_all)
        total_requests = len(all_results)

        if not successful_all:
            return {"target_hit_rate": target_hit_rate, "error": "All requests failed"}
        # 只有全部成功请求都没返回 prompt_tokens_details 才说明服务端没开
        # --enable-cache-report；个别请求命中为 0 时该字段本就缺失（引擎侧
        # _details_if_cached 在 count==0 时返回 None），不应误判成配置错误。
        if warmup_len > 0 and all(r.get("_no_cache_report") for r in successful_all):
            return {
                "target_hit_rate": target_hit_rate,
                "error": (
                    "server did not return cached_tokens; restart it with "
                    "--enable-cache-report so memory-cache hits can be verified"
                ),
            }

        total_prompt_tokens = sum(r["prompt_tokens"] for r in successful_all)
        total_cached_tokens = sum(r["cached_tokens"] for r in successful_all)
        total_completion_tokens = sum(r["completion_tokens"] for r in successful_all)
        actual_hit_rate = (
            total_cached_tokens / total_prompt_tokens
            if total_prompt_tokens > 0
            else 0.0
        )
        uncached_tokens = total_prompt_tokens - total_cached_tokens

        ttfts = [r["ttft"] for r in successful_all if r["ttft"] > 0]
        latencies = [r["latency"] for r in successful_all if r["latency"] > 0]
        mean_ttft = float(np.mean(ttfts)) * 1000 if ttfts else 0.0
        p90_ttft = float(np.percentile(ttfts, 90)) * 1000 if ttfts else 0.0
        p99_ttft = float(np.percentile(ttfts, 99)) * 1000 if ttfts else 0.0
        mean_latency = float(np.mean(latencies)) * 1000 if latencies else 0.0
        p90_latency = float(np.percentile(latencies, 90)) * 1000 if latencies else 0.0
        p99_latency = float(np.percentile(latencies, 99)) * 1000 if latencies else 0.0

        cached_tput = total_cached_tokens / bench_elapsed if bench_elapsed > 0 else 0.0
        uncached_tput = uncached_tokens / bench_elapsed if bench_elapsed > 0 else 0.0
        total_prefill_tput = (
            total_prompt_tokens / bench_elapsed if bench_elapsed > 0 else 0.0
        )
        actual_qps = total_requests / bench_elapsed if bench_elapsed > 0 else 0.0

        all_hit_chunks = [r["cached_tokens"] // page_size for r in successful_all]
        hit_chunks_hist = {
            k: all_hit_chunks.count(k) for k in sorted(set(all_hit_chunks))
        }

        print(
            f"  QPS result: sent={total_requests} ok={len(successful_all)} fail={failed} "
            f"actual_qps={actual_qps:.2f} hit_rate={actual_hit_rate:.4f} "
            f"ttft_mean={mean_ttft:.1f}ms"
        )

        return {
            "mode": "qps",
            "target_hit_rate": target_hit_rate,
            "ssd_hit_rate": ssd_hit_rate,
            "ssd_flush": ssd_info,
            "cache_fill_len": warmup_len,
            "cache_fill_requests": len(warmup_results),
            "cache_fill_successful": warmup_success,
            "cache_fill_time_s": warmup_elapsed,
            "actual_hit_rate": actual_hit_rate,
            "target_qps": qps,
            "actual_qps": actual_qps,
            "duration_s": duration,
            "total_chunks": total_chunks,
            "page_size": page_size,
            "total_len": total_len,
            "total_requests": total_requests,
            "total_tokens": total_prompt_tokens,
            "cached_tokens": total_cached_tokens,
            "uncached_tokens": uncached_tokens,
            "completion_tokens": total_completion_tokens,
            "cached_token_throughput": cached_tput,
            "uncached_token_throughput": uncached_tput,
            "total_prefill_throughput": total_prefill_tput,
            "mean_ttft_ms": mean_ttft,
            "p90_ttft_ms": p90_ttft,
            "p99_ttft_ms": p99_ttft,
            "mean_latency_ms": mean_latency,
            "p90_latency_ms": p90_latency,
            "p99_latency_ms": p99_latency,
            "total_time_s": bench_elapsed,
            "num_successful": len(successful_all),
            "num_failed": failed,
            "hit_chunks_histogram": hit_chunks_hist,
            "per_request": all_results,
        }

    # ------------------------------------------------------------------
    # Batch 模式：闭环地持续提交并发 batch，更适合直接压出服务容量上限。
    # 每一轮使用全新的 req_id，避免测量波内部前后请求相互命中。
    # ------------------------------------------------------------------
    per_round_stats: List[dict] = []
    global_req_id = 0  # monotonically increasing across all rounds

    warmup_start = time.perf_counter()
    if warmup_len > 0:
        for _ in range(num_rounds):
            warmup_prompts, _ = build_batch(
                vocab_size=vocab_size,
                total_len=warmup_len,
                page_size=page_size,
                target_hit_rate=target_hit_rate,
                batch_size=batch_size,
                req_id_offset=global_req_id,
                rng=rng,
                first_token_base=first_token_base,
                arange_offset=arange_offset,
            )
            results = await send_batch(
                api_url,
                model,
                output_len,
                warmup_prompts,
                semaphore,
                req_id_offset=global_req_id,
                bootstrap_host=bootstrap_host,
                bootstrap_port=bootstrap_port,
                dp_size=dp_size,
            )
            warmup_results.extend(results)
            global_req_id += len(warmup_prompts)

    warmup_elapsed = time.perf_counter() - warmup_start
    warmup_success = sum(r["success"] for r in warmup_results)
    if warmup_success != len(warmup_results):
        return {
            "target_hit_rate": target_hit_rate,
            "error": (
                f"cache-fill wave failed: {warmup_success}/"
                f"{len(warmup_results)} requests succeeded"
            ),
        }
    print(
        f"  Cache-fill complete: {warmup_success}/{len(warmup_results)} "
        f"requests in {warmup_elapsed:.2f}s"
    )

    print("  Flushing GPU cache (host cache retained) ...")
    try:
        flush_cache(base_url)
    except Exception as e:
        return {
            "target_hit_rate": target_hit_rate,
            "error": f"flush_cache failed: {e}",
        }
    await asyncio.sleep(0.5)

    # 清完 GPU cache 后，把对应数量的 host 内存 page 换出到 SSD，
    # 从而在随后的完整波中构造指定的 SSD 命中率。
    ssd_info = None
    if ssd_hit_rate > 0.0 and warmup_len > 0:
        try:
            ssd_info = maybe_flush_to_ssd(
                ssd_base_url=ssd_base_url,
                ssd_namespace=ssd_namespace,
                ssd_hit_rate=ssd_hit_rate,
                total_chunks=total_chunks,
                hit_chunks=hit_chunks,
                num_fill_requests=len(warmup_results),
            )
        except Exception as e:
            return {
                "target_hit_rate": target_hit_rate,
                "error": f"flushBlocksToSSD failed: {e}",
            }
        await asyncio.sleep(0.5)

    print(f"  Full-length wave: prompt_len={total_len} ...")
    global_req_id = 0
    bench_start = time.perf_counter()

    for round_idx in range(num_rounds):
        batch_prompts, expected_hit_chunks = build_batch(
            vocab_size=vocab_size,
            total_len=total_len,
            page_size=page_size,
            target_hit_rate=target_hit_rate,
            batch_size=batch_size,
            req_id_offset=global_req_id,
            rng=rng,
            first_token_base=first_token_base,
            arange_offset=arange_offset,
        )

        round_start = time.perf_counter()
        results = await send_batch(
            api_url,
            model,
            output_len,
            batch_prompts,
            semaphore,
            req_id_offset=global_req_id,
            bootstrap_host=bootstrap_host,
            bootstrap_port=bootstrap_port,
            dp_size=dp_size,
        )
        global_req_id += len(batch_prompts)
        round_elapsed = time.perf_counter() - round_start

        successful = [r for r in results if r["success"]]
        r_prompt = sum(r["prompt_tokens"] for r in successful)
        r_cached = sum(r["cached_tokens"] for r in successful)
        r_hit_rate = r_cached / r_prompt if r_prompt > 0 else 0.0
        r_ttfts = [r["ttft"] for r in successful if r["ttft"] > 0]

        # Per-request hit_chunks distribution
        actual_hit_chunks = [r["cached_tokens"] // page_size for r in successful]
        hit_chunks_dist = sorted(set(actual_hit_chunks))

        per_round_stats.append(
            {
                "round": round_idx,
                "prompt_tokens": r_prompt,
                "cached_tokens": r_cached,
                "hit_rate": r_hit_rate,
                "expected_hit_chunks_mean": float(np.mean(expected_hit_chunks)),
                "actual_hit_chunks_dist": hit_chunks_dist,
                "elapsed_s": round_elapsed,
                "num_successful": len(successful),
            }
        )

        print(
            f"  round {round_idx:02d}  "
            f"hit_rate={r_hit_rate:.3f}  "
            f"cached={r_cached}/{r_prompt}  "
            f"hit_chunks_dist={hit_chunks_dist}  "
            f"ttft_mean={np.mean(r_ttfts) * 1000 if r_ttfts else 0.0:.1f}ms  "
            f"elapsed={round_elapsed:.2f}s"
        )

        # Tag each result with its expected hit_chunks
        for i, r in enumerate(results):
            r["expected_hit_chunks"] = expected_hit_chunks[i]
        all_results.extend(results)

    bench_elapsed = time.perf_counter() - bench_start

    # 4. Aggregate
    successful_all = [r for r in all_results if r["success"]]
    failed = len(all_results) - len(successful_all)

    if not successful_all:
        return {"target_hit_rate": target_hit_rate, "error": "All requests failed"}
    # 同上：全部请求都缺 prompt_tokens_details 才是服务端没开 cache-report。
    if warmup_len > 0 and all(r.get("_no_cache_report") for r in successful_all):
        return {
            "target_hit_rate": target_hit_rate,
            "error": (
                "server did not return cached_tokens; restart it with "
                "--enable-cache-report so memory-cache hits can be verified"
            ),
        }

    total_prompt_tokens = sum(r["prompt_tokens"] for r in successful_all)
    total_cached_tokens = sum(r["cached_tokens"] for r in successful_all)
    total_completion_tokens = sum(r["completion_tokens"] for r in successful_all)
    actual_hit_rate = (
        total_cached_tokens / total_prompt_tokens if total_prompt_tokens > 0 else 0.0
    )
    uncached_tokens = total_prompt_tokens - total_cached_tokens

    ttfts = [r["ttft"] for r in successful_all if r["ttft"] > 0]
    mean_ttft = float(np.mean(ttfts)) * 1000 if ttfts else 0.0
    p90_ttft = float(np.percentile(ttfts, 90)) * 1000 if ttfts else 0.0
    p99_ttft = float(np.percentile(ttfts, 99)) * 1000 if ttfts else 0.0

    cached_tput = total_cached_tokens / bench_elapsed if bench_elapsed > 0 else 0.0
    uncached_tput = uncached_tokens / bench_elapsed if bench_elapsed > 0 else 0.0
    total_prefill_tput = (
        total_prompt_tokens / bench_elapsed if bench_elapsed > 0 else 0.0
    )

    # Per-request hit_chunks distribution across all rounds
    all_hit_chunks = [r["cached_tokens"] // page_size for r in successful_all]
    hit_chunks_hist = {k: all_hit_chunks.count(k) for k in sorted(set(all_hit_chunks))}

    return {
        "mode": "batch",
        "target_hit_rate": target_hit_rate,
        "ssd_hit_rate": ssd_hit_rate,
        "ssd_flush": ssd_info,
        "cache_fill_len": warmup_len,
        "cache_fill_requests": len(warmup_results),
        "cache_fill_successful": warmup_success,
        "cache_fill_time_s": warmup_elapsed,
        "actual_hit_rate": actual_hit_rate,
        "total_chunks": total_chunks,
        "page_size": page_size,
        "num_rounds": num_rounds,
        "batch_size": batch_size,
        "total_len": total_len,
        "total_tokens": total_prompt_tokens,
        "cached_tokens": total_cached_tokens,
        "uncached_tokens": uncached_tokens,
        "completion_tokens": total_completion_tokens,
        "cached_token_throughput": cached_tput,
        "uncached_token_throughput": uncached_tput,
        "total_prefill_throughput": total_prefill_tput,
        "mean_ttft_ms": mean_ttft,
        "p90_ttft_ms": p90_ttft,
        "p99_ttft_ms": p99_ttft,
        "total_time_s": bench_elapsed,
        "num_successful": len(successful_all),
        "num_failed": failed,
        "hit_chunks_histogram": hit_chunks_hist,
        "per_round": per_round_stats,
        "per_request": all_results,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_table(all_stats: List[dict]):
    """将所有命中率压测结果打印为终端表格。"""
    mode = all_stats[0].get("mode", "batch") if all_stats else "batch"

    if mode == "qps":
        header = (
            f"{'target_hit':>10} {'actual_hit':>10} "
            f"{'target_qps':>10} {'actual_qps':>10} "
            f"{'chunks':>6} {'chunk_sz':>8} "
            f"{'total_tok':>10} {'cached_tok':>10} {'uncached_tok':>12} "
            f"{'cached_tput':>12} {'uncached_tput':>13} {'prefill_tput':>12} "
            f"{'ttft_mean':>10} {'ttft_p90':>9} {'ttft_p99':>9} "
            f"{'lat_mean':>9} {'lat_p90':>8} {'lat_p99':>8} "
            f"{'time_s':>7}"
        )
        sep = "-" * len(header)
        print(sep)
        print(header)
        print(sep)
        for s in all_stats:
            if "error" in s:
                print(f"  target={s['target_hit_rate']:.2f}  ERROR: {s['error']}")
                continue
            print(
                f"{s['target_hit_rate']:>10.2f} {s['actual_hit_rate']:>10.4f} "
                f"{s['target_qps']:>10.2f} {s['actual_qps']:>10.2f} "
                f"{s['total_chunks']:>6} {s['page_size']:>8} "
                f"{s['total_tokens']:>10} {s['cached_tokens']:>10} {s['uncached_tokens']:>12} "
                f"{s['cached_token_throughput']:>12.1f} {s['uncached_token_throughput']:>13.1f} "
                f"{s['total_prefill_throughput']:>12.1f} "
                f"{s['mean_ttft_ms']:>10.1f} {s['p90_ttft_ms']:>9.1f} {s['p99_ttft_ms']:>9.1f} "
                f"{s['mean_latency_ms']:>9.1f} {s['p90_latency_ms']:>8.1f} {s['p99_latency_ms']:>8.1f} "
                f"{s['total_time_s']:>7.2f}"
            )
        print(sep)
    else:
        header = (
            f"{'target_hit':>10} {'actual_hit':>10} "
            f"{'chunks':>6} {'chunk_sz':>8} "
            f"{'total_tok':>10} {'cached_tok':>10} {'uncached_tok':>12} "
            f"{'cached_tput':>12} {'uncached_tput':>13} {'prefill_tput':>12} "
            f"{'ttft_mean':>10} {'ttft_p90':>9} {'ttft_p99':>9} "
            f"{'time_s':>7}"
        )
        sep = "-" * len(header)
        print(sep)
        print(header)
        print(sep)
        for s in all_stats:
            if "error" in s:
                print(f"  target={s['target_hit_rate']:.2f}  ERROR: {s['error']}")
                continue
            print(
                f"{s['target_hit_rate']:>10.2f} {s['actual_hit_rate']:>10.4f} "
                f"{s['total_chunks']:>6} {s['page_size']:>8} "
                f"{s['total_tokens']:>10} {s['cached_tokens']:>10} {s['uncached_tokens']:>12} "
                f"{s['cached_token_throughput']:>12.1f} {s['uncached_token_throughput']:>13.1f} "
                f"{s['total_prefill_throughput']:>12.1f} "
                f"{s['mean_ttft_ms']:>10.1f} {s['p90_ttft_ms']:>9.1f} {s['p99_ttft_ms']:>9.1f} "
                f"{s['total_time_s']:>7.2f}"
            )
        print(sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main(args):
    """解析运行配置，依次执行各命中率和 QPS 组合的压测。"""
    base_url = f"http://{args.host}:{args.port}"
    api_url = f"{base_url}/v1/completions"

    model = args.model
    if model is None:
        print("Auto-detecting model from server...")
        model = get_model_from_server(base_url)
        print(f"  model = {model}")

    tokenizer_path = args.tokenizer or model
    if args.vocab_size is not None:
        vocab_size = args.vocab_size
        print(f"Using explicit vocab_size = {vocab_size}")
    else:
        print(f"Reading vocab size from {tokenizer_path} ...")
        vocab_size = get_vocab_size(tokenizer_path)
        print(f"  vocab_size = {vocab_size}")

    page_size = get_page_size(base_url)
    print(f"  server page_size = {page_size}")

    # Each DP rank has its own prefix cache. Detect server dp_size and pin every
    # request to rank = req_id % dp_size, so the same req_id in a later run lands
    # on the same rank and hits the cache populated earlier.
    dp_size = get_dp_size(base_url)
    if args.dp_size and args.dp_size > 0:
        dp_size = args.dp_size
    print(f"  server dp_size = {dp_size}  (data_parallel_rank = req_id % {dp_size})")

    # Use server page_size as page_size if not explicitly set
    page_size = args.page_size if args.page_size is not None else page_size
    if page_size <= 0:
        page_size = page_size if page_size > 0 else 1

    total_chunks = args.total_len // page_size
    if total_chunks == 0:
        raise ValueError(
            f"total_len={args.total_len} is smaller than page_size={page_size}. "
            "Increase --total-len or decrease --page-size."
        )

    target_hit_rates = [float(x.strip()) for x in args.target_hit_rates.split(",")]

    qps_list = (
        [float(x.strip()) for x in args.qps.split(",")]
        if args.qps is not None
        else None
    )

    if qps_list is not None:
        print(
            f"\nConfig [QPS mode]: total_len={args.total_len}, page_size={page_size}, "
            f"total_chunks={total_chunks}, output_len={args.output_len}, "
            f"qps={qps_list}, duration={args.duration}s, "
            f"max_concurrency={args.max_concurrency}, "
            f"bootstrap={args.bootstrap_host}:{args.bootstrap_port}"
        )
    else:
        print(
            f"\nConfig [Batch mode]: total_len={args.total_len}, page_size={page_size}, "
            f"total_chunks={total_chunks}, output_len={args.output_len}, "
            f"num_rounds={args.num_rounds}, batch_size={args.batch_size}, "
            f"max_concurrency={args.max_concurrency}, "
            f"bootstrap={args.bootstrap_host}:{args.bootstrap_port}"
        )
    print(f"Target hit rates: {target_hit_rates}\n")

    # SSD 换出配置：当 --ssd-hit-rate > 0 时，在每个命中率的测量波之前，把对应
    # 数量的 host 内存 page 换出到 SSD，构造指定的 SSD 命中率。
    ssd_hit_rate = args.ssd_hit_rate
    ssd_base_url = None
    ssd_namespace = args.ssd_namespace
    if ssd_hit_rate > 0.0:
        if not ssd_namespace:
            raise ValueError(
                "--ssd-hit-rate > 0 时必须同时指定 --ssd-namespace。"
            )
        ssd_host = args.ssd_flush_host or args.host
        ssd_base_url = f"http://{ssd_host}:{args.ssd_flush_port}"
        print(
            f"SSD flush enabled: ssd_hit_rate={ssd_hit_rate}, "
            f"endpoint={ssd_base_url}/op/flushBlocksToSSD, "
            f"namespace={ssd_namespace}\n"
        )

    rng = random.Random(args.seed)
    all_stats = []

    if qps_list is not None:
        for hr in target_hit_rates:
            for qps in qps_list:
                print(f"=== hit_rate={hr:.2f}  qps={qps} ===")
                stats = await run_one_hit_rate(
                    base_url=base_url,
                    api_url=api_url,
                    model=model,
                    vocab_size=vocab_size,
                    target_hit_rate=hr,
                    total_len=args.total_len,
                    page_size=page_size,
                    output_len=args.output_len,
                    num_rounds=args.num_rounds,
                    batch_size=args.batch_size,
                    max_concurrency=args.max_concurrency,
                    rng=rng,
                    bootstrap_host=args.bootstrap_host,
                    bootstrap_port=args.bootstrap_port,
                    first_token_base=args.first_token_base,
                    arange_offset=args.arange_offset,
                    dp_size=dp_size,
                    qps=qps,
                    duration=args.duration,
                    ssd_base_url=ssd_base_url,
                    ssd_namespace=ssd_namespace,
                    ssd_hit_rate=ssd_hit_rate,
                )
                all_stats.append(stats)
                if "error" not in stats:
                    print(
                        f"  -> actual_hit_rate={stats['actual_hit_rate']:.4f}  "
                        f"actual_qps={stats['actual_qps']:.2f}  "
                        f"prefill_tput={stats['total_prefill_throughput']:.1f} tok/s  "
                        f"mean_ttft={stats['mean_ttft_ms']:.1f} ms"
                    )
                    print(
                        f"     hit_chunks_histogram: {stats['hit_chunks_histogram']}\n"
                    )
    else:
        for hr in target_hit_rates:
            print(f"=== hit_rate={hr:.2f} ===")
            stats = await run_one_hit_rate(
                base_url=base_url,
                api_url=api_url,
                model=model,
                vocab_size=vocab_size,
                target_hit_rate=hr,
                total_len=args.total_len,
                page_size=page_size,
                output_len=args.output_len,
                num_rounds=args.num_rounds,
                batch_size=args.batch_size,
                max_concurrency=args.max_concurrency,
                rng=rng,
                bootstrap_host=args.bootstrap_host,
                bootstrap_port=args.bootstrap_port,
                first_token_base=args.first_token_base,
                arange_offset=args.arange_offset,
                dp_size=dp_size,
                qps=None,
                duration=args.duration,
                ssd_base_url=ssd_base_url,
                ssd_namespace=ssd_namespace,
                ssd_hit_rate=ssd_hit_rate,
            )
            all_stats.append(stats)
            if "error" not in stats:
                print(
                    f"  -> actual_hit_rate={stats['actual_hit_rate']:.4f}  "
                    f"prefill_tput={stats['total_prefill_throughput']:.1f} tok/s  "
                    f"mean_ttft={stats['mean_ttft_ms']:.1f} ms"
                )
                print(f"     hit_chunks_histogram: {stats['hit_chunks_histogram']}\n")

    print("\n=== Summary ===")
    print_table(all_stats)

    output_path = args.output_file or f"bench_cache_hit_rate_{int(time.time())}.json"
    summary = [
        {k: v for k, v in s.items() if k not in ("per_request", "per_round")}
        for s in all_stats
    ]
    output_data = {
        "args": vars(args),
        "model": model,
        "page_size": page_size,
        "summary": summary,
        "details": all_stats,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark host-cache hit rate using a self-contained clear, cache-fill, "
            "GPU-flush, and full-length measurement flow. Cache-fill prompts are "
            "page-aligned to each target hit rate."
        )
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name. Auto-detected from /v1/models if not set.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer/model path used to read vocab_size. Defaults to --model.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Explicit vocabulary size. Bypasses tokenizer/config loading.",
    )
    parser.add_argument(
        "--total-len",
        type=int,
        default=59000,
        help="Fixed prompt token length for every request.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=64,
        help="Prefix-cache chunk size (page granularity). "
        "Defaults to server page_size from /get_server_info.",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=1,
        help="Tokens to generate per request. Keep at 1 to focus on prefill.",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=1,
        help="Number of rounds (batches) in the benchmark time window.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Concurrent requests per round."
    )
    parser.add_argument(
        "--target-hit-rates",
        type=str,
        default="0.85",
        help="Comma-separated list of target cache hit rates.",
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=256, help="Maximum concurrent requests."
    )
    parser.add_argument(
        "--output-file", type=str, default=None, help="Path for JSON output file."
    )
    parser.add_argument(
        "--seed", type=int, default=int(time.time()), help="Random seed."
    )
    parser.add_argument(
        "--bootstrap-host",
        type=str,
        default="2.2.2.2",
        help="bootstrap_host field in every request payload.",
    )
    parser.add_argument(
        "--bootstrap-port",
        type=int,
        default=8998,
        help="bootstrap_port field in every request payload.",
    )
    parser.add_argument(
        "--qps",
        type=str,
        default=None,
        help="Comma-separated target request rates. When set, uses Poisson "
        "arrival QPS mode instead of the default saturation-oriented batch "
        "mode. Pass multiple increasing values to find the saturation point.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=600.0,
        help="Benchmark duration in seconds (QPS mode only). Default: 600.",
    )
    parser.add_argument(
        "--first-token-base",
        type=int,
        default=1000,
        help="Base for per-request marker token (marker = base + req_index). "
        "The cache-fill and measured waves use the same marker.",
    )
    parser.add_argument(
        "--arange-offset",
        type=int,
        default=10000,
        help="Starting token id of the arange body. Both waves use the same "
        "offset so the cropped prompt is a strict prefix of the full prompt.",
    )
    parser.add_argument(
        "--dp-size",
        type=int,
        default=0,
        help="Data-parallel size. 0 = auto-detect from /get_server_info. "
        "Each request is pinned to data_parallel_rank = req_id %% dp_size, so "
        "both waves send the same req_id to the same rank and prefix cache.",
    )
    parser.add_argument(
        "--ssd-hit-rate",
        type=float,
        default=0.0,
        help="Target SSD cache hit rate, as a fraction of the WHOLE prompt's "
        "pages (ssd_pages_per_request = round(ssd_hit_rate * total_chunks)). "
        "When > 0, after filling the host cache and flushing the GPU cache, "
        "flushBlocksToSSD is called to swap that many pages per request out to "
        "SSD. Must be <= --target-hit-rate (clamped with a warning otherwise). "
        "0 disables SSD flushing.",
    )
    parser.add_argument(
        "--ssd-namespace",
        type=str,
        default=None,
        help="namespace passed to /op/flushBlocksToSSD. Required when "
        "--ssd-hit-rate > 0.",
    )
    parser.add_argument(
        "--ssd-flush-host",
        type=str,
        default=None,
        help="Host of the flushBlocksToSSD service. Defaults to --host.",
    )
    parser.add_argument(
        "--ssd-flush-port",
        type=int,
        default=8080,
        help="Port of the flushBlocksToSSD service. Default: 8080.",
    )

    args = parser.parse_args()
    asyncio.run(main(args))
