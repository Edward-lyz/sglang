"""压测设备侧 HBM cache 命中场景下的 prefill 吞吐。

为什么需要这个脚本
------------------
``bench_ascache.py`` 在填充 cache 后会把设备侧 KV cache flush 掉，只保留
host DRAM 里的内存 cache，因此它只能测“内存 cache 命中”，测不到 HBM
(device / radix) cache 命中。本脚本反过来：预热后不 flush 设备 cache，让
测量波直接命中 HBM 里的前缀。

方法：共享前缀 + 唯一后缀
------------------------
每个目标命中率只预热一次：

1. 清空设备 cache；host 内存 cache 也尽量清空。
2. 按目标命中率算出共享前缀长度 ``prefix_len``（对齐到 fill_block），在
   每个 DP rank 上各发一个只含该前缀的请求，把它写入 HBM。
3. 再清一次 host cache（可选），避免 device miss 时被 host 兜底。
4. 连续发 ``num_rounds * batch_size`` 个测量请求：
   ``prompt = 共享前缀 + 每个请求独有的后缀``。
   所有请求命中同一个前缀，后缀互不相同，因此不会互相命中到 100%。

这样每个目标只预热一次，而不是每个请求、每一轮都重新 fill 一遍。HBM 只需
容纳一份共享前缀 + 并发后缀，测量波可以连续跑，也更接近真实的前缀缓存
场景（大量请求共享同一段 system prompt / 文档前缀）。

命中率量化
----------
引擎按 page / AttentionStore block 对齐前缀，``prefix_len`` 只能取
``fill_block`` 的整数倍。目标命中率会四舍五入到最近的块，脚本会计算实际
命中率并检查偏差：

* ``total_len=8192, block=256, target=0.96`` -> 31/32 = 0.96875（偏差
  0.00875，默认容差内）
* ``total_len=4096, block=256, target=0.96`` -> 15/16 = 0.9375（偏差
  0.0225，超过默认容差，报错）
* ``total_len=2048, block=256, target=0.96`` -> 8/8 = 1.0（直接填满，
  和测量 prompt 完全相同，报错）

偏差超过 ``--hit-rate-tolerance``（默认 0.01）时脚本直接报错，避免把
“填满”当成 0.96、实际测到 100% 命中。

要精确测 0.96，可用 ``--total-len 6400``（25 个 256 块 -> 24/25 = 0.96，
后缀也是 256 token）。

Usage
-----
    python3 benchmark/bench_hbmcache.py --host 127.0.0.1 --port 30000 \\
        --total-len 8192 --batch-size 8 --num-rounds 4 \\
        --target-hit-rates "0.0,0.5,0.9"

注意
----
* 只支持 batch 模式；不支持 ``--qps``。开环请求无法保证测量时前缀还在 HBM。
* 预热波的时间不计入吞吐（``cache_fill_time_s`` 单独记录）；它模拟测量前 cache
  已被之前的请求填充好。真正的测量请求只计算未命中的后缀。
* 非零命中率需要能拿到缓存命中数：优先用 ``usage.cached_tokens``；服务端
  开启 HiCache / AsRadixCache 时也会回传 ``cached_tokens_details``，可从中
  推导。两者都拿不到时需要服务端加 ``--enable-cache-report``。
* 本脚本测的是“同一前缀被大量请求复用”的场景。所有并发请求会读同一段
  前缀 KV（包括 L2），对应“共享 system prompt / 文档前缀”的负载；如果要
  测每个请求各自独立前缀，需要为每个前缀单独预热，不适合用这个脚本。
"""

import argparse
import asyncio
import json
import os
import random
import subprocess
import time
from typing import List, Tuple

import aiohttp
import numpy as np
import requests

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=20 * 60 * 60)

# Suffix marker = FIRST_TOKEN_BASE + req_index，保证每个测量请求的后缀不同。
FIRST_TOKEN_BASE = 1000
ARANGE_OFFSET = 10000
# 共享前缀使用另一段 token，避免和后缀 marker 混淆（位置不同其实也不会混，
# 但分开更直观）。
PREFIX_TOKEN_BASE = 50000

# AttentionStore 落盘块粒度（python/sglang/srt/mem_cache/storage/asradix/types.py
# 的 AS_BLOCK_SIZE）。AS 开启时本地 device 前缀也会被对齐到该粒度，因此前缀
# 长度必须与之对齐。
DEFAULT_AS_BLOCK_SIZE = 256


# ---------------------------------------------------------------------------
# nsys capture control
# ---------------------------------------------------------------------------


def _run_nsys(cmd_args: List[str]) -> None:
    """Issue an nsys control command, logging failures without aborting the bench."""
    cmd = ["nsys"] + cmd_args
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except Exception as e:  # nsys missing, session gone, etc. — never fail the run.
        print(f"  [nsys] failed to run {' '.join(cmd)}: {e}")
        return
    if proc.returncode != 0:
        print(
            f"  [nsys] {' '.join(cmd_args)} -> rc={proc.returncode}: "
            f"{(proc.stderr or proc.stdout).strip()}"
        )
    else:
        print(f"  [nsys] {' '.join(cmd_args)} ok")


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


def get_server_info(base_url: str) -> dict:
    """读取 /get_server_info；失败时返回空 dict，让调用方走默认值。"""
    try:
        resp = requests.get(f"{base_url}/get_server_info", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


def get_page_size(base_url: str, info: dict = None) -> int:
    """从服务端读取 prefix cache 的 page size，查询失败时返回 1。"""
    info = get_server_info(base_url) if info is None else info
    page_size = info.get("page_size", 1)
    return page_size if isinstance(page_size, int) and page_size > 0 else 1


def get_dp_size(base_url: str, info: dict = None) -> int:
    """Detect server data-parallel size.

    Each DP rank has its own prefix cache, so the shared prefix must be warmed
    on every rank, and each request is pinned to one rank deterministically.
    """
    info = get_server_info(base_url) if info is None else info
    for key in ("data_parallel_size", "dp_size", "data_parallel_world_size"):
        value = info.get(key)
        if value:
            return int(value)
    return 1


def get_fill_block_size(
    base_url: str,
    page_size: int,
    force_as_cache: bool = None,
    info: dict = None,
) -> int:
    """返回前缀长度必须对齐的粒度。

    优先用服务端上报的 attention_store_block_size；老版本服务端没有该字段
    时，看它是否开了 enable_asradix_cache 并用编译期常量兜底；最后退回
    page_size。
    """
    info = get_server_info(base_url) if info is None else info
    for key in ("attention_store_block_size", "as_block_size"):
        value = info.get(key)
        if isinstance(value, int) and value > 0:
            return value
    as_on = info.get("enable_asradix_cache")
    if as_on is None:
        as_on = force_as_cache
    return DEFAULT_AS_BLOCK_SIZE if as_on else page_size


def get_token_capacity(server_info: dict) -> int:
    """返回每个 DP rank 的设备 KV pool 容量（token 数），读不到时返回 0。"""
    capacities = []
    for state in server_info.get("internal_states") or []:
        memory_usage = (state or {}).get("memory_usage") or {}
        capacity = memory_usage.get("token_capacity")
        if isinstance(capacity, int) and capacity > 0:
            capacities.append(capacity)
    return min(capacities) if capacities else 0


def flush_device_cache(base_url: str):
    """只清理设备侧 KV cache（HBM），保留 host 内存 cache。"""
    resp = requests.post(
        f"{base_url}/flush_cache",
        params={"timeout": 5},
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"flush_cache returned HTTP {resp.status_code}: {resp.text}")


def clear_memory_cache(
    base_url: str, max_rounds: int = 3, delay_s: float = 1.0
) -> bool:
    """尽力清理各节点的 host 内存 cache；失败只告警，不阻塞 HBM 压测。

    ``/clear_all_cache?memory_only=true`` 会遍历该实例所有节点，并检查回读
    是否为空。HBM 模式不依赖 host 层，所以这里不清空也继续；但残留的 host
    前缀可能成为 device miss 时的回退，脚本会在统计里单独暴露 host 命中。
    """
    for round_index in range(1, max_rounds + 1):
        try:
            resp = requests.post(
                f"{base_url}/clear_all_cache",
                params={"memory_only": "true"},
                timeout=5 * 60,
            )
        except Exception as e:
            print(f"  [warn] host-cache clear unavailable: {e}")
            return False
        if resp.status_code == 200:
            return True
        if resp.status_code != 207:
            print(
                f"  [warn] host-cache clear returned HTTP {resp.status_code}: "
                f"{resp.text[:200]}"
            )
            return False
        if round_index < max_rounds:
            time.sleep(delay_s)
    print(
        "  [warn] host cache still non-empty after "
        f"{max_rounds} rounds; continuing (host hits are reported separately)"
    )
    return False


# ---------------------------------------------------------------------------
# Prompt construction: one shared prefix + per-request unique suffix
# ---------------------------------------------------------------------------


def quantize_prefix_len(
    total_len: int, target_hit_rate: float, fill_block: int
) -> Tuple[int, float, int, int]:
    """把目标命中率量化到 ``fill_block`` 的整数倍。

    Returns
    -------
    prefix_len : 对齐后的共享前缀长度
    actual_rate: prefix_len / total_len
    fill_blocks: total_len // fill_block
    prefix_blocks: 实际填充的块数
    """
    fill_blocks = total_len // fill_block
    prefix_blocks = int(target_hit_rate * fill_blocks + 0.5)
    prefix_blocks = max(0, min(prefix_blocks, fill_blocks))
    prefix_len = prefix_blocks * fill_block
    actual_rate = prefix_len / total_len if total_len > 0 else 0.0
    return prefix_len, actual_rate, fill_blocks, prefix_blocks


def build_shared_prefix(
    prefix_len: int, vocab_size: int, prefix_base: int = PREFIX_TOKEN_BASE
) -> List[int]:
    """构造所有测量请求共享的前缀。"""
    return [(prefix_base + i) % vocab_size for i in range(prefix_len)]


def build_unique_suffix(
    req_index: int,
    suffix_len: int,
    vocab_size: int,
    first_token_base: int = FIRST_TOKEN_BASE,
    arange_offset: int = ARANGE_OFFSET,
) -> List[int]:
    """构造每个请求独有的后缀；第一个 token 是 marker，避免互相命中。"""
    if suffix_len <= 0:
        return []
    marker = (first_token_base + req_index) % vocab_size
    body = [(arange_offset + i) % vocab_size for i in range(suffix_len - 1)]
    return [marker] + body


def build_prompt(
    prefix_ids: List[int],
    req_index: int,
    suffix_len: int,
    vocab_size: int,
    first_token_base: int = FIRST_TOKEN_BASE,
    arange_offset: int = ARANGE_OFFSET,
) -> List[int]:
    """``共享前缀 + 唯一后缀``，长度恰好为 prefix_len + suffix_len。"""
    return prefix_ids + build_unique_suffix(
        req_index,
        suffix_len,
        vocab_size,
        first_token_base,
        arange_offset,
    )


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
    # Pin to a deterministic DP rank: each rank has its own prefix cache.
    dp_rank = req_id % dp_size if dp_size > 1 else 0
    payload = {
        "model": model,
        "rid": f"hbm-{random.randint(0, 0xFFFFFFFF):08x}",
        "prompt": prompt_ids,
        "max_tokens": output_len,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
        "return_cached_tokens_details": True,
        "bootstrap_host": bootstrap_host,
        "bootstrap_port": bootstrap_port,
        "bootstrap_room": req_id,
        "data_parallel_rank": dp_rank,
    }

    result = {
        "rid": payload["rid"],
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
                cache_details = (data.get("sglext") or {}).get("cached_tokens_details")
                if cache_details is not None:
                    result["cached_tokens_details"] = cache_details

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
            result["success"] = (
                result["prompt_tokens"] == len(prompt_ids)
                and result["completion_tokens"] >= output_len
            )
            if not result["success"]:
                result["error"] = (
                    "incomplete response usage: "
                    f"prompt={result['prompt_tokens']}/{len(prompt_ids)}, "
                    f"completion={result['completion_tokens']}/{output_len}"
                )

    except Exception as e:
        result["error"] = str(e)

    return result


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
    # 一个波次复用同一个连接池，避免建连开销污染吞吐。
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:

        async def _one(i, ids):
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


# ---------------------------------------------------------------------------
# Cache-tier accounting
# ---------------------------------------------------------------------------


def cache_breakdown(result: dict) -> Tuple[int, int, int]:
    """返回单个请求的 (device, host, storage) 命中 token 数。

    服务端只有在开启 HiCache / AsRadixCache 时才会回传
    ``cached_tokens_details``；两层都没开时不存在 host cache，所有命中必然
    来自设备侧 HBM。若服务端给出的拆分和 cached_tokens 对不上，按“全部是
    设备命中”处理，不凭空算到 host 上。
    """
    cached = int(result.get("cached_tokens", 0) or 0)
    details = result.get("cached_tokens_details")
    if not isinstance(details, dict):
        return cached, 0, 0
    device = int(details.get("device", 0) or 0)
    host = int(details.get("host", 0) or 0)
    storage = int(details.get("storage", 0) or 0)
    if device + host + storage != cached:
        if cached == 0 and device + host + storage > 0:
            # --enable-cache-report is off, so usage.cached_tokens is missing,
            # but the sglext tier split is still usable.
            return device, host, storage
        return cached, 0, 0
    return device, host, storage


def sum_cache_breakdown(results: List[dict]) -> Tuple[int, int, int]:
    """汇总一组请求的 (device, host, storage) 命中 token 数。"""
    device = host = storage = 0
    for result in results:
        d, h, s = cache_breakdown(result)
        device += d
        host += h
        storage += s
    return device, host, storage


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
    bootstrap_host: str = "2.2.2.2",
    bootstrap_port: int = 10000,
    first_token_base: int = FIRST_TOKEN_BASE,
    arange_offset: int = ARANGE_OFFSET,
    dp_size: int = 1,
    fill_block_size: int = None,
    nsys_session: str = None,
    clear_host: bool = True,
    hit_rate_tolerance: float = 0.01,
    token_capacity: int = 0,
) -> dict:
    """针对一个目标命中率执行一次共享前缀的 HBM cache 压测。

    只预热一次共享前缀，然后连续测量 ``num_rounds * batch_size`` 个带唯一
    后缀的请求；测量波不 flush 设备 cache。
    """
    fill_block = (
        fill_block_size if fill_block_size and fill_block_size > 0 else page_size
    )
    if fill_block <= 0 or total_len < fill_block:
        return {
            "target_hit_rate": target_hit_rate,
            "error": (
                f"total_len={total_len} must be at least one prefix block "
                f"(fill_block={fill_block})"
            ),
        }
    if not 0.0 <= target_hit_rate <= 1.0:
        return {
            "target_hit_rate": target_hit_rate,
            "error": f"target_hit_rate must be in [0, 1], got {target_hit_rate}",
        }
    prefix_len, actual_hit_rate, fill_blocks, prefix_blocks = quantize_prefix_len(
        total_len, target_hit_rate, fill_block
    )
    suffix_len = total_len - prefix_len

    # 目标命中率无法用当前块粒度表示时直接失败，避免“填满”被当成 0.96、
    # 实际测到 100% 命中。
    if target_hit_rate < 1.0 and prefix_len >= total_len:
        return {
            "target_hit_rate": target_hit_rate,
            "error": (
                f"target {target_hit_rate:.4f} rounds up to a full prompt at "
                f"fill_block={fill_block} ({prefix_blocks}/{fill_blocks} blocks); "
                f"the warm prefix and measured prompt would be identical. "
                f"Increase --total-len or use target <= "
                f"{(fill_blocks - 1) / fill_blocks:.4f}."
            ),
        }
    if (
        target_hit_rate < 1.0
        and abs(actual_hit_rate - target_hit_rate) > hit_rate_tolerance
    ):
        return {
            "target_hit_rate": target_hit_rate,
            "error": (
                f"target {target_hit_rate:.4f} is not representable at "
                f"fill_block={fill_block}: {prefix_blocks}/{fill_blocks} blocks "
                f"= {actual_hit_rate:.4f} (deviation "
                f"{abs(actual_hit_rate - target_hit_rate):.4f} > tolerance "
                f"{hit_rate_tolerance}). Increase --total-len or raise "
                f"--hit-rate-tolerance."
            ),
        }

    if target_hit_rate >= 1.0 and suffix_len > 0:
        print(
            f"  [note] target 1.0 capped by block alignment: "
            f"{prefix_len}/{total_len} = {actual_hit_rate:.4f}"
        )

    print(
        f"  Cache-fill wave: prompt_len={prefix_len} (shared prefix), "
        f"hit_chunks={prefix_blocks}/{fill_blocks}, fill_block={fill_block}, "
        f"target_hit_rate={target_hit_rate:.4f}"
    )

    # 一轮测量波的工作集：一份共享前缀 + 并发后缀。旧后缀会被 radix 树按
    # LRU 回收，不会按 num_rounds 成倍累积。
    per_rank_batch = (batch_size + dp_size - 1) // dp_size
    demand = prefix_len + per_rank_batch * suffix_len
    if token_capacity > 0 and demand > token_capacity:
        print(
            f"  [warn] one measured round needs about {demand} KV tokens per "
            f"DP rank (prefix={prefix_len} + {per_rank_batch} suffixes x "
            f"{suffix_len}) but the device pool holds {token_capacity}; the "
            f"prefix may be evicted. Reduce --batch-size / --total-len."
        )

    print("  Clearing GPU cache ...")
    try:
        flush_device_cache(base_url)
    except Exception as e:
        return {
            "target_hit_rate": target_hit_rate,
            "error": f"flush_cache failed: {e}",
        }
    if clear_host:
        print("  Clearing host memory cache (best effort) ...")
        clear_memory_cache(base_url)

    semaphore = asyncio.Semaphore(max_concurrency)
    prefix_ids = build_shared_prefix(prefix_len, vocab_size)
    warm_requests = 0
    warm_successful = 0
    warm_elapsed = 0.0

    # 共享前缀在每个 DP rank 上各预热一次。req_id = 0..dp_size-1 正好把
    # data_parallel_rank 固定到每个 rank。
    if prefix_len > 0:
        print(
            f"  Warming shared prefix on {dp_size} DP rank(s): {prefix_len} tokens ..."
        )
        warm_prompts = [prefix_ids] * dp_size
        warm_start = time.perf_counter()
        warm_results = await send_batch(
            api_url,
            model,
            output_len,
            warm_prompts,
            semaphore,
            req_id_offset=0,
            bootstrap_host=bootstrap_host,
            bootstrap_port=bootstrap_port,
            dp_size=dp_size,
        )
        warm_elapsed = time.perf_counter() - warm_start
        warm_requests = len(warm_results)
        warm_successful = sum(r["success"] for r in warm_results)
        if warm_successful != warm_requests:
            return {
                "target_hit_rate": target_hit_rate,
                "error": (
                    f"prefix warm wave failed: {warm_successful}/"
                    f"{warm_requests} requests succeeded"
                ),
            }
        print(
            f"  Cache-fill complete: {warm_successful}/{warm_requests} "
            f"requests in {warm_elapsed:.2f}s"
        )

        # 预热会把前缀同时写进 host 内存 cache。清掉 host 副本后，测量波的
        # device miss 只能重算，不会悄悄从 host 兜底，HBM 归因更干净。
        if clear_host:
            print("  Clearing host memory cache after warm (best effort) ...")
            clear_memory_cache(base_url)

    await asyncio.sleep(0.3)

    per_round_stats: List[dict] = []
    all_results: List[dict] = []

    print(f"  Full-length wave: prompt_len={total_len} ...")

    # 共享前缀已经 warm 好，测量波连续跑；不再逐轮 fill。
    if nsys_session:
        _run_nsys(["start", f"--session={nsys_session}"])
    try:
        measurement_started_at_unix = time.time()
        bench_start = time.perf_counter()

        for round_idx in range(num_rounds):
            req_id_offset = round_idx * batch_size
            prompts = [
                build_prompt(
                    prefix_ids,
                    req_id_offset + i,
                    suffix_len,
                    vocab_size,
                    first_token_base,
                    arange_offset,
                )
                for i in range(batch_size)
            ]

            round_start = time.perf_counter()
            results = await send_batch(
                api_url,
                model,
                output_len,
                prompts,
                semaphore,
                req_id_offset=req_id_offset,
                bootstrap_host=bootstrap_host,
                bootstrap_port=bootstrap_port,
                dp_size=dp_size,
            )
            round_elapsed = time.perf_counter() - round_start

            all_results.extend(results)

            successful = [r for r in results if r["success"]]
            r_prompt = sum(r["prompt_tokens"] for r in successful)
            r_device, r_host, r_storage = sum_cache_breakdown(successful)
            r_cached = r_device + r_host + r_storage
            r_hit_rate = r_cached / r_prompt if r_prompt > 0 else 0.0
            r_device_rate = r_device / r_prompt if r_prompt > 0 else 0.0
            r_ttfts = [r["ttft"] for r in successful if r["ttft"] > 0]
            actual_hit_chunks = [
                sum(cache_breakdown(r)) // page_size for r in successful
            ]

            per_round_stats.append(
                {
                    "round": round_idx,
                    "prompt_tokens": r_prompt,
                    "cached_tokens": r_cached,
                    "hit_rate": r_hit_rate,
                    "expected_hit_chunks_mean": float(prefix_len // page_size),
                    "actual_hit_chunks_dist": sorted(set(actual_hit_chunks)),
                    "elapsed_s": round_elapsed,
                    "num_successful": len(successful),
                    "device_cached_tokens": r_device,
                    "host_cached_tokens": r_host,
                    "storage_cached_tokens": r_storage,
                    "device_hit_rate": r_device_rate,
                }
            )

            print(
                f"  round {round_idx:02d}  hit_rate={r_hit_rate:.3f}  "
                f"cached={r_cached}/{r_prompt}  "
                f"hit_chunks_dist={sorted(set(actual_hit_chunks))}  "
                f"ttft_mean={np.mean(r_ttfts) * 1000 if r_ttfts else 0.0:.1f}ms  "
                f"elapsed={round_elapsed:.2f}s  "
                f"hbm_hit_rate={r_device_rate:.3f}"
            )

            for result in results:
                result["expected_prompt_tokens"] = total_len
                result["expected_hit_chunks"] = prefix_len // page_size

        bench_elapsed = time.perf_counter() - bench_start
        measurement_finished_at_unix = time.time()
    finally:
        if nsys_session:
            _run_nsys(["stop", f"--session={nsys_session}"])

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    successful_all = [r for r in all_results if r["success"]]
    failed = len(all_results) - len(successful_all)
    if not successful_all:
        return {"target_hit_rate": target_hit_rate, "error": "All requests failed"}

    total_device_tokens, total_host_tokens, total_storage_tokens = sum_cache_breakdown(
        successful_all
    )
    if (
        prefix_len > 0
        and all(r.get("_no_cache_report") for r in successful_all)
        and total_device_tokens + total_host_tokens + total_storage_tokens == 0
    ):
        return {
            "target_hit_rate": target_hit_rate,
            "error": (
                "server returned neither cached_tokens nor "
                "cached_tokens_details; restart it with --enable-cache-report "
                "so HBM-cache hits can be verified"
            ),
        }

    total_prompt_tokens = sum(r["prompt_tokens"] for r in successful_all)
    total_cached_tokens = total_device_tokens + total_host_tokens + total_storage_tokens
    total_completion_tokens = sum(r["completion_tokens"] for r in successful_all)
    actual_hit_rate = (
        total_cached_tokens / total_prompt_tokens if total_prompt_tokens > 0 else 0.0
    )
    device_hit_rate = (
        total_device_tokens / total_prompt_tokens if total_prompt_tokens > 0 else 0.0
    )
    host_hit_rate = (
        total_host_tokens / total_prompt_tokens if total_prompt_tokens > 0 else 0.0
    )
    fallback_tokens = total_host_tokens + total_storage_tokens
    fallback_rate = (
        fallback_tokens / total_prompt_tokens if total_prompt_tokens > 0 else 0.0
    )
    uncached_tokens = total_prompt_tokens - total_cached_tokens

    # 目标前缀非 0、也确实预热成功，却一个 device 命中都没有：前缀在测量前
    # 就被挤出了设备 pool。此时吞吐数据没有意义。
    if prefix_len > 0 and warm_successful > 0 and total_device_tokens == 0:
        detail = (
            f" ({fallback_tokens} hits were served by the host/storage tier)"
            if fallback_tokens > 0
            else ""
        )
        return {
            "target_hit_rate": target_hit_rate,
            "error": (
                "no device (HBM) cache hits: the warmed prefix was evicted from "
                "the device KV pool before measurement"
                f"{detail}; reduce --batch-size / --total-len or enlarge the "
                "KV pool"
            ),
        }
    if fallback_tokens > 0:
        print(
            f"  [warn] {fallback_tokens} cached tokens ({fallback_rate:.2%}) came "
            f"from the host/storage tier, so this run is not a pure HBM "
            f"measurement; see per_round host/storage_cached_tokens."
        )
    expected_device_rate = prefix_len / total_len if total_len > 0 else 0.0
    if prefix_len > 0 and device_hit_rate + 0.02 < expected_device_rate:
        print(
            f"  [warn] HBM hit rate {device_hit_rate:.4f} is below the expected "
            f"{expected_device_rate:.4f}; the shared prefix may be partially "
            f"evicted from the device pool."
        )
    print(
        f"  HBM (device) hit rate = {device_hit_rate:.4f}  "
        f"(host+storage fallback = {fallback_rate:.4f})"
    )

    ttfts = [r["ttft"] for r in successful_all if r["ttft"] > 0]
    mean_ttft = float(np.mean(ttfts)) * 1000 if ttfts else 0.0
    p90_ttft = float(np.percentile(ttfts, 90)) * 1000 if ttfts else 0.0
    p99_ttft = float(np.percentile(ttfts, 99)) * 1000 if ttfts else 0.0

    cached_tput = total_cached_tokens / bench_elapsed if bench_elapsed > 0 else 0.0
    device_tput = total_device_tokens / bench_elapsed if bench_elapsed > 0 else 0.0
    uncached_tput = uncached_tokens / bench_elapsed if bench_elapsed > 0 else 0.0
    total_prefill_tput = (
        total_prompt_tokens / bench_elapsed if bench_elapsed > 0 else 0.0
    )

    all_hit_chunks = [sum(cache_breakdown(r)) // page_size for r in successful_all]
    hit_chunks_hist = {k: all_hit_chunks.count(k) for k in sorted(set(all_hit_chunks))}

    return {
        "mode": "batch",
        "cache_tier": "device",
        "prompt_layout": "shared_prefix",
        "target_hit_rate": target_hit_rate,
        "fill_block_size": fill_block,
        "cache_fill_len": prefix_len,
        "cache_fill_requests": warm_requests,
        "cache_fill_successful": warm_successful,
        "cache_fill_time_s": warm_elapsed,
        "actual_hit_rate": actual_hit_rate,
        "total_chunks": total_len // page_size,
        "page_size": page_size,
        "num_rounds": num_rounds,
        "batch_size": batch_size,
        "total_len": total_len,
        "total_lens": None,
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
        "measurement_started_at_unix": measurement_started_at_unix,
        "measurement_finished_at_unix": measurement_finished_at_unix,
        "num_successful": len(successful_all),
        "num_failed": failed,
        "hit_chunks_histogram": hit_chunks_hist,
        "device_hit_rate": device_hit_rate,
        "host_hit_rate": host_hit_rate,
        "fallback_hit_rate": fallback_rate,
        "device_cached_tokens": total_device_tokens,
        "host_cached_tokens": total_host_tokens,
        "storage_cached_tokens": total_storage_tokens,
        "device_token_throughput": device_tput,
        "prefix_len": prefix_len,
        "suffix_len": suffix_len,
        "per_round": per_round_stats,
        "per_request": all_results,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_table(all_stats: List[dict]):
    """将所有命中率压测结果打印为终端表格。

    前 14 列与 bench_ascache.py 的 batch 表格保持一致，HBM 专属列追加在
    末尾，方便两个脚本的结果直接对照。
    """
    header = (
        f"{'target_hit':>10} {'actual_hit':>10} "
        f"{'chunks':>6} {'chunk_sz':>8} "
        f"{'total_tok':>10} {'cached_tok':>10} {'uncached_tok':>12} "
        f"{'cached_tput':>12} {'uncached_tput':>13} {'prefill_tput':>12} "
        f"{'ttft_mean':>10} {'ttft_p90':>9} {'ttft_p99':>9} "
        f"{'time_s':>7} "
        f"{'hbm_hit':>8} {'prefix':>8} {'suffix':>7} "
        f"{'dev_tok':>10} {'host_tok':>9} {'dev_tput':>12} {'warm_s':>7}"
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
            f"{s['total_tokens']:>10} {s['cached_tokens']:>10} "
            f"{s['uncached_tokens']:>12} "
            f"{s['cached_token_throughput']:>12.1f} "
            f"{s['uncached_token_throughput']:>13.1f} "
            f"{s['total_prefill_throughput']:>12.1f} "
            f"{s['mean_ttft_ms']:>10.1f} {s['p90_ttft_ms']:>9.1f} "
            f"{s['p99_ttft_ms']:>9.1f} {s['total_time_s']:>7.2f} "
            f"{s['device_hit_rate']:>8.4f} {s['prefix_len']:>8} "
            f"{s['suffix_len']:>7} {s['device_cached_tokens']:>10} "
            f"{s['host_cached_tokens'] + s['storage_cached_tokens']:>9} "
            f"{s['device_token_throughput']:>12.1f} "
            f"{s['cache_fill_time_s']:>7.2f}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main(args):
    """依次执行各目标命中率的 HBM cache 压测。"""
    base_url = f"http://{args.host}:{args.port}"
    api_url = f"{base_url}/v1/completions"
    if args.num_rounds < 1:
        raise ValueError("--num-rounds must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.max_concurrency < 1:
        raise ValueError("--max-concurrency must be >= 1")

    server_info = get_server_info(base_url)
    if not server_info:
        print(
            "  [warn] /get_server_info unavailable; page_size/dp_size/capacity "
            "fall back to defaults; pass --page-size/--dp-size explicitly"
        )
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

    page_size = args.page_size
    if page_size is None:
        page_size = get_page_size(base_url, server_info)
    if page_size <= 0:
        page_size = 1
    print(f"  server page_size = {page_size}")

    dp_size = get_dp_size(base_url, server_info)
    if args.dp_size and args.dp_size > 0:
        dp_size = args.dp_size
    print(f"  server dp_size = {dp_size}  (data_parallel_rank = req_id % {dp_size})")

    fill_block_size = args.fill_block_size
    if fill_block_size <= 0:
        fill_block_size = get_fill_block_size(
            base_url, page_size, args.enable_asradix_cache, server_info
        )
    if fill_block_size < page_size:
        fill_block_size = page_size
    if fill_block_size % page_size:
        raise ValueError(
            f"--fill-block-size={fill_block_size} must be a multiple of "
            f"page_size={page_size}."
        )
    print(f"  prefix alignment (fill_block) = {fill_block_size}")

    if args.total_len // page_size == 0:
        raise ValueError(
            f"total_len={args.total_len} is smaller than page_size={page_size}. "
            "Increase --total-len or decrease --page-size."
        )
    if args.total_len // fill_block_size == 0:
        raise ValueError(
            f"total_len={args.total_len} is smaller than the prefix alignment "
            f"size={fill_block_size}. Increase --total-len or pass "
            "--fill-block-size to shrink it."
        )

    target_hit_rates = [float(x.strip()) for x in args.target_hit_rates.split(",")]

    clear_host = not args.skip_host_clear
    if clear_host:
        # 服务端明确报告两层 host cache 都没开时，不必再去戳 /clear_all_cache
        # （没有 host 服务时该接口要等连接超时，白花几十秒）。字段缺失的老板本
        # 服务端仍然尝试清理。
        host_flags = [
            server_info.get("enable_asradix_cache"),
            server_info.get("enable_hierarchical_cache"),
        ]
        known_flags = [flag for flag in host_flags if flag is not None]
        if known_flags and not any(known_flags):
            print("  no host cache tier detected; skipping host-cache clear")
            clear_host = False

    token_capacity = get_token_capacity(server_info)

    print(
        f"\nConfig [Batch mode]: cache_tier=device, "
        f"prompt_layout=shared_prefix, total_len={args.total_len}, "
        f"page_size={page_size}, fill_block_size={fill_block_size}, "
        f"total_chunks={args.total_len // page_size}, "
        f"output_len={args.output_len}, num_rounds={args.num_rounds}, "
        f"batch_size={args.batch_size}, "
        f"max_concurrency={args.max_concurrency}, "
        f"hit_rate_tolerance={args.hit_rate_tolerance}, "
        f"bootstrap={args.bootstrap_host}:{args.bootstrap_port}"
    )
    print(f"Target hit rates: {target_hit_rates}\n")

    all_stats = []
    for hit_rate in target_hit_rates:
        print(f"=== hit_rate={hit_rate:.2f} ===")
        stats = await run_one_hit_rate(
            base_url=base_url,
            api_url=api_url,
            model=model,
            vocab_size=vocab_size,
            target_hit_rate=hit_rate,
            total_len=args.total_len,
            page_size=page_size,
            output_len=args.output_len,
            num_rounds=args.num_rounds,
            batch_size=args.batch_size,
            max_concurrency=args.max_concurrency,
            bootstrap_host=args.bootstrap_host,
            bootstrap_port=args.bootstrap_port,
            first_token_base=args.first_token_base,
            arange_offset=args.arange_offset,
            dp_size=dp_size,
            fill_block_size=fill_block_size,
            nsys_session=args.nsys_session,
            clear_host=clear_host,
            hit_rate_tolerance=args.hit_rate_tolerance,
            token_capacity=token_capacity,
        )
        all_stats.append(stats)
        if "error" not in stats:
            print(
                f"  -> actual_hit_rate={stats['actual_hit_rate']:.4f}  "
                f"prefill_tput={stats['total_prefill_throughput']:.1f} tok/s  "
                f"mean_ttft={stats['mean_ttft_ms']:.1f} ms  "
                f"hbm_hit_rate={stats['device_hit_rate']:.4f}"
            )
            print(f"     hit_chunks_histogram: {stats['hit_chunks_histogram']}\n")

    print("\n=== Summary ===")
    print_table(all_stats)

    output_path = (
        args.output_file or f"bench_hbm_cache_hit_rate_{int(time.time())}.json"
    )
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
        "server_info": server_info,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_path}")
    if any(s.get("error") or s.get("num_failed", 0) for s in all_stats):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark HBM (device) cache hit throughput with a shared prefix. "
            "The prefix is warmed once per DP rank, then many requests reuse it "
            "with unique suffixes; the device cache is never flushed between "
            "the warm and the measured waves. Device/host hits are reported "
            "separately to catch host fallbacks."
        )
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server host.",
    )
    parser.add_argument("--port", type=int, default=30000, help="Server port.")
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
        default=8192,
        help="Full prompt token length. The shared prefix is "
        "target_hit_rate * total_len quantised to --fill-block-size; the "
        "remaining tokens are a per-request unique suffix.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=None,
        help="Prefix-cache page size. Defaults to server page_size.",
    )
    parser.add_argument(
        "--fill-block-size",
        type=int,
        default=0,
        help="Token granularity the shared prefix must be aligned to (0 = auto). "
        "AttentionStore aligns device prefix matches to its block size (256), "
        "so a prefix shorter than the aligned target silently caps the hit "
        "rate. Auto: the server's attention_store_block_size, else 256 when "
        "asradix cache is on, else page_size.",
    )
    parser.add_argument(
        "--hit-rate-tolerance",
        type=float,
        default=0.01,
        help="Maximum allowed |actual - target| hit rate after block "
        "quantisation. Targets that cannot be represented within this "
        "tolerance fail instead of silently measuring a different rate.",
    )
    parser.add_argument(
        "--enable-asradix-cache",
        action="store_true",
        default=None,
        help="Server runs with --enable-asradix-cache. Auto-detected from "
        "/get_server_info when omitted.",
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
        default=4,
        help="Measured batches. All rounds reuse the same warmed prefix; each "
        "request gets a fresh unique suffix, so rounds stay independent.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Concurrent requests per round. The prefix is shared, so the "
        "working set is prefix_len + batch_size/dp_size * suffix_len.",
    )
    parser.add_argument(
        "--target-hit-rates",
        type=str,
        default="0.0,0.5,0.9",
        help="Comma-separated list of target HBM hit rates.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=256,
        help="Maximum concurrent requests.",
    )
    parser.add_argument(
        "--skip-host-clear",
        action="store_true",
        help="Do not try to clear the host memory cache before/after warming. "
        "Use when the server has no host cache service; host hits are still "
        "reported separately.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path for the JSON output file.",
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
        "--first-token-base",
        type=int,
        default=FIRST_TOKEN_BASE,
        help="Base for the per-request suffix marker (marker = base + req_index).",
    )
    parser.add_argument(
        "--arange-offset",
        type=int,
        default=ARANGE_OFFSET,
        help="Starting token id of the suffix arange body.",
    )
    parser.add_argument(
        "--dp-size",
        type=int,
        default=0,
        help="Data-parallel size. 0 = auto-detect from /get_server_info. The "
        "shared prefix is warmed on every rank; each request is pinned to "
        "data_parallel_rank = req_id %% dp_size.",
    )
    parser.add_argument(
        "--nsys-session",
        type=str,
        default=None,
        help="Name of a running nsys session to bracket the measured waves "
        "with `nsys start`/`nsys stop`. Requires the bench to run on the same "
        "host as the session daemon. Omit to disable nsys control.",
    )

    args = parser.parse_args()
    asyncio.run(main(args))
