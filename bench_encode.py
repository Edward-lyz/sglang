"""Benchmarking tool for encoder and transfer endpoints.

Supports multiple modes:
- encode: call /encode endpoint
- transfer: /encode + /epd/transfer (RDMA)
- zmq: /encode + ZMQ DEALER pull
"""
import argparse
import io
import json
import math
import os
import pickle
import time
from typing import List, Any, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np
import requests
import torch
import zmq
from PIL import Image


def compute_embed_digest(t: torch.Tensor) -> dict:
    """Compute a lightweight digest of a tensor for transfer verification."""
    flat = t.reshape(-1).float()
    n = flat.numel()
    return {
        "mean": flat.mean().item(),
        "std": flat.std().item(),
        "min": flat.min().item(),
        "max": flat.max().item(),
        "first8": flat[:8].tolist(),
        "last8": flat[-8:].tolist(),
        "num_elements": n,
    }


def verify_embed_digest(local: dict, remote: dict, rtol: float = 1e-3, atol: float = 1e-5) -> bool:
    """Compare two embed digests. Returns True if they match within tolerance."""
    if local["num_elements"] != remote["num_elements"]:
        return False
    for key in ("mean", "std", "min", "max"):
        if not math.isclose(local[key], remote[key], rel_tol=rtol, abs_tol=atol):
            return False
    for l_vals, r_vals in ((local["first8"], remote["first8"]),
                           (local["last8"], remote["last8"])):
        for lv, rv in zip(l_vals, r_vals):
            if not math.isclose(lv, rv, rel_tol=rtol, abs_tol=atol):
                return False
    return True


class GenID:
    """Unique ID generator."""

    def __init__(self) -> None:
        self.counter = 0
        self.lock = threading.Lock()

    def gen(self) -> str:
        """Generate next unique ID."""
        with self.lock:
            self.counter += 1
            return f"cid-{self.counter}"

genid = GenID()


def load_all_images_from_jsonl(file_path: str) -> List[str]:
    """Load all image URLs from jsonl file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    image_urls = []
    for line in lines:
        try:
            data = json.loads(line)
            for msg in data.get('messages', []):
                if isinstance(msg.get('content'), list):
                    for item in msg['content']:
                        if item.get('type') == 'image_url':
                            url = item.get('image_url', {}).get('url')
                            if url:
                                image_urls.append(url)
                                break
        except json.JSONDecodeError:
            continue

    return image_urls


# ---------------------------------------------------------------------------
# Token estimation for Qwen-VL family (Qwen2-VL / Qwen3-VL / …)
# ---------------------------------------------------------------------------

def _round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
    max_ratio: int = 200,
) -> Tuple[int, int]:
    """Resize dimensions following Qwen-VL's smart_resize logic."""
    if max(height, width) / min(height, width) > max_ratio:
        raise ValueError(
            f"aspect ratio must be < {max_ratio}, "
            f"got {max(height, width) / min(height, width):.1f}"
        )
    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _floor_by_factor(height / beta, factor)
        w_bar = _floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def load_vision_config(model_path: str) -> Dict[str, Any]:
    """Load vision config from preprocessor_config.json / config.json.

    Auto-detects model family (qwen vs kimi_k25) from config.json architectures
    or vision_config.merge_kernel_size.
    """
    pp_path = os.path.join(model_path, "preprocessor_config.json")
    cfg_path = os.path.join(model_path, "config.json")

    pp = {}
    cfg = {}
    vc = {}

    if os.path.isfile(pp_path):
        with open(pp_path) as f:
            pp = json.load(f)

    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        vc = cfg.get("vision_config", {})

    # Detect model family
    archs = cfg.get("architectures", [])
    is_kimi = any("Kimi" in a or "K2" in a for a in archs) or "merge_kernel_size" in vc

    if is_kimi:
        patch_size = vc.get("patch_size", 14)
        if isinstance(patch_size, (list, tuple)):
            patch_size = patch_size[0]
        merge_kernel_size = vc.get("merge_kernel_size", [2, 2])
        if isinstance(merge_kernel_size, int):
            merge_kernel_size = [merge_kernel_size, merge_kernel_size]
        in_patch_limit = pp.get("in_patch_limit") or vc.get("in_patch_limit", 16384)
        patch_limit_on_one_side = pp.get("patch_limit_on_one_side") or vc.get("patch_limit_on_one_side", 512)
        return {
            "model_family": "kimi_k25",
            "patch_size": patch_size,
            "merge_kernel_size": merge_kernel_size,
            "in_patch_limit": in_patch_limit,
            "patch_limit_on_one_side": patch_limit_on_one_side,
        }
    else:
        # Qwen-VL family
        patch_size = pp.get("patch_size") or vc.get("patch_size", 14)
        merge_size = pp.get("merge_size") or vc.get("spatial_merge_size", 2)
        size = pp.get("size", {})
        min_pixels = size.get("shortest_edge")
        max_pixels = size.get("longest_edge")
        factor = patch_size * merge_size
        min_pixels = min_pixels or (4 * factor * factor)
        max_pixels = max_pixels or (16384 * factor * factor)
        return {
            "model_family": "qwen",
            "patch_size": patch_size,
            "merge_size": merge_size,
            "factor": factor,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
        }


# ---------------------------------------------------------------------------
# KimiK25: NaViT-style resize
# ---------------------------------------------------------------------------

def _navit_resize(height: int, width: int, vision_cfg: Dict[str, Any]) -> Tuple[int, int]:
    """Compute resized (new_h, new_w) following KimiK25's navit_resize_image logic."""
    patch_size = vision_cfg["patch_size"]
    in_patch_limit = vision_cfg["in_patch_limit"]
    patch_limit = vision_cfg["patch_limit_on_one_side"]
    max_px = patch_limit * patch_size  # 512 * 14 = 7168

    patches_h = max(1, height // patch_size)
    patches_w = max(1, width // patch_size)

    s1 = math.sqrt(in_patch_limit / (patches_w * patches_h))
    s2 = max_px / width
    s3 = max_px / height
    scale = min(1.0, s1, s2, s3)

    new_w = min(max(1, int(width * scale)), max_px)
    new_h = min(max(1, int(height * scale)), max_px)
    return new_h, new_w


def _estimate_tokens_kimi(height: int, width: int, vision_cfg: Dict[str, Any]) -> int:
    """KimiK25: num_tokens = token_h * token_w (temporal pooled away)."""
    new_h, new_w = _navit_resize(height, width, vision_cfg)
    merge_h, merge_w = vision_cfg["merge_kernel_size"]
    factor = vision_cfg["patch_size"] * merge_h  # assume merge_h == merge_w == 2
    factor_w = vision_cfg["patch_size"] * merge_w

    # pad to nearest multiple of factor, then divide
    token_h = math.ceil(new_h / factor)
    token_w = math.ceil(new_w / factor_w)
    return token_h * token_w


# ---------------------------------------------------------------------------
# Qwen-VL: smart_resize
# ---------------------------------------------------------------------------

def _estimate_tokens_qwen(height: int, width: int, vision_cfg: Dict[str, Any]) -> int:
    """Qwen-VL family: num_tokens = grid_h * grid_w // merge_size²."""
    h_bar, w_bar = smart_resize(
        height, width,
        factor=vision_cfg["factor"],
        min_pixels=vision_cfg["min_pixels"],
        max_pixels=vision_cfg["max_pixels"],
    )
    grid_h = h_bar // vision_cfg["patch_size"]
    grid_w = w_bar // vision_cfg["patch_size"]
    return grid_h * grid_w // (vision_cfg["merge_size"] ** 2)


def estimate_image_tokens(
    height: int,
    width: int,
    vision_cfg: Dict[str, Any],
) -> int:
    """Estimate the number of LLM tokens an image will produce."""
    if vision_cfg["model_family"] == "kimi_k25":
        return _estimate_tokens_kimi(height, width, vision_cfg)
    else:
        return _estimate_tokens_qwen(height, width, vision_cfg)


def _get_image_size(url: str) -> Tuple[int, int]:
    """Return (width, height) of an image URL / local path / base64 data URI."""
    if url.startswith("data:"):
        import base64
        # data:image/...;base64,<payload>
        payload = url.split(",", 1)[1]
        img = Image.open(io.BytesIO(base64.b64decode(payload)))
    elif url.startswith(("http://", "https://")):
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
    else:
        img = Image.open(url)
    return img.size  # (width, height)


def sample_images_by_token_len(
    image_urls: List[str],
    input_len: int,
    model_path: str,
) -> List[str]:
    """Filter image URLs whose estimated token count matches input_len."""
    vision_cfg = load_vision_config(model_path)
    print(f"Vision config ({vision_cfg['model_family']}): {vision_cfg}")

    matched = []
    for url in image_urls:
        try:
            w, h = _get_image_size(url)
            tokens = estimate_image_tokens(h, w, vision_cfg)
            if abs(tokens - input_len) <= input_len * 0.1:
                matched.append(url)
        except Exception as e:
            print(f"Warning: skip image {url[:80]}... : {e}")
            continue

    return matched


def build_encode_request(request_id: str, image_url: str) -> Dict[str, Any]:
    """Build encode request."""
    return {
        "req_id": request_id,
        "mm_items": [image_url],
        "num_parts": 1,
        "part_idx": 0,
        "prefill_host": None,
        "embedding_port": None,
    }


def http_post(endpoint: str, request: Dict[str, Any]):
    """Send HTTP POST request."""
    res = requests.post(endpoint, json=request)
    if res.status_code == 200:
        return res.json()
    else:
        raise ValueError(f"Request failed with status code {res.status_code}: {res.text}")


# ---------------------------------------------------------------------------
# Transfer mode: Mooncake RDMA buffer management
# ---------------------------------------------------------------------------

class TransferContext:
    """Manages Mooncake engine and a variable-size CPU pinned receive buffer pool.

    Uses GpuBufferPool (with device="cpu", pin_memory=True) for page-based
    allocation so that each RDMA transfer gets a contiguous region sized to its
    actual embedding, avoiding the waste of fixed-slot schemes.
    """

    def __init__(self, hostname: str, gpu_id: int, ib_device: str,
                 embed_dim: int, dtype: str,
                 pool_bytes: int, page_size: int = 2 * 1024 * 1024):
        from mooncake.engine import TransferEngine
        from sglang.srt.disaggregation.gpu_buffer_pool import GpuBufferPool

        self._engine = TransferEngine()
        ret = self._engine.initialize(hostname, "P2PHANDSHAKE", "rdma", ib_device or "")
        if ret != 0:
            raise RuntimeError("Mooncake TransferEngine init failed")

        self.session_id = f"{hostname}:{self._engine.get_rpc_port()}"
        self.embed_dim = embed_dim
        self.gpu_id = gpu_id
        self.dtype_str = dtype
        self.torch_dtype = getattr(torch, dtype)
        self.elem_size = torch.tensor([], dtype=self.torch_dtype).element_size()

        self._pool = GpuBufferPool(pool_bytes, page_size, device="cpu", pin_memory=True)

        ret = self._engine.register_memory(self._pool.data_ptr, self._pool.total_bytes)
        if ret != 0:
            raise RuntimeError(
                f"register_memory failed: 0x{self._pool.data_ptr:x}, "
                f"{self._pool.total_bytes}")
        print(f"TransferContext: registered {self._pool.total_bytes / 1e6:.1f} MB CPU pinned buffer "
              f"({self._pool.num_pages} pages x {page_size / 1e6:.1f} MB), "
              f"session_id={self.session_id}")

        # Fallback allocations: torch.empty buffers not from the pool
        self._fallback_bufs: Dict[int, torch.Tensor] = {}
        self._fallback_lock = threading.Lock()

    def acquire_buffer(self, byte_size: int) -> int:
        """Allocate a contiguous region. Tries the pool first; falls back to torch.empty."""
        ptr = self._pool.try_allocate(byte_size)
        if ptr is not None:
            return ptr

        # Fallback: allocate a temporary CPU pinned tensor and register with RDMA engine
        buf = torch.empty(byte_size, dtype=torch.uint8, device="cpu", pin_memory=True)
        ptr = buf.data_ptr()
        ret = self._engine.register_memory(ptr, byte_size)
        if ret != 0:
            del buf
            raise RuntimeError(
                f"register_memory failed for fallback buffer: 0x{ptr:x}, {byte_size}")
        with self._fallback_lock:
            self._fallback_bufs[ptr] = buf
        print(f"TransferContext: pool full, fallback CPU pinned torch.empty {byte_size} bytes")
        return ptr

    def release_buffer(self, ptr: int):
        """Return a previously acquired region to the pool or free fallback buffer."""
        with self._fallback_lock:
            buf = self._fallback_bufs.pop(ptr, None)
        if buf is not None:
            self._engine.unregister_memory(ptr)
            # buf goes out of scope -> tensor freed by GC
        else:
            self._pool.free(ptr)

    def view_buffer_as_tensor(self, ptr: int, byte_size: int) -> torch.Tensor:
        """Create a typed tensor view of a buffer region (for verification).

        The returned tensor references the same GPU memory — no copy.
        """
        with self._fallback_lock:
            fb = self._fallback_bufs.get(ptr)
        if fb is not None:
            raw = fb[:byte_size]
        else:
            offset = ptr - self._pool.data_ptr
            raw = self._pool._buf[offset:offset + byte_size]
        return raw.view(self.torch_dtype)

    def release(self):
        """Release memory resources."""
        self._engine.unregister_memory(self._pool.data_ptr)
        with self._fallback_lock:
            for ptr in list(self._fallback_bufs):
                self._engine.unregister_memory(ptr)
            self._fallback_bufs.clear()


# Global transfer context (set in main when mode=transfer)
_transfer_ctx: TransferContext = None

# Global ZMQ context (set in main when mode=zmq)
_zmq_ctx: Optional["ZmqContext"] = None


class ZmqContext:
    """Manages per-thread ZMQ DEALER sockets for benchmarking embedding pulls.

    ZMQ sockets are NOT thread-safe, so each thread gets its own DEALER
    socket via get_socket(). Sockets are cleaned up on close() or at thread
    exit via threading.atexit.
    """

    def __init__(self, host: str, port: int):
        self._ctx = zmq.Context()
        self.host = host
        self.port = port
        self._local = threading.local()
        self._lock = threading.Lock()
        self._sockets: list = []
        print(f"ZmqContext: will create per-thread DEALER sockets connecting to tcp://{host}:{port}")

    def _create_socket(self):
        sock = self._ctx.socket(zmq.DEALER)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RECONNECT_IVL, 500)
        sock.setsockopt(zmq.RECONNECT_IVL_MAX, 5000)
        sock.connect(f"tcp://{self.host}:{self.port}")
        with self._lock:
            self._sockets.append(sock)
        return sock

    def get_socket(self):
        """Return a DEALER socket for the current thread (created lazily)."""
        sock = getattr(self._local, "sock", None)
        if sock is None:
            sock = self._create_socket()
            self._local.sock = sock
        return sock

    def close(self):
        """Close all sockets and terminate context."""
        with self._lock:
            for sock in self._sockets:
                sock.close(linger=0)
            self._sockets.clear()
        self._ctx.term()


def _zmq_send_and_recv(sock, json_payload: dict, timeout_ms: int = 30000):
    """Send a ZMQ request and receive the reply on a given socket.

    Returns the payload frames (after empty, task_id, req_id),
    i.e. [pickle(EmbeddingData), (opt)buffer].

    Raises TimeoutError if no reply within timeout_ms.
    On timeout, drains any stale reply to keep socket state clean.
    """
    payload = json.dumps(json_payload).encode()
    sock.send_multipart([b"", payload])
    if sock.poll(timeout=timeout_ms, flags=zmq.POLLIN) == 0:
        # Timeout: drain any late reply that may arrive later to avoid
        # corrupting subsequent requests on this socket.
        if sock.poll(timeout=0, flags=zmq.POLLIN) != 0:
            sock.recv_multipart(copy=False)
        raise TimeoutError(
            f"ZMQ recv timeout after {timeout_ms}ms "
            f"for task_id={json_payload.get('task_id')}"
        )
    frames = sock.recv_multipart(copy=False)

    # DEALER receives: [b"", task_id, req_id, pickle(EmbeddingData), (opt)buffer]
    if len(frames) < 4:
        raise ValueError(
            f"Malformed ZMQ reply: expected >=4 frames, got {len(frames)}"
        )
    return frames[3:]


class TokenBucket:
    """Token bucket rate limiter. Call acquire() before each request."""

    def __init__(self, qps: float):
        self.qps = qps
        self.capacity = max(qps, 1.0)
        self._tokens = self.capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.capacity, self._tokens + elapsed * self.qps)
        self._last_refill = now

    def acquire(self):
        """Block until a token is available."""
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
            time.sleep(0.5 / self.qps)


def _do_one_encode_request(endpoint: str, image_url: str) -> Dict[str, Any]:
    """Send a single /encode request."""
    req_id = genid.gen()
    req = build_encode_request(req_id, image_url)
    start = time.perf_counter()
    try:
        resp = http_post(endpoint, req)
        latency = time.perf_counter() - start
        return {"req_id": req_id, "latency": latency, "response": resp}
    except Exception as e:
        latency = time.perf_counter() - start
        return {"req_id": req_id, "latency": latency, "error": str(e)}


def _do_one_video_request(endpoint: str, video_url: str) -> Dict[str, Any]:
    """Send a single /encode_video_gpu_chunked request."""
    req_id = genid.gen()
    payload = {
        "req_id": req_id,
        "task_id": req_id,
        "video_items": [video_url],
        "sample_fps": _video_sample_fps,
        "skip_cache": True,
    }
    start = time.perf_counter()
    try:
        resp = http_post(endpoint, payload)
        latency = time.perf_counter() - start
        return {"req_id": req_id, "latency": latency, "response": resp}
    except Exception as e:
        latency = time.perf_counter() - start
        return {"req_id": req_id, "latency": latency, "error": str(e)}


# Video mode config (set in main)
_video_sample_fps: float = 2.0


def _do_one_transfer_request(endpoint: str, image_url: str) -> Dict[str, Any]:
    """Send a single /epd/transfer request via Mooncake RDMA.

    两阶段:
      1. /encode — 触发 server 端 encode 并缓存，拿到精确的 embedding 元数据
      2. /epd/transfer — 根据精确 byte_size 从 slot 池借 GPU buffer，RDMA 传输
    """
    ctx = _transfer_ctx
    req_id = genid.gen()
    task_id = req_id
    buf_addr = None

    start = time.perf_counter()
    try:
        # Phase 1: encode, 拿到精确的 embedding_size
        encode_req = build_encode_request(req_id, image_url)
        encode_req["task_id"] = task_id
        encode_resp = http_post(endpoint + "/encode", encode_req)

        embedding_len = encode_resp.get("embedding_len", 0)
        embedding_dim = encode_resp.get("embedding_dim", ctx.embed_dim)
        byte_size = encode_resp.get("embedding_size", 0)
        if byte_size <= 0:
            byte_size = embedding_len * embedding_dim * ctx.elem_size

        # Phase 2: 借一个 slot，发起 RDMA transfer
        buf_addr = ctx.acquire_buffer(byte_size)
        transfer_req = {
            "req_id": req_id,
            "task_id": task_id,
            "part_idx": 0,
            "num_parts": 1,
            "session_id": ctx.session_id,
            "buffer_address": buf_addr,
            "byte_size": byte_size,
            "image_url": image_url,
        }
        transfer_resp = http_post(endpoint + "/epd/transfer", transfer_req)

        # Phase 3: 验证传输数据正确性
        verify_ok = None
        remote_digest = transfer_resp.get("embed_digest")
        if remote_digest and transfer_resp.get("status_code", 200) == 200:
            try:
                recv_tensor = ctx.view_buffer_as_tensor(buf_addr, byte_size)
                local_digest = compute_embed_digest(recv_tensor)
                verify_ok = verify_embed_digest(local_digest, remote_digest)
                if not verify_ok:
                    print(f"[WARN] embed verify FAIL req_id={req_id} "
                          f"local={local_digest} remote={remote_digest}")
            except Exception as e:
                print(f"[WARN] embed verify error req_id={req_id}: {e}")
                verify_ok = None

        latency = time.perf_counter() - start
        response = {
            "embedding_len": embedding_len,
            "embedding_dim": embedding_dim,
            "embedding_size": byte_size,
            "encode_cost": encode_resp.get("embed_cost", 0),
            "transfer_cost_ms": transfer_resp.get("transfer_cost_ms", 0),
            "total_cost_ms": transfer_resp.get("total_cost_ms", 0),
            "status_code": transfer_resp.get("status_code", 200),
            "verify_ok": verify_ok,
            "embed_digest": remote_digest,
        }
        return {"req_id": req_id, "latency": latency, "response": response}
    except Exception as e:
        latency = time.perf_counter() - start
        return {"req_id": req_id, "latency": latency, "error": str(e)}
    finally:
        if buf_addr is not None:
            ctx.release_buffer(buf_addr)


def _do_one_zmq_request(endpoint: str, image_url: str) -> Dict[str, Any]:
    """Send a single /encode request followed by a ZMQ DEALER pull.

    Two phases:
      1. /encode -- trigger server-side encode and cache the embedding
      2. ZMQ DEALER pull -- retrieve the embedding via ZMQ ROUTER
    """
    ctx = _zmq_ctx
    req_id = genid.gen()
    task_id = req_id

    total_start = time.perf_counter()
    try:
        # Phase 1: HTTP /encode to cache the embedding
        encode_start = time.perf_counter()
        encode_req = build_encode_request(req_id, image_url)
        encode_req["task_id"] = task_id
        encode_resp = http_post(endpoint + "/encode", encode_req)
        encode_time = time.perf_counter() - encode_start

        embedding_len = encode_resp.get("embedding_len", 0)
        embedding_dim = encode_resp.get("embedding_dim", 0)
        embedding_size = encode_resp.get("embedding_size", 0)

        # Phase 2: ZMQ DEALER pull
        zmq_start = time.perf_counter()
        zmq_payload = {
            "req_id": req_id,
            "task_id": task_id,
            "image_url": image_url,
            "num_parts": 1,
            "part_idx": 0,
        }
        sock = ctx.get_socket()
        payload_frames = _zmq_send_and_recv(sock, zmq_payload)
        zmq_time = time.perf_counter() - zmq_start

        # Parse the ZMQ response
        recv_obj = pickle.loads(payload_frames[0])

        # Check for errors from the server
        if getattr(recv_obj, "error_msg", None) is not None:
            return {
                "req_id": req_id,
                "latency": time.perf_counter() - total_start,
                "error": f"ZMQ server error: {recv_obj.error_msg}",
            }

        # Reconstruct the tensor (clone to own the memory, avoiding
        # use-after-free when the ZMQ frame buffer is recycled)
        verify_ok = None
        remote_digest = None
        if len(payload_frames) >= 2:
            buf = payload_frames[1].buffer if hasattr(payload_frames[1], "buffer") else payload_frames[1]
            embedding = torch.frombuffer(buf, dtype=recv_obj.dtype).reshape(recv_obj.shape).clone()

            # Verify against remote digest if available
            remote_digest = getattr(recv_obj, "embed_digest", None)
            try:
                local_digest = compute_embed_digest(embedding)
                if remote_digest is not None:
                    verify_ok = verify_embed_digest(local_digest, remote_digest)
                    if not verify_ok:
                        print(f"[WARN] ZMQ embed verify FAIL req_id={req_id} "
                              f"local={local_digest} remote={remote_digest}")
                else:
                    # No remote digest to compare; just confirm we can compute one
                    verify_ok = None
            except Exception as e:
                print(f"[WARN] ZMQ embed verify error req_id={req_id}: {e}")
                verify_ok = None

            # Update embedding_size from actual received data if not set
            if embedding_size <= 0:
                embedding_size = embedding.numel() * embedding.element_size()
            if embedding_dim <= 0:
                embedding_dim = embedding.shape[-1] if embedding.dim() > 0 else 0

        total_time = time.perf_counter() - total_start

        response = {
            "embedding_len": embedding_len,
            "embedding_dim": embedding_dim,
            "embedding_size": embedding_size,
            "encode_cost": encode_resp.get("embed_cost", 0),
            "encode_time": encode_time,
            "zmq_transfer_time": zmq_time,
            "verify_ok": verify_ok,
        }
        return {"req_id": req_id, "latency": total_time, "response": response}

    except Exception as e:
        total_time = time.perf_counter() - total_start
        return {"req_id": req_id, "latency": total_time, "error": str(e)}


def send_at_qps(
    endpoint: str,
    image_urls: List[str],
    qps: float,
    do_request=_do_one_encode_request,
):
    """Send requests to the endpoint at the specified QPS."""
    bucket = TokenBucket(qps)
    results = []
    errors = []

    def _rate_limited_request(url):
        bucket.acquire()
        return do_request(endpoint, url)

    with ThreadPoolExecutor(max_workers=max(int(qps * 2), 4)) as pool:
        futures = [pool.submit(_rate_limited_request, url) for url in image_urls]
        for f in as_completed(futures):
            result = f.result()
            if "error" in result:
                errors.append(result)
            else:
                results.append(result)

    return results, errors


def send_with_concurrency(
    endpoint: str,
    image_urls: List[str],
    concurrency: int,
    do_request=_do_one_encode_request,
):
    """Send requests to the endpoint with fixed concurrency."""
    results = []
    errors = []

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(do_request, endpoint, url) for url in image_urls]
        for f in as_completed(futures):
            result = f.result()
            if "error" in result:
                errors.append(result)
            else:
                results.append(result)

    return results, errors


def print_stats(results: List[Dict], errors: List[Dict], elapsed: float):
    """Print benchmark statistics."""
    print(errors)
    total = len(results) + len(errors)
    print(f"\n{'='*60}")
    print(f"Benchmark Results")
    print(f"{'='*60}")
    print(f"Total requests: {total}")
    print(f"  Success: {len(results)}")
    print(f"  Errors:  {len(errors)}")
    if results:
        latencies = sorted([r["latency"] for r in results])
        n = len(latencies)
        avg_latency = sum(latencies) / n
        p50 = latencies[int(n * 0.50)]
        p95 = latencies[int(n * 0.95)]
        p99 = latencies[int(n * 0.99)]

        token_counts = [r["response"].get("embedding_len", 0) for r in results]
        total_tokens = sum(token_counts)
        avg_input_tokens = total_tokens / n

        if elapsed > 0:
            throughput = total_tokens / elapsed
        else:
            throughput = float('inf')

        print(f"\nLatency (s):")
        print(f"  Average : {avg_latency:.4f}")
        print(f"  P50     : {p50:.4f}")
        print(f"  P95     : {p95:.4f}")
        print(f"  P99     : {p99:.4f}")
        print(f"  Min     : {latencies[0]:.4f}")
        print(f"  Max     : {latencies[-1]:.4f}")
        print(f"\nThroughput     : {throughput:.2f} tokens/s")
        print(f"Avg input tokens: {avg_input_tokens:.1f}")

        # Verification summary (transfer mode only)
        verify_results = [r["response"].get("verify_ok") for r in results
                          if "verify_ok" in r.get("response", {})]
        if verify_results:
            n_pass = sum(1 for v in verify_results if v is True)
            n_fail = sum(1 for v in verify_results if v is False)
            n_skip = sum(1 for v in verify_results if v is None)
            print(f"\nEmbed verification:")
            print(f"  Pass : {n_pass}")
            print(f"  Fail : {n_fail}")
            print(f"  Skip : {n_skip}")

    print(f"\nTotal elapsed  : {elapsed:.4f} s")
    print(f"{'='*60}")


def print_zmq_stats(results: List[Dict], elapsed: float):
    """Print ZMQ-specific benchmark statistics."""
    if not results:
        return

    print(f"\n{'='*60}")
    print(f"ZMQ Transfer Breakdown")
    print(f"{'='*60}")

    n = len(results)
    encode_times = sorted([r["response"]["encode_time"] for r in results])
    zmq_times = sorted([r["response"]["zmq_transfer_time"] for r in results])
    embedding_sizes = [r["response"]["embedding_size"] for r in results]

    print(f"\nEncode Time (HTTP /encode) (s):")
    print(f"  Average : {sum(encode_times)/n:.4f}")
    print(f"  P50     : {encode_times[int(n*0.50)]:.4f}")
    print(f"  P95     : {encode_times[int(n*0.95)]:.4f}")
    print(f"  P99     : {encode_times[int(n*0.99)]:.4f}")

    print(f"\nZMQ Transfer Time (DEALER pull) (s):")
    print(f"  Average : {sum(zmq_times)/n:.4f}")
    print(f"  P50     : {zmq_times[int(n*0.50)]:.4f}")
    print(f"  P95     : {zmq_times[int(n*0.95)]:.4f}")
    print(f"  P99     : {zmq_times[int(n*0.99)]:.4f}")

    # Bandwidth calculation
    total_bytes = sum(embedding_sizes)
    avg_embedding_size_mb = (total_bytes / n) / (1024 * 1024) if n > 0 else 0
    bandwidths = []
    for r in results:
        t = r["response"]["zmq_transfer_time"]
        s = r["response"]["embedding_size"]
        if t > 0 and s > 0:
            bandwidths.append(s / t / (1024 * 1024))  # MB/s

    if bandwidths:
        bw_sorted = sorted(bandwidths)
        bw_n = len(bw_sorted)
        print(f"\nZMQ Transfer Bandwidth (MB/s):")
        print(f"  Average : {sum(bw_sorted)/bw_n:.2f}")
        print(f"  P50     : {bw_sorted[int(bw_n*0.50)]:.2f}")
        print(f"  P95     : {bw_sorted[int(bw_n*0.95)]:.2f}")
        print(f"  P99     : {bw_sorted[int(bw_n*0.99)]:.2f}")

    print(f"\nAvg Embedding Size: {avg_embedding_size_mb:.2f} MB")
    print(f"{'='*60}")


def _print_video_stats(results: List[Dict], elapsed: float):
    """Print video-specific stage breakdown."""
    if not results:
        return
    n = len(results)
    load = sorted([r["response"].get("load_cost", 0) for r in results])
    decode = sorted([r["response"].get("frame_extraction_cost", 0) for r in results])
    process = sorted([r["response"].get("gpu_preprocess_cost", 0) for r in results])
    embed = sorted([r["response"].get("embed_cost", 0) for r in results])
    chunks = [r["response"].get("num_chunks", 0) for r in results]

    print(f"\n{'='*60}")
    print(f"Video Pipeline Breakdown")
    print(f"{'='*60}")
    print(f"{'Stage':<12} | {'Avg':>7} | {'P50':>7} | {'P99':>7}")
    print(f"{'-'*44}")
    for name, vals in [("Load", load), ("Decode", decode), ("Process", process), ("Embed", embed)]:
        avg = sum(vals) / n
        p50 = vals[int(n * 0.50)]
        p99 = vals[int(n * 0.99)]
        print(f"{name:<12} | {avg:>7.3f} | {p50:>7.3f} | {p99:>7.3f}")
    print(f"\nAvg chunks: {sum(chunks)/n:.1f}")
    print(f"{'='*60}")


def main():
    """Main entry point for benchmark."""
    global _transfer_ctx
    parser = argparse.ArgumentParser(description="Benchmark the encode / transfer endpoint")
    parser.add_argument("--encode-url", type=str, required=True,
                        help="Encoder base URL, e.g. http://host:port")
    parser.add_argument("--mode", choices=["encode", "transfer", "zmq", "video"], default="encode",
                        help=(
                            "encode: call /encode; transfer: /encode + /epd/transfer (RDMA); "
                            "zmq: /encode + ZMQ DEALER pull; video: call /encode_video_gpu_chunked"
                        ))
    parser.add_argument("--qps", type=float, default=None,
                        help="Queries per second to send (mutually exclusive with --concurrency)")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="Number of concurrent requests (mutually exclusive with --qps)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to a jsonl file containing multimodal requests (not needed for video mode)")
    parser.add_argument("--num-reqs", type=int, default=None,
                        help="Total number of requests to send. Images are cycled if needed.")
    parser.add_argument("--input-len", type=int, default=None,
                        help="Target token count for sampling images")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Model path for estimating token count from image dimensions")

    # Transfer mode specific args
    parser.add_argument("--hostname", type=str, default=None,
                        help="[transfer] Local hostname/IP for Mooncake engine")
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="[transfer] GPU device id")
    parser.add_argument("--ib-device", type=str, default=None,
                        help="[transfer] IB device name, e.g. mlx5_0")
    parser.add_argument("--embed-dim", type=int, default=7168,
                        help="[transfer] Embedding dimension (default: 7168 for Kimi K25)")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="[transfer] Embedding dtype")
    parser.add_argument("--pool-gb", type=float, default=2.0,
                        help="[transfer] Total CPU pinned buffer pool size in GB (default: 2.0)")
    parser.add_argument("--page-size-mb", type=int, default=2,
                        help="[transfer] Pool page size in MB (default: 2)")

    # ZMQ mode specific args
    parser.add_argument("--zmq-host", type=str, default=None,
                        help="[zmq] ZMQ ROUTER host IP")
    parser.add_argument("--zmq-port", type=int, default=None,
                        help="[zmq] ZMQ ROUTER port number")

    # Video mode specific args
    parser.add_argument("--video-url", type=str, default=None,
                        help="[video] Video URL to benchmark (repeats --num-reqs times)")
    parser.add_argument("--sample-fps", type=float, default=2.0,
                        help="[video] Sample FPS for video decode (default: 2.0)")
    args = parser.parse_args()

    if args.qps is not None and args.concurrency is not None:
        parser.error("--qps and --concurrency are mutually exclusive")
    if args.qps is None and args.concurrency is None:
        parser.error("one of --qps or --concurrency is required")

    if args.mode == "transfer" and args.hostname is None:
        parser.error("--hostname is required when --mode=transfer")

    if args.mode == "zmq" and (args.zmq_host is None or args.zmq_port is None):
        parser.error("--zmq-host and --zmq-port are required when --mode=zmq")

    if args.mode == "video" and args.video_url is None:
        parser.error("--video-url is required when --mode=video")

    if args.mode != "video" and args.dataset is None:
        parser.error("--dataset is required for non-video modes")

    # Load images / build URL list
    if args.mode == "video":
        import itertools
        global _video_sample_fps
        _video_sample_fps = args.sample_fps
        num_reqs = args.num_reqs or 8
        send_urls = [args.video_url] * num_reqs
    else:
        # Load images
        image_urls = load_all_images_from_jsonl(args.dataset)
        print(f"Loaded {len(image_urls)} images from {args.dataset}")

        # Sample by token length if requested
        if args.input_len is not None:
            if args.model_path is None:
                parser.error("--model-path is required when --input-len is specified")
            image_urls = sample_images_by_token_len(image_urls, args.input_len, args.model_path)
            print(f"Sampled {len(image_urls)} images with ~{args.input_len} tokens")

        if not image_urls:
            print("No images to send. Exiting.")
            return

        # Build the final request list, cycling images if num_reqs > len(image_urls)
        if args.num_reqs is not None:
            import itertools
            send_urls = list(itertools.islice(itertools.cycle(image_urls), args.num_reqs))
        else:
            send_urls = image_urls

    # Select request function based on mode
    global _zmq_ctx
    if args.mode == "transfer":
        _transfer_ctx = TransferContext(
            hostname=args.hostname,
            gpu_id=args.gpu_id,
            ib_device=args.ib_device or "",
            embed_dim=args.embed_dim,
            dtype=args.dtype,
            pool_bytes=int(args.pool_gb * (1024 ** 3)),
            page_size=args.page_size_mb * 1024 * 1024,
        )
        do_request = _do_one_transfer_request
        # endpoint is the base URL (without /encode or /epd/transfer)
        endpoint = args.encode_url.rstrip("/")
        # strip trailing /encode if user passed it
        if endpoint.endswith("/encode"):
            endpoint = endpoint[:-len("/encode")]
    elif args.mode == "zmq":
        _zmq_ctx = ZmqContext(host=args.zmq_host, port=args.zmq_port)
        do_request = _do_one_zmq_request
        endpoint = args.encode_url.rstrip("/")
        if endpoint.endswith("/encode"):
            endpoint = endpoint[:-len("/encode")]
    elif args.mode == "video":
        do_request = _do_one_video_request
        endpoint = args.encode_url.rstrip("/")
        if not endpoint.endswith("/encode_video_gpu_chunked"):
            endpoint = endpoint + "/encode_video_gpu_chunked"
    else:
        do_request = _do_one_encode_request
        endpoint = args.encode_url
        # ensure endpoint includes /encode
        if not endpoint.endswith("/encode"):
            endpoint = endpoint.rstrip("/") + "/encode"

    # Run benchmark
    if args.qps is not None:
        mode_desc = f"{args.qps} QPS"
        print(f"Sending {len(send_urls)} requests at {mode_desc} to {endpoint} (mode={args.mode})")
        bench_start = time.perf_counter()
        results, errors = send_at_qps(endpoint, send_urls, args.qps, do_request)
    else:
        mode_desc = f"concurrency={args.concurrency}"
        print(f"Sending {len(send_urls)} requests with {mode_desc} to {endpoint} (mode={args.mode})")
        bench_start = time.perf_counter()
        results, errors = send_with_concurrency(endpoint, send_urls, args.concurrency, do_request)
    elapsed = time.perf_counter() - bench_start

    # Cleanup
    if _transfer_ctx is not None:
        _transfer_ctx.release()
    if _zmq_ctx is not None:
        _zmq_ctx.close()

    # Save all results for later analysis
    all_results = results + errors
    output_path = "bench_encode_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {output_path}")

    print_stats(results, errors, elapsed)

    if args.mode == "zmq" and results:
        print_zmq_stats(results, elapsed)

    if args.mode == "video" and results:
        _print_video_stats(results, elapsed)


if __name__ == "__main__":
    main()




"""
# Mode 1: encode only (default, same as before)
python bench_encode.py \
    --encode-url http://127.0.0.1:30000/encode \
    --concurrency 1 \
    --dataset /home/users/wangchao101/work/baidu/hac-aiacc/aiak_sglang/image2text_local.jsonl \
    --num-reqs 20 \
    --model-path /home/users/wangchao101/Qwen3-VL-32B-Instruct \
    --input-len 3000

# Mode 2: encode + RDMA transfer
python bench_encode.py \
    --encode-url http://127.0.0.1:30000 \
    --mode transfer \
    --hostname 10.51.198.16 \
    --gpu-id 7 \
    --ib-device mlx5_13 \
    --embed-dim 7168 \
    --dtype bfloat16 \
    --pool-gb 2.0 \
    --page-size-mb 2 \
    --concurrency 1 \
    --dataset /home/users/wangchao101/work/baidu/hac-aiacc/aiak_sglang/image2text_local.jsonl \
    --num-reqs 20 \
    --model-path /home/users/wangchao101/Qwen3-VL-32B-Instruct \
    --input-len 3000

# Mode 3: encode + ZMQ transfer
python bench_encode.py \
    --encode-url http://127.0.0.1:30000 \
    --mode zmq \
    --zmq-host 127.0.0.1 \
    --zmq-port 30001 \
    --concurrency 4 \
    --dataset /home/users/wangchao101/work/baidu/hac-aiacc/aiak_sglang/image2text_local.jsonl \
    --num-reqs 20 \
    --model-path /home/users/wangchao101/Qwen3-VL-32B-Instruct \
    --input-len 3000


curl http://127.0.0.1:30000/encode -X POST -H "content-type:application/json" -d '{
    "req_id": "abc",
    "mm_items": ["https://cce-ai-models.bj.bcebos.com/qianfan/images_large/0011cfd8a316edd6a363b30c206fbb0f.JPEG"],
    "num_parts": 1,
    "part_idx": 0,
    "prefill_host": null,
    "embedding_port": null
}' | python3 -m json.tool --no-ensure-ascii

# Mode 4: video encode
python bench_encode.py \
    --encode-url http://127.0.0.1:8000 \
    --mode video \
    --video-url "https://bml-test-test.bj.bcebos.com/%20tp-upload/test_15s_1080p.mp4" \
    --sample-fps 1.0 \
    --concurrency 4 \
    --num-reqs 16

"""