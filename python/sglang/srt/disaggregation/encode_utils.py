""""Utils for batch execution."""

from __future__ import annotations

import time
import queue
import logging
import subprocess
import re
import threading
from concurrent.futures import Future, wait, FIRST_COMPLETED
from dataclasses import dataclass
from typing import Dict, Generic, TypeVar, Optional, List, Protocol, Iterable, Tuple, Callable

# Mapping from encoder_transfer_backend values to wire transfer_type values.
_TRANSFER_TYPE_MAP = {
    "zmq_to_scheduler": "zmq",
    "zmq_to_tokenizer": "zmq",
}


def normalize_transfer_type(backend: str) -> str:
    """Convert encoder_transfer_backend value to the wire transfer_type.

    ``zmq_to_scheduler`` and ``zmq_to_tokenizer`` are both mapped to ``"zmq"``;
    other values (e.g. ``"mooncake"``) are returned as-is.
    """
    return _TRANSFER_TYPE_MAP.get(backend, backend)




# ==================== GPU Processor Registry ====================


# NOTE: Use `from sglang.srt.disaggregation.gpu_processor_registry import get_gpu_processor_cls`
# directly instead of this wrapper.


class WithRidProto(Protocol):
    """Protocol for objects that carry a request ID."""
    rid: str

    def num_tokens(self) -> int:
        """Return num tokens in this req."""


class WithBathInfoProto(Protocol):
    """Protocol for objects that expose batch info (token count and batch size)."""

    def num_tokens(self) -> int:
        """Return the number of tokens in this result."""
        ...

    def batch_size(self) -> int:
        """Return the batch size of this result."""
        ...


ReqType = TypeVar("ReqType", bound=WithRidProto)
ResType = TypeVar("ResType", bound=WithBathInfoProto)


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class QueueEntry:
    """Queue entry."""
    req: ReqType
    future: Future
    enqueue_time: float


class BatchImplProto(Protocol[ReqType, ResType]):
    """非流式批处理 impl：一次接收一批请求，返回全部结果。"""
    name: str

    def __call__(self, reqs: List[ReqType]) -> Iterable[ResType]:
        """Execute a batch of requests and return all results."""
        ...


class StreamingImplProto(Protocol[ReqType, ResType]):
    """流式 impl：逐个提交请求，完成一个即可调度下一个。

    适用于基于线程池/进程池的 impl，executor 维护滑动窗口，
    不必等一整批全部完成才调度新任务。
    """
    name: str

    def submit(self, req: ReqType) -> Future[ResType]:
        """Submit a single request and return a Future for its result."""
        ...


class _BaseExecutor(Generic[ReqType, ResType]):
    """BatchExecutor 和 StreamExecutor 的公共基类。"""

    def __init__(self, name: str, 
                 backlog: int = 100000):
        """Initialize the base executor.

        Args:
            name: Name identifier for this executor (used in logging).
            collector: Prometheus metrics collector instance.
            backlog: Maximum queue capacity for pending requests.
        """
        self._name = name
        self.stopped = False
        self.requests: queue.Queue[Optional[QueueEntry]] = queue.Queue(backlog)
        self._pending_num_tokens = 0
        self._pending_lock = threading.Lock()

    def __call__(self, req: ReqType) -> Future[ResType]:
        """Submit a single request to the executor queue.

        Args:
            req: The request object to enqueue.

        Returns:
            A Future that will contain the result once processed.

        Raises:
            RuntimeError: If the executor has been stopped.
        """
        if self.stopped:
            raise RuntimeError(f"batch executor {self._name} has been stopped!")
        future: Future[ResType] = Future()
        self.requests.put(QueueEntry(req=req, future=future, enqueue_time=time.monotonic()), block=False)
        with self._pending_lock:
            self._pending_num_tokens += req.num_tokens()
        return future

    def stop(self):
        """Send a stop signal to the serve loop, allowing it to drain and exit."""
        self.requests.put(None)

    def serve(self):
        """Run the main serve loop. Must be implemented by subclasses."""
        raise NotImplementedError


class BatchExecutor(_BaseExecutor[ReqType, ResType]):
    """批处理执行器：收集一批请求，一次性交给 impl 处理。"""

    def __init__(self, impl: BatchImplProto[ReqType, ResType],
                 sched: Optional[Callable[[List[ReqType]], List[ReqType]]] = None,
                 backlog: int = 100000):
        """Initialize the batch executor.

        Args:
            impl: The batch processing implementation to delegate work to.
            sched: Optional scheduling function that selects a subset of
                   candidate requests to execute. Defaults to scheduling all.
            backlog: Maximum queue capacity for pending requests.
        """
        super().__init__(impl.name, backlog=backlog)
        self._impl = impl
        self._sched = sched or (lambda reqs: reqs)
        self._waiting: List[QueueEntry] = []

        self._last_step_end_tic: float = 0.0
        # 跟踪待处理（队列+运行中）的 token 总数，用于负载均衡
        self._pending_num_tokens: int = 0


    def __get_reqs(self) -> List[QueueEntry]:
        """
        从队列里获取所有可用请求。

        如果未获取到任何请求则会block，直到有请求到达。
        当 self.stopped 为 True 或 self._waiting 非空时永远不会block。
        """
        res: List[QueueEntry] = []

        while True:
            block = not (bool(res) or self.stopped or bool(self._waiting))
            try:
                entry = self.requests.get(block=block)

                if entry is None:
                    self.stopped = True
                    logger.warning(f"batch executor {self._name} recv a stop signal, "
                                   f"serve loop(thread) will exit soon!")
                    continue

                if entry.future.cancelled():
                    logger.info(f"request {entry.req.rid} cancelled, skip it!")
                    with self._pending_lock:
                        self._pending_num_tokens -= entry.req.num_tokens()
                    continue

                res.append(entry)
            except queue.Empty:
                break
        return res
    
    def __split(self, candidates: List[QueueEntry], sched: List[ReqType]) -> List[QueueEntry]:
        """将调度结果从候选队列里分离出来。"""
        sched_rids = {r.rid for r in sched}
        batch: List[QueueEntry] = []
        for e in candidates:
            if e.future.cancelled():
                # 请求被取消，减少 pending 计数
                with self._pending_lock:
                    self._pending_num_tokens -= e.req.num_tokens()
                continue
            if e.req.rid in sched_rids:
                batch.append(e)
            else:
                self._waiting.append(e)
        return batch


    def __serve_once(self):
        """Schedule one batch of requests, invoke the impl, and resolve futures."""
        new_entries = self.__get_reqs()
        # Merge waiting queue (priority) with newly fetched requests
        candidates = self._waiting + new_entries
        self._waiting = []

        if not candidates:
            return None

        scheduled_reqs = self._sched([e.req for e in candidates])

        batch = self.__split(candidates, scheduled_reqs)


        logger.debug(f"{self._name} sched {[e.req.rid for e in batch]}"
                     + (f", waiting {len(self._waiting)}" if self._waiting else ""))

        inputs: List[ReqType] = [e.req for e in batch]
        futures: List[Future[ResType]] = [e.future for e in batch]

        batch_size: int = 0
        num_tokens: int = 0

        start = time.monotonic()



        try:
            outs = list(self._impl(inputs))

            if len(outs) != len(futures):
                raise RuntimeError(
                    f"{self._name} impl returned {len(outs)} results for {len(futures)} inputs"
                )

            for out, fut in zip(outs, futures):
                batch_size += out.batch_size()
                num_tokens += out.num_tokens()
                if not fut.done():
                    fut.set_result(out)

        except Exception as e:
            logger.error(f"{self._name} failed to process requests: {inputs}, error: {e}")
            for fut in futures:
                if not fut.done():
                    fut.set_exception(e)
            logger.error(e, exc_info=True)
        finally:
            # 减少已完成请求的 token 计数
            completed_tokens = sum(r.num_tokens() for r in inputs)
            with self._pending_lock:
                self._pending_num_tokens -= completed_tokens
            self._last_step_end_tic = now = time.monotonic()
            cost = now - start
            if batch_size and num_tokens:
                logger.debug(
                    f"{self._name} batch: {batch_size}, tokens: {num_tokens}, "
                    f"cost: {cost:.3f}s, throughput: {num_tokens/cost:.1f} tokens/s"
                )

    def serve(self):
        """Run the batch serve loop until stopped and all queued requests are drained."""
        while True:
            try:
                self.__serve_once()
                if self.stopped and self.requests.empty() and not self._waiting:
                    logger.info(f"batch executor {self._name} stopped, exiting serve loop.")
                    break
            except Exception as exc:
                logger.fatal(f"{exc}", exc_info=True)

    @property
    def pending_num_tokens(self) -> int:
        """当前正在处理 + 等待调度的 token 总数（用于负载均衡）。"""
        return self._pending_num_tokens


class StreamExecutor(_BaseExecutor[ReqType, ResType]):
    """流式执行器：逐个提交请求，任一完成即可取新请求继续提交。

    impl 需要提供 submit(req) -> Future 方法，并发度由 impl 自身控制。
    """

    def __init__(self, impl: StreamingImplProto[ReqType, ResType],
                 backlog: int = 100000):
        """Initialize the stream executor.

        Args:
            impl: The streaming implementation that provides a submit() method.
            backlog: Maximum queue capacity for pending requests.
        """
        super().__init__(impl.name, backlog=backlog)
        self._impl = impl
        # impl_future -> (user_future, submit_time)
        self.in_flight: Dict[Future[ResType], Tuple[Future[ResType], float]] = {}

    def __get_reqs(self) -> List[QueueEntry]:
        """从队列取所有可用请求。"""
        res: List[QueueEntry] = []

        while True:
            block = not (bool(res) or bool(self.in_flight) or self.stopped)
            try:
                entry = self.requests.get(block=block)
            except queue.Empty:
                break

            if entry is None:
                self.stopped = True
                logger.warning(
                    f"stream executor {self._name} recv a stop signal, "
                    "serve loop(thread) will exit soon!"
                )
                continue

            if entry.future.cancelled():
                logger.info(f"request {entry.req.rid} cancelled, skip it!")
                with self._pending_lock:
                    self._pending_num_tokens -= entry.req.num_tokens()
                continue

            res.append(entry)
        return res

    def serve(self):
        """Run the serve loop until stopped and all in-flight tasks complete."""
        while True:
            entries = self.__get_reqs()

            for entry in entries:
                logger.debug(f"{self._name} submit {entry.req.rid}")
                impl_fut = self._impl.submit(entry.req)
                self.in_flight[impl_fut] = (entry.future, time.monotonic())

            if not self.in_flight:
                if self.stopped:
                    logger.info(f"stream executor {self._name} stopped, exiting serve loop.")
                    return
                continue

            done, _ = wait(self.in_flight.keys(), return_when=FIRST_COMPLETED)

            for impl_fut in done:
                user_fut, start_time = self.in_flight.pop(impl_fut)
                try:
                    result = impl_fut.result()
                    if not user_fut.done():
                        user_fut.set_result(result)
                except Exception as e:
                    logger.error(f"{self._name} task failed: {e}", exc_info=True)
                    if not user_fut.done():
                        user_fut.set_exception(e)


class TokenLimitSched(Generic[ReqType]):

    """调度尽可能的的请求, 最接近最大token限制"""

    def __init__(self, max_num_tokens: int):
        """TokenLimitSched"""
        self.max_num_tokens = max_num_tokens

    def __call__(self, reqs: List[ReqType]) -> List[ReqType]:
        """调度"""
        res: List[ReqType] = []
        cur_tokens = 0
        for req in reqs:
            num_tokens = req.num_tokens()
            if cur_tokens + num_tokens > self.max_num_tokens:
                if not res: # 防饥饿
                    res.append(req)
                continue
            res.append(req)
            cur_tokens += num_tokens
        return res



import torch
import numpy as np


def to_tensor(data):
    """Convert various data types (ndarray, list, etc.) to a torch.Tensor.

    Args:
        data: Input data — may be a torch.Tensor, np.ndarray, list of np.ndarrays,
              or list of numeric values.

    Returns:
        A torch.Tensor if conversion is possible, otherwise returns data as-is.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.tensor(data)
    elif isinstance(data, list) and isinstance(data[0], np.ndarray):
        return torch.tensor(np.array(data))
    elif isinstance(data, list) and isinstance(data[0], (int, float)):
        return torch.tensor(data)
    else:
        return data


_image_grid_attrs = ["image_grid_thw", "image_grid_hws", "grid_thws"]


def get_image_grid_dim(images_input):
    """Extract the image grid dimension from a multimodal input.

    Searches for known grid attributes (image_grid_thw, image_grid_hws, grid_thws)
    on the object or in the dict.

    Args:
        images_input: A multimodal data object or dict that contains grid info.

    Returns:
        The grid dimension value (typically a tensor or list of [t, h, w]).

    Raises:
        ValueError: If none of the known grid attributes are found.
    """
    for attr in _image_grid_attrs:
        if hasattr(images_input, attr):
            return getattr(images_input, attr)
        elif isinstance(images_input, dict) and attr in images_input:
            return images_input[attr]
    raise ValueError(
        f"Image grid dim ({_image_grid_attrs}) not found in {images_input}"
    )


def compute_num_tokens_per_item(model, mm_item, processor_cls=None) -> int:
    """Compute the number of LLM-visible tokens an mm_item produces after the vision encoder.

    Dispatches to ``processor_cls.compute_num_tokens_per_item(model, mm_item)`` when
    a processor class is provided and implements that method.  This is the preferred
    path — all model-specific token arithmetic should live in the processor.

    """
    if processor_cls is not None and hasattr(processor_cls, "compute_num_tokens_per_item"):
        return processor_cls.compute_num_tokens_per_item(model, mm_item)



def num_tokens_from_grid(
    grid_thw,
    *,
    spatial_merge_size: Optional[int] = None,
    merge_kernel_size: Optional[Tuple[int, int]] = None,
) -> int:
    """Compute the token count a single ``[t, h, w]`` grid row expands to.

    Qwen ``spatial_merge_size`` -> ``t * (h//m) * (w//m)``; KimiK25
    ``merge_kernel_size`` -> ``(h//mh) * (w//mw)`` (temporal pooled).
    """
    t, h, w = int(grid_thw[0]), int(grid_thw[1]), int(grid_thw[2])
    if spatial_merge_size is not None:
        m = spatial_merge_size
        return t * (h // m) * (w // m)
    if merge_kernel_size is not None:
        merge_h, merge_w = merge_kernel_size
        return (h // merge_h) * (w // merge_w)
    raise ValueError(
        "num_tokens_from_grid requires either spatial_merge_size or "
        "merge_kernel_size."
    )


def get_original_image_size(image) -> List[int]:
    """Return an image's original ``[width, height]`` before preprocessing.

    Accepts the shapes an encoder sees at load/decode time: a ``{"type":
    "image", "image": ...}`` media dict, a decoded CHW/HW tensor, or a PIL
    image.
    """
    if isinstance(image, dict):
        image = image.get("image")
    if hasattr(image, "shape"):
        shape = image.shape
        if len(shape) < 2:
            raise ValueError(f"Invalid image tensor shape: {tuple(shape)}")
        return [int(shape[-1]), int(shape[-2])]
    if hasattr(image, "size"):
        width, height = image.size
        return [int(width), int(height)]
    raise TypeError(f"Cannot determine original image size from {type(image)}")


def original_image_sizes_from_task_results(
    task_results: Optional[List[Dict]],
) -> Optional[List[List[int]]]:
    """Collect ``encoder_addr.original_image_sizes`` across task_results.

    One task_result is one media item, so concatenating in task order yields
    one ``[width, height]`` per image in prompt order.  Returns ``None`` when
    no task_result carries the field — only models that need the original
    resolution (Kimi-K3) populate it.
    """
    if not task_results:
        return None
    sizes: List[List[int]] = []
    for task_result in task_results:
        encoder_addr = task_result.get("encoder_addr") or {}
        part_sizes = encoder_addr.get("original_image_sizes")
        if part_sizes:
            sizes.extend([int(v) for v in size] for size in part_sizes)
    return sizes or None


def expand_media_placeholders_with_offsets(
    input_ids: List[int],
    token_counts: List[int],
    media_token_id: int,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Expand the k-th ``media_token_id`` into ``token_counts[k]`` copies,
    recording each expanded run's inclusive ``(start, end)`` offset.

    Shared by decode (KV-cache sizing; ignores offsets) and prefill
    ``build_input_ids`` (offsets drive precomputed-embedding slicing). Surplus
    placeholders beyond ``len(token_counts)`` are kept as a single token.

    Returns ``(expanded_ids, offsets)``.
    """
    expanded_ids: List[int] = []
    offsets: List[Tuple[int, int]] = []
    media_idx = 0
    for token in input_ids:
        if token != media_token_id:
            expanded_ids.append(token)
            continue
        if media_idx < len(token_counts):
            start = len(expanded_ids)
            expanded_ids.extend([media_token_id] * token_counts[media_idx])
            offsets.append((start, len(expanded_ids) - 1))
        else:
            expanded_ids.append(media_token_id)
        media_idx += 1

    if media_idx != len(token_counts):
        logger.warning(
            "MM expansion: %d placeholders but %d grid counts; "
            "surplus placeholders kept as single tokens.",
            media_idx,
            len(token_counts),
        )
    return expanded_ids, offsets


def update_raw_text(
    text: str,
    video_prompts: List[str],
    video_placeholder: str,
) -> str:
    """replace video prompt in text with video chunk prompts."""
    video_count = text.count(video_placeholder)
    if video_count == 0:
        return text
    assert video_count == len(video_prompts)
    text_parts = text.split(video_placeholder)
    assert len(text_parts) == len(video_prompts) + 1
    text = "".join(
        [text_parts[i] + video_prompts[i] for i in range(len(video_prompts))]
    )
    text += text_parts[-1]
    return text


def _strip_ansi(text: str) -> str:
    """移除 ANSI 颜色代码."""
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    return ansi_escape.sub('', text)


def _get_rdma_nic_map() -> dict:
    """
    从 nvidia-smi topo -m 输出中提取 NIC 索引到名称的映射

    Returns:
        {nic_index: nic_name}，如 {0: "mlx5_4", 1: "mlx5_7", ...}
    """
    result = {}
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "topo", "-m"], text=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}

    in_legend = False
    for line in output.split('\n'):
        line = _strip_ansi(line)
        if "NIC Legend:" in line:
            in_legend = True
            continue
        if in_legend and ":" in line:
            # NIC0: mlx5_4
            parts = line.split(":")
            if len(parts) == 2:
                nic_idx_str = parts[0].strip().replace("NIC", "")
                nic_name = parts[1].strip()
                if nic_idx_str.isdigit():
                    result[int(nic_idx_str)] = nic_name
        # 只有在已经解析了 NIC 行之后遇到空行才 break
        elif in_legend and line.strip() == "" and result:
            break

    return result


def get_closest_rdma_nic(gpu_id: Optional[int] = None) -> Optional[str]:
    """
    获取离指定GPU最近的RDMA网卡名

    优先级: PIX > NODE > SYS

    Args:
        gpu_id: GPU ID，如果为None则使用torch.cuda.current_device()

    Returns:
        网卡名，如 "mlx5_12"，如果找不到则返回None
    """
    try:
        if gpu_id is None:
            gpu_id = torch.cuda.current_device()
    except Exception:
        gpu_id = 0

    gpu_str = str(gpu_id)
    nic_map = _get_rdma_nic_map()

    if not nic_map:
        return None

    try:
        output = subprocess.check_output(
            ["nvidia-smi", "topo", "-m"], text=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    # 解析拓扑矩阵
    lines = output.split('\n')

    # 找到header行和GPU所在行
    header_line = None
    gpu_line_content = None
    gpu_pattern = re.compile(r'^GPU(\d+)\s+(.*)$')

    for line in lines:
        line = _strip_ansi(line)

        # Header行包含GPU0和NIC0两个标记
        if header_line is None and "GPU0" in line and "NIC0" in line:
            header_line = line
            continue

        # GPU行: 以GPU开头，后面是数字，然后是空格/制表符
        match = gpu_pattern.match(line)
        if match:
            line_gpu_id = int(match.group(1))
            if line_gpu_id == gpu_id:
                gpu_line_content = match.group(2)
                break

    if not header_line or not gpu_line_content:
        return None

    headers = header_line.split()
    gpu_parts = gpu_line_content.split()

    # 按优先级查找: PIX > PXB > NODE > SYS
    priorities = ["PIX", "PXB", "NODE", "SYS"]

    for priority in priorities:
        for nic_idx, nic_name in nic_map.items():
            nic_header = f"NIC{nic_idx}"
            if nic_header not in headers:
                continue

            nic_col = headers.index(nic_header)

            if nic_col < len(gpu_parts):
                relation = gpu_parts[nic_col]
                if relation == priority:
                    return nic_name

    # fallback: 返回第一个NIC
    return list(nic_map.values())[0]




if __name__ == "__main__":
    import asyncio
    import threading

    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s %(filename)s:%(lineno)d - %(levelname)s - %(message)s')


    def _wrap(impl):
        impl.name = "test"
        return impl

    @dataclass
    class MockReq:
        """Mock request."""
        rid: str
        value: int

    def _run_with_serve(executor, coro_fn):
        """启动 serve 线程，在 asyncio.run 中执行 coro_fn，最后 stop 并 join"""
        t = threading.Thread(target=executor.serve, daemon=True)
        t.start()
        asyncio.run(coro_fn())
        executor.stop()
        t.join(timeout=2)

    def test_basic():
        """基本批量处理: 提交多个请求，验证结果正确"""
        def impl(reqs):
            return [r.value * 2 for r in reqs]
        

        executor = BatchExecutor(_wrap(impl))

        async def run():
            f1 = executor(MockReq("r1", 1))
            f2 = executor(MockReq("r2", 2))
            f3 = executor(MockReq("r3", 3))
            r1, r2, r3 = await asyncio.gather(
                asyncio.wrap_future(f1), asyncio.wrap_future(f2), asyncio.wrap_future(f3)
            )
            assert r1 == 2, f"expected 2, got {r1}"
            assert r2 == 4, f"expected 4, got {r2}"
            assert r3 == 6, f"expected 6, got {r3}"

        _run_with_serve(executor, run)
        print("test_basic passed!")

    def test_stop_graceful():
        """stop 后 serve 线程应正常退出，不会死循环"""
        def impl(reqs):
            return [None] * len(reqs)

        executor = BatchExecutor(_wrap(impl))
        t = threading.Thread(target=executor.serve)
        t.start()

        executor.stop()
        t.join(timeout=2)
        assert not t.is_alive(), "serve thread should exit after stop"
        print("test_stop_graceful passed!")

    def test_stop_drains_remaining():
        """stop 前已入队的请求应被处理完再退出"""
        def impl(reqs):
            return [r.value + 10 for r in reqs]

        executor = BatchExecutor(_wrap(impl))

        # 入队请求和 stop 信号，然后启动 serve
        fut1 = executor(MockReq("r1", 1))
        fut2 = executor(MockReq("r2", 2))
        executor.stop()

        t = threading.Thread(target=executor.serve)
        t.start()
        t.join(timeout=2)

        assert not t.is_alive(), "serve thread should exit after draining"
        assert fut1.result() == 11, f"expected 11, got {fut1.result()}"
        assert fut2.result() == 12, f"expected 12, got {fut2.result()}"
        print("test_stop_drains_remaining passed!")

    def test_error_handling():
        """impl 抛异常时，所有 future 应收到该异常"""
        def impl(reqs):
            raise ValueError("test error")

        executor = BatchExecutor(_wrap(impl))

        async def run():
            f1 = executor(MockReq("r1", 1))
            try:
                await asyncio.wrap_future(f1)
                assert False, "should have raised"
            except ValueError as e:
                assert "test error" in str(e)

        _run_with_serve(executor, run)
        print("test_error_handling passed!")

    def test_cancelled_future():
        """已取消的 future 应被跳过，不传入 impl"""
        processed_rids = []

        def impl(reqs):
            processed_rids.extend([r.rid for r in reqs])
            return [r.value for r in reqs]

        executor = BatchExecutor(_wrap(impl))

        # 入队一个已取消的 future
        cancelled_fut = executor(MockReq("cancelled", 0))
        cancelled_fut.cancel()

        async def run():
            f = executor(MockReq("r1", 42))
            r = await asyncio.wrap_future(f)
            assert r == 42

        _run_with_serve(executor, run)
        assert "cancelled" not in processed_rids, "cancelled request should be skipped"
        print("test_cancelled_future passed!")

    def test_stopped_rejects_new_requests():
        """stopped 状态下提交请求应抛 RuntimeError"""
        def impl(reqs):
            return [None] * len(reqs)

        executor = BatchExecutor(_wrap(impl))
        executor.stopped = True
        try:
            executor(MockReq("r1", 1))
            assert False, "should have raised RuntimeError"
        except RuntimeError:
            pass
        print("test_stopped_rejects_new_requests passed!")

    def test_queue_full_raises():
        """队列满时提交请求应抛 queue.Full"""
        def impl(reqs):
            return [None] * len(reqs)

        executor = BatchExecutor(_wrap(impl), backlog=3)
        # 填满队列
        for i in range(executor.backlog):
            executor(MockReq(f"r{i}", i))

        try:
            executor(MockReq("overflow", 0))
            assert False, "should have raised queue.Full"
        except queue.Full:
            pass
        print("test_queue_full_raises passed!")

    def test_concurrent_submit():
        """多协程并发提交请求，验证所有结果正确且无丢失"""
        import time

        def impl(reqs):
            time.sleep(0.01)  # 模拟处理耗时
            return [r.value * 2 for r in reqs]

        num_tasks = 50
        executor = BatchExecutor(_wrap(impl))

        async def run():
            # 分批并发提交，避免队列满
            results = []
            batch = 8
            for start in range(0, num_tasks, batch):
                end = min(start + batch, num_tasks)
                futs = [executor(MockReq(f"r{i}", i)) for i in range(start, end)]
                results.extend(await asyncio.gather(*[asyncio.wrap_future(f) for f in futs]))
            for i, r in enumerate(results):
                assert r == i * 2, f"req r{i}: expected {i * 2}, got {r}"

        _run_with_serve(executor, run)
        print("test_concurrent_submit passed!")

    def test_concurrent_submit_with_cancel():
        """并发提交时部分请求被取消，其余请求仍正确处理"""
        def impl(reqs):
            return [r.value + 100 for r in reqs]

        num_tasks = 6
        executor = BatchExecutor(_wrap(impl))

        # 入队，偶数索引的 future 预先 cancel，确保确定性
        futures = []
        for i in range(num_tasks):
            fut = executor(MockReq(f"r{i}", i))
            if i % 2 == 0:
                fut.cancel()
            futures.append(fut)
        executor.stop()

        t = threading.Thread(target=executor.serve)
        t.start()
        t.join(timeout=5)

        for i, fut in enumerate(futures):
            if i % 2 == 0:
                assert fut.cancelled(), f"req r{i} should be cancelled"
            else:
                assert fut.result() == i + 100, f"req r{i}: expected {i + 100}, got {fut.result()}"

        print("test_concurrent_submit_with_cancel passed!")

    def test_concurrent_multithread_submit():
        """多个线程同时通过 __call__ 提交请求，验证线程安全"""
        import time

        def impl(reqs):
            time.sleep(0.005)
            return [r.value * 3 for r in reqs]

        num_threads = 8
        reqs_per_thread = 10
        executor = BatchExecutor(_wrap(impl))
        t = threading.Thread(target=executor.serve, daemon=True)
        t.start()

        all_futures: list = [None] * (num_threads * reqs_per_thread)
        errors: list = []
        barrier = threading.Barrier(num_threads)

        def submitter(tid):
            try:
                barrier.wait(timeout=5)  # 所有线程同时开始提交
                for j in range(reqs_per_thread):
                    idx = tid * reqs_per_thread + j
                    all_futures[idx] = executor(MockReq(f"r{idx}", idx))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=submitter, args=(i,)) for i in range(num_threads)]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=5)

        assert not errors, f"submitter errors: {errors}"

        # 等待所有 future 完成
        for i, fut in enumerate(all_futures):
            assert fut is not None, f"future {i} not set"
            result = fut.result(timeout=5)
            assert result == i * 3, f"req r{i}: expected {i * 3}, got {result}"

        executor.stop()
        t.join(timeout=2)
        print("test_concurrent_multithread_submit passed!")

    def test_impl_returns_fewer_results():
        """impl 返回结果少于输入时，所有 future 应收到异常而非永久挂起"""
        def impl(reqs):
            return [r.value for r in reqs[:1]]  # 只返回第一个

        executor = BatchExecutor(_wrap(impl))

        async def run():
            f1 = executor(MockReq("r1", 1))
            f2 = executor(MockReq("r2", 2))
            for f in [f1, f2]:
                try:
                    await asyncio.wrap_future(f)
                except RuntimeError as e:
                    assert "returned 1 results for 2 inputs" in str(e)

        _run_with_serve(executor, run)
        print("test_impl_returns_fewer_results passed!")

    def test_impl_returns_more_results():
        """impl 返回结果多于输入时，所有 future 应收到异常"""
        def impl(reqs):
            return [r.value for r in reqs] + [999]

        executor = BatchExecutor(_wrap(impl))

        async def run():
            f1 = executor(MockReq("r1", 1))
            try:
                await asyncio.wrap_future(f1)
            except RuntimeError as e:
                assert "returned 2 results for 1 inputs" in str(e)

        _run_with_serve(executor, run)
        print("test_impl_returns_more_results passed!")

    def test_impl_generator():
        """impl 返回 generator (Iterable) 而非 list，应正常工作"""
        def impl(reqs):
            for r in reqs:
                yield r.value * 10

        executor = BatchExecutor(_wrap(impl))

        async def run():
            f1 = executor(MockReq("r1", 3))
            f2 = executor(MockReq("r2", 7))
            r1, r2 = await asyncio.gather(asyncio.wrap_future(f1), asyncio.wrap_future(f2))
            assert r1 == 30, f"expected 30, got {r1}"
            assert r2 == 70, f"expected 70, got {r2}"

        _run_with_serve(executor, run)
        print("test_impl_generator passed!")

    def test_single_batch_size():
        """逐个处理请求"""
        call_counts = []

        def impl(reqs):
            call_counts.append(len(reqs))
            return [r.value + 1 for r in reqs]

        executor = BatchExecutor(_wrap(impl))

        async def run():
            r = await asyncio.wrap_future(executor(MockReq("r1", 10)))
            assert r == 11, f"expected 11, got {r}"

        _run_with_serve(executor, run)
        assert all(c == 1 for c in call_counts), f"each batch should have size 1, got {call_counts}"
        print("test_single_batch_size passed!")

    def test_stop_during_processing():
        """impl 执行期间调用 stop，当前批次应正常完成"""
        import time
        processing = threading.Event()

        def impl(reqs):
            processing.set()
            time.sleep(0.1)  # 模拟耗时操作
            return [r.value * 5 for r in reqs]

        executor = BatchExecutor(_wrap(impl))
        fut = executor(MockReq("r1", 7))

        t = threading.Thread(target=executor.serve, daemon=True)
        t.start()

        processing.wait(timeout=2)  # 等 impl 开始执行
        executor.stop()             # impl 执行中发 stop
        t.join(timeout=2)

        assert not t.is_alive(), "serve should exit after current batch"
        assert fut.result() == 35, f"expected 35, got {fut.result()}"
        print("test_stop_during_processing passed!")

    def test_future_cancelled_after_dequeue():
        """future 在 dequeue 后、set_result 前被取消，不应影响同批次其他 future"""
        import time
        cancel_event = threading.Event()

        def impl(reqs):
            cancel_event.set()   # 通知主线程现在可以 cancel
            time.sleep(0.05)     # 给主线程时间去 cancel
            return [r.value for r in reqs]

        executor = BatchExecutor(_wrap(impl))

        # 入队两个请求
        fut1 = executor(MockReq("r1", 10))
        fut2 = executor(MockReq("r2", 20))
        executor.stop()

        t = threading.Thread(target=executor.serve, daemon=True)
        t.start()

        cancel_event.wait(timeout=2)
        fut1.cancel()  # 在 impl 执行期间 cancel

        t.join(timeout=2)
        assert fut1.cancelled()
        assert fut2.result() == 20, f"expected 20, got {fut2.result()}"
        print("test_future_cancelled_after_dequeue passed!")

    def test_multiple_stop_calls():
        """多次调用 stop 不应导致异常"""
        def impl(reqs):
            return [None] * len(reqs)

        executor = BatchExecutor(_wrap(impl))
        t = threading.Thread(target=executor.serve)
        t.start()

        executor.stop()
        executor.stop()  # 第二次 stop
        t.join(timeout=2)
        assert not t.is_alive()
        print("test_multiple_stop_calls passed!")

    # ── streaming tests (submit-based sliding window) ──

    def test_stream_basic():
        """streaming 模式基本测试：通过 submit 逐个提交，滑动窗口调度"""
        from concurrent.futures import ThreadPoolExecutor

        class StreamImpl:
            """Stream implementation for testing."""
            name = "test_stream"
            def __init__(self):
                self._pool = ThreadPoolExecutor(4)
            def submit(self, req):
                """Submit task to pool."""
                return self._pool.submit(lambda r: r.value * 2, req)

        executor = StreamExecutor(StreamImpl())

        async def run():
            f1 = executor(MockReq("r1", 1))
            f2 = executor(MockReq("r2", 2))
            f3 = executor(MockReq("r3", 3))
            r1, r2, r3 = await asyncio.gather(
                asyncio.wrap_future(f1), asyncio.wrap_future(f2), asyncio.wrap_future(f3)
            )
            assert r1 == 2 and r2 == 4 and r3 == 6

        _run_with_serve(executor, run)
        print("test_stream_basic passed!")

    def test_stream_sliding_window():
        """streaming 模式：任务完成后立即从队列取新请求，不等整批完成。

        提交 4 个请求，每完成一个就立即填入下一个。
        """
        import time
        from concurrent.futures import ThreadPoolExecutor

        submit_order = []

        class StreamImpl:
            """Stream implementation for testing."""
            name = "test_stream_sw"
            def __init__(self):
                self._pool = ThreadPoolExecutor(2)
            def submit(self, req):
                """Submit task to pool."""
                submit_order.append(req.rid)
                def work(r):
                    # r0 快(0.02s), r1 慢(0.15s), r2 快, r3 快
                    time.sleep(0.02 if r.value != 1 else 0.15)
                    return r.value * 10
                return self._pool.submit(work, req)

        executor = StreamExecutor(StreamImpl())

        # 用 _run_with_serve 异步提交，避免 queue 满
        async def run():
            futs = []
            for i in range(4):
                futs.append(executor(MockReq(f"r{i}", i)))
                await asyncio.sleep(0.01)  # 让 serve 有机会消费队列
            results = await asyncio.gather(*[asyncio.wrap_future(f) for f in futs])
            assert list(results) == [0, 10, 20, 30], f"got {list(results)}"

        _run_with_serve(executor, run)
        # 前两个立即 submit，第三个应在 r0 完成后立即 submit（不用等 r1）
        assert len(submit_order) == 4
        assert submit_order[:2] == ["r0", "r1"]  # 初始填满窗口
        print("test_stream_sliding_window passed!")

    def test_stream_error_handling():
        """streaming 模式：单个任务失败不影响其他任务"""
        from concurrent.futures import ThreadPoolExecutor

        class StreamImpl:
            """Stream implementation for testing."""
            name = "test_stream_err"
            def __init__(self):
                self._pool = ThreadPoolExecutor(4)
            def submit(self, req):
                """Submit task to pool."""
                def work(r):
                    if r.value == 2:
                        raise ValueError("task 2 failed")
                    return r.value * 10
                return self._pool.submit(work, req)

        executor = StreamExecutor(StreamImpl())

        fut1 = executor(MockReq("r1", 1))
        fut2 = executor(MockReq("r2", 2))
        fut3 = executor(MockReq("r3", 3))
        executor.stop()

        t = threading.Thread(target=executor.serve)
        t.start()
        t.join(timeout=2)

        assert fut1.result() == 10
        try:
            fut2.result()
            assert False, "should have raised"
        except ValueError as e:
            assert "task 2 failed" in str(e)
        assert fut3.result() == 30, "other tasks should not be affected"
        print("test_stream_error_handling passed!")

    def test_stream_stop_drains():
        """streaming 模式：stop 前已入队的请求应被处理完"""
        from concurrent.futures import ThreadPoolExecutor

        class StreamImpl:
            """Stream implementation for testing."""
            name = "test_stream_drain"
            def __init__(self):
                self._pool = ThreadPoolExecutor(2)
            def submit(self, req):
                """Submit task to pool."""
                return self._pool.submit(lambda r: r.value + 100, req)

        executor = StreamExecutor(StreamImpl())

        fut1 = executor(MockReq("r1", 1))
        fut2 = executor(MockReq("r2", 2))
        executor.stop()

        t = threading.Thread(target=executor.serve)
        t.start()
        t.join(timeout=2)

        assert not t.is_alive()
        assert fut1.result() == 101
        assert fut2.result() == 102
        print("test_stream_stop_drains passed!")

    def test_stream_cancelled_skipped():
        """streaming 模式：已取消的 future 不会被 submit"""
        from concurrent.futures import ThreadPoolExecutor
        submitted_rids = []

        class StreamImpl:
            """Stream implementation for testing."""
            name = "test_stream_cancel"
            def __init__(self):
                self._pool = ThreadPoolExecutor(4)
            def submit(self, req):
                """Submit task to pool."""
                submitted_rids.append(req.rid)
                return self._pool.submit(lambda r: r.value, req)

        executor = StreamExecutor(StreamImpl())

        cancelled_fut = executor(MockReq("cancelled", 0))
        cancelled_fut.cancel()

        async def run():
            f = executor(MockReq("r1", 42))
            r = await asyncio.wrap_future(f)
            assert r == 42

        _run_with_serve(executor, run)
        assert "cancelled" not in submitted_rids
        print("test_stream_cancelled_skipped passed!")

    test_basic()
    test_stop_graceful()
    test_stop_drains_remaining()
    test_error_handling()
    test_cancelled_future()
    test_stopped_rejects_new_requests()
    test_queue_full_raises()
    test_concurrent_submit()
    test_concurrent_submit_with_cancel()
    test_concurrent_multithread_submit()
    test_impl_returns_fewer_results()
    test_impl_returns_more_results()
    test_impl_generator()
    test_single_batch_size()
    test_stop_during_processing()
    test_future_cancelled_after_dequeue()
    test_multiple_stop_calls()
    test_stream_basic()
    test_stream_sliding_window()
    test_stream_error_handling()
    test_stream_stop_drains()
    test_stream_cancelled_skipped()
    print("\nAll tests passed!")
