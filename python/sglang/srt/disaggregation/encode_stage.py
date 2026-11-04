"""Multimodal stage implementation."""

import atexit
import logging
import multiprocessing as mp
import os
import queue
import threading
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, replace
from multiprocessing import shared_memory
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from transformers import AutoImageProcessor

from sglang.srt.disaggregation.encode_utils import (
    compute_num_tokens_per_item,
    get_image_grid_dim,
    to_tensor,
)
from sglang.srt.disaggregation.gpu_processor_registry import (
    get_gpu_processor_cls,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.mem_cache.scatter_mm_cache import (
    Nbytes,
    Ptr,
    ScatteredTensor,
)
from sglang.srt.server_args import (
    ServerArgs,
)
from sglang.srt.utils import (
    load_audio,
    load_image,
    load_video,
)

logger = logging.getLogger(__name__)


from io import BytesIO
from typing import Union
from urllib.parse import unquote, urlparse

import pybase64
import requests
from PIL import Image

from sglang.srt.utils import ImageData


def load_image(
    image_file: Union[Image.Image, str, ImageData, bytes],
) -> tuple[Image.Image, tuple[int, int]]:
    """overwrite `sglang.srt.utils.load_image` to fallback to 0.5.8"""
    if isinstance(image_file, ImageData):
        image_file = image_file.url

    image = image_size = None
    if isinstance(image_file, Image.Image):
        image = image_file
        image_size = (image.width, image.height)
    elif isinstance(image_file, bytes):
        image = Image.open(BytesIO(image_file))
    elif image_file.startswith("http://") or image_file.startswith("https://"):
        timeout = int(os.getenv("REQUEST_TIMEOUT", "3"))
        response = requests.get(image_file, stream=True, timeout=timeout)
        try:
            response.raise_for_status()
            image = Image.open(response.raw)
            image.load()  # Force loading to avoid issues after closing the stream
        finally:
            response.close()
    elif image_file.startswith("file://"):
        image_file = unquote(urlparse(image_file).path)
        image = Image.open(image_file)
    elif image_file.lower().endswith(("png", "jpg", "jpeg", "webp", "gif")):
        image = Image.open(image_file)
    elif image_file.startswith("data:"):
        image_file = image_file.split(",")[1]
        image = Image.open(BytesIO(pybase64.b64decode(image_file, validate=True)))
    elif isinstance(image_file, str):
        image = Image.open(BytesIO(pybase64.b64decode(image_file, validate=True)))
    else:
        raise ValueError(f"Invalid image: {image_file}")

    return image, image_size


@dataclass
class LoadReq:
    """Load request for multimodal data."""

    rid: str
    mm_items: Any
    modalities: Optional[List[str]] = None  # per-item: "image" or "video"

    def num_tokens(self) -> int:
        """Return number of tokens (placeholder)."""
        return 0


@dataclass
class LoadResult:
    """Result of loading multimodal data."""

    rid: str
    time_cost: float
    images: Any  # loaded PIL Images (single / list / nested list)
    modalities: Optional[List[str]] = None


@dataclass
class DecodeReq:
    """Decode request: raw bytes to GPU tensors."""

    rid: str
    images: List[bytes]
    modalities: Optional[List[str]] = None

    def num_tokens(self) -> int:
        """Return number of tokens (placeholder)."""
        return 0


@dataclass
class DecodeResult:
    """Decode result: decoded images (GPU tensors or PIL Images)."""

    rid: str
    time_cost: float
    images: List[Any]  # GPU mode: [C,H,W] uint8 tensors; CPU mode: PIL Images


@dataclass
class VideoLoadReq:
    """Video load request."""

    rid: str
    video_items: List[str]  # local paths or http(s) URLs
    sample_fps: Optional[float] = None
    # Per-attempt override; never changes the backend of other video tasks.
    force_decord: bool = False

    def num_tokens(self) -> int:
        """Return the number of tokens for this video load request."""
        return 0


@dataclass
class VideoChunkLoadReq:
    """Video chunk load request for streaming decode."""

    rid: str
    video_item: str
    chunk_idx: int
    sample_fps: Optional[float] = None

    def num_tokens(self) -> int:
        """Return the number of tokens for this video chunk load request."""
        return 0


@dataclass
class VideoDecodeResult:
    """Video decode-only result: raw GPU frame tensors per temporal chunk."""

    rid: str
    time_cost: float
    download_cost: float
    frame_extraction_cost: float
    chunks: List[Any] = None  # List[torch.Tensor], each (T, C, H, W)
    video_height: int = 0
    video_width: int = 0
    chunk_timestamps: List[str] = None  # per-chunk start timestamps


class MMLoaderImpl:
    """Multimodal loader implementation using a thread pool to load images in parallel."""

    def __init__(
        self, server_args: ServerArgs, max_batch_size: int, return_bytes: bool = False
    ):
        """Initialize the multimodal loader.

        Args:
            server_args: Server configuration arguments.
            max_batch_size: Maximum number of concurrent load tasks (thread pool size).
            return_bytes: If True, return raw image bytes instead of PIL Images.
        """
        self.name = "mm_loader"
        self.max_batch_size = max_batch_size
        self.server_args = server_args
        self._return_bytes = return_bytes
        labels = {
            "model_name": self.server_args.served_model_name,
        }
        if server_args.extra_metric_labels:
            labels.update(server_args.extra_metric_labels)


        from concurrent.futures import ThreadPoolExecutor

        self._pool = ThreadPoolExecutor(self.max_batch_size)

    def submit(self, req: LoadReq):
        """Submit a load request to the thread pool.

        Args:
            req: The load request containing multimodal items to load.

        Returns:
            A Future that will contain the LoadResult.
        """
        return self._pool.submit(self.load, req)

    def load(self, req: LoadReq) -> LoadResult:
        """Load all multimodal items in a single request sequentially.

        Args:
            req: The load request containing a list of mm_items.

        Returns:
            A LoadResult with loaded images and timing info.
        """
        start_time = time.monotonic()
        images = []
        for i, item in enumerate(req.mm_items):
            modality_str = (
                req.modalities[i]
                if req.modalities and i < len(req.modalities)
                else "image"
            )
            if modality_str == "video":
                image = self._download_video_to_file(item)
            elif self._return_bytes:
                image = self._download_bytes(item)
            else:
                modality = Modality.VIDEO if modality_str == "video" else Modality.IMAGE
                image = self._load_single_item(item, modality=modality)
            images.append(image)
        load_cost = time.monotonic() - start_time
        return LoadResult(
            rid=req.rid, images=images, time_cost=load_cost, modalities=req.modalities
        )

    def _download_bytes(self, url: str) -> bytes:
        """Download image from URL and return raw bytes.

        Reads the body via streaming, verifies completeness (advertised
        Content-Length must match the bytes received, and the body must be
        non-empty), and retries a few times on network errors or short/truncated
        reads. This turns a transient partial download into a retry+recovery
        instead of surfacing later as a bogus "image file is truncated" decode
        failure (HTTP 422)."""
        if url.startswith("data:"):
            import base64

            header, data = url.split(",", 1)
            if ";base64" in header:
                return base64.b64decode(data)
            return data.encode("utf-8")

        import requests as http_requests

        attempts = max(1, int(os.environ.get("SGLANG_IMAGE_DOWNLOAD_RETRIES", "3")))
        timeout = int(os.environ.get("SGLANG_IMAGE_DOWNLOAD_TIMEOUT_S", "30"))
        last_err = None
        for attempt in range(1, attempts + 1):
            try:
                resp = http_requests.get(
                    url,
                    timeout=timeout,
                    stream=True,
                    headers={"Cache-Control": "no-cache"},
                )
                try:
                    resp.raise_for_status()
                    cl = resp.headers.get("Content-Length")
                    expected = int(cl) if (cl and cl.isdigit()) else None
                    buf = bytearray()
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            buf.extend(chunk)
                finally:
                    resp.close()
                if not buf:
                    raise IOError("empty download (0 bytes)")
                if expected is not None and len(buf) != expected:
                    raise IOError(
                        f"incomplete download: got {len(buf)} of {expected} bytes"
                    )
                return bytes(buf)
            except Exception as e:  # network error / short read -> retry
                last_err = e
                logger.warning(
                    "[ImageDownload] attempt %d/%d failed url=%s err=%s",
                    attempt,
                    attempts,
                    url,
                    e,
                )
        raise IOError(f"image download failed after {attempts} attempts: {last_err}")

    def _download_video_to_file(self, url: str) -> str:
        """Download video from URL to a temp file, return the local path."""
        import tempfile

        import requests as http_requests

        resp = http_requests.get(
            url, timeout=60, stream=True, headers={"Cache-Control": "no-cache"}
        )
        resp.raise_for_status()
        # Keep the URL's real extension; do NOT fake ".mp4" for extensionless
        # URLs. An empty suffix lets _classify_video route purely by detected
        # content (e.g. content=AVI -> decord) with no false extension mismatch.
        suffix = os.path.splitext(urlparse(url).path)[1]
        fd, local_path = tempfile.mkstemp(suffix=suffix, prefix="vid_dl_")
        try:
            with os.fdopen(fd, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
        except Exception:
            os.unlink(local_path)
            raise
        return local_path

    def _load_single_item(
        self,
        data,
        modality: Modality,
        frame_count_limit=None,
        audio_sample_rate: Optional[int] = None,
        discard_alpha_channel=True,
    ):
        """Load a single multimodal item and return the decoded object.

        If data is a precomputed dict, returns it directly."""
        if isinstance(data, dict):
            return data
        try:
            if modality == Modality.IMAGE:
                img, _ = load_image(data)
                if discard_alpha_channel and img.mode != "RGB":
                    img = img.convert("RGB")
                return img
            elif modality == Modality.VIDEO:
                return load_video(data, frame_count_limit)
            elif modality == Modality.AUDIO:
                return load_audio(data, audio_sample_rate)

        except Exception as e:
            raise RuntimeError(f"Error while loading data {data}: {e}") from e


@dataclass
class ProcessReq:
    """Embed request."""

    rid: str
    images: Any
    modality: Modality = None

    def __post_init__(self):
        if self.modality is None:
            self.modality = Modality.IMAGE

    def num_tokens(self) -> int:
        """Return number of tokens (placeholder)."""
        return 0


@dataclass
class ProcessResult:
    """Embed result."""

    rid: str
    time_cost: float
    mm_item: "MultimodalDataItem"


def _convert_images_to_media_dict(images):
    """Convert image inputs to media-dict format expected by some processors."""

    # Kimi-K2.5 vision processor expects {"type": "image", "image": PIL.Image}.
    def wrap(item):
        if isinstance(item, dict):
            if item.get("type") == "image" and "image" in item:
                return item
            if "image" in item:
                return {"type": "image", **item}
        return {"type": "image", "image": item}

    if isinstance(images, (list, tuple)):
        if len(images) > 0 and isinstance(images[0], (list, tuple)):
            return [[wrap(item) for item in group] for group in images]
        return [wrap(item) for item in images]
    return [wrap(images)]


__mm_processor: Optional[AutoImageProcessor] = None


def init___mm_processor(server_args: ServerArgs):
    """Initialize the global multimodal processor in a worker process.

    Args:
        server_args: Server configuration used to load the image processor.
    """
    global __mm_processor
    __mm_processor = AutoImageProcessor.from_pretrained(
        server_args.model_path,
        trust_remote_code=server_args.trust_remote_code,
        use_fast=True,
    )


def _preprocess_images(req: ProcessReq):
    """Run the image processor and separate tensor / non-tensor outputs.

    This is the common preprocessing logic shared by both the pickle-IPC
    and shared-memory-IPC worker paths.

    Returns:
        (tensors, non_tensor_data) where tensors is a dict mapping key->Tensor
        (always contains "feature") and non_tensor_data holds everything else.
    """
    assert __mm_processor, "worker is not initialized"

    images_input = __mm_processor(images=_convert_images_to_media_dict(req.images))

    # Normalize Kimi-K2.5 HF processor's "grid_thws" to SGL standard "image_grid_thw",
    # matching KimiGPUProcessorWrapper._cpu_call (multimodal/processors/kimi_k25.py).
    if "grid_thws" in images_input and "image_grid_thw" not in images_input:
        images_input["image_grid_thw"] = images_input.pop("grid_thws")

    tensors: Dict[str, torch.Tensor] = {
        "feature": to_tensor(images_input["pixel_values"])
    }
    non_tensor_data: Dict[str, Any] = {}
    for k, v in images_input.items():
        if k == "pixel_values":
            continue
        converted = to_tensor(v)
        if isinstance(converted, torch.Tensor):
            tensors[k] = converted
        else:
            non_tensor_data[k] = converted
    return tensors, non_tensor_data


def mm_preprocess(req: ProcessReq) -> ProcessResult:
    """Worker function: preprocess images and return a ProcessResult (pickle IPC)."""
    start_time = time.monotonic()
    tensors, non_tensor_data = _preprocess_images(req)

    mm_item = MultimodalDataItem.from_dict(
        {
            "modality": Modality.IMAGE,
            "feature": tensors.pop("feature"),
        }
    )
    for k, v in tensors.items():
        mm_item.set(k, v)
    for k, v in non_tensor_data.items():
        mm_item.set(k, v)

    return ProcessResult(req.rid, time.monotonic() - start_time, mm_item)


# ── Shared-memory IPC types and helpers ──────────────────────────────────────


@dataclass
class ShmTensorMeta:
    """Metadata describing one tensor stored in a shared memory buffer."""

    key: str  # e.g. "feature", "image_grid_thw"
    dtype: str  # numpy dtype string, e.g. "float32"
    shape: tuple
    offset: int  # byte offset in the shm buffer
    nbytes: int


@dataclass
class ShmProcessResult:
    """Lightweight result returned by shm worker — tensor data lives in shm."""

    rid: str
    time_cost: float
    shm_name: str
    tensor_metas: List[ShmTensorMeta]
    modality: Modality
    non_tensor_data: Dict[str, Any]


def _write_tensors_to_shm(
    shm_name: str, tensors: Dict[str, torch.Tensor]
) -> List[ShmTensorMeta]:
    """Worker-side: write tensors into a pre-allocated shared memory buffer.

    Returns a list of lightweight metadata (no tensor payload).
    """
    shm = shared_memory.SharedMemory(name=shm_name, create=False)
    metas: List[ShmTensorMeta] = []
    offset = 0
    try:
        for key, tensor in tensors.items():
            t = tensor.contiguous()
            np_arr = t.numpy()
            nbytes = np_arr.nbytes
            # Use numpy array backed by shm buffer to avoid memoryview format mismatch
            dst = np.ndarray(nbytes, dtype=np.uint8, buffer=shm.buf, offset=offset)
            dst[:] = np.frombuffer(np_arr.tobytes(), dtype=np.uint8)
            metas.append(
                ShmTensorMeta(
                    key=key,
                    dtype=str(np_arr.dtype),
                    shape=tuple(t.shape),
                    offset=offset,
                    nbytes=nbytes,
                )
            )
            offset += nbytes
    finally:
        shm.close()  # close handle in worker; main process still owns it
    return metas


def mm_preprocess_with_shm(req: ProcessReq, shm_name: str) -> ShmProcessResult:
    """Worker function: preprocess images and write tensors to shm."""
    start_time = time.monotonic()
    tensors, non_tensor_data = _preprocess_images(req)
    tensor_metas = _write_tensors_to_shm(shm_name, tensors)

    return ShmProcessResult(
        rid=req.rid,
        time_cost=time.monotonic() - start_time,
        shm_name=shm_name,
        tensor_metas=tensor_metas,
        modality=Modality.IMAGE,
        non_tensor_data=non_tensor_data,
    )


# ---------------------------------------------------------------------------
# GPU decode implementation (bytes -> GPU tensor)
# ---------------------------------------------------------------------------


class MMDecodeImpl:
    """Image decode: bytes -> decoded images.

    GPU mode: uses nvJPEG/torchvision for GPU decode, outputs [C,H,W] uint8 tensors.
    CPU mode: uses PIL decode, outputs PIL Images.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        device: torch.device,
        use_gpu: bool = True,
        keep_alpha_channel: bool = False,
    ):
        """Initialize the decode stage.

        Args:
            server_args: Server configuration arguments.
            device: Target device (CUDA for GPU mode, ignored for CPU mode).
            use_gpu: If True, use GPU decode (nvJPEG/torchvision). If False, use PIL.
            keep_alpha_channel: If True, transparent images stay RGBA so the
                preprocessor can composite them onto its configured background
                (Kimi-K3); otherwise alpha is dropped at decode time.
        """
        self.name = "mm_decode"
        self.device = device
        self.server_args = server_args
        self._use_gpu = use_gpu
        self._keep_alpha_channel = keep_alpha_channel

        labels = (
            {
                "model_name": server_args.served_model_name,
            }
            if server_args.served_model_name
            else {}
        )

        if use_gpu:
            torch.cuda.set_device(device)
            self._decode_stream = torch.cuda.Stream(device=device)
            self._gpu_decode_available = self._probe("JPEG", self._decode_jpeg)
            self._png_decode_available = self._probe("PNG", self._decode_png)
            self._webp_decode_available = self._probe("WEBP", self._decode_webp)
        else:
            self._decode_stream = None
            self._gpu_decode_available = False
            self._png_decode_available = False
            self._webp_decode_available = False

        from concurrent.futures import ThreadPoolExecutor

        self._pool = ThreadPoolExecutor(max_workers=1)

        mode = "GPU" if use_gpu else "CPU"
        logger.info(
            f"MMDecodeImpl: mode={mode}, device={device}, "
            f"jpeg_decode={'nvJPEG' if self._gpu_decode_available else 'PIL'}, "
            f"png_decode={'torchvision' if self._png_decode_available else 'PIL'}, "
            f"webp_decode={'torchvision' if self._webp_decode_available else 'PIL'}"
        )

    def _probe(self, fmt: str, decode_fn) -> bool:
        """Probe whether a decode function works for the given format."""
        try:
            import io

            from PIL import Image as _PILImage

            buf = io.BytesIO()
            _PILImage.new("RGB", (8, 8), "red").save(buf, format=fmt)
            result = decode_fn(buf.getvalue())
            return result is not None and result.shape[0] == 3
        except Exception as e:
            logger.warning(
                f"MMDecodeImpl: GPU decode probe failed for {fmt}, will use PIL fallback: {e}"
            )
            return False

    def _decode_jpeg(self, raw: bytes) -> torch.Tensor:
        """Decode JPEG bytes using nvJPEG."""
        from torchvision.io import decode_jpeg

        data = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        return decode_jpeg(data, device=str(self.device))

    def _decode_png(self, raw: bytes) -> torch.Tensor:
        """Decode PNG bytes using torchvision."""
        from torchvision.io import ImageReadMode, decode_png

        data = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        return decode_png(data, mode=ImageReadMode.RGB).to(
            self.device, non_blocking=True
        )

    def _decode_webp(self, raw: bytes) -> torch.Tensor:
        """Decode WebP bytes using torchvision."""
        from torchvision.io import ImageReadMode, decode_webp

        data = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        return decode_webp(data, mode=ImageReadMode.RGB).to(
            self.device, non_blocking=True
        )

    def _detect_and_decode(self, image_bytes: bytes) -> torch.Tensor:
        """Detect format from magic bytes and decode using GPU, falling back to PIL."""
        # Dispatch table: (magic, offset, length, available_flag, decode_fn)
        _TABLE = [
            (b"\xff\xd8", 0, 2, self._gpu_decode_available, self._decode_jpeg),
            (b"\x89PNG", 0, 4, self._png_decode_available, self._decode_png),
        ]
        # WebP needs extra check at offset 8
        if (
            len(image_bytes) >= 12
            and image_bytes[:4] == b"RIFF"
            and image_bytes[8:12] == b"WEBP"
        ):
            if self._webp_decode_available:
                try:
                    return self._decode_webp(image_bytes)
                except Exception as e:
                    logger.warning(
                        f"MMDecodeImpl: GPU WebP decode failed, falling back to PIL: {e}"
                    )
        else:
            for magic, offset, length, available, fn in _TABLE:
                if (
                    len(image_bytes) >= offset + length
                    and image_bytes[offset : offset + length] == magic
                ):
                    if available:
                        try:
                            return fn(image_bytes)
                        except Exception as e:
                            logger.warning(
                                f"MMDecodeImpl: GPU decode failed for "
                                f"{magic!r}, falling back to PIL: {e}"
                            )
                    break

        # Fallback: PIL decode -> tensor on GPU
        logger.warning(
            f"MMDecodeImpl: using PIL fallback decode (size={len(image_bytes)} bytes)"
        )
        import io

        from PIL import Image as _PILImage

        try:
            img = _PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
            arr = np.asarray(img)
            t = torch.from_numpy(arr.copy()).permute(2, 0, 1).contiguous()
            return t.to(self.device, non_blocking=True)
        except Exception as e:
            logger.error(
                f"[ImageDecode] all backends failed (GPU + PIL) "
                f"size={len(image_bytes)} err={e}",
                exc_info=True,
            )
            raise ImageDecodeError(f"image decode failed (PIL: {e})", code=422) from e

    def _decode_image_cpu(self, image_bytes: bytes):
        """Decode image bytes using PIL (CPU path)."""
        import io

        from PIL import Image as _PILImage

        try:
            img = _PILImage.open(io.BytesIO(image_bytes))
            if (
                self._keep_alpha_channel
                and img.mode != "RGB"
                and ("A" in img.getbands() or "transparency" in img.info)
            ):
                # Same alpha probe as the GPU wrapper's _k3_to_cuda_chw: keep the
                # 4th channel so the processor composites onto its configured
                # background instead of silently discarding alpha.
                return img.convert("RGBA")
            return img.convert("RGB")
        except Exception as e:
            logger.error(
                f"[ImageDecode] PIL decode failed (CPU path) "
                f"size={len(image_bytes)} err={e}",
                exc_info=True,
            )
            raise ImageDecodeError(f"image decode failed (PIL: {e})", code=422) from e

    def _decode_video(self, video_data, use_gpu: bool):
        """Decode video using load_video. Accepts URL string or bytes."""
        if isinstance(video_data, bytes):
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(video_data)
                tmp_path = f.name
            try:
                return load_video(tmp_path, use_gpu=use_gpu)
            finally:
                os.unlink(tmp_path)
        return load_video(video_data, use_gpu=use_gpu)

    def submit(self, req: DecodeReq) -> Future[DecodeResult]:
        """Submit a decode request to the thread pool."""
        return self._pool.submit(self._decode_all, req)

    def _decode_all(self, req: DecodeReq) -> DecodeResult:
        """Decode all multimodal items in the request."""
        start = time.monotonic()
        if self._use_gpu:
            with torch.cuda.stream(self._decode_stream):
                images = []
                for i, raw in enumerate(req.images):
                    modality_str = (
                        req.modalities[i]
                        if req.modalities and i < len(req.modalities)
                        else "image"
                    )
                    if modality_str == "video":
                        images.append(self._decode_video(raw, use_gpu=True))
                    else:
                        images.append(self._detect_and_decode(raw))
            self._decode_stream.synchronize()
        else:
            images = []
            for i, raw in enumerate(req.images):
                modality_str = (
                    req.modalities[i]
                    if req.modalities and i < len(req.modalities)
                    else "image"
                )
                if modality_str == "video":
                    images.append(self._decode_video(raw, use_gpu=False))
                else:
                    images.append(self._decode_image_cpu(raw))
        return DecodeResult(
            rid=req.rid, time_cost=time.monotonic() - start, images=images
        )


# GPU single-thread processor implementation (replaces MMProcessImpl for GPU)
# ---------------------------------------------------------------------------


class MMGPUProcessImpl:
    """GPU image preprocessing: resize + normalize on GPU tensors.

    Accepts decoded GPU tensors from MMDecodeImpl.
    Uses dedicated CUDA stream to avoid blocking VIT inference.
    """

    def __init__(
        self, server_args: ServerArgs, device: torch.device, gpu_processor_cls=None
    ):
        """Initialize MMGPUProcessImpl.

        Args:
            server_args: Server configuration arguments.
            device: CUDA device for GPU operations.
            gpu_processor_cls: GPU image processor class providing
                ``create_gpu_preprocess_config`` and ``gpu_preprocess`` static methods.
                If None, auto-discovered from model architecture via registry.
        """
        if gpu_processor_cls is None:
            gpu_processor_cls = get_gpu_processor_cls(server_args.model_path)
        self._gpu_processor_cls = gpu_processor_cls

        self.name = "mm_gpu_process"
        self.server_args = server_args
        self.device = device
        labels = (
            {
                "model_name": server_args.served_model_name,
            }
            if server_args.served_model_name
            else {}
        )

        torch.cuda.set_device(device)
        self._preprocess_stream = torch.cuda.Stream(device=device)

        self._hf_processor = AutoImageProcessor.from_pretrained(
            server_args.model_path,
            trust_remote_code=server_args.trust_remote_code,
            use_fast=True,
        )
        self._gpu_config = self._gpu_processor_cls.create_gpu_preprocess_config(
            self._hf_processor, device
        )
        self._video_config = None  # lazy-loaded on first video request

        from concurrent.futures import ThreadPoolExecutor

        self._pool = ThreadPoolExecutor(max_workers=1)

        logger.info(f"MMGPUProcessImpl: device={device}")

    def submit(self, req: ProcessReq) -> Future[ProcessResult]:
        """Submit a preprocessing request to the thread pool."""
        return self._pool.submit(self._process, req)

    def _process(self, req: ProcessReq) -> ProcessResult:
        """Run GPU preprocessing on decoded GPU tensors."""
        start_time = time.monotonic()

        if req.modality == Modality.VIDEO:
            return self._process_video_chunk(req, start_time)

        with torch.cuda.stream(self._preprocess_stream):
            # req.images is List[torch.Tensor] (GPU, [C,H,W] uint8) from MMDecodeImpl
            pixel_values, grid_thws = self._gpu_processor_cls.gpu_preprocess(
                images=req.images,
                config=self._gpu_config,
            )
        self._preprocess_stream.synchronize()

        # Feature/metadata integrity guard: pixel_values row count MUST equal the
        # total patches implied by grid_thws (sum of t*h*w). A mismatch means the
        # feature tensor and its grid_thw describe different images, which crashes
        # the ViT positional-embedding add (kimi_k25 pos_emb) with an opaque 500.
        # Fail fast as 422 and name the request so the culprit image is traceable.
        expected_rows = int(grid_thws.prod(dim=1).sum().item())
        actual_rows = pixel_values.shape[0]
        if actual_rows != expected_rows:
            logger.error(
                f"[Process] feature/grid_thw mismatch rid={req.rid} "
                f"pixel_values_rows={actual_rows} grid_expected_rows={expected_rows} "
                f"grid_thws={grid_thws.tolist()}"
            )
            raise ImageDecodeError(
                f"preprocessed feature does not match grid_thw "
                f"(rows {actual_rows} != {expected_rows})",
                code=422,
            )

        mm_item = MultimodalDataItem.from_dict(
            {
                "modality": Modality.IMAGE,
                "feature": pixel_values,
            }
        )
        mm_item.set("image_grid_thw", grid_thws.clone())

        return ProcessResult(
            req.rid,
            time.monotonic() - start_time,
            mm_item,
        )

    def _process_video_chunk(self, req: ProcessReq, start_time: float) -> ProcessResult:
        """Process a single video chunk via model-specific gpu_preprocess_video_chunk."""
        if self._video_config is None:
            self._video_config = self._gpu_processor_cls.create_video_preprocess_config(
                self._hf_processor, self.device
            )
        with torch.cuda.stream(self._preprocess_stream):
            pixel_values, grid_thw = self._gpu_processor_cls.gpu_preprocess_video_chunk(
                frames=req.images[0],  # single (T, C, H, W) tensor
                config=self._video_config,
            )
        self._preprocess_stream.synchronize()

        # Same feature/grid_thw integrity guard as the image path: the video-chunk
        # path was previously unguarded, so a chunk whose patch count disagrees with
        # its grid_thw would slip through and only surface as an opaque ViT 500 in
        # mm_embed (kimi_k25 pos_emb), poisoning the whole batch.
        expected_rows = int(grid_thw.prod().item())
        actual_rows = int(pixel_values.shape[0])
        if actual_rows != expected_rows:
            logger.error(
                f"[Process] video feature/grid_thw mismatch rid={req.rid} "
                f"pixel_values_rows={actual_rows} grid_expected_rows={expected_rows} "
                f"grid_thw={grid_thw.tolist()}"
            )
            raise ImageDecodeError(
                f"preprocessed video chunk feature does not match grid_thw "
                f"(rows {actual_rows} != {expected_rows})",
                code=422,
            )

        mm_item = MultimodalDataItem.from_dict(
            {
                "modality": Modality.VIDEO,
                "feature": pixel_values,
            }
        )
        mm_item.set("image_grid_thw", grid_thw.unsqueeze(0).clone())

        return ProcessResult(
            req.rid,
            time.monotonic() - start_time,
            mm_item,
        )


class MMProcessImpl:
    """Multimodal preprocessing implementation using a process pool.

    Supports optional shared-memory IPC to avoid pickling large tensors
    between worker processes and the main process.

    Features:
    - Auto-recovery when worker processes crash (BrokenProcessPool).
    """

    _MAX_RECOVERY_ATTEMPTS = 3

    def __init__(
        self,
        server_args: ServerArgs,
        max_batch_size: int,
        /,
        use_shm_ipc: bool = False,
        shm_buf_size: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize the multimodal process pool.

        Args:
            server_args: Server configuration arguments.
            max_batch_size: Number of worker processes in the pool.
            use_shm_ipc: If True, use shared memory for tensor transfer.
            shm_buf_size: Size of each shared memory buffer in bytes.
                          Required when use_shm_ipc is True.
            device: Target device for tensors when using shm IPC.
                    If provided, tensors are transferred directly from shm
                    to device, avoiding an intermediate CPU copy.
        """
        self.name = "mm_process"
        self.max_batch_size = max_batch_size
        self.server_args = server_args
        self._use_shm_ipc = use_shm_ipc
        self._device = device

        if use_shm_ipc:
            if shm_buf_size is None:
                raise ValueError(
                    "shm_buf_size must be specified when use_shm_ipc is True"
                )
        self.shm_buf_size = shm_buf_size

        labels = {
            "model_name": self.server_args.served_model_name,
        }
        if server_args.extra_metric_labels:
            labels.update(server_args.extra_metric_labels)


        # Pool recovery state
        self._recovery_lock = threading.Lock()
        self._recovery_failures = 0

        self.shm_buffers: List[shared_memory.SharedMemory] = []

        self._pool = self._create_pool()

        if use_shm_ipc:
            self._shm_pool: queue.Queue[shared_memory.SharedMemory] = queue.Queue()
            self.__create_shm_buffer()
            atexit.register(self.close)

    def _create_pool(self):
        """Create a new ProcessPoolExecutor with the configured settings."""
        from concurrent.futures import ProcessPoolExecutor

        return ProcessPoolExecutor(
            self.max_batch_size,
            mp_context=mp.get_context("spawn"),
            initializer=init___mm_processor,
            initargs=(self.server_args,),
        )

    def __create_shm_buffer(self):
        """Allocate shared memory buffers, one per worker process."""
        if not self._use_shm_ipc:
            return

        for _ in range(self.max_batch_size):
            shm = shared_memory.SharedMemory(create=True, size=self.shm_buf_size)
            self.shm_buffers.append(shm)
            self._shm_pool.put(shm)
            logger.info(f"Created shm: {shm.name}")

    def __release_shm_buffer(self):
        """Close and unlink all shared memory buffers."""
        if not self.shm_buffers:
            return

        for shm in self.shm_buffers:
            try:
                shm.close()
                shm.unlink()
                logger.info(f"Released shm: {shm.name}")
            except Exception:
                logger.error(f"Failed to close or unlink shm: {shm.name}")
        self.shm_buffers.clear()

    # ── Pool recovery ─────────────────────────────────────────────────────

    def _recreate_pool(self):
        """Tear down the broken pool and create a fresh one.

        SHM buffers are NOT recreated — they are owned by the main process
        and remain valid across pool restarts. New workers open them by name.
        """
        logger.warning("MMProcessImpl: recreating process pool...")

        # 1. Shut down the broken pool (best-effort)
        try:
            self._pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

        # 2. Create new pool (workers will open existing shm by name)
        self._pool = self._create_pool()

        logger.warning("MMProcessImpl: pool recovered")

    def _do_submit(self, req: ProcessReq) -> Future[ProcessResult]:
        """Internal submit without recovery. Raises BrokenProcessPool on failure."""
        if not self._use_shm_ipc:
            return self._pool.submit(mm_preprocess, req)
        return self._submit_with_shm(req)

    def submit(self, req: ProcessReq) -> Future[ProcessResult]:
        """Submit a preprocessing request to the process pool.

        Automatically recovers from BrokenProcessPool by recreating the pool
        and retrying the request.

        Args:
            req: The process request containing images to preprocess.

        Returns:
            A Future that will contain the ProcessResult.
        """
        try:
            return self._do_submit(req)
        except BrokenProcessPool:
            logger.warning(
                f"MMProcessImpl: BrokenProcessPool detected in submit() for rid={req.rid}, "
                "attempting recovery..."
            )
            with self._recovery_lock:
                # Double-check: pool may have been recovered by another thread already
                try:
                    self._pool.submit(lambda: None).result(timeout=5)
                except BrokenProcessPool:
                    if self._recovery_failures >= self._MAX_RECOVERY_ATTEMPTS:
                        raise RuntimeError(
                            f"MMProcessImpl: pool recovery failed "
                            f"{self._recovery_failures} consecutive times, giving up"
                        )
                    try:
                        self._recreate_pool()
                        self._recovery_failures = 0
                    except Exception as recovery_err:
                        self._recovery_failures += 1
                        raise RuntimeError(
                            "MMProcessImpl: pool recovery failed"
                        ) from recovery_err

            # Retry on the new pool
            return self._do_submit(req)

    # ── SHM read helpers (main-process-side) ────────────────────────────

    def _read_tensors_from_shm(
        self, shm: shared_memory.SharedMemory, metas: List[ShmTensorMeta]
    ) -> Dict[str, torch.Tensor]:
        """Reconstruct tensors from a shared memory buffer.

        Each tensor is *copied* out of shm so the buffer can be safely reused.
        When a target device is set, tensors are transferred directly from shm
        to device (1 copy) instead of going through an intermediate CPU
        allocation (2 copies).
        """
        result: Dict[str, torch.Tensor] = {}
        for meta in metas:
            np_arr = np.ndarray(
                meta.shape,
                dtype=np.dtype(meta.dtype),
                buffer=shm.buf,
                offset=meta.offset,
            )
            # Zero-copy view of shm; .to(device) or .copy() materialises data.
            tmp = torch.from_numpy(np_arr)
            if self._device is not None:
                result[meta.key] = tmp.to(self._device)
            else:
                result[meta.key] = torch.from_numpy(np_arr.copy())
        return result

    def _reconstruct_process_result(
        self, shm: shared_memory.SharedMemory, shm_result: ShmProcessResult
    ) -> ProcessResult:
        """Read tensors from shm and rebuild a ProcessResult."""
        tensors = self._read_tensors_from_shm(shm, shm_result.tensor_metas)
        mm_item = MultimodalDataItem.from_dict(
            {
                "modality": shm_result.modality,
                "feature": tensors.pop("feature"),
            }
        )
        for k, v in tensors.items():
            mm_item.set(k, v)
        for k, v in shm_result.non_tensor_data.items():
            mm_item.set(k, v)
        return ProcessResult(
            rid=shm_result.rid,
            time_cost=shm_result.time_cost,
            mm_item=mm_item,
        )

    def _submit_with_shm(self, req: ProcessReq) -> Future[ProcessResult]:
        """Submit a single req via shm."""
        shm_buf = self._shm_pool.get()  # borrow a buffer from the pool
        try:
            impl_fut = self._pool.submit(mm_preprocess_with_shm, req, shm_buf.name)
        except Exception:
            self._shm_pool.put(shm_buf)  # return buffer on submit failure
            raise

        wrapper: Future[ProcessResult] = Future()

        def _on_done(f: Future):
            try:
                shm_result: ShmProcessResult = f.result()
                result = self._reconstruct_process_result(shm_buf, shm_result)
                wrapper.set_result(result)
            except BrokenProcessPool as e:
                logger.warning(f"MMProcessImpl: worker crashed for rid={req.rid}")
                if not wrapper.done():
                    wrapper.set_exception(e)
            except Exception as e:
                if not wrapper.done():
                    wrapper.set_exception(e)
            finally:
                self._shm_pool.put(shm_buf)  # always return buffer

        impl_fut.add_done_callback(_on_done)
        return wrapper

    def close(self):
        """Release all shared memory buffers."""
        self.__release_shm_buffer()


@dataclass
class EmbedReq:
    """Embed request."""

    rid: str
    mm_item: "MultimodalDataItem"
    image_url: str
    task_id: str
    num_output_tokens: int
    num_load_tokens: Optional[int] = None

    def num_tokens(self) -> int:
        """Return the token weight used for scheduling and pending accounting."""
        return self.num_load_tokens or self.num_output_tokens

    def mm_hash(self) -> str:
        """Return hash key for this request."""
        return self.task_id


@dataclass
class EmbedResult:
    """Embed result."""

    rid: str
    time_cost: float
    image_grid_dim: Any
    mm_embedding: torch.Tensor | ScatteredTensor
    num_load_tokens: Optional[int] = None
    # Original ``[width, height]`` per image, before any resize. Only models
    # whose prompt encodes the source resolution (Kimi-K3) need it.
    original_image_sizes: Optional[List[List[int]]] = None

    def batch_size(self) -> int:
        """Return the batch size of this embedding result."""
        return 1  # TODO 如何获取正确的batch_size？

    def num_tokens(self) -> int:
        """Return the token weight used for throughput metrics."""
        return self.num_load_tokens or self.mm_embedding.shape[0]


class EmbeddingImpl:
    """Vision encoder embedding implementation.

    Runs the model's vision encoder on batches of preprocessed multimodal items.
    Pure GPU inference — cache logic is handled by the caller (_encode pipeline).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        server_args: ServerArgs,
        gpu_processor_cls=None,
    ) -> None:
        """Initialize the embedding implementation.

        Args:
            model: The vision model used to extract image features.
            server_args: Server configuration arguments.
            gpu_processor_cls: Model-specific GPU processor class that owns token
                accounting logic. If None, auto-discovered from model config.
        """
        if gpu_processor_cls is None:
            gpu_processor_cls = get_gpu_processor_cls(server_args.model_path)
        self._gpu_processor_cls = gpu_processor_cls
        self.model = model
        self.name = "mm_embed"
        self.server_args = server_args
        labels = {
            "model_name": self.server_args.served_model_name,
        }
        if server_args.extra_metric_labels:
            labels.update(server_args.extra_metric_labels)


    def __call__(self, reqs: List[EmbedReq]) -> Iterable[EmbedResult]:
        """Run vision encoding on a batch of embed requests.

        Args:
            reqs: List of embedding requests, each containing a MultimodalDataItem.

        Returns:
            A list of EmbedResult, one per input request.
        """
        embed_start_time = time.monotonic()

        # Path-independent integrity check: every item entering the ViT MUST satisfy
        # feature.shape[0] == Σ(t*h*w) from its grid_thw, otherwise the pos_emb add
        # (kimi_k25:297) crashes with an opaque 500 that poisons the WHOLE batch and
        # fails unrelated co-batched requests. Validate here — the single choke point
        # all items (image + video) pass through — and on ANY mismatch flag the
        # offending item(s) so a bad chunk doesn't silently poison the batch.
        _batch_bad = False
        _batch_rows = []
        for r in reqs:
            grid = get_image_grid_dim(r.mm_item)
            if not isinstance(grid, torch.Tensor):
                grid = torch.as_tensor(grid)
            if grid.dim() == 1:
                grid = grid.unsqueeze(0)
            expected_rows = int(grid.prod(dim=1).sum().item())
            feat = getattr(r.mm_item, "feature", None)
            actual_rows = int(feat.shape[0]) if isinstance(feat, torch.Tensor) else -1
            ok = actual_rows == expected_rows
            _batch_bad = _batch_bad or not ok
            _batch_rows.append((r.rid, ok, expected_rows, actual_rows, r.mm_item))

        if _batch_bad:
            logger.error(
                "[mm_embed] BATCH feature/grid_thw mismatch batch_size=%d",
                len(reqs),
            )
            # On-error only (never fires on healthy traffic): dump an identity
            # signature for EVERY item in the batch so the swap can be localized.
            # Cross-referencing feat_ptr / feat_stor across items reveals whether a
            # BAD item's feature aliases (or was replaced by) a neighbour's tensor;
            # grid_id / grid values tell us if it was the feature or the grid that
            # got crossed after the per-item _process guard already passed.
            for rid, ok, exp, act, mm_item in _batch_rows:
                feat = getattr(mm_item, "feature", None)
                if isinstance(feat, torch.Tensor):
                    try:
                        _stor = feat.untyped_storage().data_ptr()
                        _nbytes = feat.untyped_storage().nbytes()
                    except Exception:
                        _stor, _nbytes = -1, -1
                    fsig = (
                        f"feat_id={id(feat)} feat_shape={tuple(feat.shape)} "
                        f"feat_dtype={feat.dtype} feat_dev={feat.device} "
                        f"feat_ptr={feat.data_ptr():#x} feat_stor={_stor:#x} "
                        f"feat_bytes={_nbytes} contig={feat.is_contiguous()}"
                    )
                else:
                    fsig = f"feat_type={type(feat).__name__}"
                try:
                    _grid = get_image_grid_dim(mm_item)
                    _gval = _grid.tolist() if isinstance(_grid, torch.Tensor) else _grid
                    gsig = f"grid={_gval} grid_id={id(_grid)}"
                except Exception as _ge:
                    gsig = f"grid_err={_ge!r}"
                logger.error(
                    "[mm_embed]   [%s] rid=%s modality=%s expected_rows=%s actual_rows=%s "
                    "mm_item_id=%s %s %s",
                    "OK " if ok else "BAD",
                    rid,
                    getattr(mm_item, "modality", "?"),
                    exp,
                    act,
                    id(mm_item),
                    fsig,
                    gsig,
                )

        with torch.inference_mode():
            mm_embedding: torch.Tensor = self.model.get_image_feature(
                [r.mm_item for r in reqs]
            )
            # Use non_blocking D2H + event sync instead of .cpu() to avoid
            # cudaStreamSynchronize holding the CUDA context mutex, which blocks
            # kernel launches from other threads (e.g. GPU preprocess).
            mm_embedding = mm_embedding.to("cpu", non_blocking=True)
            torch.cuda.current_stream().record_event().synchronize()

        if len(mm_embedding.shape) != 2:
            mm_embedding = mm_embedding.reshape(-1, mm_embedding.shape[-1])

        # Compute per-item token counts and split the flat embedding
        tokens_per_item = [
            compute_num_tokens_per_item(self.model, r.mm_item, self._gpu_processor_cls)
            for r in reqs
        ]
        per_item_embeddings = torch.split(mm_embedding, tokens_per_item, dim=0)

        results = []
        for r, embed in zip(reqs, per_item_embeddings):
            image_grid_dim = get_image_grid_dim(r.mm_item)
            time_cost = time.monotonic() - embed_start_time
            results.append(
                EmbedResult(
                    r.rid,
                    time_cost,
                    image_grid_dim,
                    embed,
                    num_load_tokens=r.num_load_tokens,
                )
            )

        return results


@dataclass
class TransferReq:
    """Transfer request for embedding data."""

    rid: str
    task_id: str
    session_id: str
    part_idx: int
    buffer_address: Ptr  # 接收方地址
    byte_size: Nbytes  # 接收方期望的字节数
    start_time: float
    embed: torch.Tensor | ScatteredTensor

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, __value: object) -> bool:
        return self is __value

    def num_tokens(self) -> int:
        """Return size in bytes."""
        if isinstance(self.embed, torch.Tensor):
            return self.embed.numel() * self.embed.element_size()
        else:
            return self.embed.nbytes


@dataclass
class TransferResult:
    """Transfer result."""

    rid: str
    task_id: str
    part_idx: int
    status_code: int
    message: str
    transfer_cost_ms: float
    total_cost_ms: float
    transferred_bytes: int = 0

    def batch_size(self) -> int:
        """Return batch size."""
        return 1

    def num_tokens(self) -> int:
        """Return transferred bytes."""
        return self.transferred_bytes


class TransferImpl:
    """Mooncake transfer implementation."""

    def __init__(
        self, server_args: ServerArgs, hostname: str, gpu_id: int, ib_device: str
    ):
        self.name = "mooncake_transfer"
        self.server_args = server_args

        self.server_args = server_args
        labels = {
            "model_name": self.server_args.served_model_name,
        }
        if server_args.extra_metric_labels:
            labels.update(server_args.extra_metric_labels)


        from mooncake.engine import TransferEngine

        self._engine = TransferEngine()
        ret = self._engine.initialize(hostname, "P2PHANDSHAKE", "rdma", ib_device)
        if ret != 0:
            raise RuntimeError("Mooncake TransferEngine init failed")

        self.hostname = hostname
        self.gpu_id = gpu_id
        self.session_id = f"{hostname}:{self._engine.get_rpc_port()}"

    def register_memory(self, data_ptr: int, nbytes: int) -> None:
        """注册 RDMA buffer 用于零拷贝传输。

        Args:
            data_ptr: buffer 起始地址
            nbytes: buffer 字节数

        Raises:
            RuntimeError: 注册失败时抛出
        """
        ret = self._engine.register_memory(data_ptr, nbytes)
        if ret != 0:
            raise RuntimeError(
                f"RDMA register_memory failed (ret={ret}), "
                f"ptr=0x{data_ptr:x}, size={nbytes} ({nbytes / (1024**3):.2f} GB). "
                f"Check ulimit -l and RDMA device memory registration limits."
            )
        logger.info(f"RDMA engine registered 0x{data_ptr:x}, size {nbytes}")

    def __batch_send(
        self, reqs: List[TransferReq]
    ) -> tuple[bool, List[tuple[TransferReq, TransferResult]]]:
        if not reqs:
            return True, []

        session_id = reqs[0].session_id

        # 必须是同一个接收方的
        assert all(req.session_id == session_id for req in reqs)

        # 1. 查 cache，收集所有 segments
        all_src: List[int] = []
        all_dst: List[int] = []
        all_len: List[int] = []

        for req in reqs:
            emb = req.embed
            assert isinstance(
                emb, ScatteredTensor
            ), f"expected ScatteredTensor, got {type(emb)}"

            dst_offset = 0
            for src_ptr, nbytes in emb._mems:
                actual = min(nbytes, req.byte_size - dst_offset)
                if actual <= 0:
                    break
                # 尝试与上一段合并
                if (
                    all_src
                    and src_ptr == all_src[-1] + all_len[-1]
                    and req.buffer_address + dst_offset == all_dst[-1] + all_len[-1]
                ):
                    all_len[-1] += actual
                else:
                    all_src.append(src_ptr)
                    all_dst.append(req.buffer_address + dst_offset)
                    all_len.append(actual)
                dst_offset += actual

        # 2. 一次 batch_write
        t0 = time.perf_counter()
        try:
            ret = self._engine.batch_transfer_sync_write(
                session_id, all_src, all_dst, all_len
            )
            success = ret == 0
        except Exception as exc:
            success = False
            ret = str(exc)

        end = time.perf_counter()
        elapsed = end - t0

        if not success:
            msg = f"batch_write failed: batch_write returned {ret}"
            code = 500
            logger.error(
                f"batch_write failed: batch_write returned {ret}, "
                f"remote={session_id}, "
                f"task_ids={[r.task_id for r in reqs]}"
            )
        else:
            msg = ""
            code = 200
            total_bytes = sum(all_len)
            bw = (total_bytes >> 20) / elapsed if elapsed > 0 else 0
            logger.info(
                f"embed transfer: remote={session_id}, batch_size={len(reqs)}, "
                f"task_ids={[r.task_id for r in reqs]}, "
                f"total_bytes={total_bytes}, cost={elapsed:.3f} s, bw={bw:.3f} MB/s"
            )
        results = [
            (
                req,
                TransferResult(
                    rid=req.rid,
                    task_id=req.task_id,
                    part_idx=req.part_idx,
                    status_code=code,
                    message=msg,
                    transfer_cost_ms=elapsed * 1000,
                    total_cost_ms=(end - req.start_time) * 1000,
                    transferred_bytes=req.byte_size,
                ),
            )
            for req in reqs
        ]
        return success, results

    def __group(self, reqs: List[TransferReq]) -> List[List[TransferReq]]:
        """根据 session_id 将请求分组，每组内的请求发送到同一个接收方。"""
        groups: dict = defaultdict(list)
        for req in reqs:
            groups[req.session_id].append(req)
        return list(groups.values())

    def __call__(self, reqs: List[TransferReq]) -> Iterable[TransferResult]:
        """并行向各 session 发送 embedding，一次重试失败的 session。

        按 reqs 原始顺序返回 TransferResult。
        """
        groups = [g for g in self.__group(reqs) if g]
        if not groups:
            return []

        results: List[tuple[TransferReq, TransferResult]] = []
        failed: List[List[TransferReq]] = []

        # 第一轮：各 session 并行传输
        with ThreadPoolExecutor(max_workers=len(groups)) as pool:
            fut_to_group = {pool.submit(self.__batch_send, g): g for g in groups}
            for fut in as_completed(fut_to_group):
                success, res = fut.result()
                if success:
                    results.extend(res)
                else:
                    failed.append(fut_to_group[fut])

        # 重试1次（各失败 session 仍并行）
        if failed:
            with ThreadPoolExecutor(max_workers=len(failed)) as pool:
                retry_futs = {pool.submit(self.__batch_send, g): g for g in failed}
                for fut in as_completed(retry_futs):
                    _, res = fut.result()
                    results.extend(res)

        # 按照 reqs 里的顺序对 results 排序
        temp = dict(results)
        return [temp[r] for r in reqs]


def _probe_pynvvideocodec() -> bool:
    """Check if PyNvVideoCodec with SimpleDecoder is available."""
    try:
        import PyNvVideoCodec as nvc

        assert hasattr(nvc, "SimpleDecoder")
        return True
    except (ImportError, AssertionError):
        return False


class VideoDecodeError(Exception):
    """Video content could not be decoded by any available backend, even though
    the declared format matched the container. Carries HTTP 422: the request was
    understood but the media itself is unprocessable, retrying won't help."""

    def __init__(self, message, code=422):
        super().__init__(message)
        self.code = code


class RetryableVideoDecodeError(VideoDecodeError):
    """End this NVDEC attempt; its owner may restart once with CPU decord."""


class ImageDecodeError(Exception):
    """Image content could not be decoded by any available backend (GPU nor PIL).
    Carries HTTP 422: the request was understood but the image itself is
    unprocessable (corrupt / unsupported), retrying won't help."""

    def __init__(self, message, code=422):
        super().__init__(message)
        self.code = code


class DecodeConsumerGone(Exception):
    """Raised on the decode producer side when its consumer has gone away
    (the streaming task was cancelled/timed out). It signals the producer
    (_ChunkStreamWriter.emit blocked on a full, no-longer-drained decode_queue,
    and run_video_stream around it) to abort the current video instead of
    blocking forever -- the root-cause fix for the cancel/producer deadlock that
    leaked all decode workers and wedged the video pipeline."""


def _discard_partial_embed_result(future):
    if future.cancelled():
        return
    try:
        result = future.result()
    except Exception:
        return
    try:
        result.mm_embedding = None
    except Exception:
        pass


def release_partial_embeds(embed_results, inflight, submitted_futures=()):
    """Free GPU memory held by already-produced chunk embeddings when a streaming
    video task is cancelled/failed, instead of relying on the cancelled coroutine
    being garbage-collected later. Nulls each collected result's ``mm_embedding``,
    cancels pending embed futures, and clears both lists. Best-effort: never raises.

    IMPORTANT: it must NOT touch ``mm_item.feature`` of in-flight requests. Those
    mm_items are still referenced by the SHARED embed executor's pending batch
    (get_image_feature does ``torch.cat([item.feature for item in items])``);
    nulling a feature there corrupts the whole batch and fails OTHER concurrent
    tasks with "expected Tensor ... got NoneType". We only cancel the future (a
    no-op if it is already running) and let normal GC reclaim the feature once the
    request leaves the executor."""
    for r in list(embed_results):
        try:
            if r is not None and getattr(r, "mm_embedding", None) is not None:
                r.mm_embedding = None
        except Exception:
            pass
    try:
        embed_results.clear()
    except Exception:
        pass
    # Includes the future popped by _drain_one when cancellation interrupted its
    # await. A running shared batch may complete later; discard only its result,
    # while leaving the input feature alive for the executor that still owns it.
    for future in submitted_futures:
        future.add_done_callback(_discard_partial_embed_result)
    for item in list(inflight):
        try:
            item[2].cancel()  # (idx, embed_req, future); no-op if already running
        except Exception:
            pass
    try:
        inflight.clear()
    except Exception:
        pass


# Poll interval for the bounded chunk-queue put (see _ChunkStreamWriter._put):
# how often the decode producer re-checks stop_event while blocked on a full
# queue, i.e. the max delay before a vanished consumer unblocks the producer.
_CHUNK_PUT_POLL_S = 0.5


class _ChunkStreamWriter:
    """Single definition of the decode -> consumer streaming queue protocol.

    Both the in-process (`_decode_stream`) and isolated (`_decode_stream_isolated`)
    decode paths use this to emit identical ``("chunk", ...)`` / ``("done", ...)``
    messages, so the protocol lives in one place instead of being duplicated.
    ``event`` is an optional per-chunk CUDA event the consumer must wait on
    before reading the tensor (used by the isolated CUDA-IPC path)."""

    def __init__(
        self, output_queue: queue.Queue, stop_event: "threading.Event | None" = None
    ):
        self._q = output_queue
        # Set by the consumer when it is cancelled/timed out; makes emit() stop
        # blocking on a full, no-longer-drained queue and abort instead.
        self._stop_event = stop_event
        self.n = 0
        self.frame_cost = 0.0
        self.timestamps = []
        self.height = 0
        self.width = 0

    def _put(self, item):
        """Bounded put that stays responsive to consumer cancellation.

        A plain blocking put on the bounded decode_queue deadlocks if the
        consumer was cancelled (timeout) and stopped draining: the queue stays
        full forever and the decode thread/worker are leaked. Instead poll with a
        short timeout and re-check stop_event, so a vanished consumer unblocks the
        producer within ~_CHUNK_PUT_POLL_S and aborts via DecodeConsumerGone."""
        while True:
            if self._stop_event is not None and self._stop_event.is_set():
                raise DecodeConsumerGone("decode consumer cancelled")
            try:
                self._q.put(item, timeout=_CHUNK_PUT_POLL_S)
                return
            except queue.Full:
                continue

    def emit(
        self, chunk_idx, chunk_tensor, timestamp, height, width, frame_cost, event=None
    ):
        """Enqueue one decoded chunk and accumulate its running stats."""
        if chunk_idx != self.n:
            raise VideoDecodeError(
                f"non-contiguous video chunks: expected={self.n} actual={chunk_idx}",
                code=500,
            )
        self._put(
            (
                "chunk",
                chunk_idx,
                chunk_tensor,
                timestamp,
                height,
                width,
                frame_cost,
                event,
            )
        )
        self.n += 1
        self.frame_cost += frame_cost
        self.timestamps.append(timestamp)
        self.height, self.width = height, width

    def done(self, start_time):
        """Signal end-of-stream, enqueuing the aggregated per-video summary."""
        self._put(
            (
                "done",
                self.n,
                time.monotonic() - start_time,
                self.frame_cost,
                self.height,
                self.width,
                self.timestamps,
            )
        )


class VideoDecodeImpl:
    """Video decode-only pipeline: download + NVDEC/decord decode + sample + chunk.

    Outputs raw GPU frame tensors per chunk WITHOUT preprocessing.
    Used as Stage 1 in the 3-stage video pipeline.
    """

    name = "video_decode"

    def __init__(
        self,
        server_args: ServerArgs,
        max_batch_size: int,
        _is_decode_worker: bool = False,
    ):
        self.server_args = server_args
        labels = (
            {"model_name": server_args.served_model_name}
            if server_args.served_model_name
            else {}
        )
        from concurrent.futures import ThreadPoolExecutor

        self._pool = ThreadPoolExecutor(max_batch_size)
        self._device = torch.device("cuda", torch.cuda.current_device())
        self._gpu_index = torch.cuda.current_device()

        self._decode_backend = os.environ.get(
            "SGLANG_VIDEO_DECODE_BACKEND", "auto"
        ).lower()
        self._decode_chunks_per_batch = max(
            1,
            int(os.environ.get("SGLANG_ENCODER_VIDEO_DECODE_CHUNKS_PER_BATCH", "2")),
        )
        self._pynv_available = False
        if self._decode_backend in ("pynvvideocodec", "auto"):
            self._pynv_available = _probe_pynvvideocodec()

        self._gpu_available = self._probe_gpu_decode()
        self._preprocess_stream = torch.cuda.Stream(device=self._device)

        # Use processor registry for model-specific video logic
        self._gpu_processor_cls = get_gpu_processor_cls(server_args.model_path)
        # A processor without video hooks is image-only and must still start.
        # Everything else raises: a guessed timestamp_mode or chunk size would
        # silently mis-render every video prompt.
        self._video_config = None
        self._tmks = None
        self._timestamp_mode = None
        if hasattr(self._gpu_processor_cls, "create_video_preprocess_config"):
            proc = AutoImageProcessor.from_pretrained(
                server_args.model_path,
                trust_remote_code=server_args.trust_remote_code,
            )
            self._video_config = self._gpu_processor_cls.create_video_preprocess_config(
                proc, self._device
            )
            self._tmks = self._gpu_processor_cls.get_temporal_merge_kernel_size(
                self._video_config
            )
            self._timestamp_mode = proc.media_proc_cfg.get("timestamp_mode", "mm:ss")
            logger.info(
                "VideoDecodeImpl: tmks=%d timestamp_mode=%s sample_fps=%s",
                self._tmks,
                self._timestamp_mode,
                self._video_config.sample_fps,
            )
        else:
            logger.warning(
                "%s has no video hooks; this encoder serves images only",
                self._gpu_processor_cls.__name__,
            )

        logger.info(
            f"VideoDecodeImpl: pynv={self._pynv_available}, decord_gpu={self._gpu_available}"
        )

        # NVDEC crash isolation is ALWAYS on when pynv (NVDEC) is available:
        # decode runs in subprocess workers so a native SIGFPE on a malformed
        # bitstream only kills a worker, never the encoder. Skipped inside the
        # worker itself (_is_decode_worker) to avoid recursion, and when pynv is
        # absent (no NVDEC -> no crash risk -> plain in-process decode).
        self._worker_pool = None
        if self._pynv_available and not _is_decode_worker:
            from sglang.srt.disaggregation.video_decode_worker import (
                VideoDecodeWorkerPool,
            )

            timeout_s = float(
                os.environ.get("SGLANG_VIDEO_DECODE_WORKER_TIMEOUT_S", "30")
            )
            self._worker_pool = VideoDecodeWorkerPool(
                server_args, self._gpu_index, timeout_s
            )
            logger.info("[VideoDecodeImpl] NVDEC crash isolation ENABLED")

    def _probe_gpu_decode(self) -> bool:
        if self._decode_backend == "decord_cpu":
            return False
        try:
            from decord import gpu

            gpu(0)
            return True
        except Exception:
            return False

    def submit(self, req: VideoLoadReq):
        """Submit a video load request for batch (non-streaming) decoding.

        NOTE: the encoder serves video via ``submit_stream``; this batch API is
        currently unused by the server and kept for tests / future use."""
        if self._worker_pool is not None:
            return self._pool.submit(self._decode_isolated, req)
        return self._pool.submit(self._decode, req)

    def submit_stream(
        self, req: VideoLoadReq, output_queue: queue.Queue, stop_event=None
    ):
        """Submit a video decode request that streams one chunk per queue item.

        ``stop_event`` (threading.Event) lets the consumer signal cancellation so
        the producer aborts instead of blocking on a full, undrained queue."""
        if self._worker_pool is not None:
            return self._pool.submit(
                self._decode_stream_isolated, req, output_queue, stop_event
            )
        return self._pool.submit(self._decode_stream, req, output_queue, stop_event)

    def _decode_stream_isolated(
        self, req: VideoLoadReq, output_queue: queue.Queue, stop_event=None
    ):
        """Decode one isolated attempt. Its owner handles whole-task retries."""
        start_time = time.monotonic()
        writer = _ChunkStreamWriter(output_queue, stop_event)
        _should_abort = stop_event.is_set if stop_event is not None else None
        try:
            for local_path in req.video_items:
                emitted_before = writer.n

                def on_chunk(idx, t, timestamp, height, width, frame_cost):
                    # CUDA IPC zero-copy: D2D-clone the worker's GPU frame into
                    # this process, record a per-chunk event, and let the
                    # consumer's preprocess stream wait on it -> keeps
                    # decode/upload/embed overlapped while preventing dirty reads.
                    gpu = t.clone()
                    ev = torch.cuda.current_stream().record_event()
                    try:
                        writer.emit(
                            emitted_before + idx,
                            gpu,
                            timestamp,
                            height,
                            width,
                            frame_cost,
                            ev,
                        )
                    except BaseException:
                        # The worker may be stopped after this callback fails.
                        # Finish our clone before its IPC source is destroyed.
                        ev.synchronize()
                        raise
                    # Return the clone event so the pool waits on it before acking
                    # the worker (which then frees/reuses the source IPC block).
                    return ev

                crashed = self._worker_pool.run_video_stream(
                    local_path,
                    req.sample_fps,
                    on_chunk,
                    force_decord=req.force_decord,
                    should_abort=_should_abort,
                )
                if crashed:
                    error_type = (
                        VideoDecodeError
                        if req.force_decord
                        else RetryableVideoDecodeError
                    )
                    raise error_type(
                        f"video decode worker failed: rid={req.rid} "
                        f"force_decord={req.force_decord} emitted={writer.n} path={local_path}"
                    )
            writer.done(start_time)
        except DecodeConsumerGone:
            # Consumer was cancelled/timed out and the producer aborted; nothing
            # to report back (the queue is no longer drained).
            return
        except Exception as e:
            try:
                output_queue.put(("error", e), timeout=1.0)
            except queue.Full:
                pass
            raise

    def _decode_isolated(self, req: VideoLoadReq) -> "VideoDecodeResult":
        """Batch variant of the isolated decode: collect streamed chunks into a
        single VideoDecodeResult (used by the non-streaming path)."""
        import queue as _q

        # Parity with the streaming path: pass a stop_event so a future caller of
        # this (currently server-unused) batch API has a cancellation hook. It is
        # never set here because _decode_isolated runs synchronously to completion.
        stop_event = threading.Event()
        for attempt in range(2):
            buf: _q.Queue = _q.Queue()
            attempt_req = replace(req, force_decord=req.force_decord or attempt == 1)
            try:
                self._decode_stream_isolated(attempt_req, buf, stop_event)
                break
            except RetryableVideoDecodeError:
                # No consumer has seen this batch attempt; discard all of it.
                while not buf.empty():
                    buf.get_nowait()
                if attempt_req.force_decord:
                    raise
                self._worker_pool.fallback_count += 1
        chunks, timestamps = [], []
        video_h = video_w = 0
        total_frame_cost = 0.0
        start_time = time.monotonic()
        while True:
            item = buf.get()
            if item[0] == "error":
                raise item[1]
            if item[0] == "done":
                _, _n, _dt, total_frame_cost, video_h, video_w, timestamps = item
                break
            # ("chunk", idx, chunk_tensor, ts, h, w, cost, event)
            chunk_tensor = item[2]
            video_h, video_w = item[4], item[5]
            chunks.append(chunk_tensor)
        # IPC-cloned chunks were not per-chunk synced (batch path has no
        # consumer-side event wait); ensure all clones are complete here.
        torch.cuda.current_stream().synchronize()
        return VideoDecodeResult(
            rid=req.rid,
            time_cost=time.monotonic() - start_time,
            download_cost=0.0,
            frame_extraction_cost=total_frame_cost,
            chunks=chunks,
            video_height=video_h,
            video_width=video_w,
            chunk_timestamps=timestamps,
        )

    def _decode(self, req: VideoLoadReq) -> VideoDecodeResult:
        start_time = time.monotonic()
        total_frame_cost = 0.0
        all_chunks = []
        all_timestamps = []
        video_h = video_w = 0

        for local_path in req.video_items:
            fc, chunks, h, w, timestamps = self._decode_video(local_path, req)
            total_frame_cost += fc
            all_chunks.extend(chunks)
            all_timestamps.extend(timestamps)
            video_h, video_w = h, w

        return VideoDecodeResult(
            rid=req.rid,
            time_cost=time.monotonic() - start_time,
            download_cost=0.0,
            frame_extraction_cost=total_frame_cost,
            chunks=all_chunks,
            video_height=video_h,
            video_width=video_w,
            chunk_timestamps=all_timestamps,
        )

    def _decode_stream(
        self, req: VideoLoadReq, output_queue: queue.Queue, stop_event=None
    ):
        start_time = time.monotonic()
        writer = _ChunkStreamWriter(output_queue, stop_event)
        try:
            for local_path in req.video_items:
                offset = writer.n
                for item in self._iter_decode_video_chunks(local_path, req):
                    chunk_idx, chunk_tensor, height, width, timestamp, frame_cost = item
                    writer.emit(
                        offset + chunk_idx,
                        chunk_tensor,
                        timestamp,
                        height,
                        width,
                        frame_cost,
                    )
            writer.done(start_time)
        except DecodeConsumerGone:
            return
        except Exception as e:
            try:
                output_queue.put(("error", e), timeout=1.0)
            except queue.Full:
                pass
            raise

    # Map a declared file extension to the container family it implies.
    _EXT_TO_CONTAINER = {
        ".avi": "avi",
        ".mp4": "mp4",
        ".m4v": "mp4",
        ".mov": "mp4",
        ".mkv": "mkv",
        ".webm": "mkv",
    }

    @staticmethod
    def _detect_container(local_path):
        """Detect the actual container family from magic bytes, independent of the
        file extension. Returns 'avi' | 'mp4' | 'mkv' | None (None = unknown)."""
        try:
            with open(local_path, "rb") as f:
                header = f.read(12)
        except OSError:
            return None
        if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"AVI ":
            return "avi"
        if len(header) >= 8 and header[4:8] == b"ftyp":
            return "mp4"
        if header[:4] == b"\x1a\x45\xdf\xa3":
            return "mkv"
        return None

    @classmethod
    def _classify_video(cls, local_path):
        """Decide the decode route from declared extension vs detected content.

        - mismatch (extension implies X but content is Y): reject as a client
          mislabel (400) -- the caller declared the wrong format.
        - genuine AVI content: route to decord (pynv can hard-crash on AVI).
        - otherwise (consistent, non-AVI): pynv preferred, decord fallback.

        Returns (use_decord, actual_container, mismatch)."""
        ext = os.path.splitext(local_path)[1].lower()
        expected = cls._EXT_TO_CONTAINER.get(ext)
        actual = cls._detect_container(local_path)
        mismatch = bool(expected and actual and expected != actual)
        use_decord = mismatch or actual == "avi"
        return use_decord, actual, mismatch

    @staticmethod
    def _probe_video_valid(local_path):
        """Cheap, decode-free sanity check of the container/stream headers via
        PyAV. Catches structurally-invalid files (no video stream, zero
        dimensions, zero/absent fps) that would otherwise trigger a native
        divide-by-zero (SIGFPE) deep inside the decoder. Returns None if the
        file looks decodable, else a short reason string.

        Best-effort FILTER, not a guarantee: PyAV(libav) and NVDEC are different
        parsers, so a file PyAV accepts may still crash NVDEC. If PyAV is
        unavailable we skip the probe rather than block decoding."""
        try:
            import av
        except ImportError:
            return None
        try:
            with av.open(local_path) as container:
                vstreams = [s for s in container.streams if s.type == "video"]
                if not vstreams:
                    return "no video stream"
                s = vstreams[0]
                cc = s.codec_context
                width = getattr(cc, "width", None)
                height = getattr(cc, "height", None)
                if not width or not height:
                    return f"invalid dimensions {width}x{height}"
                rate = s.average_rate or s.base_rate
                if rate is None or float(rate) == 0.0:
                    return "zero/invalid fps"
        except Exception as e:
            return f"container unparsable: {e}"
        return None

    def _reject_if_invalid_video(self, local_path, req=None):
        """Run the decode-free probe and raise 422 if the content is invalid."""
        reason = self._probe_video_valid(local_path)
        if reason is not None:
            rid = getattr(req, "rid", None)
            logger.error(
                f"[DecodePath] probe rejected video: rid={rid} "
                f"reason={reason} path={local_path}"
            )
            raise VideoDecodeError(f"video content invalid: {reason}", code=422)

    def _iter_decode_video_chunks(self, local_path, req):
        if req.force_decord:
            yield from self._iter_decode_decord_chunks(local_path, req, force_cpu=True)
            return
        # Route by declared-vs-detected format:
        #   1. mismatch (declared ext != detected content) -> reject (client mislabel, 400)
        #   2. genuine AVI content -> decord (pynv can hard-crash on AVI)
        #   3. otherwise -> pynv preferred, decord fallback
        _use_decord, actual, mismatch = self._classify_video(local_path)
        if mismatch:
            logger.error(
                f"[DecodePath] format mismatch: rid={req.rid} "
                f"declared ext != detected={actual} path={local_path}"
            )
            raise VideoDecodeError(
                f"declared file extension does not match content (detected={actual})",
                code=400,
            )
        if actual == "avi":
            try:
                yield from self._iter_decode_decord_chunks(local_path, req)
            except Exception as e:
                logger.error(
                    f"[DecodePath] decord failed (avi): rid={req.rid} "
                    f"err={e} path={local_path}",
                    exc_info=True,
                )
                raise VideoDecodeError(
                    f"video decode failed (decord: {e})", code=422
                ) from e
            return
        # Consistent, non-AVI: decode-free probe guards pynv against SIGFPE.
        self._reject_if_invalid_video(local_path, req)
        # pynv preferred, decord fallback, both fail -> 422.
        if self._pynv_available and self._decode_backend in ("auto", "pynvvideocodec"):
            emitted = 0
            try:
                for item in self._iter_decode_pynv_chunks(local_path, req):
                    emitted += 1
                    yield item
                return
            except Exception as e:
                if isinstance(e, (MemoryError, torch.cuda.OutOfMemoryError)):
                    raise
                logger.warning(
                    "NVDEC attempt failed: rid=%s emitted=%d path=%s error=%s",
                    req.rid,
                    emitted,
                    local_path,
                    e,
                )
                raise RetryableVideoDecodeError(
                    f"NVDEC failed after {emitted} chunks: {e}"
                ) from e
        try:
            yield from self._iter_decode_decord_chunks(local_path, req)
        except Exception as e:
            logger.error(
                f"[DecodePath] decord failed (direct): rid={req.rid} "
                f"err={e} path={local_path}",
                exc_info=True,
            )
            raise VideoDecodeError(
                f"video decode failed (decord: {e})", code=422
            ) from e

    def _decode_video(self, local_path, req):
        """Decode and return raw frame chunks without preprocessing (batch)."""
        # Route by declared-vs-detected format:
        #   1. mismatch (declared ext != detected content) -> reject (client mislabel, 400)
        #   2. genuine AVI content -> decord (pynv can hard-crash on AVI)
        #   3. otherwise -> pynv preferred, decord fallback
        _use_decord, actual, mismatch = self._classify_video(local_path)
        if mismatch:
            logger.error(
                f"[DecodePath] format mismatch: rid={req.rid} "
                f"declared ext != detected={actual} path={local_path}"
            )
            raise VideoDecodeError(
                f"declared file extension does not match content (detected={actual})",
                code=400,
            )
        if actual == "avi":
            try:
                return self._decode_decord(local_path, req)
            except Exception as e:
                logger.error(
                    f"[DecodePath] USED=decord result=failed (avi): rid={req.rid} "
                    f"err={e} path={local_path}",
                    exc_info=True,
                )
                raise VideoDecodeError(
                    f"video decode failed (decord: {e})", code=422
                ) from e
        # Consistent, non-AVI: decode-free probe guards pynv against SIGFPE.
        self._reject_if_invalid_video(local_path, req)
        # pynv preferred, decord fallback, both fail -> 422.
        if self._pynv_available and self._decode_backend != "decord_gpu":
            try:
                return self._decode_pynv(local_path, req)
            except Exception as e:
                logger.warning(
                    f"VideoDecodeImpl: PyNvVideoCodec failed: {e}, falling back to decord"
                )
                try:
                    return self._decode_decord(local_path, req)
                except Exception as e2:
                    logger.error(
                        f"[DecodePath] USED=decord result=failed (pynv-fallback): rid={req.rid} "
                        f"err={e2} path={local_path}",
                        exc_info=True,
                    )
                    raise VideoDecodeError(
                        f"video decode failed (pynv: {e}; decord: {e2})", code=422
                    ) from e2
        try:
            return self._decode_decord(local_path, req)
        except Exception as e:
            logger.error(
                f"[DecodePath] USED=decord result=failed (direct): rid={req.rid} "
                f"err={e} path={local_path}",
                exc_info=True,
            )
            raise VideoDecodeError(
                f"video decode failed (decord: {e})", code=422
            ) from e

    def _decode_pynv(self, local_path, req):
        import PyNvVideoCodec as nvc

        dec = nvc.SimpleDecoder(
            local_path,
            gpu_id=self._gpu_index,
            output_color_type=nvc.OutputColorType.RGBP,
        )
        try:
            meta = dec.get_stream_metadata()
            height, width = meta.height, meta.width
            num_frames, avg_fps = meta.num_frames, meta.average_fps
            frame_inds = self._gpu_processor_cls.compute_frame_sampling(
                num_frames, avg_fps, req.sample_fps
            )

            t0 = time.monotonic()
            all_rgb = dec.get_batch_frames_by_index(frame_inds)
            frame_cost = time.monotonic() - t0

            # Batch convert: stack all as uint8 first, then single float cast
            all_float = torch.stack([torch.from_dlpack(f) for f in all_rgb]).float()
            del all_rgb

            chunks = []
            for i in range(0, len(frame_inds), self._tmks):
                chunk_end = min(i + self._tmks, len(frame_inds))
                # clone() forces independent storage. A first-dimension slice is already
                # contiguous, so contiguous() alone would keep the full all_float storage alive.
                chunks.append(all_float[i:chunk_end].clone())
            del all_float  # release the large full-video tensor immediately
        finally:
            del dec
        timestamps = self._compute_chunk_timestamps(frame_inds, avg_fps)
        return frame_cost, chunks, height, width, timestamps

    def _iter_decode_pynv_chunks(self, local_path, req):
        import PyNvVideoCodec as nvc

        # Each video gets its own CUDA stream so multiple concurrent decoders
        # don't serialize on the same stream — NVDEC supports ~3 parallel sessions.
        decode_stream = torch.cuda.Stream(device=self._device)
        dec = nvc.SimpleDecoder(
            local_path,
            gpu_id=self._gpu_index,
            cuda_stream=decode_stream.cuda_stream,
            output_color_type=nvc.OutputColorType.RGBP,
        )
        try:
            meta = dec.get_stream_metadata()
            height, width = meta.height, meta.width
            num_frames, avg_fps = meta.num_frames, meta.average_fps
            frame_inds = self._gpu_processor_cls.compute_frame_sampling(
                num_frames, avg_fps, req.sample_fps
            )
            timestamps = self._compute_chunk_timestamps(frame_inds, avg_fps)
            decode_stride = self._tmks * self._decode_chunks_per_batch
            _decode_sum = 0.0
            _nchunks = 0
            for batch_start in range(0, len(frame_inds), decode_stride):
                batch_end = min(batch_start + decode_stride, len(frame_inds))
                batch_frame_inds = frame_inds[batch_start:batch_end]
                t0 = time.monotonic()
                try:
                    batch_rgb = dec.get_batch_frames_by_index(batch_frame_inds)
                except Exception:
                    logger.warning(
                        "NVDEC frame batch failed: rid=%s nframes=%s fps=%s "
                        "sampled=%d batch_start=%d batch_end=%d indices=%s "
                        "last_index=%s emitted=%d path=%s",
                        req.rid,
                        num_frames,
                        avg_fps,
                        len(frame_inds),
                        batch_start,
                        batch_end,
                        list(batch_frame_inds[:16]),
                        batch_frame_inds[-1] if len(batch_frame_inds) else None,
                        _nchunks,
                        local_path,
                    )
                    raise
                # NVDEC decodes asynchronously on decode_stream, but the reads below
                # (from_dlpack / stack / .float() / .clone()) run on the current
                # (default) stream. Without an explicit dependency the default stream
                # can read the decoder surfaces before NVDEC has finished writing them
                # -> garbage -> NaN. wait_stream inserts a GPU-side dependency only
                # (the CPU is not blocked, cross-batch pipelining is preserved).
                torch.cuda.current_stream().wait_stream(decode_stream)
                dlpack_frames = [torch.from_dlpack(f) for f in batch_rgb]
                batch_tensor = torch.stack(dlpack_frames)
                batch_tensor = batch_tensor.float()
                frame_cost = time.monotonic() - t0
                _decode_sum += frame_cost

                # Materialize every per-chunk .clone() on the default stream first,
                # then block THIS decode worker thread (never the main asyncio loop)
                # until the whole default-stream copy pipeline has finished on the GPU.
                # This closes two more GPU-side hazards that wait_stream alone does not:
                #   1. NVDEC surface recycling: the next get_batch_frames_by_index()
                #      (on decode_stream) must not overwrite the decoder surface pool
                #      while stack/float on the default stream is still reading the
                #      current batch's frames.
                #   2. Cross-thread/stream handoff: chunk_tensor is produced here on
                #      the decode worker's default stream but consumed by the preprocess
                #      step on _preprocess_stream in the main thread, with no stream
                #      dependency between them.
                batch_chunks = []
                for local_start in range(0, len(batch_frame_inds), self._tmks):
                    local_end = min(local_start + self._tmks, len(batch_frame_inds))
                    chunk_idx = (batch_start + local_start) // self._tmks
                    chunk_tensor = batch_tensor[local_start:local_end].clone()
                    batch_chunks.append((chunk_idx, chunk_tensor))
                torch.cuda.current_stream().synchronize()
                del dlpack_frames, batch_tensor, batch_rgb
                for chunk_idx, chunk_tensor in batch_chunks:
                    _nchunks += 1
                    yield chunk_idx, chunk_tensor, height, width, timestamps[
                        chunk_idx
                    ], frame_cost
            _sz_bytes = os.path.getsize(local_path)
            logger.info(
                "[DecodeTiming] backend=pynv path=%s nframes=%d sampled=%d "
                "dur=%.1fs bytes=%d size_mb=%.2f chunks=%d decode_sum=%.3fs res=%dx%d",
                local_path,
                num_frames,
                len(frame_inds),
                (num_frames / avg_fps if avg_fps else 0.0),
                _sz_bytes,
                _sz_bytes / 1048576,
                _nchunks,
                _decode_sum,
                width,
                height,
            )
        finally:
            del dec

    def _decode_decord(self, local_path, req):
        from decord import VideoReader
        from decord import cpu as decord_cpu
        from decord import gpu as decord_gpu

        ctx = decord_gpu(0) if self._gpu_available else decord_cpu(0)
        ctx_kind = "gpu" if self._gpu_available else "cpu"
        try:
            vr = VideoReader(local_path, ctx=ctx, num_threads=4)
        except Exception:
            if self._gpu_available:
                vr = VideoReader(local_path, ctx=decord_cpu(0), num_threads=4)
                ctx_kind = "cpu"
            else:
                raise

        num_frames = len(vr)
        avg_fps = vr.get_avg_fps()

        frame_inds = self._gpu_processor_cls.compute_frame_sampling(
            num_frames, avg_fps, req.sample_fps
        )

        t0 = time.monotonic()
        all_frames = vr.get_batch(frame_inds).asnumpy()
        frame_cost = time.monotonic() - t0

        height, width = all_frames[0].shape[0], all_frames[0].shape[1]

        # Batch convert: single to() and permute on the whole tensor
        all_float = (
            torch.from_numpy(all_frames).to(self._device).permute(0, 3, 1, 2).float()
        )
        chunks = []
        for i in range(0, len(frame_inds), self._tmks):
            chunk_end = min(i + self._tmks, len(frame_inds))
            # clone() forces independent storage. A first-dimension slice is already
            # contiguous, so contiguous() alone would keep the full all_float storage alive.
            chunks.append(all_float[i:chunk_end].clone())
        del all_float  # release the large full-video tensor immediately

        del vr
        timestamps = self._compute_chunk_timestamps(frame_inds, avg_fps)
        _sz_bytes = os.path.getsize(local_path)
        logger.info(
            "[DecodeTiming] backend=decord ctx=%s path=%s nframes=%d sampled=%d "
            "dur=%.1fs bytes=%d size_mb=%.2f chunks=%d decode_sum=%.3fs res=%dx%d",
            ctx_kind,
            local_path,
            num_frames,
            len(frame_inds),
            (num_frames / avg_fps if avg_fps else 0.0),
            _sz_bytes,
            _sz_bytes / 1048576,
            len(chunks),
            frame_cost,
            width,
            height,
        )
        return frame_cost, chunks, height, width, timestamps

    def _iter_decode_decord_chunks(self, local_path, req, force_cpu: bool = False):
        from decord import VideoReader
        from decord import cpu as decord_cpu
        from decord import gpu as decord_gpu

        use_gpu = self._gpu_available and not force_cpu
        ctx = decord_gpu(0) if use_gpu else decord_cpu(0)
        ctx_kind = "gpu" if use_gpu else "cpu"
        try:
            vr = VideoReader(local_path, ctx=ctx, num_threads=4)
        except Exception:
            if use_gpu:
                vr = VideoReader(local_path, ctx=decord_cpu(0), num_threads=4)
                ctx_kind = "cpu"
            else:
                raise

        try:
            num_frames = len(vr)
            avg_fps = vr.get_avg_fps()
            frame_inds = self._gpu_processor_cls.compute_frame_sampling(
                num_frames, avg_fps, req.sample_fps
            )
            timestamps = self._compute_chunk_timestamps(frame_inds, avg_fps)
            # Rec 3: CPU decord fallback guardrail. A malformed bitstream can make
            # pynv raise, then the CPU decord fallback decode a huge video and tie
            # up a worker for a very long time (or hang). On the CPU path only,
            # fail-fast videos whose sampled-frame count or duration exceeds a
            # configured cap BEFORE entering the per-chunk get_batch loop, so a
            # single oversized video cannot monopolize a decode worker. Caps of 0
            # disable the check (default). Rec 1's per-job watchdog still covers a
            # true native hang; this is the cheaper, worker-preserving fast-fail.
            if not use_gpu:
                _cpu_max_frames = int(
                    os.environ.get("SGLANG_VIDEO_DECODE_CPU_MAX_FRAMES", "0")
                )
                _cpu_max_dur = float(
                    os.environ.get("SGLANG_VIDEO_DECODE_CPU_MAX_DURATION_S", "0")
                )
                _dur = (num_frames / avg_fps) if avg_fps else 0.0
                if _cpu_max_frames > 0 and len(frame_inds) > _cpu_max_frames:
                    logger.warning(
                        "[DecodePath] CPU decord guardrail tripped: rid=%s "
                        "sampled_frames=%d > cap=%d path=%s (fast-fail 422)",
                        getattr(req, "rid", None),
                        len(frame_inds),
                        _cpu_max_frames,
                        local_path,
                    )
                    raise VideoDecodeError(
                        f"CPU decord guardrail: sampled frames {len(frame_inds)} "
                        f"> {_cpu_max_frames}",
                        code=422,
                    )
                if _cpu_max_dur > 0 and _dur > _cpu_max_dur:
                    logger.warning(
                        "[DecodePath] CPU decord guardrail tripped: rid=%s "
                        "duration=%.1fs > cap=%.0fs path=%s (fast-fail 422)",
                        getattr(req, "rid", None),
                        _dur,
                        _cpu_max_dur,
                        local_path,
                    )
                    raise VideoDecodeError(
                        f"CPU decord guardrail: duration {_dur:.1f}s "
                        f"> {_cpu_max_dur:.0f}s",
                        code=422,
                    )
            _decode_sum = 0.0
            _nchunks = 0
            _res_h = _res_w = 0
            for chunk_idx, i in enumerate(range(0, len(frame_inds), self._tmks)):
                chunk_end = min(i + self._tmks, len(frame_inds))
                chunk_frame_inds = frame_inds[i:chunk_end]
                t0 = time.monotonic()
                chunk_frames = vr.get_batch(chunk_frame_inds).asnumpy()
                frame_cost = time.monotonic() - t0
                _decode_sum += frame_cost
                chunk_tensor = (
                    torch.from_numpy(chunk_frames)
                    .to(self._device)
                    .permute(0, 3, 1, 2)
                    .float()
                )
                del chunk_frames
                _res_h, _res_w = chunk_tensor.shape[-2], chunk_tensor.shape[-1]
                _nchunks += 1
                yield (
                    chunk_idx,
                    chunk_tensor,
                    chunk_tensor.shape[-2],
                    chunk_tensor.shape[-1],
                    timestamps[chunk_idx],
                    frame_cost,
                )
            _sz_bytes = os.path.getsize(local_path)
            logger.info(
                "[DecodeTiming] backend=decord ctx=%s path=%s nframes=%d sampled=%d "
                "dur=%.1fs bytes=%d size_mb=%.2f chunks=%d decode_sum=%.3fs res=%dx%d",
                ctx_kind,
                local_path,
                num_frames,
                len(frame_inds),
                (num_frames / avg_fps if avg_fps else 0.0),
                _sz_bytes,
                _sz_bytes / 1048576,
                _nchunks,
                _decode_sum,
                _res_w,
                _res_h,
            )
        finally:
            del vr

    def _compute_chunk_timestamps(self, frame_inds, avg_fps):
        """Compute per-chunk prompts via model-specific processor."""
        return self._gpu_processor_cls.compute_chunk_timestamps(
            frame_inds, avg_fps, self._tmks, self._timestamp_mode
        )
