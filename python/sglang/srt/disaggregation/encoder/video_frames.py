"""Video -> sampled frames for encoders whose vision tower only takes images.

The factory (厂内) EPD video protocol follows Kimi-K3: one vit-server video task
yields one TaskResult whose embedding concatenates the per-chunk spans, with
``img_grid_thw`` holding one grid per chunk and ``chunk_timestamps`` one start
time per chunk. Models without a temporal vision path (DeepSeek-V4.1) encode
each sampled frame as an image, so a chunk is a single frame here.
"""

import io
import logging
import math
import os
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT_S = 60
# Reject a remote video larger than this (0 disables); an unbounded download
# would read the whole file into host memory and can OOM the encoder.
_MAX_VIDEO_BYTES = int(os.environ.get("SGLANG_ENCODER_VIDEO_MAX_BYTES", 512 << 20))


def compute_frame_sampling(
    num_frames: int, avg_fps: float, sample_fps: float
) -> List[int]:
    """Uniform frame indices at ``min(sample_fps, avg_fps)`` (same rule as K3)."""
    effective_fps = min(sample_fps, avg_fps)
    sampled = max(round(num_frames * effective_fps / avg_fps), 1)
    return np.linspace(0, num_frames - 1, sampled).round().astype(int).tolist()


def cap_frame_indices(frame_indices: List[int], max_frames: int) -> List[int]:
    """Keep at most ``max_frames`` indices, uniformly spread over the clip."""
    if max_frames <= 0 or len(frame_indices) <= max_frames:
        return frame_indices
    picks = np.linspace(0, len(frame_indices) - 1, max_frames).round().astype(int)
    return [frame_indices[i] for i in picks]


def format_timestamp(seconds: float) -> str:
    """``MM:SS`` start time of a frame, K3's default ``mm:ss`` mode."""
    total = int(seconds)
    return f"{total // 60:02d}:{total % 60:02d}"


def fit_frame(image: Image.Image, max_pixels: int) -> Image.Image:
    """Downscale (never upscale) so the frame has at most ``max_pixels`` pixels.

    A video frame tokenizes like an image, so a large frame would cost up to the
    model's per-image token cap on every frame.
    """
    if max_pixels <= 0 or image.width * image.height <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / float(image.width * image.height))
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(size, Image.BICUBIC)


def _read_video_bytes(source) -> bytes:
    """Bytes of a video given as bytes, an http(s)/file URL, or a local path."""
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if not isinstance(source, str):
        raise ValueError(f"unsupported video source type: {type(source).__name__}")
    if source.startswith(("http://", "https://")):
        import requests

        with requests.get(source, timeout=_HTTP_TIMEOUT_S, stream=True) as response:
            response.raise_for_status()
            try:
                declared = int(response.headers.get("Content-Length") or 0)
            except ValueError:
                declared = 0  # malformed header; the running total still caps it
            if _MAX_VIDEO_BYTES and declared > _MAX_VIDEO_BYTES:
                raise ValueError(
                    f"video too large: {declared} bytes > {_MAX_VIDEO_BYTES}"
                )
            chunks, total = [], 0
            for chunk in response.iter_content(chunk_size=1 << 20):
                total += len(chunk)
                if _MAX_VIDEO_BYTES and total > _MAX_VIDEO_BYTES:
                    # A missing/lying Content-Length must not let the download
                    # grow past the cap.
                    raise ValueError(
                        f"video exceeds {_MAX_VIDEO_BYTES} bytes while downloading"
                    )
                chunks.append(chunk)
            return b"".join(chunks)
    path = source[len("file://") :] if source.startswith("file://") else source
    if not os.path.exists(path):
        raise ValueError(f"video not found: {path}")
    size = os.path.getsize(path)
    if _MAX_VIDEO_BYTES and size > _MAX_VIDEO_BYTES:
        raise ValueError(f"video too large: {size} bytes > {_MAX_VIDEO_BYTES}")
    with open(path, "rb") as f:
        return f.read()


def _sampled_indices(num_frames, avg_fps, sample_fps, max_frames) -> List[int]:
    """K3 sampling rule, then the frame cap; logs when the cap bites."""
    indices = compute_frame_sampling(num_frames, avg_fps, sample_fps)
    capped = cap_frame_indices(indices, max_frames)
    if len(capped) < len(indices):
        logger.info(
            "video frames capped %d -> %d (max_frames=%d, sample_fps=%s)",
            len(indices),
            len(capped),
            max_frames,
            sample_fps,
        )
    return capped


def _decode_with_nvdec(data: bytes, sample_fps, max_frames, gpu_id: int):
    """NVDEC decode through PyNvVideoCodec, as K3's VideoDecodeImpl does.

    Returns ``(rgb_frames, indices, avg_fps)`` with HWC uint8 numpy frames.
    Needs the "video" driver capability (NVIDIA_DRIVER_CAPABILITIES) in the
    container; the caller falls back to decord when it is missing.
    """
    import tempfile

    import PyNvVideoCodec as nvc
    import torch

    # SimpleDecoder reads from a path.
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        f.write(data)
        f.flush()
        decoder = nvc.SimpleDecoder(
            f.name, gpu_id=gpu_id, output_color_type=nvc.OutputColorType.RGB
        )
        try:
            meta = decoder.get_stream_metadata()
            if not meta.num_frames:
                raise ValueError("video has no frames")
            avg_fps = float(meta.average_fps or 0.0) or 1.0
            indices = _sampled_indices(meta.num_frames, avg_fps, sample_fps, max_frames)
            decoded = decoder.get_batch_frames_by_index(indices)
            # Copy off the decoder's surfaces before they are recycled.
            frames = [torch.from_dlpack(frame).cpu().numpy() for frame in decoded]
        finally:
            del decoder
    return frames, indices, avg_fps


def _decode_with_decord(data: bytes, sample_fps, max_frames):
    """CPU decode through decord; same return shape as ``_decode_with_nvdec``."""
    from decord import VideoReader, cpu

    reader = VideoReader(io.BytesIO(data), ctx=cpu(0))
    num_frames = len(reader)
    if num_frames == 0:
        raise ValueError("video has no frames")
    avg_fps = float(reader.get_avg_fps() or 0.0) or 1.0
    indices = _sampled_indices(num_frames, avg_fps, sample_fps, max_frames)
    return list(reader.get_batch(indices).asnumpy()), indices, avg_fps


def decode_video_frames(
    source,
    sample_fps: float,
    max_frames: int,
    max_pixels: int,
    use_gpu: bool = False,
    gpu_id: int = 0,
) -> Tuple[List[Image.Image], List[str]]:
    """Decode ``source`` and return ``(frames, timestamps)`` in time order.

    ``use_gpu`` prefers NVDEC and falls back to decord on any failure, the
    same order K3 uses (pynv preferred, decord fallback).
    """
    data = _read_video_bytes(source)
    decoded = None
    if use_gpu:
        try:
            decoded = _decode_with_nvdec(data, sample_fps, max_frames, gpu_id)
        except Exception as e:
            logger.warning("NVDEC video decode failed (%s); falling back to decord", e)
    if decoded is None:
        decoded = _decode_with_decord(data, sample_fps, max_frames)
    rgb_frames, indices, avg_fps = decoded
    frames = [fit_frame(Image.fromarray(frame), max_pixels) for frame in rgb_frames]
    timestamps = [format_timestamp(i / avg_fps) for i in indices]
    return frames, timestamps


def resolve_sample_fps(sample_fps: Optional[float], default_fps: float) -> float:
    """Task-provided fps when positive, else the encoder default."""
    if sample_fps is not None and float(sample_fps) > 0:
        return float(sample_fps)
    return default_fps
