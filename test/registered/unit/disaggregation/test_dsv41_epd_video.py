"""DeepSeek-V4.1 EPD video: frame-sampled video over the K3 video protocol.

V4.1's vision tower only takes images, so the encoder samples frames and ships
one task_result per video whose embedding concatenates one image span per
frame, with one grid and one ``MM:SS`` chunk timestamp per frame. Prefill and
decode rebuild one placeholder per frame from the same task_result.
"""

import os
import tempfile
import types
from unittest import mock

import numpy as np
import pytest

from sglang.srt.disaggregation.encoder import video_frames as vf
from sglang.srt.entrypoints.openai.serving_chat import (
    _expand_dsv41_video_parts,
    _video_frames_as_images,
)


def test_frame_sampling_matches_k3_rule_and_cap():
    # 10 s at 30 fps sampled at 1 fps -> 10 frames, uniformly spread.
    idx = vf.compute_frame_sampling(300, 30.0, 1.0)
    assert len(idx) == 10 and idx[0] == 0 and idx[-1] == 299
    # sample_fps above the source fps is clamped to the source fps.
    assert len(vf.compute_frame_sampling(20, 10.0, 50.0)) == 20
    capped = vf.cap_frame_indices(idx, 4)
    assert len(capped) == 4 and capped[0] == 0 and capped[-1] == 299
    assert vf.cap_frame_indices(idx, 0) == idx


def test_timestamps_and_frame_fit():
    assert vf.format_timestamp(0.4) == "00:00"
    assert vf.format_timestamp(75.9) == "01:15"
    from PIL import Image

    big = Image.new("RGB", (1920, 1080))
    small = vf.fit_frame(big, 295936)
    assert small.width * small.height <= 295936
    assert abs(small.width / small.height - 1920 / 1080) < 0.01
    tiny = Image.new("RGB", (100, 50))
    assert vf.fit_frame(tiny, 295936) is tiny  # never upscaled here


def _write_clip(d):
    cv2 = pytest.importorskip("cv2")
    path = os.path.join(d, "clip.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(40):  # 4 s at 10 fps; brightness encodes the second
        writer.write(np.full((48, 64, 3), (i // 10) * 60, dtype=np.uint8))
    writer.release()
    return path


def _assert_four_ordered_frames(frames, stamps):
    assert len(frames) == 4 and stamps == ["00:00", "00:01", "00:02", "00:03"]
    means = [np.asarray(f).mean() for f in frames]
    assert means == sorted(means)  # time order preserved


def test_decode_video_frames_from_a_real_file():
    pytest.importorskip("decord")
    with tempfile.TemporaryDirectory() as d:
        frames, stamps = vf.decode_video_frames(_write_clip(d), 1.0, 32, 295936)
    _assert_four_ordered_frames(frames, stamps)


def test_gpu_decode_falls_back_to_decord_when_nvdec_fails():
    """K3 order: NVDEC preferred, decord when NVDEC is unavailable or fails."""
    pytest.importorskip("decord")

    def broken(*args, **kwargs):
        raise RuntimeError("Failed to load NVENC library")

    with tempfile.TemporaryDirectory() as d, mock.patch.object(
        vf, "_decode_with_nvdec", side_effect=broken
    ) as nvdec:
        frames, stamps = vf.decode_video_frames(
            _write_clip(d), 1.0, 32, 295936, use_gpu=True
        )
    assert nvdec.called
    _assert_four_ordered_frames(frames, stamps)


def _video_result(task_id, grids, stamps):
    return {
        "task_id": task_id,
        "modality": "video",
        "encoder_addr": {
            "task_id": task_id,
            "modality": "video",
            "img_grid_thw": grids,
            "chunk_timestamps": stamps,
            "embedding_len": 0,
        },
    }


def test_video_part_becomes_per_frame_timestamp_and_image_parts():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what happens?"},
                {"type": "image_url", "image_url": {"url": "img"}},
                {"type": "video_url", "video_url": {"url": "vid"}},
            ],
        }
    ]
    results = [
        {"task_id": "i", "encoder_addr": {"modality": "image"}},
        _video_result("v", [[1, 39, 39], [1, 39, 39]], ["00:00", "00:01"]),
    ]
    _expand_dsv41_video_parts(messages, results)
    content = messages[0]["content"]
    assert [p["type"] for p in content] == [
        "text", "image_url", "text", "image_url", "text", "image_url",
    ]
    assert content[2]["text"] == "00:00" and content[5]["image_url"]["url"] == "vid"

    retagged = _video_frames_as_images(results)
    assert retagged[1]["encoder_addr"]["modality"] == "image"
    assert results[1]["encoder_addr"]["modality"] == "video"  # not mutated


def test_video_without_timestamps_is_rejected():
    messages = [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": "v"}}]}]
    with pytest.raises(ValueError):
        _expand_dsv41_video_parts(messages, [])
    with pytest.raises(ValueError):
        _expand_dsv41_video_parts(
            [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": "v"}}]}],
            [_video_result("v", [[1, 39, 39]], ["00:00", "00:01"])],
        )


def test_span_lengths_match_v41_image_layout():
    from sglang.srt.multimodal.processors.deepseek_v41 import (
        DeepseekV41ImageProcessor as Proc,
    )

    proc = object.__new__(Proc)
    proc.hf_config = types.SimpleNamespace(vision_downsample_ratio=3)
    # n_llm = ceil(39/3) = 13 -> 13*(13+1)+2 = 184 tokens per frame.
    assert proc.span_lengths([[1, 39, 39], [1, 39, 39]]) == [184, 184]


def _media_request(parts, sample_fps=None):
    return types.SimpleNamespace(
        sample_fps=sample_fps,
        messages=[types.SimpleNamespace(content=parts)],
    )


def _result(task_id, modality):
    return {
        "task_id": task_id,
        "model_id": "m",
        "encoder_addr": {"model_id": "m", "modality": modality},
    }


def test_mixed_image_video_without_sample_fps_keeps_images_in_place():
    """No sample_fps: video ids cannot be recomputed, images still are."""
    from sglang.srt.entrypoints.openai.serving_chat import (
        _reorder_task_results_by_content,
    )
    from sglang.srt.multimodal.task_id import compute_media_task_id

    a, b = (compute_media_task_id("m", url) for url in ("http://a", "http://b"))
    request = _media_request(
        [
            {"type": "image_url", "image_url": {"url": "http://a"}},
            {"type": "video_url", "video_url": {"url": "http://v"}},
            {"type": "image_url", "image_url": {"url": "http://b"}},
        ]
    )
    # IC order differs from the prompt order.
    results = [_result(b, "image"), _result("vid", "video"), _result(a, "image")]
    ordered = _reorder_task_results_by_content(request, results)
    assert [r["task_id"] for r in ordered] == [a, "vid", b]


def test_reorder_without_videos_is_unchanged():
    from sglang.srt.entrypoints.openai.serving_chat import (
        _reorder_task_results_by_content,
    )
    from sglang.srt.multimodal.task_id import compute_media_task_id

    a, b = (compute_media_task_id("m", url) for url in ("http://a", "http://b"))
    request = _media_request(
        [
            {"type": "image_url", "image_url": {"url": "http://a"}},
            {"type": "image_url", "image_url": {"url": "http://b"}},
        ]
    )
    ordered = _reorder_task_results_by_content(
        request, [_result(b, "image"), _result(a, "image")]
    )
    assert [r["task_id"] for r in ordered] == [a, b]
