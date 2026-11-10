"""DeepSeek-V4.1 encoder throughput paths: ViT varlen batching and preprocessing.

``ViT.forward_batch`` packs several images into one varlen pass; it must equal
one ``forward`` per image because attention stays inside each cu_seqlens
segment. The checks run on CPU with a reference attention that honours
cu_seqlens the way the varlen backends do.
"""

import types
from unittest import mock

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from sglang.srt.models import deepseek_v41_vit as vit_module
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM, _vit_patch_groups
from sglang.srt.multimodal.deepseek_v41_image_processing import (
    load_image,
    materialize_image_gpu,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

ARGS = types.SimpleNamespace(
    vision_dim=32,
    vision_n_heads=4,
    vision_inter_dim=48,
    vision_n_layers=2,
    vision_patch_size=2,
    vision_rope_theta=10000.0,
)


class _VarlenBackend:
    """Marker standing in for a backend that honours cu_seqlens."""


class _ReferenceAttention(nn.Module):
    """Bidirectional attention bounded by ``metadata.cu_seqlens``."""

    def __init__(self, args):
        super().__init__()
        self.heads = args.vision_n_heads
        self.qkv = nn.Linear(args.vision_dim, 3 * args.vision_dim)
        self.proj = nn.Linear(args.vision_dim, args.vision_dim)
        self.qkv_backend = _VarlenBackend()

    def forward(self, x, cos, sin, metadata):
        q, k, v = self.qkv(x).view(x.shape[0], 3, self.heads, -1).unbind(1)
        q, k = vit_module.apply_rotary(q, cos, sin), vit_module.apply_rotary(k, cos, sin)
        bounds = metadata.cu_seqlens.tolist()
        out = [
            F.scaled_dot_product_attention(
                *(t[a:b].transpose(0, 1) for t in (q, k, v))
            ).transpose(0, 1)
            for a, b in zip(bounds[:-1], bounds[1:])
        ]
        return self.proj(torch.cat(out).reshape(x.shape[0], -1))


def _vit():
    torch.manual_seed(0)
    with mock.patch.object(vit_module, "Attention", _ReferenceAttention):
        return vit_module.ViT(ARGS).eval()


def _patches(grids):
    p = ARGS.vision_patch_size
    return [torch.randn(h * w, 3, p, p) for h, w in grids]


def test_forward_batch_matches_one_forward_per_image():
    model = _vit()
    grids = [(3, 4), (2, 2), (5, 1)]
    patches = _patches(grids)
    with mock.patch.object(vit_module, "_VARLEN_BACKENDS", (_VarlenBackend,)):
        with mock.patch.object(model, "forward", wraps=model.forward) as forward:
            batched = model.forward_batch(patches, grids)
            forward.assert_not_called()
    for out, p, (h, w) in zip(batched, patches, grids):
        assert out.shape == (h * w, ARGS.vision_dim)
        torch.testing.assert_close(out, model.forward(p, h, w), rtol=1e-5, atol=1e-5)


def test_forward_batch_does_not_pack_for_backends_without_varlen():
    """sdpa-style backends ignore cu_seqlens, so packing would mix images."""
    model = _vit()
    grids = [(2, 3), (3, 2)]
    patches = _patches(grids)
    with mock.patch.object(vit_module, "_VARLEN_BACKENDS", ()):
        with mock.patch.object(model, "forward", wraps=model.forward) as forward:
            outs = model.forward_batch(patches, grids)
    assert forward.call_count == 2
    for out, p, (h, w) in zip(outs, patches, grids):
        torch.testing.assert_close(out, model.forward(p, h, w))


@pytest.mark.parametrize(
    "rows,budget,expected",
    [
        ([4, 4, 4], 0, [[4], [4], [4]]),
        ([4, 4, 4], 8, [[4, 4], [4]]),
        ([4, 9, 2, 2], 8, [[4], [9], [2, 2]]),
        ([3, 3, 3], 100, [[3, 3, 3]]),
    ],
)
def test_vit_patch_groups_keep_order_and_budget(rows, budget, expected):
    prepared = [(torch.empty(n, 1), n, 1) for n in rows]
    groups = _vit_patch_groups(prepared, budget)
    assert [[entry[0].shape[0] for entry in group] for group in groups] == expected
    assert [entry for group in groups for entry in group] == prepared


V41_ARGS = types.SimpleNamespace(
    vision_patch_size=14,
    vision_downsample_ratio=3,
    vision_max_n_token=1024,
    vision_min_pixels=0,
    vision_max_wh_ratio=None,
    hidden_size=8,
)


def _image(width=100, height=60):
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (height, width, 3), dtype=np.uint8))


def _encoder_model(backend):
    model = types.SimpleNamespace(config=V41_ARGS)
    loader = load_image if backend != "gpu" else None
    model._encoder_image_backend = lambda: (backend, loader)
    model.preprocess_mm_for_encoder = (
        DeepseekV4ForCausalLM.preprocess_mm_for_encoder.__get__(model)
    )
    return model


def test_gpu_backend_defers_resize_to_the_device_and_matches_pil():
    from sglang.srt.managers.schedule_batch import Modality

    image = _image()
    gpu = _encoder_model("gpu").preprocess_mm_for_encoder([image], Modality.IMAGE, {})
    pil = _encoder_model("pil").preprocess_mm_for_encoder([image], Modality.IMAGE, {})
    gpu_item, pil_item = gpu.mm_items[0], pil.mm_items[0]

    plan = gpu_item.model_specific_data["dsv41_gpu_plan"]
    assert gpu_item.feature.dtype == torch.uint8 and gpu_item.feature.shape[0] == 3
    assert "dsv41_gpu_plan" not in pil_item.model_specific_data
    for key in ("n_vit_h", "n_vit_w"):
        assert gpu_item.model_specific_data[key] == pil_item.model_specific_data[key]
    assert gpu.item_sizes == pil.item_sizes
    assert gpu_item.hash == pil_item.hash
    patches = materialize_image_gpu(gpu_item.feature, plan).to(torch.bfloat16)
    assert patches.shape == pil_item.feature.shape
    # Same pixels within bf16 rounding of the separable bicubic passes.
    diff = (patches.float() - pil_item.feature.float()).abs()
    assert diff.mean() < 0.02


@pytest.mark.parametrize(
    "backend,disable_fast,use_gpu,rust_ok,expected",
    [
        ("pil", False, True, True, "pil"),
        ("auto", True, True, True, "pil"),
        ("auto", False, True, True, "gpu"),
        ("auto", False, False, True, "rust"),
        ("auto", False, False, False, "pil"),
    ],
)
def test_encoder_image_backend_follows_the_local_processor_rules(
    backend, disable_fast, use_gpu, rust_ok, expected
):
    import sglang.srt.models.deepseek_v4 as v4

    model = types.SimpleNamespace()
    extension = types.SimpleNamespace(
        dsv41=types.SimpleNamespace(resize_patchify=lambda *a: None)
    )

    def load(_name):
        if not rust_ok:
            raise ImportError("no rust extension")
        return extension

    with (
        mock.patch.object(
            v4,
            "get_mm",
            return_value=types.SimpleNamespace(
                image_processor_backend=backend,
                disable_fast_image_processor=disable_fast,
            ),
        ),
        mock.patch.object(
            v4.envs.SGLANG_ENCODER_IMAGE_PROCESSOR_USE_GPU, "get", return_value=use_gpu
        ),
        mock.patch("sglang.srt.rust_extensions.load_rust_extension", side_effect=load),
    ):
        chosen, loader = DeepseekV4ForCausalLM._encoder_image_backend(model)
        # Resolved once per model.
        assert DeepseekV4ForCausalLM._encoder_image_backend(model)[0] == chosen
    assert chosen == expected
    assert (loader is load_image) == (expected in ("pil", "gpu"))


def test_get_image_feature_batches_the_vit_by_patch_budget():
    import sglang.srt.models.deepseek_v4 as v4

    hidden = V41_ARGS.hidden_size
    calls = []

    def forward_batch(patches, grids):
        calls.append(list(grids))
        return [torch.ones(h * w, 4) * (h * 10 + w) for h, w in grids]

    def aligner(rows, h, w):
        r = V41_ARGS.vision_downsample_ratio
        n = ((h + r - 1) // r) * ((w + r - 1) // r)
        return rows[:1].expand(n, 4).repeat(1, hidden // 4)

    model = types.SimpleNamespace(
        image_start=torch.zeros(hidden),
        image_newline=torch.full((hidden,), -1.0),
        image_end=torch.full((hidden,), -2.0),
        tp_size=1,
        config=V41_ARGS,
        vision=types.SimpleNamespace(forward_batch=forward_batch),
        aligner=aligner,
    )
    grids = [(3, 3), (6, 3), (3, 6)]
    items = [
        types.SimpleNamespace(
            reconstruct=lambda *a, **k: None,
            n_vit_h=h,
            n_vit_w=w,
            feature=torch.zeros(h * w, 3, 14, 14),
            model_specific_data={},
        )
        for h, w in grids
    ]
    with mock.patch.object(
        v4.envs.SGLANG_DSV41_VIT_BATCH_MAX_PATCHES, "get", return_value=27
    ):
        spans = DeepseekV4ForCausalLM.get_image_feature(model, items)
    assert calls == [[(3, 3), (6, 3)], [(3, 6)]]
    for span, (h, w) in zip(spans, grids):
        r = V41_ARGS.vision_downsample_ratio
        lh, lw = (h + r - 1) // r, (w + r - 1) // r
        assert span.shape == (lh * lw + lh + 2, hidden)
        # Each image's span carries its own ViT rows, in order.
        assert float(span[1, 0]) == h * 10 + w
