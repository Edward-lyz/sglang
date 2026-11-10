"""Multimodal processor for Kimi K25 model."""

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sglang.kernels.ops.mm.process import normalize_and_patchify
from sglang.srt.disaggregation.encode_utils import (
    expand_media_placeholders_with_offsets,
    num_tokens_from_grid,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.kimi_k25 import KimiK25ForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import (
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.kimi_common import KimiGridMMDataMixin
from sglang.srt.utils.cuda_ipc_transport_utils import (
    DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY,
)


@dataclass
class KimiK25GPUPreprocessConfig:
    """Pre-computed GPU preprocessing config for the encode server."""

    patch_size: int
    merge_kernel_size: int
    in_patch_limit: int
    patch_limit_on_one_side: int
    fixed_output_tokens: int | None
    image_mean: torch.Tensor  # (1, 3, 1, 1) on GPU
    image_std_inv: torch.Tensor  # (1, 3, 1, 1) on GPU


@dataclass
class KimiK25VideoPreprocessConfig:
    """Pre-computed GPU video preprocessing config."""

    patch_size: int
    merge_kernel_size: int
    temporal_merge_kernel_size: int
    in_patch_limit_each_frame: int
    patch_limit_on_one_side: int
    sample_fps: float
    in_patch_limit_video: int | None
    max_num_frames_each_video: int | None
    fixed_output_tokens_each_frame: int | None
    image_mean: torch.Tensor  # (1, 3, 1, 1) on GPU
    image_std_inv: torch.Tensor  # (1, 3, 1, 1) on GPU


def _navit_resize_image(
    width: int,
    height: int,
    patch_size: int,
    merge_kernel_size: int,
    in_patch_limit: int,
    patch_limit_on_one_side: int,
    fixed_output_tokens: int | None,
):
    """Compute resize/pad dimensions for NaViT-style image patching."""
    s1 = math.sqrt(
        in_patch_limit
        / (max(1.0, width // patch_size) * max(1.0, height // patch_size))
    )
    s2 = patch_limit_on_one_side * patch_size / width
    s3 = patch_limit_on_one_side * patch_size / height
    scale = min(1.0, s1, s2, s3)
    new_w, new_h = max(1, int(width * scale)), max(1, int(height * scale))
    new_w = min(new_w, patch_limit_on_one_side * patch_size)
    new_h = min(new_h, patch_limit_on_one_side * patch_size)

    factor = merge_kernel_size * patch_size
    pad_height = (factor - new_h % factor) % factor
    pad_width = (factor - new_w % factor) % factor

    if fixed_output_tokens is not None:
        num_tokens = fixed_output_tokens
    else:
        token_height = (new_h + pad_height) // factor
        token_width = (new_w + pad_width) // factor
        num_tokens = token_height * token_width

    return {
        "num_tokens": num_tokens,
        "new_width": new_w,
        "new_height": new_h,
        "pad_width": pad_width,
        "pad_height": pad_height,
        "sampled_nframes": 1,
    }


def _navit_resize_video(
    width: int,
    height: int,
    nframes: int,
    avg_fps: float,
    sample_fps: float,
    patch_size: int,
    merge_kernel_size: int,
    in_patch_limit_each_frame: int,
    patch_limit_on_one_side: int,
    in_patch_limit_total: int | None = None,
    max_num_frames_each_video: int | None = None,
    fixed_output_tokens_each_frame: int | None = None,
):
    """Compute resize/pad dimensions and frame sampling count for video."""
    sample_fps = min(sample_fps, avg_fps)
    sampled_nframes = max(round(nframes * sample_fps / avg_fps), 1)
    if max_num_frames_each_video is not None:
        sampled_nframes = min(sampled_nframes, max_num_frames_each_video)

    if in_patch_limit_total is not None:
        in_patch_limit_each_frame = min(
            round(in_patch_limit_total / sampled_nframes),
            in_patch_limit_each_frame,
        )

    ret = _navit_resize_image(
        width,
        height,
        patch_size,
        merge_kernel_size,
        in_patch_limit_each_frame,
        patch_limit_on_one_side,
        fixed_output_tokens_each_frame,
    )
    ret["sampled_nframes"] = sampled_nframes
    return ret


# ---------------------------------------------------------------------------
# GPU image preprocessing utilities (resize, pad, normalize, patchify on CUDA)
# ---------------------------------------------------------------------------


def navit_resize_config(
    width: int,
    height: int,
    patch_size: int,
    merge_kernel_size: int,
    in_patch_limit: int,
    patch_limit_on_one_side: int,
    fixed_output_tokens: int | None = None,
) -> dict:
    """Compute NaViT resize target dimensions and token count.

    Pure math -- no image data needed, only (width, height).
    """
    s1 = math.sqrt(
        in_patch_limit
        / (max(1.0, width // patch_size) * max(1.0, height // patch_size))
    )
    s2 = patch_limit_on_one_side * patch_size / width
    s3 = patch_limit_on_one_side * patch_size / height
    scale = min(1.0, s1, s2, s3)
    new_w = min(max(1, int(width * scale)), patch_limit_on_one_side * patch_size)
    new_h = min(max(1, int(height * scale)), patch_limit_on_one_side * patch_size)

    factor = merge_kernel_size * patch_size
    pad_height = (factor - new_h % factor) % factor
    pad_width = (factor - new_w % factor) % factor

    if fixed_output_tokens is not None:
        num_tokens = fixed_output_tokens
    else:
        token_height = (new_h + pad_height) // factor
        token_width = (new_w + pad_width) // factor
        num_tokens = token_height * token_width

    return {
        "num_tokens": num_tokens,
        "new_width": new_w,
        "new_height": new_h,
        "pad_width": pad_width,
        "pad_height": pad_height,
    }


def _get_image_dimensions(image: Union[torch.Tensor, Image.Image]) -> tuple[int, int]:
    """Get (width, height) from a CUDA tensor or PIL Image."""
    if isinstance(image, torch.Tensor):
        if image.dim() == 2:
            return image.shape[1], image.shape[0]
        if image.dim() != 3:
            raise ValueError(
                f"Expected an HW or CHW image tensor, got shape {tuple(image.shape)}"
            )
        return image.shape[2], image.shape[1]
    return image.size  # PIL returns (width, height)


def _expand_image_token_ids(
    input_ids: Union[List[int], torch.Tensor],
    image_token_id: int,
    image_token_counts: List[int],
) -> torch.Tensor:
    """Expand one placeholder per image without tokenizing the media string again."""
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.detach().flatten().cpu().numpy()
    input_ids = np.asarray(input_ids, dtype=np.int64)

    placeholder_mask = input_ids == image_token_id
    placeholder_count = np.count_nonzero(placeholder_mask)
    if placeholder_count != len(image_token_counts):
        raise ValueError(
            f"Expected {len(image_token_counts)} image placeholder token(s), "
            f"found {placeholder_count}."
        )

    repeats = np.ones(input_ids.shape, dtype=np.int64)
    repeats[placeholder_mask] = image_token_counts
    return torch.from_numpy(np.repeat(input_ids, repeats)).unsqueeze(0)


# ---------------------------------------------------------------------------
# Kimi K2.5 GPU processor wrapper
# ---------------------------------------------------------------------------


class KimiGPUProcessorWrapper:
    """Wraps Kimi's HF processor to do GPU image preprocessing.

    GPU path: nvJPEG CUDA tensor / PIL -> _gpu_preprocess_images()
    CPU fallback: PIL -> medias kwarg -> original HF KimiK25Processor.__call__

    Exposes attributes that base class's process_mm_data needs so it behaves
    like a normal HF processor from the outside.
    """

    def __init__(
        self,
        hf_processor,
        image_token,
        image_token_id,
        patch_size,
        merge_kernel_size,
        in_patch_limit,
        patch_limit_on_one_side,
        fixed_output_tokens,
        image_mean,
        image_std,
    ):
        self._hf_processor = hf_processor
        self._image_token = image_token
        self._image_token_id = image_token_id
        self._patch_size = patch_size
        self._merge_kernel_size = merge_kernel_size
        self._in_patch_limit = in_patch_limit
        self._patch_limit_on_one_side = patch_limit_on_one_side
        self._fixed_output_tokens = fixed_output_tokens
        self._image_mean = image_mean
        self._image_std = image_std
        self._gpu_norm_tensors = None

        # Explicitly expose attributes that base class process_mm_data needs:
        # - image_processor: checked via isinstance(..., BaseImageProcessor)
        # - tokenizer: used for tokenization
        # - media_processor: used by CPU fallback path
        self.image_processor = hf_processor.image_processor
        self.tokenizer = hf_processor.tokenizer
        self.media_processor = hf_processor.media_processor

    def image_token_counts_from_sizes(
        self, image_sizes: List[tuple[int, int]]
    ) -> List[int]:
        """NaViT feature count per image, derived from (width, height) alone.

        ``navit_resize_config`` is pure math on the source dimensions, so a node
        that never touches pixels -- e.g. the decode node of a PD-disaggregated
        deployment -- can still reproduce the prompt length the vision path
        would have produced.
        """
        return [
            navit_resize_config(
                width,
                height,
                self._patch_size,
                self._merge_kernel_size,
                self._in_patch_limit,
                self._patch_limit_on_one_side,
                self._fixed_output_tokens,
            )["num_tokens"]
            for width, height in image_sizes
        ]

    def __call__(self, text=None, images=None, **kwargs):
        # process_mm_data passes images via kwargs["images"]
        images = images or kwargs.pop("images", None)
        original_input_ids = kwargs.pop("sglang_original_input_ids", None)

        if images and torch.cuda.is_available():
            return self._gpu_call(text, images, original_input_ids)
        return self._cpu_call(text, images, **kwargs)

    def _prepare_input_ids(self, input_text, resize_configs, original_input_ids):
        if original_input_ids is not None:
            return _expand_image_token_ids(
                original_input_ids,
                self._image_token_id,
                [config["num_tokens"] for config in resize_configs],
            )

        parts = input_text.split(self._image_token)
        result = [parts[0]]
        for config, part in zip(resize_configs, parts[1:]):
            result.append(self._image_token * config["num_tokens"] + part)
        expanded_text = "".join(result)
        return self._hf_processor.tokenizer(expanded_text, return_tensors="pt")[
            "input_ids"
        ]

    def _gpu_call(self, text, images, original_input_ids=None):
        """Bypass HF KimiK25VisionProcessor.preprocess entirely -- use GPU ops."""
        input_text = text[0] if isinstance(text, list) else text

        # 1. Compute resize configs (CPU math)
        resize_configs = []
        for image in images:
            w, h = _get_image_dimensions(image)
            resize_configs.append(
                navit_resize_config(
                    w,
                    h,
                    self._patch_size,
                    self._merge_kernel_size,
                    self._in_patch_limit,
                    self._patch_limit_on_one_side,
                    self._fixed_output_tokens,
                )
            )

        # 2. Reuse the request's existing tokenization when available. The
        # media placeholder expansion is exact and avoids tokenizing thousands
        # of repeated ``<|media_pad|>`` strings.
        input_ids = self._prepare_input_ids(
            input_text, resize_configs, original_input_ids
        )

        # 3. GPU image preprocessing
        image_scale, image_bias = self._get_gpu_norm_tensors()
        pixel_values, grid_thws = _gpu_preprocess_images(
            images, resize_configs, image_scale, image_bias, self._patch_size
        )

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            # Use SGL-standard key so get_new_expanded_mm_items() can split
            # per-image for cache granularity (it looks up 'image_grid_thw').
            "image_grid_thw": grid_thws,
        }

    def _cpu_call(self, text, images, **kwargs):
        """Fallback: token expansion + medias kwarg -> original HF processor."""
        input_text = text[0] if isinstance(text, list) else text

        if images:
            # Token expansion via media_tokens_calculator
            parts = input_text.split(self._image_token)
            result = [parts[0]]
            for image, part in zip(images, parts[1:]):
                num_tokens = self._hf_processor.media_processor.media_tokens_calculator(
                    {"type": "image", "image": image}
                )
                result.append(self._image_token * num_tokens + part)
            input_text = "".join(result)

            # Convert to medias format for Kimi's HF processor
            kwargs["medias"] = [{"type": "image", "image": img} for img in images]

        out = self._hf_processor(text=[input_text], **kwargs)
        grid_thws = out.pop("grid_thws", None)
        if grid_thws is not None:
            out["image_grid_thw"] = grid_thws
        return out

    def _get_gpu_norm_tensors(self, device="cuda"):
        if self._gpu_norm_tensors is None:
            image_scale = torch.tensor(
                [1.0 / (255.0 * std) for std in self._image_std],
                device=device,
                dtype=torch.float32,
            ).view(1, 3, 1, 1)
            image_bias = torch.tensor(
                [-mean / std for mean, std in zip(self._image_mean, self._image_std)],
                device=device,
                dtype=torch.float32,
            ).view(1, 3, 1, 1)
            self._gpu_norm_tensors = (image_scale, image_bias)
        return self._gpu_norm_tensors


def _pil_to_cuda_chw(image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to (C, H, W) uint8 CUDA tensor."""
    arr = np.asarray(image.convert("RGB"))
    return torch.from_numpy(arr).permute(2, 0, 1).cuda()


def _ensure_chw_rgb(image: torch.Tensor) -> torch.Tensor:
    """Coerce an already-decoded (C, H, W) image tensor to 3-channel RGB.

    PIL inputs are RGB-normalized by _pil_to_cuda_chw, but pre-decoded
    tensor inputs (e.g. nvJPEG / cached CUDA tensors) keep their native
    channel count. Grayscale (1ch) or RGBA (4ch) images then break the
    downstream torch.cat over a batch of images, which requires a
    consistent channel dimension. Normalize every tensor to 3 channels.

    Also move the tensor to the GPU (matching _pil_to_cuda_chw) so a CPU
    input does not trip a device mismatch against the CUDA normalization
    constants downstream. No-op if already on the device.
    """
    image = image.cuda()
    if image.dim() == 2:  # (H, W) grayscale -> (1, H, W)
        image = image.unsqueeze(0)
    c = image.shape[0]
    if c == 3:
        return image
    if c == 1:
        return image.repeat(3, 1, 1)
    # RGBA or other multi-channel layouts: keep the first 3 channels.
    return image[:3]


def _resize_bicubic_if_needed(
    image: torch.Tensor, target_height: int, target_width: int
) -> torch.Tensor:
    """Match the checkpoint processor's ``PIL.Image.resize(..., BICUBIC)``.

    Kimi's HF processors only ever downscale (NaViT scale <= 1.0), and PIL's
    bicubic widens the kernel support by the scale factor, i.e. it always
    antialiases; ``F.interpolate`` needs ``antialias=True`` to do the same.
    PIL also returns uint8, so quantize the resized result back to integer
    pixel values before normalization (round-to-nearest, clipped to [0, 255]);
    without this the float overshoot leaks past the [-1, 1] range the model
    was trained on.
    """
    image = image.float()
    if image.shape[-2:] == (target_height, target_width):
        return image
    return (
        F.interpolate(
            image,
            size=(target_height, target_width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        .round_()
        .clamp_(0.0, 255.0)
    )


def _grid_thw_from_resize_config(config: dict, patch_size: int) -> tuple[int, int, int]:
    height = config["new_height"] + config["pad_height"]
    width = config["new_width"] + config["pad_width"]
    return 1, height // patch_size, width // patch_size


def _default_to_cuda_chw(image: Union[torch.Tensor, Image.Image]) -> torch.Tensor:
    if isinstance(image, Image.Image):
        return _pil_to_cuda_chw(image)
    return _ensure_chw_rgb(image)


def _process_single_image(
    image: Union[torch.Tensor, Image.Image],
    config: dict,
    image_scale: torch.Tensor,
    image_bias: torch.Tensor,
    patch_size: int,
    to_chw: Callable[
        [Union[torch.Tensor, Image.Image]], torch.Tensor
    ] = _default_to_cuda_chw,
    post_resize: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    """Process a single image on GPU: resize -> pad -> normalize -> patchify.

    ``to_chw`` converts the input to a CUDA CHW tensor (models may keep an
    alpha channel here); ``post_resize`` runs on the resized ``(B, C, H, W)``
    batch before patchify (e.g. K3's transparent-background compositing).
    """
    image = to_chw(image)

    new_h, new_w = config["new_height"], config["new_width"]
    padded_h = new_h + config["pad_height"]
    padded_w = new_w + config["pad_width"]

    x = _resize_bicubic_if_needed(image.unsqueeze(0), new_h, new_w)
    if post_resize is not None:
        x = post_resize(x)

    return normalize_and_patchify(
        x, image_scale, image_bias, patch_size, padded_h, padded_w
    ).squeeze(0)


def _resize_images_by_source_shape(
    indexed_images: list[tuple[int, torch.Tensor]],
    target_height: int,
    target_width: int,
) -> list[torch.Tensor]:
    """Resize images while batching only inputs with an identical source layout.

    A NaViT target-size group can still contain images with different source
    dimensions.  Interpolation requires a rectangular batch, so preserve the
    individual path for those images and batch only equal ``(shape, dtype)``
    inputs.  The returned tensors retain the caller's original image order.
    """
    by_source_shape = defaultdict(list)
    for index, image in indexed_images:
        by_source_shape[(tuple(image.shape), image.dtype)].append((index, image))

    resized_by_index = {}
    for images in by_source_shape.values():
        if len(images) == 1:
            index, image = images[0]
            resized_by_index[index] = _resize_bicubic_if_needed(
                image.unsqueeze(0), target_height, target_width
            )
            continue

        source_batch = torch.cat([image.unsqueeze(0) for _, image in images], dim=0)
        resized_batch = _resize_bicubic_if_needed(
            source_batch, target_height, target_width
        )
        for local_index, (index, _) in enumerate(images):
            resized_by_index[index] = resized_batch[local_index : local_index + 1]

    return [resized_by_index[index] for index, _ in indexed_images]


def _gpu_preprocess_images(
    images: list[Union[torch.Tensor, Image.Image]],
    resize_configs: list[dict],
    image_scale: torch.Tensor,
    image_bias: torch.Tensor,
    patch_size: int,
    to_chw: Callable[
        [Union[torch.Tensor, Image.Image]], torch.Tensor
    ] = _default_to_cuda_chw,
    post_resize: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU preprocessing pipeline for a batch of images.

    Groups images with the same target padded size for batch processing.
    """
    n = len(images)
    if n == 0:
        device = image_scale.device
        return (
            torch.empty(0, 3, patch_size, patch_size, device=device),
            torch.empty(0, 3, dtype=torch.int64),
        )

    groups = defaultdict(list)
    for idx, (image, config) in enumerate(zip(images, resize_configs)):
        padded_h = config["new_height"] + config["pad_height"]
        padded_w = config["new_width"] + config["pad_width"]
        target_h = config["new_height"]
        target_w = config["new_width"]
        groups[(target_h, target_w, padded_h, padded_w)].append((idx, image, config))

    all_patches = [None] * n
    all_grids = [None] * n

    for (target_h, target_w, padded_h, padded_w), group in groups.items():
        if len(group) == 1:
            idx, image, config = group[0]
            patches = _process_single_image(
                image,
                config,
                image_scale,
                image_bias,
                patch_size,
                to_chw=to_chw,
                post_resize=post_resize,
            )
            all_patches[idx] = patches
            all_grids[idx] = _grid_thw_from_resize_config(config, patch_size)
        else:
            indexed_images = []
            for idx, image, _ in group:
                indexed_images.append((idx, to_chw(image)))

            # One NaViT target group can include several original resolutions.
            # Batch only source-compatible images, which removes redundant
            # bicubic launches for common multi-image requests without padding
            # random-size inputs to a larger source resolution.
            resized = _resize_images_by_source_shape(indexed_images, target_h, target_w)
            if post_resize is not None:
                # Runs before the concat: a hook may change the channel count
                # (K3 composites RGBA onto a background, returning RGB), and
                # mixed 3/4-channel sources cannot be concatenated first.
                resized = [post_resize(part) for part in resized]
            batch = torch.cat(resized, dim=0)

            T = 1
            gh, gw = padded_h // patch_size, padded_w // patch_size
            batch = normalize_and_patchify(
                batch,
                image_scale,
                image_bias,
                patch_size,
                padded_h,
                padded_w,
            )

            grid = (T, gh, gw)
            for i, (idx, _, _) in enumerate(group):
                all_patches[idx] = batch[i]
                all_grids[idx] = grid

    pixel_values = torch.cat(all_patches, dim=0)
    grid_thws = torch.tensor(all_grids, dtype=torch.int64)
    return pixel_values, grid_thws


# ---------------------------------------------------------------------------
# Kimi K2.5 SGLang multimodal processor
# ---------------------------------------------------------------------------


# Compatible with KimiVLForConditionalGeneration
class KimiK25VLImageProcessor(KimiGridMMDataMixin, SGLangBaseProcessor):
    """Multimodal image processor for Kimi K2.5 VL model."""
    models = [KimiK25ForConditionalGeneration]
    gpu_image_decode = True  # nvJPEG for JPEG, PIL fallback for others
    prefer_tokenized_input = True
    precompute_hash_before_cpu_transfer = True
    auto_mm_processor_worker_num = 2
    auto_mm_io_worker_num = 16
    supports_mm_processor_concurrency = True

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        mm_tokens = MultimodalSpecialTokens(
            image_token="<|media_pad|>",
            # TODO: could we convert in MultimodalSpecialTokens?
            image_token_id=hf_config.media_placeholder_token_id,
            image_token_regex=re.compile(r"(?:<\|media_pad\|>)+"),
        ).build(_processor)
        # Required by the disaggregated-encode path (get_mm_data -> build_input_ids).
        self.IM_TOKEN_ID = hf_config.media_placeholder_token_id
        self.IM_START_TOKEN_ID = None
        self.IM_END_TOKEN_ID = None

        media_proc_cfg = _processor.media_processor.media_proc_cfg
        processor = KimiGPUProcessorWrapper(
            _processor,
            image_token=mm_tokens.image_token,
            image_token_id=mm_tokens.image_token_id,
            patch_size=media_proc_cfg["patch_size"],
            merge_kernel_size=media_proc_cfg["merge_kernel_size"],
            in_patch_limit=media_proc_cfg["in_patch_limit"],
            patch_limit_on_one_side=media_proc_cfg["patch_limit_on_one_side"],
            fixed_output_tokens=media_proc_cfg.get("fixed_output_tokens"),
            image_mean=media_proc_cfg["image_mean"],
            image_std=media_proc_cfg["image_std"],
        )
        # Initialize the executor from the final GPU wrapper. Cloning the raw
        # HF processor here would silently bypass Kimi's GPU preprocessing.
        super().__init__(hf_config, server_args, processor, *args, **kwargs)
        self.mm_tokens = mm_tokens

    @property
    def spatial_merge_size(self) -> int:
        """
        KimiK25VisionConfig uses merge_kernel_size=(H,W); spatial factor equals H.
        """
        return self.hf_config.vision_config.merge_kernel_size[0]

    def build_input_ids(
        self, prompt, img_grid_thw=None, video_grid_thw=None, audio_seq_lens=None
    ):
        """Kimi-specific input_ids builder for the disaggregated-encode path.

        Kimi has no separate video token: image and video both use
        ``<|media_pad|>``. Each ``img_grid_thw`` row maps 1:1 to one placeholder
        (image -> 1 row, video -> N chunk rows) and is counted with the
        temporal-pooled :func:`num_tokens_from_grid`, avoiding the base builder's
        ~4x over-reserve on video rows.
        """
        if not isinstance(prompt, list):
            prompt = self._tokenizer.encode(prompt)

        merge_kernel_size = self.hf_config.vision_config.merge_kernel_size

        token_counts: List[int] = []
        if img_grid_thw is not None:
            for grid_thw in img_grid_thw:
                token_counts.append(
                    num_tokens_from_grid(grid_thw, merge_kernel_size=merge_kernel_size)
                )

        expanded, offsets = expand_media_placeholders_with_offsets(
            prompt, token_counts, self.IM_TOKEN_ID
        )
        modality_list = [Modality.IMAGE] * len(offsets)
        return expanded, offsets, modality_list

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes, Dict]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        expected_image_count = len(image_data or [])
        if self.validate_tokenized_image_placeholders(
            input_text, self.mm_tokens.image_token_id, expected_image_count
        ):
            base_output = await self.fast_load_mm_data(
                prompt=input_text,
                image_data=image_data,
                multimodal_tokens=self.mm_tokens,
            )
        else:
            base_output = await self.load_mm_data(
                prompt=input_text,
                image_data=image_data,
                multimodal_tokens=self.mm_tokens,
            )

        if len(base_output.images) != expected_image_count:
            raise ValueError(
                "Kimi image placeholders must map one-to-one to image data: "
                f"expected {expected_image_count}, loaded {len(base_output.images)}"
            )

        mm_items, input_ids, _ = await self.process_and_combine_mm_data_async(
            base_output,
            self.mm_tokens,
            sglang_original_input_ids=base_output.input_ids,
        )

        # K2.5/K2.7 encoder-DP assigns an image to exactly one TP rank. Keep
        # its IPC proxy lazy until that assignment is known, avoiding a full
        # image copy to every rank. The scheduler only honors this marker once
        # the processor has already set the item's hash and pad value.
        if self.use_cuda_ipc and self.server_args.mm_enable_dp_encoder:
            for item in mm_items:
                item.model_specific_data[DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY] = (
                    True
                )

        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
            im_token_id=self.mm_tokens.image_token_id,
        )

    def _process_and_collect_mm_items(
        self, input_text: str, images=None, audios=None, videos=None, **kwargs
    ) -> Tuple[List[MultimodalDataItem], torch.Tensor, dict]:
        """
        Helper method to process multimodal data and create mm_items in one step.

        Returns:
            Tuple of (created mm_items, input_ids)
        """

        parts = input_text.split(self.mm_tokens.image_token)

        result = [parts[0]]
        for image, part in zip(images, parts[1:]):
            num_tokens = self._processor.media_processor.media_tokens_calculator(
                {"type": "image", "image": image}
            )
            result.append(self.mm_tokens.image_token * num_tokens + part)

        input_text = "".join(result)

        if images:  # for kimi k2 vl
            mediums = []
            for image in images:
                mediums.append({"type": "image", "image": image})
            key = "_medias"[1:]  # bypass lint
            kwargs[key] = mediums
            images = None

        ret = self.process_mm_data(
            input_text=input_text, images=images, audios=audios, videos=videos, **kwargs
        )

        input_ids = ret["input_ids"].flatten()
        collected_items = self.collect_mm_items_from_processor_output(ret)

        return collected_items, input_ids, ret

    @staticmethod
    def navit_resize_config(
        width: int,
        height: int,
        patch_size: int,
        merge_kernel_size: int,
        in_patch_limit: int,
        patch_limit_on_one_side: int,
        fixed_output_tokens: int | None = None,
    ) -> dict:
        """Compute NaViT resize target dimensions and token count.

        Pure math -- no image data needed, only (width, height).
        """
        s1 = math.sqrt(
            in_patch_limit
            / (max(1.0, width // patch_size) * max(1.0, height // patch_size))
        )
        s2 = patch_limit_on_one_side * patch_size / width
        s3 = patch_limit_on_one_side * patch_size / height
        scale = min(1.0, s1, s2, s3)
        new_w = min(max(1, int(width * scale)), patch_limit_on_one_side * patch_size)
        new_h = min(max(1, int(height * scale)), patch_limit_on_one_side * patch_size)

        factor = merge_kernel_size * patch_size
        pad_height = (factor - new_h % factor) % factor
        pad_width = (factor - new_w % factor) % factor

        if fixed_output_tokens is not None:
            num_tokens = fixed_output_tokens
        else:
            token_height = (new_h + pad_height) // factor
            token_width = (new_w + pad_width) // factor
            num_tokens = token_height * token_width

        return {
            "num_tokens": num_tokens,
            "new_width": new_w,
            "new_height": new_h,
            "pad_width": pad_width,
            "pad_height": pad_height,
        }


    @staticmethod
    def _get_image_dimensions(image: Union[torch.Tensor, Image.Image]) -> tuple[int, int]:
        """Get (width, height) from a CUDA tensor or PIL Image."""
        if isinstance(image, torch.Tensor):
            if image.dim() == 2:
                return image.shape[1], image.shape[0]
            if image.dim() != 3:
                raise ValueError(
                    f"Expected an HW or CHW image tensor, got shape {tuple(image.shape)}"
                )
            return image.shape[2], image.shape[1]
        return image.size  # PIL returns (width, height)

    @staticmethod
    def _pil_to_cuda_chw(
        image: Image.Image, device: torch.device
    ) -> torch.Tensor:
        """Convert a PIL image to an RGB CHW tensor on the configured device."""
        arr = np.asarray(image.convert("RGB"))
        t = torch.from_numpy(arr.copy()).permute(2, 0, 1).contiguous()
        return t.to(device, non_blocking=True)

    @staticmethod
    def _ensure_chw_rgb(
        image: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Move a decoded HW/CHW tensor to the device and normalize it to RGB."""
        image = image.to(device, non_blocking=True)
        if image.dim() == 2:
            image = image.unsqueeze(0)
        elif image.dim() != 3:
            raise ValueError(
                f"Expected an HW or CHW image tensor, got shape {tuple(image.shape)}"
            )
        channels = image.shape[0]
        if channels == 3:
            return image
        if channels == 1:
            return image.repeat(3, 1, 1)
        if channels >= 4:
            return image[:3]
        raise ValueError(f"Expected 1, 3, or at least 4 channels, got {channels}")

    @staticmethod
    def _batch_pil_to_cuda(
        images: list[Image.Image], device: torch.device
    ) -> list[torch.Tensor]:
        """Convert PIL images to RGB CHW tensors on the configured device."""
        return [
            KimiK25VLImageProcessor._pil_to_cuda_chw(image, device)
            for image in images
        ]

    @staticmethod
    def _process_single_image(
        image: Union[torch.Tensor, Image.Image],
        config: dict,
        image_mean: torch.Tensor,
        image_std_inv: torch.Tensor,
        patch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process a single image on GPU: resize -> pad -> normalize -> patchify."""
        device = image_mean.device
        if isinstance(image, Image.Image):
            image = KimiK25VLImageProcessor._pil_to_cuda_chw(image, device)
        else:
            image = KimiK25VLImageProcessor._ensure_chw_rgb(image, device)

        new_h, new_w = config["new_height"], config["new_width"]
        pad_h, pad_w = config["pad_height"], config["pad_width"]

        x = image.unsqueeze(0).float()
        x = F.interpolate(x, size=(new_h, new_w), mode="bicubic", align_corners=False)

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), value=0.0)

        x = x / 255.0
        x = (x - image_mean) * image_std_inv

        _, C, H, W = x.shape
        T = 1
        gh, gw = H // patch_size, W // patch_size
        x = x.view(T, C, gh, patch_size, gw, patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(-1, C, patch_size, patch_size)

        grid_thw = torch.tensor([T, gh, gw], dtype=torch.int64, device=x.device)
        return x, grid_thw

    @staticmethod
    def _gpu_preprocess_images(
        images: list[Union[torch.Tensor, Image.Image]],
        resize_configs: list[dict],
        image_mean: torch.Tensor,
        image_std_inv: torch.Tensor,
        patch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """GPU preprocessing pipeline for a batch of images.

        Groups images with the same target padded size for batch processing.
        Uses batch H2D transfer via pinned memory for better throughput.
        """
        n = len(images)
        if n == 0:
            device = image_mean.device
            return (
                torch.empty(0, 3, patch_size, patch_size, device=device),
                torch.empty(0, 3, dtype=torch.int64, device=device),
            )

        device = image_mean.device

        # Phase 1: Convert PIL images to device tensors before grouping.
        pil_indices = [i for i, img in enumerate(images) if isinstance(img, Image.Image)]
        if pil_indices:
            pil_images = [images[i] for i in pil_indices]
            cuda_tensors = KimiK25VLImageProcessor._batch_pil_to_cuda(
                pil_images, device
            )
            # Replace PIL images with GPU tensors in-place (working copy)
            images = list(images)  # make a mutable copy
            for i, cuda_t in zip(pil_indices, cuda_tensors):
                images[i] = cuda_t

        images = [
            image
            if isinstance(image, Image.Image)
            else KimiK25VLImageProcessor._ensure_chw_rgb(image, device)
            for image in images
        ]

        # Phase 2: Group by target size and process on GPU
        groups = defaultdict(list)
        for idx, (image, config) in enumerate(zip(images, resize_configs)):
            padded_h = config["new_height"] + config["pad_height"]
            padded_w = config["new_width"] + config["pad_width"]
            target_h = config["new_height"]
            target_w = config["new_width"]
            groups[(target_h, target_w, padded_h, padded_w)].append((idx, image, config))

        all_patches = [None] * n
        all_grids = [None] * n

        for (target_h, target_w, padded_h, padded_w), group in groups.items():
            if len(group) == 1:
                idx, image, config = group[0]
                patches, grid = KimiK25VLImageProcessor._process_single_image(
                    image, config, image_mean, image_std_inv, patch_size
                )
                all_patches[idx] = patches
                all_grids[idx] = grid
            else:
                # All images are already GPU tensors after Phase 1
                tensors = [image.unsqueeze(0).float() for _, image, _ in group]

                resized = []
                for t in tensors:
                    r = F.interpolate(
                        t, size=(target_h, target_w), mode="bicubic", align_corners=False
                    )
                    resized.append(r)
                batch = torch.cat(resized, dim=0)

                pad_h = padded_h - target_h
                pad_w = padded_w - target_w
                if pad_h > 0 or pad_w > 0:
                    batch = F.pad(batch, (0, pad_w, 0, pad_h), value=0.0)

                batch = batch / 255.0
                batch = (batch - image_mean) * image_std_inv

                B, C, H, W = batch.shape
                T = 1
                gh, gw = H // patch_size, W // patch_size
                batch = batch.view(B, C, gh, patch_size, gw, patch_size)
                batch = batch.permute(0, 2, 4, 1, 3, 5).reshape(
                    B, -1, C, patch_size, patch_size
                )

                grid = torch.tensor([T, gh, gw], dtype=torch.int64, device=batch.device)
                for i, (idx, _, _) in enumerate(group):
                    all_patches[idx] = batch[i]
                    all_grids[idx] = grid

        pixel_values = torch.cat(all_patches, dim=0)
        grid_thws = torch.stack(all_grids, dim=0)
        return pixel_values, grid_thws

    @staticmethod
    def create_gpu_preprocess_config(
        hf_processor, device: torch.device
    ) -> "KimiK25GPUPreprocessConfig":
        """Extract preprocessing config from HF processor and pre-create GPU tensors."""
        media_proc_cfg = hf_processor.media_proc_cfg
        image_mean = torch.tensor(
            media_proc_cfg["image_mean"], device=device, dtype=torch.float32
        ).view(1, 3, 1, 1)
        image_std_inv = (
            1.0
            / torch.tensor(
                media_proc_cfg["image_std"], device=device, dtype=torch.float32
            )
        ).view(1, 3, 1, 1)
        return KimiK25GPUPreprocessConfig(
            patch_size=media_proc_cfg["patch_size"],
            merge_kernel_size=media_proc_cfg["merge_kernel_size"],
            in_patch_limit=media_proc_cfg["in_patch_limit"],
            patch_limit_on_one_side=media_proc_cfg["patch_limit_on_one_side"],
            fixed_output_tokens=media_proc_cfg.get("fixed_output_tokens"),
            image_mean=image_mean,
            image_std_inv=image_std_inv,
        )

    @staticmethod
    def gpu_preprocess(
        images: list[Union[torch.Tensor, Image.Image]],
        config: "KimiK25GPUPreprocessConfig",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Unified entry for GPU image preprocessing (encode server).

        Computes resize configs and runs the GPU batch pipeline in one call,
        so callers do not need to know about model-specific internals.
        """
        resize_configs = []
        for image in images:
            w, h = KimiK25VLImageProcessor._get_image_dimensions(image)
            resize_configs.append(
                KimiK25VLImageProcessor.navit_resize_config(
                    w, h, config.patch_size, config.merge_kernel_size,
                    config.in_patch_limit, config.patch_limit_on_one_side,
                    config.fixed_output_tokens,
                )
            )
        return KimiK25VLImageProcessor._gpu_preprocess_images(
            images, resize_configs, config.image_mean, config.image_std_inv,
            config.patch_size,
        )

    @staticmethod
    def create_video_preprocess_config(
        hf_processor, device: torch.device
    ) -> "KimiK25VideoPreprocessConfig":
        """Extract video preprocessing config from HF processor."""
        media_proc_cfg = hf_processor.media_proc_cfg
        image_mean = torch.tensor(
            media_proc_cfg["image_mean"], device=device, dtype=torch.float32
        ).view(1, 3, 1, 1)
        image_std_inv = (
            1.0
            / torch.tensor(
                media_proc_cfg["image_std"], device=device, dtype=torch.float32
            )
        ).view(1, 3, 1, 1)
        return KimiK25VideoPreprocessConfig(
            patch_size=media_proc_cfg["patch_size"],
            merge_kernel_size=media_proc_cfg["merge_kernel_size"],
            temporal_merge_kernel_size=media_proc_cfg.get("temporal_merge_kernel_size", 4),
            in_patch_limit_each_frame=media_proc_cfg.get("in_patch_limit_each_frame", media_proc_cfg["in_patch_limit"]),
            patch_limit_on_one_side=media_proc_cfg["patch_limit_on_one_side"],
            sample_fps=media_proc_cfg.get("sample_fps", 2.0),
            in_patch_limit_video=media_proc_cfg.get("in_patch_limit_video"),
            max_num_frames_each_video=media_proc_cfg.get("max_num_frames_each_video"),
            fixed_output_tokens_each_frame=media_proc_cfg.get("fixed_output_tokens"),
            image_mean=image_mean,
            image_std_inv=image_std_inv,
        )

    @staticmethod
    def gpu_preprocess_video_chunk(
        frames: torch.Tensor,
        config: "KimiK25VideoPreprocessConfig",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """GPU preprocess a single temporal chunk: resize -> pad -> normalize -> patchify.

        Args:
            frames: (T, C, H, W) float32 GPU tensor of decoded video frames.
            config: Video preprocess config with model-specific parameters.

        Returns:
            (pixel_values, grid_thw) where pixel_values is (N, C, P, P) and
            grid_thw is (3,) tensor [T, gh, gw].
        """
        T, C, H, W = frames.shape
        resize_cfg = _navit_resize_video(
            width=W, height=H, nframes=T,
            avg_fps=1.0, sample_fps=config.sample_fps,
            patch_size=config.patch_size,
            merge_kernel_size=config.merge_kernel_size,
            in_patch_limit_each_frame=config.in_patch_limit_each_frame,
            patch_limit_on_one_side=config.patch_limit_on_one_side,
            in_patch_limit_total=config.in_patch_limit_video,
            max_num_frames_each_video=config.max_num_frames_each_video,
            fixed_output_tokens_each_frame=config.fixed_output_tokens_each_frame,
        )
        new_w, new_h = resize_cfg["new_width"], resize_cfg["new_height"]
        pad_w, pad_h = resize_cfg["pad_width"], resize_cfg["pad_height"]

        x = F.interpolate(frames, size=(new_h, new_w), mode="bicubic", align_corners=False)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), value=0.0)
        x = x / 255.0
        x = (x - config.image_mean) * config.image_std_inv

        _, C, H_out, W_out = x.shape
        ps = config.patch_size
        gh, gw = H_out // ps, W_out // ps
        x = x.view(T, C, gh, ps, gw, ps)
        patches = x.permute(0, 2, 4, 1, 3, 5).reshape(-1, C, ps, ps)
        grid_thw = torch.tensor([T, gh, gw], dtype=torch.int64, device=frames.device)
        return patches, grid_thw

    @staticmethod
    def get_video_compute_weight(config: "KimiK25VideoPreprocessConfig") -> int:
        """Return the compute weight multiplier for video chunks.

        Kimi K2.5 processes T frames per video chunk in a single forward pass,
        making each chunk roughly T× more expensive than an equivalently-sized
        image.  The multiplier is used by the encode-server's free_tokens
        scheduler to account for the extra GPU pressure of video tasks.

        The value equals ``temporal_merge_kernel_size`` (i.e. the number of
        frames packed into one chunk), which defaults to 4.
        """
        return config.temporal_merge_kernel_size

    @staticmethod
    def compute_num_tokens_per_item(model, mm_item) -> int:
        """Compute the number of LLM-visible tokens a single mm_item produces.

        Kimi K2.5 applies temporal pooling inside each video chunk, so the vision
        encoder collapses the time dimension and outputs one spatial token grid per
        chunk.  The output token count is therefore:

            (gh // merge_h) * (gw // merge_w)

        where grid_thw = [T, gh, gw] and merge_kernel_size = (merge_h, merge_w).
        T is intentionally *not* multiplied — temporal merge already happened.

        For images the same formula applies (T == 1, but still not multiplied).

        This method is the single source of truth for Kimi token counting and
        should be kept in sync with the model's vision tower implementation.
        """
        from sglang.srt.disaggregation.encode_utils import get_image_grid_dim

        grid_dim = get_image_grid_dim(mm_item)
        if not isinstance(grid_dim, torch.Tensor):
            grid_dim = torch.tensor(grid_dim)
        if grid_dim.dim() == 1:
            grid_dim = grid_dim.unsqueeze(0)

        merge_kernel_size = model.config.vision_config.merge_kernel_size
        merge_h, merge_w = merge_kernel_size

        return sum(
            (int(g[1]) // merge_h) * (int(g[2]) // merge_w)
            for g in grid_dim
        )

    @staticmethod
    def compute_frame_sampling(num_frames: int, avg_fps: float, sample_fps: float) -> list[int]:
        """Compute uniform frame sampling indices based on target fps."""
        if num_frames <= 0:
            return [0]
        effective_fps = min(sample_fps, avg_fps)
        sampled = max(round(num_frames * effective_fps / avg_fps), 1)
        return np.linspace(0, num_frames - 1, sampled).round().astype(int).tolist()

    @staticmethod
    def get_temporal_merge_kernel_size(config: "KimiK25VideoPreprocessConfig") -> int:
        """Return the number of frames per temporal chunk."""
        return config.temporal_merge_kernel_size

    @staticmethod
    def compute_chunk_timestamps(
        frame_indices: list[int], avg_fps: float, tmks: int, timestamp_mode: str = "mm:ss"
    ) -> list[str]:
        """Generate per-chunk timestamp prompts for video."""
        from datetime import datetime, timezone
        prompts = []
        for i in range(0, len(frame_indices), tmks):
            start_time = frame_indices[i] / float(avg_fps)
            if timestamp_mode == "hh:mm:ss.fff":
                ts = (datetime.fromtimestamp(start_time, tz=timezone.utc).strftime("%H:%M:%S")
                      + f".{int((start_time % 1) * 1000):03d}")
            elif timestamp_mode == "mm:ss.fff":
                ts = (datetime.fromtimestamp(start_time, tz=timezone.utc).strftime("%M:%S")
                      + f".{int((start_time % 1) * 1000):03d}")
            else:
                ts = datetime.fromtimestamp(start_time, tz=timezone.utc).strftime("%M:%S")
            prompts.append(f"{ts}<|media_begin|>video<|media_content|><|media_pad|><|media_end|>")
        return prompts
