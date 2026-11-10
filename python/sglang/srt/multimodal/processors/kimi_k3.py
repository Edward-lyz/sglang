"""Kimi K3 multimodal processor.

GPU image preprocessing dedicated to K3: unlike the K2.5 wrapper it keeps
the alpha channel through the bicubic resize and then composites RGBA
images onto the checkpoint-configured background
(``transparent_bg_config`` with ``transparent_bg_fill_stage ==
"after_resize"`` in preprocessor_config.json), instead of dropping alpha
at load time.
"""

import re
from typing import Dict, List, Union

import numpy as np
import torch
from PIL import Image

from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
from sglang.srt.models.kimi_k3 import (
    KimiK3ForConditionalGeneration,
    KimiK31ForConditionalGeneration,
)
from sglang.srt.multimodal.mm_utils import has_valid_data
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import (
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.kimi_common import KimiGridMMDataMixin
from sglang.srt.multimodal.processors.kimi_k25 import (
    KimiGPUProcessorWrapper,
    KimiK25VideoPreprocessConfig,
    KimiK25VLImageProcessor,
    _get_image_dimensions,
    _gpu_preprocess_images,
    navit_resize_config,
)
from sglang.srt.utils.cuda_ipc_transport_utils import (
    DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY,
)


def _encode_k3_special_tokens(tokenizer, text: str) -> list[int]:
    """Encode K3 control tokens without allowing them to be BPE-split."""
    try:
        return list(tokenizer.encode(text, allowed_special="all"))
    except TypeError:
        # Keep the helper usable with lightweight tokenizer stubs in CPU tests.
        return list(tokenizer.encode(text))


def _expand_k3_media_prompt_token_ids(
    input_ids: Union[List[int], torch.Tensor],
    image_token_id: int,
    image_token_counts: List[int],
    image_sizes: List[tuple[int, int]],
    tokenizer,
) -> torch.Tensor:
    """Expand K3 media placeholders into the checkpoint's media contract.

    Every placeholder is expanded to its feature count. An image placeholder is
    additionally enclosed in its original uploaded dimensions; a video
    placeholder arrives already framed by the encode server and is recognised by
    its preceding ``media_content`` token. ``image_token_counts`` covers every
    placeholder in prompt order, ``image_sizes`` only the image ones.
    """
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.detach().flatten().cpu().numpy()
    input_ids = np.asarray(input_ids, dtype=np.int64)

    placeholder_count = np.count_nonzero(input_ids == image_token_id)
    if placeholder_count != len(image_token_counts):
        raise ValueError(
            f"Expected {len(image_token_counts)} media placeholder token(s), "
            f"found {placeholder_count}."
        )

    media_content_ids = _encode_k3_special_tokens(tokenizer, "<|media_content|>")
    media_content_token_id = media_content_ids[-1] if media_content_ids else None

    output = []
    image_index = 0
    placeholder_index = 0
    previous_token_id = None
    for token_id in input_ids:
        token_id = int(token_id)
        if token_id != image_token_id:
            output.append(token_id)
            previous_token_id = token_id
            continue

        num_tokens = image_token_counts[placeholder_index]
        placeholder_index += 1
        if previous_token_id == media_content_token_id:
            output.extend([image_token_id] * num_tokens)
        else:
            if image_index >= len(image_sizes):
                raise ValueError(
                    "Expected one original size for each unframed K3 image "
                    f"placeholder; ran out after {len(image_sizes)}."
                )
            width, height = image_sizes[image_index]
            image_index += 1
            output.extend(
                _encode_k3_special_tokens(
                    tokenizer,
                    f"<|media_begin|>image {width}x{height}<|media_content|>",
                )
            )
            output.extend([image_token_id] * num_tokens)
            output.extend(_encode_k3_special_tokens(tokenizer, "<|media_end|>"))
        previous_token_id = token_id

    if image_index != len(image_sizes):
        raise ValueError(
            f"Got {len(image_sizes)} original image size(s) but only "
            f"{image_index} unframed K3 image placeholder(s)."
        )

    return torch.tensor(output, dtype=torch.long).unsqueeze(0)


def _expand_k3_image_prompt_text(
    input_text: str,
    image_token: str,
    image_token_counts: List[int],
    image_sizes: List[tuple[int, int]],
) -> str:
    """Render the K3 media framing for the CPU HF-processor fallback."""
    parts = input_text.split(image_token)
    if len(parts) - 1 != len(image_token_counts):
        raise ValueError(
            f"Expected {len(image_token_counts)} image placeholder(s), "
            f"found {len(parts) - 1}."
        )

    output = [parts[0]]
    for image_token_count, (width, height), suffix in zip(
        image_token_counts, image_sizes, parts[1:]
    ):
        output.extend(
            (
                f"<|media_begin|>image {width}x{height}<|media_content|>",
                image_token * image_token_count,
                "<|media_end|>",
                suffix,
            )
        )
    return "".join(output)


def _k3_to_cuda_chw(image: Union[torch.Tensor, Image.Image]) -> torch.Tensor:
    if isinstance(image, Image.Image):
        # The checkpoint's fill_transparent_bg_with() returns RGB-mode images
        # untouched before it ever inspects the alpha bands, so an RGB image
        # carrying a stray "transparency" info key must NOT be promoted to
        # RGBA here.
        has_alpha = image.mode != "RGB" and (
            "A" in image.getbands() or "transparency" in image.info
        )
        arr = np.asarray(image.convert("RGBA" if has_alpha else "RGB"))
        return torch.from_numpy(arr).permute(2, 0, 1).cuda()

    image = image.cuda()
    if image.dim() == 2:
        image = image.unsqueeze(0)
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    return image


def _chessboard_background(
    height: int, width: int, cfg: dict, device: torch.device
) -> torch.Tensor:
    square = cfg.get("chessboard_square_size", 16)
    white = float(cfg.get("chessboard_white_value", 255))
    gray = float(cfg.get("chessboard_gray_value", 200))
    on_top_left = cfg.get("chessboard_square_on_top_left", True)

    ys = torch.arange(height, device=device) // square
    xs = torch.arange(width, device=device) // square
    parity = (ys.unsqueeze(1) + xs.unsqueeze(0)) % 2
    gray_parity = 1 if on_top_left else 0
    bg = torch.where(parity == gray_parity, gray, white)
    return bg.unsqueeze(0).expand(3, height, width)


def _fill_transparent_bg(x: torch.Tensor, bg_cfg: Union[dict, None]) -> torch.Tensor:
    """Composite a resized (1, 4, H, W) float image in [0, 255] onto the
    configured background; 3-channel input passes through."""
    if x.shape[1] == 3:
        return x
    rgb = x[:, :3]
    if bg_cfg is None:
        return rgb

    _, _, height, width = x.shape
    pattern = bg_cfg.get("pattern", "black")
    if pattern == "chessboard":
        bg = _chessboard_background(height, width, bg_cfg, x.device)
    elif pattern == "white":
        bg = torch.full((3, height, width), 255.0, device=x.device)
    elif pattern == "black":
        bg = torch.zeros(3, height, width, device=x.device)
    elif pattern == "gray":
        bg = torch.full((3, height, width), 128.0, device=x.device)
    else:
        raise ValueError(f"Invalid background pattern: {pattern}")

    alpha = (x[:, 3:4] / 255.0).clamp(0.0, 1.0)
    # The checkpoint processor casts the composited float result back with
    # numpy's astype(np.uint8), which truncates; floor matches that exactly
    # (a composite of [0, 255] inputs is always non-negative).
    return (alpha * rgb + (1.0 - alpha) * bg).clamp(0.0, 255.0).floor_()


class KimiK3GPUProcessorWrapper(KimiGPUProcessorWrapper):
    def __init__(self, *args, transparent_bg_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._transparent_bg_config = transparent_bg_config

    def _prepare_input_ids(
        self, input_text, resize_configs, original_input_ids, image_sizes
    ):
        image_token_counts = [config["num_tokens"] for config in resize_configs]
        if original_input_ids is None:
            original_input_ids = _encode_k3_special_tokens(
                self._hf_processor.tokenizer, input_text
            )
        return _expand_k3_media_prompt_token_ids(
            original_input_ids,
            self._image_token_id,
            image_token_counts,
            image_sizes,
            self._hf_processor.tokenizer,
        )

    def __call__(self, text=None, images=None, **kwargs):
        images = images or kwargs.pop("images", None)
        original_input_ids = kwargs.pop("sglang_original_input_ids", None)
        if images and torch.cuda.is_available():
            return self._gpu_call(text, images, original_input_ids)
        return self._cpu_call(text, images, original_input_ids, **kwargs)

    def _gpu_call(self, text, images, original_input_ids=None):
        input_text = text[0] if isinstance(text, list) else text

        resize_configs = []
        image_sizes = []
        for image in images:
            w, h = _get_image_dimensions(image)
            image_sizes.append((w, h))
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

        input_ids = self._prepare_input_ids(
            input_text, resize_configs, original_input_ids, image_sizes
        )

        image_scale, image_bias = self._get_gpu_norm_tensors()
        # Shared source-compatible batched pipeline (same as K2.5): RGBA
        # inputs land in their own source-shape groups, and the
        # transparent-background compositing runs on each resized batch
        # before patchify -- identical order to the previous per-image path.
        pixel_values, grid_thws = _gpu_preprocess_images(
            images,
            resize_configs,
            image_scale,
            image_bias,
            self._patch_size,
            to_chw=_k3_to_cuda_chw,
            post_resize=lambda x: _fill_transparent_bg(x, self._transparent_bg_config),
        )

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": grid_thws,
        }

    def _cpu_call(self, text, images, original_input_ids=None, **kwargs):
        """HF fallback with the same K3 media framing as the GPU path."""
        input_text = text[0] if isinstance(text, list) else text
        if not images:
            return self._hf_processor(text=[input_text], **kwargs)

        image_sizes = [_get_image_dimensions(image) for image in images]
        image_token_counts = [
            self._hf_processor.media_processor.media_tokens_calculator(
                {"type": "image", "image": image}
            )
            for image in images
        ]
        expanded_text = _expand_k3_image_prompt_text(
            input_text,
            self._image_token,
            image_token_counts,
            image_sizes,
        )
        kwargs["medias"] = [{"type": "image", "image": image} for image in images]
        out = self._hf_processor(text=[expanded_text], **kwargs)
        out["input_ids"] = self._prepare_input_ids(
            input_text,
            [{"num_tokens": count} for count in image_token_counts],
            original_input_ids,
            image_sizes,
        )
        grid_thws = out.pop("grid_thws", None)
        if grid_thws is not None:
            out["image_grid_thw"] = grid_thws
        return out


class KimiK3ImageProcessor(KimiGridMMDataMixin, SGLangBaseProcessor):
    models = [KimiK3ForConditionalGeneration, KimiK31ForConditionalGeneration]
    gpu_image_decode = True
    prefer_tokenized_input = True
    precompute_hash_before_cpu_transfer = True
    auto_mm_processor_worker_num = 2
    auto_mm_io_worker_num = 16
    supports_mm_processor_concurrency = True
    preserve_processor_input_ids = True
    # K3 composites RGBA onto the checkpoint-configured background instead of
    # dropping alpha, so the encoder's decode stage must keep the 4th channel;
    # the compositing itself runs in the GPU wrapper above, or inside the HF
    # processor (``transparent_bg_fill_stage``) on the CPU preprocessing path.
    keep_alpha_channel = True

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        mm_tokens = MultimodalSpecialTokens(
            image_token="<|media_pad|>",
            image_token_id=hf_config.media_placeholder_token_id,
            image_token_regex=re.compile(r"(?:<\|media_pad\|>)+"),
        ).build(_processor)

        media_proc_cfg = _processor.media_processor.media_proc_cfg

        processor = KimiK3GPUProcessorWrapper(
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
            transparent_bg_config=media_proc_cfg.get("transparent_bg_config"),
        )
        super().__init__(hf_config, server_args, processor, *args, **kwargs)
        self.mm_tokens = mm_tokens
        self.IM_TOKEN_ID = mm_tokens.image_token_id

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes, Dict]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        if kwargs.get("audio_data"):
            raise ValueError("Kimi-K3 does not support audio input")
        if has_valid_data(request_obj.video_data):
            # This local path only ever sees images; reaching here means the
            # request was never dispatched to an encoder.
            raise ValueError(
                "Kimi-K3 video input requires the disaggregated encode server"
            )

        expected_image_count = len(image_data or [])
        if self.validate_tokenized_image_placeholders(
            input_text, self.mm_tokens.image_token_id, expected_image_count
        ):
            # Keep structural media tokens distinct from user text that happens to
            # spell ``<|media_pad|>``. Decoding the whole prompt and matching the
            # resulting string would lose that distinction and could bind an image
            # to user-provided text instead of the renderer-inserted token.
            base_output = await self.fast_load_mm_data(
                prompt=input_text,
                image_data=image_data,
                multimodal_tokens=self.mm_tokens,
                discard_alpha_channel=False,
            )
        else:
            base_output = await self.load_mm_data(
                prompt=input_text,
                image_data=image_data,
                multimodal_tokens=self.mm_tokens,
                discard_alpha_channel=False,
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

        # K3's tower is unconditionally image-wise data-parallel (each image
        # is consumed by exactly one TP rank), so keep IPC proxies lazy until
        # that assignment is known: one tokenizer/scheduler crossing per
        # image instead of one per rank. K2.5 gates this on
        # --mm-enable-dp-encoder; K3 needs no flag.
        if getattr(self, "use_cuda_ipc", False):
            for item in mm_items:
                item.model_specific_data[DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY] = (
                    True
                )

        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
            im_token_id=self.mm_tokens.image_token_id,
        )

    def expand_input_ids_from_image_sizes(
        self, input_ids: List[int], image_sizes: List[tuple[int, int]]
    ) -> List[int]:
        """Rebuild the expanded prompt from the original image sizes alone.

        The K3 media contract -- framing control tokens plus one placeholder per
        NaViT feature -- is fully determined by each image's uploaded
        ``(width, height)``.  A node that never runs the vision tower (the decode
        node of a PD-disaggregated deployment) can therefore reproduce the
        prefill node's ``origin_input_ids`` exactly, which is what its KV cache
        allocation is sized from.
        """
        image_sizes = [tuple(size) for size in image_sizes]
        return (
            _expand_k3_media_prompt_token_ids(
                input_ids,
                self.mm_tokens.image_token_id,
                self._processor.image_token_counts_from_sizes(image_sizes),
                image_sizes,
                self._tokenizer,
            )
            .flatten()
            .tolist()
        )

    @staticmethod
    def compute_num_tokens_per_item(model, mm_item) -> int:
        """Number of LLM-visible tokens a single mm_item produces.

        K3's projector merges each ``merge_kernel_size`` patch block into one
        token, so grid ``[t, h, w]`` yields ``(h // merge_h) * (w // merge_w)``.
        """
        from sglang.srt.disaggregation.encode_utils import get_image_grid_dim

        grid_dim = get_image_grid_dim(mm_item)
        if not isinstance(grid_dim, torch.Tensor):
            grid_dim = torch.tensor(grid_dim)
        if grid_dim.dim() == 1:
            grid_dim = grid_dim.unsqueeze(0)

        merge_h, merge_w = model.config.vision_config.merge_kernel_size
        return sum(
            (int(g[1]) // merge_h) * (int(g[2]) // merge_w) for g in grid_dim
        )

    # Video hooks the encode server resolves through the architecture registry
    # (disaggregation/gpu_processor_registry.py), so they must be statics. K3's
    # released vision processor has no video branch, but the frame math is
    # identical to K2.5's, so it is reused. Decoded frames are RGB, so K3's
    # image-only transparent-background compositing has nothing to do here.

    @staticmethod
    def create_video_preprocess_config(
        hf_processor, device: torch.device
    ) -> KimiK25VideoPreprocessConfig:
        """Read the checkpoint's video knobs into a GPU-ready config."""
        return KimiK25VLImageProcessor.create_video_preprocess_config(
            hf_processor, device
        )

    @staticmethod
    def gpu_preprocess_video_chunk(
        frames: torch.Tensor, config: KimiK25VideoPreprocessConfig
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resize, normalize and patchify one temporal chunk of frames."""
        return KimiK25VLImageProcessor.gpu_preprocess_video_chunk(
            frames=frames, config=config
        )

    @staticmethod
    def get_video_compute_weight(config: KimiK25VideoPreprocessConfig) -> int:
        """Per-chunk compute multiplier for the encoder's free_tokens scheduler."""
        return KimiK25VLImageProcessor.get_video_compute_weight(config)

    @staticmethod
    def get_temporal_merge_kernel_size(
        config: KimiK25VideoPreprocessConfig,
    ) -> int:
        """Number of frames the vision tower pools into one temporal chunk."""
        return KimiK25VLImageProcessor.get_temporal_merge_kernel_size(config)

    @staticmethod
    def compute_frame_sampling(
        num_frames: int, avg_fps: float, sample_fps: float
    ) -> List[int]:
        """Uniform frame indices to decode for the requested sample rate."""
        return KimiK25VLImageProcessor.compute_frame_sampling(
            num_frames, avg_fps, sample_fps
        )

    @staticmethod
    def compute_chunk_timestamps(
        frame_indices: List[int],
        avg_fps: float,
        tmks: int,
        timestamp_mode: str = "mm:ss",
    ) -> List[str]:
        """Render one framed media prompt per temporal chunk.

        The only place K3's video prompt contract is expressed. K3 publishes no
        video framing of its own, so this provisionally follows K2.5's format;
        change it here alone once the contract is confirmed.
        """
        return KimiK25VLImageProcessor.compute_chunk_timestamps(
            frame_indices, avg_fps, tmks, timestamp_mode
        )

    def get_mm_data(self, prompt, embeddings, **kwargs):
        img_grid_thw = kwargs.get("img_grid_thw", None)
        output = self._build_kimi_mm_data_from_grids(
            prompt=prompt,
            embeddings=embeddings,
            image_token_id=self.mm_tokens.image_token_id,
            img_grid_thw=img_grid_thw,
        )
        image_sizes = kwargs.get("original_image_sizes")
        if image_sizes is None:
            # Video-only: the encode server already framed every chunk.
            return output

        counts = [self._num_image_tokens_from_grid(grid) for grid in img_grid_thw]
        output.input_ids = (
            _expand_k3_media_prompt_token_ids(
                prompt,
                self.mm_tokens.image_token_id,
                counts,
                [tuple(size) for size in image_sizes],
                self._tokenizer,
            )
            .flatten()
            .tolist()
        )

        search_start = 0
        for item, count in zip(output.mm_items, counts):
            start = output.input_ids.index(self.mm_tokens.image_token_id, search_start)
            item.offsets = [(start, start + count - 1)]
            search_start = start + count
        return output
