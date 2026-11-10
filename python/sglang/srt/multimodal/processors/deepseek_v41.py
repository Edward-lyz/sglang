"""260903 image preprocessing, preserving raw token IDs for Engram."""

import asyncio
import logging
from functools import partial
from typing import List, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.srt.multimodal.deepseek_v41_image_processing import (
    image_token_types,
    load_image,
    load_image_rust,
    prepare_image_gpu,
)
from sglang.srt.multimodal.task_id import media_task_hash
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.runtime_context import get_mm, get_serving
from sglang.srt.rust_extensions import load_rust_extension

logger = logging.getLogger(__name__)


def media_url_of(media) -> Optional[str]:
    """Where a data item's media came from: an ImageData url, a dict url, or a url."""
    if isinstance(media, dict):
        media = media.get("url")
    else:
        media = getattr(media, "url", media)
    return media if isinstance(media, str) else None


class DeepseekV41ImageProcessor(BaseMultimodalProcessor):
    models = [DeepseekV4ForCausalLM]
    preserve_processor_input_ids = True
    prefer_tokenized_input = True
    gpu_image_decode = False

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        self.image_backend = "cpu"
        self.cpu_image_loader = load_image
        backend = get_mm().image_processor_backend
        if backend == "pil" or get_mm().disable_fast_image_processor:
            self.image_backend = "cpu"
        elif envs.SGLANG_ENCODER_IMAGE_PROCESSOR_USE_GPU.get():
            self.image_backend = "gpu"
        elif backend == "auto":
            # Resolve the optional extension once, before the base class builds
            # the preprocessing cache fingerprint. Image-processing errors are
            # not caught here or retried with a different backend.
            try:
                extension = load_rust_extension(
                    "sglang.srt.rust_extensions._multimodal"
                )
                resize_patchify = extension.dsv41.resize_patchify
            except (ImportError, OSError, RuntimeError, AttributeError) as error:
                logger.warning(
                    "V4.1 Rust image processor unavailable; using PIL: %s", error
                )
            else:
                self.image_backend = "rust"
                self.cpu_image_loader = partial(
                    load_image_rust, resize_patchify=resize_patchify
                )
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.image_token_id = hf_config.image_token_id
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=_processor.convert_ids_to_tokens(self.image_token_id),
            image_token_id=self.image_token_id,
        ).build(_processor)

    def preprocess_fingerprint_payload(self):
        payload = super().preprocess_fingerprint_payload()
        payload["dsv41_image_backend"] = self.image_backend
        return payload

    async def process_mm_data_async(
        self,
        image_data,
        input_text,
        request_obj,
        image_urls: Optional[List[str]] = None,
        *args,
        **kwargs,
    ):
        base = await self.load_mm_data(
            input_text, image_data=image_data, multimodal_tokens=self.mm_tokens
        )
        ids = (
            input_text
            if isinstance(input_text, list)
            else self._processor.encode(input_text)
        )
        if ids.count(self.image_token_id) != len(base.images):
            raise ValueError("Image placeholders and images must match")
        images = iter(base.images)
        urls = iter(image_urls or [])
        tokens, items = [], []
        for token in ids:
            if token != self.image_token_id:
                tokens.append(token)
                continue
            image = next(images)
            url = media_url_of(next(urls, None))
            plan = None
            if self.image_backend == "gpu":
                patches, plan, lh, lw = await asyncio.to_thread(
                    prepare_image_gpu, image, self.hf_config
                )
                h = plan["height"] // plan["patch_size"]
                w = plan["width"] // plan["patch_size"]
            else:
                patches, h, w, lh, lw = await asyncio.to_thread(
                    self.cpu_image_loader, image, self.hf_config
                )
            if self.keep_mm_features_on_device:
                patches = patches.to(torch.device("cuda", self.server_args.base_gpu_id))
            metadata = {"n_vit_h": h, "n_vit_w": w}
            if plan is not None:
                metadata["dsv41_gpu_plan"] = plan
            count = len(image_token_types(lh, lw))
            start = len(tokens)
            tokens.extend([self.image_token_id] * count)
            item = MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=patches,
                offsets=[(start, start + count - 1)],
                model_specific_data=metadata,
            )
            # The pad value follows the media url, so a rank that never holds
            # the encoded tensor derives the same one. Media without a url
            # keeps the feature hash; a shared empty url would collide.
            if url is not None:
                item.set_hash(
                    media_task_hash(
                        model_id=get_serving().served_model_name or "",
                        media_url=url,
                    )
                )
            items.append(item)
        return MultimodalProcessorOutput(
            input_ids=tokens,
            mm_items=self._prepare_mm_items_for_transport(items),
            im_token_id=self.image_token_id,
        )

    def span_lengths(self, grids) -> List[int]:
        """Prompt span length of each image grid ``[t, n_vit_h, n_vit_w]``."""
        r = self.hf_config.vision_downsample_ratio
        lengths = []
        for _, n_vit_h, n_vit_w in torch.as_tensor(grids).reshape(-1, 3).tolist():
            lengths.append(len(image_token_types(-(-n_vit_h // r), -(-n_vit_w // r))))
        return lengths

    def get_mm_data(self, prompt, embeddings, **kwargs):
        """Rebuild the image spans for embeddings received from an encoder.

        The base implementation derives a span length from a patch grid, which
        does not describe V4.1: an image occupies
        ``n_llm_h * (n_llm_w + 1) + 2`` tokens -- the aligner grid in reading
        order plus one IMAGE_NEW_LINE per row and the two delimiters. The
        encoder returns that whole span (see
        ``DeepseekV4ForCausalLM.get_image_feature``), so span length and
        embedding rows still match one-to-one here.
        """
        grid = kwargs.get("img_grid_thw")
        if grid is None:
            raise ValueError("DeepSeek-V4.1 EPD requires img_grid_thw per image")
        grid = torch.as_tensor(grid).reshape(-1, 3)

        ids = prompt if isinstance(prompt, list) else self._processor.encode(prompt)
        num_placeholders = ids.count(self.image_token_id)
        if num_placeholders != grid.shape[0]:
            raise ValueError(
                f"{num_placeholders} image placeholders for {grid.shape[0]} "
                "encoder grids"
            )

        r = self.hf_config.vision_downsample_ratio
        image_embeddings = embeddings[Modality.IMAGE]
        tokens, items, consumed, index = [], [], 0, 0
        for token in ids:
            if token != self.image_token_id:
                tokens.append(token)
                continue
            _, n_vit_h, n_vit_w = (int(value) for value in grid[index])
            n_llm_h = -(-n_vit_h // r)
            n_llm_w = -(-n_vit_w // r)
            count = len(image_token_types(n_llm_h, n_llm_w))
            start = len(tokens)
            tokens.extend([self.image_token_id] * count)
            items.append(
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    offsets=[(start, start + count - 1)],
                    precomputed_embeddings=image_embeddings[
                        consumed : consumed + count
                    ],
                    model_specific_data={"n_vit_h": n_vit_h, "n_vit_w": n_vit_w},
                )
            )
            consumed += count
            index += 1

        return MultimodalProcessorOutput(
            input_ids=tokens,
            mm_items=items,
            im_token_id=self.image_token_id,
        )
