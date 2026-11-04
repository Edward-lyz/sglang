"""Kimi-specific grid-based multimodal data helpers.

Shared by KimiVLImageProcessor and KimiK25VLImageProcessor.
"""

from typing import Union

import numpy as np
import torch

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputFormat,
    MultimodalProcessorOutput,
)


class KimiGridMMDataMixin:
    """Mixin providing Kimi-specific grid-based multimodal data helpers.

    Expects the concrete class to supply:
      - self.hf_config  (with vision_config.merge_kernel_size)
      - self._tokenizer (with .encode())
    """

    supports_metadata_only_mm_data = True

    def resolve_image_token_counts(self, images):
        """Kimi's processor is remote-code and does not implement the
        transformers ``_get_num_multimodal_tokens`` convention; use its
        ``media_tokens_calculator`` instead.

        """
        assert images is not None
        media_tokens_calculator = (
            self._processor.media_processor.media_tokens_calculator
        )
        return [
            int(media_tokens_calculator({"type": "image", "image": image}))
            for image in images
        ]

    @staticmethod
    def validate_tokenized_image_placeholders(
        input_ids, image_token_id: int, expected_image_count: int
    ) -> bool:
        """Validate structural image tokens, returning whether IDs were supplied."""
        if not isinstance(input_ids, (list, torch.Tensor)):
            return False

        token_ids = np.asarray(
            (
                input_ids.detach().flatten().cpu()
                if isinstance(input_ids, torch.Tensor)
                else input_ids
            ),
            dtype=np.int64,
        )
        placeholder_count = int(np.count_nonzero(token_ids == image_token_id))
        if placeholder_count != expected_image_count:
            raise ValueError(
                "Kimi image placeholders must map one-to-one to image data: "
                f"expected {expected_image_count}, found {placeholder_count} token(s)"
            )
        return True

    def _num_image_tokens_from_grid(
        self, grid_thw: Union[torch.Tensor, np.ndarray, list, tuple]
    ) -> int:
        """Compute Kimi-style image token count from 2D/3D grid metadata."""
        merge_h, merge_w = self.hf_config.vision_config.merge_kernel_size

        if isinstance(grid_thw, torch.Tensor):
            vals = grid_thw.flatten().tolist()
        elif isinstance(grid_thw, np.ndarray):
            vals = grid_thw.reshape(-1).tolist()
        elif isinstance(grid_thw, (list, tuple)):
            vals = list(np.array(grid_thw).reshape(-1).tolist())
        else:
            raise TypeError(
                f"Unsupported grid type for kimi image tokens: {type(grid_thw)}"
            )

        if len(vals) >= 3:
            _t, h, w = vals[-3], vals[-2], vals[-1]
        elif len(vals) == 2:
            _t, h, w = 1, vals[0], vals[1]
        else:
            raise ValueError(
                f"Invalid grid metadata for kimi image tokens: {vals} "
                "(expected [t,h,w] or [h,w])"
            )

        h, w = int(h), int(w)
        return (h * w) // (merge_h * merge_w)

    def _build_kimi_mm_data_from_grids(
        self, prompt, embeddings, **kwargs
    ) -> MultimodalProcessorOutput:
        image_token_id = kwargs.get("image_token_id", 0)
        img_grid_thw = kwargs.get("img_grid_thw", None)

        if not isinstance(prompt, list):
            prompt = self._tokenizer.encode(prompt)

        image_token_counts = [
            self._num_image_tokens_from_grid(grid) for grid in img_grid_thw
        ]

        input_ids = []
        offsets = []
        img_idx = 0

        for token in prompt:
            if token != image_token_id:
                input_ids.append(token)
                continue

            if img_idx >= len(image_token_counts):
                raise ValueError(
                    "The number of image placeholders exceeds img_grid_thw entries."
                )

            num_tokens = image_token_counts[img_idx]
            start = len(input_ids)
            input_ids.extend([image_token_id] * num_tokens)
            offsets.append((start, len(input_ids) - 1))
            img_idx += 1

        if img_idx != len(image_token_counts):
            raise ValueError(
                "The number of image placeholders does not match img_grid_thw entries."
            )

        if embeddings is None:
            mm_items = [
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    offsets=[(start, end)],
                    precomputed_embeddings=None,
                    format=MultimodalInputFormat.PRECOMPUTED_EMBEDDING,
                )
                for start, end in offsets
            ]
            return MultimodalProcessorOutput(
                input_ids=input_ids,
                mm_items=mm_items,
                im_token_id=image_token_id,
            )

        image_embeddings = embeddings[Modality.IMAGE]
        # Embeddings may arrive as one tensor or a list of per-part tensors
        # (one per pulled EPD task); slice per-item views instead of concatenating.
        embedding_parts = (
            list(image_embeddings)
            if isinstance(image_embeddings, (list, tuple))
            else [image_embeddings]
        )
        mm_items = []
        part_idx = 0
        consumed = 0
        for start, end in offsets:
            num_tokens = end - start + 1
            while consumed >= embedding_parts[part_idx].shape[0]:
                part_idx += 1
                consumed = 0
            embedding_slice = embedding_parts[part_idx][
                consumed : consumed + num_tokens
            ]
            # An item never spans parts: each pulled task covers whole images.
            assert embedding_slice.shape[0] == num_tokens, (
                f"embedding part {part_idx} has "
                f"{embedding_parts[part_idx].shape[0] - consumed} rows left, "
                f"but the item at offset {(start, end)} needs {num_tokens}"
            )
            consumed += num_tokens
            mm_items.append(
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    offsets=[(start, end)],
                    precomputed_embeddings=embedding_slice,
                    format=MultimodalInputFormat.PRECOMPUTED_EMBEDDING,
                )
            )

        return MultimodalProcessorOutput(
            input_ids=input_ids,
            mm_items=mm_items,
            im_token_id=image_token_id,
        )
