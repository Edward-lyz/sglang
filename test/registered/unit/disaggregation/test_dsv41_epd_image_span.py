"""V4.1 EPD span arithmetic, checked against the released model's own code.

A V4.1 image does not cost ``patches // merge**2`` LLM tokens the way a
merge-based ViT does, so the generic EPD token accounting does not apply. These
tests pin the two places that have to agree: what the encoder reports as a
token count, and what the prefill-side processor rebuilds as a span.
"""

import math

import pytest
import torch

from sglang.srt.multimodal.deepseek_v41_image_processing import (
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_START,
    image_token_types,
    num_image_tokens,
)

# From DeepSeek-V4.1-Flash config.json vision_config.
PATCH_SIZE = 14
DOWNSAMPLE_RATIO = 3
MAX_IMAGE_TOKENS = 1024


def _llm_grid(height, width):
    """Aligner grid a pixel size reduces to, as the released model derives it."""
    return (
        math.ceil((height // PATCH_SIZE) / DOWNSAMPLE_RATIO),
        math.ceil((width // PATCH_SIZE) / DOWNSAMPLE_RATIO),
    )


@pytest.mark.parametrize(
    "height,width,expected_llm_grid,expected_tokens,expected_rows",
    [
        (224, 224, (6, 6), 44, 36),
        (1024, 768, (25, 18), 477, 450),
        (4000, 300, (95, 7), 762, 665),
    ],
)
def test_span_length_exceeds_aligner_rows_by_the_delimiters(
    height, width, expected_llm_grid, expected_tokens, expected_rows
):
    """The span carries one IMAGE_NEW_LINE per row plus IMAGE_START/IMAGE_END."""
    n_llm_h, n_llm_w = _llm_grid(height, width)
    assert (n_llm_h, n_llm_w) == expected_llm_grid

    span = num_image_tokens(n_llm_h, n_llm_w)
    assert span == expected_tokens
    assert span <= MAX_IMAGE_TOKENS

    types = image_token_types(n_llm_h, n_llm_w)
    assert types.numel() == span
    assert int((types == IMAGE).sum()) == expected_rows == n_llm_h * n_llm_w
    assert int((types == IMAGE_NEW_LINE).sum()) == n_llm_h
    assert int((types == IMAGE_START).sum()) == 1
    assert int((types == IMAGE_END).sum()) == 1


@pytest.mark.parametrize("n_vit_h,n_vit_w", [(16, 16), (73, 54), (285, 21), (1, 1)])
def test_encoder_token_count_matches_the_rebuilt_span(n_vit_h, n_vit_w):
    """What the encoder announces must be what the processor rebuilds.

    Both derive the aligner grid from the patch grid by ceil-division, so a
    mismatch here would desynchronize the embedding rows from input_ids.
    """
    r = DOWNSAMPLE_RATIO
    n_llm_h, n_llm_w = -(-n_vit_h // r), -(-n_vit_w // r)

    assert (n_llm_h, n_llm_w) == (
        math.ceil(n_vit_h / r),
        math.ceil(n_vit_w / r),
    )
    assert num_image_tokens(n_llm_h, n_llm_w) == len(
        image_token_types(n_llm_h, n_llm_w)
    )


def test_grid_layout_is_row_major_with_a_newline_closing_each_row():
    """Row order is what makes span position map to encoder row position."""
    types = image_token_types(2, 3).tolist()
    assert types == [
        IMAGE_START,
        IMAGE,
        IMAGE,
        IMAGE,
        IMAGE_NEW_LINE,
        IMAGE,
        IMAGE,
        IMAGE,
        IMAGE_NEW_LINE,
        IMAGE_END,
    ]


class _StubProcessor:
    """Enough of DeepseekV41ImageProcessor to exercise get_mm_data alone."""

    image_token_id = 129264

    class _HFConfig:
        """Only the field get_mm_data reads."""

        vision_downsample_ratio = DOWNSAMPLE_RATIO

    hf_config = _HFConfig()

    get_mm_data = None  # bound below


def _stub_get_mm_data(prompt, embeddings, **kwargs):
    """Call the real get_mm_data against the stub, with no HF processor loaded."""
    from sglang.srt.multimodal.processors.deepseek_v41 import (
        DeepseekV41ImageProcessor,
    )

    return DeepseekV41ImageProcessor.get_mm_data(
        _StubProcessor(), prompt, embeddings, **kwargs
    )


def test_rebuilt_spans_consume_every_encoder_row_in_order():
    """Each image owns exactly its own contiguous block of encoder rows."""
    from sglang.srt.managers.schedule_batch import Modality

    grids = [[1, 39, 39], [1, 74, 55]]
    spans = [
        num_image_tokens(-(-h // DOWNSAMPLE_RATIO), -(-w // DOWNSAMPLE_RATIO))
        for _, h, w in grids
    ]
    rows = sum(spans)
    embeddings = {Modality.IMAGE: torch.arange(rows * 4).reshape(rows, 4).float()}
    prompt = [1, _StubProcessor.image_token_id, 2, _StubProcessor.image_token_id, 3]

    output = _stub_get_mm_data(prompt, embeddings, img_grid_thw=grids)

    assert len(output.input_ids) == 3 + sum(spans)
    assert [item.offsets[0] for item in output.mm_items] == [
        (1, spans[0]),
        (spans[0] + 2, spans[0] + spans[1] + 1),
    ]
    # Rows are handed out contiguously, so item i owns exactly its own span.
    consumed = 0
    for item, span in zip(output.mm_items, spans):
        assert item.precomputed_embeddings.shape[0] == span
        assert torch.equal(
            item.precomputed_embeddings,
            embeddings[Modality.IMAGE][consumed : consumed + span],
        )
        consumed += span
    assert consumed == rows


def test_placeholder_and_grid_count_mismatch_is_rejected():
    """Fewer placeholders than grids means the prompt and the encode disagree."""
    from sglang.srt.managers.schedule_batch import Modality

    with pytest.raises(ValueError, match="image placeholders"):
        _stub_get_mm_data(
            [1, _StubProcessor.image_token_id, 2],
            {Modality.IMAGE: torch.zeros((10, 4))},
            img_grid_thw=[[1, 39, 39], [1, 39, 39]],
        )


def test_missing_grid_is_rejected():
    """Without a grid the span length is unknowable, so refuse to guess."""
    from sglang.srt.managers.schedule_batch import Modality

    with pytest.raises(ValueError, match="img_grid_thw"):
        _stub_get_mm_data(
            [_StubProcessor.image_token_id], {Modality.IMAGE: torch.zeros((10, 4))}
        )
