"""Render Kimi K3 chat prompts with the vendored official ``encoding_k3``.

K3's tokenizer (``tokenization_kimi``, loaded from the model directory) renders
messages through the ``encoding_k3`` copy shipped next to it, which lags the
official release that matches the reference API's prompt (raw tool-call
argument text, whitespace-only reasoning, ...). The model directory is not ours
to edit, so the tokenizer module's ``build_chat_segments`` is pointed at the
vendored official copy; the tokenizer keeps its own kwargs handling and
segment tokenization.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from sglang.srt.entrypoints.openai import encoding_k3

logger = logging.getLogger(__name__)


def use_official_kimi_k3_encoding(tokenizer: Any) -> None:
    """Make ``tokenizer.apply_chat_template`` render through official ``encoding_k3``.

    Idempotent; a tokenizer whose module does not render through
    ``build_chat_segments`` is left untouched with a warning.
    """
    tokenizer_module = sys.modules[type(tokenizer).__module__]
    if "build_chat_segments" not in vars(tokenizer_module):
        logger.warning(
            "Kimi K3 tokenizer %s does not render through build_chat_segments; "
            "prompts use its shipped rendering.",
            type(tokenizer).__name__,
        )
        return
    if tokenizer_module.build_chat_segments is encoding_k3.build_chat_segments:
        return
    tokenizer_module.build_chat_segments = encoding_k3.build_chat_segments
    logger.info(
        "Kimi K3 tokenizer %s renders through the official encoding_k3.",
        tokenizer_module.__name__,
    )
