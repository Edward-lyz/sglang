"""Tokenize one long prompt across Rayon threads with the exact single-call ids.

A 120k-character agent prompt takes ~100 ms in one ``tokenizer.encode`` call,
on the request's critical path in the tokenizer worker. The pre-tokenizers of
byte-level BPE tokenizers (GPT/DeepSeek style) never merge a newline with the
non-space character after it, so cutting the text right after a ``"\\n"`` that
is followed by a non-space character yields pieces whose ids concatenate to the
ids of the whole text. ``encode_batch`` then encodes the pieces in parallel.

The property is checked once per tokenizer on a probe text; tokenizers that add
special tokens in ``encode`` or that fail the probe keep the plain call.
"""

import logging
import os
from typing import List

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Covers digits, CJK, punctuation runs ending in newlines, blank lines, indents
# and tabs -- the patterns a pre-tokenizer regex could join across a newline.
_PROBE = (
    "def f(x):\n    return x + 1234567\n\n\n# 注释：中文与English混排。\n"
    'a.\n\nb,\n c\n\td\r\ne!!\n\n  \n123\n456\n<tag>\n{"k": [1, 2]}\n'
    "第一行\n第二行\n。\n——\nend"
) * 40

_checked = {}


def _split_points(text: str, num_chunks: int) -> List[int]:
    """Cut offsets right after a newline that a non-space character follows."""
    length = len(text)
    cuts = [0]
    for j in range(1, num_chunks):
        pos = text.find("\n", max(length * j // num_chunks, cuts[-1]))
        while pos != -1 and pos + 1 < length and text[pos + 1].isspace():
            pos = text.find("\n", pos + 1)
        if pos == -1 or pos + 1 >= length:
            break
        cuts.append(pos + 1)
    cuts.append(length)
    return cuts


def _encode_split(backend, text: str, num_chunks: int) -> List[int]:
    """Encode the newline-cut pieces in one parallel batch and join the ids."""
    _reset_padding_truncation(backend)
    cuts = _split_points(text, num_chunks)
    pieces = [text[a:b] for a, b in zip(cuts, cuts[1:])]
    # encode_batch_fast skips the char offsets nothing here reads (~15% of
    # the batch time on long prompts); same ids.
    encode_batch = getattr(backend, "encode_batch_fast", backend.encode_batch)
    ids = []
    for encoding in encode_batch(pieces, add_special_tokens=False):
        ids.extend(encoding.ids)
    return ids


def _added_tokens_are_safe(tokenizer) -> bool:
    """Whether no added token can join or span a newline cut.

    An added token with lstrip/rstrip absorbs the neighbouring whitespace (so a
    cut right after "\n" would change its ids), and one whose content holds a
    newline could be cut in half.
    """
    decoder = getattr(tokenizer, "added_tokens_decoder", None) or {}
    for tok in decoder.values():
        content = getattr(tok, "content", str(tok))
        if (
            getattr(tok, "lstrip", False)
            or getattr(tok, "rstrip", False)
            or "\n" in content
        ):
            return False
    return True


def _reset_padding_truncation(backend) -> None:
    """Clear padding/truncation left on the Rust tokenizer by an earlier call.

    ``tokenizer.encode`` resets both on every call (HF's
    ``set_truncation_and_padding``), ``encode_batch`` on the backend does not;
    after a processor call with ``padding=True`` on the shared tokenizer every
    piece would be padded to the longest one. Same reset HF does itself.
    """
    if getattr(backend, "padding", None) is not None:
        backend.no_padding()
    if getattr(backend, "truncation", None) is not None:
        backend.no_truncation()


def _split_is_exact(tokenizer) -> bool:
    """Whether split encoding reproduces ``tokenizer.encode`` for this tokenizer."""
    key = id(tokenizer)
    if key not in _checked:
        backend = getattr(tokenizer, "_tokenizer", None)
        ok = False
        if (
            backend is not None
            and hasattr(backend, "encode_batch")
            and _added_tokens_are_safe(tokenizer)
        ):
            specials = "".join(
                f"\n{t}" for t in getattr(tokenizer, "all_special_tokens", []) or []
            )
            probe = _PROBE + specials
            try:
                ok = tokenizer.encode(probe) == _encode_split(backend, probe, 64)
            except Exception:
                logger.exception("Parallel tokenize probe failed; keeping encode()")
        if not ok:
            logger.info("Parallel tokenize disabled for %s", type(tokenizer).__name__)
        _checked[key] = ok
    return _checked[key]


def encode_prompt(tokenizer, text: str) -> List[int]:
    """``tokenizer.encode(text)``, split across threads when the prompt is long.

    Prompts shorter than ``SGLANG_PARALLEL_TOKENIZE_MIN_CHARS`` (0 disables)
    take the plain call. The pieces only run in parallel while
    ``TOKENIZERS_PARALLELISM`` is not "false" (``TokenizerManager`` keeps it on
    when this is enabled).
    """
    min_chars = envs.SGLANG_PARALLEL_TOKENIZE_MIN_CHARS.get()
    if min_chars <= 0 or len(text) < min_chars or not _split_is_exact(tokenizer):
        return tokenizer.encode(text)
    num_chunks = max(
        1, min(envs.SGLANG_PARALLEL_TOKENIZE_CHUNKS.get(), len(text) // 4096)
    )
    return _encode_split(tokenizer._tokenizer, text, num_chunks)


def parallel_tokenize_enabled() -> bool:
    """Whether long prompts are tokenized in parallel."""
    return envs.SGLANG_PARALLEL_TOKENIZE_MIN_CHARS.get() > 0


def configure_tokenizer_parallelism() -> None:
    """Set the HF tokenizers parallelism env for a tokenizer process.

    Multimodal tokenizer processes used to force it off. With parallel
    tokenize on it stays on, and the Rayon pool (created lazily at first use)
    is capped at ``SGLANG_PARALLEL_TOKENIZE_THREADS`` per process.
    """
    if parallel_tokenize_enabled():
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        os.environ.setdefault(
            "RAYON_NUM_THREADS", str(envs.SGLANG_PARALLEL_TOKENIZE_THREADS.get())
        )
    else:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
