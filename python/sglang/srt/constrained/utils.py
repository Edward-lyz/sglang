"""
Utilities for constrained decoding.
"""

from typing import (
    Dict,
    Set,
)
import torch
import xgrammar

from sglang.srt.utils import is_hip


_is_hip = is_hip()
if _is_hip:
    from sgl_kernel import apply_token_bitmask_inplace_cuda
else:
    from sglang.kernels.ops.grammar.bitmask_ops import (
        apply_token_bitmask_inplace_triton,
    )


def is_legacy_structural_tag(obj: Dict) -> bool:
    """
    Test whether an object is a legacy structural tag.

    See `StructuralTagResponseFormat` at `sglang.srt.entrypoints.openai.protocol`.
    """
    if obj.get("structures", None) is not None:
        assert obj.get("triggers", None) is not None
        return True
    else:
        assert obj.get("format", None) is not None
        return False


def allocate_vocab_bitmask(
    batch_size: int, vocab_size: int, device
) -> torch.Tensor:
    """
    Allocate vocabulary bitmask tensor.
    """
    # TODO(liyaolang): make this function independent from xgrammar
    return xgrammar.allocate_token_bitmask(batch_size, vocab_size)


def fill_vocab_bitmask_with_valid_tokens(vocab_mask: torch.Tensor, idx: int, valid_tokens: Set[int]) -> None:
    """
    Fill vocabulary bitmask with valid tokens.
    """
    for token in valid_tokens:
        vocab_mask[idx, token // 32] |= (1 << (token % 32))


def fill_vocab_bitmask_with_invalid_tokens(vocab_mask: torch.Tensor, idx: int, invalid_tokens: Set[int]) -> None:
    """
    Fill vocabulary bitmask with invalid tokens.
    """
    for token in invalid_tokens:
        vocab_mask[idx, token // 32] &= ~(1 << (token % 32))


def apply_vocab_bitmask(logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
    """
    Apply vocabulary bitmask to logits.
    """
    if (
        logits.device.type == "cuda"
        or logits.device.type == "npu"
        or logits.device.type == "xpu"
    ):
        if _is_hip:
            apply_token_bitmask_inplace_cuda(logits, vocab_mask)
        else:
            apply_token_bitmask_inplace_triton(logits, vocab_mask)
    else:
        raise RuntimeError(f"Unsupported device: {logits.device.type}")


def move_vocab_bitmask(vocab_mask: torch.Tensor, device) -> torch.Tensor:
    """
    Move vocabulary bitmask to device.
    """
    return vocab_mask.to(device, non_blocking=True)
