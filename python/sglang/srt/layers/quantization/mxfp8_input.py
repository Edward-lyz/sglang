"""Explicit input layout for a linear consuming prequantized MXFP8 activations."""

from typing import NamedTuple

import torch


class Mxfp8SwizzledInput(NamedTuple):
    """E4M3 activations and UE8M0 scales in FlashInfer's 128x4 layout.

    A plain FP8 tuple may contain block-FP8 scales with a different layout.
    This marker lets a converted block-FP8 linear distinguish the two.
    """

    data: torch.Tensor
    scales: torch.Tensor


class Mxfp8DeepGemmInput(NamedTuple):
    """E4M3 activations and per-32 UE8M0 scales in DeepGEMM's layout: int32 packed
    [M, ceil4(K/32)/4], MN-major with the token dim aligned to 4.

    Only the DeepGEMM MXFP8 backend of a block-FP8-as-MXFP8 linear accepts it.
    """

    data: torch.Tensor
    scales: torch.Tensor
