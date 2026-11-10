"""MX2 routed-expert helpers (in-GEMM decode path).

The checkpoint stores each weight row as ``[K/4 bytes code | K/32 bytes palette indices]``
(``9K/32`` bytes per row, index format) or, for checkpoints written before the index
format, ``[K/4 code | K/16 palette]`` (``5K/16``, palette format). The MX2 grouped GEMMs
(``deep_gemm.m_grouped_fp8_mx2_gemm_nt_masked`` / ``_contiguous``) want the two parts split
and re-ordered so that one (expert, K-block) tile is one contiguous read:

    code  [E, K/128, N, 32]
    index [E, K/128, N, 4]     (palette [E, K/128, N, 8] for the palette format)

This is a pure byte shuffle: the 2-bit payload is not touched and no expanded FP4
weight is produced.
"""

from __future__ import annotations

import torch

# `w2_format` values in the checkpoint's config.json. Existing MX2 checkpoints carry
# these strings, so they keep their original spelling.
MX2_FORMAT_INDEX = "w2s_index_v1"
MX2_FORMAT_PALETTE = "w2s_palette_v1"


def mx2_row_bytes(logical_k: int, index_format: bool) -> int:
    """Physical bytes per weight row of an MX2 checkpoint with logical ``logical_k``."""
    return logical_k // 4 + (logical_k // 32 if index_format else logical_k // 16)


def reorder_mx2_to_runtime_layout(
    weight: torch.Tensor, logical_k: int, index_format: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split/re-order one expert stack, e.g. [112, 6144, 2016] uint8 (index format).

    Returns (code [E, K/128, N, 32], index [E, K/128, N, 4]) or, with ``index_format=False``,
    (code, palette [E, K/128, N, 8]).
    """
    e, n, row_bytes = weight.shape
    expected = mx2_row_bytes(logical_k, index_format)
    assert row_bytes == expected, (
        f"unexpected MX2 row width: {row_bytes} bytes for K={logical_k} "
        f"({'index' if index_format else 'palette'} format expects {expected})"
    )
    k_blocks = logical_k // 128
    code_bytes = logical_k // 4
    side_bytes = row_bytes - code_bytes
    assert weight.is_contiguous()

    code = (
        weight[:, :, :code_bytes]
        .reshape(e, n, k_blocks, 32)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    side = (
        weight[:, :, code_bytes : code_bytes + side_bytes]
        .reshape(e, n, k_blocks, side_bytes // k_blocks)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    return code, side


def mx2_runtime_weights(
    weight: torch.Tensor, logical_k: int, index_format: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Checkpoint rows -> (code, index) in the layout and bit order of the MX2 grouped GEMMs.

    An index checkpoint has as many bytes per row as code and index together, so both are written
    back into ``weight``'s own storage. The long-lived tensors then never sit in a block left by the
    temporaries, which would keep the rest of that block away from the driver and the KV pool. A
    palette checkpoint's rows are larger: its code and index are new tensors, the rows are dropped.
    """
    import deep_gemm

    code, side = reorder_mx2_to_runtime_layout(weight, logical_k, index_format)
    if not index_format:
        side = deep_gemm.mx2_palette_to_index(side)
    code = deep_gemm.mx2_permute_code(code)
    if not index_format:
        return code, side
    flat = weight.view(-1)
    n = code.numel()
    assert n + side.numel() == flat.numel()
    flat[:n].copy_(code.view(-1))
    flat[n:].copy_(side.view(-1))
    return flat[:n].view(code.shape), flat[n:].view(side.shape)
