# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Fused send-side pack for the a2a DCP reduce (see srt/layers/dcp/comm.py).

The upstream eager path stages the all-to-all payload with four separate
elementwise launches: an fp32 LSE ``contiguous()``, a strided copy of the output
partials, a bit-packed copy of the LSE tail, and after the exchange a copy that
unpacks the LSE back into an fp32 buffer.  On Kimi-K3 decode
(``[B, H, D] = [800, 96, 512]``, ``dcp_size=8``) the output copy alone is 97.5 us
and NCU attributes it to address arithmetic, not bandwidth: SM 74.87 % against
DRAM 11.00 % at 731.88 GB/s, because it writes into a view whose row pitch is
``D + lse_pack_dim`` elements (1028 B) and TensorIterator recomputes the
multi-dimensional offset per element.

This module packs the same bytes with one Triton kernel and two changes that
remove the arithmetic instead of widening the stores:

* each per-rank chunk is laid out as ``[payload block | fp32 lse block]`` rather
  than interleaving the LSE into every row.  For a fixed ``(rank, token)`` the
  source run ``out[b, n * H_per_rank : (n + 1) * H_per_rank, :]`` is then
  ``H_per_rank * D`` contiguous elements mapping onto one contiguous destination
  run, so the payload copy is a permutation of whole 16-byte-aligned head blocks
  and one program handles a whole run;
* the fp32 LSE lands directly in its block through an ``as_strided`` view, so
  both LSE staging copies disappear.

Chunk sizes stay equal across ranks, so ``all_to_all_single`` is unaffected.
Optionally the zero-KV fixup (``fixup_zero_kv_rows``) is folded in as well: each
program resolves its token's request with an in-register ``upper_bound`` search
over ``cum_seq_lens`` and predicates both the staged write and the in-place
writeback on an empty KV shard.

Numerics: the packed bytes are bit-identical to the upstream layout's, but the
combine kernel that consumes them is re-specialised by Triton on the new stride
divisibility and contracts multiply-adds differently, so the combined output can
differ from the upstream path on elements where the N partials nearly cancel.
That is why the caller gates this path behind an environment flag that defaults
to off.
"""

import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dcp_kernels import _lse_pack_dim

# Chosen by a 36-point sweep on B200 / sm_100: the kernel is register limited
# rather than bandwidth limited, so a small tile without pipelining wins, and
# every BLOCK >= 2048 shape is about 9 us slower on the 800-token scope.
_PACK_BLOCK = 1024
_PACK_NUM_WARPS = 4
_PACK_NUM_STAGES = 1


@triton.jit
def _dcp_fused_pack_kernel(
    out_ptr,
    lse_ptr,
    kv_lens_ptr,
    cum_ptr,
    staged_ptr,
    staged_f32_ptr,
    B,
    BS,
    H_PER_RANK: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    CHUNK_ELEMS: tl.constexpr,
    LSE_F32_OFFSET: tl.constexpr,
    RUN: tl.constexpr,
    BLOCK: tl.constexpr,
    HPR_BLOCK: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    HAS_FIXUP: tl.constexpr,
):
    """One program per (rank, token): copy one head-block run and one LSE row."""
    pid = tl.program_id(0)
    n = pid // B
    b = pid - n * B

    is_zero_kv = False
    if HAS_FIXUP:
        # upper_bound(cum_seq_lens, b) - 1: the request that owns token b.
        lo = 0
        hi = BS
        for _ in tl.range(0, SEARCH_STEPS):
            mid = (lo + hi) // 2
            c = tl.load(cum_ptr + mid)
            take = c <= b
            lo = tl.where(take, mid, lo)
            hi = tl.where(take, hi, mid)
        is_zero_kv = tl.load(kv_lens_ptr + lo) == 0

    src_base = b * (N * H_PER_RANK * D) + n * RUN
    dst_base = n * CHUNK_ELEMS + b * RUN

    for start in tl.range(0, RUN, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < RUN
        values = tl.load(out_ptr + src_base + offs, mask=mask, other=0)
        if HAS_FIXUP:
            values = tl.where(is_zero_kv, tl.zeros_like(values), values)
        tl.store(staged_ptr + dst_base + offs, values, mask=mask)
        if HAS_FIXUP:
            # in-place fixup of the caller's tensor, only for empty KV shards
            tl.store(
                out_ptr + src_base + offs,
                tl.zeros_like(values),
                mask=mask & is_zero_kv,
            )

    hoffs = tl.arange(0, HPR_BLOCK)
    hmask = hoffs < H_PER_RANK
    lse_src = lse_ptr + b * (N * H_PER_RANK) + n * H_PER_RANK + hoffs
    lse_values = tl.load(lse_src, mask=hmask, other=0.0)
    if HAS_FIXUP:
        lse_values = tl.where(is_zero_kv, float("-inf"), lse_values)
    tl.store(
        staged_f32_ptr
        + n * (CHUNK_ELEMS // 2)
        + LSE_F32_OFFSET
        + b * H_PER_RANK
        + hoffs,
        lse_values,
        mask=hmask,
    )
    if HAS_FIXUP:
        tl.store(lse_src, lse_values, mask=hmask & is_zero_kv)


def dcp_fused_pack_supported(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    n_ranks: int,
    local_kv_lens: Optional[torch.Tensor] = None,
    cum_seq_lens: Optional[torch.Tensor] = None,
) -> bool:
    """Host-side guard: False means the caller must use the original path."""
    if cp_attn_out.dtype not in (torch.bfloat16, torch.float16):
        return False
    if cp_attn_lse.dtype is not torch.float32:
        return False
    if not (cp_attn_out.is_contiguous() and cp_attn_lse.is_contiguous()):
        return False
    if cp_attn_out.dim() != 3 or cp_attn_lse.dim() != 2:
        return False
    B, H, D = cp_attn_out.shape
    if cp_attn_lse.shape != (B, H):
        return False
    if H % n_ranks != 0 or D % 2 != 0:
        return False
    if _lse_pack_dim(cp_attn_out.dtype) != 2:
        return False
    if (local_kv_lens is None) != (cum_seq_lens is None):
        return False
    if local_kv_lens is not None:
        if (
            local_kv_lens.dtype is not torch.int32
            or cum_seq_lens.dtype is not torch.int32
        ):
            return False
        # The folded fixup covers every token exactly once only if cum_seq_lens
        # tiles [0, B) without holes; the caller's q_indptr always does.
        if int(cum_seq_lens.numel()) != int(local_kv_lens.numel()) + 1:
            return False
    return True


def dcp_fused_pack_chunk_elems(B: int, H_per_rank: int, D: int) -> Tuple[int, int]:
    """(chunk_elems, payload_elems) of the ``[payload | fp32 lse]`` chunk layout.

    ``chunk_elems`` is rounded up to a multiple of 8 so that every chunk starts
    16-byte aligned and the fp32 view of the buffer stays element aligned.
    """
    rows = B * H_per_rank
    payload_elems = rows * D
    chunk_elems = payload_elems + rows * 2
    if chunk_elems % 8:
        chunk_elems += 8 - (chunk_elems % 8)
    return chunk_elems, payload_elems


def dcp_fused_pack(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    n_ranks: int,
    local_kv_lens: Optional[torch.Tensor] = None,
    cum_seq_lens: Optional[torch.Tensor] = None,
    send_buffer: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Stage the a2a payload into ``[N, chunk_elems]`` with one Triton launch.

    Args:
        cp_attn_out: ``[B, H, D]`` local attention partials, contiguous.
        cp_attn_lse: ``[B, H]`` fp32 log-sum-exp, contiguous.
        n_ranks: DCP world size; must divide ``H``.
        local_kv_lens: optional ``[bs]`` int32 per-request local KV lengths.  When
            given together with ``cum_seq_lens`` the zero-KV fixup is folded in
            and ``cp_attn_out`` / ``cp_attn_lse`` are updated in place, so the
            caller must not run ``fixup_zero_kv_rows`` as well.
        cum_seq_lens: optional ``[bs + 1]`` int32 token offsets per request.
        send_buffer: optional pre-allocated ``[N, chunk_elems]`` send buffer, for
            CUDA-graph capture with stable addresses.

    Returns:
        The staged send buffer, ``[N, chunk_elems]`` in ``cp_attn_out.dtype``.
    """
    N = n_ranks
    B, H, D = cp_attn_out.shape
    H_per_rank = H // N
    chunk_elems, payload_elems = dcp_fused_pack_chunk_elems(B, H_per_rank, D)

    if send_buffer is None:
        staged = torch.empty(
            N, chunk_elems, dtype=cp_attn_out.dtype, device=cp_attn_out.device
        )
    else:
        assert send_buffer.shape == (N, chunk_elems), (
            f"send_buffer {tuple(send_buffer.shape)} does not match the required "
            f"chunk layout {(N, chunk_elems)}"
        )
        staged = send_buffer

    has_fixup = local_kv_lens is not None
    bs = int(local_kv_lens.numel()) if has_fixup else 1
    _dcp_fused_pack_kernel[(N * B,)](
        cp_attn_out.view(-1),
        cp_attn_lse.view(-1),
        local_kv_lens,
        cum_seq_lens,
        staged,
        staged.view(torch.float32),
        B,
        bs,
        H_PER_RANK=H_per_rank,
        D=D,
        N=N,
        CHUNK_ELEMS=chunk_elems,
        LSE_F32_OFFSET=payload_elems // 2,
        RUN=H_per_rank * D,
        BLOCK=_PACK_BLOCK,
        HPR_BLOCK=triton.next_power_of_2(H_per_rank),
        SEARCH_STEPS=int(math.ceil(math.log2(max(bs, 2)))) + 1,
        HAS_FIXUP=has_fixup,
        num_warps=_PACK_NUM_WARPS,
        num_stages=_PACK_NUM_STAGES,
    )
    return staged


def dcp_fused_pack_recv_views(
    recv: torch.Tensor,
    n_ranks: int,
    B: int,
    H_per_rank: int,
    D: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Views of a received ``[N, chunk_elems]`` buffer for the combine kernel.

    No copy happens here: the payload becomes ``[N, B, H_per_rank, D]`` and the
    fp32 LSE block becomes ``[N, B, H_per_rank]``, which is what removes the
    upstream unpack launch.
    """
    chunk_elems, payload_elems = dcp_fused_pack_chunk_elems(B, H_per_rank, D)
    recv_output = torch.as_strided(
        recv,
        (n_ranks, B, H_per_rank, D),
        (chunk_elems, H_per_rank * D, D, 1),
        recv.storage_offset(),
    )
    recv_f32 = recv.view(torch.float32)
    recv_lse = torch.as_strided(
        recv_f32,
        (n_ranks, B, H_per_rank),
        (chunk_elems // 2, H_per_rank, 1),
        recv_f32.storage_offset() + payload_elems // 2,
    )
    return recv_output, recv_lse
