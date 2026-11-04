# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.3.post1/vllm/model_executor/layers/vocab_parallel_embedding.py

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from torch.nn.parameter import Parameter, UninitializedParameter

from sglang.kernels.ops.embeddings.vocab_parallel_embedding import (
    vocab_parallel_embedding as fused_vocab_parallel_embedding,
)
from sglang.srt.distributed import (
    divide,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.layers.amx_utils import PackWeightMethod
from sglang.srt.layers.communicator import get_attn_tp_context
from sglang.srt.layers.dp_attention import (
    attn_tp_all_reduce,
    is_allocation_symmetric,
    is_dp_attention_enabled,
)
from sglang.srt.layers.parameter import BasevLLMParameter
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
    method_has_implemented_embedding,
)
from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_compiler_backend,
    is_cpu,
    is_npu,
    set_weight_attrs,
)
from sglang.srt.utils.async_probe import maybe_detect_oob

DEFAULT_VOCAB_PADDING_SIZE = 64

_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_is_npu = is_npu()

logger = logging.getLogger(__name__)


def pad_vocab_size(vocab_size: int, pad_to: int = DEFAULT_VOCAB_PADDING_SIZE) -> int:
    """Pad the vocab size to the given value."""
    return ((vocab_size + pad_to - 1) // pad_to) * pad_to


def vocab_range_from_per_partition_vocab_size(
    per_partition_vocab_size: int, rank: int, offset: int = 0
) -> Sequence[int]:
    index_f = rank * per_partition_vocab_size
    index_l = index_f + per_partition_vocab_size
    return index_f + offset, index_l + offset


def vocab_range_from_global_vocab_size(
    global_vocab_size: int, rank: int, world_size: int, offset: int = 0
) -> Sequence[int]:
    per_partition_vocab_size = divide(global_vocab_size, world_size)
    return vocab_range_from_per_partition_vocab_size(
        per_partition_vocab_size, rank, offset=offset
    )


@dataclass
class VocabParallelEmbeddingShardIndices:
    """Indices for a shard of a vocab parallel embedding."""

    padded_org_vocab_start_index: int
    padded_org_vocab_end_index: int
    padded_added_vocab_start_index: int
    padded_added_vocab_end_index: int

    org_vocab_start_index: int
    org_vocab_end_index: int
    added_vocab_start_index: int
    added_vocab_end_index: int

    @property
    def num_org_elements(self) -> int:
        return self.org_vocab_end_index - self.org_vocab_start_index

    @property
    def num_added_elements(self) -> int:
        return self.added_vocab_end_index - self.added_vocab_start_index

    @property
    def num_org_elements_padded(self) -> int:
        return self.padded_org_vocab_end_index - self.padded_org_vocab_start_index

    @property
    def num_added_elements_padded(self) -> int:
        return self.padded_added_vocab_end_index - self.padded_added_vocab_start_index

    @property
    def num_org_vocab_padding(self) -> int:
        return self.num_org_elements_padded - self.num_org_elements

    @property
    def num_added_vocab_padding(self) -> int:
        return self.num_added_elements_padded - self.num_added_elements

    @property
    def num_elements_padded(self) -> int:
        return self.num_org_elements_padded + self.num_added_elements_padded

    def __post_init__(self):
        # sanity checks
        assert self.padded_org_vocab_start_index <= self.padded_org_vocab_end_index
        assert self.padded_added_vocab_start_index <= self.padded_added_vocab_end_index

        assert self.org_vocab_start_index <= self.org_vocab_end_index
        assert self.added_vocab_start_index <= self.added_vocab_end_index

        assert self.org_vocab_start_index <= self.padded_org_vocab_start_index
        assert self.added_vocab_start_index <= self.padded_added_vocab_start_index
        assert self.org_vocab_end_index <= self.padded_org_vocab_end_index
        assert self.added_vocab_end_index <= self.padded_added_vocab_end_index

        assert self.num_org_elements <= self.num_org_elements_padded
        assert self.num_added_elements <= self.num_added_elements_padded


@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)
def get_masked_input_and_mask(
    input_: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # torch.compile will fuse all of the pointwise ops below
    # into a single kernel, making it very fast
    org_vocab_mask = (input_ >= org_vocab_start_index) & (input_ < org_vocab_end_index)
    added_vocab_mask = (input_ >= added_vocab_start_index) & (
        input_ < added_vocab_end_index
    )
    added_offset = (
        added_vocab_start_index
        - (org_vocab_end_index - org_vocab_start_index)
        - num_org_vocab_padding
    )
    valid_offset = (org_vocab_start_index * org_vocab_mask) + (
        added_offset * added_vocab_mask
    )
    vocab_mask = org_vocab_mask | added_vocab_mask
    input_ = vocab_mask * (input_ - valid_offset)
    return input_, ~vocab_mask


def get_embedding_tp_kwargs() -> dict:
    """Vocab-parallel layout kwargs for the *input embedding* of models that
    support embedding replication (the DeepSeek-V2 target family: DeepSeek
    V3.1 / Kimi K2.5, plus their EAGLE3 / NextN drafts).

    EAGLE / NextN share the target's ``embed_tokens.weight`` tensor with the
    draft (``set_embed`` / ``set_embed_and_head``), so the target and every
    draft that shares it MUST use the same vocab-parallel layout -- otherwise
    the draft's masking/index math runs against a tensor with a different
    layout and accept_len silently drops. Route all of them through this one
    helper so they can never drift.
    """
    if envs.SGLANG_ENABLE_EMBED_REPLICATION.get():
        # Replicate the full table on every rank: skips the embed all-reduce
        # at the cost of duplicated embedding weights.
        return {"enable_tp": False}
    if envs.SGLANG_ENABLE_EMBED_DP_SHARD.get() and is_dp_attention_enabled():
        # Shard over the FULL TP group even under DP attention: the ranks of
        # that group hold different tokens, so every caller must wrap the
        # lookup in the DP row gather/scatter (see dp_sharded_embed).
        # SGLANG_VOCAB_TP_SIZE can narrow that group to an intra-node block.
        return {"enable_tp": True, "use_attn_tp_group": False}
    # Shard along the vocab dim. Under DP attention each rank owns only its
    # local tokens, so reduce within the attention-TP group, not the full TP
    # group.
    return {"enable_tp": True, "use_attn_tp_group": is_dp_attention_enabled()}


def embed_is_dp_sharded(embed_tokens: torch.nn.Module) -> bool:
    """True when the embedding's vocab shards span more than one attention-DP
    group, i.e. the lookup needs the DP row gather/scatter to be correct.

    Holds for both shard widths of the non-attn-TP scope: the whole TP group and
    the narrower intra-node block picked by SGLANG_VOCAB_TP_SIZE (the block
    still spans several DP ranks, each holding different tokens)."""
    return (
        int(getattr(embed_tokens, "tp_size", 1)) > 1
        and not getattr(embed_tokens, "use_attn_tp_group", False)
        and get_parallel().attn_tp_size < int(embed_tokens.tp_size)
    )


def dp_sharded_embed(
    embed_tokens: torch.nn.Module,
    input_ids: torch.Tensor,
    forward_batch,
) -> torch.Tensor:
    """Embed with a vocab table sharded over the full TP group while DP
    attention hands each rank only its own tokens.

    The layer's reduce spans every TP rank, so all of them must look up the
    SAME ids: gather the ids over the DP dim, embed (the layer's all-reduce
    then sums the vocab shards), and keep this rank's rows. Same shape as what
    LogitsProcessor does around a global-TP lm_head
    (_gather_dp_attn_hidden_states / _scatter_dp_attn_logits).

    Offsets are safe in both padding modes even though dp_scatter always derives
    them as a cumsum of global_num_tokens_gpu: under MAX_LEN,
    compute_dp_attention_metadata rewrites global_num_tokens to [max] * dp_size
    (forward_batch_info.py), so the cumsum equals the all_gather layout
    (rank * max_len); under SUM_LEN the all_reduce/all_gatherv layout is the
    cumsum by construction.

    When the vocab scope is narrowed (SGLANG_VOCAB_TP_SIZE) each block already
    holds a full replica, so only the block's ids have to be shared and the
    gather runs on the block's group. The same reasoning applies to the block
    cumsum: MAX_LEN equalizes the per-rank counts, so slicing them per block
    keeps gather and scatter offsets in agreement.
    """
    from sglang.srt.layers.dp_attention import (
        dp_gather_replicate,
        dp_scatter,
        get_global_dp_buffer_len,
        vocab_block_dp_gather_replicate,
        vocab_block_dp_scatter,
        vocab_dp_block_layout,
        vocab_dp_static_rows,
    )

    block = vocab_dp_block_layout(
        forward_batch.global_num_tokens_cpu, forward_batch.global_num_tokens_gpu
    )
    if block is not None:
        block_ids = input_ids.new_zeros((block.buffer_len,))
        vocab_block_dp_gather_replicate(block_ids, input_ids, block)
        block_embeds = embed_tokens(block_ids)
        local_embeds = block_embeds.new_empty(
            (input_ids.shape[0], block_embeds.shape[-1])
        )
        vocab_block_dp_scatter(local_embeds, block_embeds, block)
        return local_embeds

    global_len = get_global_dp_buffer_len()
    if global_len is None:
        raise RuntimeError(
            "dp_sharded_embed needs the DP buffer length for this forward; "
            "the caller must run inside the mlp-sync (DP gather) path."
        )
    if vocab_dp_static_rows() or forward_batch.global_num_tokens_cpu is None:
        # The global path derives both its buffer length and its offsets from
        # DP-synced per-rank counts. A static row capacity is only wired into the
        # block layout, and without the counts `get_global_dp_buffer_len()` is a
        # leftover from an earlier forward: gathering into it would silently
        # mis-place rows rather than fail.
        raise RuntimeError(
            "dp_sharded_embed fell back to the global DP gather, which needs "
            "DP-synced per-rank token counts. Narrow the vocab scope with "
            "SGLANG_VOCAB_TP_SIZE so the block layout applies, or drop "
            "SGLANG_VOCAB_DP_STATIC_ROWS / SGLANG_SCHEDULER_SKIP_ALL_GATHER."
        )
    global_ids = input_ids.new_zeros((global_len,))
    dp_gather_replicate(global_ids, input_ids, forward_batch)
    global_embeds = embed_tokens(global_ids)
    local_embeds = global_embeds.new_empty(
        (input_ids.shape[0], global_embeds.shape[-1])
    )
    dp_scatter(local_embeds, global_embeds, forward_batch)
    return local_embeds


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    Adapted from torch.nn.Embedding, note that we pad the vocabulary size to
    make sure it is divisible by the number of model parallel GPUs.

    In order to support various loading methods, we ensure that LoRA-added
    embeddings are always at the end of TP-sharded tensors. In other words,
    we shard base embeddings and LoRA embeddings separately (both padded),
    and place them in the same tensor.
    In this example, we will have the original vocab size = 1010,
    added vocab size = 16 and padding to 64. Therefore, the total
    vocab size with padding will be 1088 (because we first pad 1010 to
    1024, add 16, and then pad to 1088).
    Therefore, the tensor format looks like the following:
    TP1, rank 0 (no sharding):
                            |< --------BASE-------- >|< -BASE PADDING-- >|< -----LORA------ >|< -LORA PADDING-- >|
    corresponding token_id: |  0  |  1  | ... | 1009 |  -1  | ... |  -1  | 1010 | ... | 1015 |  -1  | ... |  -1  |
                     index: |  0  |  1  | ... | 1009 | 1010 | ... | 1023 | 1024 | ... | 1039 | 1040 | ... | 1087 |

    TP2, rank 0:
                            |< --------------------BASE--------------------- >|< -----LORA------ >|< -LORA PADDING- >|
    corresponding token_id: |  0  |  1  |  2  | ... | 497  | 498 | ...  | 511 | 1000 | ... | 1015 |  -1  | ... |  -1 |
                     index: |  0  |  1  |  2  | ... | 497  | 498 | ...  | 511 | 512  | ... | 527  |  520 | ... | 543 |
    TP2, rank 1:
                            |< -----------BASE----------- >|< -BASE PADDING- >|< -----------LORA PADDING----------- >|
    corresponding token_id: | 512 | 513 | 514 | ... | 1009 | -1  | ...  | -1  |  -1  | ... |  -1  | -1  | ... |   -1 |
                     index: |  0  |  1  |  2  | ... | 497  | 498 | ...  | 511 | 512  | ... | 519  | 520 | ... |  543 |

    Args:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        params_dtype: type of the parameters.
        org_num_embeddings: original vocabulary size (without LoRA).
        padding_size: padding size for the vocabulary.
        quant_config: quant config for the layer
        prefix: full name of the layer in the state dict
    """  # noqa: E501

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        params_dtype: Optional[torch.dtype] = None,
        org_num_embeddings: Optional[int] = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        enable_tp: bool = True,
        use_attn_tp_group: bool = False,
        use_presharded_weights: bool = False,
    ):
        super().__init__()
        self.quant_config = quant_config

        self.enable_tp = enable_tp
        self.use_attn_tp_group = use_attn_tp_group
        if self.enable_tp:
            if use_attn_tp_group:
                tp_rank = get_parallel().attn_tp_rank
                self.tp_size = get_parallel().attn_tp_size
                self.vocab_shard_group = get_parallel().attn_tp_group
            else:
                # The "global TP" scope goes through the vocab-TP group, which
                # IS the TP group unless SGLANG_VOCAB_TP_SIZE narrows it to an
                # intra-node block (one vocab replica per block).
                tp_rank = get_parallel().vocab_tp_rank
                self.tp_size = get_parallel().vocab_tp_size
                self.vocab_shard_group = get_parallel().vocab_tp_group
        else:
            assert use_attn_tp_group is False
            tp_rank = 0
            self.tp_size = 1
            self.vocab_shard_group = None
        # True when the shards span a strict subgroup of TP: the reduce/gather
        # must name that group instead of the ambient TP collectives.
        self.vocab_tp_subgroup = (
            self.enable_tp
            and not use_attn_tp_group
            and self.tp_size < get_parallel().tp_size
        )

        self.num_embeddings = num_embeddings
        self.org_vocab_size = org_num_embeddings or num_embeddings

        # Support the case where the vocab size is not divisible by the TP size.
        if (
            _is_cpu
            and pad_vocab_size(self.org_vocab_size, padding_size) % self.tp_size != 0
        ):
            padding_size *= self.tp_size
        self.padding_size = padding_size

        num_added_embeddings = num_embeddings - self.org_vocab_size
        self.use_presharded_weights = use_presharded_weights
        if use_presharded_weights:
            assert (
                num_added_embeddings == 0
            ), "Lora is not supported with presharded weights."

        self.org_vocab_size_padded = pad_vocab_size(
            self.org_vocab_size, self.padding_size
        )
        self.num_embeddings_padded = pad_vocab_size(
            self.org_vocab_size_padded + num_added_embeddings, self.padding_size
        )
        assert self.org_vocab_size_padded <= self.num_embeddings_padded

        self.shard_indices = self._get_indices(
            self.num_embeddings_padded,
            self.org_vocab_size_padded,
            self.num_embeddings,
            self.org_vocab_size,
            tp_rank,
            self.tp_size,
        )
        self.embedding_dim = embedding_dim

        quant_method = None
        if quant_config is not None:
            quant_method = quant_config.get_quant_method(self, prefix=prefix)
        if quant_method is None:
            quant_method = UnquantizedEmbeddingMethod()

        # If we are making an embedding layer, then our quantization linear
        # method must implement the embedding operation. If we are another
        # layer type like ParallelLMHead, this is not important.
        is_embedding_layer = type(self) is VocabParallelEmbedding
        quant_method_implements_embedding = method_has_implemented_embedding(
            type(quant_method)
        )
        if is_embedding_layer and not quant_method_implements_embedding:
            raise NotImplementedError(
                f"The class {type(quant_method).__name__} must implement "
                "the 'embedding' method, see UnquantizedEmbeddingMethod."
            )

        self.quant_method: QuantizeMethodBase = quant_method

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        # Divide the weight matrix along the vocaburaly dimension.
        self.num_added_embeddings = self.num_embeddings - self.org_vocab_size
        self.num_embeddings_per_partition = divide(
            self.num_embeddings_padded, self.tp_size
        )
        assert (
            self.shard_indices.num_elements_padded == self.num_embeddings_per_partition
        )
        self.num_org_embeddings_per_partition = (
            self.shard_indices.org_vocab_end_index
            - self.shard_indices.org_vocab_start_index
        )
        self.num_added_embeddings_per_partition = (
            self.shard_indices.added_vocab_end_index
            - self.shard_indices.added_vocab_start_index
        )

        self.quant_method.create_weights(
            self,
            self.embedding_dim,
            [self.num_embeddings_per_partition],
            self.embedding_dim,
            self.num_embeddings_padded,
            params_dtype=params_dtype,
            weight_loader=self.weight_loader,
        )

    @classmethod
    def _get_indices(
        cls,
        vocab_size_padded: int,
        org_vocab_size_padded: int,
        vocab_size: int,
        org_vocab_size: int,
        tp_rank: int,
        tp_size: int,
    ) -> VocabParallelEmbeddingShardIndices:
        """Get start and end indices for vocab parallel embedding, following the
        layout outlined in the class docstring, based on the given tp_rank and
        tp_size."""
        num_added_embeddings_padded = vocab_size_padded - org_vocab_size_padded
        (
            padded_org_vocab_start_index,
            padded_org_vocab_end_index,
        ) = vocab_range_from_global_vocab_size(org_vocab_size_padded, tp_rank, tp_size)
        (
            padded_added_vocab_start_index,
            padded_added_vocab_end_index,
        ) = vocab_range_from_global_vocab_size(
            num_added_embeddings_padded, tp_rank, tp_size, offset=org_vocab_size
        )
        # remove padding
        org_vocab_start_index = min(padded_org_vocab_start_index, org_vocab_size)
        org_vocab_end_index = min(padded_org_vocab_end_index, org_vocab_size)
        added_vocab_start_index = min(padded_added_vocab_start_index, vocab_size)
        added_vocab_end_index = min(padded_added_vocab_end_index, vocab_size)
        return VocabParallelEmbeddingShardIndices(
            padded_org_vocab_start_index,
            padded_org_vocab_end_index,
            padded_added_vocab_start_index,
            padded_added_vocab_end_index,
            org_vocab_start_index,
            org_vocab_end_index,
            added_vocab_start_index,
            added_vocab_end_index,
        )

    def get_sharded_to_full_mapping(self) -> Optional[List[int]]:
        """Get a mapping that can be used to reindex the gathered
        logits for sampling.

        During sampling, we gather logits from all ranks. The relationship
        of index->token_id will follow the same format as outlined in the class
        docstring. However, after the gather, we want to reindex the final
        logits tensor to map index->token_id one-to-one (the index is always
        equal the token_id it corresponds to). The indices returned by this
        method allow us to do that.
        """
        if self.tp_size < 2:
            return None

        base_embeddings: List[int] = []
        added_embeddings: List[int] = []
        padding: List[int] = []
        for tp_rank in range(self.tp_size):
            shard_indices = self._get_indices(
                self.num_embeddings_padded,
                self.org_vocab_size_padded,
                self.num_embeddings,
                self.org_vocab_size,
                tp_rank,
                self.tp_size,
            )
            range_start = self.num_embeddings_per_partition * tp_rank
            range_end = self.num_embeddings_per_partition * (tp_rank + 1)
            base_embeddings.extend(
                range(range_start, range_start + shard_indices.num_org_elements)
            )
            padding.extend(
                range(
                    range_start + shard_indices.num_org_elements,
                    range_start + shard_indices.num_org_elements_padded,
                )
            )
            added_embeddings.extend(
                range(
                    range_start + shard_indices.num_org_elements_padded,
                    range_start
                    + shard_indices.num_org_elements_padded
                    + shard_indices.num_added_elements,
                )
            )
            padding.extend(
                range(
                    range_start
                    + shard_indices.num_org_elements_padded
                    + shard_indices.num_added_elements,
                    range_start
                    + shard_indices.num_org_elements_padded
                    + shard_indices.num_added_elements_padded,
                )
            )
            assert (
                range_start
                + shard_indices.num_org_elements_padded
                + shard_indices.num_added_elements_padded
                == range_end
            )
        ret = base_embeddings + added_embeddings + padding
        assert len(ret) == self.num_embeddings_padded
        return ret

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        output_dim = getattr(param, "output_dim", None)
        packed_dim = getattr(param, "packed_dim", None)

        # If the parameter is a gguf weight, then load it directly.
        if getattr(param, "is_gguf_weight_type", None):
            param.data.copy_(loaded_weight)
            param.weight_type = loaded_weight.item()
            return
        elif isinstance(param, UninitializedParameter):
            shape = list(loaded_weight.shape)
            if output_dim is not None:
                shape[output_dim] = shape[output_dim] // self.tp_size
            param.materialize(tuple(shape), dtype=loaded_weight.dtype)

        # If parameter does not have output dim, then it should
        # be copied onto all gpus (e.g. g_idx for act_order gptq).
        if output_dim is None:
            if (
                loaded_weight.ndim == 0
                and param.data.ndim == 1
                and param.data.numel() == 1
            ):
                loaded_weight = loaded_weight.reshape(1)
            assert param.data.shape == loaded_weight.shape
            param.data.copy_(loaded_weight)
            return

        # Shard indexes for loading the weight
        start_idx = self.shard_indices.org_vocab_start_index
        shard_size = self.shard_indices.org_vocab_end_index - start_idx

        # If param packed on the same dim we are sharding on, then
        # need to adjust offsets of loaded weight by pack_factor.
        if packed_dim is not None and packed_dim == output_dim:
            packed_factor = (
                param.packed_factor
                if isinstance(param, BasevLLMParameter)
                else param.packed_factor
            )
            assert loaded_weight.shape[output_dim] == (
                self.org_vocab_size // param.packed_factor
            )
            start_idx = start_idx // packed_factor
            shard_size = shard_size // packed_factor
        else:
            assert loaded_weight.shape[output_dim] == (
                self.org_vocab_size
                // (self.tp_size if self.use_presharded_weights else 1)
            ), f"{self.org_vocab_size=} {self.use_presharded_weights=} {loaded_weight.shape[output_dim]=}"

        # Copy the data.
        if not self.use_presharded_weights:
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        param[: loaded_weight.shape[0]].data.copy_(loaded_weight)
        param[loaded_weight.shape[0] :].data.fill_(0)

    def _use_triton_embedding(self, input_: torch.Tensor) -> bool:
        """Whether the fused Triton kernel can replace the mask+gather+fill unit."""
        if _is_npu:
            return False
        if self.tp_size == 1:
            return False
        if not isinstance(self.quant_method, UnquantizedEmbeddingMethod):
            return False
        if not input_.is_cuda or not input_.is_contiguous():
            return False
        if input_.dtype not in (torch.int32, torch.int64):
            return False
        return (
            self.weight.is_cuda
            and self.weight.ndim == 2
            and self.weight.stride(1) == 1
            and self.weight.dtype in (torch.float16, torch.bfloat16, torch.float32)
        )

    def _embed_local_shard(self, input_: torch.Tensor) -> torch.Tensor:
        """Embed against the local vocab shard; out-of-shard rows are zero
        (identity when tp_size == 1).

        The output must be allocated inside the symmetric-memory context so
        the caller's all-reduce can use it; the mask temporaries and the
        in-place fill deliberately stay outside the pool.
        """
        symm_alloc = use_symmetric_memory(
            get_tp_group(),
            disabled=not is_allocation_symmetric() or self.vocab_tp_subgroup,
        )
        if self.tp_size == 1:
            with symm_alloc:
                return self.quant_method.embedding(self, input_.long())
        if self._use_triton_embedding(input_):
            with symm_alloc:
                return fused_vocab_parallel_embedding(
                    input_,
                    self.weight,
                    self.shard_indices.org_vocab_start_index,
                    self.shard_indices.org_vocab_end_index,
                    self.shard_indices.num_org_vocab_padding,
                    self.shard_indices.added_vocab_start_index,
                    self.shard_indices.added_vocab_end_index,
                )
        # Map out-of-shard ids to index 0, gather, then zero those rows.
        masked_input, input_mask = get_masked_input_and_mask(
            input_,
            self.shard_indices.org_vocab_start_index,
            self.shard_indices.org_vocab_end_index,
            self.shard_indices.num_org_vocab_padding,
            self.shard_indices.added_vocab_start_index,
            self.shard_indices.added_vocab_end_index,
        )
        with symm_alloc:
            output_parallel = self.quant_method.embedding(self, masked_input.long())
        output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
        return output_parallel

    def embed_local_shard(self, input_: torch.Tensor) -> torch.Tensor:
        """Vocab-shard lookup WITHOUT the cross-rank reduce, for callers that
        own the reduce themselves -- e.g. a speculative draft that shares this
        table while its ambient TP group is patched to the attention-TP group,
        so it must name the shard group explicitly."""
        maybe_detect_oob(
            input_, 0, self.num_embeddings, "VocabParallelEmbedding input id"
        )
        return self._embed_local_shard(input_)

    def forward(self, input_):
        # Surface a bad token id (>= vocab_size, or a negative / unmasked sentinel) as a
        # located async assert instead of a silent OOB embedding gather (tp=1 does not mask).
        maybe_detect_oob(
            input_, 0, self.num_embeddings, "VocabParallelEmbedding input id"
        )
        output_parallel = self._embed_local_shard(input_)
        if self.tp_size > 1 and not get_attn_tp_context().input_scattered:
            if self.use_attn_tp_group:
                output_parallel = attn_tp_all_reduce(output_parallel)
            elif self.vocab_tp_subgroup:
                # Vocab replicated per block: reduce inside the block only.
                output_parallel = self.vocab_shard_group.all_reduce(output_parallel)
            else:
                # Reduce across all the model parallel GPUs.
                output_parallel = tensor_model_parallel_all_reduce(output_parallel)
        return output_parallel

    def extra_repr(self) -> str:
        s = f"num_embeddings={self.num_embeddings_per_partition}"
        s += f", embedding_dim={self.embedding_dim}"
        s += f", org_vocab_size={self.org_vocab_size}"
        s += f", num_embeddings_padded={self.num_embeddings_padded}"
        if self.enable_tp:
            s += f", tp_size={self.tp_size}"
        return s


class ParallelLMHead(VocabParallelEmbedding):
    """Parallelized LM head.

    Output logits weight matrices used in the Sampler. The weight and bias
    tensors are padded to make sure they are divisible by the number of
    model parallel GPUs.

    Args:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        bias: whether to use bias.
        params_dtype: type of the parameters.
        org_num_embeddings: original vocabulary size (without LoRA).
        padding_size: padding size for the vocabulary.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        bias: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        org_num_embeddings: Optional[int] = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        enable_tp: bool = True,
        use_attn_tp_group: bool = False,
        use_presharded_weights: bool = False,
    ):
        super().__init__(
            num_embeddings,
            embedding_dim,
            params_dtype=params_dtype,
            org_num_embeddings=org_num_embeddings,
            padding_size=padding_size,
            quant_config=quant_config,
            prefix=prefix,
            enable_tp=enable_tp,
            use_attn_tp_group=use_attn_tp_group,
            use_presharded_weights=use_presharded_weights,
        )
        self.quant_config = quant_config

        # We only support pack LMHead if it's not quantized.
        if _is_cpu and _is_cpu_amx_available:
            if hasattr(self, "weight") and self.weight.dtype in [
                torch.bfloat16,
                torch.float16,
            ]:
                self.quant_method = PackWeightMethod(weight_names=["weight"])

        if bias:
            self.bias = Parameter(
                torch.empty(self.num_embeddings_per_partition, dtype=params_dtype)
            )
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)

    def tie_weights(self, embed_tokens: VocabParallelEmbedding):
        """Tie the weights with word embeddings."""
        # GGUF quantized embed_tokens.
        if self.quant_config and self.quant_config.get_name() == "gguf":
            return embed_tokens
        else:
            self.weight = embed_tokens.weight
            return self

    def forward(self, input_):
        del input_
        raise RuntimeError("LMHead's weights should be used in the sampler.")
