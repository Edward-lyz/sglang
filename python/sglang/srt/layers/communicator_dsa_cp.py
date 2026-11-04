# Copyright 2023-2024 SGLang Team
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


from functools import partial
from typing import Callable, Optional

import torch

from sglang.srt.layers.attention.dsa.utils import (
    dsa_use_prefill_cp,
)
from sglang.srt.layers.communicator import (
    CommunicateContext,
    CommunicateSimpleFn,
    CommunicateSummableTensorPairFn,
    CommunicateWithAllReduceAndLayerNormFn,
    LayerCommunicator,
    LayerScatterModes,
    ScatterMode,
)
from sglang.srt.layers.cp.utils import is_mla_cp_active
from sglang.srt.layers.dp_attention import (
    attn_cp_all_gather_into_tensor,
    attn_cp_reduce_scatter_tensor,
    get_global_dp_buffer,
    get_local_dp_buffer,
)
from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
from sglang.srt.runtime_context import get_parallel


def maybe_prefetch_next_full_attention_kv(
    forward_batch: ForwardBatch,
    next_full_attention_layer_id: Optional[int],
) -> None:
    """Prefetch (owner-broadcast) the next layer's DSA KV under layer split.

    No-op unless the current batch runs DSA prefill-CP and the active KV pool is
    a layer-sharded pool exposing ``prefetch_kv_buffer`` (i.e.
    ``LayerSplitDSATokenToKVPool``). Kicking the broadcast off one layer ahead
    overlaps it with the current layer's attention compute.
    """
    if next_full_attention_layer_id is None or not dsa_use_prefill_cp(forward_batch):
        return

    prefetch_kv_buffer = getattr(get_token_to_kv_pool(), "prefetch_kv_buffer", None)
    if prefetch_kv_buffer is not None:
        prefetch_kv_buffer(next_full_attention_layer_id)


def dsa_cp_gather_hidden_states(hidden_states: torch.Tensor):
    attn_dp_size = get_parallel().attn_dp_size
    attn_tp_size = get_parallel().attn_tp_size
    assert attn_tp_size == 1
    if attn_dp_size > 1:
        # Prefill CP together with DP>1: the attn_cp group is scoped per DP
        # group, but with moe_a2a_backend="none" the experts are sharded across
        # the FULL TP group, so every rank must see every token (from all DP
        # groups) before the per-rank expert all-reduce. Gather each rank's CP
        # shard over the full TP group into the DP-global buffer.
        hidden_states, local_hidden_states = (
            get_global_dp_buffer(get_tp_group()),
            hidden_states,
        )
        get_tp_group().all_gather_into_tensor(hidden_states, local_hidden_states)
        return hidden_states
    hidden_states, local_hidden_states = (
        get_local_dp_buffer(get_parallel().attn_cp_group),
        hidden_states,
    )
    attn_cp_all_gather_into_tensor(hidden_states, local_hidden_states)
    return hidden_states


def dsa_cp_reduce_scatter_hidden_states(hidden_states: torch.Tensor):
    attn_dp_size = get_parallel().attn_dp_size
    attn_tp_size = get_parallel().attn_tp_size
    assert attn_tp_size == 1
    if attn_dp_size > 1:
        # Symmetric inverse of the full-TP-group gather above: SUM the per-rank
        # expert partials across the full TP group and scatter each rank its own
        # CP shard back. reduce_scatter_tensor splits the global buffer into
        # tp_size equal chunks (chunk r -> rank r), matching the all_gather order.
        tp_group = get_tp_group()
        tp_size = get_parallel().tp_size
        input_hidden_states = hidden_states
        hidden_states = hidden_states.tensor_split(tp_size)[get_parallel().tp_rank]
        tp_group.reduce_scatter_tensor(
            hidden_states.contiguous(), input_hidden_states
        )
        return hidden_states
    cp_size = get_parallel().attn_cp_size
    cp_rank = get_parallel().attn_cp_rank
    input_hidden_states = hidden_states
    hidden_states = hidden_states.tensor_split(cp_size)[cp_rank]
    attn_cp_reduce_scatter_tensor(hidden_states, input_hidden_states)
    return hidden_states


class DSACPLayerCommunicator(LayerCommunicator):
    def __init__(
        self,
        layer_scatter_modes: LayerScatterModes,
        input_layernorm: torch.nn.Module,
        post_attention_layernorm: torch.nn.Module,
        # Reduce scatter requires skipping all-reduce in model code after MoE/MLP, so only enable for models which have that implemented. Remove flag once done for all models that use LayerCommunicator.
        allow_reduce_scatter: bool = False,
        is_last_layer: bool = False,
        qkv_latent_func: Optional[Callable] = None,
    ):
        super().__init__(
            layer_scatter_modes,
            input_layernorm,
            post_attention_layernorm,
            allow_reduce_scatter,
            is_last_layer,
            qkv_latent_func,
        )

    def _post_init_communicate(self):
        # SCATTERED in attn tp is different from SCATTERED in global tp when dp_size > 1.
        # With CP the MLP gather/scatter (dsa_cp_gather/reduce_scatter) spans the
        # full TP group when attn_dp_size > 1, so DP>1 is supported here; only the
        # attn-tp dimension must be collapsed (attn_tp_size == 1 under CP).
        if self.layer_scatter_modes.mlp_mode != ScatterMode.SCATTERED:
            assert self._context.attn_tp_size == 1, (
                "attn_tp_size should be 1 under CP when moe_runner_backend is none"
            )
        self._communicate_simple_fn = DSACPCommunicateSimpleFn.get_fn(
            input_mode=ScatterMode.SCATTERED,
            output_mode=ScatterMode.SCATTERED,
            context=self._context,
        )
        self._communicate_with_all_reduce_and_layer_norm_fn = DSACPCommunicateWithAllReduceAndLayerNormFn.get_fn(
            hidden_states_input_mode=ScatterMode.SCATTERED,
            residual_input_mode=ScatterMode.SCATTERED,
            hidden_states_output_mode=self.layer_scatter_modes.mlp_mode,  # SCATTERED, FULL
            residual_output_mode=ScatterMode.SCATTERED,
            context=self._context,
        )
        self._communicate_summable_tensor_pair_fn = DSACPCommunicateSummableTensorPairFn.get_fn(
            hidden_states_input_mode=self.layer_scatter_modes.mlp_mode,  # SCATTERED, FULL
            residual_input_mode=ScatterMode.SCATTERED,
            output_mode=ScatterMode.SCATTERED,
            context=self._context,
        )


class DSACPCommunicateSimpleFn(CommunicateSimpleFn):
    @staticmethod
    def get_fn(
        input_mode: ScatterMode,
        output_mode: ScatterMode,
        context: CommunicateContext,
    ):
        if context.is_same_group_size(input_mode, output_mode):
            return DSACPCommunicateSimpleFn._trivial

        raise NotImplementedError(f"{input_mode=} {output_mode=}")


class DSACPCommunicateWithAllReduceAndLayerNormFn(
    CommunicateWithAllReduceAndLayerNormFn
):
    """Besides communication, needs to
    1. All reduce in tp_attn_group on hidden_states
    2. Apply layer norm
    """

    @staticmethod
    def get_fn(
        hidden_states_input_mode: ScatterMode,
        residual_input_mode: ScatterMode,
        hidden_states_output_mode: ScatterMode,
        residual_output_mode: ScatterMode,
        context: CommunicateContext,
    ):
        assert hidden_states_input_mode == ScatterMode.SCATTERED
        assert residual_input_mode == ScatterMode.SCATTERED
        assert residual_output_mode == ScatterMode.SCATTERED
        if hidden_states_output_mode == ScatterMode.SCATTERED:
            return DSACPCommunicateWithAllReduceAndLayerNormFn._simple

        if hidden_states_output_mode == ScatterMode.FULL:
            return partial(
                DSACPCommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual,
                residual_input_mode=residual_input_mode,
            )

        raise NotImplementedError(
            f"{hidden_states_input_mode=} {residual_input_mode=} {hidden_states_output_mode=} {residual_output_mode=}"
        )

    @staticmethod
    def _gather_hidden_states_and_residual(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        layernorm: torch.nn.Module,
        context: CommunicateContext,
        *,
        residual_input_mode,
    ):
        if hidden_states.shape[0] != 0:
            hidden_states, residual = layernorm(hidden_states, residual)
        # for prefill: attn tp scattered -> full
        # for decode: attn tp full -> full
        if dsa_use_prefill_cp(forward_batch) or is_mla_cp_active(forward_batch):
            hidden_states = dsa_cp_gather_hidden_states(hidden_states)
        return hidden_states, residual


class DSACPCommunicateSummableTensorPairFn(CommunicateSummableTensorPairFn):
    """It is allowed to make (hidden_states, residual) := (hidden_states + residual, None) if needed."""

    @staticmethod
    def get_fn(
        hidden_states_input_mode: ScatterMode,
        residual_input_mode: ScatterMode,
        output_mode: ScatterMode,
        context: CommunicateContext,
    ):
        # Check exact enum match first: even if group sizes happen to be equal
        # (e.g. tp_size == attn_cp_size makes FULL and SCATTERED both size 1),
        # FULL and SCATTERED have different data layouts under CP and require
        # an explicit scatter operation.
        if (
            (hidden_states_input_mode == ScatterMode.FULL)
            and (residual_input_mode == ScatterMode.SCATTERED)
            and (output_mode == ScatterMode.SCATTERED)
        ):
            return DSACPCommunicateSummableTensorPairFn._scatter_hidden_states

        if context.is_same_group_size(
            hidden_states_input_mode, output_mode
        ) and context.is_same_group_size(residual_input_mode, output_mode):
            return DSACPCommunicateSummableTensorPairFn._trivial

        raise NotImplementedError(
            f"{hidden_states_input_mode=} {residual_input_mode=} {output_mode=}"
        )

    @staticmethod
    def _scatter_hidden_states(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
        allow_reduce_scatter: bool = False,
    ):
        # for prefill: full -> attn tp scattered
        # for decode: full -> attn tp full
        if dsa_use_prefill_cp(forward_batch) or is_mla_cp_active(forward_batch):
            hidden_states = dsa_cp_reduce_scatter_hidden_states(hidden_states)
        return hidden_states, residual
