from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.managers.overlap_utils import RelayPayload
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.utils.common import to_device_nonblocking

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.server_args import ServerArgs


def build_eagle_disagg_draft_input(
    batch: ScheduleBatch,
    server_args: ServerArgs,
    last_tokens_tensor: torch.Tensor,
    future_map: FutureMap,
) -> EagleDraftInput:
    num_states = server_args.speculative_eagle_topk
    if server_args.enable_multi_layer_eagle:
        num_states *= server_args.speculative_num_steps

    # Stack on the host and copy once without blocking: per-request
    # pageable H2D copies would each synchronize the scheduler stream.
    topk_p = to_device_nonblocking(
        torch.stack(
            [
                torch.as_tensor(req.output_topk_p[:num_states], dtype=torch.float32)
                for req in batch.reqs
            ],
            dim=0,
        ),
        torch.float32,
        batch.device,
    )
    topk_index = to_device_nonblocking(
        torch.stack(
            [
                torch.as_tensor(req.output_topk_index[:num_states], dtype=torch.int64)
                for req in batch.reqs
            ],
            dim=0,
        ),
        torch.int64,
        batch.device,
    )

    hidden_states = to_device_nonblocking(
        torch.stack([req.hidden_states_tensor for req in batch.reqs], dim=0),
        None,
        batch.device,
    )

    dsa_topk_indices = None
    dsa_indices_list = [req.output_dsa_topk_indices for req in batch.reqs]
    if dsa_indices_list and all(t is not None for t in dsa_indices_list):
        # Check validity where the data lives (host for PD transfers) before
        # the copy, so .item() does not wait on the scheduler stream.
        dsa_stacked = torch.stack(dsa_indices_list, dim=0)
        if torch.any(torch.all(dsa_stacked < 0, dim=1)).item():
            dsa_topk_indices = None
        else:
            dsa_topk_indices = to_device_nonblocking(dsa_stacked, None, batch.device)

    spec_info = EagleDraftInput(
        topk_p=topk_p,
        topk_index=topk_index,
        hidden_states=hidden_states,
        bonus_tokens=last_tokens_tensor,
        dsa_topk_indices=dsa_topk_indices,
    )
    spec_info.capture_hidden_mode = CaptureHiddenMode.LAST

    if batch.enable_overlap:
        spec_info.future_dsa_topk_indices_available = dsa_topk_indices is not None
        spec_info.future_indices = batch.req_pool_indices
        # Seed the relay buf with the known seq_lens; publish's chained record
        # keeps the in-flight forward's fence intact (see FutureMap.publish).
        future_map.publish(spec_info.future_indices, batch.seq_lens)
        future_map.stash(
            spec_info.future_indices, RelayPayload.from_draft_input(spec_info)
        )

    return spec_info
