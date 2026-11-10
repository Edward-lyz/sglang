"""Pure topology helpers for PP-aware AttentionStore namespaces."""

from typing import Sequence


def attention_store_node_pp_range(
    pp_size: int, nnodes: int, node_rank: int
) -> tuple[int, int]:
    """Return the contiguous PP role hosted by one node."""
    if pp_size <= 0 or nnodes <= 0 or not 0 <= node_rank < nnodes:
        raise ValueError(
            f"invalid AttentionStore topology: pp_size={pp_size}, "
            f"nnodes={nnodes}, node_rank={node_rank}"
        )
    if pp_size >= nnodes:
        if pp_size % nnodes != 0:
            raise ValueError(
                f"AttentionStore requires pp_size ({pp_size}) to be "
                f"divisible by nnodes ({nnodes})"
            )
        stages_per_node = pp_size // nnodes
        start = node_rank * stages_per_node
        return start, start + stages_per_node

    if nnodes % pp_size != 0:
        raise ValueError(
            f"AttentionStore requires nnodes ({nnodes}) to be "
            f"divisible by pp_size ({pp_size})"
        )
    nodes_per_stage = nnodes // pp_size
    start = node_rank // nodes_per_stage
    return start, start + 1


def attention_store_endpoint_geometry(
    layer_lists: Sequence[Sequence[int]],
    local_pp_offset: int,
    tp_rank: int,
    tp_size: int,
) -> tuple[int, int, int]:
    """Map active node-local PP stages onto uniform AttentionStore shards."""
    if tp_size <= 0 or not 0 <= tp_rank < tp_size:
        raise ValueError(f"invalid attention TP rank {tp_rank}/{tp_size}")
    if not 0 <= local_pp_offset < len(layer_lists):
        raise ValueError(
            f"local PP offset {local_pp_offset} is outside {len(layer_lists)} stages"
        )
    active_offsets = [
        offset for offset, layer_ids in enumerate(layer_lists) if layer_ids
    ]
    shard_num = len(active_offsets) * tp_size
    layer_num = max(map(len, layer_lists), default=0)
    if local_pp_offset not in active_offsets:
        return -1, shard_num, layer_num
    stage_shard = active_offsets.index(local_pp_offset)
    shard_id = stage_shard * tp_size + tp_rank
    return shard_id, shard_num, layer_num
