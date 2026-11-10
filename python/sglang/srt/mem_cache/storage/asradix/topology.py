"""Pure topology helpers for PP-aware AttentionStore namespaces."""

from typing import Sequence


def attention_store_kda_namespaces(
    model_id: str, node_pp_start: int
) -> tuple[str, str]:
    """Return stable node-local KDA page and checkpoint namespaces."""
    if not model_id or node_pp_start < 0:
        raise ValueError(
            "AttentionStore KDA namespace requires a model ID and PP start rank"
        )
    return f"{model_id}:{node_pp_start}", f"{model_id}:kda:{node_pp_start}"


def attention_store_node_pp_range(
    pp_size: int, nnodes: int, node_rank: int
) -> tuple[int, int]:
    """Return the contiguous PP role hosted by one node."""
    if pp_size <= 0 or nnodes <= 0 or not 0 <= node_rank < nnodes:
        raise ValueError(
            f"invalid AttentionStore topology: pp_size={pp_size}, "
            f"nnodes={nnodes}, node_rank={node_rank}"
        )
    if nnodes > pp_size:
        raise ValueError(
            f"AttentionStore requires nnodes ({nnodes}) <= pp_size ({pp_size})"
        )
    if pp_size % nnodes != 0:
        raise ValueError(
            f"AttentionStore requires pp_size ({pp_size}) to be "
            f"divisible by nnodes ({nnodes})"
        )
    stages_per_node = pp_size // nnodes
    start = node_rank * stages_per_node
    return start, start + stages_per_node


def attention_store_agent_group_membership(
    pp_size: int, nnodes: int, node_rank: int
) -> tuple[int, int]:
    """Return the Agent member count and local member ID for one cache group."""
    if pp_size <= 0 or nnodes <= 0 or not 0 <= node_rank < nnodes:
        raise ValueError(
            f"invalid AttentionStore topology: pp_size={pp_size}, "
            f"nnodes={nnodes}, node_rank={node_rank}"
        )
    if pp_size == 1:
        # In non-PP DP attention, each DP rank owns a node-local cache and Agent.
        return 1, 0
    attention_store_node_pp_range(pp_size, nnodes, node_rank)
    return nnodes, node_rank


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


def attention_store_endpoint_bytes_per_block(
    layer_lists: Sequence[Sequence[int]],
    bytes_per_layer: int,
    tp_size: int,
) -> int:
    """Return the whole node-local namespace payload for one logical block."""
    if bytes_per_layer <= 0 or tp_size <= 0:
        raise ValueError(
            "AttentionStore endpoint byte geometry must be positive: "
            f"bytes_per_layer={bytes_per_layer}, tp_size={tp_size}"
        )
    active_stage_count = sum(bool(layer_ids) for layer_ids in layer_lists)
    layer_num = max(map(len, layer_lists), default=0)
    if active_stage_count == 0 or layer_num == 0:
        return 0
    shard_num = active_stage_count * tp_size
    return bytes_per_layer * layer_num * shard_num
