"""Adapters for the factory encoder-task result protocol.

The New EPD runtime deals in ``EmbeddingData`` objects and should not know
about the ``task_results`` wire format an external encoder pool produces. This
module is the boundary between the two protocols.

The two decompose a request differently, which is the main thing reconciled
here:

- ``task_results`` is a positional list with one entry per multimodal item. The
  factory receiver reads it as ``enumerate(task_results)``, and entries carry no
  part numbering.
- New EPD numbers parts, because it shards one request across encoder replicas
  and reassembles by ``part_idx``.

Part numbering is therefore assigned here rather than taken as given: from
``index`` when the payload numbers its items, otherwise from list position,
which is how the factory receiver reads the list. Optional fields default to
what the factory receiver defaults them to, so a payload it accepts is accepted
here too.

``task_id`` names an encode, not a request: it is ``md5(model_id + media_uri)``,
so two items of one request have two task ids and the same media twice has one.
Request identity comes from ``request_id`` or from the caller's rid, never from
``task_id`` -- the aggregator rejects parts whose ``req_id`` disagree.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Any, Mapping

from sglang.srt.managers.schedule_batch import Modality

# The wire transfer types the factory receiver implements. "rdma" is defined in
# the protocol but is a different flow -- it advertises the encoder's own
# item_addr / cache_size for the receiver to read from, rather than taking a
# buffer_address to write into -- and the factory rejects it as NOT_IMPLEMENTED,
# so it is not accepted here either.
ZMQ_TRANSFER_TYPE = "zmq"
MOONCAKE_TRANSFER_TYPE = "mooncake"
SUPPORTED_TRANSFER_TYPES = (ZMQ_TRANSFER_TYPE, MOONCAKE_TRANSFER_TYPE)

_DEFAULT_TRANSFER_TYPE = "mooncake"
_DEFAULT_MODALITY = "image"
_DEFAULT_EMBEDDING_DTYPE = "bfloat16"

# Model-specific metadata a processor may need to rebuild the prompt layout.
# Optional for every model, and readable from either level of the payload.
_PASSTHROUGH_FIELDS = (
    "img_grid_thw",
    "video_grid_thw",
    "original_image_sizes",
    "video_timestamps",
    "chunk_timestamps",
    "second_per_grid_ts",
    "task_id",
    "cache_source",
)


def _get(value: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a mapping or an object, so both wire shapes work."""
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def normalize_modality(value: Any) -> Modality:
    """Coerce a wire modality (name, enum value or ``Modality``) to ``Modality``."""
    if isinstance(value, Modality):
        return value
    if isinstance(value, str):
        name = value.upper()
        if name in Modality.__members__:
            return Modality[name]
    if isinstance(value, int):
        try:
            return Modality(value)
        except ValueError:
            pass
    raise ValueError(f"Unsupported EPD modality: {value!r}")


def task_result_request_id(result: Any) -> Any:
    """``task_id`` identifies the encode; ``req_id`` is accepted as an alias."""
    return _get(result, "task_id") or _get(result, "req_id")


def task_result_owner_req_id(result: Any) -> Any:
    """The request this task result belongs to.

    Not ``task_id``: that is ``md5(model_id + media_uri)``, so it differs per
    item within one request and repeats when the same media does. Parts are
    aggregated per request, and the aggregator rejects parts whose ``req_id``
    disagree, so identity has to come from ``request_id`` -- or, when the
    payload omits it, from the caller's rid.
    """
    return _get(result, "request_id") or task_result_request_id(result)


def encoder_addr_of(result: Any) -> Any:
    """Nested endpoint block; absent is tolerated so callers can default."""
    return _get(result, "encoder_addr") or {}


def field_of(result: Any, name: str, default: Any = None) -> Any:
    """Read a field from encoder_addr, falling back to the task result."""
    return _get(encoder_addr_of(result), name, _get(result, name, default))


def task_result_error(result: Any):
    """``(error_msg, error_code)`` for a failed encode, else ``(None, None)``.

    A failed K3 task result is not marked with an error field: it reports the
    failure in ``status_code`` / ``message`` and carries
    ``embedding_len=embedding_dim=0`` with no grid. Reading only ``error_msg``
    would take that for a valid zero-row embedding and assemble an empty span.
    """
    error_msg = _get(result, "error_msg")
    error_code = _get(result, "error_code")
    if error_msg is not None or error_code is not None:
        return error_msg, error_code
    status_code = _get(result, "status_code")
    if status_code is not None and int(status_code) != int(HTTPStatus.OK):
        return _get(result, "message") or f"encoder returned {status_code}", int(
            status_code
        )
    return None, None


def validate_task_result(result: Any) -> None:
    """Reject a task result the receiver could not act on.

    Only the encode identity is mandatory, matching the factory receiver, which
    fails a task result for a missing ``task_id`` and nothing else. Anything
    optional is either defaulted here or validated where it is consumed.
    """
    if task_result_request_id(result) is None:
        raise ValueError("EPD task result is missing task_id")
    normalize_modality(field_of(result, "modality", _DEFAULT_MODALITY))


def _validate_part_numbering(part_idx: Any, num_parts: Any) -> None:
    """Reject part numbering that would index the aggregation buffers wrongly."""
    if not isinstance(num_parts, int) or num_parts <= 0:
        raise ValueError(f"Invalid EPD num_parts: {num_parts!r}")
    if not isinstance(part_idx, int) or not 0 <= part_idx < num_parts:
        raise ValueError(
            f"Invalid EPD part_idx={part_idx!r} for num_parts={num_parts!r}"
        )


def task_result_to_embedding_data(
    result: Any,
    embedding: Any = None,
    part_idx: int = 0,
    num_parts: int = 1,
    req_id: Any = None,
):
    """Convert one factory task result to New EPD ``EmbeddingData``.

    ``part_idx`` / ``num_parts`` come from the entry's position in
    ``task_results``; the factory protocol does not put them on the wire.

    ``req_id`` is the request every part of this assembly belongs to -- pass the
    receiving request's rid. It falls back to ``request_id`` and only then to
    ``task_id``, which names the encode rather than the request.

    Imported lazily to avoid an import cycle: ``receiver`` owns the wire
    container while this module only owns protocol normalization.
    """
    validate_task_result(result)
    _validate_part_numbering(part_idx, num_parts)
    from sglang.srt.disaggregation.encoder.receiver import EmbeddingData

    shape = _get(result, "embedding_shape", _get(result, "shape"))
    if shape is None:
        rows = field_of(result, "embedding_len")
        dim = field_of(result, "embedding_dim")
        if rows and dim:
            shape = [int(rows), int(dim)]
    if shape is not None:
        shape = list(shape)
    error_msg, error_code = task_result_error(result)
    data = EmbeddingData(
        req_id=req_id if req_id is not None else task_result_owner_req_id(result),
        num_parts=num_parts,
        part_idx=part_idx,
        grid_dim=_get(result, "grid_dim", field_of(result, "img_grid_thw")),
        modality=normalize_modality(field_of(result, "modality", _DEFAULT_MODALITY)),
        embedding=embedding,
        embedding_shape=shape,
        error_msg=error_msg,
        error_code=error_code,
    )
    for name in _PASSTHROUGH_FIELDS:
        value = field_of(result, name)
        if value is not None:
            setattr(data, name, value)
    return data


def task_result_to_transport_descriptor(
    result: Any, part_idx: int = 0, num_parts: int = 1
) -> dict:
    """Normalize one factory task result into a transport descriptor.

    The descriptor is metadata only -- embedding bytes always arrive through the
    configured New EPD transport. Keeping the conversion here stops the receiver
    from depending on the factory's Pydantic models.
    """
    validate_task_result(result)
    _validate_part_numbering(part_idx, num_parts)

    def addr(name, default=None):
        """Endpoint-first lookup, as the factory receiver reads its fields."""
        return field_of(result, name, default)

    descriptor = {
        "req_id": task_result_request_id(result),
        "task_id": addr("task_id"),
        "model_id": addr("model_id"),
        "part_idx": part_idx,
        "num_parts": num_parts,
        "ip": addr("ip"),
        "http_port": addr("http_port"),
        "transfer_type": str(addr("transfer_type", _DEFAULT_TRANSFER_TYPE)).lower(),
        "item_addr": addr("item_addr", 0),
        "rdma_port": addr("rdma_port", 0),
        "cache_size": addr("cache_size", 0),
        "header_size": addr("header_size", 0),
        "mooncake_session_id": addr("mooncake_session_id", ""),
        "embedding_size": addr("embedding_size", 0),
        "embedding_len": addr("embedding_len", 0),
        "embedding_dim": addr("embedding_dim", 0),
        "embedding_dtype": addr("embedding_dtype", _DEFAULT_EMBEDDING_DTYPE),
        "modality": normalize_modality(addr("modality", _DEFAULT_MODALITY)),
        "image_url": addr("image_url"),
        "img_grid_thw": addr("img_grid_thw"),
        "original_image_sizes": addr("original_image_sizes"),
        "chunk_timestamps": addr("chunk_timestamps"),
    }
    if not descriptor["ip"] or not descriptor["http_port"]:
        raise ValueError("EPD encoder_addr has an incomplete endpoint")
    return descriptor


def task_results_to_descriptors(task_results: Any) -> list:
    """Validate a whole ``task_results`` list and return its descriptors.

    Descriptors stay in the caller's order so they can be zipped with
    ``task_results``; the ordering that aligns embedding rows with prompt
    placeholders travels in ``part_idx``, which is what the aggregator places
    parts by. ``index`` -- the media item's position in the original messages --
    supplies it when present, otherwise list position does, which is how the
    factory receiver reads the list.
    """
    if not isinstance(task_results, (list, tuple)) or not task_results:
        raise ValueError("EPD task_results must be a non-empty list")
    for result in task_results:
        error_msg, error_code = task_result_error(result)
        if error_msg is not None:
            # The encoder already failed this item; there is nothing to receive.
            raise ValueError(
                f"encoder failed task {task_result_request_id(result)!r}: "
                f"{error_msg} (code={error_code})"
            )
    num_parts = len(task_results)
    positions = [
        _get(result, "index", position) for position, result in enumerate(task_results)
    ]
    if sorted(positions) != list(range(num_parts)):
        # ``index`` is advisory: it aligns embedding rows with prompt
        # placeholders only when the whole set is a 0..n-1 permutation. Some
        # routers leave it unset (every item defaults to 0), which is not a
        # gap/reorder of a *populated* index but simply "index not provided".
        # In that case the list order is the canonical placeholder order the
        # factory receiver reads, so fall back to it instead of rejecting an
        # otherwise well-formed multi-image request.
        positions = list(range(num_parts))
    descriptors = [
        task_result_to_transport_descriptor(result, part_idx, num_parts)
        for part_idx, result in zip(positions, task_results)
    ]
    transfer_types = {d["transfer_type"] for d in descriptors}
    if len(transfer_types) != 1:
        # One request's parts land through one receiver, so a mixed payload has
        # no rank that could receive all of it.
        raise ValueError(f"EPD task_results mixes transports: {sorted(transfer_types)}")
    transfer_type = descriptors[0]["transfer_type"]
    if transfer_type not in SUPPORTED_TRANSFER_TYPES:
        raise ValueError(
            f"Unsupported EPD transfer_type={transfer_type}; supported: "
            + ", ".join(SUPPORTED_TRANSFER_TYPES)
        )
    return descriptors


def unusable_task_results_reason(task_results: Any):
    """Why ``task_results`` cannot be received from the encoder, or None.

    Covers the two cases a language-only node falls back to local encoding
    for: nothing came back from the encoder pool (missing / empty), or what
    came back cannot be pulled (a failed item, or a malformed payload). The
    answer depends only on the payload, so prefill and decode -- which get the
    same payload -- always make the same call and keep their prompt lengths
    aligned.
    """
    if not task_results:
        return "no task_results"
    try:
        task_results_to_descriptors(task_results)
    except (ValueError, TypeError, KeyError) as e:
        return str(e)
    return None


def descriptor_encoder_url(descriptor: Mapping) -> str:
    """HTTP base URL of the encoder that holds this descriptor's embedding."""
    return f"http://{descriptor['ip']}:{descriptor['http_port']}"
