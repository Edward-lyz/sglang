"""The task_results protocol as the factory encoder pool actually sends it.

The fixtures deliberately mirror a real payload: no part numbering on the wire,
metadata nested in ``encoder_addr``, and most fields optional. The factory
receiver fails a task result only for a missing ``task_id``, so anything it
accepts has to be accepted here too.
"""

import pytest
import torch

from sglang.srt.disaggregation.encoder.internal_protocol import (
    descriptor_encoder_url,
    task_result_error,
    task_result_to_embedding_data,
    task_result_to_transport_descriptor,
    task_results_to_descriptors,
    validate_task_result,
)
from sglang.srt.disaggregation.encoder.receiver import _aggregate_embedding_part
from sglang.srt.managers.schedule_batch import Modality


def _encoder_addr(**kwargs):
    """One ``encoder_addr`` block, with the fields a real payload carries."""
    value = {
        "ip": "10.0.0.1",
        "http_port": 18000,
        "transfer_type": "mooncake",
        "task_id": "t-1",
        "model_id": "dsv41",
        "embedding_size": 4112384,
        "embedding_len": 502,
        "embedding_dim": 4096,
        "embedding_dtype": "bfloat16",
        "mooncake_session_id": "sess-1",
        "item_addr": 4096,
        "img_grid_thw": [[1, 74, 55]],
    }
    value.update(kwargs)
    return value


def _result(**kwargs):
    """One task result, shaped as the factory encoder pool sends it."""
    value = {
        "model_id": "dsv41",
        "task_id": "t-1",
        "request_id": "r-1",
        "status_code": 200,
        "token_num": 502,
        "encoder_addr": _encoder_addr(),
    }
    value.update(kwargs)
    return value


def test_part_numbering_comes_from_list_position():
    """Without ``index``, list position carries the order."""
    descriptors = task_results_to_descriptors(
        [
            _result(task_id="t-1", encoder_addr=_encoder_addr(task_id="t-1")),
            _result(task_id="t-2", encoder_addr=_encoder_addr(task_id="t-2")),
            _result(task_id="t-3", encoder_addr=_encoder_addr(task_id="t-3")),
        ]
    )

    assert [d["part_idx"] for d in descriptors] == [0, 1, 2]
    assert {d["num_parts"] for d in descriptors} == {3}
    assert [d["task_id"] for d in descriptors] == ["t-1", "t-2", "t-3"]


def test_index_assigns_part_numbering_regardless_of_arrival_order():
    """``index`` is the item's position in the original messages.

    Descriptors keep the caller's order so they can be zipped with the task
    results they came from; the prompt ordering rides in ``part_idx``, which is
    what the aggregator places parts by.
    """
    results = [
        _result(task_id="dog", index=1, encoder_addr=_encoder_addr(task_id="dog")),
        _result(task_id="cat", index=0, encoder_addr=_encoder_addr(task_id="cat")),
    ]

    descriptors = task_results_to_descriptors(results)

    assert [(d["part_idx"], d["task_id"]) for d in descriptors] == [
        (1, "dog"),
        (0, "cat"),
    ]
    # Position-wise pairing with the source list must stay valid, or a caller
    # zipping the two would attach one item's metadata to another's rows.
    assert all(
        result["task_id"] == descriptor["task_id"]
        for result, descriptor in zip(results, descriptors)
    )


@pytest.mark.parametrize("indices", [(0, 2), (1, 1), (0, 0), (1, 2)])
def test_indices_that_are_not_a_permutation_fall_back_to_list_order(indices):
    """``index`` is advisory: it aligns rows with placeholders only when the
    whole set is a 0..n-1 permutation. Some routers leave it unset (every item
    defaults to 0) or populate it oddly; that is "index not provided", not a
    corrupt reorder, so fall back to list order -- the order the factory
    receiver reads -- instead of rejecting an otherwise well-formed request.
    """
    descriptors = task_results_to_descriptors(
        [_result(task_id=f"t-{i}", index=i) for i in indices]
    )
    assert [d["part_idx"] for d in descriptors] == [0, 1]
    assert {d["num_parts"] for d in descriptors} == {2}


def test_parts_of_one_request_aggregate_under_the_receiving_rid():
    """Two images are two task ids but one request.

    ``task_id`` is ``md5(model_id + media_uri)``, so it cannot identify the
    request: the aggregator rejects parts whose ``req_id`` disagree, which would
    fail every multi-image request.
    """
    results = [
        _result(task_id="t-dog", encoder_addr=_encoder_addr(task_id="t-dog")),
        _result(task_id="t-cat", encoder_addr=_encoder_addr(task_id="t-cat")),
    ]
    descriptors = task_results_to_descriptors(results)

    merged = None
    for result, descriptor in zip(results, descriptors):
        part = task_result_to_embedding_data(
            result,
            torch.zeros((502, 4096), dtype=torch.bfloat16),
            part_idx=descriptor["part_idx"],
            num_parts=descriptor["num_parts"],
            req_id="rid-42",
        )
        assert part.req_id == "rid-42"
        merged = _aggregate_embedding_part(merged, part, "deepseek_v41")

    assert merged.ready


def test_request_id_identifies_the_request_when_no_rid_is_given():
    """Falling back to ``task_id`` would make each item its own request."""
    data = task_result_to_embedding_data(_result(task_id="t-1"))
    assert data.req_id == "r-1"


def test_only_task_id_and_an_endpoint_are_mandatory():
    """Everything else defaults exactly as the factory receiver defaults it."""
    descriptor = task_result_to_transport_descriptor(
        {"task_id": "t-9", "encoder_addr": {"ip": "10.0.0.9", "http_port": 18000}}
    )

    assert descriptor["transfer_type"] == "mooncake"
    assert descriptor["modality"] == Modality.IMAGE
    assert descriptor["embedding_dtype"] == "bfloat16"
    assert descriptor["mooncake_session_id"] == ""
    assert descriptor["item_addr"] == 0


def test_missing_task_id_is_rejected():
    """The one field the factory receiver also refuses to default."""
    result = _result()
    del result["task_id"]
    result["encoder_addr"] = _encoder_addr(task_id=None)
    with pytest.raises(ValueError, match="task_id"):
        validate_task_result(result)


def test_req_id_is_accepted_as_a_task_id_alias():
    """Some payloads name the encode ``req_id``; both spellings must work."""
    descriptor = task_result_to_transport_descriptor(
        {"req_id": "r-7", "encoder_addr": {"ip": "10.0.0.7", "http_port": 18000}}
    )
    assert descriptor["req_id"] == "r-7"


def test_transport_descriptor_carries_the_whole_endpoint():
    """Everything a pull needs has to survive the conversion."""
    descriptor = task_result_to_transport_descriptor(_result())

    assert descriptor_encoder_url(descriptor) == "http://10.0.0.1:18000"
    assert descriptor["transfer_type"] == "mooncake"
    assert descriptor["mooncake_session_id"] == "sess-1"
    assert descriptor["item_addr"] == 4096
    assert descriptor["embedding_size"] == 4112384
    assert descriptor["embedding_len"] == 502
    assert descriptor["embedding_dim"] == 4096
    assert descriptor["model_id"] == "dsv41"
    assert descriptor["modality"] == Modality.IMAGE
    assert descriptor["img_grid_thw"] == [[1, 74, 55]]


def test_video_chunk_timestamps_are_carried():
    """Video needs them downstream, so they must survive the conversion."""
    stamps = ["0.00", "1.50", "3.00"]
    video = _result(
        encoder_addr=_encoder_addr(modality="video", chunk_timestamps=stamps)
    )

    descriptor = task_result_to_transport_descriptor(video)
    assert descriptor["modality"] == Modality.VIDEO
    assert descriptor["chunk_timestamps"] == stamps

    data = task_result_to_embedding_data(
        video, torch.zeros((502, 4), dtype=torch.bfloat16)
    )
    assert data.chunk_timestamps == stamps


def test_embedding_shape_falls_back_to_the_announced_geometry():
    """No ``embedding_shape`` on the wire: use embedding_len x embedding_dim."""
    data = task_result_to_embedding_data(
        _result(), torch.zeros((502, 4096), dtype=torch.bfloat16)
    )
    assert data.req_id == "r-1"
    assert data.shape == [502, 4096]
    assert data.modality == Modality.IMAGE
    assert data.img_grid_thw == [[1, 74, 55]]


def test_incomplete_endpoint_is_rejected():
    """An endpoint that cannot be dialled must fail before any transfer."""
    with pytest.raises(ValueError, match="endpoint"):
        task_result_to_transport_descriptor(
            _result(encoder_addr=_encoder_addr(http_port=None))
        )


def test_unknown_modality_is_rejected():
    """Guessing a modality would assemble the wrong prompt layout."""
    with pytest.raises(ValueError, match="modality"):
        validate_task_result(_result(encoder_addr=_encoder_addr(modality="hologram")))


def test_empty_task_results_is_rejected():
    """IC resolves every media item or none, so an empty list is a bug."""
    with pytest.raises(ValueError, match="non-empty"):
        task_results_to_descriptors([])


def test_mixed_transports_are_rejected():
    """One request's parts land through one receiver, so a mix has no owner."""
    with pytest.raises(ValueError, match="mixes transports"):
        task_results_to_descriptors(
            [_result(), _result(encoder_addr=_encoder_addr(transfer_type="zmq"))]
        )


@pytest.mark.parametrize("part_idx,num_parts", [(0, 0), (1, 1), (-1, 2), (2, 2)])
def test_out_of_range_part_numbering_is_rejected(part_idx, num_parts):
    """Part numbering indexes the aggregation buffers, so validate it early."""
    with pytest.raises(ValueError, match="num_parts|part_idx"):
        task_result_to_transport_descriptor(_result(), part_idx, num_parts)


def _failed_result():
    """A failed encode as the factory reports it: no error field, just a status.

    It also zeroes embedding_len/embedding_dim and drops the grid, so reading
    only ``error_msg`` would take this for a valid zero-row embedding.
    """
    return {
        "model_id": "dsv41",
        "task_id": "t-err",
        "request_id": "r-1",
        "image_url": "http://example/a.jpg",
        "status_code": 500,
        "message": "decode failed",
        "token_num": 0,
        "encoder_addr": _encoder_addr(
            embedding_size=0, embedding_len=0, embedding_dim=0, img_grid_thw=None
        ),
    }


def test_status_code_reports_a_failed_encode():
    """The failure travels in status_code / message, not in an error field."""
    assert task_result_error(_failed_result()) == ("decode failed", 500)
    assert task_result_error(_result()) == (None, None)

    data = task_result_to_embedding_data(_failed_result())
    assert data.error_msg == "decode failed"
    assert data.error_code == 500


def test_a_failed_task_result_is_not_turned_into_a_receive():
    """Zero rows with no grid must not assemble as an empty span."""
    with pytest.raises(ValueError, match="encoder failed task"):
        task_results_to_descriptors([_failed_result()])


def test_an_explicit_error_field_takes_precedence():
    """A payload that does set error_msg / error_code is reported verbatim."""
    result = _failed_result()
    result["error_msg"] = "explicit"
    result["error_code"] = 503
    assert task_result_error(result) == ("explicit", 503)
