"""The pull side of the factory EPD protocol (ku doc interfaces 4 and 5).

These pin the request bodies and reply parsing, which is where a mismatch with
the encoder pool would show up as a silent misread rather than an exception.
"""

import pickle

import pytest
import torch

from sglang.srt.disaggregation.encoder.factory_transport import (
    FactoryZmqTransport,
    build_epd_transfer_payload,
    descriptor_to_pull_target,
    epd_transfer_error,
    parse_torch_dtype,
)
from sglang.srt.disaggregation.encoder.internal_protocol import (
    task_results_to_descriptors,
)


def _task_result(transfer_type="mooncake", **addr):
    """One task result for a 576x4096 image embedding, as the ku doc shows it."""
    encoder_addr = {
        "ip": "10.0.1.1",
        "http_port": 8001,
        "transfer_type": transfer_type,
        "task_id": "md5-hash-cat",
        "model_id": "vit_model",
        "embedding_size": 4718592,
        "embedding_len": 576,
        "embedding_dim": 4096,
        "embedding_dtype": "bfloat16",
        "img_grid_thw": [[1, 38, 50]],
        "mooncake_session_id": "mc-session-encoder-1",
    }
    encoder_addr.update(addr)
    return {
        "model_id": "vit_model",
        "task_id": "md5-hash-cat",
        "request_id": "req-001",
        "index": 0,
        "status_code": 200,
        "message": "",
        "token_num": 576,
        "encoder_addr": encoder_addr,
    }


def _pull_target(**kwargs):
    """The pull target one task result normalizes into."""
    descriptor = task_results_to_descriptors([_task_result(**kwargs)])[0]
    return descriptor_to_pull_target(descriptor, "req-001")


def test_pull_target_takes_geometry_from_the_descriptor():
    """Knowing the size up front is what removes the metadata round trip."""
    target = _pull_target()

    assert target.endpoint == "10.0.1.1:8001"
    assert target.shape == [576, 4096]
    assert target.dtype == "bfloat16"
    assert target.byte_size == 4718592
    assert (target.part_idx, target.num_parts) == (0, 1)
    assert target.modality == "image"


def test_byte_size_is_derived_when_the_encoder_omits_it():
    """Without a byte count the landing buffer would be sized from nothing."""
    target = _pull_target(embedding_size=0)
    assert target.byte_size == 576 * 4096 * 2  # bfloat16


def test_epd_transfer_payload_matches_the_documented_body():
    """The request body is the whole contract; a missing field stalls the pull."""
    payload = build_epd_transfer_payload(
        _pull_target(), session_id="192.168.1.195:62354", buffer_address=140234567890432
    )

    assert payload == {
        "req_id": "req-001",
        "task_id": "md5-hash-cat",
        "part_idx": 0,
        "num_parts": 1,
        "session_id": "192.168.1.195:62354",
        "buffer_address": 140234567890432,
        "byte_size": 4718592,
        "dtype": "bfloat16",
        "shape": [576, 4096],
        "image_url": None,
        "modality": "image",
    }


@pytest.mark.parametrize("dtype", ["bfloat16", "torch.bfloat16"])
def test_both_dtype_spellings_on_the_wire_are_accepted(dtype):
    """Encoders send either spelling; misparsing one would halve the rows."""
    assert parse_torch_dtype(dtype) is torch.bfloat16


def test_shard_fields_stay_single_shard_whatever_the_item_position():
    """The encoder only echoes part_idx/num_parts back.

    Pulls are deduplicated per task_id, so one call fetches one whole embedding;
    reporting this item's position instead would hand the encoder a shard index
    that means nothing on its side.
    """
    descriptors = task_results_to_descriptors(
        [
            _task_result(),
            {**_task_result(), "task_id": "md5-hash-dog", "index": 1},
        ]
    )
    second = descriptor_to_pull_target(descriptors[1], "req-001")
    assert second.part_idx == 1

    payload = build_epd_transfer_payload(second, "sess", 4096)
    assert payload["part_idx"] == 0
    assert payload["num_parts"] == 1


def test_rdma_transfer_type_is_rejected():
    """"rdma" advertises the encoder's own item_addr to read from, not a
    buffer_address to write into, and the factory receiver rejects it too."""
    with pytest.raises(ValueError, match="Unsupported EPD transfer_type"):
        task_results_to_descriptors([_task_result(transfer_type="rdma")])


def test_transfer_ack_status_decides_success():
    """The ack's status is the completion signal, so it decides the outcome."""
    assert epd_transfer_error({"status_code": 200, "message": ""}) is None
    assert "500" in epd_transfer_error({"status_code": 500, "message": "oom"})
    assert "oom" in epd_transfer_error({"status_code": 500, "message": "oom"})


def _zmq_frames(meta, embedding=None):
    """The [meta, embedding] frames an encoder replies with over ZMQ."""
    frames = [pickle.dumps(meta)]
    if embedding is not None:
        frames.append(bytes(embedding.view(torch.uint8).numpy().tobytes()))
    return frames


def test_zmq_reply_yields_the_embedding_and_grid():
    """A well-formed reply must reshape to the announced geometry."""
    embedding = torch.arange(8, dtype=torch.bfloat16)
    meta = {
        "shape": [2, 4],
        "dtype": "torch.bfloat16",
        "image_grid_thw": [[1, 38, 50]],
    }

    result = FactoryZmqTransport._parse_reply("md5-hash-cat", _zmq_frames(meta, embedding))

    assert result.success
    assert tuple(result.embedding.shape) == (2, 4)
    assert torch.equal(result.embedding.reshape(-1), embedding)
    assert result.grid_thw.tolist() == [[1, 38, 50]]


def test_zmq_reply_surfaces_the_encoders_error():
    """An encoder-side failure must fail the request, not wait for rows."""
    result = FactoryZmqTransport._parse_reply(
        "md5-hash-cat", _zmq_frames({"error_msg": "404 not found", "error_code": 404})
    )
    assert result.error_msg == "404 not found"
    assert result.error_code == 404


def test_a_reply_without_an_embedding_frame_is_an_error_not_an_empty_tensor():
    """An empty span would otherwise be prefilled as if it were valid."""
    result = FactoryZmqTransport._parse_reply(
        "md5-hash-cat", _zmq_frames({"shape": [2, 4], "dtype": "bfloat16"})
    )
    assert result.embedding is None
    assert "carried no embedding frame" in result.error_msg


def test_an_undecodable_reply_fails_the_request_rather_than_raising():
    """A malformed frame must not take down the scheduler loop."""
    result = FactoryZmqTransport._parse_reply("md5-hash-cat", [b"not-a-pickle"])
    assert result.embedding is None
    assert "undecodable encoder reply" in result.error_msg
