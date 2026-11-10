"""Factory (厂内) EPD producer↔consumer protocol closure.

The encoder submits a TaskResult/EncoderAddr to the vit-server; the vit-server
hands that same payload to prefill, which parses it with PR1's consumer. This
pins that the two halves agree field-for-field — no GPU or network needed.
"""

from sglang.srt.disaggregation.encode_workflow import (
    EncoderAddr,
    SubmitResultRequest,
    TaskResult,
)
from sglang.srt.disaggregation.encoder.internal_protocol import (
    task_result_owner_req_id,
    task_results_to_descriptors,
)


def _submitted_task_result():
    """A TaskResult shaped exactly as the factory TaskProcessor submits it."""
    return TaskResult(
        model_id="dsv41",
        task_id="md5-cat",
        request_id="r-1",
        image_url="http://x/a.jpg",
        status_code=200,
        message="success",
        token_num=502,
        encoder_addr=EncoderAddr(
            ip="10.0.0.1",
            http_port=7000,
            transfer_type="zmq",
            task_id="md5-cat",
            model_id="dsv41",
            embedding_size=502 * 4096 * 2,
            embedding_len=502,
            embedding_dim=4096,
            embedding_dtype="bfloat16",
            img_grid_thw=[[1, 74, 55]],
            modality="image",
        ),
    )


def test_submitted_task_result_parses_on_the_prefill_side():
    """What the encoder submits must be what PR1's consumer can parse."""
    wire = SubmitResultRequest(
        model_id="dsv41", task_results=[_submitted_task_result()]
    ).model_dump()

    descriptors = task_results_to_descriptors(wire["task_results"])
    d = descriptors[0]

    assert d["task_id"] == "md5-cat"
    assert d["transfer_type"] == "zmq"
    assert (d["ip"], d["http_port"]) == ("10.0.0.1", 7000)
    assert (d["embedding_len"], d["embedding_dim"]) == (502, 4096)
    assert d["img_grid_thw"] == [[1, 74, 55]]


def test_request_identity_travels_in_request_id_not_task_id():
    """task_id names the encode; the request is identified by request_id."""
    wire = SubmitResultRequest(
        model_id="dsv41", task_results=[_submitted_task_result()]
    ).model_dump()
    assert task_result_owner_req_id(wire["task_results"][0]) == "r-1"


def test_advertised_port_is_transport_aware():
    """mooncake advertises the HTTP port (/epd/transfer); zmq the ROUTER port.

    A single port for both would send mooncake pulls to the zmq ROUTER port.
    """
    import os
    from types import SimpleNamespace

    from sglang.srt.disaggregation.encode_workflow import TaskProcessor

    os.environ["POD_IP"] = "10.0.0.1"

    def _processor(backend):
        """A TaskProcessor bound to a stub encoder using the given backend."""
        encoder = SimpleNamespace(
            server_args=SimpleNamespace(
                encoder_transfer_backend=backend, port=8000, served_model_name="m"
            ),
            pull_zmq_port=7000,
        )
        return TaskProcessor(encoder)

    assert _processor("mooncake")._advertised_port == 8000
    assert _processor("zmq_to_scheduler")._advertised_port == 7000


def test_metric_failure_never_breaks_a_task():
    """A broken collector must not stop the TaskResult from being produced."""
    from unittest import mock

    from sglang.srt.disaggregation import encode_workflow as workflow_module

    class _Boom:
        def __getattr__(self, name):
            raise RuntimeError("collector exploded")

    with mock.patch.object(
        workflow_module, "logger", mock.Mock()
    ), mock.patch(
        "sglang.srt.disaggregation.encoder.server.encoder_metrics_collector",
        _Boom(),
        create=True,
    ):
        workflow_module._new_epd_metric("inc_requests_received", modality="image")


def test_unknown_time_cost_keys_do_not_fail_the_task():
    """Stage costs are filtered to TimeCostInfo fields before the reply."""
    from types import SimpleNamespace

    from sglang.srt.disaggregation.encode_workflow import TaskProcessor

    encoder = SimpleNamespace(
        server_args=SimpleNamespace(
            encoder_transfer_backend="mooncake", port=8000, served_model_name="m"
        ),
        pull_zmq_port=7000,
        pop_task_time_costs=lambda task_id: {
            "process_time_cost": 1.0,
            "embed_time_cost": 2.0,
            "decode_time_cost": 3.0,
            "some_future_field": 4.0,
        },
    )
    costs = TaskProcessor(encoder)._known_time_costs("t1")
    assert costs == {
        "process_time_cost": 1.0,
        "embed_time_cost": 2.0,
        "decode_time_cost": 3.0,
    }
