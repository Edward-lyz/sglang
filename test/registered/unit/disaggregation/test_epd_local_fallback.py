"""Local-encode fallback for multimodal requests the encoder pool did not serve.

A language-only node keeps a request alive when its task_results are missing,
empty, failed or malformed: they are dropped and the media is encoded locally.
Prefill and decode receive the same payload and must make the same call.
"""

import pytest

from sglang.srt.disaggregation.encoder.internal_protocol import (
    unusable_task_results_reason,
)
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.environ import envs

IMAGE_MSG = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "http://x/a.jpg"}},
        ],
    }
]


def _result(**kwargs):
    """One well-formed factory task result."""
    value = {
        "model_id": "dsv41",
        "task_id": "t-1",
        "request_id": "r-1",
        "status_code": 200,
        "encoder_addr": {
            "ip": "10.0.0.1",
            "http_port": 18000,
            "transfer_type": "mooncake",
            "task_id": "t-1",
            "embedding_len": 502,
            "embedding_dim": 4096,
            "embedding_dtype": "bfloat16",
        },
    }
    value.update(kwargs)
    return value


def _chat(**kwargs):
    """Build a chat request the way the gateway forwards it."""
    return ChatCompletionRequest(model="m", messages=IMAGE_MSG, **kwargs)


@pytest.mark.parametrize(
    "task_results",
    [
        None,
        [],
        [_result(status_code=500, message="vit failed")],
        [_result(), _result(task_id="t-2", error_msg="boom", error_code=503)],
        [_result(encoder_addr={"transfer_type": "mooncake"})],
    ],
    ids=["missing", "empty", "failed", "one-of-two-failed", "no-endpoint"],
)
def test_unusable_task_results_fall_back_to_local_encoding(task_results):
    """Anything the prefill could not pull is dropped instead of a 400."""
    assert unusable_task_results_reason(task_results) is not None
    req = _chat(is_mm_request=True, task_results=task_results)
    assert req.task_results is None


def test_usable_task_results_are_kept():
    """A pullable payload still goes through the encoder pool path."""
    assert unusable_task_results_reason([_result()]) is None
    req = _chat(is_mm_request=True, task_results=[_result()])
    assert req.task_results and req.task_results[0]["task_id"] == "t-1"


def test_failed_task_results_without_mm_flag_also_fall_back():
    """The flag is optional; a failed payload alone must not reach the pull."""
    req = _chat(task_results=[_result(status_code=500)])
    assert req.task_results is None


def test_task_results_on_a_text_request_is_still_rejected():
    """is_mm_request=False with task_results is a caller bug, not a fallback."""
    with pytest.raises(ValueError, match="only accepted when is_mm_request"):
        _chat(is_mm_request=False, task_results=[_result()])


def test_strict_mode_restores_the_400():
    """SGLANG_EPD_LOCAL_FALLBACK=0 keeps the original IC contract."""
    with envs.SGLANG_EPD_LOCAL_FALLBACK.override(False):
        with pytest.raises(ValueError, match="requires task_results"):
            _chat(is_mm_request=True)
        req = _chat(is_mm_request=True, task_results=[_result(status_code=500)])
        assert req.task_results is not None
