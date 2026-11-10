"""PP + EPD: embeddings never travel down the pipeline, metadata does.

Only the first PP stage embeds tokens. Precomputed EPD rows are zero-copy views
of the pinned landing pool, so forwarding them would pickle the whole pool;
later stages only need the offsets/hashes/pad values.
"""

import pickle

import torch

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.managers.scheduler_pp_mixin import strip_mm_embeddings_for_next_stage


class _Req:
    """Stand-in carrying the one field the stripper touches."""

    def __init__(self, mm_inputs):
        self.rid = "r"
        self.mm_inputs = mm_inputs


def _mm_inputs():
    pool = torch.zeros(1 << 20, dtype=torch.bfloat16)  # stands in for the pool
    rows = pool[: 6 * 4].view(6, 4)
    item = MultimodalDataItem(modality=Modality.IMAGE, offsets=[(3, 8)])
    item.precomputed_embeddings = rows
    item.hash = 123
    mm = MultimodalInputs(mm_items=[item])
    mm.pool_buffer_addrs = [0xABC]
    return mm


def test_downstream_copy_has_metadata_but_no_rows():
    mm = _mm_inputs()
    req = _Req(mm)
    out = strip_mm_embeddings_for_next_stage([req, "not-a-req"])
    sent = out[0]
    assert sent is not req and out[1] == "not-a-req"
    item = sent.mm_inputs.mm_items[0]
    assert item.precomputed_embeddings is None
    assert item.offsets == [(3, 8)] and item.hash == 123
    assert sent.mm_inputs.pool_buffer_addrs is None
    # This stage still owns and uses the rows.
    assert req.mm_inputs.mm_items[0].precomputed_embeddings is not None
    assert req.mm_inputs.pool_buffer_addrs == [0xABC]
    # And the payload no longer drags the 2 MB backing storage along.
    assert len(pickle.dumps(sent.mm_inputs)) < 64 * 1024
    assert len(pickle.dumps(req.mm_inputs)) > 1 << 20


def test_requests_without_rows_are_passed_through_untouched():
    plain = _Req(None)
    data = [plain]
    assert strip_mm_embeddings_for_next_stage(data) is data
    assert strip_mm_embeddings_for_next_stage({"x": 1}) == {"x": 1}


def test_real_tokenized_request_is_copied_not_mutated():
    """The real IPC struct (msgspec) must survive the shallow copy."""
    from sglang.srt.managers.io_struct import TokenizedGenerateReqInput

    mm = _mm_inputs()
    req = TokenizedGenerateReqInput(
        input_text="x", input_ids=[1, 2, 3], input_embeds=None, mm_inputs=mm,
        token_type_ids=None, sampling_params=None, return_logprob=False,
        logprob_start_len=-1, top_logprobs_num=0, token_ids_logprob=None,
        stream=False,
    )
    (sent,) = strip_mm_embeddings_for_next_stage([req])
    assert isinstance(sent, TokenizedGenerateReqInput) and sent is not req
    assert sent.input_ids == [1, 2, 3]
    assert sent.mm_inputs.mm_items[0].precomputed_embeddings is None
    assert req.mm_inputs.mm_items[0].precomputed_embeddings is not None
