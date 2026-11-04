import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import List
import unittest
from unittest.mock import Mock, patch

import torch


def load_staging_method():
    source = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/managers/scheduler.py"
    )
    scheduler = next(
        node for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "Scheduler"
    )
    method = next(
        node for node in scheduler.body
        if isinstance(node, ast.FunctionDef) and node.name == "_stage_as_swa_copies"
    )
    method.decorator_list = []
    namespace = {
        "torch": torch, "List": List, "Req": object, "logger": Mock(),
        "free_chunk_tail_swa_slots": Mock(),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_stage_as_swa_copies"]


STAGE = load_staging_method()


def make_scheduler(requests, target_copy, draft_copy, device="cpu"):
    target = SimpleNamespace(
        swa_win=128, copy_swa_page_half_batch=target_copy,
    )
    draft = SimpleNamespace(copy_swa_page_half_batch=draft_copy)
    mapping = torch.zeros(3, dtype=torch.int64, device=device)
    mapping[1] = 256
    req_to_token = torch.ones(
        (len(requests), 256), dtype=torch.int64, device=device,
    )
    cache = SimpleNamespace(
        chunk_tail_swa_pool=target, swa_policy=object(),
        full_to_swa_index_mapping=mapping, swa_page_size=256,
        swa_kv_pool=object(),
    )
    scheduler = SimpleNamespace(
        token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: cache),
        draft_worker=SimpleNamespace(primary_draft_kv_pool=SimpleNamespace(
            chunk_tail_swa_pool=draft, swa_kv_pool=object(),
        )),
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        swa_copy_stream=Mock(), prep_done_event=object(), forward_stream=object(),
        device_module=SimpleNamespace(stream=lambda stream: nullcontext()),
        _plan_swa_copies=lambda req, policy, window: (req, [0], [128]),
    )
    return scheduler


def make_requests(events):
    return [
        SimpleNamespace(
            as_kv_ready_event=event,
            kv=SimpleNamespace(req_pool_idx=index),
            as_swa_kv_slots=[index],
        )
        for index, event in enumerate(events)
    ]


class TestSwaStagingReady(unittest.TestCase):
    def test_waits_for_every_distinct_batch_event_before_both_copies(self):
        first, second = object(), object()
        order = Mock()
        requests = make_requests([first, first, second])
        scheduler = make_scheduler(requests, order.target_copy, order.draft_copy)
        stage_stream = Mock()
        stage_stream.wait_event.side_effect = order.wait_event
        with patch.object(torch.cuda, "current_stream", return_value=stage_stream):
            STAGE(scheduler, requests)
        self.assertEqual(
            [entry[0] for entry in order.mock_calls],
            ["wait_event", "wait_event", "target_copy", "draft_copy"],
        )
        self.assertEqual(stage_stream.wait_event.call_args_list[0].args, (first,))
        self.assertEqual(stage_stream.wait_event.call_args_list[1].args, (second,))
        stage_stream.wait_stream.assert_not_called()

    def test_missing_event_falls_back_to_forward_stream_before_copy(self):
        order = Mock()
        requests = make_requests([object(), None])
        scheduler = make_scheduler(requests, order.target_copy, order.draft_copy)
        stage_stream = Mock()
        stage_stream.wait_stream.side_effect = order.wait_stream
        with patch.object(torch.cuda, "current_stream", return_value=stage_stream):
            STAGE(scheduler, requests)
        self.assertEqual(
            [entry[0] for entry in order.mock_calls],
            ["wait_stream", "target_copy", "draft_copy"],
        )
        stage_stream.wait_stream.assert_called_once_with(scheduler.forward_stream)
        stage_stream.wait_event.assert_not_called()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_delayed_draft_producer_is_visible_to_real_cuda_copies(self):
        device = torch.device("cuda:0")
        source = torch.zeros((2, 4096), dtype=torch.uint8, device=device)
        staged = torch.zeros_like(source)
        producer = torch.cuda.Stream(device=device)
        ready = torch.cuda.Event()
        requests = make_requests([ready])
        scheduler = make_scheduler(
            requests,
            lambda *args, **kwargs: staged[0].copy_(source[0]),
            lambda *args, **kwargs: staged[1].copy_(source[1]),
            device=device,
        )
        scheduler.swa_copy_stream = torch.cuda.Stream(device=device)
        scheduler.prep_done_event = torch.cuda.Event()
        scheduler.forward_stream = producer
        scheduler.device_module = torch.cuda
        scheduler.prep_done_event.record()
        torch.cuda.synchronize(device)
        with torch.cuda.stream(producer):
            torch.cuda._sleep(500_000_000)
            source.fill_(173)
            ready.record()
        self.assertFalse(ready.query())
        STAGE(scheduler, requests)
        torch.cuda.current_stream(device).synchronize()
        self.assertTrue(torch.all(staged == 173).item())


if __name__ == "__main__":
    unittest.main()
