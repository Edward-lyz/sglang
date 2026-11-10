"""CPU checks of the actual prefill worker's producer-event and warmup chain."""

import ast
import queue
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch


def load_forward_methods(fence_enabled=True):
    source = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/disaggregation/prefill.py"
    )
    names = {
        "_ensure_prefill_fwd_thread",
        "_prefill_fwd_thread_main",
        "_fwd_collect",
    }
    methods = [
        node
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    for method in methods:
        method.decorator_list = []
    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                *methods,
            ],
            type_ignores=[],
        )
    )
    namespace = {
        "torch": torch,
        "logger": Mock(),
        "envs": SimpleNamespace(
            SGLANG_AS_SPAG_BATCH_WRITE_FENCE=SimpleNamespace(
                get=lambda: fence_enabled
            ),
            SGLANG_OPT_PREFILL_FWD_INFERENCE_MODE=SimpleNamespace(get=lambda: True),
        ),
    }
    exec(compile(module, str(source), "exec"), namespace)
    return namespace


def make_batch(order):
    batch = SimpleNamespace(
        reqs=[SimpleNamespace(as_kv_ready_event=None) for _ in range(2)],
        as_kv_ready_event=None,
    )

    def copy_batch():
        order.append("copy")
        return SimpleNamespace(
            reqs=batch.reqs, as_kv_ready_event=batch.as_kv_ready_event
        )

    batch.copy = copy_batch
    return batch


def make_worker(batches, order):
    incoming = queue.Queue()
    for batch in batches:
        incoming.put((batch, True, [256, 512]))
    incoming.put(None)
    events = []
    forward_stream = object()

    def make_event():
        event = Mock()
        event.record.side_effect = lambda **kwargs: order.append("record")
        events.append(event)
        return event

    def run_batch(batch, **kwargs):
        if not torch.is_inference_mode_enabled():
            raise AssertionError("The worker must preserve inference_mode")
        order.append("forward")
        return object()

    worker = SimpleNamespace(
        ps=SimpleNamespace(gpu_id=0),
        device_module=SimpleNamespace(
            set_device=Mock(), Event=Mock(side_effect=make_event)
        ),
        forward_stream=forward_stream,
        _fwd_in_q=incoming,
        _fwd_out_q=queue.Queue(),
        run_batch=Mock(side_effect=run_batch),
        _apply_war_barrier=Mock(side_effect=lambda: order.append("war")),
    )
    return worker, events


class TestPrefillForwardReady(unittest.TestCase):
    def test_records_per_batch_before_snapshot_and_transfers_to_requests(self):
        methods = load_forward_methods()
        order = []
        batches = [make_batch(order), make_batch(order)]
        worker, events = make_worker(batches, order)

        methods["_prefill_fwd_thread_main"](worker)

        self.assertEqual(order, ["forward", "record", "war", "copy"] * 2)
        self.assertEqual(len(events), 2)
        self.assertIsNot(events[0], events[1])
        for batch, event in zip(batches, events):
            event.record.assert_called_once_with(stream=worker.forward_stream)
            self.assertTrue(all(req.as_kv_ready_event is None for req in batch.reqs))
            _, collected = methods["_fwd_collect"](worker)
            self.assertIs(collected.as_kv_ready_event, event)
            self.assertTrue(all(req.as_kv_ready_event is event for req in batch.reqs))
        self.assertTrue(worker._fwd_out_q.empty())

    def test_disabled_fence_creates_no_event_and_keeps_fallback(self):
        methods = load_forward_methods(fence_enabled=False)
        order = []
        batch = make_batch(order)
        worker, events = make_worker([batch], order)

        methods["_prefill_fwd_thread_main"](worker)
        _, collected = methods["_fwd_collect"](worker)

        self.assertEqual(order, ["forward", "war", "copy"])
        self.assertEqual(events, [])
        self.assertIsNone(collected.as_kv_ready_event)
        self.assertTrue(all(req.as_kv_ready_event is None for req in batch.reqs))

    def test_forward_failure_is_relayed_without_publishing_an_event(self):
        methods = load_forward_methods()
        batch = make_batch([])
        worker, events = make_worker([batch], [])
        failure = RuntimeError("forward failed")
        worker.run_batch.side_effect = failure

        methods["_prefill_fwd_thread_main"](worker)
        with self.assertRaises(RuntimeError) as caught:
            methods["_fwd_collect"](worker)

        self.assertIs(caught.exception, failure)
        self.assertEqual(events, [])
        worker._apply_war_barrier.assert_not_called()
        self.assertTrue(all(req.as_kv_ready_event is None for req in batch.reqs))

    def test_paged_cuda_allocator_warms_once_before_starting_thread(self):
        methods = load_forward_methods()
        order = Mock()
        paged_module = ModuleType("sglang.srt.mem_cache.allocator.paged")
        paged_module.warmup_alloc_extend_kernel = order.warmup
        worker = SimpleNamespace(
            device="cuda:0",
            token_to_kv_pool_allocator=SimpleNamespace(page_size=256),
            max_running_requests=128,
            _prefill_fwd_thread_main=Mock(),
        )
        with (
            patch.dict(sys.modules, {paged_module.__name__: paged_module}),
            patch("threading.Thread") as thread,
        ):
            thread.return_value.start.side_effect = order.start
            methods["_ensure_prefill_fwd_thread"](worker)
            methods["_ensure_prefill_fwd_thread"](worker)
            thread.assert_called_once()
            thread.return_value.start.assert_called_once()

        order.warmup.assert_called_once_with("cuda:0", 256, 128)
        self.assertEqual([call[0] for call in order.mock_calls], ["warmup", "start"])

    def test_cpu_and_unpaged_allocators_skip_warmup(self):
        methods = load_forward_methods()
        paged_module = ModuleType("sglang.srt.mem_cache.allocator.paged")
        paged_module.warmup_alloc_extend_kernel = Mock()
        for device, page_size in [("cpu", 256), ("cuda:0", 1)]:
            with self.subTest(device=device, page_size=page_size):
                worker = SimpleNamespace(
                    device=device,
                    token_to_kv_pool_allocator=SimpleNamespace(page_size=page_size),
                    max_running_requests=128,
                    _prefill_fwd_thread_main=Mock(),
                )
                with (
                    patch.dict(sys.modules, {paged_module.__name__: paged_module}),
                    patch("threading.Thread") as thread,
                ):
                    methods["_ensure_prefill_fwd_thread"](worker)
                    thread.return_value.start.assert_called_once()
        paged_module.warmup_alloc_extend_kernel.assert_not_called()

    def test_failed_warmup_prevents_starting_the_forward_thread(self):
        methods = load_forward_methods()
        failure = RuntimeError("warmup failed")
        paged_module = ModuleType("sglang.srt.mem_cache.allocator.paged")
        paged_module.warmup_alloc_extend_kernel = Mock(side_effect=failure)
        worker = SimpleNamespace(
            device="cuda:0",
            token_to_kv_pool_allocator=SimpleNamespace(page_size=256),
            max_running_requests=128,
        )
        with (
            patch.dict(sys.modules, {paged_module.__name__: paged_module}),
            patch("threading.Thread") as thread,
        ):
            with self.assertRaises(RuntimeError) as caught:
                methods["_ensure_prefill_fwd_thread"](worker)
            self.assertIs(caught.exception, failure)
            thread.assert_not_called()


if __name__ == "__main__":
    unittest.main()
