"""
Unit tests for the parallel scheduler-launch gating.

Verifies use_parallel_scheduler_launch is enabled only when the flag is set,
the start method is "spawn", and no per-child global-state launch feature
(data-parallel controller, memory saver, one-visible-device remap) is active,
and that _make_scheduler_process keeps its argument tuple aligned with the
run_scheduler_process signature.
"""

import dataclasses
import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.entrypoints.engine import _make_scheduler_process
from sglang.srt.environ import envs
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.numa_utils import use_parallel_scheduler_launch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _server_args(**overrides):
    args = dict(dp_size=1, ep_join_mode="prefill", enable_memory_saver=False)
    args.update(overrides)
    return SimpleNamespace(**args)


def _spawn():
    return patch("multiprocessing.get_start_method", return_value="spawn")


class TestParallelSchedulerLaunchGating(CustomTestCase):
    def test_disabled_when_flag_off(self):
        with envs.SGLANG_PARALLEL_SCHEDULER_LAUNCH.override(False):
            self.assertFalse(use_parallel_scheduler_launch(_server_args()))

    def test_enabled_when_flag_on_and_clear(self):
        with envs.SGLANG_PARALLEL_SCHEDULER_LAUNCH.override(
            True
        ), envs.SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS.override(False), _spawn():
            self.assertTrue(use_parallel_scheduler_launch(_server_args()))

    def test_disabled_under_fork_start_method(self):
        with envs.SGLANG_PARALLEL_SCHEDULER_LAUNCH.override(
            True
        ), envs.SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS.override(False), patch(
            "multiprocessing.get_start_method", return_value="fork"
        ):
            self.assertFalse(use_parallel_scheduler_launch(_server_args()))

    def test_disabled_by_incompatible_features(self):
        with envs.SGLANG_PARALLEL_SCHEDULER_LAUNCH.override(
            True
        ), envs.SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS.override(False), _spawn():
            self.assertFalse(use_parallel_scheduler_launch(_server_args(dp_size=2)))
            self.assertFalse(
                use_parallel_scheduler_launch(_server_args(ep_join_mode="scale"))
            )
            self.assertFalse(
                use_parallel_scheduler_launch(_server_args(enable_memory_saver=True))
            )

    def test_disabled_by_one_visible_device(self):
        with envs.SGLANG_PARALLEL_SCHEDULER_LAUNCH.override(
            True
        ), envs.SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS.override(True), _spawn():
            self.assertFalse(use_parallel_scheduler_launch(_server_args()))

    def test_real_server_args_has_expected_fields(self):
        # Guard against the mock drifting from the real ServerArgs contract.
        names = {f.name for f in dataclasses.fields(ServerArgs)}
        self.assertTrue({"dp_size", "ep_join_mode", "enable_memory_saver"} <= names)


class TestMakeSchedulerProcess(CustomTestCase):
    def _process(self, **overrides):
        args = dict(
            server_args=object(),
            port_args=object(),
            gpu_id=3,
            tp_rank=1,
            attn_cp_rank=0,
            moe_dp_rank=0,
            moe_ep_rank=0,
            pp_rank=2,
            pipe_writer=object(),
        )
        args.update(overrides)
        return _make_scheduler_process(run_scheduler_process, **args)

    def test_positional_args_align_with_target_signature(self):
        # The args tuple is positional, so a signature change on either side
        # would silently shift ranks; bind the two together here.
        proc = self._process()
        params = list(inspect.signature(run_scheduler_process).parameters)
        bound = dict(zip(params, proc._args))
        self.assertEqual(len(proc._args), 10)
        self.assertEqual(bound["gpu_id"], 3)
        self.assertEqual(bound["tp_rank"], 1)
        self.assertEqual(bound["pp_rank"], 2)
        self.assertIsNone(bound["dp_rank"])
        self.assertIs(bound["pipe_writer"], proc._args[9])

    def test_bind_numa_in_child_only_sent_when_enabled(self):
        self.assertEqual(self._process()._kwargs, {})
        self.assertEqual(
            self._process(bind_numa_in_child=True)._kwargs,
            {"bind_numa_in_child": True},
        )


if __name__ == "__main__":
    unittest.main()
