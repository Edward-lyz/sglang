import unittest
from types import MethodType, SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.aux_hidden_states import pack_aux_hidden_states
from sglang.srt.models.dspark import DSparkDraftMixin
from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
    TargetHiddenKvInjector,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _Attention:
    num_kv_heads = 1
    head_dim = 3

    def __init__(self) -> None:
        self.attn = SimpleNamespace(k_scale=0.5, v_scale=0.25)
        self.input = None

    def kv_proj_only(self, hidden_states: torch.Tensor):
        self.input = hidden_states
        return hidden_states + 1, hidden_states + 2

    def apply_k_norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + 3

    def apply_k_rope(
        self, _positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        return hidden_states + 4


class DSparkTargetHiddenProjectionTest(CustomTestCase):
    def test_prefill_snapshot_tracks_tail_and_rank_major_rows(self) -> None:
        """SWA injection survives mapping reuse between target metadata and its output."""
        from sglang.srt.environ import envs
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
            DSparkWorkerV2,
        )

        mapping = torch.arange(20, dtype=torch.int64) * 7
        pool = SimpleNamespace(
            set_swa_key_buffer_radix_fused_norm_rope=mock.Mock(),
            translate_loc_from_full_to_swa=mock.Mock(
                side_effect=lambda indices: mapping[indices]
            ),
        )
        draft_model = SimpleNamespace(write_target_hidden_kv=mock.Mock())
        runner = SimpleNamespace(
            device=torch.device("cpu"), prefill_attention_backend_str="dsv4"
        )
        injector = TargetHiddenKvInjector(
            draft_model=draft_model,
            draft_model_runner=SimpleNamespace(token_to_kv_pool=pool),
            model_runner=runner,
            device=torch.device("cpu"),
            verify_num_draft_tokens=2,
            block_pos_offsets=torch.arange(2),
        )
        cache_loc = torch.tensor([2, 4, 7, 9, 12])
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            seq_lens=torch.tensor([10, 12]),
            prefix_lens=[7, 10], extend_lens=[3, 2],
            out_cache_loc=cache_loc,
        )
        output = SimpleNamespace(
            next_token_ids=torch.tensor([1, 2]),
            logits_output=SimpleNamespace(
                hidden_states_row_token_index=torch.tensor([2, 0, 1, 3]),
                hidden_states_rank_major=torch.arange(12).reshape(4, 3).float(),
                hidden_states=None,
                hidden_states_token_indices=torch.tensor([1, 2, 4]),
                dspark_commit_mask=None,
            ),
        )

        def finish_target(*args, **kwargs):
            mapping.fill_(999)
            return output

        worker = SimpleNamespace()
        worker.target_worker = SimpleNamespace(
            forward_batch_generation=mock.Mock(side_effect=finish_target),
            model_runner=runner,
        )
        worker.model_runner = runner
        worker._kv_injector = injector
        worker._target_hidden_projection_enabled = False
        worker._tp_sync = SimpleNamespace(sync=mock.Mock())
        module = "sglang.srt.speculative.dspark_components"
        with (
            envs.SGLANG_ENABLE_PREFILL_WAR_READ_DONE.override(True),
            mock.patch(module + ".dspark_kv_inject.is_unified_kv_triton", return_value=False),
            mock.patch(module + ".dspark_worker_v2.is_unified_kv_triton", return_value=False),
            mock.patch(module + ".dspark_worker_v2.is_pin_memory_available", return_value=False),
            mock.patch(
                module + ".dspark_worker_v2.compute_position",
                return_value=(torch.tensor([7, 8, 9, 10, 11]), None),
            ),
            mock.patch(module + ".dspark_worker_v2.make_next_draft_input"),
        ):
            DSparkWorkerV2._forward_prefill(worker, batch, on_publish=None)
        pool.translate_loc_from_full_to_swa.assert_called_once_with(cache_loc)
        write = draft_model.write_target_hidden_kv.call_args.kwargs
        torch.testing.assert_close(
            write["swa_loc"], torch.tensor([84, 28, 49, -1], dtype=torch.int32)
        )
        torch.testing.assert_close(write["positions"], torch.tensor([11, 8, 9, 8]))

    def test_single_aux_hidden_state_is_returned_without_copy(self) -> None:
        hidden_states = torch.empty(2, 3)

        self.assertIs(hidden_states, pack_aux_hidden_states([hidden_states]))

    def test_preprojected_hidden_is_not_projected_again(self) -> None:
        attention = _Attention()
        draft_model = SimpleNamespace(
            layers=[SimpleNamespace(self_attn=attention)],
            project_target_hidden=mock.Mock(
                side_effect=AssertionError("projection must not run twice")
            ),
            _fused_kv_write_bundle=lambda _pool: None,
            _stacked_ctx_kv_params=lambda: None,
        )
        draft_model.write_target_hidden_kv = MethodType(
            DSparkDraftMixin.write_target_hidden_kv, draft_model
        )
        pool = SimpleNamespace(set_kv_buffer=mock.Mock())
        injector = TargetHiddenKvInjector(
            draft_model=draft_model,
            draft_model_runner=SimpleNamespace(token_to_kv_pool=pool),
            model_runner=SimpleNamespace(device=torch.device("cpu")),
            device=torch.device("cpu"),
            verify_num_draft_tokens=2,
            block_pos_offsets=torch.arange(2),
        )
        projected_hidden = torch.arange(6, dtype=torch.float32).reshape(2, 3)

        injector.inject_target_hidden(
            target_hidden=projected_hidden,
            cache_loc=torch.arange(2),
            positions=torch.arange(2),
            target_hidden_is_projected=True,
        )

        draft_model.project_target_hidden.assert_not_called()
        self.assertIs(attention.input, projected_hidden)


if __name__ == "__main__":
    unittest.main()
