import unittest
from unittest import mock

from sglang.srt.utils.weight_versions import WeightVersionSpan
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (
    BatchStrOutput,
    BatchTokenIDOutput,
    unwrap_from_pickle,
    wrap_as_pickle,
)
from sglang.srt.managers.multi_tokenizer_mixin import (
    MultiDetokenizerRouter,
    TokenizerWorker,
    _handle_output_by_index,
    _handle_output_by_indices,
    _NotGroupable,
    get_tokenizer_worker_class,
)

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class CustomTokenizerWorker(TokenizerWorker):
    pass


class NotAWorker:
    pass


class DefaultServerArgs:
    def get_tokenizer_worker_class(self):
        return TokenizerWorker


class CustomServerArgs:
    def get_tokenizer_worker_class(self):
        return CustomTokenizerWorker


class InvalidServerArgs:
    def get_tokenizer_worker_class(self):
        return NotAWorker


def _make_batch_str_output() -> BatchStrOutput:
    return BatchStrOutput(
        rids=["rid-0", "rid-1"],
        spec_verify_ct=[0, 0],
        spec_num_correct_drafts=[0, 0],
        spec_correct_drafts_histogram=[[], []],
        finished_reasons=[None, {"type": "length"}],
        output_strs=["first", "second"],
        output_ids=[[1], [2]],
        prompt_tokens=[10, 20],
        completion_tokens=[1, 2],
        reasoning_tokens=[0, 0],
        cached_tokens=[3, 4],
        cached_tokens_details=[
            {"device": 3, "host": 0},
            {"device": 1, "host": 3},
        ],
        input_token_logprobs_val=[[], []],
        input_token_logprobs_idx=[[], []],
        output_token_logprobs_val=[[], []],
        output_token_logprobs_idx=[[], []],
        input_top_logprobs_val=[[], []],
        input_top_logprobs_idx=[[], []],
        output_top_logprobs_val=[[], []],
        output_top_logprobs_idx=[[], []],
        input_token_ids_logprobs_val=[[], []],
        input_token_ids_logprobs_idx=[[], []],
        output_token_ids_logprobs_val=[[], []],
        output_token_ids_logprobs_idx=[[], []],
        output_token_entropy_val=[0.0, 0.0],
        output_token_sampling_mask=[[], []],
        output_token_sampling_logprobs=[[], []],
        output_hidden_states=[None, None],
        routed_experts=[None, None],
        indexer_topk=[None, None],
        placeholder_tokens_idx=[None, None],
        placeholder_tokens_val=[None, None],
        retraction_counts=[0, 0],
        weight_versions=[
            [
                WeightVersionSpan(version="v1", start=0, end=3),
                WeightVersionSpan(version="v2", start=3, end=5),
            ],
            [WeightVersionSpan(version="v2", start=0, end=2)],
        ],
    )


class TestMultiTokenizerMixin(unittest.TestCase):
    def test_batch_str_output_preserves_cached_tokens_details(self):
        output = _make_batch_str_output()

        single_output = _handle_output_by_index(output, 1)

        self.assertEqual(single_output.rids, ["rid-1"])
        self.assertEqual(single_output.cached_tokens, [4])
        self.assertEqual(
            single_output.cached_tokens_details,
            [{"device": 1, "host": 3}],
        )

    def test_batch_str_output_keeps_weight_versions_nested_per_request(self):
        """Per-request segment lists stay one level nested after the split."""
        output = _make_batch_str_output()

        self.assertEqual(
            _handle_output_by_index(output, 0).weight_versions,
            [
                [
                    WeightVersionSpan(version="v1", start=0, end=3),
                    WeightVersionSpan(version="v2", start=3, end=5),
                ]
            ],
        )
        self.assertEqual(
            _handle_output_by_index(output, 1).weight_versions,
            [[WeightVersionSpan(version="v2", start=0, end=2)]],
        )

    def test_batch_str_output_without_weight_versions_stays_none(self):
        """An output from an older server without the field splits into None."""
        output = _make_batch_str_output()
        output.weight_versions = None

        self.assertIsNone(_handle_output_by_index(output, 0).weight_versions)

    def test_get_tokenizer_worker_class_uses_default(self):
        self.assertIs(get_tokenizer_worker_class(DefaultServerArgs()), TokenizerWorker)

    def test_get_tokenizer_worker_class_resolves_custom_class(self):
        self.assertIs(
            get_tokenizer_worker_class(CustomServerArgs()),
            CustomTokenizerWorker,
        )

    def test_get_tokenizer_worker_class_rejects_non_worker(self):
        with self.assertRaisesRegex(TypeError, "TokenizerWorker"):
            get_tokenizer_worker_class(InvalidServerArgs())


def _make_batch_token_id_output(n: int) -> BatchTokenIDOutput:
    """An n-item scheduler output with every routed field populated per item."""
    return BatchTokenIDOutput(
        rids=[f"rid-{i}" for i in range(n)],
        http_worker_ipcs=[f"ipc://worker-{i % 3}" for i in range(n)],
        spec_verify_ct=list(range(n)),
        spec_num_correct_drafts=[i * 2 for i in range(n)],
        spec_correct_drafts_histogram=[[i, 1] for i in range(n)],
        time_stats=wrap_as_pickle([{"t": i} for i in range(n)]),
        finished_reasons=[None if i % 2 else {"type": "stop"} for i in range(n)],
        decoded_texts=[f"text-{i}" for i in range(n)],
        decode_ids=[[i, i + 1] for i in range(n)],
        read_offsets=list(range(n)),
        output_ids=[[100 + i] for i in range(n)],
        skip_special_tokens=[True] * n,
        spaces_between_special_tokens=[False] * n,
        no_stop_trim=[False] * n,
        prompt_tokens=[1000 + i for i in range(n)],
        completion_tokens=[i for i in range(n)],
        reasoning_tokens=[0] * n,
        cached_tokens=[i for i in range(n)],
        cached_tokens_details=[{"device": i, "host": 0} for i in range(n)],
        input_token_logprobs_val=[[] for _ in range(n)],
        input_token_logprobs_idx=[[] for _ in range(n)],
        output_token_logprobs_val=[[] for _ in range(n)],
        output_token_logprobs_idx=[[] for _ in range(n)],
        input_top_logprobs_val=[[] for _ in range(n)],
        input_top_logprobs_idx=[[] for _ in range(n)],
        output_top_logprobs_val=[[] for _ in range(n)],
        output_top_logprobs_idx=[[] for _ in range(n)],
        input_token_ids_logprobs_val=[[] for _ in range(n)],
        input_token_ids_logprobs_idx=[[] for _ in range(n)],
        output_token_ids_logprobs_val=[[] for _ in range(n)],
        output_token_ids_logprobs_idx=[[] for _ in range(n)],
        output_token_entropy_val=[0.0] * n,
        output_token_sampling_mask=[[] for _ in range(n)],
        output_token_sampling_logprobs=[[] for _ in range(n)],
        output_hidden_states=[None] * n,
        routed_experts=[None] * n,
        indexer_topk=[None] * n,
        placeholder_tokens_idx=None,
        placeholder_tokens_val=None,
        retraction_counts=[0] * n,
        customized_info=wrap_as_pickle({"k": [f"c-{i}" for i in range(n)]}),
        dp_ranks=[i % 8 for i in range(n)],
    )


def _items(output):
    """Per-item view of a split output: every list field indexed per item."""
    out = []
    for i in range(len(output.rids)):
        item = {}
        for name in output.__struct_fields__:
            v = getattr(output, name)
            if name in ("time_stats", "customized_info"):
                v = unwrap_from_pickle(v)
            if isinstance(v, dict):
                item[name] = {k: vv[i] for k, vv in v.items()}
            elif isinstance(v, list):
                item[name] = v[i]
            else:
                item[name] = v
        out.append(item)
    return out


class TestGroupedSplit(unittest.TestCase):
    def test_indices_split_equals_concatenated_single_splits(self):
        output = _make_batch_token_id_output(7)
        idx = [5, 0, 3]
        grouped = _items(_handle_output_by_indices(output, idx))
        singles = [_items(_handle_output_by_index(output, i))[0] for i in idx]
        self.assertEqual(grouped, singles)

    def test_str_output_grouped_split(self):
        output = _make_batch_str_output()
        grouped = _items(_handle_output_by_indices(output, [1, 0]))
        singles = [_items(_handle_output_by_index(output, i))[0] for i in (1, 0)]
        self.assertEqual(grouped, singles)

    def test_partial_length_field_is_not_groupable(self):
        output = _make_batch_token_id_output(4)
        output.read_offsets = [0, 1]  # covers only some items
        with self.assertRaises(_NotGroupable):
            _handle_output_by_indices(output, [1, 3])
        # The single-item split keeps its historical behavior.
        self.assertIsNone(_handle_output_by_index(output, 3).read_offsets)
        self.assertEqual(_handle_output_by_index(output, 1).read_offsets, [1])

    def test_router_sends_one_sub_batch_per_detokenizer(self):
        output = _make_batch_token_id_output(9)
        router = MultiDetokenizerRouter.__new__(MultiDetokenizerRouter)
        router.ipc_name_list = ["ipc://detok-0", "ipc://detok-1"]
        router.num_workers = 2
        router._pick_cache = {}
        sent = []
        router._send = lambda ipc, obj: sent.append((ipc, obj))
        recv = iter([output])

        def fake_recv(_sock):
            try:
                return next(recv)
            except StopIteration:
                raise KeyboardInterrupt

        with mock.patch(
            "sglang.srt.managers.multi_tokenizer_mixin.sock_recv", fake_recv
        ):
            router.recv_from_scheduler = None
            with self.assertRaises(KeyboardInterrupt):
                router.event_loop()

        targets = {router._pick(ipc) for ipc in output.http_worker_ipcs}
        self.assertEqual(len(sent), len(targets))
        got = {}
        for ipc, obj in sent:
            for item in _items(obj):
                self.assertEqual(router._pick(item["http_worker_ipcs"]), ipc)
                got[item["rids"]] = item
        expected = {
            it["rids"]: it
            for it in (
                dict(_items(_handle_output_by_index(output, i))[0],
                     http_worker_ipcs=output.http_worker_ipcs[i])
                for i in range(9)
            )
        }
        self.assertEqual(got, expected)


if __name__ == "__main__":
    unittest.main()
