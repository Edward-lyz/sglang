import pickle
import unittest
from array import array
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.observability.req_time_stats import APIServerReqTimeStats
from sglang.srt.sampling.sampling_params import SamplingParams

try:
    from attentionstore_sdk.sdk import Tokens
except ImportError:
    Tokens = None

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


@unittest.skipIf(Tokens is None, "AttentionStore SDK is optional")
class TestAttentionStoreTokensIpc(CustomTestCase):
    def test_prompt_storage_is_shared_across_pickle_hops(self):
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.server_args = SimpleNamespace(enable_asradix_cache=True)
        manager.preferred_sampling_params = None
        manager.sampling_params_class = SamplingParams
        manager.tokenizer = None
        manager.model_config = SimpleNamespace(vocab_size=129280)
        prompt = [1024] + list(range(61439))
        request = GenerateReqInput(
            rid="compact-tokens",
            input_ids=prompt,
            sampling_params={"temperature": 0, "max_new_tokens": 1},
            return_logprob=False,
            logprob_start_len=61440,
            top_logprobs_num=0,
            bootstrap_room=0,
        )
        manager.rid_to_state = {
            request.rid: SimpleNamespace(time_stats=APIServerReqTimeStats())
        }
        tokenized = manager._create_tokenized_object(request, None, prompt)
        reference_hashes = Tokens(prompt, 256).hash_ids
        for hop in range(3):
            with self.subTest(hop=hop):
                self.assertIsInstance(tokenized.input_ids, array)
                self.assertIs(
                    tokenized.input_ids, tokenized.as_block_hashs.token_list
                )
                self.assertEqual(list(tokenized.input_ids), prompt)
                self.assertEqual(tokenized.as_block_hashs.hash_ids, reference_hashes)
                payload = pickle.dumps(
                    tokenized, protocol=pickle.HIGHEST_PROTOCOL
                )
                self.assertLess(len(payload), len(prompt) * 8 + 32768)
                tokenized = pickle.loads(payload)

    def test_sdk_hashes_match_at_page_boundaries(self):
        for length in (0, 1, 255, 256, 257, 55296, 61440):
            with self.subTest(length=length):
                prompt = [(200 + index) % 129280 for index in range(length)]
                packed = array("q", prompt)
                reference = Tokens(prompt, 256)
                compact = Tokens(packed, 256)
                self.assertEqual(compact.hash_ids, reference.hash_ids)
                self.assertEqual(list(compact.token_list), prompt)


if __name__ == "__main__":
    unittest.main()
