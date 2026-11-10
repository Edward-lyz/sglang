"""SGLANG_OPT_MM_PAD_INPUT_IDS_ARRAY must pad exactly like the tensor path.

The padded ids are the prefix-cache key and the AttentionStore block-hash
input, so any divergence (an off-by-one at a span end, a skipped modality, a
mutated input) silently changes cache hits between the two paths.
"""

import unittest
from array import array

from sglang.srt.environ import envs
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

IMAGE_TOKEN = 9


def _item(modality, offsets, pad_value):
    item = MultimodalDataItem(modality=modality, offsets=offsets)
    item.pad_value = pad_value
    return item


def _inputs(items):
    return MultimodalInputs(mm_items=items, im_token_id=IMAGE_TOKEN)


class TestPadInputIdsArray(CustomTestCase):
    def _pad(self, ids, mm_inputs, *, use_array):
        with envs.SGLANG_OPT_MM_PAD_INPUT_IDS_ARRAY.override(use_array):
            return MultiModalityDataPaddingPatternMultimodalTokens().pad_input_tokens(
                ids, mm_inputs
            )

    def test_matches_tensor_path_and_keeps_input(self):
        ids = array("q", [1, IMAGE_TOKEN, IMAGE_TOKEN, 2, IMAGE_TOKEN, 3, IMAGE_TOKEN])
        snapshot = list(ids)
        mm_inputs = _inputs(
            [
                _item(Modality.IMAGE, [(1, 2)], 1001),
                # Last span ends on the final token: the inclusive end must be kept.
                _item(Modality.IMAGE, [(4, 4), (6, 6)], 1002),
                # No token id for audio here, so the tensor path leaves it untouched.
                _item(Modality.AUDIO, [(0, 0)], 1003),
            ]
        )
        expected = self._pad(list(ids), mm_inputs, use_array=False)
        got = self._pad(ids, mm_inputs, use_array=True)
        self.assertEqual(list(got), expected)
        self.assertEqual(got.typecode, "q")
        self.assertEqual(list(ids), snapshot)


if __name__ == "__main__":
    unittest.main()
