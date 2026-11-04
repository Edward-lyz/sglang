import sys
import unittest
from types import SimpleNamespace

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    sys.modules["torch"] = SimpleNamespace(Tensor=object)

from sglang.srt.mem_cache.storage.asradix.transport import _BufferPageIO


class _FakeTensor:
    def __init__(self, shape, stride0, element_size=4, data_ptr=1000):
        self.shape = shape
        self._stride0 = stride0
        self._element_size = element_size
        self._data_ptr = data_ptr

    def stride(self, dim):
        assert dim == 0
        return self._stride0

    def element_size(self):
        return self._element_size

    def data_ptr(self):
        return self._data_ptr


class TestAttentionStoreBlockStaging(unittest.TestCase):
    def _io(self):
        return _BufferPageIO(
            sdk=object(),
            buffers=SimpleNamespace(),
            layer_ids=(0,),
            buffer_tensors={"k": (_FakeTensor((16, 64, 2), 128),)},
            buffer_pools={"k": (SimpleNamespace(size=1024, page_size=64),)},
            local_page_size=64,
            block_size=256,
        )

    def test_groups_four_local_pages_per_attention_store_block(self):
        io = self._io()

        self.assertEqual(
            io._logical_blocks((11, 22), (4, 5, 6, 7, 13, 14, 15, 16)),
            ((11, (4, 5, 6, 7)), (22, (13, 14, 15, 16))),
        )

    def test_accepts_non_consecutive_local_pages(self):
        io = self._io()

        self.assertEqual(
            io._logical_blocks((11,), (4, 5, 7, 8)),
            ((11, (4, 5, 7, 8)),),
        )


if __name__ == "__main__":
    unittest.main()
