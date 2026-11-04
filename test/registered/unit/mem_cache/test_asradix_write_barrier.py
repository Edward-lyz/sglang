"""页写路径必须在读 HBM 之前等生产者事件。

线上现象:引擎侧写摘要读到全零页或上一个占用者的内容,而 SDK 的 gather kernel
几毫秒后拿到的是最终数据 —— 说明发起写的时候页还在被填充。谁先跑纯看调度,
gather 抢在生产者前面时脏数据会直接进缓存(2026-09-09 实测 4 例)。

同时覆盖摘要日志的运行时开关:它带一次 D2H 拷贝和一次哈希,默认关闭。
"""

import sys
import unittest
from types import SimpleNamespace

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    sys.modules["torch"] = SimpleNamespace(Tensor=object)

if "attentionstore_sdk" not in sys.modules:

    class _BlockBatch:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _WriteRequest:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    sys.modules["attentionstore_sdk"] = SimpleNamespace()
    sys.modules["attentionstore_sdk.sdk"] = SimpleNamespace(
        BlockBatch=_BlockBatch,
        WriteRequest=_WriteRequest,
    )

from sglang.srt.mem_cache.storage.asradix import transport as transport_mod
from sglang.srt.mem_cache.storage.asradix.page_coordinator import (
    FullPageCacheCoordinator,
    _PageWriteTask,
)
from sglang.srt.mem_cache.storage.asradix.transport import _BufferPageIO


class _Page:
    """一页 HBM 显存:生产者算完之前读到旧内容,算完之后读到最终内容。"""

    def __init__(self, page_id: int):
        self.page_id = page_id
        self.finalized = False

    def content(self) -> str:
        return ("final-%d" if self.finalized else "stale-%d") % self.page_id


class _ProducerEvent:
    """模拟 CUDA event。

    第一次轮询时生产者还没写完,之后才完成 —— 只有真的等过这个事件的消费者
    才能读到最终内容,没等的消费者必然读到旧内容。
    """

    def __init__(self, pages):
        self._pages = pages
        self._polled = False

    def query(self) -> bool:
        if not self._polled:
            self._polled = True
            return False
        for page in self._pages:
            page.finalized = True
        return True


class _RecordingSdk:
    """代替真 SDK:在 write 被调用的那一刻记录页里的内容。"""

    def __init__(self, pages):
        self._pages = pages
        self.gathered = None

    def write(self, request) -> int:
        self.gathered = [
            self._pages[page_id].content()
            for block in request.blocks.page_ids
            for page_id in block
        ]
        return len(request.blocks.hash_ids)


class _RecordingStore:
    def __init__(self):
        self.kwargs = None

    def write(self, block_hash_ids, page_ids, **kwargs) -> int:
        self.kwargs = kwargs
        return len(block_hash_ids)


class TestAttentionStorePageWriteBarrier(unittest.TestCase):
    def setUp(self):
        self.pages = {page_id: _Page(page_id) for page_id in range(4)}
        self.sdk = _RecordingSdk(self.pages)
        self._digest_enabled = transport_mod.cache_digest_enabled
        transport_mod.cache_digest_enabled = lambda: False

    def tearDown(self):
        transport_mod.cache_digest_enabled = self._digest_enabled

    def _io(self):
        return _BufferPageIO(
            sdk=self.sdk,
            buffers=SimpleNamespace(),
            layer_ids=(0,),
            buffer_tensors={"k": (object(),)},
            buffer_pools={"k": (SimpleNamespace(size=1024, page_size=64),)},
            local_page_size=64,
            block_size=256,
        )

    def test_gather_without_producer_event_reads_unfinished_page(self):
        """没有事件可等时,gather 读到的就是还没写完的页 —— 线上脏数据的形状。"""
        self._io().write((11,), (0, 1, 2, 3))

        self.assertEqual(
            self.sdk.gathered,
            ["stale-0", "stale-1", "stale-2", "stale-3"],
        )

    def test_gather_sees_final_content_only_if_write_waits(self):
        event = _ProducerEvent(self.pages.values())

        self._io().write((11,), (0, 1, 2, 3), producer_ready=event)

        self.assertEqual(
            self.sdk.gathered,
            ["final-0", "final-1", "final-2", "final-3"],
        )

    def test_do_write_forwards_the_producer_event(self):
        store = _RecordingStore()
        event = _ProducerEvent(())
        task = _PageWriteTask(
            rid="r0",
            endpoint=SimpleNamespace(store=store),
            block_hash_ids=(11,),
            data_id_list=SimpleNamespace(materialize=lambda: (0, 1, 2, 3)),
            write_params={},
            producer_ready=event,
        )

        FullPageCacheCoordinator().do_write(
            SimpleNamespace(_log_as_result=lambda *a, **k: None),
            task,
        )

        self.assertIs(store.kwargs.get("producer_ready"), event)



if __name__ == "__main__":
    unittest.main()
