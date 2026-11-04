"""Fake KV transfer implementations for testing disaggregation without actual data movement."""

from sglang.srt.disaggregation.fake.conn import (
    FakeKVManager,
    FakeKVReceiver,
    FakeKVSender,
)
