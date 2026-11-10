from types import SimpleNamespace
from unittest import mock

import pytest

from sglang.srt.mem_cache.storage.asradix.cache import AttentionStoreUnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def fixture(available, capacity=960, after=960):
    pool = SimpleNamespace(
        num_pages=capacity, available_size=mock.Mock(return_value=available)
    )
    cache = SimpleNamespace(
        attention_store_plan=SimpleNamespace(name="SPAG"), pp_world_size=1,
        _all_reduce_attn_groups=mock.Mock(),
        _as_min=mock.Mock(return_value=after),
        drain_attention_store_writes=mock.Mock(),
    )
    return cache, pool


def run(cache, pool, required):
    return AttentionStoreUnifiedRadixCache.ensure_attention_store_swa_staging_capacity(
        cache, pool, required
    )


@pytest.mark.parametrize("required", [0, 117, 960])
def test_complete_capacity_does_not_wait_on_the_writer(required):
    cache, pool = fixture(960)
    run(cache, pool, required)
    cache._all_reduce_attn_groups.assert_called_once()
    cache.drain_attention_store_writes.assert_not_called()
    cache._as_min.assert_not_called()


def test_shortage_drains_before_retrying_complete_capacity():
    cache, pool = fixture(24, after=960)
    order = []
    cache._all_reduce_attn_groups.side_effect = lambda state, op: order.append("agree")
    cache.drain_attention_store_writes.side_effect = lambda: order.append("drain")
    cache._as_min.side_effect = lambda available: order.append("recheck") or 960
    run(cache, pool, 117)
    assert order == ["agree", "drain", "recheck"]


def test_another_rank_shortage_also_drains_this_rank():
    cache, pool = fixture(960)
    def reduce(state, op):
        state[0] = 24
        state[1] = -234
    cache._all_reduce_attn_groups.side_effect = reduce
    run(cache, pool, 117)
    cache.drain_attention_store_writes.assert_called_once()


def test_impossible_batch_fails_before_allocating_any_partial_backups():
    cache, pool = fixture(960)
    with pytest.raises(RuntimeError, match="pool capacity"):
        run(cache, pool, 961)
    cache.drain_attention_store_writes.assert_not_called()


def test_unreleased_inflight_capacity_fails_without_invalid_sentinel_publication():
    cache, pool = fixture(24, after=116)
    with pytest.raises(RuntimeError, match="after draining"):
        run(cache, pool, 117)
    cache.drain_attention_store_writes.assert_called_once()


@pytest.mark.parametrize("plan,pp", [("SPA", 1), ("KDA", 1), ("SPAG", 2)])
def test_other_plans_and_pipeline_protocols_keep_their_own_lifecycle(plan, pp):
    cache, pool = fixture(24)
    cache.attention_store_plan.name = plan
    cache.pp_world_size = pp
    run(cache, pool, 117)
    cache._all_reduce_attn_groups.assert_not_called()
    cache.drain_attention_store_writes.assert_not_called()
