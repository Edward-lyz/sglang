from __future__ import annotations

import logging
import time
import warnings
from typing import TYPE_CHECKING

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.utils.common import ceil_align, raise_error_or_warn
from sglang.srt.utils.request_logger import disable_request_logging
from sglang.srt.utils.watchdog import WatchdogRaw

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class SchedulerRuntimeCheckerMixin:
    def _get_token_info(self: Scheduler):
        available_size = self.token_to_kv_pool_allocator.available_size()
        evictable_size = self.tree_cache.evictable_size()
        num_used = self.max_total_num_tokens - (available_size + evictable_size)
        token_usage = num_used / self.max_total_num_tokens
        return num_used, token_usage, available_size, evictable_size

    def _get_mamba_token_info(self: Scheduler):
        is_mamba_radix_cache = (
            self.tree_cache.supports_mamba() and self.tree_cache.is_tree_cache()
        )
        full_available_size = self.token_to_kv_pool_allocator.available_size()
        full_evictable_size = (
            self.tree_cache.full_evictable_size() if is_mamba_radix_cache else 0
        )
        mamba_available_size = self.req_to_token_pool.mamba_pool.available_size()
        mamba_evictable_size = (
            self.tree_cache.mamba_evictable_size() if is_mamba_radix_cache else 0
        )
        full_num_used = self.token_to_kv_pool_allocator.size - (
            full_available_size + full_evictable_size
        )
        mamba_num_used = self.req_to_token_pool.mamba_pool.size - (
            mamba_available_size + mamba_evictable_size
        )
        full_token_usage = full_num_used / self.token_to_kv_pool_allocator.size
        mamba_usage = mamba_num_used / self.req_to_token_pool.mamba_pool.size
        return (
            full_num_used,
            mamba_num_used,
            full_token_usage,
            mamba_usage,
            full_available_size,
            full_evictable_size,
            mamba_available_size,
            mamba_evictable_size,
        )

    def _get_swa_token_info(self: Scheduler):
        full_available_size = self.token_to_kv_pool_allocator.full_available_size()
        full_evictable_size = self.tree_cache.full_evictable_size()
        swa_available_size = self.token_to_kv_pool_allocator.swa_available_size()
        swa_evictable_size = self.tree_cache.swa_evictable_size()
        full_num_used = self.full_tokens_per_layer - (
            full_available_size + full_evictable_size
        )
        swa_num_used = self.swa_tokens_per_layer - (
            swa_available_size + swa_evictable_size
        )
        full_token_usage = full_num_used / self.full_tokens_per_layer
        swa_token_usage = swa_num_used / self.swa_tokens_per_layer
        return (
            full_num_used,
            swa_num_used,
            full_token_usage,
            swa_token_usage,
            full_available_size,
            full_evictable_size,
            swa_available_size,
            swa_evictable_size,
        )

    def _check_hybrid_memory(self: Scheduler):
        (
            full_num_used,
            swa_num_used,
            _,
            _,
            full_available_size,
            full_evictable_size,
            swa_available_size,
            swa_evictable_size,
        ) = self._get_swa_token_info()
        memory_leak = full_num_used != 0 or swa_num_used != 0
        token_msg = (
            f"{self.full_tokens_per_layer=}, {full_available_size=}, {full_evictable_size=}, {self.tree_cache.full_protected_size()=}\n"
            f"{self.swa_tokens_per_layer=}, {swa_available_size=}, {swa_evictable_size=}, {self.tree_cache.swa_protected_size()=}\n"
        )
        return memory_leak, token_msg

    def _check_mamba_memory(self: Scheduler):
        (
            full_num_used,
            mamba_num_used,
            _,
            _,
            full_available_size,
            full_evictable_size,
            mamba_available_size,
            mamba_evictable_size,
        ) = self._get_mamba_token_info()
        memory_leak = (
            full_num_used != self.tree_cache.full_protected_size()
            or mamba_num_used != self.tree_cache.mamba_protected_size()
        )
        if memory_leak:
            free_full_pages = set(
                self.token_to_kv_pool_allocator.free_pages.tolist()
                + self.token_to_kv_pool_allocator.release_pages.tolist()
            )
            cached_full_pages = set(self.tree_cache.all_values_flatten().tolist())
            expected_full_pages = set(
                range(1, self.token_to_kv_pool_allocator.size + 1)
            )
            leaked_full_pages = (
                expected_full_pages - free_full_pages - cached_full_pages
            )
            free_mamba_pages = set(
                self.req_to_token_pool.mamba_pool.free_slots.tolist()
            )
            cached_mamba_pages = set(
                self.tree_cache.all_mamba_values_flatten().tolist()
            )
            expected_mamba_pages = set(range(self.req_to_token_pool.mamba_pool.size))
            leaked_mamba_pages = (
                expected_mamba_pages - free_mamba_pages - cached_mamba_pages
            )
            token_msg = (
                f"{full_available_size=}, {full_evictable_size=}, {self.token_to_kv_pool_allocator.size=}, {self.tree_cache.full_protected_size()=}\n"
                f"{mamba_available_size=}, {mamba_evictable_size=}, {self.req_to_token_pool.mamba_pool.size=}, {self.tree_cache.mamba_protected_size()=}, leaked_full_pages={leaked_full_pages if len(leaked_full_pages) > 0 else None}, leaked_mamba_pages={leaked_mamba_pages if len(leaked_mamba_pages) > 0 else None}\n"
            )
        else:
            token_msg = (
                f"{full_available_size=}, {full_evictable_size=}, {self.token_to_kv_pool_allocator.size=}, {self.tree_cache.full_protected_size()=}\n"
                f"{mamba_available_size=}, {mamba_evictable_size=}, {self.req_to_token_pool.mamba_pool.size=}, {self.tree_cache.mamba_protected_size()=}\n"
            )
        return memory_leak, token_msg

    def _check_radix_cache_memory(self: Scheduler):
        _, _, available_size, evictable_size = self._get_token_info()
        protected_size = self.tree_cache.protected_size()
        memory_leak = (available_size + evictable_size) != (
            # self.max_total_num_tokens
            # if not self.enable_hierarchical_cache
            # else self.max_total_num_tokens - protected_size
            self.max_total_num_tokens
            - protected_size
        )
        token_msg = f"{self.max_total_num_tokens=}, {available_size=}, {evictable_size=}, {protected_size=}\n"
        return memory_leak, token_msg

    def _get_batch_uncached_size(self: Scheduler, batch: ScheduleBatch) -> int:
        ret = 0
        for req in batch.reqs:
            assert req.kv_committed_freed == req.kv_overallocated_freed
            uncached_len = 0
            if not req.kv_committed_freed:
                allocated_len = req.kv_allocated_len
                if self.page_size > 1:
                    allocated_len = ceil_align(allocated_len, self.page_size)
                    assert req.cache_protected_len % self.page_size == 0
                uncached_len = allocated_len - req.cache_protected_len

            ret += uncached_len

        return ret

    def self_check_during_busy(self: Scheduler):
        current_batch: ScheduleBatch = self.last_batch

        if current_batch is None:
            return

        spec_topk = self.server_args.speculative_eagle_topk or 1
        if spec_topk > 1:
            warnings.warn(
                "Runtime memory check (busy) is not supported when speculation topk > 1."
            )
            return

        _, _, available_size, evictable_size = self._get_token_info()
        protected_size = self.tree_cache.protected_size()

        uncached_size = self._get_batch_uncached_size(current_batch)

        if (
            current_batch.forward_mode.is_extend()
            and self.running_batch is not None
            and not self.running_batch.is_empty()
        ):
            uncached_size += self._get_batch_uncached_size(self.running_batch)

        if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get() > 1:
            log_msg = f"[Mem Check (BUSY)] {available_size=}, {evictable_size=}, {protected_size=}, {uncached_size=}"
            logger.info(log_msg)

        total_tokens = available_size + evictable_size + protected_size + uncached_size
        assert (
            total_tokens == self.max_total_num_tokens
        ), f"Mem Leak Detected! {total_tokens=} vs {self.max_total_num_tokens=}"

    def _check_req_pool(self: Scheduler):
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            req_total_size = (
                self.req_to_token_pool.size + self.req_to_token_pool.pre_alloc_size
            )
        else:
            req_total_size = self.req_to_token_pool.size

        if len(self.req_to_token_pool.free_slots) != req_total_size:
            msg = (
                "req_to_token_pool memory leak detected!"
                f"available_size={len(self.req_to_token_pool.free_slots)}, "
                f"total_size={self.req_to_token_pool.size}\n"
            )
            raise_error_or_warn(
                self,
                envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.get(),
                "count_req_pool_leak_warnings",
                msg,
            )

    def _refresh_idle_metrics(self: Scheduler):
        """Refresh the pool-level gauges while the engine is idle.

        Split out of ``check_memory`` on purpose. This is a read-only snapshot
        of the allocator, so it is safe to run under a *light* gate. The leak
        checks in ``check_memory`` are not: they raise, so they need the full
        ``_is_no_request()`` judgment. Keeping them together would freeze
        ``num_used_tokens`` at a stale value whenever a request lingers in
        ``running_batch`` or a health check sits in an input queue, which is
        exactly the "metrics never fall back" symptom.
        """
        if not (
            self.enable_metrics
            and self.current_scheduler_metrics_enabled
            and time.perf_counter() > self.metrics_collector.last_log_time + 30
        ):
            return
        # During idle time, also collect metrics every 30 seconds.
        if self.is_hybrid_swa:
            (
                full_num_used,
                swa_num_used,
                full_token_usage,
                swa_token_usage,
                _,
                _,
                _,
                _,
            ) = self._get_swa_token_info()
            num_used = max(full_num_used, swa_num_used)
            token_usage = max(full_token_usage, swa_token_usage)
        elif self.is_hybrid_ssm:
            (
                num_used,
                _,
                token_usage,
                _,
                _,
                _,
                _,
                _,
            ) = self._get_mamba_token_info()
        else:
            num_used, token_usage, _, _ = self._get_token_info()
        self.stats.num_running_reqs = len(self.running_batch.reqs)
        self.stats.num_used_tokens = num_used
        self.stats.token_usage = round(token_usage, 2)
        self.stats.gen_throughput = 0
        self.stats.num_queue_reqs = len(self.waiting_queue)
        self.stats.num_grammar_queue_reqs = len(self.grammar_manager)
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.stats.num_prefill_prealloc_queue_reqs = len(
                self.disagg_prefill_bootstrap_queue.queue
            )
            self.stats.num_prefill_inflight_queue_reqs = len(
                self.disagg_prefill_inflight_queue
            )
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            self.stats.num_decode_prealloc_queue_reqs = len(
                self.disagg_decode_prealloc_queue.queue
            )
            self.stats.num_decode_transfer_queue_reqs = len(
                self.disagg_decode_transfer_queue.queue
            )
            self.stats.decode_sum_seq_lens = (
                self.running_batch.seq_lens_cpu.sum().item()
                if (
                    self.running_batch is not None
                    and not self.running_batch.is_empty()
                    and self.running_batch.seq_lens_cpu is not None
                )
                else 0
            )
        self.metrics_collector.log_stats(self.stats)

    def check_memory(self: Scheduler):
        if self.is_hybrid_swa:
            memory_leak, token_msg = self._check_hybrid_memory()
        elif self.is_hybrid_ssm and self.tree_cache.supports_mamba():
            memory_leak, token_msg = self._check_mamba_memory()
        elif (
            self.disaggregation_mode == DisaggregationMode.DECODE
            and getattr(self.tree_cache, "disable", False)
        ):
            # The decode engine disables radix cache, so the radix-based leak
            # check does not apply to it: the pool always keeps a reserved
            # padding page and health-check pages that the check would
            # misreport as a leak. Only skip it for the decode engine; other
            # radix-disabled deployments still go through the radix check.
            memory_leak, token_msg = False, ""
        else:
            memory_leak, token_msg = self._check_radix_cache_memory()

        if memory_leak:
            msg = "token_to_kv_pool_allocator memory leak detected! " f"{token_msg}"
            raise_error_or_warn(
                self,
                envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.get(),
                "count_memory_leak_warnings",
                msg,
            )

        self._check_req_pool()

        self._refresh_idle_metrics()
        self._publish_kv_events()

    def check_tree_cache(self: Scheduler):
        if (
            self.tree_cache.is_tree_cache()
            and (self.is_hybrid_swa and self.tree_cache.supports_swa())
            or (self.is_hybrid_ssm and self.tree_cache.supports_mamba())
        ):
            self.tree_cache.sanity_check()


    def _decode_idle_ready(self: Scheduler) -> bool:
        """Complete idle judgment for the decode engine's leak checks.

        ``_is_no_request()`` cannot be reused verbatim here. With overlap
        scheduling the *idle* event loop still builds an idle batch every
        iteration, runs it, and pushes it onto ``result_queue``; the previous
        iteration's entry is popped right after. So ``len(result_queue) == 0``
        never holds while the loop spins, and any check behind that condition
        would be skipped forever. Treat a ``result_queue`` that only carries
        empty batches as idle instead.
        """
        if not self.running_batch.is_empty():
            return False
        if self.last_batch is not None and not self.last_batch.is_empty():
            return False
        if self.cur_batch is not None and not self.cur_batch.is_empty():
            return False
        if any(b is not None and not b.is_empty() for b, _ in self.result_queue):
            return False
        if self.pp_size > 1 and any(
            not x.is_empty() for x in getattr(self, "running_mbs", []) or []
        ):
            return False
        if (
            len(self.waiting_queue)
            or len(self.disagg_decode_prealloc_queue.queue)
            or len(self.disagg_decode_transfer_queue.queue)
        ):
            return False
        # Quarantined requests have left the transfer queue but still hold
        # their KV pages and req_to_token slot on purpose.
        if self.disagg_decode_transfer_queue.num_quarantined() > 0:
            return False
        # Requests being offloaded still own their KV until the async
        # Device->Host copy completes.
        if (
            self.server_args.disaggregation_decode_enable_offload_kvcache
            and len(self.decode_offload_manager.ongoing_offload) > 0
        ):
            return False
        return True

    def self_check_during_idle(self: Scheduler):
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            if len(self.disagg_prefill_inflight_queue) > 0:
                return
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            # Refresh the gauges *before* the leak-check gate below.
            # num_used_tokens / decode_sum_seq_lens are read-only allocator and
            # batch snapshots, so publishing them is safe even with work still
            # in flight. Keeping them behind the full idle gate is what made
            # the pool look like it never fell back: the gauges froze at their
            # last value from log_decode_stats — i.e. the last running
            # request's size — long after the pages had actually been freed.
            self._refresh_idle_metrics()

            if not self._decode_idle_ready():
                return

        self.check_memory()
        self.check_tree_cache()
        self.new_token_ratio = self.init_new_token_ratio
        self.maybe_sleep_on_idle()


def create_scheduler_watchdog(
    scheduler: Scheduler, watchdog_timeout: float, soft: bool = False
) -> WatchdogRaw:
    def dump_info() -> str:
        if scheduler.is_initializing or disable_request_logging():
            return ""
        if scheduler.is_hybrid_swa:
            _, info_msg = scheduler._check_hybrid_memory()
        elif scheduler.is_hybrid_ssm and scheduler.tree_cache.supports_mamba():
            _, info_msg = scheduler._check_mamba_memory()
        else:
            _, info_msg = scheduler._check_radix_cache_memory()
        return (
            f"{scheduler.cur_batch.batch_size()=}\n"
            f"{scheduler.cur_batch.reqs=}\n"
            f"{info_msg}"
        )

    return WatchdogRaw(
        debug_name="Scheduler",
        get_counter=lambda: getattr(scheduler, "forward_ct", 0),
        is_active=lambda: scheduler.is_initializing
        or getattr(scheduler, "cur_batch", None) is not None,
        watchdog_timeout=watchdog_timeout,
        soft=soft,
        dump_info=dump_info,
    )
