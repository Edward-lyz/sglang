#!/usr/bin/env python3
"""Four CPU Gloo ranks: identical refill slots through failures and readiness."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TREE = ROOT / "python/sglang/srt/mem_cache/storage/asradix"


def worker(args):
    """Run one CPU rank through allocation, read and publication failure cases.

    Args:
        args: Parsed qualification options including rank and output directory.

    Returns:
        None; the rank writes a qualification result file.

    Raises:
        AssertionError: A collective, publication or cleanup invariant is
            violated.
    """
    from types import SimpleNamespace
    from unittest import mock

    import torch
    import torch.distributed as dist
    from sglang.srt.mem_cache.storage.asradix import checkpoint_coordinator as module

    torch.set_num_threads(1)
    rank = args.rank
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(args.output / "rendezvous"),
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=30),
    )
    try:
        coordinator = module.SPAGCacheCoordinator()
        coordinator._native_physical_writer = lambda cache: True
        inserted, events, freed, released, reductions, requests, futures = (
            [],
            [],
            [],
            [],
            [],
            [],
            {},
        )
        scalar_votes = []
        current = [0]
        tasks = 18
        failed = {1, 2, 3, 4, 17}

        class NativeFuture:
            """Model native read completion and injected per-rank failures."""
            def __init__(self, index, loaded):
                self.index = index
                self.ready = not (index == 0 and rank == 0)
                pages = [int(loaded[i * 4]) // 4 for i in range(3)]
                selected = sorted({i for i in range(3) if i * 4 // 3 == rank} | {2})
                self.native = SimpleNamespace(
                    _completion=dict(
                        prepared_pages=pages,
                        local_indices=selected,
                        prepared_swa_page=7,
                    )
                )
                self.count = len(selected)

            def done(self):
                """Expose the configured completion readiness for this rank.

                Returns:
                    bool: Whether this fixture is ready to drain.
                """
                return self.ready

            def result(self):
                """Return the configured rank-local read result.

                Returns:
                    int: Number of successfully read blocks.

                Raises:
                    RuntimeError: This rank is configured to fail the read.
                """
                assert self.ready
                if self.index == 4 and rank == 0:
                    raise RuntimeError("injected native read failure")
                return self.count

        def allocate(*args):
            index = current[0]
            if (index == 1 and rank == 1) or index == 17:
                return None
            if index == 2 and rank == 2:
                raise RuntimeError("injected allocation failure")
            return torch.arange(index * 12 + 4, index * 12 + 16, dtype=torch.int64)

        def free_full(loaded, **kwargs):
            freed.append(int(loaded[0]) // 12)

        allocator = SimpleNamespace(
            device="cpu",
            available_size=lambda: 1000,
            alloc_extend_swa_tail=allocate,
            free_swa_segment=lambda *a, **k: None,
            free_full_segment=free_full,
        )

        def reduce(tensor, op):
            before = tensor.tolist()
            dist.all_reduce(tensor, op=op)
            reductions.append(
                dict(before=before, after=tensor.tolist(), shape=list(tensor.shape))
            )

        def scalar_min(value):
            scalar_votes.append(value)
            tensor = torch.tensor(value, dtype=torch.int64)
            reduce(tensor, dist.ReduceOp.MIN)
            return int(tensor)

        def emit(action, hashes, pages):
            events.append(("digest", (int(pages[0]) - 1) // 3))

        def diffuse(pages, count, **kwargs):
            events.append(("diffuse", (int(pages[0]) - 1) // 3))

        def diffuse_many(entries, **kwargs):
            for pages, count, extra in entries:
                diffuse(pages, count)

        cache = SimpleNamespace(
            page_size=4,
            pp_size=1,
            pp_world_size=1,
            token_to_kv_pool_allocator=allocator,
            full_evictable_size=lambda: 0,
            _as_min=scalar_min,
            _all_reduce_attn_groups=reduce,
            _log_as_result=mock.Mock(),
            _log_as_failure=mock.Mock(),
            _as_read_diffuser=SimpleNamespace(
                cp_rank=rank, cp_size=4, diffuse=diffuse, diffuse_many=diffuse_many
            ),
            ascli=SimpleNamespace(emit_content_digest=emit),
        )

        def submit(cache, task):
            index = current[0]
            if index == 3 and rank == 3:
                raise RuntimeError("injected native submit failure")
            future = NativeFuture(index, task.loaded)
            futures[index] = future
            return future

        coordinator._try_native_metadata = submit
        coordinator._ensure_load_back_executor = mock.Mock(
            side_effect=AssertionError("Python reader must not be used")
        )

        def publish(cache, task, tokens):
            index = int(task.rid.split("-")[1])
            assert index not in failed
            assert tokens == 12
            inserted.append(index)
            events.append(("insert", index))
            task.req._as_materialized_host_span = (0, tokens)
            task.req.as_read_pending_lock = True

        coordinator._insert_loaded_prefix = publish
        with mock.patch.object(
            module,
            "_acquire_node_release",
            side_effect=lambda cache, anchor: lambda: released.append(anchor),
        ):
            for index in range(tasks):
                current[0] = index
                req = SimpleNamespace(
                    rid=f"req-{index}",
                    prefix_indices=torch.empty(0, dtype=torch.int64),
                    _as_pending_load_back=None,
                    _as_materialized_host_span=None,
                    as_read_pending_lock=False,
                )
                task = coordinator._submit_load_back(
                    cache,
                    req,
                    dict(
                        block_count=3,
                        start_block=0,
                        hashes=[index * 3 + j for j in range(3)],
                    ),
                    index,
                )
                assert task is not None, (rank, index, "queue slot was dropped")
                req._as_pending_load_back = task
                requests.append(req)
        assert len(coordinator._pending_load_backs) == tasks
        assert scalar_votes == [1], scalar_votes
        coordinator.commit_load_backs(cache)
        assert len(coordinator._pending_load_backs) == tasks
        assert not inserted, inserted
        if rank == 0:
            futures[0].ready = True
        dist.barrier()
        coordinator.commit_load_backs(cache)
        assert not coordinator._pending_load_backs
        expected = [i for i in range(tasks) if i not in failed]
        assert inserted == expected, (rank, inserted)
        for index, req in enumerate(requests):
            assert req._as_pending_load_back is None
            assert bool(req.as_read_pending_lock) == (index not in failed)
            assert (req._as_materialized_host_span is not None) == (index not in failed)
        for index in expected:
            trio = [name for name, i in events if i == index]
            assert trio == ["diffuse", "digest", "insert"], trio
        for index in failed:
            assert not [event for event in events if event[1] == index]
            owned = not (
                (index == 1 and rank == 1) or (index == 2 and rank == 2) or index == 17
            )
            assert freed.count(index) == int(owned), (rank, index, freed)
            assert released.count(index) == int(owned), (rank, index, released)
        assert [entry["shape"] for entry in reductions] == [
            [],
            [1],
            [1],
            [tasks],
        ], reductions
        assert reductions[-1]["after"] == [
            0 if i in failed else 3 for i in range(tasks)
        ]
        coordinator._ensure_load_back_executor.assert_not_called()
        assert not torch.cuda.is_initialized()
        record = dict(
            passed=True,
            rank=rank,
            grouped=args.grouped,
            requests=tasks,
            failed_indices=sorted(failed),
            published=inserted,
            freed=freed,
            released=released,
            collectives=reductions,
            CPU_only=True,
            CUDA_initialized=False,
            actual_Gloo=True,
            queue_and_collective_order_preserved=True,
            publication_before_CP_agreement=False,
        )
        (args.output / f"rank{rank}.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def main():
    """Run the selected four-rank CPU qualification scenario.

    Returns:
        None.

    Raises:
        AssertionError: A rank fails the ownership or publication checks.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--grouped", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--enabled",
        type=int,
        choices=(0, 1),
        default=1,
        help="0 is an expected-failure compatibility-path control",
    )
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.rank is not None:
        try:
            worker(args)
        except Exception:
            traceback.print_exc()
            return 1
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "harness.py").write_bytes(Path(__file__).read_bytes())
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="99",
        OMP_NUM_THREADS="1",
        PYTHONPATH=str(ROOT / "python"),
        SGLANG_AS_DEFER_ALLOCATION_GATE=str(args.enabled),
        SGLANG_AS_READ_DIFFUSE_GROUP_COMMIT=str(args.grouped),
    )
    processes = []
    try:
        for rank in range(4):
            output = (args.output / f"rank{rank}.log").open("x")
            argv = [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                str(args.output),
                "--rank",
                str(rank),
                "--grouped",
                str(args.grouped),
                "--enabled",
                str(args.enabled),
            ]
            process = subprocess.Popen(
                argv, env=environment, stdout=output, stderr=subprocess.STDOUT
            )
            output.close()
            processes.append(process)
        deadline = time.monotonic() + 90
        codes = [
            process.wait(timeout=max(1, deadline - time.monotonic()))
            for process in processes
        ]
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
    records = [
        json.loads(p.read_text()) for p in sorted(args.output.glob("rank*.json"))
    ]
    report = dict(
        passed=codes == [0] * 4 and len(records) == 4,
        CPU_only=True,
        actual_Gloo_ranks=4,
        grouped=args.grouped,
        enabled=args.enabled,
        rank_results=records,
        returncodes=codes,
        serving_integrated=False,
        performance_proven=False,
    )
    report["serving_tree"] = str(TREE)
    report["serving_files"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in TREE.glob("*.py")
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(passed=report["passed"], returncodes=codes, rank_results=len(records)),
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
