"""Isolated NVDEC (PyNvVideoCodec) video-decode worker pool.

Root-cause hardening for malformed-bitstream videos that make NVDEC
divide-by-zero (SIGFPE, core dumped) during decoder construction. Such a native
crash cannot be caught by a Python try/except and, when the decode runs in a
thread of the main encoder process, it takes the whole service down.

The NVDEC decode runs inside persistent, spawned subprocesses. If a subprocess
dies from a native signal (e.g. SIGFPE), only that worker is lost: the caller
detects the crash, respawns the worker, and falls back to the in-process (CPU)
decord decoder, which decodes these files correctly. The main encoder process
stays alive and keeps serving other requests.

Decoded GPU frames are handed to the parent process zero-copy via CUDA IPC
(``torch.multiprocessing`` transfers a CUDA tensor by memory handle; the parent
D2D-clones it into its own context). A pool of ``SGLANG_VIDEO_DECODE_WORKERS``
(default 4) workers preserves decode concurrency.
"""

from __future__ import annotations

import logging
import os
import queue as _queue
import threading
import time

import torch
import torch.multiprocessing as tmp

logger = logging.getLogger(__name__)

_SHUTDOWN = None  # sentinel asking a worker to exit

# Backpressure window: max chunks a worker may keep decoded-and-unacked at once.
# The parent acks each chunk after it has D2D-cloned it, and the downstream
# decode_queue is bounded to the same size, so decode is paced to embed and a
# worker's retained GPU footprint stays ~W chunks regardless of video length
# (root-cause fix for the "worker retains ALL chunks -> 60GB+ -> OOM" blowup).
_DECODE_AHEAD = int(
    os.environ.get(
        "SGLANG_VIDEO_DECODE_AHEAD",
        os.environ.get("SGLANG_ENCODER_VIDEO_MAX_INFLIGHT_CHUNKS", "8"),
    )
)
# Upper bound the worker waits for an ack when the window is full. Exceeding it
# means the parent consumer stalled/died; the worker aborts the job so the pool's
# deadline detection can respawn it instead of hanging forever.
# Aligned to the 30s task-level timeout (SGLANG_ENCODER_TASK_TIMEOUT_S): a stalled
# task already fails at 30s, so waiting longer here only lets a stuck worker
# linger (holding its slot/GPU); 30s of no ack means the consumer is truly stuck.
_ACK_WAIT_TIMEOUT = float(os.environ.get("SGLANG_VIDEO_DECODE_ACK_TIMEOUT_S", "30"))

# Per-job wall-clock watchdog: a hard upper bound on how long ONE decode job may
# run, computed once at dispatch and NEVER reset by incremental progress. This
# closes the "native decode hang" gap that the per-message silence deadline and
# the crash/ack timeouts all miss: a malformed bitstream can make pynv raise a
# catchable error, then the in-job decord (CPU) fallback hangs INSIDE the native
# decode call -- the process stays alive (no crash -> no respawn) and never
# reaches the ack-wait (no ack timeout), so the worker is wedged indefinitely
# (observed: 4 workers eaten one by one -> video pipeline dead for 6-7h).
# When a job exceeds this bound the pool force-kills and respawns the worker.
# Set to 0 to disable.
#
# Default = the task-level timeout (SGLANG_ENCODER_TASK_TIMEOUT_S, 30s): once the
# task is cancelled at 30s its decode output is worthless, so there is no value in
# keeping a stuck worker busy longer -- reclaim it at the task budget. Tracking
# that env (instead of a hardcoded 30) keeps them aligned if the task timeout is
# adjusted, avoiding killing a decode the (longer-lived) task still wants.
# NOTE: after the consumer-cancel abort path (run_video_stream should_abort), this
# watchdog is a BACKSTOP -- the primary recovery is the abort that fires within
# ~0.5s of the task timeout; the watchdog mainly covers the should_abort=None
# (batch) path and any case where the cancel signal is missed.
_TASK_TIMEOUT_S = float(os.environ.get("SGLANG_ENCODER_TASK_TIMEOUT_S", "30"))
_JOB_TIMEOUT = float(
    os.environ.get("SGLANG_VIDEO_DECODE_JOB_TIMEOUT_S", str(_TASK_TIMEOUT_S))
)


def _set_pdeathsig_kill():
    """Ask the kernel to SIGKILL this worker when its parent dies (Linux).

    ``daemon=True`` only reaps children on the parent's *normal* exit; a hard
    ``kill -9`` of the parent would otherwise orphan these workers and leak
    their GPU memory. PR_SET_PDEATHSIG makes the kernel kill us regardless."""
    try:
        import ctypes
        import signal as _sig

        # Use the current process's already-loaded libc symbols instead of a
        # hard-coded "libc.so.6" soname, so it also works on musl-based distros
        # (e.g. Alpine) where that soname does not exist.
        libc = ctypes.CDLL(None, use_errno=True)
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, _sig.SIGKILL)
        if os.getppid() == 1:  # parent already gone
            os._exit(0)
    except Exception:
        pass


# Distinctive process name (comm, <=15 chars) stamped on every decode worker so
# an orphan-reaper can identify leftover workers by /proc/<pid>/comm without
# risking unrelated processes.
_WORKER_COMM = "sgl-vdworker"


def _set_worker_proctitle():
    """Rename this worker's process comm to ``_WORKER_COMM`` (via prctl
    PR_SET_NAME) so orphaned workers are identifiable in /proc for reaping."""
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        PR_SET_NAME = 15
        libc.prctl(PR_SET_NAME, ctypes.c_char_p(_WORKER_COMM.encode()), 0, 0, 0)
    except Exception:
        pass


def _reap_orphan_workers():
    """Kill decode-worker subprocesses left over from a PREVIOUS encoder that was
    hard-killed (a SIGKILL of the parent can bypass PR_SET_PDEATHSIG in some
    races / when the workers were mid-init), which otherwise keep holding CUDA
    contexts and GPU memory.

    Safety: only targets processes whose comm == ``_WORKER_COMM`` AND whose
    parent is init (ppid == 1) -- i.e. definitely orphaned (their encoder is
    gone). A LIVE sibling encoder's workers have ppid = that encoder's pid, so
    they are never touched. Best-effort; never raises."""
    import signal as _sig

    try:
        entries = os.listdir("/proc")
    except Exception:
        return
    me = os.getpid()
    reaped = 0
    for d in entries:
        if not d.isdigit():
            continue
        pid = int(d)
        if pid == me:
            continue
        try:
            with open(f"/proc/{d}/comm", "r") as f:
                if f.read().strip() != _WORKER_COMM:
                    continue
            with open(f"/proc/{d}/stat", "r") as f:
                # "pid (comm) state ppid ..."; comm may contain spaces/parens,
                # so parse after the LAST ')'.
                after = f.read().rsplit(")", 1)[1].split()
                ppid = int(after[1])  # after[0]=state, after[1]=ppid
            if ppid != 1:
                continue  # still owned by a live encoder -> leave it
            os.kill(pid, _sig.SIGKILL)
            reaped += 1
            logger.warning(
                "[VideoDecodeWorkerPool] reaped orphaned decode worker pid=%s "
                "(leftover from a hard-killed encoder)",
                pid,
            )
        except (FileNotFoundError, ProcessLookupError):
            continue  # process vanished between listing and read/kill
        except Exception:
            continue
    if reaped:
        logger.warning(
            "[VideoDecodeWorkerPool] reaped %d orphaned decode worker(s)", reaped
        )


def _close_worker_queues(w) -> None:
    """Release the pipe file descriptors behind a worker's mp queues.

    Called on respawn/shutdown so long-running services don't leak FDs as
    workers are recycled."""
    if w is None:
        return
    for q in (
        getattr(w, "req_q", None),
        getattr(w, "resp_q", None),
        getattr(w, "ack_q", None),
    ):
        try:
            if q is not None:
                q.cancel_join_thread()
                q.close()
        except Exception:
            pass


def _disable_core_dumps():
    """Prevent this worker from writing a (multi-GB) core file when it dies from
    a native SIGFPE. NVDEC crashes on malformed videos are expected here; without
    this, repeated crashes would quickly fill the disk with core dumps."""
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except Exception:
        pass


def _worker_main(server_args, gpu_index, req_q, resp_q, ack_q):
    """Decode subprocess entry point: build an in-process VideoDecodeImpl and
    stream one GPU chunk tensor per ``chunk`` message via CUDA IPC. A native
    crash (SIGFPE) terminates this process; the parent observes exitcode and
    recovers."""
    _disable_core_dumps()  # native SIGFPE must not dump multi-GB core files
    _set_pdeathsig_kill()  # die if the parent is hard-killed (avoid GPU leak)
    _set_worker_proctitle()  # tag comm so orphans can be reaped if PDEATHSIG races
    # Spawned subprocesses start with NO logging handler, so this worker's
    # logger.info(...) (ready / crash lines) would otherwise be dropped
    # and never reach encode.log. Configure a handler here; stdout/stderr are
    # inherited from the parent (redirected to encode.log).
    try:
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [vdworker pid=%(process)d] %(message)s",
            )
    except Exception:
        pass
    try:
        torch.cuda.set_device(gpu_index)
    except Exception:
        pass

    from sglang.srt.disaggregation.encode_stage import (
        RetryableVideoDecodeError,
        VideoDecodeImpl,
        VideoLoadReq,
    )

    # _is_decode_worker=True: this instance decodes in-process and must NOT spawn
    # its own worker pool (would recurse).
    impl = VideoDecodeImpl(server_args, max_batch_size=1, _is_decode_worker=True)
    # Keep CUDA-IPC payloads of the CURRENT job alive until the next job arrives.
    # By then the parent's run_video_stream() for the previous job has returned
    # (it blocks until "done"), i.e. all chunks were D2D-cloned into the parent,
    # so releasing here is safe and bounds memory to one video.
    # chunk_idx -> GPU payload, kept alive for CUDA-IPC until the parent acks it
    # (parent acks right after it has D2D-cloned the chunk). Bounded to
    # ~_DECODE_AHEAD entries by the window check below, so a worker's retained GPU
    # footprint no longer grows with video length.
    _alive = {}

    def _drain_acks(cur_job_id, block=False):
        """Release chunks the parent has consumed. Acks are (job_id, chunk_idx);
        only acks for the CURRENT job may free entries — a late ack from a previous
        video must never pop the next video's same-index chunk (chunk_idx restarts
        at 0 per video), which would free CUDA-IPC memory before the parent consumed
        it. When ``block`` is True, wait (bounded) for one ack to hold outstanding
        <= _DECODE_AHEAD; returns False if it timed out waiting (parent stalled)."""
        if block:
            try:
                jid, aidx = ack_q.get(timeout=_ACK_WAIT_TIMEOUT)
            except _queue.Empty:
                return False
            except Exception:
                return False
            if jid == cur_job_id:
                _alive.pop(aidx, None)
        while True:
            try:
                jid, aidx = ack_q.get_nowait()
            except _queue.Empty:
                break
            except Exception:
                break
            if jid == cur_job_id:
                _alive.pop(aidx, None)
        return True

    logger.info("[VideoDecodeWorker] ready gpu_index=%s pid=%s", gpu_index, os.getpid())

    while True:
        job = req_q.get()
        if job is _SHUTDOWN:
            break
        job_id, local_path, sample_fps, force_decord = job
        _drain_acks(job_id)  # discard stale acks; clear() drops leftover payloads
        _alive.clear()
        try:
            req = VideoLoadReq(
                rid=str(job_id), video_items=[local_path], sample_fps=sample_fps
            )
            # force_decord: NVDEC already crashed on this file, so skip pynv and
            # decode with CPU decord directly -- still isolated in this subprocess,
            # so a decord native crash also only kills this worker, not the server.
            if force_decord:
                chunk_iter = impl._iter_decode_decord_chunks(
                    local_path, req, force_cpu=True
                )
            else:
                chunk_iter = impl._iter_decode_video_chunks(local_path, req)
            n = 0
            for item in chunk_iter:
                chunk_idx, chunk_tensor, height, width, timestamp, frame_cost = item
                # Zero-copy: hand the GPU tensor to the parent via CUDA IPC.
                # torch records a producer-side event at send time that the
                # parent waits on when it opens the tensor, so we don't do a
                # whole-device synchronize here -> the worker keeps decoding the
                # next chunk while the parent consumes this one.
                payload = chunk_tensor.contiguous()
                # Backpressure: never keep more than _DECODE_AHEAD chunks decoded
                # but unacked. Block for an ack (parent consumed a chunk) when the
                # window is full -> decode is paced to the embed consumer and the
                # retained GPU footprint stays bounded regardless of video length.
                while len(_alive) >= _DECODE_AHEAD:
                    if not _drain_acks(job_id, block=True):
                        raise RuntimeError(
                            f"no chunk ack within {_ACK_WAIT_TIMEOUT:.0f}s "
                            f"(parent consumer stalled?)"
                        )
                _alive[chunk_idx] = payload  # keep alive until parent acks
                resp_q.put(
                    (
                        job_id,
                        "chunk",
                        chunk_idx,
                        payload,
                        timestamp,
                        height,
                        width,
                        frame_cost,
                    )
                )
                n += 1
                _drain_acks(job_id)  # sweep any acks that arrived meanwhile
            _drain_acks(job_id)  # final sweep of any acks that arrived during decode
            resp_q.put((job_id, "done", n))
        except BaseException as e:  # noqa: BLE001
            # Queued IPC payloads must survive until the parent clones/acks them,
            # including on ordinary decode errors. The next job clears leftovers.
            resp_q.put(
                (
                    job_id,
                    "error",
                    f"{type(e).__name__}: {e}",
                    isinstance(e, RetryableVideoDecodeError),
                    getattr(e, "code", 422),
                )
            )


class _Worker:
    """Handle to one decode subprocess and its request/response/ack queues."""

    __slots__ = ("proc", "req_q", "resp_q", "ack_q")

    def __init__(self, proc, req_q, resp_q, ack_q):
        self.proc = proc
        self.req_q = req_q
        self.resp_q = resp_q
        self.ack_q = ack_q


class VideoDecodeWorkerPool:
    """Pool of persistent NVDEC decode subprocesses with per-worker crash
    recovery. ``run_video_stream`` borrows an idle worker, so up to
    ``num_workers`` videos decode concurrently."""

    def __init__(self, server_args, gpu_index, timeout_s=120.0, num_workers=None):
        self._server_args = server_args
        self._gpu_index = gpu_index
        self._timeout_s = timeout_s
        if num_workers is None:
            num_workers = int(os.environ.get("SGLANG_VIDEO_DECODE_WORKERS", "4"))
        self._n = max(1, num_workers)
        self._ctx = tmp.get_context("spawn")
        self._job_seq = 0
        self._seq_lock = threading.Lock()
        self._workers = [None] * self._n
        self._idle: _queue.Queue[int] = _queue.Queue()
        self.crash_count = 0
        self.fallback_count = 0
        # Clean up any decode workers orphaned by a previously hard-killed encoder
        # (they would otherwise keep holding GPU memory) before spawning ours.
        _reap_orphan_workers()
        for i in range(self._n):
            self._workers[i] = self._spawn_one()
            self._idle.put(i)
        logger.info("[VideoDecodeWorkerPool] started with %d workers", self._n)

    # -- lifecycle -----------------------------------------------------------
    def _spawn_one(self) -> _Worker:
        req_q = self._ctx.Queue()
        # Bounded resp_q: when the parent stops reading (because embed is slow and
        # the downstream decode_queue is full), the worker blocks on resp_q.put,
        # which paces decoding to the embed consumer instead of racing ahead and
        # retaining the whole video in GPU memory.
        resp_q = self._ctx.Queue(maxsize=_DECODE_AHEAD + 2)
        ack_q = self._ctx.Queue()  # parent -> worker: chunk_idx consumed (safe to free)
        proc = self._ctx.Process(
            target=_worker_main,
            args=(self._server_args, self._gpu_index, req_q, resp_q, ack_q),
            daemon=True,
            name="video-decode-worker",
        )
        proc.start()
        logger.info("[VideoDecodeWorkerPool] spawned worker pid=%s", proc.pid)
        return _Worker(proc, req_q, resp_q, ack_q)

    def _respawn_slot(self, i: int):
        """Terminate the dead/broken worker in slot ``i``, release its queue
        FDs, and start a fresh worker in its place."""
        old = self._workers[i]
        try:
            if old is not None and old.proc is not None and old.proc.is_alive():
                old.proc.terminate()
                old.proc.join(timeout=3)
                # A worker stuck inside a native CUDA/decode call may ignore
                # SIGTERM; escalate to SIGKILL so it can't linger holding GPU
                # memory / queue FDs after we close its queues below.
                if old.proc.is_alive():
                    old.proc.kill()
                    old.proc.join(timeout=1)
        except Exception:
            pass
        _close_worker_queues(old)
        self._workers[i] = self._spawn_one()

    def shutdown(self):
        """Gracefully stop all worker processes and release their queue FDs."""
        for w in self._workers:
            try:
                if w is not None and w.proc is not None and w.proc.is_alive():
                    w.req_q.put(_SHUTDOWN)
            except Exception:
                pass
        for w in self._workers:
            try:
                if w is not None and w.proc is not None:
                    w.proc.join(timeout=3)
                    if w.proc.is_alive():
                        w.proc.terminate()
                        w.proc.join(timeout=2)
                        # Still alive after SIGTERM (e.g. blocked in native
                        # CUDA) -> force SIGKILL so shutdown never leaks a worker.
                        if w.proc.is_alive():
                            w.proc.kill()
            except Exception:
                pass
        for w in self._workers:
            _close_worker_queues(w)

    def _next_job_id(self) -> int:
        """Return a process-unique, monotonically increasing job id (thread-safe)."""
        with self._seq_lock:
            self._job_seq += 1
            return self._job_seq

    # -- decode --------------------------------------------------------------
    def run_video_stream(
        self, local_path, sample_fps, on_chunk, force_decord=False, should_abort=None
    ) -> bool:
        """Decode one video via a borrowed worker, streaming GPU (CUDA-IPC)
        chunks to ``on_chunk``. ``force_decord`` makes the worker skip pynv and
        use CPU decord (for the isolated fallback after an NVDEC crash).
        ``should_abort`` is an optional predicate; when it returns True (the
        consumer was cancelled/timed out) the job is aborted, the worker is
        force-respawned (it may be blocked on a full resp_q), and
        DecodeConsumerGone is raised so the caller stops quietly. Returns
        True if the worker CRASHED (native death) so the caller can escalate;
        False on clean completion. Raises VideoDecodeError for ordinary python
        decode errors."""
        from sglang.srt.disaggregation.encode_stage import (
            DecodeConsumerGone,
            RetryableVideoDecodeError,
            VideoDecodeError,
        )

        # Fix D-1: acquiring a worker must itself be abortable. A plain blocking
        # _idle.get() sits BEFORE the poll loop where should_abort/job-watchdog
        # live, so if every worker is busy (e.g. many concurrent native decode
        # hangs) a cancelled task's decode thread would block here forever, leak
        # a decode-threadpool thread, and never release -> a second wedge. Poll
        # with a timeout and bail out (nothing borrowed yet) when the consumer is
        # already gone.
        i = None
        while i is None:
            if should_abort is not None and should_abort():
                raise DecodeConsumerGone("consumer gone before worker acquired")
            try:
                i = self._idle.get(timeout=1.0)
            except _queue.Empty:
                continue
        crashed = False
        aborted = False
        recovered = False
        worker_reported_error = False
        try:
            w = self._workers[i]
            if w is None or w.proc is None or not w.proc.is_alive():
                self._respawn_slot(i)
                w = self._workers[i]
            job_id = self._next_job_id()
            w.req_q.put((job_id, local_path, sample_fps, force_decord))

            deadline = time.monotonic() + self._timeout_s
            # Per-job wall-clock watchdog (Rec 1): computed ONCE and never reset
            # by incremental progress, unlike `deadline` above. Bounds a native
            # decode hang (worker alive but stuck inside a decode call) that the
            # crash-detection and ack-timeout paths cannot catch.
            job_deadline = (
                (time.monotonic() + _JOB_TIMEOUT) if _JOB_TIMEOUT > 0 else None
            )
            while True:
                # Consumer cancelled/timed out: stop pulling from this worker and
                # abort. The worker may be blocked on a full resp_q.put, so it is
                # force-respawned below to reset it.
                if should_abort is not None and should_abort():
                    logger.warning(
                        "[VideoDecodeWorkerPool] consumer gone; aborting job path=%s",
                        local_path,
                    )
                    aborted = True
                    break
                try:
                    item = w.resp_q.get(timeout=1.0)
                except _queue.Empty:
                    if w.proc.exitcode is not None:
                        self.crash_count += 1
                        sig = -w.proc.exitcode if (w.proc.exitcode or 0) < 0 else "n/a"
                        logger.error(
                            "[VideoDecodeWorkerPool] worker died exitcode=%s signal=%s path=%s",
                            w.proc.exitcode,
                            sig,
                            local_path,
                        )
                        crashed = True
                        break
                    if job_deadline is not None and time.monotonic() > job_deadline:
                        self.crash_count += 1
                        logger.error(
                            "[VideoDecodeWorkerPool] worker job exceeded %.0fs "
                            "(native decode hang?) path=%s; force-respawn",
                            _JOB_TIMEOUT,
                            local_path,
                        )
                        crashed = True
                        break
                    if time.monotonic() > deadline:
                        self.crash_count += 1
                        logger.error(
                            "[VideoDecodeWorkerPool] worker silent >%.0fs path=%s; treating as crash",
                            self._timeout_s,
                            local_path,
                        )
                        crashed = True
                        break
                    continue

                deadline = time.monotonic() + self._timeout_s
                if item[0] != job_id:
                    continue  # stale message
                kind = item[1]
                if kind == "chunk":
                    (
                        _,
                        _,
                        chunk_idx,
                        gpu_tensor,
                        timestamp,
                        height,
                        width,
                        frame_cost,
                    ) = item
                    # on_chunk performs the D2D clone into the parent's context and
                    # returns the CUDA event recording the clone's completion (or
                    # None). We MUST wait on that event before acking: the clone is
                    # async, and once we ack the worker frees/reuses the source
                    # CUDA-IPC block -> acking before the clone finished would risk a
                    # cross-process dirty read. After the event, ack to (a) free the
                    # worker's _alive entry and (b) release one decode-ahead slot.
                    # on_chunk (writer.emit) can raise DecodeConsumerGone if the
                    # consumer was cancelled and stopped draining -- abort instead
                    # of blocking forever inside it.
                    try:
                        clone_ev = on_chunk(
                            chunk_idx, gpu_tensor, timestamp, height, width, frame_cost
                        )
                    except DecodeConsumerGone:
                        del gpu_tensor
                        aborted = True
                        break
                    del gpu_tensor
                    if clone_ev is not None:
                        clone_ev.synchronize()
                    try:
                        w.ack_q.put((job_id, chunk_idx))
                    except Exception:
                        pass
                elif kind == "done":
                    break
                elif kind == "error":
                    error_type = (
                        RetryableVideoDecodeError
                        if len(item) > 3 and item[3]
                        else VideoDecodeError
                    )
                    error = error_type(
                        f"video decode failed (worker job={job_id} path={local_path}: {item[2]})",
                        code=item[4] if len(item) > 4 else 422,
                    )
                    # This terminal message follows all chunks from the job.
                    # Their clones/acks have completed, and the worker is ready
                    # for another request. Ordinary decode errors need no respawn.
                    worker_reported_error = True
                    raise error
            if crashed or aborted:
                # abort: the worker may be blocked on a full resp_q.put; respawn
                # to reset it and reclaim a clean slot.
                self._respawn_slot(i)
                recovered = True
            if aborted:
                raise DecodeConsumerGone("decode aborted: consumer gone")
            return crashed
        except BaseException:
            # A callback/protocol error can leave the worker blocked on an ack
            # or a full response queue. Never return that slot to the idle pool.
            if not recovered and not worker_reported_error:
                self._respawn_slot(i)
            raise
        finally:
            self._idle.put(i)
