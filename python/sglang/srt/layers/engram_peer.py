"""[luno-5437] Engram tables sharded across same-node instances over CUDA IPC.

N instances (fake-dp replicas, or independent servers that see every GPU of the
node) each hold 1 / (N * tp) of an engram table instead of 1 / tp. After loading,
every shard is published as a CUDA IPC handle in a rendezvous directory; each
rank maps all peer shards into its address space and gathers the rows of its own
tokens directly from the owner GPU over NVLink. No collective is involved, so
the instances only share memory, not process groups.
"""

from __future__ import annotations

import atexit
import ctypes
import logging
import os
import pickle
import time
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

_POSIX_FD = 1  # CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
_LOC_DEVICE = 1  # CU_MEM_LOCATION_TYPE_DEVICE
_ACCESS_RW = 3  # CU_MEM_ACCESS_FLAGS_PROT_READWRITE
_SYS_PIDFD_OPEN, _SYS_PIDFD_GETFD = 434, 438


class _Loc(ctypes.Structure):
    """CUmemLocation."""

    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _AllocFlags(ctypes.Structure):
    """CUmemAllocationProp::allocFlags."""

    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class _AllocProp(ctypes.Structure):
    """CUmemAllocationProp."""

    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", _Loc),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _AllocFlags),
    ]


class _AccessDesc(ctypes.Structure):
    """CUmemAccessDesc."""

    _fields_ = [("location", _Loc), ("flags", ctypes.c_int)]


_libcuda = None
_libc = None


def _cu():
    """Return the lazily loaded libcuda driver library."""
    global _libcuda
    if _libcuda is None:
        _libcuda = ctypes.CDLL("libcuda.so.1")
    return _libcuda


def _check(err: int, what: str) -> None:
    """Raise RuntimeError if a CUDA driver call returned a non-zero CUresult."""
    if err != 0:
        msg = ctypes.c_char_p()
        _cu().cuGetErrorString(err, ctypes.byref(msg))
        raise RuntimeError(f"engram peer table: {what} failed: {err} ({msg.value})")


def _alloc_prop(device: int) -> _AllocProp:
    """Pinned device allocation on `device`, exportable as a POSIX fd."""
    prop = _AllocProp()
    prop.type = 1  # CU_MEM_ALLOCATION_TYPE_PINNED
    prop.requestedHandleTypes = _POSIX_FD
    prop.location = _Loc(_LOC_DEVICE, device)
    return prop


def _map_for(handle: int, size: int, device: int) -> int:
    """Reserve VA for `size` bytes, map the physical allocation `handle` there
    and make it accessible from `device` only (no context on the owner GPU)."""
    ptr = ctypes.c_uint64()
    _check(
        _cu().cuMemAddressReserve(
            ctypes.byref(ptr),
            ctypes.c_size_t(size),
            ctypes.c_size_t(0),
            ctypes.c_uint64(0),
            ctypes.c_uint64(0),
        ),
        "cuMemAddressReserve",
    )
    _check(
        _cu().cuMemMap(
            ptr,
            ctypes.c_size_t(size),
            ctypes.c_size_t(0),
            ctypes.c_uint64(handle),
            ctypes.c_uint64(0),
        ),
        "cuMemMap",
    )
    desc = _AccessDesc(_Loc(_LOC_DEVICE, device), _ACCESS_RW)
    _check(
        _cu().cuMemSetAccess(
            ptr, ctypes.c_size_t(size), ctypes.byref(desc), ctypes.c_size_t(1)
        ),
        "cuMemSetAccess",
    )
    return ptr.value


def _steal_fd(pid: int, fd: int) -> int:
    """Duplicate another process's fd (pidfd_getfd: same user + ptrace access)."""
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL(None, use_errno=True)
    pidfd = _libc.syscall(_SYS_PIDFD_OPEN, pid, 0)
    if pidfd < 0:
        raise OSError(ctypes.get_errno(), f"pidfd_open({pid})")
    try:
        nfd = _libc.syscall(_SYS_PIDFD_GETFD, pidfd, fd, 0)
        if nfd < 0:
            raise OSError(
                ctypes.get_errno(),
                f"pidfd_getfd({pid}, {fd}); the instances need the same PID namespace "
                "and CAP_SYS_PTRACE (or a permissive ptrace_scope)",
            )
        return nfd
    finally:
        os.close(pidfd)


class _CudaArray:
    """Exposes a raw device allocation to torch.as_tensor."""

    def __init__(self, ptr: int, nbytes: int):
        """Wrap `nbytes` bytes at device address `ptr` as a uint8 array."""
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 2,
        }


def peer_instances() -> int:
    """Number of instances sharing engram tables (0 when peer sharing is off)."""
    n = envs.SGLANG_DSV41_ENGRAM_PEER_INSTANCES.get()
    return n if n and n > 1 else 0


class EngramPeerTable:
    """One engram layer's table, row-sharded over N instances x tp ranks.

    Shard bytes: [rows, dim] fp8 e4m3 then [rows, dim // block] e8m0, in one
    cuMemCreate allocation exported as a POSIX fd. Peers import it with
    pidfd_getfd and map it for their own GPU only. Legacy cudaIpc* mappings were
    ~60x slower for these random row reads (13.6 vs 0.21 ms per 20480-token
    layer) and create a context on every peer GPU (~0.6 GiB each).
    """

    def __init__(self, num_embeddings: int, dim: int, block: int, layer_id: int):
        """Allocate and map this rank's shard; peers are mapped later by connect()."""
        n = peer_instances()
        assert n > 1
        parallel = get_parallel()
        tp, tp_rank = parallel.tp_size, parallel.tp_rank
        device = torch.cuda.current_device()
        index = envs.SGLANG_DSV41_ENGRAM_PEER_INDEX.get()
        if index < 0:
            index = device // tp
        if not 0 <= index < n:
            raise RuntimeError(
                f"engram peer table: instance index {index} outside [0, {n}); set "
                "SGLANG_DSV41_ENGRAM_PEER_INDEX or launch with every GPU visible"
            )
        if torch.cuda.device_count() < n * tp:
            raise RuntimeError(
                f"engram peer table: {n} instances x tp {tp} need {n * tp} visible GPUs, "
                f"this process sees {torch.cuda.device_count()} (CUDA_VISIBLE_DEVICES "
                "must list the whole node; pick GPUs with --base-gpu-id instead)"
            )
        self.layer_id = layer_id
        self.dim, self.block = dim, block
        self.num_embeddings = num_embeddings
        self.num_shards = n * tp
        self.shard = index * tp + tp_rank
        self.device = device
        self.starts = [
            num_embeddings * i // self.num_shards for i in range(self.num_shards + 1)
        ]
        self.row_start = self.starts[self.shard]
        self.rows = self.starts[self.shard + 1] - self.row_start
        width = dim + dim // block
        torch.cuda.synchronize(device)  # the runtime's primary context is current
        prop = _alloc_prop(device)
        gran = ctypes.c_size_t()
        _check(
            _cu().cuMemGetAllocationGranularity(
                ctypes.byref(gran), ctypes.byref(prop), 1
            ),
            "cuMemGetAllocationGranularity",
        )
        self.nbytes = -(-max(1, self.rows * width) // gran.value) * gran.value
        handle = ctypes.c_uint64()
        _check(
            _cu().cuMemCreate(
                ctypes.byref(handle),
                ctypes.c_size_t(self.nbytes),
                ctypes.byref(prop),
                ctypes.c_uint64(0),
            ),
            "cuMemCreate",
        )
        fd = ctypes.c_int()
        _check(
            _cu().cuMemExportToShareableHandle(
                ctypes.byref(fd),
                ctypes.c_uint64(handle.value),
                _POSIX_FD,
                ctypes.c_uint64(0),
            ),
            "cuMemExportToShareableHandle",
        )
        # Kept open for the process lifetime: peers duplicate it with pidfd_getfd.
        self.fd = fd.value
        self.ptr = _map_for(handle.value, self.nbytes, device)
        raw = torch.as_tensor(
            _CudaArray(self.ptr, self.nbytes), device=f"cuda:{device}"
        )
        self.weight = (
            raw[: self.rows * dim].view(torch.float8_e4m3fn).view(self.rows, dim)
        )
        self.scale = (
            raw[self.rows * dim : self.rows * width]
            .view(torch.float8_e8m0fnu)
            .view(self.rows, dim // block)
        )
        self.bases: Optional[torch.Tensor] = None
        self.starts_dev: Optional[torch.Tensor] = None
        self._file: Optional[str] = None

    @property
    def ready(self) -> bool:
        """Whether every peer shard has been mapped (connect() finished)."""
        return self.bases is not None

    def _path(self, shard: int) -> str:
        """Rendezvous file that publishes `shard`."""
        d = envs.SGLANG_DSV41_ENGRAM_PEER_DIR.get()
        tag = envs.SGLANG_DSV41_ENGRAM_PEER_TAG.get()
        return os.path.join(
            d, f"{tag}_L{self.layer_id}_s{shard}of{self.num_shards}.pkl"
        )

    def _meta(self) -> dict:
        """Layout fields every instance's shard must agree on."""
        return {
            "num_embeddings": self.num_embeddings,
            "dim": self.dim,
            "block": self.block,
            "num_shards": self.num_shards,
            "starts": self.starts,
        }

    def connect(self) -> None:
        """Publish this shard (call once every row is written) and map every peer
        shard; blocks until all num_shards shards are published."""
        if self.ready:
            return
        torch.cuda.synchronize(self.device)
        record = {
            "pid": os.getpid(),
            "device": self.device,
            "fd": self.fd,
            "nbytes": self.nbytes,
            **self._meta(),
        }
        path = self._path(self.shard)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(record, f)
        os.replace(tmp, path)
        self._file = path
        atexit.register(self._unpublish)

        bases = [0] * self.num_shards
        bases[self.shard] = self.ptr
        pending = set(range(self.num_shards)) - {self.shard}
        deadline = time.monotonic() + envs.SGLANG_DSV41_ENGRAM_PEER_TIMEOUT.get()
        while pending:
            for s in sorted(pending):
                peer = self._read_peer(s)
                if peer is None:
                    continue
                nfd = _steal_fd(peer["pid"], peer["fd"])
                remote = ctypes.c_uint64()
                try:
                    _check(
                        _cu().cuMemImportFromShareableHandle(
                            ctypes.byref(remote), ctypes.c_void_p(nfd), _POSIX_FD
                        ),
                        f"cuMemImportFromShareableHandle(shard {s} on GPU {peer['device']})",
                    )
                finally:
                    os.close(nfd)
                bases[s] = _map_for(remote.value, peer["nbytes"], self.device)
                pending.discard(s)
            if pending:
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"engram peer table layer {self.layer_id}: shards {sorted(pending)} "
                        f"never published under {os.path.dirname(path)} (tag "
                        f"{envs.SGLANG_DSV41_ENGRAM_PEER_TAG.get()!r}); are all "
                        f"{peer_instances()} instances up?"
                    )
                time.sleep(1.0)
        torch.cuda.synchronize(self.device)
        dev = torch.device("cuda", self.device)
        self.bases = torch.tensor(bases, dtype=torch.int64, device=dev)
        self.starts_dev = torch.tensor(self.starts, dtype=torch.int64, device=dev)
        logger.info(
            "engram peer table layer %d: shard %d/%d rows [%d, %d) on GPU %d "
            "(%.2f GiB), %d peer shards mapped",
            self.layer_id,
            self.shard,
            self.num_shards,
            self.row_start,
            self.row_start + self.rows,
            self.device,
            self.rows * (self.dim + self.dim // self.block) / 2**30,
            self.num_shards - 1,
        )

    def _read_peer(self, shard: int) -> Optional[dict]:
        """Load a peer's published record; None if absent or left by a dead process."""
        try:
            with open(self._path(shard), "rb") as f:
                peer = pickle.load(f)
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            return None
        # A file left by a dead process holds a stale handle: wait for its successor.
        if not os.path.exists(f"/proc/{peer['pid']}"):
            return None
        mine = self._meta()
        if any(peer.get(k) != v for k, v in mine.items()):
            raise RuntimeError(
                f"engram peer table layer {self.layer_id}: shard {shard} was published "
                f"with a different layout ({ {k: peer.get(k) for k in mine} } vs {mine}); "
                "all instances must load the same checkpoint with the same "
                "SGLANG_DSV41_ENGRAM_PEER_INSTANCES and tp size"
            )
        return peer

    def _unpublish(self) -> None:
        """Remove this shard's rendezvous file at exit if this process still owns it."""
        try:
            if self._file is not None:
                with open(self._file, "rb") as f:
                    if pickle.load(f)["pid"] == os.getpid():
                        os.remove(self._file)
        except Exception:
            pass
