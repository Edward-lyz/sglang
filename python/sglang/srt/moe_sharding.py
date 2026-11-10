"""Runtime sharding for replicated Kimi K3 MoE weights.

Provides:
- Kimi K3 MoE weight-shard configuration.
- ShardedWeight: holds (shard, full_buffer, ready_event) for one weight.
- FlatAGScheduler: launches a single flat all_gather on a side stream and tracks per-layer events.

Compatibility constraints:
- Must work in cuda graph capture mode (no Python branching that depends on
  graph state; all decisions baked at __init__).
- All collective calls go through `torch.distributed` (NCCL backend).
- Side stream: a single shared `torch.cuda.Stream` leased from RuntimeContext
  via `get_moe_sharding_side_stream`; all MoE-sharded layers'
  AllGather runs on the same stream, kept separate from model-level
  alt_streams to avoid sharing slots with other consumers.
"""
from __future__ import annotations

import functools
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger(__name__)

_K3_CE_GROUP_SIZE = 8


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MoeShardingConfig:
    """Snapshot of Kimi K3 MoE weight-shard settings.

    Frozen so cuda graph capture sees consistent state.
    """
    enabled: bool = False
    # Comma-separated list of layer ids, or "all". Stored as raw string here;
    # use `targets(num_layers)` to materialise to set[int].
    layer_idx_spec: str = "all"

    @classmethod
    def from_env(cls) -> "MoeShardingConfig":
        """Build a config from the SGLANG MoE weight-shard env vars."""
        from sglang.srt.environ import envs

        return cls(
            enabled=envs.SGLANG_ENABLE_K3_MOE_WEIGHT_SHARD.get(),
            layer_idx_spec=envs.SGLANG_K3_MOE_WEIGHT_SHARD_LAYERS.get(),
        )


    def targets(self, num_layers: int) -> List[int]:
        """Materialise spec → sorted list of layer ids in [0, num_layers).

        Spec syntax:
          - "all"       → all layers
          - "2"         → [2]
          - "2,5,7"     → [2, 5, 7]
          - "2-10"      → [2..10]
          - "2-10,15"   → mix
        """
        ids = _parse_layer_spec(self.layer_idx_spec)
        if ids is None:
            return list(range(num_layers))
        return sorted({i for i in ids if 0 <= i < num_layers})


def _parse_layer_spec(spec: str) -> Optional[List[int]]:
    """Parse a layer-id spec string.

    Returns None when the spec selects "all" layers (caller materialises
    against its own num_layers). Otherwise returns the explicit list of
    ids in declaration order, unfiltered.
    """
    s = spec.strip().lower()
    if s == "all":
        return None
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def get_moe_sharding_config() -> MoeShardingConfig:
    """Return the MoE sharding config resolved from the environment."""
    return MoeShardingConfig.from_env()


@dataclass
class _WeightShardResources:
    shard_group: Optional[dist.ProcessGroup] = None
    shard_size: Optional[int] = None
    shard_rank: int = 0
    shard_kind: str = "none"
    nccl_ce_ext: object = None
    nccl_ce_ext_init_error: Optional[BaseException] = None
    allocations: list = field(default_factory=list)
    alloc_by_ptr: dict = field(default_factory=dict)
    windows: list = field(default_factory=list)
    comm_cache: dict = field(default_factory=dict)

    def shutdown(self) -> None:
        """Tear down NCCL windows and release weight-shard resources."""
        ext = self.nccl_ce_ext
        if ext is not None:
            try:
                torch.cuda.synchronize()
            except Exception as e:
                logger.warning(
                    "[K3WeightShard] device synchronization before teardown "
                    "failed: %r",
                    e,
                )
            while self.windows:
                comm_ptr, win_handle, _ = self.windows.pop()
                try:
                    ext.win_deregister(comm_ptr, win_handle)
                except Exception as e:
                    logger.warning(
                        "[K3WeightShard] window deregistration failed: %r", e
                    )
            while self.allocations:
                alloc = self.allocations.pop()
                try:
                    ext.free(alloc.data_ptr)
                except Exception as e:
                    logger.warning("[K3WeightShard] CE free failed: %r", e)
        self.windows.clear()
        self.allocations.clear()
        self.alloc_by_ptr.clear()
        self.comm_cache.clear()
        if self.shard_group is not None and dist.is_initialized():
            try:
                dist.destroy_process_group(self.shard_group)
            except Exception as e:
                logger.warning("[K3WeightShard] group destroy failed: %r", e)
        self.shard_group = None
        self.shard_size = None
        self.shard_rank = 0
        self.shard_kind = "none"


def _weight_shard_resources() -> _WeightShardResources:
    from sglang.srt.runtime_context import get_buffer

    return get_buffer("k3_weight_shard_manager", _WeightShardResources)


def get_moe_sharding_group_info() -> Tuple[Optional[dist.ProcessGroup], int, int, str]:
    """Resolve a node-local CE-capable subgroup of the tensor-parallel group."""
    resources = _weight_shard_resources()
    if resources.shard_size is not None:
        return (
            resources.shard_group,
            resources.shard_size,
            resources.shard_rank,
            resources.shard_kind,
        )

    def _cache_passthrough(kind: str = "none"):
        resources.shard_group = None
        resources.shard_size = 1
        resources.shard_rank = 0
        resources.shard_kind = kind
        return (
            resources.shard_group,
            resources.shard_size,
            resources.shard_rank,
            resources.shard_kind,
        )

    if not dist.is_initialized():
        return _cache_passthrough()

    try:
        from sglang.srt.distributed import get_tp_group

        tp_group = get_tp_group()
        tp_rank = tp_group.rank_in_group
    except Exception as e:
        logger.warning(
            "[K3WeightShard] Cannot resolve TP shard group: %s; "
            "falling back to passthrough",
            e,
        )
        return _cache_passthrough()

    if tp_group.world_size <= 1:
        return _cache_passthrough()

    # NCCL CE collectives are limited to one NVLink domain. TP ranks are laid
    # out contiguously per node, so cap each weight-shard communicator at the
    # eight local ranks instead of spanning a multi-node TP group.
    group_start = (tp_rank // _K3_CE_GROUP_SIZE) * _K3_CE_GROUP_SIZE
    member_ranks = list(
        tp_group.ranks[group_start : group_start + _K3_CE_GROUP_SIZE]
    )
    shard_size = len(member_ranks)
    shard_rank = tp_rank - group_start

    # Local synchronization lets each node create its CE communicator without
    # requiring ranks in the other node-local TP subgroups to enter this call.
    my_group = dist.new_group(
        ranks=member_ranks,
        use_local_synchronization=True,
    )
    resources.shard_group = my_group
    resources.shard_size = shard_size
    resources.shard_rank = shard_rank
    resources.shard_kind = "local_tp"
    logger.info(
        "[K3WeightShard] Built local TP shard group: members=%s shard_size=%d "
        "shard_rank=%d",
        member_ranks,
        shard_size,
        shard_rank,
    )
    return (
        resources.shard_group,
        resources.shard_size,
        resources.shard_rank,
        resources.shard_kind,
    )


def get_moe_sharding_side_stream() -> torch.cuda.Stream:
    """Lease the K3 weight-shard AllGather stream from RuntimeContext."""
    from sglang.srt.runtime_context import get_stream

    return get_stream("k3_weight_shard")
# data_ptr -> _CEAllocation; survives tensor view/contiguous/clone since the
# underlying storage retains the same address. Used by `_ce_alloc_for_tensor`
# to recover the CE record after Python attributes are lost across ops like
# `.t().contiguous()` performed by `process_weights_after_loading`.
# (comm_ptr, win_handle, owning data_ptr) tuples, kept so we can deregister
# windows in reverse registration order before freeing memory at shutdown.
# pg id (id(group)) -> ncclComm_t pointer


@dataclass
class _CEAllocation:
    """Bookkeeping for one ncclMemAlloc-backed buffer."""
    data_ptr: int
    num_bytes: int
    device_id: int
    storage_tensor: torch.Tensor  # uint8 view, keeps Python-side ref alive
    win_handles: List[Tuple[int, int]]  # list of (comm_ptr, win_handle)


# C++ source for the NCCL Copy-Engine / Symmetric Memory bindings, compiled
# on first use via torch.utils.cpp_extension.load_inline (see
# _load_nccl_ce_ext). Exposes thin wrappers over ncclMemAlloc/Free,
# ncclCommWindowRegister/Deregister, ncclGroupStart/End and ncclGetVersion.
_NCCL_CE_EXT_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <stdexcept>
#include <string>

#define NCCL_CHECK(cmd)                                                  \
  do {                                                                   \
    ncclResult_t r = cmd;                                                \
    if (r != ncclSuccess)                                                \
      throw std::runtime_error("NCCL error: " + std::to_string((int)r)); \
  } while (0)

// Allocate `num_bytes` of NCCL symmetric memory. Returns a 1-D uint8
// tensor that aliases the allocation (no free-on-DTOR; caller must call
// `free(t.data_ptr())` when done).
torch::Tensor nccl_alloc_bytes(int64_t num_bytes, int64_t device_id) {
    if (num_bytes <= 0) {
        throw std::runtime_error("alloc: num_bytes must be > 0");
    }
    void* ptr = nullptr;
    NCCL_CHECK(ncclMemAlloc(&ptr, static_cast<size_t>(num_bytes)));
    auto options = torch::TensorOptions()
                     .dtype(torch::kUInt8)
                     .device(torch::kCUDA, static_cast<int>(device_id));
    // Note: from_blob does NOT take ownership; release is via nccl_free.
    return torch::from_blob(ptr, {num_bytes}, options);
}

void nccl_free(int64_t data_ptr) {
  if (data_ptr == 0) return;
  NCCL_CHECK(ncclMemFree(reinterpret_cast<void*>(data_ptr)));
}

int64_t nccl_win_register(int64_t comm_ptr, int64_t data_ptr, int64_t num_bytes) {
  ncclWindow_t win = nullptr;
  NCCL_CHECK(ncclCommWindowRegister(
      reinterpret_cast<ncclComm_t>(comm_ptr),
      reinterpret_cast<void*>(data_ptr),
      static_cast<size_t>(num_bytes),
      &win,
      NCCL_WIN_COLL_SYMMETRIC));
  return reinterpret_cast<int64_t>(win);
}

void nccl_win_deregister(int64_t comm_ptr, int64_t win) {
  NCCL_CHECK(ncclCommWindowDeregister(
      reinterpret_cast<ncclComm_t>(comm_ptr),
      reinterpret_cast<ncclWindow_t>(win)));
}

void nccl_group_start() { NCCL_CHECK(ncclGroupStart()); }
void nccl_group_end() { NCCL_CHECK(ncclGroupEnd()); }

// Returns NCCL runtime version (e.g. 22803 for 2.28.3).
int64_t nccl_version() {
  int v = 0;
  NCCL_CHECK(ncclGetVersion(&v));
  return static_cast<int64_t>(v);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("alloc", &nccl_alloc_bytes,
        "ncclMemAlloc num_bytes; returns 1-D uint8 cuda tensor aliasing the allocation");
  m.def("free", &nccl_free, "ncclMemFree by raw data_ptr");
  m.def("win_register", &nccl_win_register,
        "ncclCommWindowRegister(NCCL_WIN_COLL_SYMMETRIC); returns ncclWindow_t handle");
  m.def("win_deregister", &nccl_win_deregister, "ncclCommWindowDeregister");
  m.def("group_start", &nccl_group_start);
  m.def("group_end", &nccl_group_end);
  m.def("version", &nccl_version, "NCCL runtime version (major*10000+minor*100+patch)");
}
"""


def _load_nccl_ce_ext():
    """JIT-compile and load the NCCL CE / Symmetric Memory bindings.

    The C++ source is inlined here and built on first use via
    ``torch.utils.cpp_extension.load_inline`` (no separate setup step or
    prebuilt ``.so`` required). Result is cached so compilation and any
    error message happen only once.

    Build requires the NCCL headers/libs. Resolution order for the NCCL
    install prefix:
      1. ``NCCL_HOME`` (then ``<prefix>/include`` and ``<prefix>/lib``),
         overridable individually via ``NCCL_INCLUDE_DIR`` / ``NCCL_LIB_DIR``.
      2. Otherwise rely on the toolchain's default search paths.
    """
    resources = _weight_shard_resources()
    if resources.nccl_ce_ext is not None:
        return resources.nccl_ce_ext
    init_error = resources.nccl_ce_ext_init_error
    if init_error is not None:
        raise RuntimeError(
            f"[MoeSharding CE] nccl_ce_ext previously failed to build/load: {init_error!r}"
        )

    try:
        from torch.utils.cpp_extension import load_inline

        nccl_home = os.environ.get("NCCL_HOME", "/usr/local/nccl")
        nccl_include = os.environ.get(
            "NCCL_INCLUDE_DIR", os.path.join(nccl_home, "include")
        )
        nccl_lib = os.environ.get("NCCL_LIB_DIR", os.path.join(nccl_home, "lib"))

        extra_include = [nccl_include] if os.path.isdir(nccl_include) else []
        extra_ldflags = ["-lnccl"]
        if os.path.isdir(nccl_lib):
            extra_ldflags = [f"-L{nccl_lib}", f"-Wl,-rpath,{nccl_lib}", "-lnccl"]

        nccl_ce_ext = load_inline(
            name="nccl_ce_ext",
            cpp_sources=[_NCCL_CE_EXT_CPP_SOURCE],
            extra_include_paths=extra_include,
            extra_ldflags=extra_ldflags,
            with_cuda=True,
            verbose=False,
        )
    except Exception as e:  # pragma: no cover - depends on local toolchain
        resources.nccl_ce_ext_init_error = e
        logger.error(
            "[MoeSharding CE] failed to JIT-build nccl_ce_ext: %r. Ensure NCCL "
            "headers/libs are discoverable (set NCCL_HOME, or "
            "NCCL_INCLUDE_DIR / NCCL_LIB_DIR) and a CUDA toolchain (nvcc) is "
            "available. NCCL >= 2.28.3 is required for CE/Symmetric-Memory "
            "AllGather.",
            e,
        )
        raise

    resources.nccl_ce_ext = nccl_ce_ext
    _cta_policy = os.environ.get("NCCL_CTA_POLICY")
    if _cta_policy != "2":
        raise RuntimeError(
            "[MoeSharding CE] NCCL_CTA_POLICY=2 is required for the NCCL 2.28.3+ Copy "
            "Engine / Symmetric Memory AllGather algorithm to be selected, "
            f"but got NCCL_CTA_POLICY={_cta_policy!r}. Set it in the worker "
            "environment BEFORE the NCCL communicator is created (i.e. before "
            "any torch.distributed collective)."
        )
    try:
        ver = nccl_ce_ext.version()
    except Exception:
        ver = -1
    logger.info(f"[MoeSharding CE] nccl_ce_ext loaded; NCCL packed version = {ver}")
    if ver != -1 and ver < 22803:
        logger.warning(
            f"[MoeSharding CE] NCCL version {ver} < 22803; AllGather over symmetric "
            f"memory may not select Copy Engine algorithms. Consider NCCL >= 2.28.3."
        )
    return resources.nccl_ce_ext


def get_ncclcomm_from_pg(group: Optional[dist.ProcessGroup]) -> int:
    """Extract the underlying ``ncclComm_t`` (as int64) from a torch PG.

    Tries multiple PyTorch-private APIs in sequence to handle 2.5/2.6/2.7
    drift. Caches the result keyed by ``id(group)``.
    """
    if group is None:
        raise RuntimeError("get_ncclcomm_from_pg: group is None")
    resources = _weight_shard_resources()
    cache_key = id(group)
    cached = resources.comm_cache.get(cache_key)
    if cached is not None:
        return cached

    # PyTorch ProcessGroupNCCL lazy-creates the ncclComm_t on first
    # collective. Force creation with a tiny allreduce on the group's
    # default device.
    try:
        dev_idx = torch.cuda.current_device()
        probe = torch.zeros(1, dtype=torch.float32, device=f"cuda:{dev_idx}")
        dist.all_reduce(probe, group=group)
        torch.cuda.synchronize()
    except Exception as e:
        logger.warning(f"[MoeSharding CE] probe all_reduce on shard group failed: {e!r}")

    backend = None
    try:
        backend = group._get_backend(torch.device("cuda"))  # type: ignore[attr-defined]
    except Exception:
        try:
            backend = dist.distributed_c10d._get_pg_default_device(group)  # type: ignore[attr-defined]
            backend = group._get_backend(backend)  # type: ignore[attr-defined]
        except Exception as e:
            raise RuntimeError(
                f"[MoeSharding CE] Cannot resolve ProcessGroupNCCL backend from group: {e!r}"
            )

    last_err: Optional[BaseException] = None
    tried: List[str] = []
    for attr in (
        "_comm_ptr",        # observed on this build (test_allgather_ce_simple.py)
        "_get_nccl_comm",
        "get_nccl_comm",
        "_getNcclComm",
        "comm",
        "ncclComm",
        "nccl_comm",
    ):
        fn = getattr(backend, attr, None)
        if fn is None:
            continue
        tried.append(attr)
        try:
            comm = fn() if callable(fn) else fn
            if isinstance(comm, int) and comm != 0:
                resources.comm_cache[cache_key] = int(comm)
                logger.info(
                    f"[MoeSharding CE] ncclComm_t resolved via backend.{attr} -> "
                    f"0x{int(comm):x}"
                )
                return int(comm)
            # Some bindings return an opaque capsule; reject for now.
        except Exception as e:
            last_err = e
            continue

    # Diagnostic dump: enumerate every comm-ish attribute on backend so the
    # user can tell us which symbol the local PyTorch actually exposes.
    diag = sorted(
        a for a in dir(backend)
        if ("comm" in a.lower() or "nccl" in a.lower())
        and not a.startswith("__")
    )
    raise RuntimeError(
        "[MoeSharding CE] Failed to extract ncclComm_t from torch ProcessGroupNCCL "
        f"(backend={type(backend).__name__}, tried={tried}, "
        f"last_err={last_err!r}, comm-like attrs on backend={diag}). "
        f"Consider upgrading PyTorch; MoE sharding requires CE AllGather."
    )


def ce_alloc_tensor(
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Allocate a tensor of the requested shape/dtype/device backed by
    ``ncclMemAlloc`` storage. The returned tensor is a typed view over a
    uint8 byte buffer; the underlying allocation is tracked in
    the RuntimeContext resource manager so registration and freeing can find it.
    """
    ext = _load_nccl_ce_ext()
    if device.type != "cuda":
        raise RuntimeError(f"ce_alloc_tensor only supports CUDA, got {device}")
    dev_idx = device.index if device.index is not None else torch.cuda.current_device()
    elem_size = torch.tensor([], dtype=dtype).element_size()
    numel = 1
    for s in shape:
        numel *= int(s)
    num_bytes = numel * elem_size
    if num_bytes == 0:
        # Symmetric-memory windows must be non-empty; fall back to a
        # regular empty tensor for the deferred-buffer placeholder case.
        return torch.empty(shape, dtype=dtype, device=device)

    byte_tensor = ext.alloc(num_bytes, int(dev_idx))
    data_ptr = int(byte_tensor.data_ptr())
    typed = byte_tensor.view(dtype).view(*shape)
    alloc = _CEAllocation(
        data_ptr=data_ptr,
        num_bytes=num_bytes,
        device_id=int(dev_idx),
        storage_tensor=byte_tensor,
        win_handles=[],
    )
    resources = _weight_shard_resources()
    resources.allocations.append(alloc)
    resources.alloc_by_ptr[data_ptr] = alloc
    # Stash the allocation record on the tensor so callers can re-find it
    # (e.g. for window registration after dp_group is known). Note: this
    # attribute is lost across ops that produce a new Tensor object (view,
    # contiguous, clone, narrow, t). The data_ptr-keyed registry above is
    # the durable lookup path.
    typed._as_ce_alloc = alloc  # type: ignore[attr-defined]
    return typed


def ce_alloc_concat_views(
    specs: Sequence[Tuple[str, Tuple[int, ...], torch.dtype]],
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Tuple[int, int]], int]:
    """Allocate ONE CE-backed byte buffer holding all `(name, shape, dtype)`
    entries laid out back-to-back (each entry padded to 16B alignment so
    typed views are dtype-aligned).

    Returns ``(byte_buffer, {name: typed_view}, {name: (off, nb)}, total_padded_bytes)``:
      - ``byte_buffer``      : 1D uint8 CE allocation covering the whole concat region.
      - ``typed_views``      : per-name typed view onto its sub-region.
      - ``offsets``          : per-name ``(byte_offset, byte_size)`` (size is the
                               actual unpadded payload; offset is into the byte buffer).
      - ``total_padded_bytes``: full size of ``byte_buffer`` (sum of padded entries).

    All returned views share the SAME underlying `_CEAllocation` as the
    byte buffer, so a SINGLE ``ce_register_window(byte_buffer, group)``
    call covers every typed view for both AG endpoints (shard staging
    side as well as full_buf side). This is what gets the per-comm
    symmetric-window count down from O(num_layers × num_weights) to
    O(num_pingpong_slots).
    """
    if device.type != "cuda":
        raise RuntimeError(f"ce_alloc_concat_views only supports CUDA, got {device}")
    entries: List[Tuple[str, Tuple[int, ...], torch.dtype, int, int, int]] = []
    total_bytes = 0
    for name, shape, dtype in specs:
        elem = torch.tensor([], dtype=dtype).element_size()
        numel = 1
        for s in shape:
            numel *= int(s)
        nb = numel * elem
        # Pad to 16B so the next entry's offset is aligned for any common dtype.
        nb_padded = (nb + 15) // 16 * 16
        entries.append((name, tuple(shape), dtype, nb, nb_padded, total_bytes))
        total_bytes += nb_padded
    if total_bytes == 0:
        raise RuntimeError("ce_alloc_concat_views: empty spec list")
    byte_buf = ce_alloc_tensor((total_bytes,), torch.uint8, device)
    views: Dict[str, torch.Tensor] = {}
    offsets: Dict[str, Tuple[int, int]] = {}
    for name, shape, dtype, nb, _nb_padded, off in entries:
        seg = byte_buf.narrow(0, off, nb)
        typed = seg.view(dtype).view(*shape)
        # Stash the same alloc on each view so fast-path lookup works.
        typed._as_ce_alloc = byte_buf._as_ce_alloc  # type: ignore[attr-defined]
        views[name] = typed
        offsets[name] = (off, nb)
    return byte_buf, views, offsets, total_bytes


def _ce_alloc_for_tensor(tensor: torch.Tensor) -> Optional["_CEAllocation"]:
    """Return the `_CEAllocation` backing `tensor` if any, else None.

    Searches in this order:
      1. The `_as_ce_alloc` attribute (fast path; lost across view ops).
      2. The data_ptr → alloc registry (handles `.view`, `.contiguous`,
         `.t`, `.narrow`, etc., which produce a new Tensor object but
         share or sit inside the original CE storage).
      3. Linear scan over tracked allocations for `tensor.data_ptr()` falling
         within an allocation's `[data_ptr, data_ptr + num_bytes)` range
         (covers narrowed / sliced tensors whose data_ptr is offset).
    """
    alloc = getattr(tensor, "_as_ce_alloc", None)
    if alloc is not None:
        return alloc
    if tensor.numel() == 0:
        return None
    ptr = int(tensor.data_ptr())
    resources = _weight_shard_resources()
    alloc = resources.alloc_by_ptr.get(ptr)
    if alloc is not None:
        return alloc
    for a in resources.allocations:
        if a.data_ptr <= ptr < a.data_ptr + a.num_bytes:
            return a
    return None


def ce_register_window(
    tensor: torch.Tensor,
    group: Optional[dist.ProcessGroup],
) -> Optional[int]:
    """Register a CE-allocated tensor as a NCCL_WIN_COLL_SYMMETRIC window
    on the given dp group. Returns the window handle, or None if the
    tensor is not CE-backed (e.g. dp_size == 1 passthrough).

    All ranks in `group` MUST call this on equivalent tensors in the same
    order with the same byte size; the calling code is responsible for
    enforcing that ordering (the model-level finalize loop walks MoE-sharded
    layers in sorted layer_id order).
    """
    if group is None:
        return None
    alloc = _ce_alloc_for_tensor(tensor)
    if alloc is None:
        # Not CE-backed (e.g. fell back to torch.empty for 0-byte placeholder,
        # or storage was replaced by `process_weights_after_loading` with a
        # regular caching-allocator buffer).
        return None
    # Skip duplicate registration on the same comm.
    comm_ptr = get_ncclcomm_from_pg(group)
    for existing_comm, _ in alloc.win_handles:
        if existing_comm == comm_ptr:
            return None
    ext = _load_nccl_ce_ext()
    win = ext.win_register(comm_ptr, alloc.data_ptr, alloc.num_bytes)
    alloc.win_handles.append((comm_ptr, int(win)))
    _weight_shard_resources().windows.append(
        (comm_ptr, int(win), alloc.data_ptr)
    )
    return int(win)


def ce_register_windows_grouped(
    tensors: Sequence[torch.Tensor],
    group: Optional[dist.ProcessGroup],
) -> List[Optional[int]]:
    """Register a list of CE-allocated tensors as
    NCCL_WIN_COLL_SYMMETRIC windows on the given dp group.

    TODO: the current implementation reaches into NCCL via a hand-written
    C++ extension (``_load_nccl_ce_ext``) for ``ncclCommWindowRegister``.
    This should be replaced with ``nccl4py`` once it exposes window
    register/deregister APIs, so we can drop the custom C++ extension
    entirely.

    Returns one entry per input tensor: the registered window handle, or
    None when the tensor is passthrough / not CE-backed / already
    registered on this comm.

    All ranks in ``group`` MUST call this with equivalent tensor lists in
    the same order with the same per-tensor byte size.
    """
    n = len(tensors)
    out: List[Optional[int]] = [None] * n
    if group is None or n == 0:
        logger.info(
            f"[MoeSharding CE] ce_register_windows_grouped: SKIP (group is None or empty), n={n}"
        )
        return out

    # Resolve allocations + dedup-against-existing per tensor.
    comm_ptr = get_ncclcomm_from_pg(group)
    pending: List[Tuple[int, "_CEAllocation"]] = []  # (output_index, alloc)
    miss_not_ce = 0
    miss_dup = 0
    miss_details: List[str] = []
    for i, t in enumerate(tensors):
        alloc = _ce_alloc_for_tensor(t)
        if alloc is None:
            miss_not_ce += 1
            if len(miss_details) < 4:
                miss_details.append(
                    f"  [{i}] not CE-backed: shape={tuple(t.shape)} dtype={t.dtype} "
                    f"data_ptr=0x{t.data_ptr():x} numel={t.numel()}"
                )
            continue
        if any(c == comm_ptr for c, _ in alloc.win_handles):
            miss_dup += 1
            continue
        pending.append((i, alloc))

    rank0 = (not dist.is_initialized()) or (dist.get_rank() == 0)
    if rank0:
        logger.info(
            f"[MoeSharding CE] ce_register_windows_grouped on comm 0x{comm_ptr:x}: "
            f"in={n} not_ce_backed={miss_not_ce} already_registered={miss_dup} "
            f"to_register={len(pending)}"
        )
        for line in miss_details:
            logger.debug(f"[MoeSharding CE]{line}")

    if not pending:
        return out

    ext = _load_nccl_ce_ext()
    if rank0:
        sample = pending[0][1]
        logger.info(
            f"[MoeSharding CE] registering {len(pending)} windows (per-tensor); "
            f"sample alloc: data_ptr=0x{sample.data_ptr:x} num_bytes={sample.num_bytes}"
        )
    # NOTE: NCCL 22803 segfaults if multiple ncclCommWindowRegister calls
    # are batched inside ncclGroupStart/ncclGroupEnd; do per-tensor
    # registration without group wrapping (this matches the original
    # ce_register_window semantics, just batched-iterated and logged).
    wins: List[int] = []
    resources = _weight_shard_resources()
    try:
        for idx, alloc in pending:
            win = int(ext.win_register(comm_ptr, alloc.data_ptr, alloc.num_bytes))
            wins.append(win)
            alloc.win_handles.append((comm_ptr, win))
            resources.windows.append((comm_ptr, win, alloc.data_ptr))
            out[idx] = win
    except Exception:
        for win in reversed(wins):
            try:
                ext.win_deregister(comm_ptr, win)
            except Exception:
                logger.exception(
                    "[K3WeightShard] failed to roll back window 0x%x", win
                )
        registered = set(wins)
        resources.windows[:] = [
            item
            for item in resources.windows
            if not (item[0] == comm_ptr and item[1] in registered)
        ]
        for _, alloc in pending:
            alloc.win_handles[:] = [
                item
                for item in alloc.win_handles
                if not (item[0] == comm_ptr and item[1] in registered)
            ]
        raise
    if rank0:
        logger.info(
            f"[MoeSharding CE] ce_register_windows_grouped DONE: registered {len(wins)} windows "
            f"on comm 0x{comm_ptr:x} (handles: "
            f"{[hex(w) for w in wins[:4]]}{'...' if len(wins) > 4 else ''})"
        )
    return out


def _assert_ce_registered_pair(
    shard: torch.Tensor,
    full_buf: torch.Tensor,
    group: Optional[dist.ProcessGroup],
    name: str = "<unnamed>",
) -> None:
    """When CE backend is active, both AG endpoints must own a registered
    NCCL_WIN_COLL_SYMMETRIC window on the dp comm. Raises if either side
    is missing a `_window_handle`-equivalent (i.e. an `_as_ce_alloc`
    record carrying at least one win_handle for this comm).

    No-op when ``group`` is None (passthrough / dp_size==1).
    """
    if group is None:
        return
    try:
        comm_ptr = get_ncclcomm_from_pg(group)
    except Exception as e:
        raise RuntimeError(
            f"[MoeSharding CE] cannot resolve ncclComm_t for {name}: {e!r}"
        )
    for label, t in (("shard", shard), ("full_buf", full_buf)):
        alloc = _ce_alloc_for_tensor(t)
        if alloc is None:
            raise RuntimeError(
                f"[MoeSharding CE] {name}.{label} is not CE-allocated; expected "
                f"ncclMemAlloc-backed storage for CE AllGather"
            )
        if not any(c == comm_ptr for c, _ in alloc.win_handles):
            raise RuntimeError(
                f"[MoeSharding CE] {name}.{label} has no symmetric window registered "
                f"for comm 0x{comm_ptr:x}; ce_register_window must run for "
                f"both shard and full_buf before launch"
            )


def ce_free_tensor(tensor: torch.Tensor) -> bool:
    """Free the CE allocation backing `tensor` (deregister any windows,
    then ``ncclMemFree``) and drop it from the resource manager. Returns
    True if a CE allocation was found and freed, else False.

    Caller MUST ensure the tensor (and any view of it) is no longer used
    after this call. Intended for one-shot CE buffers that are not part of
    the persistent hot-path set (e.g. per-weight shards consumed once by
    the finalize redistribute).
    """
    alloc = _ce_alloc_for_tensor(tensor)
    if alloc is None:
        return False
    resources = _weight_shard_resources()
    ext = resources.nccl_ce_ext
    if ext is None:
        return False
    # Deregister any symmetric windows opened on this allocation.
    for comm_ptr, win_handle in list(alloc.win_handles):
        try:
            ext.win_deregister(comm_ptr, win_handle)
        except Exception as e:
            logger.warning(f"[MoeSharding CE] win_deregister failed during free: {e!r}")
            return False
        alloc.win_handles.remove((comm_ptr, win_handle))
        for i, (c, w, _p) in enumerate(resources.windows):
            if c == comm_ptr and w == win_handle:
                resources.windows.pop(i)
                break
    try:
        ext.free(alloc.data_ptr)
    except Exception as e:
        logger.warning(f"[MoeSharding CE] ncclMemFree failed during free: {e!r}")
        return False
    try:
        resources.allocations.remove(alloc)
    except ValueError:
        pass
    resources.alloc_by_ptr.pop(alloc.data_ptr, None)
    return True


def ce_shutdown() -> None:
    """Best-effort teardown: deregister all symmetric windows and free
    every CE-allocated buffer. Safe to call multiple times.
    """
    _weight_shard_resources().shutdown()


# --------------------------------------------------------------------------
# ShardedWeight: holds (shard, full_buf, ready_event) for one weight
# --------------------------------------------------------------------------


@dataclass
class ShardedWeight:
    """Holds the trio (shard, full_buf, ready_event) for one weight.

    Layout convention:
        full weight has shape (D0, D1, ..., Dn) and dtype W.
        shard splits along ``shard_dim`` (dim 0 for the supported CP/DP engine
        path used by MoE sharding).
        After AG, full_buf.shape == (D0, D1, ..., Dn).

    Caller responsibilities:
        - Allocate `shard` and load weights into it (custom weight_loader).
        - Allocate `full_buf` with same dtype/device.
        - Pass in a `ready_event` (CUDA event) — recorded when AG completes.

    Notes:
        - If dp_size == 1, full_buf == shard (no AG needed); we set
          `is_passthrough = True` and skip all_gather.
    """
    name: str
    shard: torch.Tensor
    full_buf: torch.Tensor
    ready_event: torch.cuda.Event
    dp_size: int
    dp_group: Optional[dist.ProcessGroup]
    shard_dim: int = 0  # 0 = split out_features (Linear weight default)
    is_passthrough: bool = False

    def __post_init__(self):
        assert self.shard.dtype == self.full_buf.dtype, "dtype mismatch"
        assert self.shard.device == self.full_buf.device, "device mismatch"
        if self.dp_size == 1:
            assert self.shard.data_ptr() == self.full_buf.data_ptr(), (
                "dp_size=1 should pass shard==full_buf to avoid copy"
            )
            self.is_passthrough = True
            return
        ndim = self.shard.ndim
        sd = self.shard_dim if self.shard_dim >= 0 else ndim + self.shard_dim
        assert 0 <= sd < ndim, f"shard_dim {self.shard_dim} out of range"
        # full_buf may be a deferred 0-element placeholder (load-time path);
        # only validate full-shape consistency when full_buf is materialised.
        if self.full_buf.numel() > 0:
            expected_full = self.shard.shape[sd] * self.dp_size
            assert self.full_buf.shape[sd] == expected_full, (
                f"full_buf dim {sd} = {self.full_buf.shape[sd]} != "
                f"shard dim {sd} = {self.shard.shape[sd]} × dp_size {self.dp_size}"
            )

    @classmethod
    def allocate(
        cls,
        name: str,
        full_shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        dp_size: int,
        dp_group: Optional[dist.ProcessGroup],
        shard_dim: int = 0,
        full_buf_override: Optional[torch.Tensor] = None,
        defer_full_buf: bool = False,
        shard_storage: Optional[torch.Tensor] = None,
        use_ce: bool = False,
    ) -> "ShardedWeight":
        """Convenience factory that allocates both buffers + a fresh event.

        Args:
            full_buf_override: If provided, use this tensor as `full_buf`
                instead of allocating a fresh one. Caller (e.g. workspace
                pool) is responsible for shape/dtype/device match. This lets
                multiple `ShardedWeight` instances share a small pool of
                workspace buffers (ping-pong) instead of each owning a
                full-sized buffer permanently.
            defer_full_buf: If True, skip allocating `full_buf` and store a
                0-element placeholder instead. The caller is responsible for
                installing a real full-shape buffer (via direct attribute
                assignment to `sharded.full_buf`) before any AG runs. Used
                by the load-time DP-shard path where the pool buffer is
                attached during model-level finalization.
            shard_storage: If provided, reuse this tensor as `shard` instead
                of allocating a fresh one. Lets the caller share storage
                with `target_mod.weight.data` (load-time path) so the
                weight_loader writes directly into the shard.
            use_ce: If True, allocate every freshly created buffer (`shard`
                when ``shard_storage is None``, and `full_buf` when neither
                ``full_buf_override`` nor ``defer_full_buf`` is set) via
                ``ncclMemAlloc`` so the dp comm can later register them as
                NCCL_WIN_COLL_SYMMETRIC windows. Buffers passed in by the
                caller (override / shard_storage) are taken as-is and the
                caller is responsible for their CE-eligibility.
        """
        def _alloc(shape: Tuple[int, ...]) -> torch.Tensor:
            if use_ce:
                return ce_alloc_tensor(tuple(shape), dtype, device)
            return torch.empty(shape, dtype=dtype, device=device)

        if dp_size == 1:
            buf = (
                shard_storage
                if shard_storage is not None
                else _alloc(full_shape)
            )
            return cls(
                name=name,
                shard=buf,
                full_buf=buf,
                ready_event=torch.cuda.Event(),
                dp_size=1,
                dp_group=dp_group,
                shard_dim=shard_dim,
                is_passthrough=True,
            )
        sd = shard_dim if shard_dim >= 0 else len(full_shape) + shard_dim
        assert full_shape[sd] % dp_size == 0, (
            f"full_shape dim {sd} = {full_shape[sd]} not divisible by dp_size {dp_size}"
        )
        shard_shape = list(full_shape)
        shard_shape[sd] = shard_shape[sd] // dp_size
        if shard_storage is not None:
            assert tuple(shard_storage.shape) == tuple(shard_shape), (
                f"shard_storage shape {tuple(shard_storage.shape)} != "
                f"expected shard_shape {tuple(shard_shape)}"
            )
            assert shard_storage.dtype == dtype, "shard_storage dtype mismatch"
            assert shard_storage.device == device, "shard_storage device mismatch"
            shard = shard_storage
        else:
            shard = _alloc(tuple(shard_shape))
        if defer_full_buf:
            assert full_buf_override is None, (
                "defer_full_buf=True is incompatible with full_buf_override"
            )
            full_buf = torch.empty(0, dtype=dtype, device=device)
        elif full_buf_override is not None:
            assert tuple(full_buf_override.shape) == tuple(full_shape), (
                f"full_buf_override shape {tuple(full_buf_override.shape)} "
                f"!= expected full_shape {tuple(full_shape)}"
            )
            assert full_buf_override.dtype == dtype, "full_buf_override dtype mismatch"
            assert full_buf_override.device == device, "full_buf_override device mismatch"
            full_buf = full_buf_override
        else:
            full_buf = _alloc(tuple(full_shape))
        return cls(
            name=name,
            shard=shard,
            full_buf=full_buf,
            ready_event=torch.cuda.Event(),
            dp_size=dp_size,
            dp_group=dp_group,
            shard_dim=shard_dim,
        )


@dataclass(frozen=True)
class FlatLayout:
    """Static byte layout of T = concat(weight_full, padded).

    Fields
        entries        : per-name (name, byte_offset, nb_unpadded, nb_padded,
                         full_shape, dtype). Offset is into T.
        total_bytes    : |T|. Multiple of (16 * dp_size).
        chunk_bytes    : |T| / dp_size. Per-rank chunk size in bytes.
    """
    entries: Tuple[Tuple[str, int, int, int, Tuple[int, ...], torch.dtype], ...]
    total_bytes: int
    chunk_bytes: int


def build_flat_layout(
    specs: Sequence[Tuple[str, Tuple[int, ...], torch.dtype]],
    dp_size: int,
) -> FlatLayout:
    """Compute a FlatLayout from (name, full_shape, dtype) specs.

    Each entry is laid out back-to-back with 16B padding between entries.
    The total is then rounded up to a multiple of `16 * dp_size` so the
    per-rank chunk is 16B-aligned (CE/NVLink prefer 16B alignment).
    """
    if dp_size < 1:
        raise ValueError(f"dp_size must be >= 1 (got {dp_size})")
    entries: List[Tuple[str, int, int, int, Tuple[int, ...], torch.dtype]] = []
    off = 0
    for name, shape, dtype in specs:
        elem = torch.tensor([], dtype=dtype).element_size()
        numel = 1
        for s in shape:
            numel *= int(s)
        nb = numel * elem
        nb_padded = (nb + 15) // 16 * 16
        entries.append((name, off, nb, nb_padded, tuple(shape), dtype))
        off += nb_padded
    if off == 0:
        raise RuntimeError("build_flat_layout: empty spec list")
    align = 16 * dp_size
    total = (off + align - 1) // align * align
    chunk = total // dp_size
    return FlatLayout(
        entries=tuple(entries),
        total_bytes=total,
        chunk_bytes=chunk,
    )


class FlatAGStagingPool:
    """Ping-pong pool for FlatAGScheduler.

    Each slot holds:
      - flat_buf  (CE uint8, |T| bytes)         : AG output endpoint.
      - views     (Dict[name, typed Tensor])    : narrow+view onto flat_buf.

    ``flat_buf`` is CE-allocated and registered as a symmetric window on
    the dp comm before first AG. The AG *input* is the per-layer
    ``flat_shard`` (owned by ``FlatAGScheduler``, also CE-registered) —
    the pool no longer holds a separate shard staging buffer.
    """

    def __init__(
        self,
        layout: FlatLayout,
        device: torch.device,
        dp_size: int,
        n_buffers: int = 2,
    ):
        assert n_buffers >= 1, "need at least 1 buffer"
        self.layout = layout
        self.dp_size = int(dp_size)
        self.n_buffers = n_buffers
        self.device = device
        self.flat_bufs: List[torch.Tensor] = []
        self.views_per_slot: List[Dict[str, torch.Tensor]] = []
        for _ in range(n_buffers):
            fb = ce_alloc_tensor((layout.total_bytes,), torch.uint8, device)
            self.flat_bufs.append(fb)
            views: Dict[str, torch.Tensor] = {}
            for name, off, nb, _nb_padded, shape, dtype in layout.entries:
                seg = fb.narrow(0, off, nb)
                typed = seg.view(dtype).view(*shape)
                # Stash CE alloc for fast _ce_alloc_for_tensor lookup on
                # the typed view (narrow+view drop the attribute).
                typed._as_ce_alloc = fb._as_ce_alloc  # type: ignore[attr-defined]
                views[name] = typed
            self.views_per_slot.append(views)
        total_mib = n_buffers * layout.total_bytes / (1024 ** 2)
        logger.info(
            f"[MoeSharding] FlatAGStagingPool n={n_buffers} dp_size={dp_size} "
            f"|T|={layout.total_bytes} chunk={layout.chunk_bytes} "
            f"entries={[(n, sh, dt) for n, _, _, _, sh, dt in layout.entries]} "
            f"total={total_mib:.1f} MiB"
        )

    def borrow_views(self, seq_idx: int) -> Dict[str, torch.Tensor]:
        """Return typed full-tensor views for the staging slot."""
        return self.views_per_slot[seq_idx % self.n_buffers]

    def borrow_flat(self, seq_idx: int) -> torch.Tensor:
        """Return the flat all-gather buffer for the staging slot."""
        return self.flat_bufs[seq_idx % self.n_buffers]


def _redistribute_to_flat_shard(
    layer,
    layout: FlatLayout,
    dp_group: dist.ProcessGroup,
    dp_size: int,
    dp_rank: int,
    device: torch.device,
) -> torch.Tensor:
    """One-time finalize redistribute: per-weight 1/dp_size shards -> flat 1/dp_size chunk.

    Steps (off the hot path):
      1. AG each per-weight shard into a temporary full tensor.
      2. For each weight, copy only the bytes that fall inside this rank's
         chunk window directly into the persistent CE-allocated flat_shard
         buffer. No |T|-sized scratch tensor is materialised.
      3. Free per-weight temporaries as we go.

    Returns the persistent CE flat_shard tensor (registered downstream by
    the caller via `ce_register_window` / `ce_register_windows_grouped`).
    """
    slot_attrs = {
        role: sharded_attr
        for role, sharded_attr, _target_path in layer._moe_shard_targets
    }
    chunk = layout.chunk_bytes
    my_lo = dp_rank * chunk
    my_hi = my_lo + chunk
    # CE-allocated persistent destination; zero so any padding bytes that
    # don't get overwritten by a weight slice are deterministic.
    flat_shard = ce_alloc_tensor((chunk,), torch.uint8, device)
    flat_shard.zero_()

    for name, off, nb, _nb_padded, shape, dtype in layout.entries:
        sharded = getattr(layer, slot_attrs[name], None)
        if sharded is None:
            continue
        if sharded.is_passthrough:
            full = sharded.shard
        else:
            if not sharded.shard.is_contiguous():
                raise RuntimeError(
                    f"flat redistribute requires contiguous shard for {name}: "
                    f"shape={tuple(sharded.shard.shape)} stride={tuple(sharded.shard.stride())}"
                )
            full = torch.empty(list(shape), dtype=dtype, device=device)
            # Regular NCCL AG (off hot path). CE registration not required
            # for this one-shot redistribute.
            dist.all_gather_into_tensor(full, sharded.shard, group=dp_group)

        # Intersection of weight's byte range [off, off+nb) with this rank's
        # chunk window [my_lo, my_hi).
        lo = max(off, my_lo)
        hi = min(off + nb, my_hi)
        if lo < hi:
            src_bytes = full.reshape(-1).view(torch.uint8)
            flat_shard.narrow(0, lo - my_lo, hi - lo).copy_(
                src_bytes.narrow(0, lo - off, hi - lo)
            )
        # Drop the temporary full tensor before processing the next weight
        # so peak HBM stays at one weight + flat_shard rather than sum.
        if not sharded.is_passthrough:
            del full

    return flat_shard


class FlatAGScheduler:
    """One CE all_gather of byte chunk -> flat T; per-weight tensors are
    narrow+view onto the AG output. No scatter, no per-weight AG.

    Constraints
        - All weights must shard along dim 0.
        - Same dp_size / dp_group for all weights.
        - CE backend (NCCL >= 2.28.3 with symmetric windows).
        - flat_shard, pool.flat_buf, pool.shard_buf must be registered as
          symmetric windows on the dp comm BEFORE first launch.
    """

    def __init__(
        self,
        side_stream: torch.cuda.Stream,
        weights: Sequence[ShardedWeight],
        dp_size: int,
        dp_group: Optional[dist.ProcessGroup],
        layout: FlatLayout,
        role_names: Optional[Sequence[str]] = None,
    ):
        assert len(weights) >= 1, "FlatAGScheduler needs at least one weight"
        for w in weights:
            assert w.shard_dim == 0, (
                f"FlatAGScheduler requires shard_dim=0 (got {w.shard_dim} on {w.name})"
            )
            assert w.dp_size == dp_size, "dp_size mismatch among flat weights"
        self.side_stream = side_stream
        self.weights: List[ShardedWeight] = list(weights)
        self.dp_size = dp_size
        self.dp_group = dp_group
        self.layout = layout
        if role_names is None:
            role_names = [w.name for w in weights]
        assert len(role_names) == len(weights), "role_names mismatch"
        self.role_names: List[str] = list(role_names)
        for w, role in zip(self.weights, self.role_names):
            w.name_for_pool = role
        self._shared_event = self.weights[0].ready_event
        self._waited_launch_id = 0
        self._launch_id = 0
        self._finalized = False

    def finalize(
        self,
        pool: Optional[FlatAGStagingPool],
        seq_idx: int,
        flat_shard: torch.Tensor,
    ) -> None:
        """Bind each weight's `full_buf` to the pool slot's typed view and
        cache the per-layer persistent flat_shard buffer.
        """
        if self._finalized:
            return
        self._pool = pool
        self._seq_idx = seq_idx
        self._flat_shard = flat_shard
        if self.dp_size == 1:
            for w in self.weights:
                w.is_passthrough = True
            self._finalized = True
            return
        assert pool is not None, "FlatAGScheduler.finalize requires pool when dp_size > 1"
        views = pool.borrow_views(seq_idx)
        for w in self.weights:
            w.full_buf = views[w.name_for_pool]
        # Single AG kernel finishes all weights simultaneously, so one
        # scheduler-level ready_event is enough for the whole concat buffer.
        for w in self.weights[1:]:
            w.ready_event = self._shared_event
        logger.info(
            "[MoeSharding] FlatAGScheduler.finalize dp_size=%d n_weights=%d "
            "|T|=%d chunk=%d",
            self.dp_size, len(self.weights),
            self.layout.total_bytes, self.layout.chunk_bytes,
        )
        _assert_ce_registered_pair(
            self._flat_shard, pool.borrow_flat(seq_idx), self.dp_group, "flat"
        )
        self._finalized = True

    def wait_ready_once(self) -> None:
        """Wait for the all-gather to complete, at most once per launch."""
        if self._waited_launch_id == self._launch_id:
            return
        torch.cuda.current_stream().wait_event(self._shared_event)
        self._waited_launch_id = self._launch_id

    def launch(self, gate_event: Optional[torch.cuda.Event] = None) -> None:
        """Launch the flat all-gather on the side stream.

        AG reads directly from the persistent per-layer ``flat_shard``
        (already a registered CE symmetric window) — no per-forward
        staging copy into a pool shard buffer.
        """
        if not self._finalized:
            raise RuntimeError("FlatAGScheduler.launch called before finalize()")
        self._launch_id += 1
        if self.dp_size == 1:
            self._shared_event.record()
            return
        cur = torch.cuda.current_stream()
        self.side_stream.wait_stream(cur)
        if gate_event is not None:
            self.side_stream.wait_event(gate_event)
        flat_buf = self._pool.borrow_flat(self._seq_idx)
        with torch.cuda.stream(self.side_stream):
            dist.all_gather_into_tensor(
                output_tensor=flat_buf,
                input_tensor=self._flat_shard,
                group=self.dp_group,
            )
            self._shared_event.record(self.side_stream)


class MoeShardingForwardPlanner:
    """Depth=1 lookahead AG dispatcher for MoE-sharded layers.

    The Kimi K3 model caches one planner for its local pipeline-stage range.

    Instances are cheap to reuse: construct once per (start, end) layer
    range, cache on the model, then call ``reset()`` at the top of every
    forward instead of rebuilding the layer list and planner.

    Usage::

        planner = MoeShardingForwardPlanner(as_layers)  # once, then cached
        planner.reset()                               # every forward
        planner.pre_dispatch(start_layer)
        for i in range(start_layer, end_layer):
            planner.maybe_dispatch(i + 1)
            hidden_states = layer(...)
            planner.update_done_event(i, layer)
    """

    def __init__(self, as_layers: Sequence[Tuple[int, object]]):
        self._sharded_layers: List[Tuple[int, object]] = list(as_layers)
        self._next = 0
        self._prev_done_event: Optional[torch.cuda.Event] = None

    def reset(self) -> None:
        """Rewind dispatch state for a new forward pass.

        Lets one planner instance (cached per (start, end) layer range) be
        reused across forwards instead of being rebuilt each time.
        """
        self._next = 0
        self._prev_done_event = None

    def _dispatch(self, module, gate_event: Optional[torch.cuda.Event]) -> None:
        module._moe_shard_sched.launch(gate_event=gate_event)

    def pre_dispatch(self, start_layer: int) -> None:
        """Pre-launch AG for MoE-sharded layers at *start_layer*."""
        while (
            self._next < len(self._sharded_layers)
            and self._sharded_layers[self._next][0] == start_layer
        ):
            _, module = self._sharded_layers[self._next]
            self._dispatch(module, gate_event=None)
            # NOTE: do NOT stash this layer's compute_done_event as the next
            # gate -- it has not been recorded in this forward/graph yet.
            # Waiting on it would reference a record from a previous forward
            # (eager: gate silently no-ops) or from another cuda graph
            # (split-graph capture: illegal cross-capture event dependency).
            # Slot safety for the next AG is already guaranteed by
            # stream/graph ordering; update_done_event() rebuilds the gate
            # chain from this layer on.
            self._next += 1

    def maybe_dispatch(self, next_layer_idx: int) -> None:
        """Before running layer *next_layer_idx - 1*, dispatch AG for
        *next_layer_idx* if it is the next MoE-sharded layer."""
        if self._next >= len(self._sharded_layers):
            return
        nidx, module = self._sharded_layers[self._next]
        if nidx == next_layer_idx:
            self._dispatch(module, gate_event=self._prev_done_event)
            self._next += 1

    def update_done_event(self, layer_idx: int, layer) -> None:
        """After running *layer_idx*, stash its compute_done_event if it
        is a MoE-sharded layer."""
        module = getattr(layer, "mlp", None)
        if module is None or getattr(module, "_moe_shard_sched", None) is None:
            return
        ev = getattr(module, "_moe_shard_done_event", None)
        if ev is not None:
            ev.record(torch.cuda.current_stream())
            self._prev_done_event = ev


# --------------------------------------------------------------------------
# Module-level helpers for replicated linear weights
# --------------------------------------------------------------------------


def _layer_id(layer) -> int:
    return int(getattr(layer, "layer_idx", getattr(layer, "layer_id", -1)))


def resolve_weight_module(root, path: str):
    """Resolve a dotted attribute path to a module under root, or None."""
    module = root
    for part in path.split("."):
        module = getattr(module, part, None)
        if module is None:
            return None
    return module


def setup_layer_moe_sharding_weight(
    layer,
    target_attr: str,
    side_stream: Optional[torch.cuda.Stream] = None,
):
    """Install a ShardedWeight for one linear submodule.

    Behavior:
        - Returns `sharded` or None if MoE sharding is disabled or the layer /
          target module missing.
        - Allocates `<target>_shard` (1/dp_size of out_features).
        - Defers shard population & round-trip validation to the flat-AG
          finalize (weights uninitialised at __init__).
    """
    cfg = get_moe_sharding_config()
    if not cfg.enabled:
        return None
    layer_id = _layer_id(layer)
    ids = _parse_layer_spec(cfg.layer_idx_spec)
    if ids is not None and layer_id not in set(ids):
        return None

    shard_group, shard_size, shard_rank, shard_kind = get_moe_sharding_group_info()

    target_mod = resolve_weight_module(layer, target_attr)
    if target_mod is None or not hasattr(target_mod, "weight"):
        rank0 = (not dist.is_initialized()) or (dist.get_rank() == 0)
        if rank0:
            logger.warning(
                f"[K3WeightShard layer={layer_id}] target={target_attr!r} not found; "
                f"MoE sharding disabled for this layer"
            )
        return None

    weight = target_mod.weight.data  # nn.Linear weight: (out_per_tp, in)
    quant_method = getattr(target_mod, "quant_method", None)
    if (
        weight.ndim != 2
        or not weight.is_contiguous()
        or weight.dtype not in (torch.bfloat16, torch.float16)
        or quant_method is None
        or type(quant_method).__name__ != "UnquantizedLinearMethod"
    ):
        raise RuntimeError(
            f"K3 weight sharding only supports contiguous, unquantized bf16/fp16 "
            f"linear weights; {target_attr} has shape={tuple(weight.shape)}, "
            f"dtype={weight.dtype}, contiguous={weight.is_contiguous()}, "
            f"quant_method={type(quant_method).__name__}"
        )
    out_per_tp, in_dim = weight.shape
    rank0 = (not dist.is_initialized()) or (dist.get_rank() == 0)
    module_tp_size = getattr(target_mod, "tp_size", None)
    module_tp_rank = getattr(target_mod, "tp_rank", None)
    input_partition = getattr(target_mod, "input_size_per_partition", None)
    reduce_results = getattr(target_mod, "reduce_results", None)
    input_is_parallel = getattr(target_mod, "input_is_parallel", None)
    if rank0:
        logger.debug(
            f"[K3WeightShard layer={layer_id}] {target_attr} weight shape = "
            f"{tuple(weight.shape)} dtype={weight.dtype} device={weight.device}"
        )
        logger.info(
            f"[K3WeightShard layer={layer_id}] {target_attr} "
            f"module_tp_size={module_tp_size} module_tp_rank={module_tp_rank} "
            f"input_partition={input_partition} input_is_parallel={input_is_parallel} "
            f"reduce_results={reduce_results}"
        )
        logger.debug(
            f"[K3WeightShard layer={layer_id}] shard_size={shard_size} "
            f"shard_rank={shard_rank} shard_kind={shard_kind} "
            f"-> shard out_per_tp by shard_size"
        )
        if hasattr(target_mod, "weight_scale_inv"):
            ws = target_mod.weight_scale_inv
            logger.debug(
                f"[K3WeightShard layer={layer_id}] {target_attr} "
                f"weight_scale_inv shape = {tuple(ws.shape)}"
            )
    if shard_size == 1:
        if rank0:
            logger.info(
                f"[K3WeightShard layer={layer_id}] shard_size=1, "
                f"passthrough mode (no AG)"
            )
        sharded = ShardedWeight.allocate(
            name=f"layer{layer_id}.{target_attr}",
            full_shape=tuple(weight.shape),
            dtype=weight.dtype,
            device=weight.device,
            dp_size=1,
            dp_group=None,
            shard_dim=0,
        )
        sharded.full_buf.copy_(weight)
        return sharded

    if out_per_tp % shard_size != 0:
        logger.warning(
            "[K3WeightShard layer=%d] %s rows=%d are not divisible by "
            "shard_size=%d; weight sharding is disabled for this layer",
            layer_id,
            target_attr,
            out_per_tp,
            shard_size,
        )
        return None

    shard_rows = out_per_tp // shard_size

    # CE AllGather is the only backend; every freshly created buffer is
    # ncclMemAlloc-backed for symmetric-window registration.
    use_ce = True

    # Requant-then-shard: leave target_mod.weight as the FULL weight and
    # do NOT wrap the loader. Each rank loads the full checkpoint weight +
    # full scale, and process_weights_after_loading requants the FULL
    # weight -> the ue8m0 weight_scale_inv is derived from the full N and
    # stays replicated (never sharded, never AG'd). We allocate a fresh
    # per-rank shard buffer here; finalize copies this rank's slice of the
    # requanted full weight into it, then the flat AG reconstructs the
    # full weight exactly as in the load-time path.
    sharded = ShardedWeight.allocate(
        name=f"layer{layer_id}.{target_attr}",
        full_shape=tuple(weight.shape),
        dtype=weight.dtype,
        device=weight.device,
        dp_size=shard_size,
        dp_group=shard_group,
        shard_dim=0,
        defer_full_buf=True,
        use_ce=use_ce,
    )
    if rank0:
        logger.info(
            f"[K3WeightShard layer={layer_id}] shard after load: "
            f"{target_attr}.weight kept FULL ({out_per_tp}, {in_dim}) "
            f"dtype={weight.dtype}; per-rank shard buffer "
            f"({shard_rows}, {in_dim}) allocated (use_ce={use_ce}); "
            f"shard_rank={shard_rank} shard_kind={shard_kind}; "
            f"weight_scale_inv NOT sharded"
        )

    return sharded


def setup_layer_weight_shard_flat(
    layer,
    targets: Sequence[Tuple[str, str]],
    side_stream: Optional[torch.cuda.Stream] = None,
):
    """Set up one flat AllGather for a set of replicated linear weights."""
    if not get_moe_sharding_config().enabled:
        return None
    _group, shard_size, _rank, _kind = get_moe_sharding_group_info()
    if shard_size <= 1:
        return None

    named_actives = []
    weight_targets = []
    try:
        for index, (role, target_path) in enumerate(targets):
            sharded = setup_layer_moe_sharding_weight(
                layer, target_path, side_stream
            )
            if sharded is None:
                continue
            sharded_attr = f"_moe_shard_weight_{index}"
            setattr(layer, sharded_attr, sharded)
            named_actives.append((role, sharded))
            weight_targets.append((role, sharded_attr, target_path))
    except Exception:
        for _, sharded in reversed(named_actives):
            ce_free_tensor(sharded.shard)
        for _, sharded_attr, _ in weight_targets:
            delattr(layer, sharded_attr)
        raise

    layer._moe_shard_targets = weight_targets
    if not named_actives:
        return None

    active_roles = [name for name, _ in named_actives]
    actives = [w for _, w in named_actives]
    dp_size = actives[0].dp_size
    dp_group = actives[0].dp_group

    # Build a layout from the per-rank shard shape × dp_size on shard_dim.
    # NOTE: full_buf may be a deferred 0-element placeholder at setup time
    # (load-time path), so we cannot read full_buf.shape here — derive it
    # from the shard shape instead. This works for both load-time and
    # legacy paths.
    specs = []
    for w, role in zip(actives, active_roles):
        if w.is_passthrough:
            full_shape = tuple(w.shard.shape)
        else:
            sd = (
                w.shard_dim
                if w.shard_dim >= 0
                else w.shard.ndim + w.shard_dim
            )
            full_shape = list(w.shard.shape)
            full_shape[sd] = full_shape[sd] * w.dp_size
            full_shape = tuple(full_shape)
        specs.append((role, full_shape, w.shard.dtype))
    layout = build_flat_layout(specs, dp_size if dp_size >= 1 else 1)

    flat_sched = FlatAGScheduler(
        side_stream=side_stream or torch.cuda.current_stream(),
        weights=actives,
        dp_size=dp_size,
        dp_group=dp_group,
        layout=layout,
        role_names=active_roles,
    )
    layer._moe_shard_sched = flat_sched
    layer._moe_shard_done_event = torch.cuda.Event()
    return flat_sched


def prepare_layer_weight_shard_forward(layer) -> None:
    """Wait for one layer's AllGather and bind reconstructed full weights."""
    sched = getattr(layer, "_moe_shard_sched", None)
    if sched is None:
        return
    if not isinstance(sched, FlatAGScheduler) or not sched._finalized:
        raise RuntimeError(
            f"K3 weight sharding layer {_layer_id(layer)} was not finalized"
        )
    sched.wait_ready_once()
    for _role, sharded_attr, target_path in layer._moe_shard_targets:
        sharded = getattr(layer, sharded_attr)
        target_mod = resolve_weight_module(layer, target_path)
        if target_mod is None:
            raise RuntimeError(
                f"K3 weight sharding target {target_path!r} disappeared"
            )
        target_mod.weight.data = sharded.full_buf


def finalize_layer_weight_shard_flat(
    layer,
    seq_idx: int,
    pool: Optional["FlatAGStagingPool"],
) -> None:
    """Finalize one Kimi K3 MoE module's flat weight shard.

    Steps (off forward path):
      1. Re-bind each ShardedWeight.shard to target_mod.weight.data.
      2. One-time redistribute: per-weight 1/dp_size shards -> persistent
         CE-allocated `flat_shard` (1/dp_size of T) for this layer.
      3. Register `flat_shard` as a symmetric window on the dp comm.
      4. flat_sched.finalize(pool, seq_idx, flat_shard) -> binds full_buf
         narrow+views and shared ready_event.
      5. Free per-weight shard storage on target_mod.weight.data
         (replaced with the same view onto the AG output for forward),
         since the source of truth for forward reads is now full_buf.

    `weight_scale_inv` is left replicated on every rank, same as concat.
    """
    sched = getattr(layer, "_moe_shard_sched", None)
    if sched is None:
        return
    if not isinstance(sched, FlatAGScheduler):
        return

    if sched.dp_size > 1 and pool is None:
        raise RuntimeError(
            "finalize_layer_weight_shard_flat requires a pool when dp_size > 1"
        )

    rank0 = (not dist.is_initialized()) or (dist.get_rank() == 0)

    slot_target = [
        (sharded_attr, target_path)
        for _role, sharded_attr, target_path in layer._moe_shard_targets
    ]
    # 1. Re-bind shard storage.
    #    target_mod.weight is the FULL requanted weight -> copy this rank's
    #    slice into the freshly-allocated shard buffer, then free the full
    #    weight (rebind target_mod.weight to the shard view so no full-sized
    #    copy lingers; forward rebinds to full_buf anyway).
    _, _, shard_rank, shard_kind = get_moe_sharding_group_info()
    for sharded_attr, target_attr in slot_target:
        sharded = getattr(layer, sharded_attr, None)
        target_mod = resolve_weight_module(layer, target_attr)
        if sharded is None or target_mod is None:
            continue
        if sharded.is_passthrough:
            sharded.full_buf.copy_(target_mod.weight.data)
            continue
        full_w = target_mod.weight.data
        shard_rows = sharded.shard.shape[0]
        src = full_w.narrow(0, shard_rank * shard_rows, shard_rows)
        sharded.shard.copy_(src)
        # Drop the full weight: forward will rebind weight.data to the
        # AG'd full_buf, so target_mod.weight only needs to hold the shard.
        target_mod.weight.data = sharded.shard
        if rank0:
            logger.info(
                f"[K3WeightShard finalize layer={_layer_id(layer)} target={target_attr}] "
                f"requant-then-shard: sliced full {tuple(full_w.shape)} -> "
                f"shard {tuple(sharded.shard.shape)} "
                f"(shard_rank={shard_rank} shard_kind={shard_kind}); "
                f"full weight released"
            )
        del full_w


    # 2. Redistribute (no-op when dp_size==1).
    if sched.dp_size > 1:
        device = sched.weights[0].shard.device
        flat_shard = _redistribute_to_flat_shard(
            layer,
            sched.layout,
            sched.dp_group,
            sched.dp_size,
            shard_rank,
            device,
        )
        # The one-shot NCCL gathers and copies above are enqueued on the
        # current stream. Their CE-backed source shards must remain alive until
        # every read has completed.
        torch.cuda.current_stream().synchronize()
        # 3. Register the flat shard as a CE symmetric window on the dp comm.
        ce_register_window(flat_shard, sched.dp_group)
        # Stash on layer so caller can free old per-weight shards if
        # desired and so registration survives.
        layer._moe_shard_flat = flat_shard

        # 2b. Release the per-weight shards. They were consumed exactly
        #     once by the redistribute above; forward reads the AG output
        #     (full_buf) and the AG input is the persistent flat_shard, so
        #     these 1/dp_size per-weight buffers are now dead. They are
        #     CE (ncclMemAlloc) buffers tracked by the resource manager for the
        #     process lifetime, so a plain `del` would NOT reclaim them and
        #     the 1/dp_size weight would stay permanently duplicated (the
        #     flat_shard already holds this rank's slice), shrinking the KV
        #     pool. `ce_free_tensor` deregisters windows + ncclMemFree's the
        #     buffer so the memory is returned. The empty placeholder keeps
        #     weight.data a valid tensor until forward rebinds it to full_buf.
        for sharded_attr, target_attr in slot_target:
            sharded = getattr(layer, sharded_attr, None)
            target_mod = resolve_weight_module(layer, target_attr)
            if sharded is None or sharded.is_passthrough:
                continue
            old_shard = sharded.shard
            empty = torch.empty(0, dtype=old_shard.dtype, device=device)
            # `old_shard` is the only remaining handle to the CE storage;
            # free it (windows + ncclMemFree) then drop the handle.
            freed = ce_free_tensor(old_shard)
            if freed:
                if target_mod is not None:
                    target_mod.weight.data = empty
                sharded.shard = empty
            free_status = torch.tensor(
                [int(freed)], dtype=torch.int32, device=device
            )
            dist.all_reduce(
                free_status, op=dist.ReduceOp.MIN, group=sched.dp_group
            )
            if free_status.item() != 1:
                raise RuntimeError(
                    f"failed to free temporary K3 weight shard "
                    f"for layer {_layer_id(layer)} target {target_attr}"
                )
            del old_shard
    else:
        flat_shard = torch.empty(
            0, dtype=torch.uint8, device=sched.weights[0].shard.device
        )

    # 4. Bind scheduler buffers.
    sched.finalize(pool=pool, seq_idx=seq_idx, flat_shard=flat_shard)
    layer._moe_shard_seq_idx = seq_idx

    # 5. weight_scale_inv is left FULL and replicated on every rank: the
    #    framework requanted the full weight at load time, so the ue8m0 scale
    #    is already correct for the AG-reconstructed full weight. No AG needed.
    if rank0:
        logger.info(
            f"[K3WeightShard finalize layer={_layer_id(layer)}] complete"
        )


def finalize_model_weight_shard(
    as_layers: Sequence,
    mode: str = "lazy",
) -> Optional[FlatAGStagingPool]:
    """Build the shared staging pool and finalize every Kimi K3 MoE shard.

    This is the top-level entry point called by the model's
    ``force_weight_shard_finalize`` / ``_run_weight_shard_finalize`` hook.
    The caller is responsible for:

      - Idempotency / cuda-graph-capture guards.
      - Collecting and sorting ``as_layers`` by ``layer_id``.
      - Storing the returned pool for the forward path.
      - Calling ``torch.cuda.empty_cache()`` afterwards.

    Args:
        as_layers: Kimi K3 MoE modules with ``_moe_shard_sched``, sorted by
            ``layer_idx``.
        mode: ``"force"`` (pre-pool, driven by ModelRunner) or ``"lazy"``
            (forward fallback). Used only for logging.

    Returns the shared :class:`FlatAGStagingPool`, or ``None`` when
    ``as_layers`` is empty or all schedulers are passthrough.
    """
    if not as_layers:
        return None

    sample = as_layers[0]
    sample_sched = sample._moe_shard_sched
    dp_size = sample_sched.dp_size
    dp_group = sample_sched.dp_group

    if dp_size == 1:
        for seq_idx, sa in enumerate(as_layers):
            finalize_layer_weight_shard_flat(sa, seq_idx, pool=None)
        logger.info(
            f"[MoeSharding] {mode} finalize (passthrough): "
            f"finalized {len(as_layers)} layers"
        )
        return None
    layout = sample_sched.layout
    for layer in as_layers[1:]:
        if layer._moe_shard_sched.layout != layout:
            raise RuntimeError(
                "K3 weight sharding requires identical target layouts across layers"
            )

    # Force NCCL comm init (lazy Symmetric VA setup) before ce_alloc_tensor.
    device = sample_sched.weights[0].shard.device
    _dummy = torch.zeros(1, dtype=torch.float32, device=device)
    dist.all_reduce(_dummy, group=dp_group)
    torch.cuda.synchronize()

    pool = FlatAGStagingPool(
        layout=layout,
        device=device,
        dp_size=dp_size,
        n_buffers=1 if len(as_layers) <= 1 else 2,
    )

    # Register pool flat_bufs as symmetric windows on the dp comm.
    ce_register_windows_grouped(list(pool.flat_bufs), dp_group)

    for seq_idx, sa in enumerate(as_layers):
        finalize_layer_weight_shard_flat(sa, seq_idx, pool=pool)

    logger.info(
        f"[K3WeightShard] {mode} finalize: "
        f"finalized {len(as_layers)} layers, "
        f"shared flat pool n_buffers={pool.n_buffers} |T|={layout.total_bytes}"
    )
    return pool
