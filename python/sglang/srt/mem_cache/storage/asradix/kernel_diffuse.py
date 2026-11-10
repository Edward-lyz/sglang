"""NVLink read-diffuse transport: raw-IPC + SM scatter kernel.

Ported from the layer_overlap L2-pipe prototype (validated on 8xGB100). Rationale:
  - cross-process ``cudaMemcpyPeerAsync`` over IPC memory falls back to PCIe
    (~36 GB/s);
  - a CUDA kernel doing SM LD/ST directly on a RAW-IPC-opened peer pointer runs
    at NVLink speed (~750 GB/s) -- but ONLY via ``cudaIpcOpenMemHandle`` (torch's
    shared-tensor mapping is copy-engine-only and faults an SM kernel);
  - scattered (discrete) page order costs ~0 vs contiguous at >=8KB pages.

Transport-only: compiles the kernel, wraps raw IPC get/open, and copies page
regions peer->local. Orchestration (handle exchange, sharding, gating) lives in
``read_diffuse.py``.
"""

from __future__ import annotations

import ctypes
import functools

from sglang.srt.environ import envs

_LAZY = 1  # cudaIpcMemLazyEnablePeerAccess
_HANDLE_BYTES = 64


def _diffuse_blocks() -> int:
    # Small grid on purpose: diffuse volume is modest, and a large grid needlessly
    # occupies SMs and inflates forward's latency-bound collectives.
    return max(1, int(envs.SGLANG_AS_READ_DIFFUSE_BLOCKS.get()))


_CUDA = r"""
#include <cuda_runtime.h>
// contiguous region copy (dst/src already offset), nbytes multiple of 16
__global__ void _copy_region(char* dst, const char* src, long n4) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for (; i < n4; i += stride) ((float4*)dst)[i] = ((const float4*)src)[i];
}
void copy_region(long long dst, long long src, long nbytes, int blocks) {
    _copy_region<<<blocks, 256>>>((char*)dst, (const char*)src, nbytes / 16);
}
// PHYSICALLY DISCRETE scatter: page k lives at the SAME scattered byte offset
// idx[k]*page_bytes in BOTH src and dst windows (real paged-KV layout: identical
// slot map on every rank). Reads and writes are both page-granular, discrete.
__global__ void _scatter_pages(char* dst, const char* src, const int* idx,
                               int page_bytes, int npages) {
    int v4 = page_bytes / 16;
    for (int k = blockIdx.x; k < npages; k += gridDim.x) {
        long o = (long)idx[k] * page_bytes;
        const float4* s = (const float4*)(src + o);
        float4* d = (float4*)(dst + o);
        for (int i = threadIdx.x; i < v4; i += blockDim.x) d[i] = s[i];
    }
}
void scatter_pages(long long dst, long long src, long long idx,
                   int page_bytes, int npages, int blocks) {
    _scatter_pages<<<blocks, 256>>>((char*)dst, (const char*)src,
                                    (const int*)idx, page_bytes, npages);
}
"""
_CPP = (
    "void copy_region(long long dst, long long src, long nbytes, int blocks);\n"
    "void scatter_pages(long long dst, long long src, long long idx, "
    "int page_bytes, int npages, int blocks);"
)


class _Handle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * _HANDLE_BYTES)]


@functools.lru_cache(maxsize=1)
def _rt() -> ctypes.CDLL:
    rt = ctypes.CDLL("libcudart.so")
    rt.cudaSetDevice.argtypes = [ctypes.c_int]
    rt.cudaIpcGetMemHandle.argtypes = [ctypes.POINTER(_Handle), ctypes.c_void_p]
    rt.cudaIpcOpenMemHandle.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        _Handle,
        ctypes.c_uint,
    ]
    rt.cudaGetErrorString.restype = ctypes.c_char_p
    return rt


@functools.lru_cache(maxsize=1)
def _kernel():
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name="asradix_kernel_diffuse",
        cpp_sources=_CPP,
        cuda_sources=_CUDA,
        functions=["copy_region", "scatter_pages"],
        verbose=False,
    )
# PLACEHOLDER_APPEND


def ipc_get_handle(dev_ptr: int) -> bytes:
    """Raw 64B cudaIpc handle for a device allocation base pointer."""
    rt = _rt()
    h = _Handle()
    rc = rt.cudaIpcGetMemHandle(ctypes.byref(h), ctypes.c_void_p(dev_ptr))
    if rc != 0:
        raise RuntimeError(
            f"cudaIpcGetMemHandle rc={rc}: {rt.cudaGetErrorString(rc).decode()}"
        )
    return ctypes.string_at(ctypes.addressof(h), _HANDLE_BYTES)


def share_torch_window(t) -> tuple[bytes, int]:
    """Return (framed_handle_bytes, byte_offset) for a torch KV window tensor.

    torch frames the segment handle (>64B: a leading type byte + the 64B
    cudaIpcMemHandle); ship the whole thing, the opener strips it. Offset is the
    segment storage offset plus this tensor's own storage offset.
    """
    shared = t.untyped_storage()._share_cuda_()  # (dev, handle, size, off, ...)
    handle = bytes(shared[1])
    seg_off = int(shared[3])
    tensor_off = t.storage_offset() * t.element_size()
    return handle, seg_off + tensor_off


def ipc_open_handle(handle_bytes: bytes, device: int) -> int:
    """Raw-open a peer window handle while ``device`` is current (lazy peer
    access) -> a kernel-addressable, NVLink-reachable base pointer (as int).

    Accepts a raw 64B handle or torch's framed handle (scans the leading offset
    for the real 64B window).
    """
    rt = _rt()
    rt.cudaSetDevice(device)
    n = len(handle_bytes)
    # 64B raw -> offset 0; torch frames it (66B seen: 2B prefix) -> try 2,1,0,3.
    candidates = (0,) if n == _HANDLE_BYTES else (2, 1, 0, 3)
    last_rc = -1
    for base in candidates:
        if base + _HANDLE_BYTES > n:
            continue
        h = _Handle()
        ctypes.memmove(
            ctypes.addressof(h),
            handle_bytes[base : base + _HANDLE_BYTES],
            _HANDLE_BYTES,
        )
        ptr = ctypes.c_void_p()
        last_rc = rt.cudaIpcOpenMemHandle(ctypes.byref(ptr), h, _LAZY)
        if last_rc == 0:
            return int(ptr.value)
    raise RuntimeError(
        f"cudaIpcOpenMemHandle failed (rc={last_rc}: "
        f"{rt.cudaGetErrorString(last_rc).decode()}) for {n}B handle"
    )


def copy_region(dst_ptr: int, src_ptr: int, nbytes: int) -> None:
    """Contiguous peer->local copy on the current stream (nbytes % 16 == 0)."""
    if nbytes <= 0:
        return
    _kernel().copy_region(dst_ptr, src_ptr, nbytes, _diffuse_blocks())


def scatter_pages(
    dst_ptr: int, src_ptr: int, idx_ptr: int, page_bytes: int, npages: int
) -> None:
    """Discrete copy: page k at scattered offset idx[k]*page_bytes is copied
    src->dst at the SAME offset (real paged-KV layout). idx is an int32 device
    array of ``npages`` scattered page indices. Runs on the current stream.
    """
    if npages <= 0 or page_bytes <= 0:
        return
    _kernel().scatter_pages(
        dst_ptr, src_ptr, idx_ptr, page_bytes, npages, _diffuse_blocks()
    )

