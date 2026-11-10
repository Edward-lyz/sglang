"""[fusion-opt] Prefill-CP SWA KV store, local-first (SGLANG_FUSE_CP_KV_STORE_LOCAL_FIRST).

The stock CP path all-gathers the raw bf16 kv [T, 512] (1024 B/row), reorders it into
global token order, then every rank runs RMSNorm + RoPE + fp8 quant on all T rows into
its SWA cache. All three steps are row-wise, so each rank instead
  1. norm + RoPE + quantizes only its local rows, with the same fused kernel, into a
     staging buffer laid out exactly like the paged cache (528 B/row for V4.1 fp8),
  2. all-gathers the staging pages,
  3. copies every token's 512 B data row + 16 B scale row into its cache slot.
Every stored byte is produced by the same kernel on the same inputs (row, position),
so the cache contents are bitwise identical to the stock path. Paged layouts V4 (576+8 B)
and V4.1 fp8 (512+16 B) are supported.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.elementwise import fused_k_norm_rope_flashmla
from sglang.srt.environ import envs

_CE_GATHER_GROUPS: set = set()


def _ce_all_gather(x: torch.Tensor) -> torch.Tensor:
    """[fusion-opt] SGLANG_FUSE_CP_SWA_GATHER_CE: CP all-gather of ``x`` (rank-major)
    through torch symmetric memory's low-contention all-gather -- peer pulls on the
    copy engines instead of an NCCL kernel, so the gather takes no SMs from compute
    it overlaps (DeepInfer gathers its CP KV through userbuffer copies likewise)."""
    import torch.distributed._symmetric_memory as symm_mem

    from sglang.srt.distributed.parallel_state import get_attn_cp_group

    group_name = get_attn_cp_group().device_group.group_name
    if group_name not in _CE_GATHER_GROUPS:
        symm_mem.enable_symm_mem_for_group(group_name)
        _CE_GATHER_GROUPS.add(group_name)
    out = torch.ops.symm_mem._low_contention_all_gather(x.contiguous(), group_name)
    return torch.ops._c10d_functional.wait_tensor(out)


_CU_STREAM_MEM_OP_WAIT_VALUE_32 = 1
_CU_STREAM_MEM_OP_WRITE_VALUE_32 = 2
_CU_STREAM_WAIT_VALUE_GEQ = 0x0
_libcuda = None
_MemOp = None


def _cu():
    """libcuda plus the CUstreamBatchMemOpParams layout (48 B union; wait/write
    value: op u32 @0, address u64 @8, value u64 @16, flags u32 @24, alias u64 @32)."""
    global _libcuda, _MemOp
    if _libcuda is None:
        import ctypes

        class MemOp(ctypes.Structure):
            """CUstreamBatchMemOpParams (wait / write value 32 members)."""

            _fields_ = [
                ("operation", ctypes.c_uint32),
                ("_pad0", ctypes.c_uint32),
                ("address", ctypes.c_uint64),
                ("value", ctypes.c_uint64),
                ("flags", ctypes.c_uint32),
                ("_pad1", ctypes.c_uint32),
                ("alias", ctypes.c_uint64),
                ("_pad2", ctypes.c_uint64),
            ]

        assert ctypes.sizeof(MemOp) == 48
        _MemOp = MemOp
        _libcuda = ctypes.CDLL("libcuda.so.1")
    return _libcuda


class _MemOpBatch:
    """A fixed cuStreamBatchMemOp batch (stream memory ops run in the stream front
    end and hold no SM); only the compared / written value changes per call."""

    def __init__(self, op: int, addresses):
        """Build one ``op`` (wait / write value 32) per device address."""
        import ctypes

        lib = _cu()
        self.arr = (_MemOp * len(addresses))()
        for k, addr in enumerate(addresses):
            self.arr[k].operation = op
            self.arr[k].address = addr
            self.arr[k].flags = _CU_STREAM_WAIT_VALUE_GEQ
        self.n = len(addresses)
        self.fn = lib.cuStreamBatchMemOp_v2
        self.fn.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint]
        self.ptr = ctypes.addressof(self.arr)

    def __call__(self, stream_handle: int, value: int) -> None:
        """Enqueue the batch on ``stream_handle`` with every value set to ``value``."""
        for k in range(self.n):
            self.arr[k].value = value
        err = self.fn(stream_handle, self.n, self.ptr, 0)
        if err != 0:
            raise RuntimeError(f"cuStreamBatchMemOp failed: {err}")


class _PushGather:
    """[fusion-opt] SGLANG_FUSE_CP_SWA_GATHER_PUSH: CP all-gather that holds no SM.

    DeepInfer's CP gather (DeepEP BucketBuffer): the producer kernel writes its
    shard straight into a pre-registered symmetric buffer, copy engines move it,
    and readiness travels as stream memory ops -- so nothing of the gather ever
    occupies an SM next to the persistent 148-CTA GEMMs. Here:
      * buf[parity] holds the rank-major gather ``[cp, nbytes]``; rank r's
        producer writes buf[parity][r] in place (no staging copy),
      * the copy stream pushes that slice into every peer's buf[parity][r]
        (cuMemcpyAsync over NVLink), then writes the peer's ready[r] = epoch,
      * the consumer stream waits ready[*] >= epoch, reads the gather, then
        writes done[r] = epoch into every peer,
      * a push into a peer's buf[parity] first waits done >= epoch - 2 (the last
        epoch that used that parity) from that peer.
    Every rank calls it with the same sizes in the same order (one call per
    layer). Host cost per call: 2 event ops, 4 batched memop calls, cp-1 memcpys
    (the batches are built once; only their values change).
    """

    def __init__(self, group, capacity_bytes: int):
        """Rendezvous the gather buffer and flags on ``group`` (collective)."""
        import torch.distributed as dist
        import torch.distributed._symmetric_memory as symm_mem

        symm_mem.enable_symm_mem_for_group(group.group_name)
        self.rank, self.world = group.rank(), group.size()
        self.capacity = capacity_bytes
        self.buf = symm_mem.empty(
            2 * self.world * capacity_bytes, dtype=torch.uint8, device="cuda"
        )
        handle = symm_mem.rendezvous(self.buf, group.group_name)
        # ready[0:world] then done[world:2*world], one u32 per source rank
        self.flags = symm_mem.empty(2 * self.world, dtype=torch.int32, device="cuda")
        self.flags.zero_()
        flag_handle = symm_mem.rendezvous(self.flags, group.group_name)
        assert handle.offset == 0 and flag_handle.offset == 0
        torch.cuda.synchronize()
        dist.barrier(group)
        self.buf_ptrs = list(handle.buffer_ptrs)
        self.flag_ptrs = list(flag_handle.buffer_ptrs)
        self._handles = (handle, flag_handle)  # keep the mappings alive
        self.peers = [(self.rank + k) % self.world for k in range(1, self.world)]
        mine, w = self.flag_ptrs[self.rank], self.world
        W32, WAIT32 = _CU_STREAM_MEM_OP_WRITE_VALUE_32, _CU_STREAM_MEM_OP_WAIT_VALUE_32
        self.wait_done = _MemOpBatch(WAIT32, [mine + 4 * (w + p) for p in self.peers])
        self.signal_ready = _MemOpBatch(
            W32, [self.flag_ptrs[p] + 4 * self.rank for p in self.peers]
        )
        self.wait_ready = _MemOpBatch(WAIT32, [mine + 4 * p for p in self.peers])
        self.signal_done = _MemOpBatch(
            W32, [self.flag_ptrs[p] + 4 * (w + self.rank) for p in self.peers]
        )
        import ctypes

        lib = _cu()
        self._memcpy = lib.cuMemcpyAsync
        self._memcpy.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_void_p]
        # raw driver events / calls: torch's Event.record and current_stream() cost
        # ~5 us each, and this runs once per layer on the host-bound tail
        self._ev_record = lib.cuEventRecord
        self._ev_record.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._wait_event = lib.cuStreamWaitEvent
        self._wait_event.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
        self.produced, self.pushed = [], []
        for _ in range(2):
            for lst in (self.produced, self.pushed):
                ev = ctypes.c_void_p()
                if lib.cuEventCreate(ctypes.byref(ev), ctypes.c_uint(0x2)) != 0:  # DISABLE_TIMING
                    raise RuntimeError("cuEventCreate failed")
                lst.append(ev.value)
        self.copy_stream = torch.cuda.Stream()
        self.copy_handle = self.copy_stream.cuda_stream
        self.device_index = torch.cuda.current_device()
        self.pushed_used = [False, False]
        self.views = {}
        self.epoch = 0

    def _main(self) -> int:
        """Raw handle of the current stream (cheaper than torch.cuda.current_stream())."""
        return torch._C._cuda_getCurrentRawStream(self.device_index)

    def _views(self, parity: int, nbytes: int):
        """(this rank's slice, rank-major gather) of buf[parity] for nbytes/rank."""
        key = (parity, nbytes)
        v = self.views.get(key)
        if v is None:
            base = parity * self.world * self.capacity
            v = self.views[key] = (
                self.buf[base + self.rank * nbytes : base + (self.rank + 1) * nbytes],
                self.buf[base : base + self.world * nbytes],
            )
        return v

    def _base(self, epoch: int) -> int:
        """Byte offset of this epoch's (parity's) gather region."""
        return (epoch & 1) * self.world * self.capacity

    def begin(self, nbytes: int):
        """Next epoch and this rank's slice to produce into (current stream)."""
        epoch = self.epoch + 1
        parity = epoch & 1
        if self.pushed_used[parity]:  # our pushes still reading this slice
            self._wait_event(self._main(), self.pushed[parity], 0)
        return epoch, self._views(parity, nbytes)[0]

    def push(self, epoch: int, nbytes: int) -> None:
        """After the producer wrote its slice on the current stream."""
        self.epoch = epoch
        parity = epoch & 1
        h = self.copy_handle
        self._ev_record(self.produced[parity], self._main())
        self._wait_event(h, self.produced[parity], 0)
        if epoch > 2:
            self.wait_done(h, epoch - 2)
        off = self._base(epoch) + self.rank * nbytes
        src = self.buf_ptrs[self.rank] + off
        for p in self.peers:
            if self._memcpy(self.buf_ptrs[p] + off, src, nbytes, h) != 0:
                raise RuntimeError("cuMemcpyAsync failed")
        self.signal_ready(h, epoch)
        self._ev_record(self.pushed[parity], h)
        self.pushed_used[parity] = True

    def wait(self, epoch: int, nbytes: int) -> torch.Tensor:
        """Current stream waits for every peer's slice; returns the gather."""
        self.wait_ready(self._main(), epoch)
        return self._views(epoch & 1, nbytes)[1]

    def release(self, epoch: int) -> None:
        """Current stream finished reading this epoch's gather."""
        self.signal_done(self._main(), epoch)


_PUSH_GATHER = {}


def _push_gather(capacity_bytes: int) -> "_PushGather":
    """The attention-CP group's _PushGather, created on first use (collective)."""
    from sglang.srt.distributed.parallel_state import get_attn_cp_group

    group = get_attn_cp_group().device_group
    pg = _PUSH_GATHER.get(group.group_name)
    if pg is None:
        pg = _PUSH_GATHER[group.group_name] = _PushGather(group, capacity_bytes)
    return pg


@triton.jit
def _scatter_kv_rows_kernel(
    src_ptr,  # int32 view of the gathered staging pages [cp * npages, PB / 4]
    dst_ptr,  # int32 view of the SWA cache [num_pages, PB / 4]
    swa_loc_ptr,  # [T] dst slot per global token (< 0: no write)
    src_row_ptr,  # [T] int64 flat staging row (rank * P + local row) per global token
    total,  # T
    rank_len,  # P: rows per rank in the staging layout
    npages_per_rank,
    page_words,  # PB / 4
    cp_size,
    INTERLEAVE: tl.constexpr,  # src row = (t % cp) * P + t // cp, src_row_ptr unused
    PAGE_SIZE: tl.constexpr,
    DATA_WORDS: tl.constexpr,  # data bytes per token / 4 (V4: 144, V4.1: 128)
    SCALE_WORDS: tl.constexpr,  # scale bytes per token / 4 (V4: 2, V4.1: 4)
    DATA_P2: tl.constexpr,
    SCALE_P2: tl.constexpr,
    BLOCK_T: tl.constexpr,  # tokens per program
):
    """Scatter all-gathered staging rows (data + scale words) into SWA cache slots."""
    t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    tm = t < total
    dst = tl.load(swa_loc_ptr + t, mask=tm, other=-1).to(tl.int64)
    ok = tm & (dst >= 0)
    if INTERLEAVE:
        m = (t % cp_size).to(tl.int64) * rank_len + t // cp_size
    else:
        m = tl.load(src_row_ptr + t, mask=tm, other=0).to(tl.int64)
    r = m // rank_len
    j = m % rank_len
    s_base = (r * npages_per_rank + j // PAGE_SIZE) * page_words
    s_slot = j % PAGE_SIZE
    dst_c = tl.where(ok, dst, 0)
    d_base = (dst_c // PAGE_SIZE) * page_words
    d_slot = dst_c % PAGE_SIZE
    w = tl.arange(0, DATA_P2)
    wm = ok[:, None] & (w[None, :] < DATA_WORDS)
    v = tl.load(src_ptr + (s_base + s_slot * DATA_WORDS)[:, None] + w[None, :], mask=wm)
    tl.store(dst_ptr + (d_base + d_slot * DATA_WORDS)[:, None] + w[None, :], v, mask=wm)
    sw = tl.arange(0, SCALE_P2)
    sm = ok[:, None] & (sw[None, :] < SCALE_WORDS)
    scale_base = PAGE_SIZE * DATA_WORDS
    sv = tl.load(
        src_ptr + (s_base + scale_base + s_slot * SCALE_WORDS)[:, None] + sw[None, :],
        mask=sm,
    )
    tl.store(
        dst_ptr + (d_base + scale_base + d_slot * SCALE_WORDS)[:, None] + sw[None, :],
        sv,
        mask=sm,
    )


_SCATTER_BLOCK_T = 8


_ARANGE_CACHE = {}


def _arange_i32(n: int, dev) -> torch.Tensor:
    """Return a cached int32 arange(n) on dev (read-only staging slots 0..n-1).

    Shared by every layer of a forward, so callers must not modify it. One
    buffer per device, grown on demand; callers on the SWA store side stream
    and the main stream share it, so each use records its stream, otherwise a
    buffer dropped on growth could be reused while the other stream still
    reads it.
    """
    t = _ARANGE_CACHE.get(dev)
    if t is None or t.numel() < n:
        t = _ARANGE_CACHE[dev] = torch.arange(
            max(n, 2 * (t.numel() if t is not None else 0)),
            dtype=torch.int32,
            device=dev,
        )
    t.record_stream(torch.cuda.current_stream(t.device))
    return t[:n]


def cp_store_swa_kv_local_first(
    *,
    kvcache: torch.Tensor,
    page_size: int,
    layout,
    kv_local: torch.Tensor,
    positions_local: torch.Tensor,
    swa_loc: torch.Tensor,
    forward_batch,
    kv_weight: torch.Tensor,
    eps: float,
    freqs_cis: torch.Tensor,
    cp_rank: int,
    cp_size: int,
    defer: bool = False,
):
    """Store prefill-CP SWA KV local-first: quantize local rows, all-gather pages.

    Each CP rank norm + RoPE + quantizes only its local tokens into staging pages,
    all-gathers the pages (rank-major) and scatters every global token into its
    SWA cache slot, instead of all-gathering the raw bf16 KV.

    ``defer`` with SGLANG_FUSE_CP_SWA_GATHER_PUSH: the staging pages are produced in
    place in the symmetric gather buffer and pushed by copy engines; returns a
    ``finish()`` that waits for the peers and scatters (call it on the same stream
    before the SWA cache is read). Otherwise stores synchronously, returns None.
    """
    from sglang.srt.layers.dp_attention import attn_cp_all_gather_into_tensor

    meta = forward_batch.attn_cp_metadata
    total = int(meta.total_seq_lens)
    logical = meta.per_rank_logical_token or meta.per_rank_actual_token
    rank_len = max(meta.per_rank_actual_token)
    if rank_len == 0 or total == 0:
        return
    local_len = logical[cp_rank]
    page_bytes = kvcache.shape[1]
    npages = (rank_len + page_size - 1) // page_size
    dev = kv_local.device

    push = epoch = None
    if defer and envs.SGLANG_FUSE_CP_SWA_GATHER_PUSH.get():
        from sglang.srt.runtime_context import get_schedule

        chunk = get_schedule().chunked_prefill_size
        cap_pages = ((chunk + cp_size - 1) // cp_size + page_size - 1) // page_size + 1
        capacity = cap_pages * page_bytes
        # same npages on every rank (padded rank_len), so the choice is rank-consistent
        if npages * page_bytes <= capacity:
            push = _push_gather(capacity)
            if npages * page_bytes > push.capacity:
                push = None

    # 1. local rows -> staging pages (same kernel as the cache store)
    if push is not None:
        epoch, mine = push.begin(npages * page_bytes)
        staging = mine.view(npages, page_bytes)
    else:
        staging = torch.empty((npages, page_bytes), dtype=torch.uint8, device=dev)
    stage_loc = _arange_i32(local_len, dev)
    fused_k_norm_rope_flashmla(
        kv=kv_local[:local_len],
        kv_weight=kv_weight,
        eps=eps,
        freqs_cis=freqs_cis,
        positions=positions_local[:local_len],
        out_loc=stage_loc,
        kvcache=staging,
        page_size=page_size,
        layout=layout,
    )
    # 2. all-gather the staging pages (rank-major)
    if push is not None:
        push.push(epoch, npages * page_bytes)

        def finish():
            """Wait for every peer's pages, scatter them into the SWA cache, release."""
            gathered = push.wait(epoch, npages * page_bytes).view(
                cp_size * npages, page_bytes
            )
            _scatter_gathered(gathered, kvcache, swa_loc, meta, total, rank_len,
                              npages, page_bytes, page_size, layout, cp_size)
            push.release(epoch)

        return finish
    if envs.SGLANG_FUSE_CP_SWA_GATHER_CE.get():
        gathered = _ce_all_gather(staging)
    else:
        gathered = torch.empty(
            (cp_size * npages, page_bytes), dtype=torch.uint8, device=dev
        )
        attn_cp_all_gather_into_tensor(gathered, staging)

    _scatter_gathered(gathered, kvcache, swa_loc, meta, total, rank_len,
                      npages, page_bytes, page_size, layout, cp_size)
    return None


def _scatter_gathered(gathered, kvcache, swa_loc, meta, total, rank_len,
                      npages, page_bytes, page_size, layout, cp_size) -> None:
    """Copy every global token's row of the rank-major gathered pages into its SWA slot."""
    # 3. global token t -> flat staging row; same map as the interleave gather
    # (interleave map computed in the kernel: no index tensors / launches per layer)
    interleave = meta.gather_index is None
    src_row = swa_loc  # placeholder pointer when INTERLEAVE
    if not interleave:
        src_row = meta.gather_index[:total].to(torch.int64)
    loc = swa_loc[:total]
    _scatter_kv_rows_kernel[(triton.cdiv(total, _SCATTER_BLOCK_T),)](
        gathered.view(torch.int32),
        kvcache.view(torch.int32),
        loc,
        src_row,
        total,
        rank_len,
        npages,
        page_bytes // 4,
        cp_size,
        INTERLEAVE=interleave,
        PAGE_SIZE=page_size,
        DATA_WORDS=layout.data_bytes // 4,
        SCALE_WORDS=layout.scale_bytes // 4,
        DATA_P2=triton.next_power_of_2(layout.data_bytes // 4),
        SCALE_P2=max(2, triton.next_power_of_2(layout.scale_bytes // 4)),
        BLOCK_T=_SCATTER_BLOCK_T,
        num_warps=4,
    )
