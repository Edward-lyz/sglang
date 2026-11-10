"""Operator-level numeric gate for the _pre_alloc item-sync fix.

HybridReqToTokenPool.alloc used the 0-dim CUDA tensor req.mamba_pool_idx as a
tensor index to reset the ReplaySSM per-slot state, which forces an implicit
.item()/_local_scalar_dense D2H sync on every admitted request. The fix indexes
those resets by the host-side slot id from MambaSlotAllocator.alloc_with_host.

These CPU tests prove the fix is numerically a no-op: the host id addresses the
identical slot, and resetting the state buffers by the host int yields buffers
bit-identical to resetting by the device-scalar index.
"""
import torch
from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator


def test_host_id_matches_device_slot_direct():
    a = MambaSlotAllocator(size=16, device="cpu")
    slot, host = a.alloc_with_host(1)
    assert slot is not None and host is not None
    assert int(slot[0]) == int(host[0])


def test_host_id_matches_device_slot_group_path():
    a = MambaSlotAllocator(size=16, device="cpu")
    a.alloc_group_begin(4)
    for _ in range(4):
        slot, host = a.alloc_with_host(1)
        assert slot is not None and host is not None
        assert int(slot[0]) == int(host[0])
    a.alloc_group_end()


def test_host_index_reset_equals_device_index_reset():
    a = MambaSlotAllocator(size=16, device="cpu")
    slot, host = a.alloc_with_host(1)
    dev_idx = slot[0]      # old path: 0-dim tensor index (triggers D2H .item())
    host_idx = host[0]     # new path: python int index (no sync)
    for proto in (
        torch.arange(17, dtype=torch.int32),
        torch.ones(17, dtype=torch.int32),
        torch.full((17,), 7, dtype=torch.int32),
    ):
        b_dev = proto.clone()
        b_host = proto.clone()
        b_dev[dev_idx] = 0
        b_host[host_idx] = 0
        assert torch.equal(b_dev, b_host)
        assert int(b_host[host_idx]) == 0


def test_host_index_is_python_int_not_tensor():
    a = MambaSlotAllocator(size=16, device="cpu")
    _, host = a.alloc_with_host(1)
    assert isinstance(host[0], int)


def test_zero_fill_equals_scalar_assign():
    """.zero_() on the selected slot view is numerically identical to `= 0`,
    but avoids the synchronous pageable-H2D aten::copy_ that `buf[idx] = 0`
    emits (lift_fresh CPU scalar -> copy_ into device element)."""
    a = MambaSlotAllocator(size=16, device="cpu")
    _, host = a.alloc_with_host(1)
    idx = host[0]
    for proto in (
        torch.arange(17, dtype=torch.int32),
        torch.full((17,), 5, dtype=torch.int8),
        torch.ones(17, dtype=torch.int32),
    ):
        b_assign = proto.clone()
        b_zero = proto.clone()
        b_assign[idx] = 0
        b_zero[idx].zero_()
        assert torch.equal(b_assign, b_zero)
        assert int(b_zero[idx]) == 0


def test_host_built_index_tensor_equals_stack_cast():
    """Building the mamba slot-id tensor from host ints equals the device
    torch.stack(0-dim views).to(int32) result, but avoids the stream-blocking
    aten::to on the admission path."""
    a = MambaSlotAllocator(size=32, device="cpu")
    dev_ids = []
    host_ids = []
    for _ in range(5):
        slot, host = a.alloc_with_host(1)
        dev_ids.append(slot[0])
        host_ids.append(host[0])
    via_stack = torch.stack(dev_ids).to(dtype=torch.int32)
    via_host = torch.tensor(host_ids, dtype=torch.int32)
    assert torch.equal(via_stack, via_host)
    assert via_host.dtype == torch.int32
