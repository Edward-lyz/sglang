#!/usr/bin/env python3
"""CPU wire differential, fallback/collective order, and GIL controls."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "99"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MAX_JOBS"] = "2"

import copyreg
import ctypes
import hashlib
import importlib.util
import json
import pickle
import random
import sys
import threading
import time
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "python"))
import sglang.srt.mem_cache.cpp_utils as package
import torch


def load(name, path):
    """Load a qualification module from an explicit source path.

    Args:
        name: Import name under which the source module is registered.
        path: Explicit filesystem path of the source module to load.

    Returns:
        ModuleType: The module registered under name.

    Raises:
        ImportError: The module or one of its dependencies cannot be loaded.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


codec = load(
    package.__name__ + ".request_oob",
    ROOT / "python/sglang/srt/mem_cache/cpp_utils/request_oob.py",
)
package.request_oob = codec
candidate = load("common", ROOT / "python/sglang/srt/utils/common.py")
candidate.envs.SGLANG_ENABLE_REQUEST_OOB_BROADCAST = SimpleNamespace(get=lambda: True)
from sglang.srt.managers.io_struct import (
    BatchTokenizedGenerateReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.sampling.sampling_params import SamplingParams

torch.set_num_threads(1)


class RegisteredOnly:
    """Represent a value serialized through a registered pickle reducer."""
    def __init__(self, value):
        self.value = value

    def __reduce_ex__(self, protocol):
        raise AssertionError("copyreg dispatch was lost")


def restore_registered(value):
    """Reconstruct a registered-only pickle fixture from its value.

    Args:
        value: Configured fixture value or injected failure.

    Returns:
        RegisteredOnly: The reconstructed fixture.
    """
    return RegisteredOnly(value)


class ArraySubclass(array):
    """Exercise legacy serialization for non-exact array instances."""
    pass


def differential():
    """Compare OOB round trips with the independent pickle wire oracle.

    Returns:
        int: Number of checked wire cases.

    Raises:
        AssertionError: A payload differs from the wire oracle.
    """
    count = 0
    rng = random.Random(98001)
    for code in "bBhHiIlLqQfdu":
        for length in (0, 1, 17, 1024, 61440):
            if code == "u":
                original = array(code, "图像😀a" * ((length + 3) // 4))[:length]
            elif code in "fd":
                original = array(code, (rng.uniform(-100, 100) for _ in range(length)))
            else:
                original = array(code, (rng.randrange(100) for _ in range(length)))
            payload = {"tokens": original, "alias": original, "nested": [complex(1, 2)]}
            packet = codec.encode(payload, [original])
            assert packet is not None
            decoded = codec.decode(packet)
            legacy = pickle.loads(pickle.dumps(payload))
            assert pickle.dumps(decoded, 4) == pickle.dumps(legacy, 4), (code, length)
            assert decoded["tokens"] is decoded["alias"]
            if length and code != "u":
                original[0] = 42
                assert decoded["tokens"].tobytes() == legacy["tokens"].tobytes()
            count += 1
    copyreg.pickle(RegisteredOnly, lambda value: (restore_registered, (value.value,)))
    try:
        a = array("q", [-(2**63), 2**63 - 1])
        subclass = ArraySubclass("q", [4, 5])
        subclass.extra = {"x": "preserved"}
        data = [a, a, RegisteredOnly(98), subclass]
        result = codec.decode(codec.encode(data, [a, subclass]))
        assert result[0] is result[1]
        assert result[0].tobytes() == a.tobytes()
        assert result[2].value == 98 and result[3].extra == subclass.extra
        assert type(result[3]) is ArraySubclass
        count += 1
    finally:
        copyreg.dispatch_table.pop(RegisteredOnly)
    # Malformed transport frames must fail before any unchecked buffer access.
    packet = codec.encode([array("q", [1, 2])], [])
    corrupt = [
        torch.empty(0, dtype=torch.uint8),
        packet[:-1],
        torch.cat((packet, torch.zeros(1, dtype=torch.uint8))),
    ]
    for start in (0, 8, 16):
        value = packet.clone()
        value[start : start + 8] = 255
        corrupt.append(value)
    for value in corrupt:
        try:
            codec.decode(value)
        except (ValueError, pickle.UnpicklingError):
            count += 1
        else:
            raise AssertionError("malformed frame accepted")
    native = codec.load_native_pack()
    for buffer in (memoryview(bytearray(8))[::2], memoryview(array("q", [1, 2]))):
        try:
            native(b"x", [buffer])
        except ValueError:
            count += 1
        else:
            raise AssertionError("native buffer contract not enforced")
    return count


def request(rid, **overrides):
    """Build a tokenized request fixture with explicit field overrides.

    Args:
        rid: Request identifier used by the fixture.
        overrides: Field overrides for the fixture or supported contract.

    Returns:
        TokenizedGenerateReqInput: The constructed request fixture.

    Raises:
        TypeError: An override is incompatible with the request structure.
    """
    values = dict(
        rid=rid,
        input_text=None,
        input_ids=array("q", range(61440)),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(max_new_tokens=1, temperature=0),
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=True,
    )
    values.update(overrides)
    return TokenizedGenerateReqInput(**values)


def protocol_contract():
    """Check request eligibility, legacy fallback and collective ordering.

    Returns:
        int: Number of checked protocol cases.

    Raises:
        AssertionError: A fallback or collective contract is violated.
    """
    text = [request(f"cpu:{i}") for i in range(3)]
    from attentionstore_sdk.sdk import Tokens

    as_text = [request(f"as:{i}") for i in range(3)]
    for row in as_text:
        row.as_block_hashs = Tokens(row.input_ids, 1024)
        row.as_kv_cache_params = {"namespace": "cpu", "cache_key": "preserved"}
    cases = [
        (text, True),
        (as_text, True),
        ([BatchTokenizedGenerateReqInput(batch=text)], True),
        ([request("mm", mm_inputs={"opaque": "MM fallback"})] * 3, False),
        ([request("embeds", input_embeds=[[0.1, 0.2]])] * 3, False),
        ([request("epd", task_results=[{"transport": "mooncake"}])] * 3, False),
        ([request("encoder", encoder_urls=["http://encoder.invalid"])] * 3, False),
        ([request("pending-epd", need_wait_for_mm_inputs=True)] * 3, False),
        ([request("session", session_id="keep-session-path")] * 3, False),
        ([*text, SimpleNamespace(control=True)], False),
        ([request("small")], False),
        ([], False),
        ([request("list", input_ids=[1, 2, 3])] * 3, False),
    ]
    count = 0
    for data, expect_oob in cases:
        for enabled in (False, True):
            channel = []
            modes = []

            def send(tensor, **kwargs):
                channel.append(tensor.clone())
                modes.append((tensor.dtype, tensor.numel()))

            def receive(tensor, **kwargs):
                tensor.copy_(channel.pop(0))

            with mock.patch.object(
                candidate.envs.SGLANG_ENABLE_REQUEST_OOB_BROADCAST,
                "get",
                return_value=enabled,
            ), mock.patch.object(candidate.dist, "broadcast", side_effect=send):
                assert candidate.broadcast_pyobj(data, 0, src=0) is data
            expected_collectives = 1 if not data else 2
            assert len(channel) == expected_collectives
            header = channel[0].item()
            assert (header < 0) == (enabled and expect_oob)
            with mock.patch.object(candidate.dist, "broadcast", side_effect=receive):
                result = candidate.broadcast_pyobj(None, 1, src=0)
            assert not channel
            assert pickle.dumps(result, 4) == pickle.dumps(
                pickle.loads(pickle.dumps(data)), 4
            )
            if data is as_text:
                assert all(
                    row.as_block_hashs.token_list is row.input_ids for row in result
                )
                assert (
                    result[0].as_block_hashs.hash_ids
                    == as_text[0].as_block_hashs.hash_ids
                )
            count += 1
    # Failure happens before the header, so all ranks retain the legacy sequence.
    with mock.patch.object(codec, "_PACK", None), mock.patch.object(
        codec, "_PACK_ATTEMPTED", True
    ):
        assert codec.maybe_encode_requests(text) is None
        count += 1
    with mock.patch.dict(copyreg.dispatch_table, {array: lambda value: None}):
        assert codec.maybe_encode_requests(text) is None
        count += 1
    return count


def progress():
    """Compare Python progress during native packing and GIL-held controls.

    Returns:
        list[dict]: Progress and timing measurements for each control.

    Raises:
        AssertionError: Native packing fails the progress controls.
    """
    tokens = array("q", range(8 * 1024 * 1024))
    control = ctypes.PyDLL(None).memcmp
    control.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    control.restype = ctypes.c_int
    other = tokens[:]
    native = codec.load_native_pack()
    raw = memoryview(tokens).cast("B")
    stop, ready = threading.Event(), threading.Event()
    active = False
    stamps = []

    def counter():
        ready.set()
        while not stop.is_set():
            if active:
                stamps.append(time.monotonic_ns())
            time.sleep(0.00001)

    thread = threading.Thread(target=counter)
    thread.start()
    ready.wait()
    old = sys.getswitchinterval()
    sys.setswitchinterval(60)
    rows = []
    try:
        for repeat in range(3):
            for kind in ("PyDLL_control", "native_pack"):
                # Retire the previous tensor outside the measured interval.
                # Its destructor can release the GIL independently of memcmp.
                result = None
                stamps.clear()
                active = True
                start = time.monotonic_ns()
                if kind == "PyDLL_control":
                    result = control(
                        tokens.buffer_info()[0],
                        other.buffer_info()[0],
                        len(tokens) * tokens.itemsize,
                    )
                else:
                    result = native(b"metadata", [raw])
                end = time.monotonic_ns()
                active = False
                interior = sum(start + 100000 < t < end - 100000 for t in stamps)
                assert interior == 0 if kind == "PyDLL_control" else interior > 0, (
                    kind,
                    repeat,
                    (end - start) / 1e6,
                    interior,
                )
                if kind == "PyDLL_control":
                    assert result == 0
                else:
                    assert result.numel() == len(raw) + 40
                rows.append(
                    dict(
                        kind=kind,
                        repeat=repeat,
                        call_ms=(end - start) / 1e6,
                        Python_progress=interior,
                        margin_us=100,
                    )
                )
    finally:
        active = False
        sys.setswitchinterval(old)
        stop.set()
        thread.join(timeout=10)
        assert not thread.is_alive()
    return rows


if __name__ == "__main__":
    result = dict(
        wire_cases=differential(),
        protocol_cases=protocol_contract(),
        native_GIL_progress=progress(),
        CUDA_initialized=torch.cuda.is_initialized(),
        source_sha256={
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                ROOT / "python/sglang/srt/utils/common.py",
                ROOT / "python/sglang/srt/mem_cache/cpp_utils/request_oob.py",
                ROOT / "python/sglang/srt/mem_cache/cpp_utils/request_oob_pack.cpp",
                Path(__file__),
            )
        },
        passed=True,
        serving_speedup_proven=False,
        limitations=["CPU transport controls, not an engine/MM/PD/byte digest proof"],
    )
    assert not result["CUDA_initialized"]
    Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
