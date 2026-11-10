"""Optional text-request OOB broadcast; legacy serialization remains available."""
from array import array, _array_reconstructor
from io import BytesIO
import copyreg
import logging
import pickle
from pathlib import Path
import struct

logger = logging.getLogger(__name__)
_PACK = None
_PACK_ATTEMPTED = False
_EXERCISED = 0
_MIN_TOKEN_BYTES = 1024 * 1024
_FORMATS = {}


def _machine_format(typecode):
    if typecode not in _FORMATS:
        reduced = array(typecode).__reduce_ex__(5)
        _FORMATS[typecode] = reduced[1][2]
    return _FORMATS[typecode]


def _restore_array(typecode, machine_format, buffer):
    # A Python frame between arrays allows the normal switch interval to run.
    # Preserve built-in mutable array semantics and portable machine formats.
    if machine_format != _machine_format(typecode):
        return _array_reconstructor(array, typecode, machine_format, bytes(buffer))
    result = array(typecode)
    result.frombytes(buffer)
    return result


def load_native_pack():
    """Load the CPU OOB packer once, retaining legacy fallback on failure.

    Returns:
        Optional[Callable]: Native pack function, or None if loading failed.
    """
    global _PACK, _PACK_ATTEMPTED
    if _PACK_ATTEMPTED:
        return _PACK
    _PACK_ATTEMPTED = True
    try:
        from torch.utils.cpp_extension import load

        module = load(
            name="request_oob",
            sources=[str(Path(__file__).with_name("request_oob_pack.cpp"))],
            extra_cflags=["-O3"], with_cuda=False, verbose=False)
        _PACK = module.pack
    except Exception:
        logger.exception("Request OOB native pack unavailable; retaining legacy broadcast")
    return _PACK


def encode(data, selected_arrays):
    """Serialize only selected exact arrays out of band; no source mutation."""
    selected_ids = {id(value) for value in selected_arrays}
    stream = BytesIO()
    buffers = []
    pickler = pickle.Pickler(
        stream, protocol=5, buffer_callback=buffers.append)

    def reduce_array(value):
        if id(value) not in selected_ids:
            return value.__reduce_ex__(5)
        return (_restore_array, (
            value.typecode, _machine_format(value.typecode),
            pickle.PickleBuffer(value)))

    pickler.dispatch_table = {**copyreg.dispatch_table, array: reduce_array}
    pickler.dump(data)
    pack = load_native_pack()
    if pack is None:
        return None
    return pack(stream.getvalue(), [buffer.raw() for buffer in buffers])


def decode(tensor):
    """Validate the complete frame before unpickling; no large bytes copy."""
    if tensor.device.type != "cpu" or tensor.dtype.is_floating_point:
        raise ValueError("request OOB packet must be a CPU byte tensor")
    import torch

    if tensor.dtype != torch.uint8 or tensor.ndim != 1 or not tensor.is_contiguous():
        raise ValueError("request OOB packet must be contiguous uint8")
    view = memoryview(tensor.numpy())
    if len(view) < 24 or view[:8] != b"SGBOOB01":
        raise ValueError("invalid request OOB packet header")
    metadata_len, count = struct.unpack_from(">QQ", view, 8)
    if count > (len(view) - 24) // 8:
        raise ValueError("invalid request OOB buffer count")
    cursor = 24 + count * 8
    if metadata_len > len(view) - cursor:
        raise ValueError("invalid request OOB metadata length")
    metadata = view[cursor:cursor + metadata_len]
    cursor += metadata_len
    buffers = []
    for i in range(count):
        size = struct.unpack_from(">Q", view, 24 + i * 8)[0]
        if size > len(view) - cursor:
            raise ValueError("invalid request OOB buffer length")
        buffers.append(view[cursor:cursor + size])
        cursor += size
    if cursor != len(view):
        raise ValueError("trailing bytes in request OOB frame")
    return pickle.loads(metadata, buffers=buffers)


def maybe_encode_requests(data):
    """The fast path only sees private, received text requests.

    MM, embedding, RPC/control, input embeddings and custom token arrays retain
    the legacy wire format. A mixed work/control list also keeps that format.
    """
    if array in copyreg.dispatch_table:
        return None
    from sglang.srt.managers.io_struct import (
        BatchTokenizedGenerateReqInput, TokenizedGenerateReqInput,
    )
    if type(data) is not list or not data:
        return None
    rows = []
    for item in data:
        if type(item) is TokenizedGenerateReqInput:
            rows.append(item)
        elif type(item) is BatchTokenizedGenerateReqInput:
            rows.extend(item.batch)
        else:
            return None
    arrays = []
    for row in rows:
        if (type(row) is not TokenizedGenerateReqInput
                or row.mm_inputs is not None or row.input_embeds is not None
                or row.task_results is not None or row.encoder_urls is not None
                or row.need_wait_for_mm_inputs or row.num_items_assigned is not None
                or row.positional_embed_overrides is not None
                or row.session_id is not None or row.session_params is not None
                or type(row.input_ids) is not array or row.input_ids.typecode != "q"):
            return None
        arrays.append(row.input_ids)
    if sum(len(value) * value.itemsize for value in arrays) < _MIN_TOKEN_BYTES:
        return None
    global _EXERCISED
    try:
        packet = encode(data, arrays)
    except Exception:
        # No collective has occurred yet. The caller will use the legacy packet.
        logger.exception("Request OOB pack failed; retaining legacy broadcast")
        return None
    if packet is not None and not _EXERCISED & 1:
        _EXERCISED |= 1
        logger.info("Request OOB native pack exercised: bytes=%d arrays=%d",
                    packet.numel(), len(arrays))
    return packet


def decode_requests(tensor):
    """Decode a CPU OOB packet and record first-use diagnostics.

    Args:
        tensor: Mutable CPU vote tensor representing rank agreement.

    Returns:
        Any: The request objects reconstructed from the packet.

    Raises:
        ValueError: The OOB packet is malformed.
        pickle.UnpicklingError: The packet metadata cannot be unpickled.
    """
    global _EXERCISED
    result = decode(tensor)
    if not _EXERCISED & 2:
        _EXERCISED |= 2
        logger.info("Request OOB decode exercised: bytes=%d", tensor.numel())
    return result
