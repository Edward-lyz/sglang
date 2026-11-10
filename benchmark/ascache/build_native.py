#!/usr/bin/env python3
"""Build the CP-aware radix extension and request packer before serving (CPU only)."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    """Build and prewarm the checked-in native components without GPU
    inference.

    Returns:
        None.

    Raises:
        Exception: Toolchain, compilation or library loading failed.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["SGLANG_RUST_BUILD_MODE"] = "auto"
    os.environ.setdefault("CARGO_BUILD_JOBS", "2")
    os.environ.setdefault("MAX_JOBS", "2")
    sys.path.insert(0, str(ROOT / "python"))
    import torch
    from sglang.srt.mem_cache.cpp_utils.request_oob import load_native_pack
    from sglang.srt.mem_cache.rust_tree_core.extension import bindings

    if load_native_pack() is None:
        raise RuntimeError("Request OOB packer compilation failed")
    assert not torch.cuda.is_initialized()
    print(json.dumps({"radix_extension": bindings.__file__, "CUDA_initialized": False}))


if __name__ == "__main__":
    main()
