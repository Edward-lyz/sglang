"""Kimi K3 KV-balanced FP8 MLA decode (vLLM PR #63638).

``kernel.py`` / ``helpers.py`` / ``core.py`` are vendored unchanged except for the
Triton import. ``KvBalancedMlaDecoder`` adapts it to the tokenspeed_mla backend.
Needs nvidia-cutlass-dsl >= 4.8 and an SM10x GPU; import the submodules lazily.
"""
