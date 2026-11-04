"""Utility helpers for DeepSeek V4 JIT kernels."""


def make_name(name: str) -> str:
    """Return the cache key prefix used by DeepSeek V4 kernels."""
    return f"dpsk_v4_{name}"
