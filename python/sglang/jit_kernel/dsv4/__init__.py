"""DeepSeek V4 JIT kernel helpers."""

from .moe import mega_moe_pre_dispatch
from .utils import make_name

__all__ = [
    "make_name",
    "mega_moe_pre_dispatch",
]
