"""JIT wrappers for DeepSeek V4 Mega-MoE kernels."""

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

from .utils import make_name


@cache_once
def _jit_mega_moe_pre_dispatch_module(quant_group_size: int):
    """Load the Mega-MoE pre-dispatch CUDA module."""
    args = make_cpp_args(quant_group_size, is_arch_support_pdl())
    return load_jit(
        make_name("mega_moe_pre_dispatch"),
        *args,
        cuda_files=["deepseek_v4/mega_moe_pre_dispatch.cuh"],
        cuda_wrappers=[("run", f"MegaMoEPreDispatchKernel<{args}>::run")],
    )


def mega_moe_pre_dispatch(
    x,
    topk_idx,
    topk_weights,
    buf_x,
    buf_x_sf,
    buf_topk_idx,
    buf_topk_weights,
    quant_group_size: int = 32,
) -> None:
    """Quantize and arrange tokens before the DeepGEMM Mega-MoE kernel."""
    module = _jit_mega_moe_pre_dispatch_module(quant_group_size)
    module.run(
        x,
        topk_idx,
        topk_weights,
        buf_x,
        buf_x_sf,
        buf_topk_idx,
        buf_topk_weights,
    )
