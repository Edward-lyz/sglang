"""Utilities for dispatching layer operators across supported platforms."""

from typing import Callable, ClassVar

from torch import nn
from sglang.srt.platforms import current_platform
from sglang.srt.utils import (
    cpu_has_amx_support,
    is_cpu,
    is_cuda,
    is_hip,
    is_musa,
    is_npu,
    is_xpu,
)

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_cpu = is_cpu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_npu = is_npu()
_is_xpu = is_xpu()
_is_musa = is_musa()


class MultiPlatformOp(nn.Module):
    """Base module that dispatches forward implementations by platform."""

    # OOT forward registry: maps dispatch_key -> {op_cls -> forward_fn}
    _oot_forward_registry: ClassVar[dict[str, dict[type, Callable]]] = {}

    @classmethod
    def register_oot_forward(cls, op_cls: type, fn: Callable, platform_key: str):
        """Register an OOT forward implementation for a specific op class and platform."""
        cls._oot_forward_registry.setdefault(platform_key, {})[op_cls] = fn

    def __init__(self):
        """Initialize the dispatched forward method state."""
        super().__init__()
        self._forward_method: Callable = self.dispatch_forward()

        # States for torch.compile
        self._original_forward_method = None
        self.is_torch_compile = False

    def enter_torch_compile(self, num_tokens: int):
        """Switch to the torch.compile-compatible forward implementation."""
        # Skip if Op is already entered compile mode.
        # NOTE(alcanderian): Some Ops(for example RotaryEmbedding) will be reused
        # among layers and `enter_torch_compile` will be called many times.
        # We should prevent `self._original_forward_method` from being overridden when
        # it is not the first time `enter_torch_compile` called.
        if self.is_torch_compile:
            return

        self._original_forward_method = self._forward_method
        # NOTE: Temporarily workaround MoE
        # The performance of torch.compile on this layer is not always good when bs > 1,
        # so we decide to only use torch.compile when bs=1
        if "FusedMoE" in self.__class__.__name__:
            if num_tokens == 1:
                from sglang.srt.layers.moe.fused_moe_native import (
                    fused_moe_forward_native,
                )

                self._forward_method = fused_moe_forward_native
        elif "TopK" in self.__class__.__name__:
            if num_tokens == 1:
                self._forward_method = self.forward_native
        else:
            self._forward_method = self.forward_native
        self.is_torch_compile = True

    def leave_torch_compile(self):
        """Restore the forward implementation used before torch.compile."""
        # Skip if Op is already exited compile mode.
        if not self.is_torch_compile:
            return

        self._forward_method = self._original_forward_method
        self._original_forward_method = None
        self.is_torch_compile = False

    # Please do not override this method, because `self._forward_method` can change when in torch compile mode
    def forward(self, *args, **kwargs):
        """Run the platform-dispatched forward implementation."""
        return self._forward_method(*args, **kwargs)

    def forward_native(self, *args, **kwargs):
        """Run the native PyTorch forward implementation."""
        raise NotImplementedError

    def forward_cuda(self, *args, **kwargs):
        """Run the CUDA forward implementation."""
        raise NotImplementedError

    def forward_npu(self, *args, **kwargs):
        """Run the NPU forward implementation."""
        raise NotImplementedError

    def forward_hip(self, *args, **kwargs):
        """Run the HIP forward implementation."""
        return self.forward_cuda(*args, **kwargs)

    def forward_xpu(self, *args, **kwargs):
        """Run the XPU forward implementation."""
        return self.forward_native(*args, **kwargs)

    def forward_musa(self, *args, **kwargs):
        """Run the MUSA forward implementation."""
        # XXX (MUSA): MUSA kernels follow the CUDA path by default.
        # At this stage, sgl-kernel support for MUSA is still under active
        # development, so we fall back to the PyTorch-native implementation.
        return self.forward_native(*args, **kwargs)

    def forward_hpu(self, *args, **kwargs):
        """Run the HPU forward implementation."""
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs):
        """Run the CPU forward implementation."""
        return self.forward_native(*args, **kwargs)

    def dispatch_forward(self):
        """Select the forward implementation for the current platform."""
        # OOT platform dispatch: check registry then method lookup
        if current_platform.is_out_of_tree():
            key = current_platform.get_dispatch_key_name()
            oot = self._oot_forward_registry.get(key, {})
            if type(self) in oot:
                return oot[type(self)].__get__(self)
            method = getattr(self, f"forward_{key}", None)
            if method is not None:
                return method
            return self.forward_native

        if _is_cuda:
            return self.forward_cuda
        elif _is_hip:
            return self.forward_hip
        elif _is_cpu and _is_cpu_amx_available:
            return self.forward_cpu
        elif _is_npu:
            return self.forward_npu
        elif _is_xpu:
            return self.forward_xpu
        elif _is_musa:
            return self.forward_musa
        else:
            return self.forward_native
