"""Cache compatible FlashInfer MXFP8 host dispatch (upstream #42156; the runner
factory and the skip_check kwarg it relies on already exist in FlashInfer 0.6.18).

Keep FlashInfer's output allocation, autotuner and kernel invocation intact.
Only reuse the stateless CuTe runner and validation of identical metadata.
"""

from functools import lru_cache
from inspect import Parameter, signature

from packaging.version import InvalidVersion, Version

try:
    from flashinfer import __version__ as flashinfer_version
    from flashinfer.gemm import gemm_base
except ImportError:
    flashinfer_version = ""
    gemm_base = None

_cached_runner_factories = set()


def maybe_cache_mxfp8_dispatch(raw_mm):
    """Keep unsupported FlashInfer versions and APIs on the original path."""
    if gemm_base is None:
        return raw_mm
    try:
        if Version(flashinfer_version) < Version("0.6.18"):
            return raw_mm
    except InvalidVersion:
        return raw_mm
    try:
        if not callable(gemm_base._cute_dsl_gemm_mxfp8_runner):
            return raw_mm
        parameters = signature(raw_mm, follow_wrapped=False).parameters
    except (AttributeError, TypeError, ValueError):
        return raw_mm
    skip_check = parameters.get("skip_check")
    accepts_keyword = skip_check is not None and skip_check.kind in (
        Parameter.POSITIONAL_OR_KEYWORD,
        Parameter.KEYWORD_ONLY,
    )
    if not accepts_keyword and not any(
        p.kind == Parameter.VAR_KEYWORD for p in parameters.values()
    ):
        return raw_mm
    return Mxfp8DispatchCache(raw_mm, gemm_base)


class Mxfp8DispatchCache:
    """Cache FlashInfer MXFP8 host dispatch."""
    def __init__(self, raw_mm, gemm_module):
        self.raw_mm = raw_mm
        self.validated_set = set()
        factory = gemm_module._cute_dsl_gemm_mxfp8_runner
        # This process-wide patch only memoizes the stateless
        # runner factory. The factory arguments include SM, PDL and output dtype.
        if factory not in _cached_runner_factories:
            factory = lru_cache(maxsize=16)(factory)
            gemm_module._cute_dsl_gemm_mxfp8_runner = factory
            _cached_runner_factories.add(factory)
        self.runner_factory = factory

    def __call__(self, a, b, a_scale, b_scale, out_dtype, use_8x4_sf_layout, backend):
        kwargs = dict(
            out_dtype=out_dtype,
            use_8x4_sf_layout=use_8x4_sf_layout,
            backend=backend,
        )
        if backend != "cute-dsl":
            return self.raw_mm(a, b, a_scale, b_scale, **kwargs)

        # Validation depends on metadata, never tensor contents. A changed
        # shape, stride, dtype or device must pass the original checks again.
        key = (
            out_dtype,
            use_8x4_sf_layout,
            tuple(
                (t.shape, t.stride(), t.dtype, t.device, t.layout)
                for t in (a, b, a_scale, b_scale)
            ),
        )
        is_validated = key in self.validated_set
        if is_validated:
            from sglang.srt.utils.community_opt import log_community_opt_once

            log_community_opt_once(
                "#42156 mxfp8_dispatch_cache",
                "cute-dsl mm_mxfp8 skip_check hit (%d cached keys)",
                len(self.validated_set),
            )
        result = self.raw_mm(a, b, a_scale, b_scale, skip_check=is_validated, **kwargs)
        if not is_validated:
            # Bound metadata memory for varying workloads. Only successful
            # executions qualify; a failed validation is never cached.
            if len(self.validated_set) >= 1024:
                self.validated_set.clear()
            self.validated_set.add(key)
        return result
