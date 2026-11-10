""" flashkda backend """
from typing import Optional, Union

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.linear.kernels.kda_cute_checkpoint import (
    CuTeCheckpointAdapter,
    CuTeCheckpointPlan,
)
from sglang.srt.layers.attention.linear.kernels.kda_native_checkpoint import (
    KDANativeCheckpointResult,
    probe_pinned_style_native,
)
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)


def _load_flash_kda():
    """Return the ``flash_kda`` fwd/get_workspace_size module.

    With ``SGLANG_OPT_FLASH_KDA_CUTE`` enabled, use the vendored CuTe DSL
    drop-in (no external dependency); otherwise import the external MoonshotAI
    ``flash_kda`` CUTLASS package.
    """
    if envs.SGLANG_OPT_FLASH_KDA_CUTE.get():
        from sglang.kernels.ops.attention.linear import flash_kda_cute

        return flash_kda_cute
    try:
        import flash_kda
    except ImportError as e:
        raise ImportError(
            "The 'flashkda' KDA prefill backend requires the flash_kda module, "
            "which is not installed. Install it from source:\n"
            "    pip install git+https://github.com/MoonshotAI/FlashKDA.git\n"
            "or enable the in-tree CuTe DSL drop-in with "
            "SGLANG_OPT_FLASH_KDA_CUTE=1 (Blackwell SM100+)."
        ) from e
    return flash_kda


def _triton_fallback(
    q,
    k,
    v,
    g,
    beta,
    ssm_states,
    cache_indices,
    query_start_loc,
    A_log=None,
    dt_bias=None,
    lower_bound=None,
    return_intermediate_states=False,
):
    """Fall back to the Triton chunk_kda kernel (handles all preprocessing).

    `g` is the RAW gate; chunk_kda applies the gate activation internally when
    A_log is provided, so A_log/dt_bias/lower_bound must be threaded through too
    -- otherwise the fallback silently skips activation. chunk_kda updates the
    ssm state in-place via cache_indices and returns only the output tensor
    (or (output, h) when return_intermediate_states is set).
    """
    from sglang.kernels.ops.attention.fla.kda import chunk_kda

    return chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=ssm_states,
        initial_state_indices=cache_indices,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=query_start_loc,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        output_intermediate_states=return_intermediate_states,
    )


class FlashKDAKernel(LinearAttnKernelBase):
    """FlashKDA (MoonshotAI) fully-fused CUTLASS KDA prefill backend.

    Wraps the external ``flash_kda`` package (https://github.com/MoonshotAI/FlashKDA).

    FlashKDA fuses q/k L2 norm, beta sigmoid, and the KDA gate *inside* the
    kernel, so we pass RAW tensors plus ``A_log``/``dt_bias``/``lower_bound``.
    It is prefill-only, bf16, K == V == 128, HV == H (no GVA), and requires the
    safe (bounded) gate (``lower_bound`` set). The non-safe path and sequences
    outside [chunk_size, max_seq_len] fall back to Triton ``chunk_kda``.
    Requires an SM90+ GPU with the ``flash_kda`` package.
    """

    def __init__(self):
        # Extend-length window for the fused path, read once (not per forward).
        self.min_seq_len = envs.SGLANG_FLASHKDA_MIN_SEQ_LEN.get()
        self.max_seq_len = envs.SGLANG_FLASHKDA_MAX_SEQ_LEN.get()
        self._native_checkpoint_adapter = None
        self.native_checkpoint_capability = "disabled"
        # Register optional ops before reflection.
        flash_kda = _load_flash_kda()
        if envs.SGLANG_OPT_FLASH_KDA_CUTE.get():
            # CuTe has its own explicit adapter. Never infer its support from
            # an unrelated external torch.ops.flash_kda registration.
            self._native_checkpoint_adapter = CuTeCheckpointAdapter(flash_kda.fwd)
            self.native_checkpoint_capability = "cute_prefix_suffix_fp32_c32"
        else:
            (
                self._native_checkpoint_adapter,
                self.native_checkpoint_capability,
            ) = probe_pinned_style_native(torch, enabled=True)
        self.native_checkpoint_supported = self._native_checkpoint_adapter is not None

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError("FlashKDAKernel only supports prefill (extend)")

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        lower_bound: Optional[float] = None,
        extend_seq_lens_cpu: Optional[list] = None,
        is_spec_decode: bool = False,
        return_intermediate_states: bool = False,
        **kwargs,
    ) -> Union[
        torch.Tensor, tuple[torch.Tensor, torch.Tensor], KDANativeCheckpointResult
    ]:
        # Interior snapshots require an explicit external ABI or CuTe adapter
        # and an exact validated plan. Other cases keep whole-batch fallback.
        # Boundary-aligned tracks read the committed final state out of the pool
        # instead (track_ssm_final_src), never index h, and stay on the fused
        # path with a zero-row h stand-in -- same escape hatch as NvidiaKDAKernel.
        track_h_src = kwargs.get("track_ssm_h_src")
        needs_interior_snapshot = (
            return_intermediate_states
            and track_h_src is not None
            and track_h_src.numel() > 0
        )
        offsets = kwargs.get("native_checkpoint_offsets")
        owner_rows = kwargs.get("native_checkpoint_rows")
        cute_plan = kwargs.get("cute_checkpoint_plan")
        native_checkpoint_capable_for_plan = bool(
            needs_interior_snapshot
            and self._native_plan_is_eligible(
                q,
                k,
                v,
                g,
                beta,
                ssm_states,
                cache_indices,
                query_start_loc,
                A_log,
                dt_bias,
                offsets,
                owner_rows,
                track_h_src,
                cute_plan,
            )
        )
        # Preserve the legacy short-circuit (and avoid extra length readbacks)
        # when an interior snapshot cannot use this explicitly supported ABI.
        fall_back = (
            needs_interior_snapshot and not native_checkpoint_capable_for_plan
        ) or self._should_fall_back(
            lower_bound, is_spec_decode, query_start_loc, extend_seq_lens_cpu
        )
        native_checkpoint_used = native_checkpoint_capable_for_plan and not fall_back
        if fall_back:
            return _triton_fallback(
                q,
                k,
                v,
                g,
                beta,
                ssm_states,
                cache_indices,
                query_start_loc,
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=lower_bound,
                return_intermediate_states=return_intermediate_states,
            )

        out = self._flashkda_extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            native_checkpoint_offsets=offsets if native_checkpoint_used else None,
            cute_checkpoint_plan=cute_plan if native_checkpoint_used else None,
        )
        if isinstance(out, KDANativeCheckpointResult):
            return out
        if return_intermediate_states:
            # Keep the (out, h) contract; h is never indexed in this branch.
            h_empty = q.new_empty(
                (1, 0) + tuple(ssm_states.shape[1:]), dtype=torch.float32
            )
            return out, h_empty
        return out

    def _native_plan_is_eligible(
        self,
        q,
        k,
        v,
        g,
        beta,
        ssm_states,
        cache_indices,
        query_start_loc,
        A_log,
        dt_bias,
        offsets,
        owner_rows,
        track_h_src,
        cute_plan=None,
    ) -> bool:
        """Check tensor metadata only, without reading CUDA values per layer.

        Offset values/owner ordering were validated once from CPU inputs by the
        exact KDA metadata builder; no caller may synthesize an unvalidated plan.
        """
        if getattr(self, "_native_checkpoint_adapter", None) is None:
            return False
        if q.ndim != 4 or q.shape[0] != 1 or q.shape[3] != 128:
            return False
        if isinstance(self._native_checkpoint_adapter, CuTeCheckpointAdapter):
            if (
                not isinstance(cute_plan, CuTeCheckpointPlan)
                or q.shape[2] != 96
                or ssm_states.dtype != torch.float32
                or cute_plan.prefix_tokens.device != q.device
                or sum(cute_plan.lengths) != q.shape[1]
                or len(cute_plan.lengths) != cache_indices.numel()
                or cute_plan.active_rows.numel() != track_h_src.numel()
            ):
                return False
        if offsets is None or owner_rows is None or A_log is None or dt_bias is None:
            return False
        if any(x.shape != q.shape for x in (k, v, g)):
            return False
        if beta.shape != q.shape[:3] or any(
            x.dtype != torch.bfloat16 for x in (q, k, v, g)
        ):
            return False
        if q.device.type != "cuda" or any(
            x.device != q.device
            for x in (
                k,
                v,
                g,
                beta,
                ssm_states,
                cache_indices,
                query_start_loc,
                A_log,
                dt_bias,
                offsets,
                owner_rows,
            )
        ):
            return False
        n = cache_indices.numel()
        if (
            cache_indices.ndim != 1
            or query_start_loc.ndim != 1
            or query_start_loc.numel() != n + 1
            or offsets.shape != (n,)
            or owner_rows.ndim != 1
            or owner_rows.numel() != track_h_src.numel()
            or not owner_rows.numel()
        ):
            return False
        if (
            offsets.dtype != torch.int64
            or owner_rows.dtype != torch.int64
            or not offsets.is_contiguous()
            or not owner_rows.is_contiguous()
            or ssm_states.ndim != 4
            or tuple(ssm_states.shape[1:]) != (q.shape[2], 128, 128)
            or ssm_states.dtype not in (torch.bfloat16, torch.float32)
            or A_log.numel() != q.shape[2]
            or dt_bias.numel() != q.shape[2] * 128
        ):
            return False
        return True

    def _should_fall_back(
        self,
        lower_bound: Optional[float],
        is_spec_decode: bool,
        query_start_loc: torch.Tensor,
        extend_seq_lens_cpu: Optional[list],
    ) -> bool:
        """Whether to use the Triton chunk_kda path instead of the fused kernel."""
        # Safe-gate only: the fused kernel does not support the unbounded gate
        # (-exp(A_log)*softplus); those models leave lower_bound unset.
        if lower_bound is None:
            return True
        # FlashKDA writes the committed recurrent state back in place, so it is
        # unsafe for speculative verify / draft-extend forwards (which must stay
        # rollback-able). Those reach this backend through forward_extend, so
        # gate them here rather than relying on the decode/target_verify stubs.
        if is_spec_decode:
            return True
        # Short sequences (< chunk size) and long sequences (> the crossover
        # where Triton's chunked prefill wins) are faster on Triton. Read the
        # per-request lengths from the CPU-side extend_seq_lens to avoid a
        # GPU->CPU sync on every layer; derive from query_start_loc (one sync)
        # only if they are unavailable.
        if extend_seq_lens_cpu is not None:
            if torch.is_tensor(extend_seq_lens_cpu):
                lo = int(extend_seq_lens_cpu.min())
                hi = int(extend_seq_lens_cpu.max())
            else:
                lo = min(extend_seq_lens_cpu)
                hi = max(extend_seq_lens_cpu)
        else:
            seq_lens = query_start_loc[1:] - query_start_loc[:-1]
            lo_t, hi_t = torch.aminmax(seq_lens)
            lo, hi = int(lo_t), int(hi_t)
        return lo < self.min_seq_len or hi > self.max_seq_len

    def _flashkda_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        lower_bound: Optional[float] = None,
        native_checkpoint_offsets: Optional[torch.Tensor] = None,
        cute_checkpoint_plan: Optional[CuTeCheckpointPlan] = None,
    ) -> Union[torch.Tensor, KDANativeCheckpointResult]:
        flash_kda = _load_flash_kda()

        # Input shapes (varlen, B == 1, matching chunk_kda's contract):
        #   q, k = [1, packed_seq, H, K]   v = [1, packed_seq, HV, V]
        #   g    = [1, packed_seq, HV, K]  beta = [1, packed_seq, H]
        # flash_kda wants these 4D tensors directly and RAW (it fuses l2norm /
        # beta sigmoid / gate activation in-kernel).
        num_heads = q.shape[2]
        head_dim = q.shape[3]
        scale = head_dim**-0.5

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        g = g.contiguous()

        # KimiDeltaAttention.forward already applies sigmoid to beta on the
        # prefill path, but flash_kda expects beta LOGITS (it sigmoids
        # internally). Invert back so the kernel recovers the intended value:
        # sigmoid(logit(p)) == p. (triton/cuLA consume the post-sigmoid beta.)
        beta = torch.logit(beta.float().clamp_(1e-7, 1.0 - 1e-7)).to(torch.bfloat16)
        beta = beta.contiguous()

        # flash_kda wants A_log [H] fp32 and dt_bias [H, K] fp32. The model
        # stores A_log as [1, 1, H, 1] and dt_bias as 1D [H*K], so reshape both.
        A_log = A_log.reshape(-1).float().contiguous()
        if dt_bias is not None:
            dt_bias = dt_bias.reshape(num_heads, -1).float().contiguous()

        # cu_seqlens must be int64 for flash_kda (FLA casts to long).
        cu_seqlens = query_start_loc.to(torch.int64)

        # flash_kda varlen state is [N, H, V, K] -- the SAME layout as sglang's
        # KDA pool, so no transpose is needed. Advanced indexing copies, so the
        # final state is written back in-place below (matching chunk_kda).
        initial_state = ssm_states[cache_indices].contiguous()

        out_buf = torch.empty_like(v)
        final_state = torch.empty_like(initial_state)

        checkpoint_states = None
        if native_checkpoint_offsets is None:
            # Final-only/feature-off path preserves the old standalone call.
            flash_kda.fwd(
                q,
                k,
                v,
                g,
                beta,
                scale,
                out_buf,
                A_log,
                dt_bias,
                lower_bound,
                initial_state=initial_state,
                final_state=final_state,
                cu_seqlens=cu_seqlens,
            )
        elif isinstance(self._native_checkpoint_adapter, CuTeCheckpointAdapter):
            checkpoint_states = torch.empty_like(initial_state)
            self._native_checkpoint_adapter.invoke(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=float(scale),
                out=out_buf,
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=lower_bound,
                initial_state=initial_state,
                final_state=final_state,
                checkpoint_state=checkpoint_states,
                plan=cute_checkpoint_plan,
            )
        else:
            # This exact ABI and plan were checked before dispatch. Buffers are
            # per call/layer, never stored in shared metadata or on the backend.
            adapter = self._native_checkpoint_adapter
            checkpoint_states = torch.empty_like(initial_state)
            workspace = torch.empty(
                adapter.get_workspace_size(
                    q.shape[1], num_heads, cache_indices.numel()
                ),
                dtype=torch.uint8,
                device=q.device,
            )
            adapter.invoke(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=float(scale),
                out=out_buf,
                workspace=workspace,
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=lower_bound,
                initial_state=initial_state,
                final_state=final_state,
                cu_seqlens=cu_seqlens,
                checkpoint_state=checkpoint_states,
                checkpoint_offsets=native_checkpoint_offsets,
            )
            # Never catch launch errors and retry after a possibly-mutating op.

        ssm_states[cache_indices] = final_state

        # out_buf is already [1, packed_seq, HV, V].
        if checkpoint_states is not None:
            return KDANativeCheckpointResult(out_buf, checkpoint_states)
        return out_buf
