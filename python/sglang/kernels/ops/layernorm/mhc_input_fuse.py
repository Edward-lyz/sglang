"""Fused DSV4.1-Flash Reuse-layer attention-input glue kernel (ONE Triton launch).

Collapses the decode-path glue that the profiler sees as ops 51-54:
  op52 hc_split_sinkhorn : pre[4]=sigmoid(mixes[:4]*s0+base[:4])+eps; post[4]; comb[4,4] sinkhorn
  op51 hc_combine        : y[h]=sum_{k<4} pre[k]*x_flat[k*H+h] (fp32) -> bf16
  op53 input RMSNorm     : normed=(y.f32*rsqrt(mean(y.f32^2)+eps)*w.f32).to(bf16)
  op54 mxfp8 swizzled quant (variant A only): per-group(32) ue8m0 e4m3 + 128x4 swizzled scale

Two entry points, gated by SGLANG_FUSE_MHC_PRE_INPUT (default OFF):
  fused_mhc_pre_input_norm  -> (normed_bf16, post, comb)   (3->1; norm_fused=True drop-in for
                                DeepseekV4DecoderLayer.hc_pre; the vendor mxfp8 GEMM does op54)
  fused_mhc_pre_input_quant -> (q_fp8, sf, post, comb)      (4->1; folds op54 too; needs the
                                caller to consume Mxfp8SwizzledInput -- see the run report)

Frozen from record_kernel_opt/deepseekv4.1_flash_decode_kernel_mhc_0922 (winner v3: block_m=1,
num_warps=4). Correctness on B200 sm_100: ue8m0 scale bytes 0/0 vs the 4-launch vendor path;
e4m3 activation ~3e-6 (RMSNorm reduction-order ULP). Only the DSV4.1-Flash decode geometry
(hc_mult=4, hidden=5120, group=32) is supported; other shapes raise NotImplementedError so the
caller falls back to the original path.
"""
import numpy as np
import torch
import triton
import triton.language as tl

INV448 = float(np.float32(1.0 / 448.0))
_HID = 5120
_HC = 4
_KFLAT = _HC * _HID          # 20480
_GROUP = 32
_G = _HID // _GROUP          # 160
_GPAD = triton.next_power_of_2(_G)   # 256
_KT = _G // 4                # 40
_MIX = (2 + _HC) * _HC       # 24


@triton.jit
def sglfuse_mhc_pre_input(
    mixes_ptr, xflat_ptr, scale_ptr, base_ptr, w_ptr,
    q_ptr, sf_ptr, post_ptr, comb_ptr, normed_ptr,
    M, rms_eps, hc_eps, inv448,
    HC: tl.constexpr, HID: tl.constexpr, KFLAT: tl.constexpr,
    G: tl.constexpr, GPAD: tl.constexpr, GROUP: tl.constexpr, KT: tl.constexpr,
    MIX: tl.constexpr, ITERS: tl.constexpr, BLOCK_M: tl.constexpr, EMIT_NORMED: tl.constexpr,
):
    """Fused DSV4.1-Flash pre-input glue kernel."""
    pid = tl.program_id(0)
    s0 = tl.load(scale_ptr + 0)
    s1 = tl.load(scale_ptr + 1)
    s2 = tl.load(scale_ptr + 2)
    gs = tl.arange(0, GPAD)
    ks = tl.arange(0, GROUP)
    col = gs[:, None] * GROUP + ks[None, :]
    gmask = gs < G
    m2 = gmask[:, None]
    kt = gs // 4
    ki = gs % 4
    jv = tl.arange(0, HC)
    jj = jv[:, None]
    kk = jv[None, :]
    for i in tl.static_range(BLOCK_M):
        row = pid * BLOCK_M + i
        if row < M:
            mbase = row * MIX
            acc = tl.zeros([GPAD, GROUP], dtype=tl.float32)
            for k in tl.static_range(HC):
                pk = tl.sigmoid(tl.load(mixes_ptr + mbase + k) * s0 + tl.load(base_ptr + k)) + hc_eps
                xk = tl.load(xflat_ptr + row * KFLAT + k * HID + col, mask=m2, other=0.0).to(tl.float32)
                acc += pk * xk
            postj = 2.0 * tl.sigmoid(
                tl.load(mixes_ptr + mbase + HC + jv) * s1 + tl.load(base_ptr + HC + jv))
            tl.store(post_ptr + row * HC + jv, postj)
            comb = tl.load(mixes_ptr + mbase + 2 * HC + jj * HC + kk) * s2 + tl.load(base_ptr + 2 * HC + jj * HC + kk)
            rmax = tl.max(comb, axis=1)
            comb = tl.exp(comb - rmax[:, None])
            comb = comb / tl.sum(comb, axis=1)[:, None] + hc_eps
            comb = comb / (tl.sum(comb, axis=0)[None, :] + hc_eps)
            for _ in tl.static_range(ITERS - 1):
                comb = comb / (tl.sum(comb, axis=1)[:, None] + hc_eps)
                comb = comb / (tl.sum(comb, axis=0)[None, :] + hc_eps)
            tl.store(comb_ptr + row * HC * HC + jj * HC + kk, comb)
            y = acc.to(tl.bfloat16).to(tl.float32)
            sq = tl.where(m2, y * y, 0.0)
            rq = tl.rsqrt(tl.sum(sq) / HID + rms_eps)
            w = tl.load(w_ptr + col, mask=m2, other=0.0).to(tl.float32)
            normed = (y * rq * w).to(tl.bfloat16).to(tl.float32)
            if EMIT_NORMED == 1:
                tl.store(normed_ptr + row * HID + col, normed.to(tl.bfloat16), mask=m2)
            else:
                amax = tl.maximum(tl.max(tl.abs(normed), axis=1), 1e-10)
                bits = (amax * inv448).to(tl.int32, bitcast=True)
                e8 = tl.minimum(((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32), 255)
                yscale = ((254 - e8) << 23).to(tl.float32, bitcast=True)
                q = tl.minimum(tl.maximum(normed * yscale[:, None], -448.0), 448.0).to(tl.float8e4nv)
                tl.store(q_ptr + row * HID + col, q, mask=m2)
                mi = row % 128
                off = (((row // 128 * KT + kt) * 32 + mi % 32) * 16 + (mi // 32) * 4 + ki)
                tl.store(sf_ptr + off, e8.to(tl.uint8), mask=gmask)


def is_supported(x_flat_shape, hc_mult, hidden):
    """Only the frozen DSV4.1-Flash decode geometry is supported."""
    return hc_mult == _HC and hidden == _HID and x_flat_shape[1] == _KFLAT


def _prep(mixes, x_flat, hc_scale, hc_base, weight, hc_mult, hidden):
    if not is_supported(tuple(x_flat.shape), hc_mult, hidden):
        raise NotImplementedError(
            "mhc_input_fuse supports only hc_mult=4, hidden=5120 (got hc_mult=%s hidden=%s)"
            % (hc_mult, hidden))
    if x_flat.dtype != torch.bfloat16 or not x_flat.is_contiguous():
        raise NotImplementedError("mhc_input_fuse needs contiguous bf16 x_flat")
    M = x_flat.shape[0]
    mixes_f = mixes.reshape(M, _MIX).float().contiguous()
    return (M, x_flat.device, mixes_f, hc_scale.float().contiguous(),
            hc_base.float().contiguous(), weight.to(torch.bfloat16).contiguous())


def _launch(mixes_f, x_flat, scale, base, w, q, sf, post, comb, normed,
            M, rms_eps, hc_eps, sinkhorn_iters, emit_normed, block_m=1, num_warps=4):
    sglfuse_mhc_pre_input[(triton.cdiv(M, block_m),)](
        mixes_f, x_flat, scale, base, w, q, sf, post, comb, normed,
        M, float(rms_eps), float(hc_eps), INV448,
        HC=_HC, HID=_HID, KFLAT=_KFLAT, G=_G, GPAD=_GPAD, GROUP=_GROUP, KT=_KT,
        MIX=_MIX, ITERS=sinkhorn_iters, BLOCK_M=block_m, EMIT_NORMED=emit_normed,
        num_warps=num_warps)


def fused_mhc_pre_input_norm(mixes, x_flat, hc_scale, hc_base, weight,
                             rms_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20,
                             hc_mult=_HC, hidden=_HID):
    """3->1 (op51+52+53). -> (normed[M,5120] bf16, post[M,4], comb[M,4,4]).
    Drop-in for hc_pre's norm_fused=True path; the vendor mxfp8 GEMM does op54."""
    M, dev, mixes_f, scale, base, w = _prep(mixes, x_flat, hc_scale, hc_base, weight, hc_mult, hidden)
    empty = torch.empty
    q = empty((1,), device=dev, dtype=torch.float8_e4m3fn)
    sf = empty((1,), device=dev, dtype=torch.uint8)
    post = empty((M, _HC), device=dev, dtype=torch.float32)
    comb = empty((M, _HC, _HC), device=dev, dtype=torch.float32)
    normed = empty((M, _HID), device=dev, dtype=torch.bfloat16)
    _launch(mixes_f, x_flat, scale, base, w, q, sf, post, comb, normed,
            M, rms_eps, hc_eps, sinkhorn_iters, emit_normed=1)
    return normed, post, comb


def fused_mhc_pre_input_quant(mixes, x_flat, hc_scale, hc_base, weight,
                              rms_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20,
                              hc_mult=_HC, hidden=_HID):
    """4->1 (op51+52+53+54). -> (q_fp8[M,5120] e4m3, sf[M*160] uint8 128x4-swizzled ue8m0,
    post[M,4], comb[M,4,4]). Caller wraps (q_fp8, sf) as Mxfp8SwizzledInput for the GEMM."""
    M, dev, mixes_f, scale, base, w = _prep(mixes, x_flat, hc_scale, hc_base, weight, hc_mult, hidden)
    q = torch.empty((M, _HID), device=dev, dtype=torch.float8_e4m3fn)
    sf = torch.empty((M * _G,), device=dev, dtype=torch.uint8)
    post = torch.empty((M, _HC), device=dev, dtype=torch.float32)
    comb = torch.empty((M, _HC, _HC), device=dev, dtype=torch.float32)
    normed = torch.empty((1,), device=dev, dtype=torch.bfloat16)
    _launch(mixes_f, x_flat, scale, base, w, q, sf, post, comb, normed,
            M, rms_eps, hc_eps, sinkhorn_iters, emit_normed=0)
    return q, sf, post, comb
