# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused V4.1 HC transition: expand (in place) + next mixes + delayed collapse.

One sublayer boundary of DeepSeek-V4.1 is

    residual = expand(y, residual, post, comb)
    pre', post', comb' = sinkhorn(norm_proj(residual, fn_next))
    inputs = rms_norm(collapse(residual, pre_delayed)) * norm_weight

which previously took five launches and a fresh [T, 4, H] FP32 residual. Here
K1 updates the residual in place and emits fixed-order partial reductions; K2
finishes the mixes, Sinkhorn and the normalized FP16 collapse. Reduction order
is fixed per row, so M1..M6 graphs produce identical routes for a token.
"""

import os

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

H = 5120
COLS = 256
CHUNKS = H // COLS  # 20

_MHC_PREFILL = envs.VLLM_V41_MHC_PREFILL == "1"
if _MHC_PREFILL:
    if envs.VLLM_V41_HC_TOKEN_BLOCKED == "1":
        raise ValueError(
            "L2c requires the 9Z windowed route (VLLM_V41_HC_TOKEN_BLOCKED=0)"
        )
    from .deepseek_v41_regs import MIN_ROWS, register_transition



@triton.jit
def _transition_parts(
    residual,
    y,
    post,
    comb,
    fn,
    cpre,
    cbuf,
    dots,
    squares,
    csquares,
    MIX: tl.constexpr,
    H: tl.constexpr,
    COLS: tl.constexpr,
    CHUNKS: tl.constexpr,
):
    t, chunk = tl.program_id(0), tl.program_id(1)
    col = chunk * COLS + tl.arange(0, COLS)
    base = residual + t * 4 * H
    r0 = tl.load(base + 0 * H + col)
    r1 = tl.load(base + 1 * H + col)
    r2 = tl.load(base + 2 * H + col)
    r3 = tl.load(base + 3 * H + col)
    yv = tl.load(y + t * H + col).to(tl.float32)
    cb = comb + t * 16
    pb = post + t * 4
    # comb[source * 4 + output], same as the reference expand.
    n0 = (
        tl.load(cb + 0) * r0
        + tl.load(cb + 4) * r1
        + tl.load(cb + 8) * r2
        + tl.load(cb + 12) * r3
        + tl.load(pb + 0) * yv
    )
    n1 = (
        tl.load(cb + 1) * r0
        + tl.load(cb + 5) * r1
        + tl.load(cb + 9) * r2
        + tl.load(cb + 13) * r3
        + tl.load(pb + 1) * yv
    )
    n2 = (
        tl.load(cb + 2) * r0
        + tl.load(cb + 6) * r1
        + tl.load(cb + 10) * r2
        + tl.load(cb + 14) * r3
        + tl.load(pb + 2) * yv
    )
    n3 = (
        tl.load(cb + 3) * r0
        + tl.load(cb + 7) * r1
        + tl.load(cb + 11) * r2
        + tl.load(cb + 15) * r3
        + tl.load(pb + 3) * yv
    )
    tl.store(base + 0 * H + col, n0)
    tl.store(base + 1 * H + col, n1)
    tl.store(base + 2 * H + col, n2)
    tl.store(base + 3 * H + col, n3)
    pc = cpre + t * 4
    coll = (
        n0 * tl.load(pc + 0)
        + n1 * tl.load(pc + 1)
        + n2 * tl.load(pc + 2)
        + n3 * tl.load(pc + 3)
    )
    tl.store(cbuf + t * H + col, coll)
    tl.store(csquares + t * CHUNKS + chunk, tl.sum(coll * coll, 0))
    if MIX:
        tl.store(
            squares + t * CHUNKS + chunk,
            tl.sum(n0 * n0 + n1 * n1 + n2 * n2 + n3 * n3, 0),
        )
        for r in tl.static_range(24):
            w = fn + r * 4 * H
            d = (
                tl.load(w + 0 * H + col) * n0
                + tl.load(w + 1 * H + col) * n1
                + tl.load(w + 2 * H + col) * n2
                + tl.load(w + 3 * H + col) * n3
            )
            tl.store(dots + (t * CHUNKS + chunk) * 24 + r, tl.sum(d, 0))


COLS_T = 128
CHUNKS_T = H // COLS_T  # 40
TOK_T = 16
_TOKEN_BLOCKED = envs.VLLM_V41_HC_TOKEN_BLOCKED == "1"


@triton.jit(do_not_specialize=["TOKENS"])
def _transition_parts_tok(
    residual,
    y,
    post,
    comb,
    fn,
    cpre,
    cbuf,
    dots,
    squares,
    csquares,
    TOKENS,
    MIX: tl.constexpr,
    H: tl.constexpr,
    COLS: tl.constexpr,
    CHUNKS: tl.constexpr,
    TOK: tl.constexpr,
):
    block, chunk = tl.program_id(0), tl.program_id(1)
    t = block * TOK + tl.arange(0, TOK)
    live = t < TOKENS
    col = chunk * COLS + tl.arange(0, COLS)
    m2 = live[:, None]
    base = residual + t[:, None].to(tl.int64) * 4 * H + col[None, :]
    r0 = tl.load(base + 0 * H, m2, 0.0)
    r1 = tl.load(base + 1 * H, m2, 0.0)
    r2 = tl.load(base + 2 * H, m2, 0.0)
    r3 = tl.load(base + 3 * H, m2, 0.0)
    yv = tl.load(y + t[:, None] * H + col[None, :], m2, 0.0).to(tl.float32)
    cb = comb + t * 16
    pb = post + t * 4
    pc = cpre + t * 4
    coll = tl.zeros((TOK, COLS), tl.float32)
    sq = tl.zeros((TOK,), tl.float32)
    acc = tl.zeros((TOK, 32), tl.float32)
    r32 = tl.arange(0, 32)
    for o in tl.static_range(4):
        n = (
            tl.load(cb + o, live, 0.0)[:, None] * r0
            + tl.load(cb + 4 + o, live, 0.0)[:, None] * r1
            + tl.load(cb + 8 + o, live, 0.0)[:, None] * r2
            + tl.load(cb + 12 + o, live, 0.0)[:, None] * r3
            + tl.load(pb + o, live, 0.0)[:, None] * yv
        )
        tl.store(base + o * H, n, m2)
        coll += n * tl.load(pc + o, live, 0.0)[:, None]
        if MIX:
            sq += tl.sum(n * n, 1)
            w = tl.load(
                fn + r32[None, :] * 4 * H + o * H + col[:, None],
                (r32 < 24)[None, :],
                0.0,
            )  # [COLS, 32]
            acc += tl.dot(n, w, allow_tf32=False)
    tl.store(cbuf + t[:, None] * H + col[None, :], coll, m2)
    tl.store(csquares + t * CHUNKS + chunk, tl.sum(coll * coll, 1), live)
    if MIX:
        tl.store(squares + t * CHUNKS + chunk, sq, live)
        tl.store(
            dots + (t[:, None] * CHUNKS + chunk) * 24 + r32[None, :],
            acc,
            m2 & (r32 < 24)[None, :],
        )


@triton.jit
def _transition_finish(
    dots,
    squares,
    csquares,
    cbuf,
    norm,
    out,
    scale,
    base,
    pre,
    post,
    comb,
    MIX: tl.constexpr,
    H: tl.constexpr,
    CHUNKS: tl.constexpr,
    NORM_EPS: tl.constexpr,
    HC_EPS: tl.constexpr,
    ITERS: tl.constexpr,
    BLOCK: tl.constexpr,
    COLLAPSE: tl.constexpr = True,
):
    t = tl.program_id(0)
    if COLLAPSE:
        total = tl.full((), 0, tl.float32)
        for c in range(CHUNKS):
            total += tl.load(csquares + t * CHUNKS + c)
        inv = tl.rsqrt(total / H + NORM_EPS)
        for first in range(0, H, BLOCK):
            col = first + tl.arange(0, BLOCK)
            v = tl.load(cbuf + t * H + col) * inv * tl.load(norm + col)
            tl.store(out + t * H + col, v.to(tl.float16))
    if MIX:
        i = tl.arange(0, 4)
        sq = tl.full((), 0, tl.float32)
        for c in range(CHUNKS):
            sq += tl.load(squares + t * CHUNKS + c)
        rs = tl.rsqrt(sq / (4 * H) + NORM_EPS)
        p = tl.zeros((4,), tl.float32)
        q = tl.zeros((4,), tl.float32)
        location = i[:, None] * 4 + i[None, :]
        m = tl.zeros((4, 4), tl.float32)
        for c in range(CHUNKS):
            row = dots + (t * CHUNKS + c) * 24
            p += tl.load(row + i)
            q += tl.load(row + 4 + i)
            m += tl.load(row + 8 + location)
        p = p * rs * tl.load(scale) + tl.load(base + i)
        q = q * rs * tl.load(scale + 1) + tl.load(base + 4 + i)
        tl.store(pre + t * 4 + i, tl.sigmoid(p) + HC_EPS)
        tl.store(post + t * 4 + i, 2.0 * tl.sigmoid(q))
        value = m * rs * tl.load(scale + 2) + tl.load(base + 8 + location)
        value = tl.exp(value - tl.max(value, 1)[:, None])
        value = value / tl.sum(value, 1)[:, None] + HC_EPS
        value = value / (tl.sum(value, 0)[None, :] + HC_EPS)
        for _ in range(ITERS - 1):
            value = value / (tl.sum(value, 1)[:, None] + HC_EPS)
            value = value / (tl.sum(value, 0)[None, :] + HC_EPS)
        tl.store(comb + t * 16 + location, value)



_WINDOW = int(envs.VLLM_V41_HC_WINDOW)
COLS2 = int(envs.VLLM_V41_HC_COLS)
CHUNKS2 = H // COLS2
PARTS_WARPS = int(envs.VLLM_V41_HC_PW)
FINISH_WARPS = int(envs.VLLM_V41_HC_FW)
_V2 = envs.VLLM_V41_HC_V2 == "1"


@triton.jit
def _transition_parts2(
    residual,
    y,
    post,
    comb,
    fn,
    cpre,
    cbuf,
    dots,
    squares,
    csquares,
    TOK: tl.constexpr,
    MIX: tl.constexpr,
    H: tl.constexpr,
    COLS: tl.constexpr,
    CHUNKS: tl.constexpr,
):
    # One program per column chunk serves every token, so fn is read once.
    chunk = tl.program_id(0)
    col = chunk * COLS + tl.arange(0, COLS)
    s4 = tl.arange(0, 4)
    r32 = tl.arange(0, 32)
    if MIX:
        w = tl.load(
            fn + (r32[:, None, None] * 4 + s4[None, :, None]) * H + col[None, None, :],
            mask=(r32 < 24)[:, None, None],
            other=0.0,
        )
    for t in tl.static_range(TOK):
        base = residual + t * 4 * H
        r = tl.load(base + s4[:, None] * H + col[None, :])  # [4, COLS] source
        yv = tl.load(y + t * H + col).to(tl.float32)
        cb = tl.load(comb + t * 16 + s4[:, None] * 4 + s4[None, :])  # [src, out]
        pb = tl.load(post + t * 4 + s4)
        # n[o, c] = sum_s comb[s, o] * r[s, c] + post[o] * y[c]
        n = tl.sum(cb[:, :, None] * r[:, None, :], 0) + pb[:, None] * yv[None, :]
        tl.store(base + s4[:, None] * H + col[None, :], n)
        pc = tl.load(cpre + t * 4 + s4)
        coll = tl.sum(n * pc[:, None], 0)
        tl.store(cbuf + t * H + col, coll)
        tl.store(csquares + t * CHUNKS + chunk, tl.sum(coll * coll, 0))
        if MIX:
            tl.store(squares + t * CHUNKS + chunk, tl.sum(tl.sum(n * n, 1), 0))
            d = tl.sum(tl.sum(w * n[None, :, :], 2), 1)
            tl.store(dots + (t * CHUNKS + chunk) * 24 + r32, d, mask=r32 < 24)


@triton.jit
def _transition_finish2(
    dots,
    squares,
    csquares,
    cbuf,
    norm,
    out,
    scale,
    base,
    pre,
    post,
    comb,
    MIX: tl.constexpr,
    H: tl.constexpr,
    CHUNKS: tl.constexpr,
    CP2: tl.constexpr,
    NORM_EPS: tl.constexpr,
    HC_EPS: tl.constexpr,
    ITERS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t, part = tl.program_id(0), tl.program_id(1)
    c = tl.arange(0, CP2)
    cm = c < CHUNKS
    total = tl.sum(tl.load(csquares + t * CHUNKS + c, mask=cm, other=0.0), 0)
    inv = tl.rsqrt(total / H + NORM_EPS)
    col = part * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(cbuf + t * H + col) * inv * tl.load(norm + col)
    tl.store(out + t * H + col, v.to(tl.float16))
    if MIX:  # noqa: SIM102
        if part == 0:
            i = tl.arange(0, 4)
            sq = tl.sum(tl.load(squares + t * CHUNKS + c, mask=cm, other=0.0), 0)
            rs = tl.rsqrt(sq / (4 * H) + NORM_EPS)
            r32 = tl.arange(0, 32)
            tile = tl.load(
                dots + (t * CHUNKS + c[:, None]) * 24 + r32[None, :],
                mask=cm[:, None] & (r32 < 24)[None, :],
                other=0.0,
            )
            acc = tl.sum(tile, 0)  # [32]
            location = i[:, None] * 4 + i[None, :]
            p = tl.sum(tl.where(r32[None, :] == i[:, None], acc[None, :], 0.0), 1)
            q = tl.sum(tl.where(r32[None, :] == (i[:, None] + 4), acc[None, :], 0.0), 1)
            m = tl.sum(
                tl.where(
                    r32[None, None, :] == (location[:, :, None] + 8),
                    acc[None, None, :],
                    0.0,
                ),
                2,
            )
            p = p * rs * tl.load(scale) + tl.load(base + i)
            q = q * rs * tl.load(scale + 1) + tl.load(base + 4 + i)
            tl.store(pre + t * 4 + i, tl.sigmoid(p) + HC_EPS)
            tl.store(post + t * 4 + i, 2.0 * tl.sigmoid(q))
            value = m * rs * tl.load(scale + 2) + tl.load(base + 8 + location)
            value = tl.exp(value - tl.max(value, 1)[:, None])
            value = value / tl.sum(value, 1)[:, None] + HC_EPS
            value = value / (tl.sum(value, 0)[None, :] + HC_EPS)
            for _ in range(ITERS - 1):
                value = value / (tl.sum(value, 1)[:, None] + HC_EPS)
                value = value / (tl.sum(value, 0)[None, :] + HC_EPS)
            tl.store(comb + t * 16 + location, value)


def hc_transition(
    residual,
    y,
    post,
    comb,
    collapse_pre,
    norm_weight,
    fn=None,
    scale=None,
    base=None,
    *,
    norm_eps=1e-20,
    hc_eps=1e-6,
    iters=20,
):
    """In-place expand; returns (inputs_fp16, (pre, post, comb) or None)."""
    tokens = residual.shape[0]
    if (
        residual.shape != (tokens, 4, H)
        or residual.dtype != torch.float32
        or not residual.is_contiguous()
        or y.shape != (tokens, H)
        or not y.is_contiguous()
    ):
        raise ValueError("V4.1 fused HC transition requires FP32 [T, 4, 5120]")
    if _MHC_PREFILL and tokens >= MIN_ROWS:
        return register_transition(
            residual, y, post, comb, collapse_pre, norm_weight, fn, scale, base,
            norm_eps=norm_eps, hc_eps=hc_eps, iters=iters,
        )
    mix = fn is not None
    device = residual.device
    if _FUSE_TRANSITION and _V3 and 0 < tokens <= _FT_TOKENS:
        return _hc_transition_fused(
            residual,
            y,
            post,
            comb,
            collapse_pre,
            norm_weight,
            fn,
            scale,
            base,
            norm_eps,
            hc_eps,
            iters,
        )
    if _V3 and 0 < tokens <= 8:
        return _hc_transition3(
            residual,
            y,
            post,
            comb,
            collapse_pre,
            norm_weight,
            fn,
            scale,
            base,
            norm_eps,
            hc_eps,
            iters,
        )
    if _CUDA and 0 < tokens <= 8 and _cuda_ready():
        return _hc_transition_cuda(
            residual,
            y,
            post,
            comb,
            collapse_pre,
            norm_weight,
            fn,
            scale,
            base,
            norm_eps,
            hc_eps,
            iters,
        )
    if _V2 and 0 < tokens <= 8:
        return _hc_transition2(
            residual,
            y,
            post,
            comb,
            collapse_pre,
            norm_weight,
            fn,
            scale,
            base,
            norm_eps,
            hc_eps,
            iters,
        )
    if _TOKEN_BLOCKED and tokens > 8:
        return _hc_transition_tok(
            residual,
            y,
            post,
            comb,
            collapse_pre,
            norm_weight,
            fn,
            scale,
            base,
            norm_eps,
            hc_eps,
            iters,
        )
    # Per-token kernels: process bounded windows so the FP32 collapse buffer
    # does not scale with the prefill chunk.
    window = min(tokens, _WINDOW)
    cbuf = torch.empty(window, H, dtype=torch.float32, device=device)
    csq = torch.empty(window, CHUNKS, dtype=torch.float32, device=device)
    if mix:
        dots = torch.empty(window, CHUNKS, 24, dtype=torch.float32, device=device)
        sq = torch.empty(window, CHUNKS, dtype=torch.float32, device=device)
        pre = torch.empty(tokens, 4, dtype=torch.float32, device=device)
        post_n = torch.empty_like(pre)
        comb_n = torch.empty(tokens, 4, 4, dtype=torch.float32, device=device)
    else:
        dots = sq = csq
    out = torch.empty(tokens, H, dtype=torch.float16, device=device)
    for a in range(0, tokens, window):
        b = min(a + window, tokens)
        n = b - a
        if mix:
            pre_w, post_w, comb_w = pre[a:b], post_n[a:b], comb_n[a:b]
        else:
            pre_w = post_w = comb_w = csq
        _transition_parts[(n, CHUNKS)](
            residual[a:b],
            y[a:b],
            post[a:b],
            comb[a:b],
            fn if mix else post,
            collapse_pre[a:b],
            cbuf,
            dots,
            sq,
            csq,
            MIX=mix,
            H=H,
            COLS=COLS,
            CHUNKS=CHUNKS,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _transition_finish[(n,)](
            dots,
            sq,
            csq,
            cbuf,
            norm_weight,
            out[a:b],
            scale if mix else csq,
            base if mix else csq,
            pre_w,
            post_w,
            comb_w,
            MIX=mix,
            H=H,
            CHUNKS=CHUNKS,
            NORM_EPS=norm_eps,
            HC_EPS=hc_eps,
            ITERS=iters,
            BLOCK=1024,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out, ((pre, post_n, comb_n) if mix else None)


def _hc_transition_tok(
    residual,
    y,
    post,
    comb,
    collapse_pre,
    norm_weight,
    fn,
    scale,
    base,
    norm_eps,
    hc_eps,
    iters,
):
    tokens = residual.shape[0]
    mix = fn is not None
    device = residual.device
    window = min(tokens, _WINDOW)
    cbuf = torch.empty(window, H, dtype=torch.float32, device=device)
    csq = torch.empty(window, CHUNKS_T, dtype=torch.float32, device=device)
    if mix:
        dots = torch.empty(window, CHUNKS_T, 24, dtype=torch.float32, device=device)
        sq = torch.empty(window, CHUNKS_T, dtype=torch.float32, device=device)
        pre = torch.empty(tokens, 4, dtype=torch.float32, device=device)
        post_n = torch.empty_like(pre)
        comb_n = torch.empty(tokens, 4, 4, dtype=torch.float32, device=device)
    else:
        dots = sq = csq
    out = torch.empty(tokens, H, dtype=torch.float16, device=device)
    for a in range(0, tokens, window):
        b = min(a + window, tokens)
        n = b - a
        if mix:
            pre_w, post_w, comb_w = pre[a:b], post_n[a:b], comb_n[a:b]
        else:
            pre_w = post_w = comb_w = csq
        _transition_parts_tok[(triton.cdiv(n, TOK_T), CHUNKS_T)](
            residual[a:b],
            y[a:b],
            post[a:b],
            comb[a:b],
            fn if mix else post,
            collapse_pre[a:b],
            cbuf,
            dots,
            sq,
            csq,
            n,
            MIX=mix,
            H=H,
            COLS=COLS_T,
            CHUNKS=CHUNKS_T,
            TOK=TOK_T,
            num_warps=8,
            enable_fp_fusion=False,
        )
        _transition_finish[(n,)](
            dots,
            sq,
            csq,
            cbuf,
            norm_weight,
            out[a:b],
            scale if mix else csq,
            base if mix else csq,
            pre_w,
            post_w,
            comb_w,
            MIX=mix,
            H=H,
            CHUNKS=CHUNKS_T,
            NORM_EPS=norm_eps,
            HC_EPS=hc_eps,
            ITERS=iters,
            BLOCK=1024,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out, ((pre, post_n, comb_n) if mix else None)


def _hc_transition2(
    residual,
    y,
    post,
    comb,
    collapse_pre,
    norm_weight,
    fn,
    scale,
    base,
    norm_eps,
    hc_eps,
    iters,
):
    tokens = residual.shape[0]
    mix = fn is not None
    device = residual.device
    cbuf = torch.empty(tokens, H, dtype=torch.float32, device=device)
    csq = torch.empty(tokens, CHUNKS2, dtype=torch.float32, device=device)
    if mix:
        dots = torch.empty(tokens, CHUNKS2, 24, dtype=torch.float32, device=device)
        sq = torch.empty(tokens, CHUNKS2, dtype=torch.float32, device=device)
        pre = torch.empty(tokens, 4, dtype=torch.float32, device=device)
        post_n = torch.empty_like(pre)
        comb_n = torch.empty(tokens, 4, 4, dtype=torch.float32, device=device)
    else:
        dots = sq = pre = post_n = comb_n = csq
    out = torch.empty(tokens, H, dtype=torch.float16, device=device)
    _transition_parts2[(CHUNKS2,)](
        residual,
        y,
        post,
        comb,
        fn if mix else post,
        collapse_pre,
        cbuf,
        dots,
        sq,
        csq,
        TOK=tokens,
        MIX=mix,
        H=H,
        COLS=COLS2,
        CHUNKS=CHUNKS2,
        num_warps=PARTS_WARPS,
        enable_fp_fusion=False,
    )
    _transition_finish2[(tokens, H // 1024)](
        dots,
        sq,
        csq,
        cbuf,
        norm_weight,
        out,
        scale if mix else csq,
        base if mix else csq,
        pre,
        post_n,
        comb_n,
        MIX=mix,
        H=H,
        CHUNKS=CHUNKS2,
        CP2=triton.next_power_of_2(CHUNKS2),
        NORM_EPS=norm_eps,
        HC_EPS=hc_eps,
        ITERS=iters,
        BLOCK=1024,
        num_warps=FINISH_WARPS,
        enable_fp_fusion=False,
    )
    return out, ((pre, post_n, comb_n) if mix else None)


_CUDA = envs.VLLM_V41_HC_CUDA == "1"


def _cuda_ready():
    from vllm.model_executor.layers.quantization.gguf_v41_moe import _expert_major_ready

    return _expert_major_ready() and hasattr(torch.ops.v41k, "hc_transition")


def _hc_transition_cuda(
    residual,
    y,
    post,
    comb,
    collapse_pre,
    norm_weight,
    fn,
    scale,
    base,
    norm_eps,
    hc_eps,
    iters,
):
    tokens = residual.shape[0]
    mix = fn is not None
    device = residual.device
    chunks = H // 32
    cbuf = torch.empty(tokens, H, dtype=torch.float32, device=device)
    csq = torch.empty(tokens, chunks, dtype=torch.float32, device=device)
    out = torch.empty(tokens, H, dtype=torch.float16, device=device)
    if mix:
        dots = torch.empty(tokens, chunks, 24, dtype=torch.float32, device=device)
        sq = torch.empty(tokens, chunks, dtype=torch.float32, device=device)
        pre = torch.empty(tokens, 4, dtype=torch.float32, device=device)
        post_n = torch.empty_like(pre)
        comb_n = torch.empty(tokens, 4, 4, dtype=torch.float32, device=device)
    else:
        dots = sq = pre = post_n = comb_n = None
    torch.ops.v41k.hc_transition(
        residual,
        y,
        post.contiguous(),
        comb.contiguous(),
        collapse_pre.contiguous(),
        norm_weight,
        out,
        cbuf,
        csq,
        fn,
        scale,
        base,
        dots,
        sq,
        pre,
        post_n,
        comb_n,
        norm_eps,
        hc_eps,
        iters,
    )
    return out, ((pre, post_n, comb_n) if mix else None)


_V3 = envs.VLLM_V41_HC_V3 == "1"
_V3_GROUPS = int(envs.VLLM_V41_HC_V3_GROUPS)


@triton.jit
def _v3_new_residual(residual, y, post, comb, t, col, H: tl.constexpr):
    base = residual + t * 4 * H
    r0 = tl.load(base + 0 * H + col)
    r1 = tl.load(base + 1 * H + col)
    r2 = tl.load(base + 2 * H + col)
    r3 = tl.load(base + 3 * H + col)
    yv = tl.load(y + t * H + col).to(tl.float32)
    cb = comb + t * 16
    pb = post + t * 4
    n0 = (
        tl.load(cb + 0) * r0
        + tl.load(cb + 4) * r1
        + tl.load(cb + 8) * r2
        + tl.load(cb + 12) * r3
        + tl.load(pb + 0) * yv
    )
    n1 = (
        tl.load(cb + 1) * r0
        + tl.load(cb + 5) * r1
        + tl.load(cb + 9) * r2
        + tl.load(cb + 13) * r3
        + tl.load(pb + 1) * yv
    )
    n2 = (
        tl.load(cb + 2) * r0
        + tl.load(cb + 6) * r1
        + tl.load(cb + 10) * r2
        + tl.load(cb + 14) * r3
        + tl.load(pb + 2) * yv
    )
    n3 = (
        tl.load(cb + 3) * r0
        + tl.load(cb + 7) * r1
        + tl.load(cb + 11) * r2
        + tl.load(cb + 15) * r3
        + tl.load(pb + 3) * yv
    )
    return n0, n1, n2, n3


@triton.jit
def _transition_parts3(
    residual,
    y,
    post,
    comb,
    fn,
    cpre,
    cbuf,
    dots,
    squares,
    csquares,
    MIX: tl.constexpr,
    H: tl.constexpr,
    COLS: tl.constexpr,
    CHUNKS: tl.constexpr,
    ROWS: tl.constexpr,
):
    # v1 arithmetic, but the 24 fn rows are split over program_id(2) and the
    # residual is left untouched here (finish3 stores the identical update).
    t, chunk, grp = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    col = chunk * COLS + tl.arange(0, COLS)
    n0, n1, n2, n3 = _v3_new_residual(residual, y, post, comb, t, col, H)
    if grp == 0:
        pc = cpre + t * 4
        coll = (
            n0 * tl.load(pc + 0)
            + n1 * tl.load(pc + 1)
            + n2 * tl.load(pc + 2)
            + n3 * tl.load(pc + 3)
        )
        tl.store(cbuf + t * H + col, coll)
        tl.store(csquares + t * CHUNKS + chunk, tl.sum(coll * coll, 0))
        if MIX:
            tl.store(
                squares + t * CHUNKS + chunk,
                tl.sum(n0 * n0 + n1 * n1 + n2 * n2 + n3 * n3, 0),
            )
    if MIX:
        for i in tl.static_range(ROWS):
            r = grp * ROWS + i
            w = fn + r * 4 * H
            d = (
                tl.load(w + 0 * H + col) * n0
                + tl.load(w + 1 * H + col) * n1
                + tl.load(w + 2 * H + col) * n2
                + tl.load(w + 3 * H + col) * n3
            )
            tl.store(dots + (t * CHUNKS + chunk) * 24 + r, tl.sum(d, 0))


@triton.jit
def _transition_finish3(
    residual,
    y,
    post_in,
    comb_in,
    dots,
    squares,
    csquares,
    cbuf,
    norm,
    out,
    scale,
    base,
    pre,
    post,
    comb,
    MIX: tl.constexpr,
    H: tl.constexpr,
    COLS: tl.constexpr,
    CHUNKS: tl.constexpr,
    NORM_EPS: tl.constexpr,
    HC_EPS: tl.constexpr,
    ITERS: tl.constexpr,
):
    t, chunk = tl.program_id(0), tl.program_id(1)
    col = chunk * COLS + tl.arange(0, COLS)
    n0, n1, n2, n3 = _v3_new_residual(residual, y, post_in, comb_in, t, col, H)
    rb = residual + t * 4 * H
    total = tl.full((), 0, tl.float32)
    for c in tl.static_range(CHUNKS):
        total += tl.load(csquares + t * CHUNKS + c)
    inv = tl.rsqrt(total / H + NORM_EPS)
    v = tl.load(cbuf + t * H + col) * inv * tl.load(norm + col)
    tl.store(out + t * H + col, v.to(tl.float16))
    if MIX:  # noqa: SIM102
        if chunk == 0:
            i = tl.arange(0, 4)
            sq = tl.full((), 0, tl.float32)
            for c in tl.static_range(CHUNKS):
                sq += tl.load(squares + t * CHUNKS + c)
            rs = tl.rsqrt(sq / (4 * H) + NORM_EPS)
            p = tl.zeros((4,), tl.float32)
            q = tl.zeros((4,), tl.float32)
            location = i[:, None] * 4 + i[None, :]
            m = tl.zeros((4, 4), tl.float32)
            for c in tl.static_range(CHUNKS):
                row = dots + (t * CHUNKS + c) * 24
                p += tl.load(row + i)
                q += tl.load(row + 4 + i)
                m += tl.load(row + 8 + location)
            p = p * rs * tl.load(scale) + tl.load(base + i)
            q = q * rs * tl.load(scale + 1) + tl.load(base + 4 + i)
            tl.store(pre + t * 4 + i, tl.sigmoid(p) + HC_EPS)
            tl.store(post + t * 4 + i, 2.0 * tl.sigmoid(q))
            value = m * rs * tl.load(scale + 2) + tl.load(base + 8 + location)
            value = tl.exp(value - tl.max(value, 1)[:, None])
            value = value / tl.sum(value, 1)[:, None] + HC_EPS
            value = value / (tl.sum(value, 0)[None, :] + HC_EPS)
            for _ in range(ITERS - 1):
                value = value / (tl.sum(value, 1)[:, None] + HC_EPS)
                value = value / (tl.sum(value, 0)[None, :] + HC_EPS)
            tl.store(comb + t * 16 + location, value)
    # Every program has read its residual chunk (n above) before storing it.
    tl.store(rb + 0 * H + col, n0)
    tl.store(rb + 1 * H + col, n1)
    tl.store(rb + 2 * H + col, n2)
    tl.store(rb + 3 * H + col, n3)


def _hc_transition3(
    residual,
    y,
    post,
    comb,
    collapse_pre,
    norm_weight,
    fn,
    scale,
    base,
    norm_eps,
    hc_eps,
    iters,
):
    tokens = residual.shape[0]
    mix = fn is not None
    device = residual.device
    cbuf = torch.empty(tokens, H, dtype=torch.float32, device=device)
    csq = torch.empty(tokens, CHUNKS, dtype=torch.float32, device=device)
    if mix:
        dots = torch.empty(tokens, CHUNKS, 24, dtype=torch.float32, device=device)
        sq = torch.empty(tokens, CHUNKS, dtype=torch.float32, device=device)
        pre = torch.empty(tokens, 4, dtype=torch.float32, device=device)
        post_n = torch.empty_like(pre)
        comb_n = torch.empty(tokens, 4, 4, dtype=torch.float32, device=device)
    else:
        dots = sq = pre = post_n = comb_n = csq
    out = torch.empty(tokens, H, dtype=torch.float16, device=device)
    groups = _V3_GROUPS if mix else 1
    _transition_parts3[(tokens, CHUNKS, groups)](
        residual,
        y,
        post,
        comb,
        fn if mix else post,
        collapse_pre,
        cbuf,
        dots,
        sq,
        csq,
        MIX=mix,
        H=H,
        COLS=COLS,
        CHUNKS=CHUNKS,
        ROWS=24 // groups,
        num_warps=4,
        enable_fp_fusion=False,
    )
    _transition_finish3[(tokens, CHUNKS)](
        residual,
        y,
        post,
        comb,
        dots,
        sq,
        csq,
        cbuf,
        norm_weight,
        out,
        scale if mix else csq,
        base if mix else csq,
        pre,
        post_n,
        comb_n,
        MIX=mix,
        H=H,
        COLS=COLS,
        CHUNKS=CHUNKS,
        NORM_EPS=norm_eps,
        HC_EPS=hc_eps,
        ITERS=iters,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out, ((pre, post_n, comb_n) if mix else None)


# --- VLLM_V41_FUSE_TRANSITION: parts3 + finish3 in one launch -----------------
#
# Same grid and phase-1 code as _transition_parts3. Instead of a second launch,
# CTAs take tickets on small per-device counters and the last arrival does the
# work finish3 did, with finish3's arithmetic and summation order:
#   * last of the GROUPS CTAs of (token, chunk): stores the updated residual
#     chunk (every group has read the old one by then);
#   * last of the CHUNKS group-0 CTAs of a token: sums csquares in order and
#     writes the whole normalized FP16 collapse row;
#   * last of the CHUNKS chunk completions of a token (MIX): the mixes and the
#     Sinkhorn, i.e. finish3's chunk==0 block.
# Each counter is reset by its last arrival, so it is zero between launches
# (CUDA-graph replays included). Counters live outside any graph pool.

_FUSE_TRANSITION = envs.VLLM_V41_FUSE_TRANSITION == "1"
_FT_TOKENS = 8
_FT_NORM_AT = _FT_TOKENS * CHUNKS  # per-(token, chunk) slots come first
_FT_MIX_AT = _FT_NORM_AT + _FT_TOKENS
_FT_SLOTS = _FT_MIX_AT + _FT_TOKENS
# The interpreter runs programs one by one and has no inline PTX.
_FT_FENCE = os.environ.get("TRITON_INTERPRET", "0") != "1"
# Cap at parts3's 32 registers (sm_70: 16 CTAs/SM, so 6 x 20 x 8 = 960 CTAs
# stay one wave); ptxas fits the finish paths in 32 without spills.
_FT_MAXNREG = 32
_FT_COUNTERS: dict = {}
_FT_LOGGED = False
_FT_CAPTURE_WARNED = False


@triton.jit
def _ft_fence(FENCE: tl.constexpr):
    if FENCE:
        tl.inline_asm_elementwise(
            "fence.acq_rel.gpu; mov.u32 $0, 0;",
            "=r",
            [],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )


@triton.jit
def _ft_arrive(counter, FENCE: tl.constexpr):
    # Every thread releases its stores, then one ticket per CTA.
    _ft_fence(FENCE)
    tl.debug_barrier()
    return tl.atomic_add(counter, 1, sem="acq_rel", scope="gpu")


@triton.jit
def _ft_take(counter, FENCE: tl.constexpr):
    # Called by the last arrival: re-arm the counter and acquire.
    tl.atomic_xchg(counter, 0, sem="relaxed", scope="gpu")
    _ft_fence(FENCE)


@triton.jit
def _ft_store_residual(residual, n0, n1, n2, n3, t, col, H: tl.constexpr):
    rb = residual + t * 4 * H
    tl.store(rb + 0 * H + col, n0)
    tl.store(rb + 1 * H + col, n1)
    tl.store(rb + 2 * H + col, n2)
    tl.store(rb + 3 * H + col, n3)


@triton.jit
def _ft_mix(
    counters,
    dots,
    squares,
    scale,
    base,
    pre,
    post,
    comb,
    t,
    H: tl.constexpr,
    CHUNKS: tl.constexpr,
    NORM_EPS: tl.constexpr,
    HC_EPS: tl.constexpr,
    ITERS: tl.constexpr,
    MIX_AT: tl.constexpr,
    FENCE: tl.constexpr,
):
    # One arrival per completed chunk of token t; the last runs finish3's MIX
    # block (same order), reading other CTAs' partials through L2.
    if _ft_arrive(counters + MIX_AT + t, FENCE) == CHUNKS - 1:
        _ft_take(counters + MIX_AT + t, FENCE)
        i = tl.arange(0, 4)
        sq = tl.full((), 0, tl.float32)
        for c in tl.static_range(CHUNKS):
            sq += tl.load(squares + t * CHUNKS + c, cache_modifier=".cg")
        rs = tl.rsqrt(sq / (4 * H) + NORM_EPS)
        p = tl.zeros((4,), tl.float32)
        q = tl.zeros((4,), tl.float32)
        location = i[:, None] * 4 + i[None, :]
        m = tl.zeros((4, 4), tl.float32)
        for c in tl.static_range(CHUNKS):
            row = dots + (t * CHUNKS + c) * 24
            p += tl.load(row + i, cache_modifier=".cg")
            q += tl.load(row + 4 + i, cache_modifier=".cg")
            m += tl.load(row + 8 + location, cache_modifier=".cg")
        p = p * rs * tl.load(scale) + tl.load(base + i)
        q = q * rs * tl.load(scale + 1) + tl.load(base + 4 + i)
        tl.store(pre + t * 4 + i, tl.sigmoid(p) + HC_EPS)
        tl.store(post + t * 4 + i, 2.0 * tl.sigmoid(q))
        value = m * rs * tl.load(scale + 2) + tl.load(base + 8 + location)
        value = tl.exp(value - tl.max(value, 1)[:, None])
        value = value / tl.sum(value, 1)[:, None] + HC_EPS
        value = value / (tl.sum(value, 0)[None, :] + HC_EPS)
        for _ in range(ITERS - 1):
            value = value / (tl.sum(value, 1)[:, None] + HC_EPS)
            value = value / (tl.sum(value, 0)[None, :] + HC_EPS)
        tl.store(comb + t * 16 + location, value)


@triton.jit
def _transition_fused3(
    residual,
    y,
    post_in,
    comb_in,
    fn,
    cpre,
    cbuf,
    dots,
    squares,
    csquares,
    counters,
    norm,
    out,
    scale,
    base,
    pre,
    post,
    comb,
    MIX: tl.constexpr,
    H: tl.constexpr,
    COLS: tl.constexpr,
    CHUNKS: tl.constexpr,
    ROWS: tl.constexpr,
    GROUPS: tl.constexpr,
    NORM_EPS: tl.constexpr,
    HC_EPS: tl.constexpr,
    ITERS: tl.constexpr,
    NORM_AT: tl.constexpr,
    MIX_AT: tl.constexpr,
    FENCE: tl.constexpr,
):
    t, chunk, grp = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    col = chunk * COLS + tl.arange(0, COLS)
    n0, n1, n2, n3 = _v3_new_residual(residual, y, post_in, comb_in, t, col, H)
    # Phase 1: _transition_parts3, unchanged.
    if grp == 0:
        pc = cpre + t * 4
        coll = (
            n0 * tl.load(pc + 0)
            + n1 * tl.load(pc + 1)
            + n2 * tl.load(pc + 2)
            + n3 * tl.load(pc + 3)
        )
        tl.store(cbuf + t * H + col, coll)
        tl.store(csquares + t * CHUNKS + chunk, tl.sum(coll * coll, 0))
        if MIX:
            tl.store(
                squares + t * CHUNKS + chunk,
                tl.sum(n0 * n0 + n1 * n1 + n2 * n2 + n3 * n3, 0),
            )
    if MIX:
        for i in tl.static_range(ROWS):
            r = grp * ROWS + i
            w = fn + r * 4 * H
            d = (
                tl.load(w + 0 * H + col) * n0
                + tl.load(w + 1 * H + col) * n1
                + tl.load(w + 2 * H + col) * n2
                + tl.load(w + 3 * H + col) * n3
            )
            tl.store(dots + (t * CHUNKS + chunk) * 24 + r, tl.sum(d, 0))
    # Residual chunk: written once every group has read the old values.
    if GROUPS == 1:
        _ft_store_residual(residual, n0, n1, n2, n3, t, col, H)
        if MIX:
            _ft_mix(
                counters,
                dots,
                squares,
                scale,
                base,
                pre,
                post,
                comb,
                t,
                H,
                CHUNKS,
                NORM_EPS,
                HC_EPS,
                ITERS,
                MIX_AT,
                FENCE,
            )
    else:
        slot = counters + t * CHUNKS + chunk
        if _ft_arrive(slot, FENCE) == GROUPS - 1:
            _ft_take(slot, FENCE)
            _ft_store_residual(residual, n0, n1, n2, n3, t, col, H)
            if MIX:
                _ft_mix(
                    counters,
                    dots,
                    squares,
                    scale,
                    base,
                    pre,
                    post,
                    comb,
                    t,
                    H,
                    CHUNKS,
                    NORM_EPS,
                    HC_EPS,
                    ITERS,
                    MIX_AT,
                    FENCE,
                )
    # Normalized collapse: finish3's sum order and expression, all chunks.
    # (Nested on purpose: only group-0 CTAs may take a ticket.)
    if grp == 0:  # noqa: SIM102
        if _ft_arrive(counters + NORM_AT + t, FENCE) == CHUNKS - 1:
            _ft_take(counters + NORM_AT + t, FENCE)
            total = tl.full((), 0, tl.float32)
            for c in tl.static_range(CHUNKS):
                total += tl.load(csquares + t * CHUNKS + c, cache_modifier=".cg")
            inv = tl.rsqrt(total / H + NORM_EPS)
            for c in tl.static_range(CHUNKS):
                cc = c * COLS + tl.arange(0, COLS)
                v = tl.load(cbuf + t * H + cc, cache_modifier=".cg") * inv
                v = v * tl.load(norm + cc)
                tl.store(out + t * H + cc, v.to(tl.float16))


def _ft_counters(device):
    """Zeroed int32 tickets for one device, allocated outside graph capture."""
    key = (device.type, device.index)
    counters = _FT_COUNTERS.get(key)
    if counters is None:
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            return None
        counters = torch.zeros(_FT_SLOTS, dtype=torch.int32, device=device)
        _FT_COUNTERS[key] = counters
    return counters


def _hc_transition_fused(
    residual,
    y,
    post,
    comb,
    collapse_pre,
    norm_weight,
    fn,
    scale,
    base,
    norm_eps,
    hc_eps,
    iters,
):
    global _FT_LOGGED, _FT_CAPTURE_WARNED
    tokens = residual.shape[0]
    device = residual.device
    counters = _ft_counters(device)
    if counters is None:
        if not _FT_CAPTURE_WARNED:
            _FT_CAPTURE_WARNED = True
            logger.warning(
                "VLLM_V41_FUSE_TRANSITION=1: first use is inside CUDA-graph "
                "capture, so this graph keeps the two-kernel HC transition. "
                "Run one eager decode step before capture to enable it."
            )
        return _hc_transition3(
            residual,
            y,
            post,
            comb,
            collapse_pre,
            norm_weight,
            fn,
            scale,
            base,
            norm_eps,
            hc_eps,
            iters,
        )
    mix = fn is not None
    groups = _V3_GROUPS if mix else 1
    if not _FT_LOGGED:
        _FT_LOGGED = True
        logger.info(
            "VLLM_V41_FUSE_TRANSITION=1: HC transition parts3+finish3 fused "
            "into one launch (last-arrival finish, groups=%d).",
            _V3_GROUPS,
        )
    cbuf = torch.empty(tokens, H, dtype=torch.float32, device=device)
    csq = torch.empty(tokens, CHUNKS, dtype=torch.float32, device=device)
    if mix:
        dots = torch.empty(tokens, CHUNKS, 24, dtype=torch.float32, device=device)
        sq = torch.empty(tokens, CHUNKS, dtype=torch.float32, device=device)
        pre = torch.empty(tokens, 4, dtype=torch.float32, device=device)
        post_n = torch.empty_like(pre)
        comb_n = torch.empty(tokens, 4, 4, dtype=torch.float32, device=device)
    else:
        dots = sq = pre = post_n = comb_n = csq
    out = torch.empty(tokens, H, dtype=torch.float16, device=device)
    _transition_fused3[(tokens, CHUNKS, groups)](
        residual,
        y,
        post,
        comb,
        fn if mix else post,
        collapse_pre,
        cbuf,
        dots,
        sq,
        csq,
        counters,
        norm_weight,
        out,
        scale if mix else csq,
        base if mix else csq,
        pre,
        post_n,
        comb_n,
        MIX=mix,
        H=H,
        COLS=COLS,
        CHUNKS=CHUNKS,
        ROWS=24 // groups,
        GROUPS=groups,
        NORM_EPS=norm_eps,
        HC_EPS=hc_eps,
        ITERS=iters,
        NORM_AT=_FT_NORM_AT,
        MIX_AT=_FT_MIX_AT,
        FENCE=_FT_FENCE,
        num_warps=4,
        enable_fp_fusion=False,
        maxnreg=_FT_MAXNREG,
    )
    return out, ((pre, post_n, comb_n) if mix else None)
