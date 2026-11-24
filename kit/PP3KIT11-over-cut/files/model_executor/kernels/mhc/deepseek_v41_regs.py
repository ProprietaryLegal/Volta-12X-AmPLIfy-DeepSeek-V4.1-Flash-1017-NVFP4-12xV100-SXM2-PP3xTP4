# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wide V4.1 mHC transition using the standalone v41mhc SM70 library."""

import math
from pathlib import Path

import torch

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)
MIN_ROWS = 30
_STREAMS = 4
_MIXES = 24

override = envs.VLLM_V41_MHC_LIB
if override is None:
    path = Path(__file__).resolve().parent / "v41mhc" / "build" / "v41mhc.so"
    if not path.is_file():
        raise ValueError(f"missing in-tree mHC library: {path}")
else:
    path = Path(override)
    if not path.is_absolute() or not path.is_file():
        raise ValueError(
            "VLLM_V41_MHC_LIB must name an existing absolute path to v41mhc.so: "
            f"{override!r}"
        )
torch.ops.load_library(str(path))
_expand_norm = torch.ops.v41mhc.expand_norm.default
_coefficient_partials = torch.ops.v41mhc.coefficient_partials.default


def _check_mix_input(
    name: str,
    tensor: torch.Tensor | None,
    shape: tuple[int, ...],
    device: torch.device,
) -> None:
    if (
        tensor is None
        or tensor.shape != shape
        or tensor.device != device
        or tensor.dtype != torch.float32
        or not tensor.is_contiguous()
    ):
        raise ValueError(f"V4.1 register mHC requires contiguous FP32 {name} {shape}")


def register_transition(
    residual: torch.Tensor,
    y: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    collapse_pre: torch.Tensor,
    norm_weight: torch.Tensor,
    fn: torch.Tensor | None,
    scale: torch.Tensor | None,
    base: torch.Tensor | None,
    *,
    norm_eps: float,
    hc_eps: float,
    iters: int,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None]:
    from .deepseek_v41_fused import CHUNKS, H, _transition_finish

    rows = residual.size(0)
    device = residual.device
    if not math.isfinite(norm_eps) or not 0 < norm_eps <= 1:
        raise ValueError("V4.1 register mHC requires 0 < norm_eps <= 1")

    # Validate the coefficient inputs before updating the residual.
    if fn is not None:
        _check_mix_input("fn", fn, (_MIXES, _STREAMS * H), device)
        _check_mix_input("scale", scale, (3,), device)
        _check_mix_input("base", base, (_MIXES,), device)
        if not math.isfinite(hc_eps) or not 0 < hc_eps <= 1 or not 0 < iters < 2**31:
            raise ValueError("V4.1 register mHC requires positive hc_eps and iters")

    out = torch.empty((rows, H), device=device, dtype=torch.float16)
    result = None
    if fn is not None:
        dots = torch.empty((rows, CHUNKS, _MIXES), device=device, dtype=torch.float32)
        squares = torch.empty((rows, CHUNKS), device=device, dtype=torch.float32)
        pre = torch.empty((rows, _STREAMS), device=device, dtype=torch.float32)
        next_post = torch.empty_like(pre)
        next_comb = torch.empty(
            (rows, _STREAMS, _STREAMS), device=device, dtype=torch.float32
        )

    _expand_norm(out, residual, y, post, comb, collapse_pre, norm_weight, norm_eps)
    if fn is not None:
        _coefficient_partials(residual, fn, dots, squares)
        _transition_finish[(rows,)](
            dots, squares, squares, out, norm_weight, out, scale, base,
            pre, next_post, next_comb,
            MIX=True, H=H, CHUNKS=CHUNKS, NORM_EPS=norm_eps,
            HC_EPS=hc_eps, ITERS=iters, BLOCK=1024, COLLAPSE=False,
            num_warps=4, enable_fp_fusion=False,
        )
        result = (pre, next_post, next_comb)
    logger.info_once(
        "VLLM_V41_MHC_PREFILL=1 first dispatch rows=%d",
        rows,
        key="v41mhc-prefill-first-dispatch",
        scope="process",
    )
    return out, result
