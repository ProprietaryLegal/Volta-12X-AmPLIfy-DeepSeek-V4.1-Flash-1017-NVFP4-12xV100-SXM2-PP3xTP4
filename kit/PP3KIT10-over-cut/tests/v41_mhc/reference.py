# SPDX-License-Identifier: Apache-2.0
"""Deterministic, model-free numerical inputs and independent FP64 oracle."""

from collections.abc import Iterator
from dataclasses import dataclass

import torch

H = 5120
STREAMS = 4
MIXES = 24
ROWS_PER_CHUNK = 4092
MAX_CONTEXT = 524_288
NORM_EPS = 1e-20
HC_EPS = 1e-6
ITERS = 20


@dataclass(frozen=True)
class Inputs:
    residual: torch.Tensor
    y: torch.Tensor
    post: torch.Tensor
    comb: torch.Tensor
    cpre: torch.Tensor
    weight: torch.Tensor
    fn: torch.Tensor
    scale: torch.Tensor
    base: torch.Tensor

    def args(self, residual: torch.Tensor, mix: bool) -> tuple[torch.Tensor, ...]:
        tensors = (residual, self.y, self.post, self.comb, self.cpre, self.weight)
        return tensors + (self.fn, self.scale, self.base) if mix else tensors

    def to(self, device: str) -> "Inputs":
        return Inputs(*(x.to(device) for x in self.args(self.residual, True)))


def make_inputs(
    rows: int,
    *,
    y_dtype: torch.dtype = torch.float16,
    weight_dtype: torch.dtype = torch.float32,
    seed: int = 1588,
) -> Inputs:
    g = torch.Generator().manual_seed(seed)
    residual = torch.randn((rows, STREAMS, H), generator=g)
    y = torch.randn((rows, H), generator=g).to(y_dtype)
    post = 2 * torch.rand((rows, STREAMS), generator=g)
    comb = torch.rand((rows, STREAMS, STREAMS), generator=g).softmax(dim=1)
    cpre = torch.rand((rows, STREAMS), generator=g)
    weight = torch.rand(H, generator=g).mul_(1.125).add_(0.125).to(weight_dtype)
    fn = torch.randn((MIXES, STREAMS * H), generator=g).mul_(0.008)
    scale = torch.tensor([1.0, 0.85, 0.9], dtype=torch.float32)
    base = torch.randn(MIXES, generator=g).mul_(0.4)
    # Positive and negative logits near saturation, with ordinary logits too.
    base[[0, 7, 8, 13, 23]] = torch.tensor([50., -50., -42., 37., -35.])
    # V4.1 residuals can exceed the FP16 range before normalization.
    residual[::11].mul_(2.0**19)
    residual[3::11].mul_(2.0**-28)
    y[3::11].mul_(2.0**-10)
    post[3::11].mul_(2.0**-18)
    # Exact cancellation with a nonzero expanded residual, and one zero row.
    residual[1::13, 1].copy_(-residual[1::13, 0])
    comb[1::13] = torch.eye(STREAMS)
    post[1::13].zero_()
    cpre[1::13] = torch.tensor([0.5, 0.5, 0., 0.])
    residual[2::13].zero_()
    y[2::13].zero_()
    return Inputs(residual, y, post, comb, cpre, weight, fn, scale, base)


def partitions(rows: int, step: int) -> Iterator[slice]:
    for start in range(0, rows, step):
        yield slice(start, min(start + step, rows))


def elementwise_errors(
    original: Inputs,
    residual: torch.Tensor,
    out: torch.Tensor,
    coeffs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    *,
    mixed: bool,
) -> dict[str, tuple[float, float]]:
    """Return (max absolute deviation, max error divided by its limit)."""
    rows = original.residual.shape[0]
    assert residual.dtype == torch.float32 and residual.shape == (rows, STREAMS, H)
    assert out.dtype == torch.float16 and out.shape == (rows, H)
    assert (coeffs is not None) == mixed
    if coeffs is not None:
        for tensor, shape in zip(coeffs, ((rows, 4), (rows, 4), (rows, 4, 4)), strict=True):
            assert tensor.dtype == torch.float32 and tensor.shape == shape
    metrics: dict[str, tuple[float, float]] = {}

    def compare(
        name: str, actual: torch.Tensor, exact: torch.Tensor, limit: torch.Tensor
    ) -> None:
        err = (actual.cpu().double() - exact).abs()
        error, ratio = (float(err.max()), float((err / limit).max()))
        if not torch.isfinite(err).all():
            error = ratio = float("inf")
        previous = metrics.get(name, (0.0, 0.0))
        metrics[name] = max(error, previous[0]), max(ratio, previous[1])

    weight = original.weight.double()
    fn_transposed = original.fn.double().t()
    for selection in partitions(rows, 128):
        inputs = original.residual[selection].double()
        y = original.y[selection].double()
        comb = original.comb[selection].double()
        post = original.post[selection].double()
        update = post[..., None] * y[:, None, :]
        expanded = torch.einsum("tsh,tso->toh", inputs, comb) + update
        envelope = torch.einsum("tsh,tso->toh", inputs.abs(), comb.abs()) + update.abs()
        compare("residual", residual[selection], expanded, 6e-7 * envelope + 2e-7)

        collapse = torch.einsum("toh,to->th", expanded, original.cpre[selection].double())
        normed = collapse * (collapse.square().mean(-1, keepdim=True) + NORM_EPS).rsqrt()
        normed *= weight
        compare(
            "out",
            out[selection],
            normed,
            5e-4 * normed.abs() + 3e-5 * weight.abs() + 2**-25,
        )
        if coeffs is None:
            continue
        flat = expanded.flatten(1)
        dots = flat @ fn_transposed
        dots /= (flat.square().mean(-1, keepdim=True) + NORM_EPS).sqrt()
        p = torch.sigmoid(dots[:, :4] * original.scale[0] + original.base[:4]) + HC_EPS
        q = 2 * torch.sigmoid(dots[:, 4:8] * original.scale[1] + original.base[4:8])
        matrix = (dots[:, 8:] * original.scale[2] + original.base[8:]).reshape(-1, 4, 4)
        matrix = matrix.softmax(dim=-1) + HC_EPS
        matrix /= matrix.sum(dim=-2, keepdim=True) + HC_EPS
        for _ in range(ITERS - 1):
            matrix /= matrix.sum(dim=-1, keepdim=True) + HC_EPS
            matrix /= matrix.sum(dim=-2, keepdim=True) + HC_EPS
        for name, tensor, exact in zip(("pre", "post", "comb"), coeffs, (p, q, matrix), strict=True):
            compare(name, tensor[selection], exact, 2e-5 + 2e-5 * exact.abs())

    return metrics
