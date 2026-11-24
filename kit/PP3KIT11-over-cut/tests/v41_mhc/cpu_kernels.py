# SPDX-License-Identifier: Apache-2.0
"""Use the loaded CUDA library's schemas with a test-only CPU implementation."""

import torch

from reference import H, STREAMS, partitions


def _ordered_products(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    result = values[:, 0] * weights[:, 0, None]
    for i in range(1, STREAMS):
        result += values[:, i] * weights[:, i, None]
    return result


def _tree_sum(values: torch.Tensor) -> torch.Tensor:
    # CUDA shuffle-down order in the first warp, then among the 8 warp totals.
    while values.shape[-1] > 1:
        half = values.shape[-1] // 2
        values = values[..., :half] + values[..., half:]
    return values[..., 0]


def _tile_sum(values: torch.Tensor) -> torch.Tensor:
    rows, columns = values.shape
    pair = values.reshape(rows, columns // 256, 4, 32, 2).sum(dim=-1)
    return _tree_sum(_tree_sum(pair))


def expand_norm(
    out: torch.Tensor,
    residual: torch.Tensor,
    y: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    collapse_pre: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> None:
    vector_inputs = (residual, y, weight)
    if any(t.data_ptr() % (4 * t.element_size()) for t in vector_inputs):
        raise ValueError("v41mhc: inputs must be aligned to four elements")
    for block in partitions(residual.size(0), 128):
        current = residual[block]
        expanded = torch.stack(
            [
                _ordered_products(current, comb[block, :, dst])
                + post[block, dst, None] * y[block].float()
                for dst in range(STREAMS)
            ],
            dim=1,
        )
        current.copy_(expanded)
        collapse = _ordered_products(expanded, collapse_pre[block])
        tiles = _tile_sum(collapse.square())
        square = torch.zeros(tiles.shape[0], dtype=torch.float32)
        for tile in tiles.unbind(dim=1):
            square += tile
        out[block].copy_(collapse * (square[:, None] / H + eps).rsqrt() * weight.float())


def coefficient_partials(
    residual: torch.Tensor, fn: torch.Tensor, dots: torch.Tensor, squares: torch.Tensor
) -> None:
    weights = fn.reshape(24, STREAMS, H)
    for block in partitions(residual.size(0), 128):
        x = residual[block]
        squares[block].copy_(_tile_sum(
            x[:, 0].square() + x[:, 1].square() + x[:, 2].square() + x[:, 3].square()
        ))
        for mix, w in enumerate(weights):
            products = x * w
            ordered = products[:, 0] + products[:, 1] + products[:, 2] + products[:, 3]
            dots[block, :, mix].copy_(_tile_sum(ordered))
