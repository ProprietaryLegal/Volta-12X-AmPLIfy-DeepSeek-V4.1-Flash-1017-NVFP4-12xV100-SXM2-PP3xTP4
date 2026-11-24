# SPDX-License-Identifier: Apache-2.0
import ast
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

import cpu_kernels
from reference import MAX_CONTEXT, ROWS_PER_CHUNK, elementwise_errors, make_inputs
from vllm.model_executor.kernels.mhc import deepseek_v41_fused as current

BASELINE = (
    Path(os.environ["V41_MHC_BASELINE"])
    / "vllm/model_executor/kernels/mhc/deepseek_v41_fused.py"
)


def _load(path: Path, suffix: str) -> ModuleType:
    name = current.__name__ + "_" + suffix
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def baseline():
    return _load(BASELINE, "pristine_9z")


@pytest.fixture(scope="module")
def candidate():
    with pytest.MonkeyPatch.context() as patch:
        patch.delenv("VLLM_V41_MHC_PREFILL", raising=False)
        patch.delenv("VLLM_V41_MHC_LIB", raising=False)
        module = _load(Path(current.__file__), "enabled_test")
    dispatch = torch.library.Library("v41mhc", "IMPL", "CPU")
    dispatch.impl("expand_norm", cpu_kernels.expand_norm)
    dispatch.impl("coefficient_partials", cpu_kernels.coefficient_partials)
    yield module


def test_legacy_implementation_unchanged_at_all_widths():
    """The register branch is the sole executable difference from the 9Z module."""
    ours = ast.parse(Path(current.__file__).read_text())
    original = ast.parse(BASELINE.read_text())
    for node in tuple(ours.body):
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "_MHC_PREFILL":
            ours.body.remove(node)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "_MHC_PREFILL":
            ours.body.remove(node)
        if isinstance(node, ast.FunctionDef) and node.name == "_transition_finish":
            assert node.args.args[-1].arg == "COLLAPSE"
            assert ast.literal_eval(node.args.defaults.pop()) is True
            node.args.args.pop()
            collapse = node.body[1]
            assert isinstance(collapse, ast.If) and ast.unparse(collapse.test) == "COLLAPSE"
            assert not collapse.orelse
            node.body[1:2] = collapse.body
        if isinstance(node, ast.FunctionDef) and node.name == "hc_transition":
            branch = next(
                x
                for x in node.body
                if isinstance(x, ast.If)
                and ast.unparse(x.test) == "_MHC_PREFILL and tokens >= MIN_ROWS"
            )
            node.body.remove(branch)
    assert ast.dump(ours) == ast.dump(original)


def _assert_same_bytes(left: torch.Tensor, right: torch.Tensor) -> None:
    assert left.shape == right.shape and left.dtype == right.dtype
    assert torch.equal(left.view(torch.uint8), right.view(torch.uint8))


@pytest.mark.parametrize("rows", [1, 6, 9, 29, 30])
def test_legacy_dispatch_is_byte_exact(baseline, candidate, rows):
    inputs = make_inputs(rows)
    # The narrow widths also exercise a full coefficient update under the
    # interpreter. Wide 9Z reads are checked structurally and in the GPU probe.
    mixed = rows <= 6
    reference_residual = inputs.residual.clone()
    off_residual = inputs.residual.clone()
    assert not current._MHC_PREFILL
    expected, expected_coeff = baseline.hc_transition(*inputs.args(reference_residual, mixed))
    actual, actual_coeff = current.hc_transition(*inputs.args(off_residual, mixed))
    _assert_same_bytes(off_residual, reference_residual)
    _assert_same_bytes(actual, expected)
    if mixed:
        for a, b in zip(actual_coeff, expected_coeff, strict=True):
            _assert_same_bytes(a, b)
    else:
        assert actual_coeff is expected_coeff is None
    if rows < 30:
        on_residual = inputs.residual.clone()
        on, on_coeff = candidate.hc_transition(*inputs.args(on_residual, mixed))
        _assert_same_bytes(on_residual, reference_residual)
        _assert_same_bytes(on, expected)
        if mixed:
            for a, b in zip(on_coeff, expected_coeff, strict=True):
                _assert_same_bytes(a, b)
        else:
            assert on_coeff is None


@pytest.mark.parametrize(
    "rows,y_dtype,weight_dtype,mixed",
    [
        (30, torch.float16, torch.float32, True),
        (30, torch.float16, torch.float16, False),
        (129, torch.float32, torch.float16, True),  # partial 128-row finish grid
        (129, torch.float32, torch.float32, False),
        (2044, torch.float16, torch.float32, True),
        (ROWS_PER_CHUNK, torch.float16, torch.float32, True),
        (ROWS_PER_CHUNK, torch.float16, torch.float32, False),
        (MAX_CONTEXT % ROWS_PER_CHUNK, torch.float16, torch.float32, True),
    ],
)
def test_active_transition_against_independent_double(
    candidate, rows, y_dtype, weight_dtype, mixed
):
    inputs = make_inputs(rows, y_dtype=y_dtype, weight_dtype=weight_dtype)
    residual = inputs.residual.clone()
    ptr = residual.data_ptr()
    out, coeff = candidate.hc_transition(*inputs.args(residual, mixed))
    assert residual.data_ptr() == ptr
    metrics = elementwise_errors(inputs, residual, out, coeff, mixed=mixed)
    print(f"rows={rows} mix={int(mixed)} errors={metrics}")
    assert all(ratio <= 1 for _, ratio in metrics.values()), metrics


def test_bad_mix_fails_before_mutation(candidate):
    inputs = make_inputs(30)
    residual = inputs.residual.clone()
    args = inputs.args(residual, True)
    with pytest.raises(ValueError, match="FP32 scale"):
        candidate.hc_transition(*args[:-2], inputs.scale.half(), inputs.base)
    _assert_same_bytes(residual, inputs.residual)


def test_unaligned_slice_fails_before_mutation(candidate):
    inputs = make_inputs(30)
    storage = torch.empty(inputs.residual.numel() + 1, dtype=torch.float32)
    residual = storage[1:].view_as(inputs.residual)
    residual.copy_(inputs.residual)
    assert residual.is_contiguous()
    with pytest.raises(ValueError, match="aligned"):
        candidate.hc_transition(*inputs.args(residual, True))
    _assert_same_bytes(residual, inputs.residual)


@pytest.mark.parametrize("rows", [30, 129])
def test_cpu_model_differential_against_9z(baseline, candidate, rows):
    """Exercise both hc_transition entry points on the reviewer's fixed seed."""
    inputs = make_inputs(rows)
    expected_residual = inputs.residual.clone()
    actual_residual = inputs.residual.clone()
    expected, expected_coeffs = baseline.hc_transition(*inputs.args(expected_residual, True))
    actual, actual_coeffs = candidate.hc_transition(*inputs.args(actual_residual, True))
    _assert_same_bytes(actual_residual, expected_residual)
    assert torch.isfinite(expected).all() and torch.isfinite(actual).all()
    low = torch.nextafter(expected, torch.full_like(expected, -float("inf")))
    high = torch.nextafter(expected, torch.full_like(expected, float("inf")))
    assert torch.all((low <= actual) & (actual <= high))
    assert actual_coeffs is not None and expected_coeffs is not None
    maxima = []
    for measured, reference in zip(actual_coeffs, expected_coeffs, strict=True):
        error = (measured.double() - reference.double()).abs()
        magnitude = reference.double().abs()
        assert torch.all(error <= 2e-5 * magnitude)
        maxima.append(float(torch.where(error == 0, 0.0, error / magnitude).max()))
    differing = int((actual.view(torch.int16) != expected.view(torch.int16)).sum())
    print(f"MEASURED CPU interpreter rows={rows} residual_byte_equal=True "
          f"out_differences={differing}/{actual.numel()} "
          f"pre_post_comb_relative={maxima}")
