# SPDX-License-Identifier: Apache-2.0
"""Exercise the engine's transition entry point with CUDA hidden."""
import os
import subprocess
import sys

import pytest

MODULE = "vllm.model_executor.kernels.mhc.deepseek_v41_fused"


@pytest.mark.parametrize("value", ["", " 1", "01", "2", "-1", "true"])
def test_bad_value_refuses_model_import(value):
    result = subprocess.run(
        [sys.executable, "-c", f"import {MODULE}"],
        env=dict(os.environ, VLLM_V41_MHC_PREFILL=value),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode
    assert "VLLM_V41_MHC_PREFILL" in result.stderr
    assert "Valid options" in result.stderr


def test_off_switch_preserves_legacy_import():
    env = dict(os.environ, VLLM_V41_MHC_PREFILL="0", VLLM_V41_MHC_LIB="/unused/v41mhc.so")
    command = (
        f"import {MODULE}; import sys; "
        "assert 'vllm.model_executor.kernels.mhc.deepseek_v41_regs' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", command], env=env, check=True)


def test_token_blocked_incompatible_with_exact_coefficients():
    env = dict(os.environ, VLLM_V41_MHC_PREFILL="1", VLLM_V41_HC_TOKEN_BLOCKED="1")
    result = subprocess.run(
        [sys.executable, "-c", f"import {MODULE}"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "L2c requires the 9Z windowed route" in result.stderr
