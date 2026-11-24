# SPDX-License-Identifier: Apache-2.0
"""Check the standing kit with the actual vLLM imports and L2c shared object."""

import ast
from hashlib import file_digest
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

MODULE = "vllm.model_executor.kernels.mhc.deepseek_v41_fused"
KERNELS = "vllm/model_executor/kernels/mhc"
LIB = "v41mhc/build/v41mhc.so"
LIB_SHA = "247cb22ca1863c4ef2aaac60ad006ddf6c00e3dae521ba615eaa73c9cb1bb3a7"
ENGINE = Path(os.environ["V41_MHC_ENGINE"])
BASELINE = Path(os.environ["V41_MHC_BASELINE"])


def model_import(code, **overrides):
    env = {name: value for name, value in os.environ.items()
           if name not in ("VLLM_V41_MHC_PREFILL", "VLLM_V41_MHC_LIB")}
    return subprocess.run([sys.executable, "-B", "-c", code],
                          env=env | overrides, capture_output=True, text=True, check=False)


CHECK = f"""
import {MODULE} as fused
import os
from pathlib import Path
from hashlib import file_digest
from vllm import envs
from vllm.model_executor.kernels.mhc import deepseek_v41_regs as regs
assert fused._MHC_PREFILL
assert envs.VLLM_V41_MHC_PREFILL == "1"
assert envs.VLLM_V41_MHC_LIB == os.getenv("VLLM_V41_MHC_LIB")
assert Path(regs.path) == expected, regs.path
with expected.open("rb") as library:
    assert file_digest(library, "sha256").hexdigest() == "{LIB_SHA}"
assert regs._expand_norm is fused.register_transition.__globals__["_expand_norm"]
"""


def test_environment_defaults_and_in_tree_registration():
    result = model_import(f"from pathlib import Path\nexpected = Path({str(ENGINE / KERNELS / LIB)!r})\n" + CHECK)
    assert result.returncode == 0, result.stderr


def test_absolute_override_wins(tmp_path):
    lib = tmp_path / "override.so"
    shutil.copyfile(ENGINE / KERNELS / LIB, lib)
    result = model_import(f"from pathlib import Path\nexpected = Path({str(lib)!r})\n" + CHECK,
                          VLLM_V41_MHC_LIB=str(lib))
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("override", ["", "relative.so", "/missing/v41mhc.so"])
def test_invalid_override_refuses_even_with_in_tree_library(override):
    result = model_import(f"import {MODULE}", VLLM_V41_MHC_LIB=override)
    assert result.returncode != 0
    assert "ValueError:" in result.stderr and "VLLM_V41_MHC_LIB" in result.stderr


def test_missing_default_library_names_its_path(tmp_path):
    # Load the real fused module and regs from a copy omitting only the library.
    kernels = tmp_path / "mhc"
    shutil.copytree(ENGINE / KERNELS, kernels, ignore=shutil.ignore_patterns("v41mhc"))
    result = model_import(
        "from vllm.model_executor.kernels import mhc\n"
        f"mhc.__path__ = [{str(kernels)!r}]\n"
        f"import {MODULE}"
    )
    assert result.returncode != 0
    assert f"ValueError: missing in-tree mHC library: {kernels / LIB}" in result.stderr


def test_declared_default():
    source = (ENGINE / "vllm/envs.py").read_text()
    assert 'VLLM_V41_MHC_PREFILL: str = "1"' in source
    tree = ast.parse(source)
    entry, = (value for node in ast.walk(tree) if isinstance(node, ast.Dict)
              for key, value in zip(node.keys, node.values, strict=True)
              if isinstance(key, ast.Constant) and key.value == "VLLM_V41_MHC_PREFILL")
    metadata = {kw.arg: ast.literal_eval(kw.value) for kw in entry.keywords
                if kw.arg in ("declared_default", "effective_default")}
    assert metadata == {"declared_default": "'1'", "effective_default": "'1'"}


def test_fingerprint_and_serving_environment():
    store = "vllm/v1/core/sched/v41_store_files.py"
    assert (ENGINE / store).read_bytes() == (BASELINE / store).read_bytes()
    adopted = Path(os.environ["V41_MHC_ADOPTED"]).read_bytes()
    lines = {line.split(b"=", 1)[0] for line in adopted.splitlines()}
    assert {b"VLLM_V41_MHC_PREFILL", b"VLLM_V41_MHC_LIB"}.isdisjoint(lines)
    result = model_import(f"import {MODULE}\nimport os\n"
                          "assert not any(name.startswith('VLLM_V41_MHC_') for name in os.environ)",
                          **dict(line.decode().split("=", 1) for line in adopted.splitlines()))
    assert result.returncode == 0, result.stderr
