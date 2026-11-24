# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frozen S4 batched wo_a adapter with optional target 4K admission.

The native artifact identity describes its historical M8188 qualification.
The Python adapter independently admits M4092 and M4096 with the 4K switch.

Each worker device/compute-stream owns22MiB: canonical codes4MiB,
canonical statistics2MiB and restored FP16 weights16MiB. The owner is shared
across projections on that stream, never across independent streams. Nothing
is a per-layer weight cache. No runtime compiler or alternative library exists.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

from . import dense_tm

ADMITTED_ROWS = (8188,)
ALGORITHM = -1
OWNER_BYTES = 22 * 1024**2
logger = init_logger(__name__)
_LOGGED = False
_LOGGED_4K_ROWS: set[int] = set()


def _digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


@dataclass(frozen=True)
class VendorFile:
    relative_path: str
    sha256: str


@dataclass(frozen=True)
class NativeArtifact:
    library: Path
    library_sha256: str
    cpp_sha256: str
    metadata_sha256: str
    cublas_version: int
    vendors: tuple[VendorFile, ...]


def _load_artifact() -> NativeArtifact:
    here = Path(__file__).resolve().parent
    metadata = here / "k3s4_blas.json"
    raw = json.loads(metadata.read_text())
    if (
        raw["schema"] != "k3.s4.native_artifact.v1"
        or raw["torch"] != str(torch.__version__)
        or raw["cuda"] != torch.version.cuda
        or raw["algorithm"] != ALGORITHM
        or raw["admitted_rows"] != list(ADMITTED_ROWS)
        or raw["owner_bytes"] != OWNER_BYTES
        or raw["cuda_graph_capture"] != "REFUSED"
    ):
        raise ValueError("Frozen S4 Torch/CUDA/algorithm/admission identity differs")
    library, cpp = here / "k3s4_blas.so", here / "s4_blas.cpp"
    if (
        _digest(library) != raw["library_sha256"]
        or _digest(cpp) != raw["cpp_source_sha256"]
    ):
        raise ValueError("Frozen S4 ELF/source bytes differ")
    torch.ops.load_library(str(library))
    mapped = {
        Path(line.split(None, 5)[5]).resolve()
        for line in Path("/proc/self/maps").read_text().splitlines()
        if len(line.split(None, 5)) == 6 and "libcublas" in line.split(None, 5)[5]
    }
    vendors = []
    for entry in raw["vendor_libraries"]:
        relative = Path(entry["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Frozen S4 vendor path must be within the serving venv")
        actual = (Path(sys.prefix) / relative).resolve()
        if (
            actual not in mapped
            or not entry["byte_identical"]
            or entry["sha256"] != entry["fart_sha256"]
            or _digest(actual) != entry["sha256"]
        ):
            raise ValueError(
                "Actual mapped S4 cuBLAS vendor differs from frozen live bytes"
            )
        vendors.append(VendorFile(str(relative), entry["sha256"]))
    if {Path(v.relative_path).name for v in vendors} != {
        "libcublas.so.12",
        "libcublasLt.so.12",
    }:
        raise ValueError("Frozen S4 requires both observed vendor library identities")
    return NativeArtifact(
        library,
        raw["library_sha256"],
        raw["cpp_source_sha256"],
        _digest(metadata),
        raw["cublas_runtime_version"],
        tuple(vendors),
    )


_ARTIFACT = _load_artifact()


@dataclass(frozen=True)
class ScratchOwner:
    stream: torch.cuda.Stream
    weight: torch.Tensor
    stats: torch.Tensor
    restored: torch.Tensor

    def bytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (self.weight, self.stats, self.restored)
        )


_OWNERS: dict[tuple[torch.device, int], ScratchOwner] = {}


def native_library_path() -> Path:
    return _ARTIFACT.library


def artifact_identity() -> dict[str, object]:
    return {
        "library_sha256": _ARTIFACT.library_sha256,
        "cpp_source_sha256": _ARTIFACT.cpp_sha256,
        "metadata_sha256": _ARTIFACT.metadata_sha256,
        "algorithm": ALGORITHM,
        "admitted_rows": list(ADMITTED_ROWS),
        "owner_bytes": OWNER_BYTES,
        "cublas_runtime_version": _ARTIFACT.cublas_version,
    }


def scratch_state() -> dict[str, object]:
    all_storage = [
        t.untyped_storage().data_ptr()
        for owner in _OWNERS.values()
        for t in (owner.weight, owner.stats, owner.restored)
    ]
    return {
        "owner_count": len(_OWNERS),
        "resident_bytes": sum(o.bytes() for o in _OWNERS.values()),
        "distinct_storage": all(
            len(
                {
                    t.untyped_storage().data_ptr()
                    for t in (o.weight, o.stats, o.restored)
                }
            )
            == 3
            for o in _OWNERS.values()
        ),
        "per_owner_bytes": [o.bytes() for o in _OWNERS.values()],
        "all_owner_storages_distinct": len(set(all_storage)) == len(all_storage),
    }


class SegmentProjection(Protocol):
    codes: torch.Tensor
    stats: torch.Tensor
    segment_high: torch.Tensor
    segment_format: int
    logical_output_size: int
    gguf_tm_k_ld: int
    gguf_tm_q_ld: int


def _project(
    output: torch.Tensor,
    codes: torch.Tensor,
    high: torch.Tensor,
    scales: torch.Tensor,
    k_ld: int,
    q_ld: int,
) -> torch.Tensor:
    global _LOGGED
    four_k = (
        dense_tm.WO_A_BATCHED_4K_ENABLED
        and output.ndim == 3
        and output.shape[0] in dense_tm.WO_A_BATCHED_4K_ROWS
    )
    if four_k:
        dense_tm.validate_wo_a_4k_geometry(output)
    elif output.shape != (8188, 2, 4096) or output.dtype != torch.float16:
        raise ValueError("S4 batched wo_a row count or TP4-rank geometry is not admitted")
    if not output.is_cuda or (k_ld, q_ld) != (131072, 2048):
        raise ValueError("S4 batched wo_a requires CUDA and the TP4-rank Q5_K layout")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("S4 scratch has no CUDA-graph-slot ownership")
    stream = torch.cuda.current_stream(output.device)
    key = (output.device, int(stream.cuda_stream))
    if key not in _OWNERS:
        if torch.cuda.get_device_capability(output.device) != (7, 0):
            raise ValueError("S4 frozen native numerics require qualified SM70")
        owner = ScratchOwner(
            stream,
            torch.empty((4096, 256), dtype=torch.int32, device=output.device),
            torch.empty((128, 2048), dtype=torch.int64, device=output.device),
            torch.empty((4096, 2048), dtype=torch.float16, device=output.device),
        )
        if owner.bytes() != OWNER_BYTES:
            raise AssertionError("S4 owned scratch footprint differs")
        _OWNERS[key] = owner
    owner = _OWNERS[key]
    pointers = {
        t.untyped_storage().data_ptr()
        for t in (owner.weight, owner.stats, owner.restored)
    }
    if any(
        t.untyped_storage().data_ptr() in pointers
        for t in (output, codes, high, scales)
    ):
        raise AssertionError("S4 input/source aliases a private scratch owner")
    if not four_k and not _LOGGED:
        logger.info("V4.1 S4 prefill wo_a native-batched candidate active")
        _LOGGED = True
    torch.ops._C.gguf_dense_restore_canonical_sm70_out(
        owner.weight, owner.stats, codes, high, scales, 1, 4096, 2048
    )
    torch.ops._C.gguf_affine_dequantize_sm70_out(
        owner.restored, owner.weight, owner.stats, 5, 32
    )
    result = torch.ops.k3s4.project(output.contiguous(), owner.restored, 2, ALGORITHM)
    if result.untyped_storage().data_ptr() in {
        t.untyped_storage().data_ptr()
        for t in (owner.weight, owner.stats, owner.restored)
    }:
        raise AssertionError("S4 live output aliases an owned scratch buffer")
    if four_k and output.shape[0] not in _LOGGED_4K_ROWS:
        logger.info(
            "%s=1 first projection rows=%d",
            dense_tm.WO_A_BATCHED_4K_KNOB,
            output.shape[0],
        )
        _LOGGED_4K_ROWS.add(output.shape[0])
    return result


def _fake(
    output: torch.Tensor,
    codes: torch.Tensor,
    high: torch.Tensor,
    scales: torch.Tensor,
    k_ld: int,
    q_ld: int,
) -> torch.Tensor:
    return output.new_empty((output.shape[0], 2048))


direct_register_custom_op(
    op_name="v41_prefill_woa_blas", op_func=_project, fake_impl=_fake
)


def batched_wo_a(projection: SegmentProjection, output: torch.Tensor) -> torch.Tensor:
    """Return owned output; all canonical/FP16 temporaries stay stream-owned."""
    if projection.segment_format != 1 or projection.logical_output_size != 2048:
        raise ValueError("S4 native adapter requires the original Q5_K segment bank")
    return torch.ops.vllm.v41_prefill_woa_blas(
        output,
        projection.codes,
        projection.segment_high,
        projection.stats,
        projection.gguf_tm_k_ld,
        projection.gguf_tm_q_ld,
    )
