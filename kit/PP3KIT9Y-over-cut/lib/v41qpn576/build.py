#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build only the packed W2 op from verified serving sources, without CUDA access."""
import argparse
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINS = {
    "skinny_moe_qpn_sm70.cu":
        "31d6ea4fcb310a57276ad26874d644c51f2f277896da3c99d81d744decc8c16b",
    "mma_fragments_sm70.cuh":
        "1d895ebb61fe6fb59ebb21b9f67cb276b1279fa1e9884e7ec264bc9a847a66a2",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine_src", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("build requires CUDA_VISIBLE_DEVICES= (empty)")
    source = args.engine_src / "csrc/sm70_turbomind/ops"
    out = args.output.resolve()
    out.mkdir()
    for name, pin in PINS.items():
        data = (source / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != pin:
            raise ValueError(f"serving source digest mismatch: {name}")
        (out / name).write_bytes(data)
    subprocess.run(
        ["patch", "--batch", "--forward", "--fuzz=0", "-p0",
         "-i", str(HERE / "compact.patch")], cwd=out, check=True,
    )
    generated = out / "v41qpn576.cu"
    generated.write_bytes(
        (out / "skinny_moe_qpn_sm70.cu").read_bytes() + (HERE / "w2.cu").read_bytes()
    )
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0"
    os.environ["MAX_JOBS"] = "1"
    from torch.utils.cpp_extension import load

    path = load(
        name="v41qpn576", sources=[str(generated)], build_directory=str(out),
        extra_cuda_cflags=["-O3", "-Xptxas=-v", "-Xcompiler=-fvisibility=hidden",
                           "--expt-relaxed-constexpr"],
        is_python_module=False, verbose=True,
    )
    library = out / "v41qpn576.so"
    if Path(path).resolve() != library:
        shutil.copyfile(path, library)
    digest = hashlib.sha256(library.read_bytes()).hexdigest()
    library.with_suffix(".sha256").write_text(digest + "\n")
    print(f"RUN built {library} sha256={digest}")


if __name__ == "__main__":
    main()
