# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the 9Z exact-coefficient v41mhc.so with the serving Torch ABI."""

import argparse
import hashlib
from pathlib import Path

from torch.utils import cpp_extension

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", nargs="?", type=Path, default=HERE / "build")
    directory = parser.parse_args().build_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    cpp_extension.load(
        name="v41mhc",
        sources=[str(HERE / "v41mhc.cu")],
        extra_cuda_cflags=[
            "-O3",
            "-gencode",
            "arch=compute_70,code=sm_70",
            "-std=c++17",
            "--fmad=false",
            "-Xptxas=-v",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        is_python_module=False,
        build_directory=str(directory),
    )
    library = directory / "v41mhc.so"
    with library.open("rb") as data:
        digest = hashlib.file_digest(data, "sha256").hexdigest()
    print(f"{digest}  {library}")


if __name__ == "__main__":
    main()
