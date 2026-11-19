# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the v41idx3 measuring library (SM70) into ./build/v41idx3.so, the way v41k's build.py builds v41k.so.

Run with the python of the venv whose torch will load it (torch's C++ ABI), on the machine that will run it:
    VENV/bin/python build.py [BUILD_DIR]
No card is needed to build. The library registers torch.ops.v41idx3.index_scores_tc and nothing else.
"""

import os
import sys

from torch.utils import cpp_extension as ce

here = os.path.dirname(os.path.abspath(__file__))
build = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(here, "build")
os.makedirs(build, exist_ok=True)
ce.load(
    name="v41idx3",
    sources=[os.path.join(here, "v41idx3.cu")],
    extra_cuda_cflags=[
        "-O3",
        "-gencode",
        "arch=compute_70,code=sm_70",
        "--expt-relaxed-constexpr",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
    ],
    extra_cflags=["-O3", "-std=c++17"],
    is_python_module=False,
    build_directory=build,
    verbose=False,
)
print("built", os.path.join(build, "v41idx3.so"))
