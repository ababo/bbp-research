"""Build script for extensions."""

from setuptools import Extension, setup
from torch import cuda

from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,  # type: ignore
    CUDAExtension,  # type: ignore
)


ext_modules: list[Extension] = [
    CppExtension(
        name="bitwise2_ext_cpu",
        sources=[
            "src/bitwise2/ext_cpu/bitwise_or.cc",
        ],
    ),
]

if cuda.is_available():
    ext_modules.append(
        CUDAExtension(
            name="bitwise2_ext_cuda",
            sources=[
                "src/bitwise2/ext_cuda/bitwise_or.cu",
            ],
        )
    )
else:
    print("CUDA not found, only building CPU extensions.")

setup(
    name="ext",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
