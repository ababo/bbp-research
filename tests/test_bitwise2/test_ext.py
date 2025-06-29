"""Tests for operations implemented in native extensions."""

from dataclasses import dataclass, field
from typing import Callable, Optional, TypeAlias

import pytest
from pytest_benchmark.fixture import BenchmarkFixture  # type: ignore
import torch

import bitwise2_ext_cpu

try:
    import bitwise2_ext_cuda
except ImportError:
    bitwise2_ext_cuda = None


def _bitwise_or_reduce(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    result = tensor
    for i in range(tensor.shape[dim]):
        if i == 0:
            result = tensor.select(dim, i)
        else:
            result = torch.bitwise_or(result, tensor.select(dim, i))
    return result


_BitwiseOrReduce: TypeAlias = Callable[[torch.Tensor, int], torch.Tensor]
_ErrorProjection: TypeAlias = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class _BitwiseOrReduceFunction:
    name: str
    device_type: str
    function: _BitwiseOrReduce


@dataclass
class _BitwiseOrReduceCase:
    function: _BitwiseOrReduceFunction
    tensor: torch.Tensor
    dim: int
    expected: Optional[torch.Tensor] = field(default=None)
    index: Optional[int] = field(default=-1)

    def __repr__(self) -> str:
        return (
            f"{self.index:03d}-{self.function.name}({self.function.device_type}), "
            + f"shape={list(self.tensor.shape)}, dim={self.dim}"
        )


def _get_ext_bitwise_or_reduces() -> list[_BitwiseOrReduceFunction]:
    ext_bitwise_or_reduces = [
        _BitwiseOrReduceFunction(
            "ext",
            "cpu",
            bitwise2_ext_cpu.bitwise_or_reduce,
        )
    ]
    if bitwise2_ext_cuda:
        ext_bitwise_or_reduces.append(
            _BitwiseOrReduceFunction(
                "ext",
                "cuda",
                # Use CPU version of the C++ extension that calls CUDA via Torch.
                bitwise2_ext_cpu.bitwise_or_reduce,
            )
        )
    return ext_bitwise_or_reduces


def _generate_bitwise_or_reduce_test_cases() -> list[_BitwiseOrReduceCase]:
    shape = [10, 20, 30, 40]
    tensor = 1 << torch.randint(0, 32, shape, dtype=torch.int32)
    return [
        _BitwiseOrReduceCase(
            func,
            tensor.to(device=func.device_type),
            dim,
            _bitwise_or_reduce(tensor, dim),
        )
        for func in _get_ext_bitwise_or_reduces()
        for dim in range(len(shape))
    ]


@pytest.mark.parametrize("case", _generate_bitwise_or_reduce_test_cases(), ids=repr)
def test_bitwise_or_reduce(case: _BitwiseOrReduceCase):
    """Test bitwise2_ext_cpu.bitwise_or_reduce()."""
    assert case.expected is not None
    assert torch.equal(
        case.function.function(case.tensor, case.dim).to(device="cpu"), case.expected
    )


def _get_py_bitwise_or_reduces() -> list[_BitwiseOrReduceFunction]:
    py_bitwise_or_reduces = [_BitwiseOrReduceFunction("py", "cpu", _bitwise_or_reduce)]
    if bitwise2_ext_cuda:
        py_bitwise_or_reduces.append(
            _BitwiseOrReduceFunction("py", "cuda", _bitwise_or_reduce)
        )
    return py_bitwise_or_reduces


def _generate_bitwise_or_reduce_bench_cases() -> list[_BitwiseOrReduceCase]:
    specimen = [10, 20, 30]

    shapes = [
        [s * 10000 if i == j else s for i, s in enumerate(specimen)]
        for j in range(len(specimen))
    ]

    tensors = list(
        map(lambda shape: 1 << torch.randint(0, 32, shape, dtype=torch.int32), shapes)
    )

    cases = [
        _BitwiseOrReduceCase(
            func,
            tensor.to(device=func.device_type),
            dim,
        )
        for func in _get_py_bitwise_or_reduces() + _get_ext_bitwise_or_reduces()
        for dim in range(len(specimen))
        for tensor in tensors
    ]

    cases = sorted(cases, key=lambda c: (c.tensor.shape, c.dim, c.function.device_type))
    for i, case in enumerate(cases):
        case.index = i
    return cases


@pytest.mark.benchmark
@pytest.mark.parametrize("case", _generate_bitwise_or_reduce_bench_cases(), ids=repr)
def test_bitwise_or_reduce_perf(
    benchmark: BenchmarkFixture, case: _BitwiseOrReduceCase
):
    """Benchmark _bitwise_or_reduce() and bitwise2_ext_cpu.bitwise_or_reduce()."""
    benchmark(case.function.function, case.tensor, case.dim)


@dataclass
class _ErrorProjectionFunction:
    name: str
    device_type: str
    function: _ErrorProjection


@dataclass
class _ErrorProjectionCase:
    function: _ErrorProjectionFunction
    sm: torch.Tensor
    e: torch.Tensor
    expected: Optional[torch.Tensor] = field(default=None)
    index: Optional[int] = field(default=-1)

    def __repr__(self) -> str:
        return (
            f"{self.index:03d}-{self.function.name}({self.function.device_type}), "
            + f"shape={list(self.sm.shape)}"
        )


def _get_ext_error_projections() -> list[_ErrorProjectionFunction]:
    ext_error_projections = [
        _ErrorProjectionFunction(
            "ext-cpu",
            "cpu",
            bitwise2_ext_cpu.error_projection,
        )
    ]
    if bitwise2_ext_cuda:
        ext_error_projections.append(
            _ErrorProjectionFunction(
                "ext-cpu",
                "cuda",
                # Use CPU version of the C++ extension that calls CUDA via Torch.
                bitwise2_ext_cpu.error_projection,
            )
        )
        ext_error_projections.append(
            _ErrorProjectionFunction(
                "ext-cuda",
                "cuda",
                bitwise2_ext_cuda.error_projection,
            )
        )
    return ext_error_projections


def _generate_error_projection_test_cases() -> list[_ErrorProjectionCase]:
    # fmt: off
    sm = torch.tensor(
        [
            [
                [0b10110000000000000000000000000000, 0b00100000000000000000000000000000], # 🟢
                [0b00000000000000000000000000000000, 0b00000000000000000000000000000000],
                [0b00011100100000000000000000000001, 0b00100000000011100000000000000000], # 🔴
                [0b00000000000000011011000000000000, 0b00000010000000000000000110000000], # 🟢
                [0b00000001010110000000000000000000, 0b00101000000110000000000000000000], # 🟢
                [0b00000000000000000000000000000000, 0b00000000000000000000000000000000],
                [0b10000000000000000000000000000000, 0b00100000000000000000000000000000], # 🔴
                [0b10110000000001101100000000000100, 0b00111100000000000100000000000000], # 🔴
            ],
            [
                [0b10101010101010101010101010101010, 0b11001100110011001100110011001100], # 🟢
                [0b00000000000000000000000000000000, 0b00000000000000000000000000000000],
                [0b01010101010101010101010101010101, 0b00110011001100110011001100110011], # 🟢
                [0b11111111111111111111111111111100, 0b11111111111111111111111111111010], # 🔴
                [0b00000000000000000000000000000000, 0b00000000000000000000000000000000],
                [0b11111111111111111111111111111111, 0b11111111111111111111111111111111], # 🔴
                [0b01000000000000000000000100000000, 0b00101000000100000100000000010000], # 🔴
                [0b00000000000000000000000000000000, 0b00000000000000000000000000000000],
            ],
            [
                [0b00000000000000000000000000000000, 0b00000000000000000000000000000000],
                [0b00000000000000000000000000000000, 0b00000000000000000000000000000000],
                [0b00000000000000000000000000000000, 0b00000000000000000000000000000000],
                [0b00000000000000000000000000000000, 0b00000000000000000000000000000000],
                [0b00000000000000000000000000000000, 0b00000000000000000000000000000000],
                [0b00000000000000000000000000000000, 0b00000000000000000000000000000000],
                [0b00000000000000000000000000000000, 0b00000000000000000000000000000000],
                [0b00000000000000000000000000000000, 0b00000000000000000000000000000000],
            ],
            [
                [0b11111111111111111111111111111111, 0b11111111111111111111111111111111],  # 🔴
                [0b00000000000000000000000000000000, 0b00000000000000000000000000000000],
                [0b11111111111111111111111111111111, 0b11111111111111111111111111111111],  # 🔴
                [0b00000000000000000000000000000000, 0b00000000000000000000000000000000],
                [0b11111111111111111111111111111111, 0b11111111111111111111111111111111],  # 🔴
                [0b00000000000000000000000000000000, 0b00000000000000000000000000000000],
                [0b11111111111111111111111111111111, 0b11111111111111111111111111111111],  # 🔴
                [0b00000000000000000000000000000000, 0b00000000000000000000000000000000],
            ],
            [
                [0b10000000000000000000000000000000, 0b00000000000000000000000000000000],  # 🟢
                [0b00000000000000000000000000000000, 0b00000000000000000000000000000000],
                [0b01000000000000000000000000000000, 0b00000000000000000000000000000000],  # 🔴
                [0b00100000000000000000000000000000, 0b00000000000000000000000000000000],  # 🟢
                [0b00010000000000000000000000000000, 0b00000000000000000000000000000000],  # 🔴
                [0b00001000000000000000000000000000, 0b00000000000000000000000000000000],  # 🟢
                [0b00000100000000000000000000000000, 0b00000000000000000000000000000000],  # 🔴
                [0b00000010000000000000000000000000, 0b00000000000000000000000000000000],  # 🟢
            ],
            [
                [0b11000000000000000000000000000000, 0b00000000000000000000000000000000],  # 🔴
                [0b00000000000000000000000000000000, 0b00000000000000000000000000000000],
                [0b00110000000000000000000000000000, 0b00000000000000000000000000000000],  # 🔴
                [0b01001000000000000000000000000000, 0b00000000000000000000000000000000],  # 🟢
                [0b10001011000000000000000000000000, 0b00000000000000000000000000000000],  # 🔴
                [0b11010000000000000000000000000000, 0b00000000000000000000000000000000],  # 🟢
                [0b01001011001100000000000000000000, 0b00000000000000000000000000000001],  # 🔴
                [0b01001001000111000000000000000000, 0b00000000000000000000000000000000],  # 🟢
            ],
        ],
        dtype = torch.uint32,
    ).view(dtype=torch.int32)

    e = torch.tensor(
        [
            [0b11000100],
            [0b01101000],
            [0b00000000],
            [0b01010101],
            [0b01010100],
            [0b01010101],
        ],
        dtype=torch.int32,
    )

    result = torch.tensor(
        [
            [0b10011100100000000000000000000001, 0b00100000000011100000000000000000],
            [0b11111111111111111111111111111100, 0b11111111111111111111111111111010],
            [0b00000000000000000000000000000000, 0b00000000000000000000000000000000],
            [0b11111111111111111111111111111111, 0b11111111111111111111111111111111],
            [0b01010100000000000000000000000000, 0b00000000000000000000000000000000],
            [0b11000000000000000000000000000000, 0b00000000000000000000000000000000],
        ],
        dtype=torch.uint32,
    ).view(dtype=torch.int32)
    # fmt: on

    cases = [
        _ErrorProjectionCase(
            func,
            sm.to(device=func.device_type),
            e.to(device=func.device_type),
            result,
        )
        for func in _get_ext_error_projections()
    ]
    return cases


@pytest.mark.parametrize("case", _generate_error_projection_test_cases(), ids=repr)
def test_error_projection(case: _ErrorProjectionCase):
    """Test error_projection()."""
    assert case.expected is not None
    assert torch.equal(
        case.function.function(case.sm, case.e).to(device="cpu"), case.expected
    )


def _generate_error_projection_bench_cases() -> list[_ErrorProjectionCase]:
    specimen = [512, 8, 16]
    factors = [16, 4, 16]

    sm_shapes = [specimen] + [
        [s * factors[i] if i == j else s for i, s in enumerate(specimen)]
        for j in range(len(specimen))
    ]

    sms = list(
        map(
            lambda shape: 1 << torch.randint(0, 32, shape, dtype=torch.int32), sm_shapes
        )
    )

    es = list(
        map(
            lambda shape: torch.randint(
                -(2**31),
                2**31 - 1,
                [shape[0], (shape[1] + 31) // 32],
                dtype=torch.int32,
            ),
            sm_shapes,
        )
    )

    cases = [
        _ErrorProjectionCase(
            func,
            sm.to(device=func.device_type),
            e.to(device=func.device_type),
        )
        for func in _get_ext_error_projections()
        for sm, e in zip(sms, es)
    ]

    cases = sorted(cases, key=lambda c: (c.sm.shape, c.function.device_type))
    for i, case in enumerate(cases):
        case.index = i
    return cases


@pytest.mark.benchmark
@pytest.mark.parametrize("case", _generate_error_projection_bench_cases(), ids=repr)
def test_error_projection_perf(benchmark: BenchmarkFixture, case: _ErrorProjectionCase):
    """Benchmark bitwise2_ext_cpu.bitwise_or_reduce() and bitwise2_ext_cuda.bitwise_or_reduce()."""
    benchmark(case.function.function, case.sm, case.e)
