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
    ext_bitwise_or_reduces: list[_BitwiseOrReduceFunction]
    ext_bitwise_or_reduces = [
        _BitwiseOrReduceFunction(
            "ext", "cpu", bitwise2_ext_cpu.bitwise_or_reduce,
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
