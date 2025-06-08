"""Tests for operations implemented in native extensions."""

from typing import Callable, Dict

import torch

import bitwise2_ext_cpu

try:
    import bitwise2_ext_cuda  # type: ignore
except ImportError:
    bitwise2_ext_cuda = None


def test_bitwise_or_reduce():
    """Test bitwise_or_reduce()."""

    def slow_bitwise_or_reduce(tensor: torch.Tensor, dim: int) -> torch.Tensor:
        result = tensor
        for i in range(tensor.shape[dim]):
            if i == 0:
                result = tensor.select(dim, i)
            else:
                result = torch.bitwise_or(result, tensor.select(dim, i))
        return result

    functions: Dict[str, Callable[[torch.Tensor, int], torch.Tensor]]
    functions = {"cpu": bitwise2_ext_cpu.bitwise_or_reduce}  # type: ignore
    if bitwise2_ext_cuda:
        functions["cuda"] = bitwise2_ext_cuda.bitwise_or_reduce  # type: ignore

    shape = [10, 20, 30, 40]
    dim = len(shape)
    tensor = 1 << torch.randint(0, 32, shape, dtype=torch.int32)

    expected = [slow_bitwise_or_reduce(tensor, i) for i in range(0, dim)]

    for dev_type, function in functions.items():
        tensor2 = tensor.to(dev_type)
        for i in range(0, dim):
            print(f"device type {dev_type}, dim {i}")
            result = function(tensor2, i)
            assert torch.equal(result, expected[i].to(dev_type))
