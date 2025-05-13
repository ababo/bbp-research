"""BitTensor and related generic Boolean tensor operations."""

from enum import Enum
from typing import TypeAlias, Union

import torch


class BitTensor:
    """A bit-packed Boolean tensor."""

    __slots__ = ("_bit_length", "_data")

    _bit_length: int
    _data: torch.Tensor

    def __init__(self, bit_length: int, data: torch.Tensor):
        assert data.dtype == torch.int32
        assert data.shape[-1] == (bit_length - 1) // 32 + 1

        self._bit_length = bit_length
        self._data = data

    @property
    def shape(self) -> torch.Size:
        """Return the shape of the tensor."""
        return self._data.shape[:-1] + (self._bit_length,)

    @property
    def data(self) -> torch.Tensor:
        """Return the underlying torch.Tensor."""
        return self._data


BitLiteral: TypeAlias = Union[str, list["BitLiteral"]]


class Device(Enum):
    """Supported device types."""

    CPU = "cpu"
    CUDA = "cuda"


def bit_tensor(literal: BitLiteral, device=Device.CPU) -> BitTensor:
    """Convert a nested list of bit strings to a BitTensor.

    Parses and pads the bit representation to create a BitTensor on the specified device.

    Args:
        literal: Nested list of bit strings.
        device: Target device for the tensor (default: CPU).

    Returns:
        A BitTensor containing the parsed bit data.

    Raises:
        ValueError: If bit strings contain invalid characters or have inconsistent lengths.
    """

    def parse_bits(bits: str) -> tuple[list[int], int]:
        bits = bits.replace("_", "")
        if not all(c in "01" for c in bits):
            raise ValueError("malformed bits (expected only 1 or 0)")

        padding = (32 - len(bits) % 32) % 32
        padded_bits = bits + "0" * padding
        return [
            int(padded_bits[i : i + 32][::-1], 2)
            for i in range(0, len(padded_bits), 32)
        ], len(bits)

    IntLiteral: TypeAlias = list[Union[int, "IntLiteral"]]

    def parse_literal(literal: BitLiteral) -> Union[IntLiteral, int]:
        if isinstance(literal, str):
            return parse_bits(literal)

        result, bit_length = [], None
        for part in literal:
            int_part, length = parse_literal(part)
            if bit_length is not None and length != bit_length:
                raise ValueError("inconsistent bit length")
            bit_length = length
            result.append(int_part)

        return result, bit_length

    int_literal, bit_length = parse_literal(literal)
    data = torch.tensor(int_literal, dtype=torch.uint32, device=device.value).view(
        torch.int32
    )
    return BitTensor(bit_length, data)
