"""BitTensor and related generic Boolean tensor operations."""

from enum import Enum
from typing import cast, TypeAlias, Union

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

    def __getitem__(self, index: int) -> "BitTensor":
        if len(self._data.shape) < 2:
            raise ValueError("bit indexing not supported")
        return BitTensor(self._bit_length, self._data[index])

    def __len__(self) -> int:
        return self.shape[0]

    def __str__(self) -> str:
        return self.format()

    @property
    def shape(self) -> list[int]:
        """Return the shape of the tensor."""
        return list(self._data.shape[:-1]) + [self._bit_length]

    @property
    def data(self) -> torch.Tensor:
        """Return the underlying torch.Tensor."""
        return self._data

    def format(
        self,
        margin: int = 0,
        max_height: int = 24,
        max_width: int = 80,
    ) -> str:
        """
        Format the tensor as a string with optional indentation and line prefixes.

        Args:
            margin: String prepended to every line.
            max_height: Maximum number of lines to display.
            max_width: Maximum number of characters per line.

        Returns:
            A formatted string representation of the tensor.
        """
        if len(self.shape) == 1:
            gen = (f"{(cast(int, v.item()) & 2**32-1):032b}"[::-1] for v in self._data)
            bits = "".join(gen)[: self._bit_length]
            if len(bits) > max_width:
                ellipsis = "..."
                eff_width = max_width - len(ellipsis)
                bits = (
                    bits[: eff_width // 2] + ellipsis + bits[-((eff_width + 1) // 2) :]
                )
            return bits

        def fmt(item: "BitTensor") -> str:
            return item.format(
                margin=margin + 1,
                max_height=(max_height - 1) // 2,
                max_width=max_width - 2,
            )

        item0 = fmt(self[0])
        item_height = len(item0.splitlines())

        prefix = (margin + 1) * " "
        result: str = "[" + item0

        if item_height * len(self) > max_height:
            for i in range(1, max_height // 2 // item_height):
                result += "\n" + prefix + fmt(self[i])
            result += "\n" + prefix + "..."
            for i in range(len(self) - (max_height + 1) // 2 // item_height, len(self)):
                result += "\n" + prefix + fmt(self[i])
        else:
            for i in range(1, len(self)):
                result += "\n" + prefix + fmt(self[i])

        return result + "]"


BitLiteral: TypeAlias = Union[str, list["BitLiteral"]]
_IntLiteral: TypeAlias = Union[list[int], list["_IntLiteral"]]


class Device(Enum):
    """Supported device types."""

    CPU = "cpu"
    CUDA = "cuda"


def bit_tensor(literal: BitLiteral, device: Device = Device.CPU) -> BitTensor:
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
        if bits == "" or not all(c in "01" for c in bits):
            raise ValueError("malformed or empty bits (expected only 1, 0 or _)")

        padding = (32 - len(bits) % 32) % 32
        padded_bits = bits + "0" * padding
        return [
            int(padded_bits[i : i + 32][::-1], 2)
            for i in range(0, len(padded_bits), 32)
        ], len(bits)

    def parse_literal(literal: BitLiteral) -> tuple[_IntLiteral, int]:
        if isinstance(literal, str):
            return parse_bits(literal)

        result: _IntLiteral = []
        bit_length = -1
        for part in literal:
            int_part, length = parse_literal(part)
            if bit_length not in (-1, length):
                raise ValueError("inconsistent bit length")
            bit_length = length
            result.append(int_part)

        return result, bit_length

    int_literal, bit_length = parse_literal(literal)
    data = torch.tensor(int_literal, dtype=torch.uint32, device=device.value).view(
        torch.int32
    )
    return BitTensor(bit_length, data)
