"""BitTensor and related generic Boolean tensor operations."""

from enum import Enum
import math
from typing import cast, Any, TypeAlias, Union

import torch

import bitwise2_ext_cpu


class BitTensor:
    """A bit-packed Boolean tensor."""

    __slots__ = ("_bit_length", "_data")

    _bit_length: int
    _data: torch.Tensor

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, BitTensor)
            and self._bit_length == other._bit_length
            and torch.equal(self._data, other._data)
        )

    def __getitem__(self, index: int) -> "BitTensor":
        if len(self._data.shape) < 2:
            raise ValueError("bit indexing not supported")
        return BitTensor(self._bit_length, self._data[index])

    def __init__(self, bit_length: int, data: torch.Tensor):
        assert data.dtype == torch.int32
        assert len(data) != 0
        assert data.shape[-1] == (bit_length - 1) // 32 + 1

        self._bit_length = bit_length
        self._data = data

    def __len__(self) -> int:
        return self.shape[0]

    def __repr__(self):
        return self.__str__()

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

    def reduce_and(self, dim: int, keepdim: bool = False) -> "BitTensor":
        """
        Reduce by AND along a specified dimension.

        Args:
            dim: Dimension to reduce over (cannot be the last dimension).
            keepdim: Whether to retain the reduced dimension with size 1.

        Returns:
            A BitTensor containing the result of the bitwise OR reduction.
        """

        if dim < 0 or dim >= self._data.dim() - 1:
            raise ValueError("invalid dimension")

        data = bitwise2_ext_cpu.bitwise_and_reduce(self._data, dim)
        if keepdim:
            data.unsqueeze_(dim)

        return BitTensor(self._bit_length, data)

    def reduce_or(self, dim: int, keepdim: bool = False) -> "BitTensor":
        """
        Reduce by OR along a specified dimension.

        Args:
            dim: Dimension to reduce over (cannot be the last dimension).
            keepdim: Whether to retain the reduced dimension with size 1.

        Returns:
            A BitTensor containing the result of the bitwise OR reduction.
        """

        if dim < 0 or dim >= self._data.dim() - 1:
            raise ValueError("invalid dimension")

        data = bitwise2_ext_cpu.bitwise_or_reduce(self._data, dim)
        if keepdim:
            data.unsqueeze_(dim)

        return BitTensor(self._bit_length, data)

    def sample_random_bit_(self) -> "BitTensor":
        """
        Randomly select one active bit per row.

        Returns:
            The same BitTensor, modified in-place with at most one bit set per row.
        """

        # pylint: disable-msg=too-many-locals,too-many-statements

        def zero_last_n_bits_(tensor: torch.Tensor, n: int) -> torch.Tensor:
            num_words = tensor.shape[-1]
            full_words = n // 32
            remainder = n % 32
            start = max(0, num_words - full_words)
            tensor[..., start:] = 0
            partial_idx = num_words - full_words - 1
            if partial_idx >= 0 and remainder:
                keep_bits = 32 - remainder
                mask = (1 << keep_bits) - 1
                tensor[..., partial_idx] &= mask
            return tensor

        def popcount(tensor: torch.Tensor) -> torch.Tensor:
            x = tensor.clone()
            x += -((x >> 1) & 0x55555555)
            x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
            x += x >> 4
            x &= 0x0F0F0F0F
            x += x >> 8
            x += x >> 16
            x &= 0x3F
            return x

        def sample_random_bit_(tensor: torch.Tensor) -> torch.Tensor:
            device = tensor.device
            shape = tensor.shape
            batch_shape = shape[:-2]
            num_rows = shape[-2]
            num_words = shape[-1]
            flat_batch = math.prod(batch_shape) if batch_shape else 1
            tensor_flat = tensor.reshape(flat_batch, num_rows, num_words)

            # Popcount per word per row
            word_pops = popcount(tensor_flat)

            # Total set bits per row
            total_pops = word_pops.sum(dim=-1)

            # Random index k in [0, total_pops)
            random_k = torch.zeros_like(total_pops, dtype=torch.int64)
            mask = total_pops > 0
            if mask.any():
                r = torch.rand(total_pops[mask].shape, device=device)
                random_k[mask] = torch.floor(r * total_pops[mask].float()).to(
                    torch.int64
                )

            # Cumulative popcounts to find word
            cum_pops = word_pops.cumsum(dim=-1)
            col_mask = cum_pops > random_k.unsqueeze(-1)
            inv_mask = torch.ones_like(col_mask, dtype=torch.uint8) - col_mask.to(
                torch.uint8
            )
            selected_word = torch.argmin(inv_mask, dim=-1)

            # Previous cumulative
            prev_word = (selected_word - 1).clamp(min=0)
            prev_cum = torch.gather(cum_pops, -1, prev_word.unsqueeze(-1)).squeeze(-1)
            prev_cum = torch.where(
                selected_word > 0, prev_cum, torch.zeros_like(prev_cum)
            )

            # Local k within word
            local_k = random_k - prev_cum

            # Get selected word value
            selected_val = torch.gather(
                tensor_flat, -1, selected_word.unsqueeze(-1)
            ).squeeze(-1)

            # Find bit position in word
            bit_indices = torch.arange(32, device=device, dtype=torch.int32)
            bits_set = ((selected_val.unsqueeze(-1) & (1 << bit_indices)) != 0).to(
                torch.int32
            )
            cum_bits = bits_set.cumsum(dim=-1)
            bit_mask = cum_bits >= (local_k + 1).unsqueeze(-1)
            bit_inv_mask = torch.ones_like(bit_mask, dtype=torch.uint8) - bit_mask.to(
                torch.uint8
            )
            selected_bit = torch.argmin(bit_inv_mask, dim=-1)

            # Create bit value
            bit_value = (1 << selected_bit).to(tensor_flat.dtype)

            # Modify in place
            tensor_flat.zero_()
            active_batches, active_rows = torch.nonzero(mask, as_tuple=True)
            if active_batches.numel() > 0:
                active_selected_words = selected_word[active_batches, active_rows]
                active_bit_values = bit_value[active_batches, active_rows]
                tensor_flat[active_batches, active_rows, active_selected_words] = (
                    active_bit_values
                )

            return tensor

        n = self._data.shape[-1] * 32 - self._bit_length
        zero_last_n_bits_(self._data, n)
        sample_random_bit_(self._data)
        return self

    def to_bool_tensor(self) -> torch.Tensor:
        """Construct a corresponding Boolean PyTorch tensor."""

        positions = torch.arange(self._bit_length, device=self._data.device)
        words = positions // 32
        shifts = positions % 32

        batch_shape = self._data.shape[:-1]
        ndim = len(batch_shape)
        words = words.reshape((1,) * ndim + (-1,)).expand(*batch_shape, -1)
        shifts = shifts.reshape((1,) * ndim + (-1,)).expand(*batch_shape, -1)

        selected_words = torch.gather(self._data, -1, words)
        bits = (selected_words.to(torch.int64) >> shifts) & 1
        return bits.to(torch.bool)


BitLiteral: TypeAlias = Union[str, list["BitLiteral"]]
_IntLiteral: TypeAlias = Union[list[int], list["_IntLiteral"]]


class Device(Enum):
    """Supported device types."""

    CPU = "cpu"
    CUDA = "cuda"


def bit_tensor(literal: BitLiteral, device: Device = Device.CPU) -> BitTensor:
    """
    Convert a nested list of bit strings to a BitTensor.

    Parses and pads the bit representation to create a BitTensor on the specified device.

    Args:
        literal: Nested list of bit strings.
        device: Target device for the tensor (default: CPU).

    Returns:
        A BitTensor containing the parsed bit data.
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

        if len(literal) == 0:
            raise ValueError("empty literal")

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


def from_bool_tensor(tensor: torch.Tensor) -> BitTensor:
    """
    Convert a Boolean PyTorch tensor to a BitTensor.

    Args:
        tensor: torch.Tensor with dtype=torch.bool.

    Returns:
        A BitTensor that corresponds to the given tensor.
    """

    if tensor.dtype != torch.bool:
        raise ValueError("unexpected tensor dtype")
    if len(tensor) == 0 or tensor.shape[-1] == 0:
        raise ValueError("scalar or empty tensor")

    shape = tensor.shape
    length = shape[-1]
    num_words = (length + 31) // 32

    indices = torch.arange(length, dtype=torch.int32, device=tensor.device)
    shifts, words = indices % 32, indices // 32

    packed_shape = shape[:-1] + (num_words,)
    packed = torch.zeros(packed_shape, dtype=torch.int32, device=tensor.device)

    words = words.expand(*shape[:-1], -1)
    shifts = shifts.expand_as(tensor)

    packed.scatter_add_(-1, words.to(torch.int64), tensor << shifts)
    return BitTensor(tensor.shape[-1], packed)
