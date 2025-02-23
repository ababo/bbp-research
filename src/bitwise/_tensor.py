from typing import Union

import numpy as np
import torch

TensorLiteral = list[Union[str, "TensorLiteral"]]
_TensorIntLiteral = list[Union[int, "_TensorIntLiteral"]]


def tensor(literal: TensorLiteral, device=None) -> torch.Tensor:
    """Converts a nested list of bit strings into a PyTorch
    torch.int32 tensor, parsing and padding the bit representation."""

    def parse_bits(bits: str) -> tuple[list[int], int]:
        bits = bits.replace("_", "")
        if not all(c in "01" for c in bits):
            raise ValueError("malformed bits (expected only 1 or 0)")

        padding = (32 - len(bits) % 32) % 32
        padded_bits = bits + "0" * padding
        return [
            int(padded_bits[i : i + 32], 2) for i in range(0, len(padded_bits), 32)
        ], len(bits)

    def parse_literal(literal: TensorLiteral) -> tuple[_TensorIntLiteral, int]:
        if isinstance(literal, str):
            return parse_bits(literal)

        result, bit_length = [], None
        for part in literal:
            int_part, len = parse_literal(part)
            if bit_length is not None and len != bit_length:
                raise ValueError("inconsistent bit length")
            bit_length = len
            result.append(int_part)

        return result, bit_length

    int_literal, _ = parse_literal(literal)
    return torch.tensor(int_literal, dtype=torch.uint32, device=device).view(
        torch.int32
    )


def pack(tensor: torch.Tensor) -> torch.Tensor:
    """Packs tensor values into bits and returns a PyTorch
    torch.int32 tensor, padding the bit representation."""

    shape = tensor.shape
    length = shape[-1]
    num_words = (length + 31) // 32

    indices = torch.arange(length, dtype=torch.int32, device=tensor.device)
    shifts, words = 31 - indices % 32, indices // 32

    packed_shape = shape[:-1] + (num_words,)
    packed = torch.zeros(packed_shape, dtype=torch.int32, device=tensor.device)

    words = words.expand(*shape[:-1], -1)
    shifts = shifts.expand_as(tensor)

    packed.scatter_add_(-1, words.to(torch.int64), (tensor != 0) << shifts)
    return packed


def to_str(tensor: torch.Tensor):
    """Returns a string representation of a PyTorch torch.int32 tensor in binary form."""
    np_array = tensor.numpy().astype(np.uint32)
    np_array = np.vectorize(lambda x: f"{x:032b}")(np_array)
    np_array = np.apply_along_axis(lambda row: "_".join(row), -1, np_array)
    return str(np_array)
