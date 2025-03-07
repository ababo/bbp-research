from typing import Union

import numpy as np
import torch

Tensor = torch.Tensor
TensorLiteral = list[Union[str, "TensorLiteral"]]
_TensorIntLiteral = list[Union[int, "_TensorIntLiteral"]]


def mask_rows(tensor: Tensor, mask: Tensor) -> Tensor:
    """Masks rows of a matrix based on a mask."""

    # Validate input dimensions and shapes
    if tensor.ndim == 2:
        m, n = tensor.shape
        assert mask.ndim == 2 and mask.shape[0] == 1 and mask.shape[1] * 32 >= m
    elif tensor.ndim == 3:
        batch_size, m, n = tensor.shape
        assert (
            mask.ndim == 3
            and mask.shape[0] == batch_size
            and mask.shape[1] == 1
            and mask.shape[2] * 32 >= m
        )
    else:
        raise ValueError("tensor must be 2D or 3D")

    # Number of rows to mask
    m = tensor.shape[-2]

    # Generate row indices
    indices = torch.arange(m, device=tensor.device)

    # Compute mask element index (k) and bit position for each row
    k = indices // 32  # Which 32-bit mask element
    bit_positions = 31 - (indices % 32)  # Bit position within the element, reversed

    # Extract the relevant mask values for each row
    if tensor.ndim == 2:
        mask_values = mask[0, k]  # Shape: (m,)
    else:
        mask_values = mask[:, 0, k]  # Shape: (batch_size, m)

    # Extract bits: check if the bit at bit_positions is set
    bit_mask = (
        mask_values & (1 << bit_positions)
    ) != 0  # Shape: (m,) or (batch_size, m)

    # Convert boolean mask to int32 (True -> 1, False -> 0)
    row_mask = bit_mask.to(tensor.dtype)

    # Apply the mask to the tensor
    if tensor.ndim == 2:
        result = tensor * row_mask.unsqueeze(1)  # (m, n) * (m, 1) -> (m, n)
    else:
        result = tensor * row_mask.unsqueeze(
            2
        )  # (batch_size, m, n) * (batch_size, m, 1) -> (batch_size, m, n)

    return result


def pack(tensor: Tensor) -> Tensor:
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


def tensor(literal: TensorLiteral, device=None) -> Tensor:
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


def to_str(tensor: Tensor):
    """Returns a string representation of a PyTorch torch.int32 tensor in binary form."""
    np_array = tensor.numpy().astype(np.uint32)
    np_array = np.vectorize(lambda x: f"{x:032b}")(np_array)
    np_array = np.apply_along_axis(lambda row: "_".join(row), -1, np_array)
    return str(np_array)
