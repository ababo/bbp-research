"""Miscellaneous utility operations."""

import torch


def map_bit_counts(tensor: torch.Tensor) -> torch.Tensor:
    """Map int32 tensor elements to their bit counts."""

    if tensor.dtype != torch.int32:
        raise ValueError("unexpected argument dtype")

    x = tensor.clone()
    x += -((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x += x >> 4
    x &= 0x0F0F0F0F
    x += x >> 8
    x += x >> 16
    x &= 0x3F
    return x
