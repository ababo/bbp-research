from typing import Union

import torch


class Tensor:
    _data: torch.Tensor


TensorLiteral = list[Union[str, "TensorLiteral"]]
_TensorIntLiteral = list[Union[int, "_TensorIntLiteral"]]


def tensor(literal: TensorLiteral, device=None) -> Tensor:
    def parse_bits(bits: str) -> tuple[list[int], int]:
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
            bit_length = int_part
            result.append(int_part)

        return result, bit_length

    int_literal, _ = parse_literal(literal)
    data = torch.tensor(int_literal, dtype=torch.uint32, device=device)

    result = Tensor()
    result._data = data
    return result
