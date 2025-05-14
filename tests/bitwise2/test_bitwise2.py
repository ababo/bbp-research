"""Tests for BitTensor and related generic Boolean tensor operations."""

import pytest
import torch

import bitwise2


def test_bit_tensor():
    """Test bit_tensor()."""

    with pytest.raises(ValueError):
        bitwise2.bit_tensor("100a1")

    with pytest.raises(ValueError):
        bitwise2.bit_tensor(["1001", "111"])

    t = bitwise2.bit_tensor(
        [
            "10000000000000000000000000000000_01000000000000000000000000000000"
            + "11000000000000000000000000000000",
            "00100000000000000000000000000000_10100000000000000000000000000000"
            + "01100000000000000000000000000000",
        ]
    )
    assert torch.equal(t.data, torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32))

    t = bitwise2.bit_tensor(
        [
            [
                ["10000000000000000000000000000000_01000000000000000000000000000000"],
                ["11000000000000000000000000000000_00100000000000000000000000000000"],
                ["10100000000000000000000000000000_01100000000000000000000000000000"],
            ]
        ]
    )
    assert torch.equal(
        t.data, torch.tensor([[[[1, 2]], [[3, 4]], [[5, 6]]]], dtype=torch.int32)
    )

    t = bitwise2.bit_tensor("11111111111111111111111111111111")
    assert torch.equal(t.data, torch.tensor([-1], dtype=torch.int32))


def test_bit_tensor_shape():
    """Test BitTensor.shape."""

    t = bitwise2.bit_tensor("0001010")
    assert t.shape == [7]

    t = bitwise2.bit_tensor(
        [["10000000000000000000000000000000_01", "10000000000000000000000000000000_10"]]
    )
    assert t.shape == [1, 2, 34]
