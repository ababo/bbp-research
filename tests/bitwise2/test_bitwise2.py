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


def test_bit_tensor_format():
    """Test BitTensor.format()."""

    t = bitwise2.bit_tensor("10110011100011110000111110000011111100000011111110000000")
    assert (
        t.format(max_width=56)
        == "10110011100011110000111110000011111100000011111110000000"
    )
    assert (
        t.format(max_width=55)
        == "10110011100011110000111110...11111100000011111110000000"
    )
    assert (
        t.format(max_width=54)
        == "1011001110001111000011111...11111100000011111110000000"
    )

    t = bitwise2.bit_tensor(
        [
            [["0000"], ["0001"], ["0010"], ["0011"]],
            [["0100"], ["0101"], ["0110"], ["0111"]],
            [["1000"], ["1001"], ["1010"], ["1011"]],
        ]
    )
    assert (
        t.format(max_height=11)
        == "[[[0000]\n  [0001]\n  [0010]\n  [0011]]\n ...\n [[1000]\n  [1001]\n  [1010]\n  [1011]]]"
    )
    assert (
        t.format(max_height=12)
        == "[[[0000]\n  [0001]\n  [0010]\n  [0011]]\n [[0100]\n  [0101]\n  [0110]\n  [0111]]\n "
        + "[[1000]\n  [1001]\n  [1010]\n  [1011]]]"
    )


def test_bit_tensor_shape():
    """Test BitTensor.shape."""

    t = bitwise2.bit_tensor("0001010")
    assert t.shape == [7]

    t = bitwise2.bit_tensor(
        [["10000000000000000000000000000000_01", "10000000000000000000000000000000_10"]]
    )
    assert t.shape == [1, 2, 34]
