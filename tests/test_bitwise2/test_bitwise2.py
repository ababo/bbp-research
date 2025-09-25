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


def test_bit_tensor_sample_random_bit_():
    """Test BitTensor.sample_random_bit_()."""

    d = torch.tensor(
        [
            [
                0b00001000000000000000000000000001,
                0b10000000000000000000000000000000,
                0b11111111111111111111111111111010,
            ],
            [
                0b10000000000000100000000000000000,
                0b00000000000000000000000000000001,
                0b11111111111111111111111111110101,
            ],
        ],
        dtype=torch.uint32,
    ).to(dtype=torch.int32)

    row0_ok = [False, False, False, False, False]
    row1_ok = [False, False, False, False, False]

    for _ in range(100):
        tensor = bitwise2.BitTensor(68, d.clone())
        tensor.sample_random_bit_()

        row0_0 = tensor[0] == bitwise2.bit_tensor(
            "10000000000000000000000000000000_00000000000000000000000000000000_0000",
        )
        row0_1 = tensor[0] == bitwise2.bit_tensor(
            "00000000000000000000000000010000_00000000000000000000000000000000_0000",
        )
        row0_2 = tensor[0] == bitwise2.bit_tensor(
            "00000000000000000000000000000000_00000000000000000000000000000001_0000",
        )
        row0_3 = tensor[0] == bitwise2.bit_tensor(
            "00000000000000000000000000000000_00000000000000000000000000000000_0100",
        )
        row0_4 = tensor[0] == bitwise2.bit_tensor(
            "00000000000000000000000000000000_00000000000000000000000000000000_0001",
        )
        assert row0_0 or row0_1 or row0_2 or row0_3 or row0_4
        row0_ok[0] |= row0_0
        row0_ok[1] |= row0_1
        row0_ok[2] |= row0_2
        row0_ok[3] |= row0_3
        row0_ok[4] |= row0_4

        row1_0 = tensor[1] == bitwise2.bit_tensor(
            "00000000000000000000000000000001_00000000000000000000000000000000_0000",
        )
        row1_1 = tensor[1] == bitwise2.bit_tensor(
            "00000000000000000100000000000000_00000000000000000000000000000000_0000",
        )
        row1_2 = tensor[1] == bitwise2.bit_tensor(
            "00000000000000000000000000000000_10000000000000000000000000000000_0000",
        )
        row1_3 = tensor[1] == bitwise2.bit_tensor(
            "00000000000000000000000000000000_00000000000000000000000000000000_1000",
        )
        row1_4 = tensor[1] == bitwise2.bit_tensor(
            "00000000000000000000000000000000_00000000000000000000000000000000_0010",
        )
        assert row1_0 or row1_1 or row1_2 or row1_3 or row1_4
        row1_ok[0] |= row1_0
        row1_ok[1] |= row1_1
        row1_ok[2] |= row1_2
        row1_ok[3] |= row1_3
        row1_ok[4] |= row1_4

    assert row0_ok == [True, True, True, True, True]
    assert row1_ok == [True, True, True, True, True]


def test_bit_tensor_shape():
    """Test BitTensor.shape."""

    t = bitwise2.bit_tensor("0001010")
    assert t.shape == [7]

    t = bitwise2.bit_tensor(
        [["10000000000000000000000000000000_01", "10000000000000000000000000000000_10"]]
    )
    assert t.shape == [1, 2, 34]


def test_bit_tensor_to_bool_tensor():
    """Test BitTensor.to_bool_tensor()."""

    t = bitwise2.bit_tensor(
        [
            [
                "10110011100011110000111110000011_1111000000",
                "01001100011100001111000001111100_0000111111",
            ],
            [
                "01100111000111100001111100000111_1110000001",
                "10011000111000011110000011111000_0001111110",
            ],
            [
                "11001110001111000011111000001111_1100000010",
                "00110001110000111100000111110000_0011111101",
            ],
        ]
    )

    bt = torch.tensor(
        [
            [
                # fmt: off
                [1,0,1,1,0,0,1,1,1,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,
                 1,1,1,1,0,0,0,0,0,0],
                [0,1,0,0,1,1,0,0,0,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,
                 0,0,0,0,1,1,1,1,1,1],
                # fmt: on
            ],
            [
                # fmt: off
                [0,1,1,0,0,1,1,1,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,
                 1,1,1,0,0,0,0,0,0,1],
                [1,0,0,1,1,0,0,0,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,
                 0,0,0,1,1,1,1,1,1,0],
                # fmt: on
            ],
            [
                # fmt: off
                [1,1,0,0,1,1,1,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,1,
                 1,1,0,0,0,0,0,0,1,0],
                [0,0,1,1,0,0,0,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,
                 0,0,1,1,1,1,1,1,0,1],
                # fmt: on
            ],
        ],
        dtype=torch.bool,
    )

    assert torch.equal(t.to_bool_tensor(), bt)


def test_from_bool_tensor():
    """Test bit_tensor_shape()."""
    t = torch.tensor(
        [
            [
                # fmt: off
                [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,  0,],
                [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,  1,],
                # fmt: on
            ]
        ]
    ).to(dtype=torch.bool)
    assert bitwise2.from_bool_tensor(t) == bitwise2.bit_tensor(
        [
            [
                "01010101010101010101010101010101_0",
                "10101010101010101010101010101010_1",
            ]
        ]
    )
