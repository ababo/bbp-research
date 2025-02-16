import bitwise
import pytest
import torch


def test_tensor():
    with pytest.raises(ValueError):
        bitwise.tensor("100a1")

    with pytest.raises(ValueError):
        bitwise.tensor(["1001", "111"])

    t = bitwise.tensor(
        [
            "00000000000000000000000000000001_00000000000000000000000000000010_00000000000000000000000000000011",
            "00000000000000000000000000000100_00000000000000000000000000000101_00000000000000000000000000000110",
        ]
    )
    assert torch.equal(t, torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32))

    t = bitwise.tensor(
        [
            [
                ["00000000000000000000000000000001_00000000000000000000000000000010"],
                ["00000000000000000000000000000011_00000000000000000000000000000100"],
                ["00000000000000000000000000000101_00000000000000000000000000000110"],
            ]
        ]
    )
    assert torch.equal(
        t, torch.tensor([[[[1, 2]], [[3, 4]], [[5, 6]]]], dtype=torch.int32)
    )

    t = bitwise.tensor("11111111111111111111111111111111")
    assert torch.equal(t, torch.tensor([-1], dtype=torch.int32))


def test_pack():
    # fmt: off
    tensor = torch.tensor(
        [
            [0,2,1,1,0,0,1,1,1,0,0,0,1,0,0,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,1,1,0,0,0,1,],
            [0,1,2,1,0,0,1,0,1,0,0,0,1,0,0,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,1,1,0,0,1,1,],
            [0,1,1,2,0,0,1,1,0,0,0,0,1,0,0,1,1,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,0,1,0,0,],
            [0,1,1,1,0,0,2,1,1,0,0,0,1,0,0,1,1,1,0,1,0,0,0,1,1,0,0,0,1,1,1,1,0,1,0,1,]
        ],
        dtype=torch.int32
    )
    # fmt: on

    packed_tensor = bitwise.pack(tensor)
    expected_tensor = torch.tensor(
        [
            [0b01110011100010011111000110001111, 0b00010000000000000000000000000000],
            [0b01110010100010011111000110001111, 0b00110000000000000000000000000000],
            [0b01110011000010011011000110001111, 0b01000000000000000000000000000000],
            [0b01110011100010011101000110001111, 0b01010000000000000000000000000000],
        ],
        dtype=torch.int32,
    )
    assert torch.equal(packed_tensor, expected_tensor)

    tensor = torch.cat([tensor, tensor, tensor])
    packed_tensor = bitwise.pack(tensor)
    expected_tensor = torch.cat([expected_tensor, expected_tensor, expected_tensor])
    assert torch.equal(packed_tensor, expected_tensor)
