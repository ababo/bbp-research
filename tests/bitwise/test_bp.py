import bitwise
from bitwise import bp
import torch


def test_row_activation():
    inputs = bitwise.tensor(
        [
            "11010011100010110101101000111001_01100101110110011101100010011010",
            "00101110011010100011110010110100_10101001010100010111000100011110",
            "11100011001101011110100111010001_00011011101001101110001100110101",
            "01010010111100011011001000001111_11001100100010101100101010011000",
        ]
    )

    weights = bitwise.tensor(
        [
            "10101010101010101010101010101010_10101010101010101010101010101010",
            "00101100011101001010010111000110_10011010001001100010011101100101",
            "11001100110011001100110011001100_11001100110011001100110011001100",
            "00011100110010100001011000101110_11100100010110010001110011001010",
        ]
    )

    results = bp.row_activation(inputs, weights)

    expected = bitwise.tensor(["1011", "1111", "1110", "1111"])

    assert torch.equal(results, expected)


def test_activation_sensitivity():
    vectors = bitwise.tensor(
        [
            ["00000000000000000000000000000000_00000000000000000000000000000000"],
            ["11111111111111111111111111111111_11111111111111111111111111111111"],
            ["01010101010101010101010101010101_01010101010101010101010101010101"],
            ["10000000000000000000000000000000_00000000000000000000000000000000"],
            ["11010101010101010101010101010101_01010101010101010101010101010101"],
            ["11000000000000000000000000000000_00000000000000000000000000000000"],
        ]
    )

    weights = bitwise.tensor(
        [
            "00000000000000000000000000000000_00000000000000000000000000000000",
            "11111111111111111111111111111111_11111111111111111111111111111111",
            "10101010101010101010101010101010_10101010101010101010101010101010",
        ]
    )

    results = bp.activation_sensitivity(vectors, weights)

    expected = bitwise.tensor(
        [
            [
                "00000000000000000000000000000000_00000000000000000000000000000000",
                "11111111111111111111111111111111_11111111111111111111111111111111",
                "10101010101010101010101010101010_10101010101010101010101010101010",
            ],
            [
                "00000000000000000000000000000000_00000000000000000000000000000000",
                "00000000000000000000000000000000_00000000000000000000000000000000",
                "00000000000000000000000000000000_00000000000000000000000000000000",
            ],
            [
                "00000000000000000000000000000000_00000000000000000000000000000000",
                "00000000000000000000000000000000_00000000000000000000000000000000",
                "10101010101010101010101010101010_10101010101010101010101010101010",
            ],
            [
                "00000000000000000000000000000000_00000000000000000000000000000000",
                "10000000000000000000000000000000_00000000000000000000000000000000",
                "10000000000000000000000000000000_00000000000000000000000000000000",
            ],
            [
                "00000000000000000000000000000000_00000000000000000000000000000000",
                "00000000000000000000000000000000_00000000000000000000000000000000",
                "10000000000000000000000000000000_00000000000000000000000000000000",
            ],
            [
                "00000000000000000000000000000000_00000000000000000000000000000000",
                "00000000000000000000000000000000_00000000000000000000000000000000",
                "10000000000000000000000000000000_00000000000000000000000000000000",
            ],
        ]
    )

    assert torch.equal(results, expected)

    results = bp.activation_sensitivity(weights, vectors)

    expected = bitwise.tensor(
        [
            [
                "00000000000000000000000000000000_00000000000000000000000000000000",
                "00000000000000000000000000000000_00000000000000000000000000000000",
                "00000000000000000000000000000000_00000000000000000000000000000000",
            ],
            [
                "11111111111111111111111111111111_11111111111111111111111111111111",
                "00000000000000000000000000000000_00000000000000000000000000000000",
                "00000000000000000000000000000000_00000000000000000000000000000000",
            ],
            [
                "01010101010101010101010101010101_01010101010101010101010101010101",
                "00000000000000000000000000000000_00000000000000000000000000000000",
                "01010101010101010101010101010101_01010101010101010101010101010101",
            ],
            [
                "10000000000000000000000000000000_00000000000000000000000000000000",
                "10000000000000000000000000000000_00000000000000000000000000000000",
                "10000000000000000000000000000000_00000000000000000000000000000000",
            ],
            [
                "11010101010101010101010101010101_01010101010101010101010101010101",
                "00000000000000000000000000000000_00000000000000000000000000000000",
                "10000000000000000000000000000000_00000000000000000000000000000000",
            ],
            [
                "11000000000000000000000000000000_00000000000000000000000000000000",
                "00000000000000000000000000000000_00000000000000000000000000000000",
                "10000000000000000000000000000000_00000000000000000000000000000000",
            ],
        ]
    )

    assert torch.equal(results, expected)


def test_error_projection():
    s = bitwise.tensor(
        [
            [
                "10001100000000001000000000000000_00000000000010000000000000000001",
                "10000000100000000000000000000000_00000000000000000001000000000000",
                "00000000000000000000001000000000_00000001000000000000000000000000",
                "00001100000000000010000000000000_00000000000000000000000010000001",
            ],
            [
                "11001100000000001000000000000000_00000000000010000000000000000001",
                "10000000100000000000000000000000_00000000000000000001000000000000",
                "00000000000000000000001000000000_00000001000000000000000000000000",
                "00001100000000000010000000000000_00000000000000000000000010000001",
            ],
            [
                "10001100000000001000000000000000_00000000000010000000000000000001",
                "11000000100000000000000000000000_00000000000000000001000000000000",
                "00000000000000000000001000000000_00000001000000000000000000000000",
                "00001100000000000010000000000000_00000000000000000000000010000001",
            ],
            [
                "10001100000000001000000000000000_00000000000010000000000000000001",
                "10000000100000000000000000000000_00000000000000000001000000000000",
                "01000000000000000000001000000000_00000001000000000000000000000000",
                "00001100000000000010000000000000_00000000000000000000000010000001",
            ],
        ]
    )

    e = bitwise.tensor(
        [
            ["10100000000000000000000000000000"],
            ["01010000000000000000000000000000"],
            ["00001010000001000000000000000000"],
            ["11111010000001000000000000000000"],
        ]
    )

    expected = bitwise.tensor(
        [
            ["00000000000000001000001000000000_00000001000010000000000000000000"],
            ["00000000100000000010000000000000_00000000000000000001000010000000"],
            ["00000000000000000000000000000000_00000000000000000000000000000000"],
            ["11001100100000001010001000000000_00000001000010000001000010000001"],
        ]
    )

    results = bp.error_projection(s, e)

    assert torch.equal(results, expected)


def test_pick_bit_per_row():
    batch_size = 10
    tensor = torch.randint(0, 2**32, (batch_size, 10, 10)).to(dtype=torch.int32)
    result = bp.pick_bit_per_row(tensor)

    # Ensure that (result & tensor) == result (i.e., no extra bits are set)
    assert torch.all((result & tensor) == result)

    # Sum across rows to get integer values
    row_sums = result.sum(dim=2).int()

    for batch in range(batch_size):
        # Check that each row sum is a power of 2 (or zero)
        assert all(x & (x - 1) == 0 for x in row_sums[batch])
