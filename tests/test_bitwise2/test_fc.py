"""Tests for operations related to fully connected layer."""

from bitwise2 import bit_tensor
from bitwise2.fc import FullyConnectedLayer


def test_fully_connected_layer_update():
    """Test FullyConnectedLayer.update()."""

    weights = bit_tensor(["100101", "011010", "111001"])
    biases = bit_tensor("010")
    fcl = FullyConnectedLayer(weights, biases)

    inputs = bit_tensor(["110101", "001000", "000001", "101100"])
    errors = bit_tensor(["011", "001", "001", "110"])

    inputs_clone = inputs.clone()
    errors_clone = errors.clone()

    result = fcl.update(inputs, errors, sample_random_bit=False)

    assert inputs == inputs_clone
    assert errors == errors_clone

    # ss=[[000000
    #      010000
    #      000000]
    #     [001000
    #      001000
    #      001000]
    #     [000001
    #      000001
    #      000001]
    #     [000000
    #      001000
    #      000000]]
    # w_mask=[000000
    #         010000
    #         001001]
    # weights'=[100101
    #           001010
    #           110000]
    # errors'=[001
    #          000
    #          000
    #          110]
    # sm=[[100101
    #      000000
    #      110000]
    #     [000000
    #      001000
    #      000000]
    #     [000001
    #      000000
    #      000000]
    #     [100100
    #      001000
    #      100000]]
    # w_mask'=[100100
    #          000000
    #          000000]
    # weights''=[000001
    #            001010
    #            110000]
    # errors''=[001
    #           000
    #           000
    #           010]
    # b_mask=000
    # ss'=[[000001
    #       000001
    #       000001
    #       000001]
    #      [001010
    #       001000
    #       001010
    #       001000]
    #      [000000
    #       110000
    #       110000
    #       100000]]
    # i_mask=[000000
    #         000000
    #         000000
    #         001000]
    # inputs'=[110101
    #          001000
    #          000001
    #          100100]
    # errors'''=[001
    #            000
    #            000
    #            000]
    # sm'=[[000001
    #       000000
    #       000001
    #       000000]
    #      [000000
    #       001000
    #       000000
    #       000000]
    #      [110000
    #       000000
    #       000000
    #       100000]]
    # i_mask'=[110000
    #          000000
    #          000000
    #          000000]
    # inputs''=[000101
    #           001000
    #           000001
    #           100100]

    assert weights == bit_tensor(["000001", "001010", "110000"])
    assert biases == bit_tensor("010")
    assert result == bit_tensor(["110000", "000000", "000000", "001000"])
