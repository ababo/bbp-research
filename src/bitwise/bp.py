import bitwise
import torch


def row_activation(x: bitwise.Tensor, w: bitwise.Tensor) -> bitwise.Tensor:
    """Computes Row Activation: Z[i] is True if any (X[j] AND W[i, j]) is True."""
    conjunct = torch.bitwise_and(x[:, None, :], w)
    collapsed = conjunct.any(dim=-1)
    return bitwise.pack(collapsed)


class Layer:
    _weights: bitwise.Tensor
    _bias: bitwise.Tensor

    def __init__(self, weights: bitwise.Tensor, bias: bitwise.Tensor):
        self._weights = weights
        self._bias = bias

    def eval(self, inputs: bitwise.Tensor) -> bitwise.Tensor:
        return row_activation(inputs, self._weights).bitwise_xor_(self._bias)
