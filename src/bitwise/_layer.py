import torch

class Layer:
    _weights: torch.Tensor
    _bias: torch.Tensor

    def __init__(self, weights: torch.Tensor, bias: torch.Tensor):
        self._weights = weights
        self._bias = bias

    def eval(self, inputs: torch.Tensor) -> torch.Tensor:
        tmp = torch.bitwise_and(inputs[:, None, :], self._weights)
        return tmp
