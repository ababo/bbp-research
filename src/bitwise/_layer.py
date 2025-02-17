import torch
import bitwise

class Layer:
    _weights: torch.Tensor
    _bias: torch.Tensor

    def __init__(self, weights: torch.Tensor, bias: torch.Tensor):
        self._weights = weights
        self._bias = bias

    def eval(self, inputs: torch.Tensor) -> torch.Tensor:
        conjunct = torch.bitwise_and(inputs[:, None, :], self._weights)
        collapsed = conjunct.any(dim=-1)
        packed = bitwise.pack(collapsed)
        return packed.bitwise_xor_(self._bias)
