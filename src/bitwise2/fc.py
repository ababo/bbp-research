"""Operations related to fully connected layer."""

import torch

from bitwise2 import BitTensor, from_bool_tensor
from bitwise2 import bp
import bitwise2_ext_cpu


class FullyConnectedLayer:
    """Represents a fully-connected layer."""

    _weights: BitTensor
    _biases: BitTensor

    def __init__(
        self,
        weights: BitTensor,
        biases: BitTensor,
    ):
        """
        Constructs a fully-connected layer for the given weights and biases.

        Args:
            weights: Weights with shape [m, n].
            biases: Biases with shape [m].
        """

        if (
            len(weights.shape) != 2
            or len(biases.shape) != 1
            or weights.shape[0] != biases.shape[0]
        ):
            raise ValueError("unexpected or non-matching argument shapes")

        self._weights = weights
        self._biases = biases

    def eval(self, inputs: BitTensor) -> BitTensor:
        """
        Computes the layer's output for the given inputs.

        Args:
            inputs: A batch of inputs with shape [b, n].

        Returns:
            A batch of input activations of shape [b, m].
        """

        z = bp.row_activation(inputs, self._weights)
        z.data.bitwise_xor_(self._biases.data)
        return z

    def update(self, inputs: BitTensor, errors: BitTensor) -> BitTensor:
        """Updates weights and biases and returns estimated input errors."""

        sp, sm = bp.activation_sensitivity(inputs, self._weights)
        ss = bp.specialized_activation_sensitivity(sp, sm)
        ss.data.transpose_(0, 1)
        te = errors.to_bool_tensor().transpose_(0, 1)
        w_mask = bp.specialized_error_projection(ss, from_bool_tensor(te))
        self._weights.data.bitwise_xor_(w_mask.data)

        ss.data.transpose_(0, 1).bitwise_and_(w_mask.data)
        te ^= torch.any(ss.data != 0, dim=2).transpose_(0, 1)

        _, sm = bp.activation_sensitivity(inputs, self._weights)
        sm.data.transpose_(0, 1)
        w_mask = bp.error_projection(sm, from_bool_tensor(te))
        self._weights.data.bitwise_xor_(w_mask.data)

        sm.data.transpose_(0, 1)
        e_mask = (sm.data & w_mask.data) == sm.data
        e_mask &= sm.data.sum(dim=-1, keepdim=True) != 0
        te ^= e_mask.squeeze_().transpose_(0, 1)

        te.transpose_(0, 1)
        b_mask = bitwise2_ext_cpu.bitwise_and_reduce(te, 0)
        self._biases.data.bitwise_xor_(from_bool_tensor(b_mask).data)
        te.bitwise_xor_(b_mask)

        sp, sm = bp.activation_sensitivity(self._weights, inputs)
        ss = bp.specialized_activation_sensitivity(sp, sm)
        ss.data.transpose_(0, 1)
        i_mask = bp.specialized_error_projection(ss, from_bool_tensor(te))
        inputs2 = BitTensor(inputs.shape[-1], inputs.data ^ i_mask.data)

        ss.data.transpose_(0, 1).bitwise_and_(i_mask.data)
        te ^= torch.any(ss.data != 0, dim=2).transpose_(0, 1)

        _, sm = bp.activation_sensitivity(self._weights, inputs2)
        sm.data.transpose_(0, 1)
        i_mask = bp.error_projection(sm, from_bool_tensor(te))
        inputs2.data.bitwise_xor_(i_mask.data)

        inputs2.data.bitwise_xor_(inputs.data)
        return inputs2
