"""Operations related to fully connected layer."""

from bitwise2 import BitTensor
from bitwise2 import bp


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

        raise NotImplementedError
