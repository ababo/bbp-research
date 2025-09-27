"""Model and related operations."""

from typing import List, Tuple

import torch

from bitwise2 import fc, BitTensor


class Model:
    """Generic multi-layer model."""

    _layers: List[fc.FullyConnectedLayer]

    def __init__(self, layers: List[fc.FullyConnectedLayer]):
        self._layers = layers

    def eval(self, inputs: BitTensor) -> BitTensor:
        """Evaluate for the given inputs."""

        outputs = inputs
        for layer in self._layers:
            outputs = layer.eval(outputs)
        return outputs

    def update(
        self, inputs: BitTensor, expected_outputs: BitTensor
    ) -> Tuple[BitTensor, int]:
        """
        Perform a forward and backward propagation pass.

        Returns:
            outputs: Outputs after forward propagation pass.
            updated_layers: Number of layers whose parameters were updated.
        """

        layer_outputs = [inputs]
        for layer in self._layers:
            layer_outputs.append(layer.eval(layer_outputs[-1]))

        model_outputs = layer_outputs.pop()

        layer_errors = model_outputs.clone()
        layer_errors.data.bitwise_xor_(expected_outputs.data)

        for i, layer in enumerate(reversed(self._layers)):
            if torch.all(layer_errors.data == 0):
                return model_outputs, i
            layer_errors = layer.update(layer_outputs.pop(), layer_errors)

        return model_outputs, len(self._layers)
