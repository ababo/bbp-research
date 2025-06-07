"""Operations related to Boolean backpropagation."""

from enum import Enum
from typing import Tuple

import torch

from bitwise2 import BitTensor, from_bool_tensor


class SensitivityDependency(Enum):
    """Specifies what the activation is sensitive to changes in."""

    WEIGHTS = "weights"
    INPUTS = "inputs"


def activation_sensitivity(
    x: BitTensor, w: BitTensor, dependency: SensitivityDependency
) -> Tuple[BitTensor, BitTensor]:
    """
    Compute positive and negative activation sensitivity tensors.

    Args:
        x: A batch of inputs with shape [b, 1, n].
        w: Weights with shape [m, n].
        dependency: Whether to compute sensitivity with respect to weights or inputs.

    Returns:
        A positive and negative activation sensitivity tensors, each of shape [b, m, n].
    """
    if (
        len(x.shape) != 3
        or len(w.shape) != 2
        or x.shape[-2] != 1
        or x.shape[-1] != w.shape[-1]
    ):
        raise ValueError("unexpected or non-matching argument shapes")

    sm = x.data & w.data
    non_zero = torch.any(sm != 0, dim=-1).unsqueeze(-1)
    src = x if dependency == SensitivityDependency.WEIGHTS else w
    sp = torch.where(non_zero, torch.zeros_like(sm), src.data)
    bit_length = x.shape[-1]
    return BitTensor(bit_length, sp), BitTensor(bit_length, sm)


def row_activation(x: BitTensor, w: BitTensor) -> BitTensor:
    """
    Compute Row Activation for a batch of inputs and weights.

    Args:
        x: A batch of inputs with shape [b, 1, n].
        w: Weights with shape [m, n].

    Returns:
        A batch of input activations of shape [b, 1, m].
    """

    if (
        len(x.shape) != 3
        or len(w.shape) != 2
        or x.shape[-2] != 1
        or x.shape[-1] != w.shape[-1]
    ):
        raise ValueError("unexpected or non-matching argument shapes")

    conjunct = torch.bitwise_and(x.data, w.data)
    collapsed = conjunct.any(dim=-1)
    return from_bool_tensor(collapsed[:, None, :])
