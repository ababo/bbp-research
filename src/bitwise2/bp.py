"""Operations related to Boolean backpropagation."""

from enum import Enum
from typing import Tuple

import torch

from bitwise2 import BitTensor, from_bool_tensor
import bitwise2_ext_cpu

try:
    import bitwise2_ext_cuda
except ImportError:
    bitwise2_ext_cuda = None


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
        x: A batch of inputs with shape [b, n].
        w: Weights with shape [m, n].
        dependency: Whether to compute sensitivity with respect to weights or inputs.

    Returns:
        A positive and negative activation sensitivity tensors, each of shape [b, m, n].
    """
    if len(x.shape) != 2 or len(w.shape) != 2 or x.shape[1] != w.shape[1]:
        raise ValueError("unexpected or non-matching argument shapes")

    sm = x.data.unsqueeze(1) & w.data
    non_zero = (sm != 0).any(dim=-1, keepdim=True)
    src = x.data.unsqueeze(1) if dependency == SensitivityDependency.WEIGHTS else w.data
    sp = torch.where(non_zero, 0, src)
    return BitTensor(x.shape[-1], sp), BitTensor(x.shape[-1], sm)


def error_projection(sm: BitTensor, e: BitTensor) -> BitTensor:
    """
    Perform error projection for a batch of negative sensitivity sequences.

    Args:
        sm: A batch of sensitivity rows with shape [b, m, n].

        e: A batch of error rows of shape [b, m]. Each row bit designates an
           error state of element in the corresponding sensitivity sequence.

    Returns:
        A batch of near-optimal difference masks of shape [b, n].
    """

    if (
        len(sm.shape) != 3
        or len(e.shape) != 2
        or sm.shape[0] != e.shape[0]
        or sm.shape[1] != e.shape[1]
        or sm.data.device != e.data.device
    ):
        raise ValueError("unexpected or non-matching argument shapes or devices")

    dev_type = sm.data.device.type
    if dev_type == "cuda" and bitwise2_ext_cuda is not None:
        data = bitwise2_ext_cuda.error_projection(sm.data, e.data)
    elif dev_type == "cpu":
        data = bitwise2_ext_cpu.error_projection(sm.data, e.data)
    else:
        raise NotImplementedError(f"non-supported device type {dev_type}")

    return BitTensor(sm.shape[-1], data)


def row_activation(x: BitTensor, w: BitTensor) -> BitTensor:
    """
    Compute Row Activation for a batch of inputs and weights.

    Args:
        x: A batch of inputs with shape [b, n].
        w: Weights with shape [m, n].

    Returns:
        A batch of input activations of shape [b, m].
    """

    if len(x.shape) != 2 or len(w.shape) != 2 or x.shape[1] != w.shape[1]:
        raise ValueError("unexpected or non-matching argument shapes")

    x_exp = x.data.unsqueeze(1)
    conjunct = torch.bitwise_and(x_exp, w.data)
    collapsed = conjunct.any(dim=-1)
    return from_bool_tensor(collapsed)
