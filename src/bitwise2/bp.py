"""Operations related to Boolean backpropagation."""

import torch

from bitwise2 import BitTensor, from_bool_tensor


def row_activation(x: BitTensor, w: BitTensor) -> BitTensor:
    """
    Compute Row Activation for a batch of inputs and weights.

    Args:
        x: A batch of inputs with shape [b, 1, n].
        w: Weights with shape [m, n].

    Returns:
        A batch of input activations of shape [b, 1, m].

    Raises:
        ValueError: If the arguments shapes are wrong or don't match.
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
