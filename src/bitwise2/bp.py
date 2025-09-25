"""Operations related to Boolean backpropagation."""

from typing import Tuple

import torch

from bitwise2 import BitTensor, from_bool_tensor
import bitwise2_ext_cpu

try:
    import bitwise2_ext_cuda
except ImportError:
    bitwise2_ext_cuda = None


def activation_sensitivity(
    signal: BitTensor,
    target: BitTensor,
) -> Tuple[BitTensor, BitTensor]:
    """
    Compute positive and negative activation sensitivity tensors.

    Args:
        signal: Tensor of shape [b, n] that serves as an input for row activation.
        target: Tensor of shape [m, n] whose changes the sensitivity is measured against.

    Returns:
        A positive and negative activation sensitivity tensors, each of shape [b, m, n].
    """
    if (
        len(signal.shape) != 2
        or len(target.shape) != 2
        or signal.shape[1] != target.shape[1]
    ):
        raise ValueError("unexpected or non-matching argument shapes")

    sm = signal.data.unsqueeze(1) & target.data
    non_zero = (sm != 0).any(dim=-1, keepdim=True)
    src = signal.data.unsqueeze(1)
    sp = torch.where(non_zero, 0, src)
    return BitTensor(signal.shape[-1], sp), BitTensor(signal.shape[-1], sm)


def error_projection(sm: BitTensor, e: BitTensor) -> BitTensor:
    """
    Perform error projection for sensitivity row groups.

    Args:
        sm: Sensitivity row groups of shape [m, b, n].

        e: Error rows of shape [m, b]. Each error row bit marks the error
           state of the corresponding row within the sensitivity group.

    Returns:
        Near-optimal difference rows of shape [m, n].
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


def specialized_activation_sensitivity(sp: BitTensor, sm: BitTensor) -> BitTensor:
    """
    Compute specialized activation sensitivity tensor for given sensitivity tensors.

    Args:
        sp: A positive sensitivity tensor with shape [b, m, n].
        sm: A negative sensitivity tensor with shape [b, m, n].

    Returns:
        A specialized sensitivity tensor of shape [b, m, n].
    """

    if len(sp.shape) != 3 or sp.shape != sm.shape:
        raise ValueError("unexpected or non-matching argument shapes")

    keep_rows = (sm.data & (sm.data - 1) != 0).any(dim=-1).logical_not_()
    keep_rows.logical_and_((sm.data != 0).sum(dim=-1) <= 1)
    sm_data = sm.data * keep_rows.unsqueeze_(-1)
    return BitTensor(sp.shape[-1], sm_data.bitwise_or_(sp.data))


def specialized_error_projection(ss: BitTensor, e: BitTensor) -> BitTensor:
    """
    Perform error projection for specialized sensitivity row groups.

    Args:
        sm: Sensitivity row groups of shape [m, b, n].

        e: Error rows of shape [m, b]. Each error row bit marks the error
           state of the corresponding row within the sensitivity group.

    Returns:
        Optimal difference rows of shape [m, n].
    """

    if len(ss.shape) != 3 or ss.shape[:-1] != e.shape:
        raise ValueError("unexpected or non-matching argument shapes")

    row_indices = (
        torch.arange(ss.shape[1], device=ss.data.device)
        .unsqueeze_(0)
        .expand(ss.shape[0], ss.shape[1])
    )
    bit_masks = 1 << (row_indices % 32)
    e_group = torch.gather(e.data, 1, row_indices // 32)

    i_bits = e_group & bit_masks != 0
    i_ss = ss.data * i_bits.unsqueeze(2)
    i_row = bitwise2_ext_cpu.bitwise_or_reduce(i_ss, 1)

    c_bits = e_group & bit_masks == 0
    c_ss = ss.data * c_bits.unsqueeze(2)
    c_row = bitwise2_ext_cpu.bitwise_or_reduce(c_ss, 1)

    mask = i_row.bitwise_and_(c_row.bitwise_not_())
    return BitTensor(ss.shape[-1], mask)
