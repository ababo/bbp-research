import bitwise
import torch


def row_activation(x: bitwise.Tensor, w: bitwise.Tensor) -> bitwise.Tensor:
    """
    Computes Row Activation.

    Given x and w of shape [1, n] and [m, n], computes z where z[i] is True
    if there exists a column j such that both x[0, j] and w[i, j] are True.
    """
    conjunct = torch.bitwise_and(x[:, None, :], w)
    collapsed = conjunct.any(dim=-1)
    return bitwise.pack(collapsed)


def activation_sensitivity(a: bitwise.Tensor, b: bitwise.Tensor) -> bitwise.Tensor:
    """
    Computes Activation Sensitivity.

    Given a and b of shape [m, n], computes c where c[i, j] is
    True if flipping a[i, j] changes the row activation outcome.
    Supports broadcasting when one input has shape [1, n].
    """
    # Validate input dimensions
    if a.dim() < 2 or b.dim() < 2:
        raise ValueError("inputs must be at least 2D tensors")

    # Extract shapes
    *batch_shape_a, m_a, k_a = a.shape
    *batch_shape_b, m_b, k_b = b.shape

    if k_a != k_b:
        raise ValueError("number of columns must match")

    # Determine batch shape
    batch_shape = torch.broadcast_shapes(tuple(batch_shape_a), tuple(batch_shape_b))

    # Expand a and b to match batch shape
    a = a.expand(*batch_shape, m_a, k_a)
    b = b.expand(*batch_shape, m_b, k_b)

    # Determine max rows for broadcasting
    m = max(m_a, m_b)
    k = k_a

    # Broadcast a and b to [*, m, k]
    if m_a == 1 and m_a < m:
        a = a.expand(*batch_shape, m, k)
    if m_b == 1 and m_b < m:
        b = b.expand(*batch_shape, m, k)

    # Compute bitwise AND
    d = a & b  # Shape [*, m, k]

    # Identify elements with exactly one bit set
    has_one_bit = (d != 0) & ((d & (d - 1)) == 0)  # True where d[i, p] has one bit

    # Number of non-zero elements per row
    num_nonzero = (d != 0).sum(dim=-1)  # Shape [*, m]

    # Rows where Z_i = 0
    mask_all_zero = num_nonzero == 0  # Shape [*, m]

    # Rows where d[i, :] has exactly one bit set
    mask_exactly_one_bit = (has_one_bit.sum(dim=-1) == 1) & (num_nonzero == 1)

    # Initialize output
    c = torch.zeros_like(d, dtype=torch.int32)

    # When Z_i = 0, set C[i, p] to bits in B that can activate Z_i if flipped in A
    c[mask_all_zero, :] = b[mask_all_zero, :] & ~a[mask_all_zero, :]

    # When Z_i = 1 and exactly one bit set, set C[i, p] to that bit
    c[mask_exactly_one_bit, :] = d[mask_exactly_one_bit, :]

    return c


class Layer:
    _weights: bitwise.Tensor
    _bias: bitwise.Tensor

    def __init__(self, weights: bitwise.Tensor, bias: bitwise.Tensor):
        self._weights = weights
        self._bias = bias

    def eval(self, inputs: bitwise.Tensor) -> bitwise.Tensor:
        return row_activation(inputs, self._weights).bitwise_xor_(self._bias)
