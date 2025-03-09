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
    num_nonzero = (d != 0).to(torch.int32).sum(dim=-1)  # Shape [*, m], dtype=int32

    # Rows where Z_i = 0
    mask_all_zero = num_nonzero == 0  # Shape [*, m]

    # Rows where d[i, :] has exactly one bit set
    has_one_bit_sum = has_one_bit.to(torch.int32).sum(
        dim=-1
    )  # Shape [*, m], dtype=int32
    mask_exactly_one_bit = (has_one_bit_sum == 1) & (num_nonzero == 1)

    # Initialize output
    c = torch.zeros_like(d, dtype=torch.int32)

    # When Z_i = 0, set C[i, p] to bits in B that can activate Z_i if flipped in A
    c[mask_all_zero, :] = b[mask_all_zero, :] & ~a[mask_all_zero, :]

    # When Z_i = 1 and exactly one bit set, set C[i, p] to that bit
    c[mask_exactly_one_bit, :] = d[mask_exactly_one_bit, :]

    return c


def _bitwise_or_reduce(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    size = tensor.shape[dim]
    if size == 1:
        return tensor.squeeze(dim)

    # Pad to the next power of two for pairwise reduction
    next_power_of_two = 1 << (size - 1).bit_length()
    if size < next_power_of_two:
        pad_size = next_power_of_two - size
        tensor = torch.cat(
            [
                tensor,
                torch.zeros(
                    *tensor.shape[:dim],
                    pad_size,
                    *tensor.shape[dim + 1 :],
                    dtype=tensor.dtype,
                    device=tensor.device,
                ),
            ],
            dim=dim,
        )

    # Recursively reduce by pairwise OR until size is 1
    while tensor.shape[dim] > 1:
        half_size = tensor.shape[dim] // 2
        left = tensor.narrow(dim, 0, half_size)
        right = tensor.narrow(dim, half_size, half_size)
        tensor = torch.bitwise_or(left, right)
    return tensor.squeeze(dim)


def error_projection(s: bitwise.Tensor, e: bitwise.Tensor) -> bitwise.Tensor:
    """
    Computes Activation Sensitivity.

    Given a and b of shape [m, n], computes c where c[i, j] is
    True if flipping a[i, j] changes the row activation outcome.
    Supports broadcasting when one input has shape [1, n].
    """
    # Batch and shape parameters
    B = s.shape[0]
    m = s.shape[1]
    n_packed = s.shape[2]
    num_groups = e.shape[2]
    m_padded = num_groups * 32  # Align rows to multiple of 32

    # Step 1: Pad s along the row dimension if necessary (for each batch)
    if m < m_padded:
        pad_rows = m_padded - m
        s_padded = torch.cat(
            [s, torch.zeros(B, pad_rows, n_packed, dtype=s.dtype, device=s.device)],
            dim=1,
        )
    else:
        s_padded = s

    # Reshape each batch's s into groups of 32 rows:
    # Resulting shape: (B, num_groups, 32, n_packed)
    s_grouped = s_padded.view(B, num_groups, 32, n_packed)

    # Step 2: Unpack e into a boolean mask.
    # Assume e has shape (B, 1, num_groups). The following shifts out 32 bits per group.
    shifts = torch.arange(31, -1, -1, device=e.device)  # [31, 30, ..., 0]
    e_unpacked = ((e[:, 0, :, None] >> shifts) & 1).to(
        torch.bool
    )  # (B, num_groups, 32)
    e_unpacked_expanded = e_unpacked[..., None]  # (B, num_groups, 32, 1)

    # Step 3: Mask s based on the bit values in e.
    s_e1_masked = torch.where(
        e_unpacked_expanded, s_grouped, torch.tensor(0, dtype=s.dtype, device=s.device)
    )
    s_e0_masked = torch.where(
        ~e_unpacked_expanded, s_grouped, torch.tensor(0, dtype=s.dtype, device=s.device)
    )

    # Reduce within each group (over the 32 rows)
    or_e1_groups = _bitwise_or_reduce(s_e1_masked, dim=2)  # (B, num_groups, n_packed)
    or_e0_groups = _bitwise_or_reduce(s_e0_masked, dim=2)  # (B, num_groups, n_packed)

    # Then reduce across groups
    or_e1_global = _bitwise_or_reduce(or_e1_groups, dim=1)  # (B, n_packed)
    or_e0_global = _bitwise_or_reduce(or_e0_groups, dim=1)  # (B, n_packed)

    # Step 4: Compute the final projection.
    e_prime_packed = or_e1_global & ~or_e0_global  # (B, n_packed)

    return e_prime_packed[:, None, :]  # (B, 1, n_packed)


def pick_bit_per_row(tensor: bitwise.Tensor) -> bitwise.Tensor:
    """
    Picks a single active bit per row at random.

    Given a matrix of shape [m, n], returns a matrix of the same shape
    where each row retains exactly one randomly chosen active bit, with
    all other bits zeroed. Rows with no active bits remain zero.
    """

    bits = 32
    shape = tensor.shape
    tensor = tensor.view(-1, shape[-1])  # Flatten all except the last dimension

    # Convert each int32 value into a 32-bit binary representation
    bit_masks = (
        tensor.unsqueeze(-1) >> torch.arange(bits, device=tensor.device)
    ) & 1  # (batch, cols, 32)

    # Compute which bits are set at least once per row
    row_bit_presence = bit_masks.any(
        dim=1
    )  # (batch, 32) - True if the bit is set in any column

    # Generate a random choice for each row, selecting one bit from the active ones
    rand_values = torch.rand_like(row_bit_presence.float())
    rand_values[~row_bit_presence] = -1  # Ignore unset bits
    selected_bit = rand_values.argmax(
        dim=-1
    )  # (batch,) - Index of the chosen bit per row

    # Convert selected bit index to a bit mask
    selected_mask = (
        (1 << selected_bit).unsqueeze(-1).to(torch.int32)
    )  # Ensure correct dtype

    # Identify which elements originally had this selected bit
    valid_positions = (tensor & selected_mask).bool()

    # Randomly select **one** column per row where the selected bit exists
    valid_positions_float = valid_positions.float()
    rand_col_selector = torch.rand_like(
        valid_positions_float
    )  # Generate random values for tie-breaking
    rand_col_selector[~valid_positions] = -1  # Ignore invalid positions

    selected_col = rand_col_selector.argmax(dim=-1, keepdim=True)  # (batch, 1)

    # Construct final mask: only one element in each row should keep the selected bit
    final_mask = torch.zeros_like(tensor, dtype=torch.int32)
    src = selected_mask.expand_as(final_mask)  # Expand selected bit mask to match shape
    final_mask.scatter_(dim=-1, index=selected_col, src=src)  # Scatter the bit mask

    result = tensor & final_mask  # Apply the final mask

    return result.view(shape)


class Layer:
    _weights: bitwise.Tensor
    _bias: bitwise.Tensor

    def __init__(self, weights: bitwise.Tensor, bias: bitwise.Tensor):
        self._weights = weights
        self._bias = bias

    def eval(self, inputs: bitwise.Tensor) -> bitwise.Tensor:
        return row_activation(inputs, self._weights).bitwise_xor_(self._bias)
