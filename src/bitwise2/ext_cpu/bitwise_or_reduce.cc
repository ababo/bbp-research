#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/extension.h>

#include <vector>

torch::Tensor bitwise_or_reduce(torch::Tensor input, int64_t dim) {
    // Check input validity
    if (input.numel() == 0) {
        TORCH_CHECK(false, "Input tensor is empty");
    }
    if (input.dim() < 1) {
        TORCH_CHECK(false, "Input tensor must have at least 1 dimension");
    }

    // Normalize dimension
    if (dim < 0) {
        dim += input.dim();
    }
    TORCH_CHECK(dim >= 0 && dim < input.dim(), "Dimension out of range");

    // Compute permutation to move reduction dimension to last
    std::vector<int64_t> perm;
    for (int64_t d = 0; d < input.dim(); d++) {
        if (d != dim) {
            perm.push_back(d);
        }
    }
    perm.push_back(dim);

    // Permute input for contiguous memory in reduction dimension
    auto permuted = input.permute(perm);

    // Check permuted tensor
    if (permuted.dim() < 2) {
        TORCH_CHECK(false, "Permuted tensor must have at least 2 dimensions");
    }

    // Calculate sizes
    int64_t dr = permuted.size(-1);  // Size of reduction dimension
    int64_t leading_dims =
        permuted.numel() / dr;  // Product of other dimensions
    if (dr <= 0) {
        TORCH_CHECK(false, "Reduction dimension size must be positive");
    }

    // Compute strides of permuted tensor
    auto strides = permuted.strides();

    // Create output tensor
    auto output = torch::empty({leading_dims}, input.options());

    // Dispatch for integral types and perform reduction
    AT_DISPATCH_INTEGRAL_TYPES(
        input.scalar_type(), "bitwise_or_reduce_cpu", [&]() {
            auto input_data = permuted.data_ptr<scalar_t>();
            auto output_data = output.data_ptr<scalar_t>();

            // Precompute divisors for indexing
            std::vector<int64_t> divisors(permuted.dim() - 1);
            int64_t divisor = 1;
            for (int64_t d = permuted.dim() - 2; d >= 0; d--) {
                divisors[d] = divisor;
                divisor *= permuted.size(d);
            }

            // Parallelize over independent reductions
            at::parallel_for(
                0, leading_dims, 0, [&](int64_t start, int64_t end) {
                    for (int64_t i = start; i < end; i++) {
                        // Compute indices for all non-reduction dimensions
                        std::vector<int64_t> indices(permuted.dim() - 1);
                        int64_t remainder = i;
                        for (int64_t d = 0; d < permuted.dim() - 1; d++) {
                            indices[d] = remainder / divisors[d];
                            remainder = remainder % divisors[d];
                            if (indices[d] < 0 ||
                                indices[d] >= permuted.size(d)) {
                                TORCH_CHECK(false, "Index out of bounds");
                            }
                        }
                        // Compute base index
                        int64_t base_idx = 0;
                        for (int64_t d = 0; d < permuted.dim() - 1; d++) {
                            base_idx += indices[d] * strides[d];
                        }
                        if (base_idx < 0 || base_idx >= permuted.numel()) {
                            TORCH_CHECK(false, "base_idx out of bounds");
                        }

                        // Initialize result to 0 to match Python reference
                        scalar_t result = 0;
                        for (int64_t j = 0; j < dr; j++) {
                            int64_t idx =
                                base_idx + j * strides[permuted.dim() - 1];
                            if (idx < 0 || idx >= permuted.numel()) {
                                TORCH_CHECK(false, "idx out of bounds");
                            }
                            scalar_t value = input_data[idx];
                            result |= value;
                        }
                        output_data[i] = result;
                    }
                });
        });

    // Compute output shape by removing reduction dimension
    auto output_shape = input.sizes().vec();
    output_shape.erase(output_shape.begin() + dim);

    // Reshape and print output tensor
    auto final_output = output.reshape(output_shape);

    return final_output;
}
