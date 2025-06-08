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

    // Compute output shape
    auto output_shape = input.sizes().vec();
    output_shape.erase(output_shape.begin() + dim);

    if (dim == input.dim() - 1) {
        // Reduction along the last dimension with contiguous access
        auto reduction_size = input.size(dim);
        auto num_reductions = input.numel() / reduction_size;
        auto output = torch::empty({num_reductions}, input.options());
        AT_DISPATCH_INTEGRAL_TYPES(
            input.scalar_type(), "bitwise_or_reduce_cpu", [&]() {
                auto data = input.data_ptr<scalar_t>();
                auto output_data = output.data_ptr<scalar_t>();
                at::parallel_for(
                    0, num_reductions, 0, [&](int64_t start, int64_t end) {
                        for (int64_t i = start; i < end; i++) {
                            scalar_t result = 0;
                            auto start_ptr = data + i * reduction_size;
                            for (int64_t j = 0; j < reduction_size; j++) {
                                result |= start_ptr[j];
                            }
                            output_data[i] = result;
                        }
                    });
            });
        return output.reshape(output_shape);
    } else {
        // Permute to move dim to last and make contiguous
        std::vector<int64_t> perm;
        for (int64_t d = 0; d < input.dim(); d++) {
            if (d != dim) {
                perm.push_back(d);
            }
        }
        perm.push_back(dim);
        auto permuted = input.permute(perm).contiguous();
        // Reduce along the last dimension
        auto reduction_size = permuted.size(-1);
        auto num_reductions = permuted.numel() / reduction_size;
        auto output = torch::empty({num_reductions}, permuted.options());
        AT_DISPATCH_INTEGRAL_TYPES(
            permuted.scalar_type(), "bitwise_or_reduce_cpu", [&]() {
                auto data = permuted.data_ptr<scalar_t>();
                auto output_data = output.data_ptr<scalar_t>();
                at::parallel_for(
                    0, num_reductions, 0, [&](int64_t start, int64_t end) {
                        for (int64_t i = start; i < end; i++) {
                            scalar_t result = 0;
                            auto start_ptr = data + i * reduction_size;
                            for (int64_t j = 0; j < reduction_size; j++) {
                                result |= start_ptr[j];
                            }
                            output_data[i] = result;
                        }
                    });
            });
        return output.reshape(output_shape);
    }
}
