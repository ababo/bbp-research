#include <ATen/Parallel.h>
#include <torch/torch.h>

#include <vector>

using ReduceOp = torch::Tensor (*)(const torch::Tensor&, const torch::Tensor&);

static torch::Tensor reduce(torch::Tensor input, ReduceOp op, int64_t dim) {
    TORCH_CHECK(input.numel() > 0, "Input tensor is empty");
    TORCH_CHECK(input.dim() >= 1,
                "Input tensor must have at least 1 dimension");

    dim = dim < 0 ? dim + input.dim() : dim;
    TORCH_CHECK(dim >= 0 && dim < input.dim(), "Dimension out of range");

    int64_t reduce_size = input.size(dim);
    if (reduce_size == 0) {
        return torch::zeros_like(input.select(dim, 0));
    }

    const int64_t kParallelThreshold = 256;

    if (reduce_size < kParallelThreshold) {
        // Serial execution for small reduction sizes.
        torch::Tensor result = input.select(dim, 0).clone();
        for (int64_t i = 1; i < reduce_size; i++) {
            result = op(result, input.select(dim, i));
        }
        return result;
    } else {
        // Parallel execution for larger reduction sizes.
        int64_t num_threads = at::get_num_threads();
        int64_t chunk_size = (reduce_size + num_threads - 1) / num_threads;
        int64_t num_chunks = (reduce_size + chunk_size - 1) / chunk_size;

        // Vector to store partial results.
        std::vector<torch::Tensor> partial_results(num_chunks);

        // Parallel computation of partial reductions.
        at::parallel_for(0, num_chunks, 0, [&](int64_t start, int64_t end) {
            for (int64_t s = start; s < end; s++) {
                int64_t chunk_start = s * chunk_size;
                int64_t chunk_end =
                    std::min(chunk_start + chunk_size, reduce_size);
                if (chunk_start >= reduce_size) {
                    continue;
                }

                // Compute partial OR for this chunk.
                torch::Tensor partial = input.select(dim, chunk_start).clone();
                for (int64_t i = chunk_start + 1; i < chunk_end; i++) {
                    partial = op(partial, input.select(dim, i));
                }
                partial_results[s] = partial;
            }
        });

        // Combine partial results.
        torch::Tensor result = partial_results[0];
        for (int64_t i = 1; i < num_chunks; i++) {
            if (partial_results[i].defined()) {
                result = op(result, partial_results[i]);
            }
        }
        return result;
    }
}

torch::Tensor bitwise_and_reduce(torch::Tensor input, int64_t dim) {
    return reduce(input, torch::bitwise_and, dim);
}

torch::Tensor bitwise_or_reduce(torch::Tensor input, int64_t dim) {
    return reduce(input, torch::bitwise_or, dim);
}
