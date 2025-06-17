#include <ATen/Parallel.h>
#include <torch/torch.h>

static inline bool is_error(torch::Tensor e_item, int64_t index) {
    return e_item[index / 32].item<int32_t>() & (1 << (index % 32));
}

static void process_batch_item(const torch::Tensor &sm_item,
                               const torch::Tensor &e_item,
                               torch::Tensor result_item) {
    for (int64_t i = 0; i < sm_item.sizes()[0]; ++i) {
        if (!is_error(e_item, i) || !sm_item[i].any().item<bool>()) {
            continue;
        }

        torch::Tensor tmp = result_item | sm_item[i];
        for (int64_t j = 0; j < sm_item.sizes()[0]; ++j) {
            if (is_error(e_item, j) || !sm_item[j].any().item<bool>()) {
                continue;
            }

            if ((sm_item[j] & tmp).equal(sm_item[j])) {
                goto L;
            }
        }

        result_item.copy_(tmp);
    L:;
    }
}

torch::Tensor error_projection(torch::Tensor sm, torch::Tensor e) {
    TORCH_CHECK(sm.dtype() == torch::kInt32, "Bad dtype for sm");
    TORCH_CHECK(e.dtype() == torch::kInt32, "Bad dtype for e");
    TORCH_CHECK(sm.dim() == 3, "Bad number of dimensions for sm");
    TORCH_CHECK(e.dim() == 2, "Bad number of dimensions for e");
    TORCH_CHECK(sm.sizes()[0] == e.sizes()[0],
                "Incompatible first dimension sizes");
    TORCH_CHECK(sm.sizes()[1] <= e.sizes()[1] * 32,
                "Incompatible second dimension sizes");

    const int64_t kParallelThreshold = 16;

    int64_t batch_size = sm.sizes()[0];
    torch::Tensor result = torch::zeros(
        {batch_size, sm.sizes()[2]},
        torch::TensorOptions().dtype(torch::kInt32).device(sm.device()));

    if (batch_size < kParallelThreshold) {
        for (int64_t i = 0; i < batch_size; ++i) {
            process_batch_item(sm[i], e[i], result[i]);
        }
    } else {
        int64_t num_threads = at::get_num_threads();
        int64_t chunk_size = (batch_size + num_threads - 1) / num_threads;
        int64_t num_chunks = (batch_size + chunk_size - 1) / chunk_size;

        at::parallel_for(0, num_chunks, 0, [&](int64_t start, int64_t end) {
            for (int64_t s = start; s < end; ++s) {
                int64_t chunk_start = s * chunk_size;
                int64_t chunk_end =
                    std::min(chunk_start + chunk_size, batch_size);
                if (chunk_start >= batch_size) {
                    continue;
                }

                for (int64_t i = chunk_start; i < chunk_end; i++) {
                    process_batch_item(sm[i], e[i], result[i]);
                }
            }
        });
    }

    return result;
}
