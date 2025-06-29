#include <c10/cuda/CUDAException.h>
#include <torch/torch.h>

enum class RowKind : uint8_t { kZeroed = 0, kCorrect, kIncorrect };

static const int kThreadsPerBlock = 32;

static __device__ inline bool is_error(const int32_t* e_item, int64_t index) {
    return e_item[index / 32] & (1 << (index % 32));
}

__global__ void process_batch_item_kernel(const int32_t* sm, const int32_t* e,
                                          int32_t* result, int64_t height,
                                          int64_t width, int64_t e_stride,
                                          int64_t batch_size) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) {
        return;
    }

    int thread_id = threadIdx.x;
    int num_threads = blockDim.x;

    const int32_t* sm_item = sm + batch_idx * height * width;
    const int32_t* e_item = e + batch_idx * e_stride;
    int32_t* result_item = result + batch_idx * width;

    extern __shared__ int32_t shared_mem[];
    int32_t* tmp_row = shared_mem;
    RowKind* row_kinds = reinterpret_cast<RowKind*>(shared_mem + width);

    for (int64_t i = thread_id; i < height; i += num_threads) {
        const int32_t* sm_item_row = sm_item + i * width;
        bool non_zero = false;
        for (int64_t j = 0; j < width; ++j) {
            if (sm_item_row[j] != 0) {
                non_zero = true;
                break;
            }
        }
        if (non_zero) {
            row_kinds[i] =
                is_error(e_item, i) ? RowKind::kIncorrect : RowKind::kCorrect;
        } else {
            row_kinds[i] = RowKind::kZeroed;
        }
    }
    __syncthreads();

    for (int64_t i = 0; i < height; ++i) {
        if (row_kinds[i] != RowKind::kIncorrect) {
            continue;
        }

        for (int64_t j = thread_id; j < width; j += num_threads) {
            tmp_row[j] = result_item[j] | sm_item[i * width + j];
        }

        __shared__ bool update_result;
        update_result = true;
        __syncthreads();

        for (int64_t j = thread_id; j < height; j += num_threads) {
            if (row_kinds[j] != RowKind::kCorrect) {
                continue;
            }

            bool spoils = true;
            for (int64_t k = 0; k < width; ++k) {
                int32_t val = sm_item[j * width + k] & tmp_row[k];
                if (val != sm_item[j * width + k]) {
                    spoils = false;
                    break;
                }
            }

            if (spoils) {
                update_result = false;
            }
        }
        __syncthreads();

        if (update_result) {
            for (int64_t j = thread_id; j < width; j += num_threads) {
                result_item[j] = tmp_row[j];
            }
        }
        __syncthreads();
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

    int64_t batch_size = sm.sizes()[0];
    int64_t height = sm.sizes()[1];
    int64_t width = sm.sizes()[2];
    int64_t e_stride = e.sizes()[1];

    sm = sm.contiguous();
    e = e.contiguous();
    torch::Tensor result = torch::zeros(
        {batch_size, width},
        torch::TensorOptions().dtype(torch::kInt32).device(sm.device()));

    int num_blocks = batch_size;
    size_t shared_mem_size = width * sizeof(int32_t) + height * sizeof(RowKind);

    process_batch_item_kernel<<<num_blocks, kThreadsPerBlock,
                                shared_mem_size>>>(
        sm.data_ptr<int32_t>(), e.data_ptr<int32_t>(),
        result.data_ptr<int32_t>(), height, width, e_stride, batch_size);

    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaDeviceSynchronize());

    return result;
}
