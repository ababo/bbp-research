#include <torch/extension.h>

torch::Tensor bitwise_or_reduce(torch::Tensor input, int64_t dim)
{
    printf("bitwise_or_reduce (CPU)\n");
    return input;
}
