#include <torch/extension.h>

torch::Tensor bitwise_or(torch::Tensor input, c10::optional<int64_t> dim, bool keepdim)
{
    printf("works!\n");
    return input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("bitwise_or", &bitwise_or, "Bitwise OR reduction on int32 tensor",
          py::arg("input"), py::arg("dim") = py::none(), py::arg("keepdim") = false);
}
