#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace py = pybind11;

torch::Tensor bitwise_or_reduce(torch::Tensor input, int64_t dim);

torch::Tensor error_projection(torch::Tensor sm, torch::Tensor e);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Native CPU extension for bitwise2 module.";

    m.def("bitwise_or_reduce", &bitwise_or_reduce,
          "Perform bitwise OR reduction on an int32 PyTorch tensor.\n\n"
          "Args:\n"
          "    tensor (torch.Tensor): Input tensor of type int32.\n"
          "    dim (int): Dimension along which to perform reduction.\n\n"
          "Returns:\n"
          "    torch.Tensor: Reduced tensor.",
          py::arg("tensor"), py::arg("dim"));

    m.def("error_projection", &error_projection,
          "Perform error projection for a batch of Boolean sensitivity "
          "sequences.\n\n"
          "Args:\n"
          "    sm (torch.Tensor): int32 tensor of shape [b, m, n].\n"
          "        Batch of bit-packed negative sensitivity row sequences.\n"
          "    e (torch.Tensor): int32 tensor of shape [b, k] (k * 32 >= m).\n"
          "        Batch of bit-packed error rows. Each row bit designates \n"
          "        an error state of element in the corresponding sensitivity "
          "sequence.\n\n"
          "Returns:\n"
          "    torch.Tensor: int32 tensor of shape [b, n]. Batch of\n"
          "        near-optimal bit-packed difference masks.",
          py::arg("sm"), py::arg("e"));
}
