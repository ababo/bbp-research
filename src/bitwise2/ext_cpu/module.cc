#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace py = pybind11;

torch::Tensor bitwise_and_reduce(torch::Tensor input, int64_t dim);
torch::Tensor bitwise_or_reduce(torch::Tensor input, int64_t dim);
torch::Tensor error_projection(torch::Tensor sm, torch::Tensor e);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Native CPU extension for bitwise2 module.";

    m.def("bitwise_and_reduce", &bitwise_and_reduce,
          "Perform bitwise AND reduction on an int32 PyTorch tensor.\n\n"
          "Args:\n"
          "    tensor (torch.Tensor): Input tensor of type int32.\n"
          "    dim (int): Dimension along which to perform reduction.\n\n"
          "Returns:\n"
          "    torch.Tensor: Reduced tensor.",
          py::arg("tensor"), py::arg("dim"));

    m.def("bitwise_or_reduce", &bitwise_or_reduce,
          "Perform bitwise OR reduction on an int32 PyTorch tensor.\n\n"
          "Args:\n"
          "    tensor (torch.Tensor): Input tensor of type int32.\n"
          "    dim (int): Dimension along which to perform reduction.\n\n"
          "Returns:\n"
          "    torch.Tensor: Reduced tensor.",
          py::arg("tensor"), py::arg("dim"));

    m.def("error_projection", &error_projection,
          "Perform error projection for sensitivity row groups.\n\n"
          "Args:\n"
          "    sm (torch.Tensor): int32 tensor of shape [m, b, n].\n"
          "        Bit-packed sensitivity row groups.\n"
          "    e (torch.Tensor): int32 tensor of shape [m, k] (k * 32 >= b).\n"
          "        Bit-packed error rows. Each error row bit marks the error \n"
          "        state of the corresponding row within the sensitivity group.\n\n"
          "Returns:\n"
          "    torch.Tensor: int32 tensor of shape [m, n].\n"
          "        Near-optimal difference rows.",
          py::arg("sm"), py::arg("e"));
}
