#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace py = pybind11;

torch::Tensor error_projection(torch::Tensor sm, torch::Tensor e);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Native CUDA extension for bitwise2 module.";

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
          "        optimal bit-packed difference masks.",
          py::arg("sm"), py::arg("e"));

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
          "        optimal bit-packed difference masks.",
          py::arg("sm"), py::arg("e"));
}
