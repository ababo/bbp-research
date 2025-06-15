#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Native CUDA extension for bitwise2 module.";
}
