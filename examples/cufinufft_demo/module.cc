#include <pybind11/pybind11.h>
#include "native.h"

namespace py = pybind11;

PYBIND11_MODULE(_bridge, module) {
  module.def("cufinufft_address", []() {
    return reinterpret_cast<uintptr_t>(JbCufinufft);
  });
  module.def("native_synchronize", &JbCufinufftSynchronize,
             py::call_guard<py::gil_scoped_release>());
  module.def("native_configure", &JbCufinufftConfigure,
             py::call_guard<py::gil_scoped_release>());
  module.def("native_cached", &JbCufinufftCached);
}
