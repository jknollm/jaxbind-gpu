// SPDX-License-Identifier: BSD-2-Clause
// A deliberately small typed FFI baseline for the dispatch measurement.
// It launches the same externally compiled kernel as every other variant.
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include "../src/jaxbind_gpu/include/jaxbind_gpu.h"
#include <xla/ffi/api/ffi.h>

namespace ffi = xla::ffi;

static int LaunchScale(void* stream, int64_t kernel, void* x, void* y, int n) {
  using Launch = int (*)(void*, unsigned, unsigned, unsigned, unsigned,
                        unsigned, unsigned, unsigned, void*, void**, void**);
  static void* library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  static auto launch = library ? reinterpret_cast<Launch>(
      dlsym(library, "cuLaunchKernel")) : nullptr;
  if (!launch) return -1;
  void* parameters[] = {&x, &y, &n};
  return launch(reinterpret_cast<void*>(kernel), (n + 255) / 256, 1, 1,
                      256, 1, 1, 0, stream, parameters, nullptr);
}

static ffi::Error Scale(void* stream, int64_t kernel,
                        ffi::Buffer<ffi::F32> input,
                        ffi::Result<ffi::Buffer<ffi::F32>> output) {
  int status = LaunchScale(stream, kernel, input.typed_data(),
                           output->typed_data(), input.element_count());
  return status ? ffi::Error::Internal("Reference CUDA launch failed")
                : ffi::Error::Success();
}

extern "C" int dispatch_cabi(const jb_gpu_context* context,
    const jb_gpu_buffer* inputs, size_t ninputs,
    jb_gpu_buffer* outputs, size_t noutputs, const void* config,
    size_t config_size, char* error, size_t error_capacity) {
  if (ninputs != 1 || noutputs != 1 || config_size != sizeof(int64_t)) return 1;
  int64_t kernel;
  std::memcpy(&kernel, config, sizeof(kernel));
  int status = LaunchScale(context->stream, kernel, inputs[0].data,
                           outputs[0].data, inputs[0].nbytes / sizeof(float));
  if (status && error_capacity) {
    const char* message = "Reference CUDA launch failed";
    std::strncpy(error, message, error_capacity - 1);
    error[error_capacity - 1] = '\0';
  }
  return status;
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    dispatch_reference, Scale,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<void*>>().Attr<int64_t>("kernel")
        .Arg<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>(),
    {ffi::Traits::kCmdBufferCompatible});
