// SPDX-License-Identifier: BSD-2-Clause
#include "kernel_bridge.h"
#include "inline_storage.h"
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <dlfcn.h>
#include <vector>

namespace {
struct Config {
  uint32_t version;
  int32_t device;
  uint64_t function;
  uint32_t grid[3], block[3], shared_bytes, parameter_count;
};
struct Parameter { uint32_t kind, index, offset, size; };
static_assert(sizeof(Config) == 48 && sizeof(Parameter) == 16);

int Fail(char* error, size_t size, const char* message) {
  if (size) std::snprintf(error, size, "%s", message);
  return 1;
}
}

int JbLaunchKernel(const jb_gpu_context* context, const jb_gpu_buffer* inputs,
                   size_t ninputs, jb_gpu_buffer* outputs, size_t noutputs,
                   const void* data, size_t size, char* error, size_t error_size) {
  if (context->abi_version != JB_GPU_ABI_VERSION || size < sizeof(Config))
    return Fail(error, error_size, "Invalid native CUDA kernel configuration");
  Config config;
  std::memcpy(&config, data, sizeof(config));
  if (config.version != 1 || !config.function ||
      size < sizeof(Config) + config.parameter_count * sizeof(Parameter))
    return Fail(error, error_size, "Invalid native CUDA kernel parameter layout");
  if (config.device != context->device)
    return Fail(error, error_size, "CUDA kernel was compiled for a different device; bind a kernel for this device");
  for (int i = 0; i < 3; ++i)
    if (!config.grid[i] || !config.block[i])
      return Fail(error, error_size, "CUDA launch dimensions must be positive");
  const auto* bytes = static_cast<const char*>(data);
  // Addresses passed to cuLaunchKernel must stay stable while building the
  // parameter list. Reserve storage before taking addresses of pointer values.
  InlineStorage<void*, 32> pointers(config.parameter_count), parameters(config.parameter_count);
  for (uint32_t i = 0; i < config.parameter_count; ++i) {
    Parameter parameter;
    std::memcpy(&parameter, bytes + sizeof(Config) + i * sizeof(Parameter), sizeof(parameter));
    if (parameter.kind == 0 && parameter.index < ninputs) {
      pointers[i] = inputs[parameter.index].data;
      parameters[i] = &pointers[i];
    } else if (parameter.kind == 1 && parameter.index < noutputs) {
      pointers[i] = outputs[parameter.index].data;
      parameters[i] = &pointers[i];
    } else if (parameter.kind == 2 && parameter.size && parameter.offset <= size &&
               parameter.size <= size - parameter.offset) {
      parameters[i] = const_cast<char*>(bytes + parameter.offset);
    } else {
      return Fail(error, error_size, "Invalid CUDA kernel argument reference");
    }
  }
  using Launch = int (*)(void*, unsigned int, unsigned int, unsigned int,
                         unsigned int, unsigned int, unsigned int, unsigned int,
                         void*, void**, void**);
  static void* library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  static auto launch = library ? reinterpret_cast<Launch>(dlsym(library, "cuLaunchKernel")) : nullptr;
  if (!launch) return Fail(error, error_size, "CUDA driver launch API unavailable");
  int status = launch(reinterpret_cast<void*>(config.function),
                      config.grid[0], config.grid[1], config.grid[2],
                      config.block[0], config.block[1], config.block[2],
                      config.shared_bytes, context->stream, parameters.data(), nullptr);
  if (status && error_size) std::snprintf(error, error_size, "cuLaunchKernel failed with CUDA driver error %d", status);
  return status;
}
