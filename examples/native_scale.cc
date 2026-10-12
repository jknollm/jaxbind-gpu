// SPDX-License-Identifier: BSD-2-Clause
// Launch a precompiled CUDA kernel through the standalone jaxbind-gpu C ABI.
#include <jaxbind_gpu.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstring>

using Launch = int (*)(void*, unsigned, unsigned, unsigned, unsigned, unsigned,
                      unsigned, unsigned, void*, void**, void**);
static void* driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
static Launch launch = driver ? reinterpret_cast<Launch>(dlsym(driver, "cuLaunchKernel")) : nullptr;

extern "C" int scale_buffer(const jb_gpu_context* ctx, const jb_gpu_buffer* in,
                           size_t ni, jb_gpu_buffer* out, size_t no,
                           const void* config, size_t size, char* error, size_t capacity) {
  if (!launch || ctx->abi_version != JB_GPU_ABI_VERSION || ni != 1 || no != 1 ||
      in[0].dtype != JB_GPU_F32 || out[0].dtype != JB_GPU_F32 ||
      in[0].nbytes != out[0].nbytes || size != 12) {
    std::snprintf(error, capacity, "Invalid native scale configuration"); return 1;
  }
  uint64_t kernel; float factor;
  std::memcpy(&kernel, config, 8);
  std::memcpy(&factor, static_cast<const char*>(config) + 8, 4);
  int64_t n = in[0].nbytes / sizeof(float);
  if (!n) return 0;
  void* a = in[0].data; void* b = out[0].data;
  void* params[] = {&a, &b, &n, &factor};
  int status = launch(reinterpret_cast<void*>(kernel), (n + 255) / 256, 1, 1,
                      256, 1, 1, 0, ctx->stream, params, nullptr);
  if (status) std::snprintf(error, capacity, "CUDA launch failed: %d", status);
  return status;
}
