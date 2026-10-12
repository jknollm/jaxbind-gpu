// SPDX-License-Identifier: BSD-2-Clause
// Bare typed FFI baseline. The benchmark owns and compiles both CUDA kernels.
#include <cstdint>
#include <dlfcn.h>
#include <xla/ffi/api/ffi.h>

namespace ffi = xla::ffi;
static ffi::Error Call(void* stream, int64_t kernel,
                       ffi::RemainingArgs args, ffi::RemainingRets rets) {
  using Launch = int (*)(void*, unsigned, unsigned, unsigned, unsigned,
                        unsigned, unsigned, unsigned, void*, void**, void**);
  static void* library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  static auto launch = library ? reinterpret_cast<Launch>(
      dlsym(library, "cuLaunchKernel")) : nullptr;
  if (!launch || (args.size() != 1 && args.size() != 2) || rets.size() != 1)
    return ffi::Error::InvalidArgument("Expected one or two inputs and one output");
  auto input = args.get<ffi::Buffer<ffi::F32>>(0);
  auto output = rets.get<ffi::Buffer<ffi::F32>>(0);
  if (!input.has_value() || !output.has_value())
    return ffi::Error::InvalidArgument("Expected float32 buffers");
  void* x = input->typed_data();
  void* y = output.value()->typed_data();
  int n = input->element_count();
  void* direction = nullptr;
  if (args.size() == 2) {
    auto second = args.get<ffi::Buffer<ffi::F32>>(1);
    if (!second.has_value() || second->element_count() != n)
      return ffi::Error::InvalidArgument("Expected matching direction buffer");
    direction = second->typed_data();
  }
  void* parameters[] = {&x, &direction, &y, &n};
  void* unary[] = {&x, &y, &n};
  int status = launch(reinterpret_cast<void*>(kernel), (n + 255) / 256, 1, 1,
                      256, 1, 1, 0, stream,
                      args.size() == 2 ? parameters : unary, nullptr);
  return status ? ffi::Error::Internal("Reference CUDA launch failed")
                : ffi::Error::Success();
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    nonlinear_reference, Call,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<void*>>().Attr<int64_t>("kernel")
        .RemainingArgs().RemainingRets(), {ffi::Traits::kCmdBufferCompatible});
