// SPDX-License-Identifier: BSD-2-Clause
// Minimal FFI control invoking the identical external C ABI provider.
#include <cstdint>
#include <string_view>
#include "../src/jaxbind_gpu/include/jaxbind_gpu.h"
#include <xla/ffi/api/ffi.h>

namespace ffi = xla::ffi;

static jb_gpu_buffer Buffer(ffi::AnyBuffer value) {
  auto dims = value.dimensions();
  return {value.untyped_data(), value.size_bytes(), dims.begin(), dims.size(),
          static_cast<uint32_t>(value.element_type())};
}

static ffi::Error Call(void* stream, int32_t device, int64_t address,
                       std::string_view config, ffi::RemainingArgs args,
                       ffi::RemainingRets rets) {
  if (args.size() > 8 || rets.size() > 8)
    return ffi::Error::InvalidArgument("Reference supports at most eight buffers");
  jb_gpu_buffer inputs[8], outputs[8];
  for (size_t i = 0; i < args.size(); ++i) {
    auto input = args.get<ffi::AnyBuffer>(i);
    if (!input.has_value()) return ffi::Error::InvalidArgument("Array required");
    inputs[i] = Buffer(input.value());
  }
  for (size_t i = 0; i < rets.size(); ++i) {
    auto output = rets.get<ffi::AnyBuffer>(i);
    if (!output.has_value()) return ffi::Error::InvalidArgument("Array required");
    outputs[i] = Buffer(*output.value());
  }
  jb_gpu_context context{JB_GPU_ABI_VERSION, device, stream};
  char error[2048] = {};
  auto operation = reinterpret_cast<jb_gpu_operation>(address);
  int status = operation(&context, inputs, args.size(), outputs, rets.size(),
                         config.data(), config.size(), error, sizeof(error));
  error[sizeof(error) - 1] = '\0';
  return status ? ffi::Error::Internal(error) : ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(provider_reference, Call,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<void*>>().Ctx<ffi::DeviceOrdinal>()
        .Attr<int64_t>("address").Attr<std::string_view>("config")
        .RemainingArgs().RemainingRets());
