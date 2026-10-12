#include <jaxbind_gpu.h>
#include <xla/ffi/api/ffi.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <atomic>

static std::atomic<uint64_t> scale_calls{0};
extern "C" uint64_t scale_count() { return scale_calls.load(); }
static std::atomic<uint64_t> input_pointer{0}, output_pointer{0};
extern "C" uint64_t last_input_pointer() { return input_pointer.load(); }
extern "C" uint64_t last_output_pointer() { return output_pointer.load(); }
static std::atomic<uint64_t> effect_calls{0};
extern "C" uint64_t effect_count() { return effect_calls.load(); }

extern "C" int effect_buffer(const jb_gpu_context*, const jb_gpu_buffer*, size_t,
                            jb_gpu_buffer*, size_t, const void*, size_t,
                            char*, size_t) {
  ++effect_calls;
  return 0;
}

extern "C" int alias_buffer(const jb_gpu_context*, const jb_gpu_buffer* in, size_t,
                           jb_gpu_buffer* out, size_t, const void*, size_t,
                           char* error, size_t capacity) {
  if (in[0].data == out[0].data) return 0;
  std::snprintf(error, capacity, "Expected aliased input and output");
  return 1;
}

static void* driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
using Copy = int (*)(uint64_t, uint64_t, size_t, void*);
using Launch = int (*)(void*, unsigned, unsigned, unsigned, unsigned, unsigned,
                      unsigned, unsigned, void*, void**, void**);
static Copy copy = driver ? reinterpret_cast<Copy>(dlsym(driver, "cuMemcpyDtoDAsync_v2")) : nullptr;
static Launch launch = driver ? reinterpret_cast<Launch>(dlsym(driver, "cuLaunchKernel")) : nullptr;

extern "C" int copy_buffer(const jb_gpu_context* ctx, const jb_gpu_buffer* in,
                          size_t ni, jb_gpu_buffer* out, size_t no,
                          const void*, size_t, char* error, size_t capacity) {
  if (!copy || ctx->abi_version != JB_GPU_ABI_VERSION || ni != 1 || no != 1 ||
      in[0].nbytes != out[0].nbytes) {
    std::snprintf(error, capacity, "Invalid native copy configuration"); return 1;
  }
  input_pointer = reinterpret_cast<uint64_t>(in[0].data);
  output_pointer = reinterpret_cast<uint64_t>(out[0].data);
  if (in[0].nbytes == 0) return 0;
  int status = copy(reinterpret_cast<uint64_t>(out[0].data),
                    reinterpret_cast<uint64_t>(in[0].data), in[0].nbytes, ctx->stream);
  if (status) std::snprintf(error, capacity, "CUDA copy failed: %d", status);
  return status;
}

extern "C" int scale_buffer(const jb_gpu_context* ctx, const jb_gpu_buffer* in,
                           size_t ni, jb_gpu_buffer* out, size_t no,
                           const void* config, size_t size, char* error, size_t capacity) {
  ++scale_calls;
  if (!launch || ni != 1 || no != 1 || in[0].dtype != JB_GPU_F32 || size != 12) {
    std::snprintf(error, capacity, "Invalid native scale configuration"); return 1;
  }
  uint64_t kernel; float factor;
  std::memcpy(&kernel, config, 8); std::memcpy(&factor, static_cast<const char*>(config) + 8, 4);
  int64_t n = in[0].nbytes / sizeof(float);
  if (!n) return 0;
  void* a = in[0].data; void* b = out[0].data;
  void* params[] = {&a, &b, &n, &factor};
  int status = launch(reinterpret_cast<void*>(kernel), (n + 255) / 256, 1, 1,
                      256, 1, 1, 0, ctx->stream, params, nullptr);
  if (status) std::snprintf(error, capacity, "CUDA launch failed: %d", status);
  return status;
}

extern "C" int fail_buffer(const jb_gpu_context*, const jb_gpu_buffer*, size_t,
                          jb_gpu_buffer*, size_t, const void*, size_t,
                          char* error, size_t capacity) {
  std::snprintf(error, capacity, "Deliberate external native error"); return 1;
}

namespace ffi = xla::ffi;
static ffi::Error TypedCopy(void* stream, int32_t device, ffi::AnyBuffer in,
                           ffi::Result<ffi::AnyBuffer> out) {
  auto dims = in.dimensions(); auto odims = out->dimensions();
  jb_gpu_context ctx{JB_GPU_ABI_VERSION, device, stream};
  jb_gpu_buffer a{in.untyped_data(), in.size_bytes(), dims.begin(), dims.size(),
                  static_cast<uint32_t>(in.element_type())};
  jb_gpu_buffer b{out->untyped_data(), out->size_bytes(), odims.begin(), odims.size(),
                  static_cast<uint32_t>(out->element_type())};
  char error[256] = {};
  int status = copy_buffer(&ctx, &a, 1, &b, 1, nullptr, 0, error, sizeof(error));
  return status ? ffi::Error::Internal(error) : ffi::Error::Success();
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(typed_copy, TypedCopy,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<void*>>().Ctx<ffi::DeviceOrdinal>()
    .Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>(), {ffi::Traits::kCmdBufferCompatible});
