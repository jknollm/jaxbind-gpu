// SPDX-License-Identifier: BSD-2-Clause
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <string>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <deque>
#include <dlfcn.h>
#include "../src/jaxbind_gpu/include/jaxbind_gpu.h"
#include "kernel_bridge.h"
#include "inline_storage.h"

#include <pybind11/pybind11.h>
#include <xla/ffi/api/ffi.h>

namespace py = pybind11;
namespace ffi = xla::ffi;

// CUDA user-object destructors must not call CUDA or wait for Python's GIL.
// Transfer graph owners to a queue. The Python reaper releases them on their
// device after CUDA has retired every graph/launch referencing the allocation.
struct RetiredGraphs {
  std::mutex mutex;
  std::deque<PyObject*> owners;
};
static RetiredGraphs& Retired() {
  // CUDA may invoke destructors during interpreter/process shutdown.
  static auto* retired = new RetiredGraphs;
  return *retired;
}
static void RetireGraph(void* owner) {
  auto& retired = Retired();
  std::lock_guard<std::mutex> lock(retired.mutex);
  retired.owners.push_back(static_cast<PyObject*>(owner));
}

struct NativeEntry {
  jb_gpu_operation operation;
  std::string process_key;
};
static std::mutex native_mutex;
static std::unordered_map<int64_t, NativeEntry> native_operations;
static int64_t next_native_id = 0;

struct NativeState {
  static ffi::TypeId id;
  jb_gpu_operation operation;
  std::string config;
};
ffi::TypeId NativeState::id = {};
static constexpr auto native_type_info = ffi::MakeTypeInfo<NativeState>();

static ffi::ErrorOr<std::unique_ptr<NativeState>> InstantiateNative(
    int64_t native_id, std::string_view process_key, std::string_view config) {
  std::lock_guard<std::mutex> lock(native_mutex);
  auto found = native_operations.find(native_id);
  if (found == native_operations.end() || found->second.process_key != process_key)
    return ffi::Error::InvalidArgument("Native operation belongs to a different process");
  return std::make_unique<NativeState>(
      NativeState{found->second.operation, std::string(config)});
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(JaxbindGpuNativeInstantiate, InstantiateNative,
    ffi::Ffi::BindInstantiate().Attr<int64_t>("native_id")
                              .Attr<std::string_view>("process_key")
                              .Attr<std::string_view>("config"));

static jb_gpu_buffer NativeBuffer(ffi::AnyBuffer buffer) {
  auto dims = buffer.dimensions();
  return {buffer.untyped_data(), buffer.size_bytes(), dims.begin(), dims.size(),
          static_cast<uint32_t>(buffer.element_type())};
}

static ffi::Error NativeCall(NativeState* state, void* stream, int32_t device,
                            ffi::Dictionary,
                            ffi::RemainingArgs args, ffi::RemainingRets rets) {
  try {
    InlineStorage<jb_gpu_buffer, 8> inputs(args.size()), outputs(rets.size());
    for (size_t i = 0; i < args.size(); ++i) {
      auto arg = args.get<ffi::AnyBuffer>(i);
      if (!arg.has_value()) return ffi::Error::InvalidArgument("Expected array input");
      inputs[i] = NativeBuffer(arg.value());
    }
    for (size_t i = 0; i < rets.size(); ++i) {
      auto ret = rets.get<ffi::AnyBuffer>(i);
      if (!ret.has_value()) return ffi::Error::InvalidArgument("Expected array output");
      outputs[i] = NativeBuffer(*ret.value());
    }
    jb_gpu_context context{JB_GPU_ABI_VERSION, device, stream};
    char error[2048] = {};
    int status = state->operation(&context, inputs.data(), args.size(), outputs.data(),
                           rets.size(), state->config.data(), state->config.size(), error, sizeof(error));
    error[sizeof(error) - 1] = '\0';
    return status == 0 ? ffi::Error::Success() : ffi::Error::Internal(
        error[0] ? error : "External native GPU operation failed");
  } catch (const std::exception& error) {
    return ffi::Error::Internal(error.what());
  } catch (...) {
    return ffi::Error::Internal("Unknown exception in native GPU operation");
  }
}

#define JB_NATIVE_BINDING ffi::Ffi::Bind() \
    .Ctx<ffi::State<NativeState>>() \
    .Ctx<ffi::PlatformStream<void*>>() .Ctx<ffi::DeviceOrdinal>() \
    .Attrs() .RemainingArgs() .RemainingRets()
XLA_FFI_DEFINE_HANDLER_SYMBOL(JaxbindGpuNativeCall, NativeCall, JB_NATIVE_BINDING);
XLA_FFI_DEFINE_HANDLER_SYMBOL(JaxbindGpuNativeGraphCall, NativeCall, JB_NATIVE_BINDING,
                             {ffi::Traits::kCmdBufferCompatible});

static const char* Dtype(ffi::DataType type) {
  switch (type) {
    case ffi::DataType::PRED: return "bool";
    case ffi::DataType::S8: return "int8";
    case ffi::DataType::S16: return "int16";
    case ffi::DataType::S32: return "int32";
    case ffi::DataType::S64: return "int64";
    case ffi::DataType::U8: return "uint8";
    case ffi::DataType::U16: return "uint16";
    case ffi::DataType::U32: return "uint32";
    case ffi::DataType::U64: return "uint64";
    case ffi::DataType::F16: return "float16";
    case ffi::DataType::F32: return "float32";
    case ffi::DataType::F64: return "float64";
    case ffi::DataType::C64: return "complex64";
    case ffi::DataType::C128: return "complex128";
    default: throw std::invalid_argument("Unsupported GPU buffer dtype");
  }
}

static py::tuple Describe(ffi::AnyBuffer buffer) {
  auto dims = buffer.dimensions();
  py::tuple shape(dims.size());
  for (size_t i = 0; i < dims.size(); ++i) shape[i] = py::int_(dims[i]);
  return py::make_tuple(reinterpret_cast<uintptr_t>(buffer.untyped_data()),
                        buffer.size_bytes(), shape, Dtype(buffer.element_type()));
}

static ffi::Error PythonCall(PyObject* graph_owner, void* stream, int32_t device, int64_t callback_id,
                       std::string_view process_key, std::string_view kwargs_dump,
                       ffi::RemainingArgs args,
                       ffi::RemainingRets rets, int64_t workspace_bytes = 0,
                       void* resource_stream = nullptr) {
  py::gil_scoped_acquire gil;
  try {
    py::tuple inputs(args.size()), outputs(rets.size());
    for (size_t i = 0; i < args.size(); ++i) {
      auto arg = args.get<ffi::AnyBuffer>(i);
      if (!arg.has_value()) return ffi::Error::InvalidArgument("Expected array input");
      inputs[i] = Describe(arg.value());
    }
    for (size_t i = 0; i < rets.size(); ++i) {
      auto ret = rets.get<ffi::AnyBuffer>(i);
      if (!ret.has_value()) return ffi::Error::InvalidArgument("Expected array output");
      outputs[i] = Describe(*ret.value());
    }
    // The runtime is loaded before any callback is registered. Avoid running
    // the import machinery on every invocation, without retaining globals in
    // C++ beyond the interpreter's lifetime.
    auto runtime = py::reinterpret_borrow<py::object>(
        PyDict_GetItemString(PyImport_GetModuleDict(), "jaxbind_gpu._runtime"));
    if (!runtime) runtime = py::module_::import("jaxbind_gpu._runtime");
    runtime.attr("_dispatch")(
        callback_id, py::str(process_key.data(), process_key.size()),
        py::bytes(kwargs_dump.data(), kwargs_dump.size()),
        reinterpret_cast<uintptr_t>(stream), device, inputs, outputs,
        graph_owner ? py::reinterpret_borrow<py::object>(graph_owner) : py::none(), workspace_bytes,
        reinterpret_cast<uintptr_t>(resource_stream));
    return ffi::Error::Success();
  } catch (const py::error_already_set& error) {
    return ffi::Error::Internal(error.what());
  } catch (const std::exception& error) {
    return ffi::Error::InvalidArgument(error.what());
  } catch (...) {
    return ffi::Error::Internal("Unknown exception in jaxbind_gpu callback");
  }
}

static ffi::Error Call(void* stream, int32_t device, int64_t callback_id,
                       std::string_view process_key, std::string_view kwargs_dump,
                       ffi::RemainingArgs args, ffi::RemainingRets rets) {
  return PythonCall(nullptr, stream, device, callback_id, process_key, kwargs_dump,
                    args, rets);
}

static ffi::Error GraphCallWithWorkspace(void* stream, int32_t device,
                            int64_t callback_id, std::string_view process_key,
                            std::string_view kwargs_dump, ffi::RemainingArgs args,
                            ffi::RemainingRets rets, int64_t workspace_bytes,
                            bool* cacheable = nullptr, void* resource_stream = nullptr) {
  // cudaMalloc does not become a graph node. Allow it during capture and keep
  // its owner attached to the graph. Async allocation nodes cannot be nested in
  // XLA command buffers. This does not permit synchronizing a capturing stream.
  using Exchange = int (*)(int*);
  using CaptureInfo = int (*)(void*, int*, uint64_t*, void**, void*, size_t*);
  using UserCreate = int (*)(void**, void*, void (*)(void*), unsigned int, unsigned int);
  using UserRetain = int (*)(void*, void*, unsigned int, unsigned int);
  using UserRelease = int (*)(void*, unsigned int);
  static void* driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  static auto exchange = driver ? reinterpret_cast<Exchange>(
      dlsym(driver, "cuThreadExchangeStreamCaptureMode")) : nullptr;
  static auto capture_info = driver ? reinterpret_cast<CaptureInfo>(dlsym(driver, "cuStreamGetCaptureInfo_v2")) : nullptr;
  static auto create_user = driver ? reinterpret_cast<UserCreate>(dlsym(driver, "cuUserObjectCreate")) : nullptr;
  static auto retain_user = driver ? reinterpret_cast<UserRetain>(dlsym(driver, "cuGraphRetainUserObject")) : nullptr;
  static auto release_user = driver ? reinterpret_cast<UserRelease>(dlsym(driver, "cuUserObjectRelease")) : nullptr;
  if (!exchange || !capture_info || !create_user || !retain_user || !release_user)
    return ffi::Error::Internal("CUDA graph lifetime API unavailable");
  int status = 0;
  void* graph = nullptr;
  if (capture_info(stream, &status, nullptr, &graph, nullptr, nullptr))
    return ffi::Error::Internal("Cannot query CUDA capture graph");
  py::gil_scoped_acquire gil;
  py::object owner = py::none();
  if (status == 1) {
    owner = py::list();
    Py_INCREF(owner.ptr());  // transferred to the CUDA user object
    void* user = nullptr;
    if (create_user(&user, owner.ptr(), RetireGraph, 1, 1)) {
      Py_DECREF(owner.ptr());
      return ffi::Error::Internal("Cannot create CUDA graph resource owner");
    }
    if (retain_user(graph, user, 1, 1)) {  // CU_GRAPH_USER_OBJECT_MOVE
      release_user(user, 1);
      return ffi::Error::Internal("Cannot attach resources to CUDA graph");
    }
  }
  int mode = 2;  // CU_STREAM_CAPTURE_MODE_RELAXED
  if (exchange(&mode)) return ffi::Error::Internal("Cannot set CUDA capture mode");
  auto error = PythonCall(status == 1 ? owner.ptr() : nullptr, stream, device, callback_id, process_key,
                          kwargs_dump, args, rets, workspace_bytes, resource_stream);
  if (cacheable && error.success()) {
    try {
      *cacheable = true;
      for (auto context : owner)
        *cacheable = *cacheable && context.attr("_can_cache_graph")().cast<bool>();
    } catch (const py::error_already_set& exception) {
      error = ffi::Error::Internal(exception.what());
    }
  }
  if (exchange(&mode)) return ffi::Error::Internal("Cannot restore CUDA capture mode");
  return error;
}

#include "graph_cache.h"

static ffi::Error GraphCall(CallbackGraphState* state, void* stream, int32_t device,
                            int64_t callback_id, std::string_view process_key,
                            std::string_view kwargs_dump, ffi::RemainingArgs args,
                            ffi::RemainingRets rets) {
  return CachedGraphCall(state, stream, device, callback_id, process_key,
                         kwargs_dump, args, rets, 0);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    JaxbindGpuGraphCall, GraphCall,
    ffi::Ffi::Bind()
        .Ctx<ffi::State<CallbackGraphState>>()
        .Ctx<ffi::PlatformStream<void*>>().Ctx<ffi::DeviceOrdinal>()
        .Attr<int64_t>("callback_id").Attr<std::string_view>("process_key")
        .Attr<std::string_view>("kwargs_dump").RemainingArgs().RemainingRets(),
    {ffi::Traits::kCmdBufferCompatible});

static ffi::Error WorkspaceCall(void* stream, int32_t device,
                               int64_t callback_id, std::string_view process_key,
                               std::string_view kwargs_dump, int64_t workspace_bytes,
                               ffi::RemainingArgs args, ffi::RemainingRets rets) {
  return PythonCall(nullptr, stream, device, callback_id, process_key, kwargs_dump,
                    args, rets, workspace_bytes);
}
static ffi::Error GraphWorkspaceCall(void* stream, int32_t device,
                                    int64_t callback_id, std::string_view process_key,
                                    std::string_view kwargs_dump, int64_t workspace_bytes,
                                    ffi::RemainingArgs args, ffi::RemainingRets rets,
                                    CallbackGraphState* state) {
  return CachedGraphCall(state, stream, device, callback_id, process_key, kwargs_dump,
                   args, rets, workspace_bytes);
}
#define JB_WORKSPACE_BIND \
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<void*>>().Ctx<ffi::DeviceOrdinal>() \
      .Attr<int64_t>("callback_id").Attr<std::string_view>("process_key") \
      .Attr<std::string_view>("kwargs_dump").Attr<int64_t>("workspace_bytes") \
      .RemainingArgs().RemainingRets()
XLA_FFI_DEFINE_HANDLER_SYMBOL(JaxbindGpuWorkspaceCall, WorkspaceCall, JB_WORKSPACE_BIND);
XLA_FFI_DEFINE_HANDLER_SYMBOL(JaxbindGpuGraphWorkspaceCall, GraphWorkspaceCall,
                             JB_WORKSPACE_BIND.Ctx<ffi::State<CallbackGraphState>>(),
                             {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    JaxbindGpuCall, Call,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<void*>>()
        .Ctx<ffi::DeviceOrdinal>()
        .Attr<int64_t>("callback_id")
        .Attr<std::string_view>("process_key")
        .Attr<std::string_view>("kwargs_dump")
        .RemainingArgs()
        .RemainingRets());

PYBIND11_MODULE(_bridge, module) {
  module.def("release_recording_streams", &ReleaseRecordingStreams);
  module.def("recording_stream_count", []() {
    auto& registry = Recordings();
    std::lock_guard<std::mutex> lock(registry.mutex);
    return registry.streams.size();
  });
  module.def("callback_graph_type", []() {
    py::dict type;
    type["type_id"] = py::capsule(&CallbackGraphState::id);
    type["type_info"] = py::capsule(const_cast<XLA_FFI_TypeInfo*>(&callback_graph_type_info));
    return type;
  });
  module.def("native_type", []() {
    py::dict type;
    type["type_id"] = py::capsule(&NativeState::id);
    type["type_info"] = py::capsule(const_cast<XLA_FFI_TypeInfo*>(&native_type_info));
    return type;
  });
  module.def("kernel_address", []() { return reinterpret_cast<uintptr_t>(JbLaunchKernel); });
  module.def("take_retired_graphs", []() {
    std::deque<PyObject*> owners;
    {
      auto& retired = Retired();
      std::lock_guard<std::mutex> lock(retired.mutex);
      owners.swap(retired.owners);
    }
    py::list result;
    for (auto* owner : owners) result.append(py::reinterpret_steal<py::object>(owner));
    return result;
  });
  module.def("register_native", [](uintptr_t pointer, std::string process_key) {
    if (pointer == 0) throw std::invalid_argument("A native function pointer cannot be null");
    std::lock_guard<std::mutex> lock(native_mutex);
    int64_t id = ++next_native_id;
    native_operations.emplace(id, NativeEntry{reinterpret_cast<jb_gpu_operation>(pointer),
                                              std::move(process_key)});
    return id;
  });
  module.def("capsule_address", [](py::capsule capsule) {
    return reinterpret_cast<uintptr_t>(capsule.get_pointer());
  });
  module.def("registrations", []() {
    py::dict registrations;
    registrations["jaxbind_gpu_pycall"] = py::capsule(
        reinterpret_cast<void*>(JaxbindGpuCall), "xla._CUSTOM_CALL_TARGET");
    registrations["jaxbind_gpu_native"] = py::capsule(
        reinterpret_cast<void*>(JaxbindGpuNativeCall), "xla._CUSTOM_CALL_TARGET");
    registrations["jaxbind_gpu_native_graph"] = py::capsule(
        reinterpret_cast<void*>(JaxbindGpuNativeGraphCall), "xla._CUSTOM_CALL_TARGET");
    for (auto name : {"jaxbind_gpu_native", "jaxbind_gpu_native_graph"}) {
      py::dict handlers;
      handlers["execute"] = registrations[name];
      handlers["instantiate"] = py::capsule(
          reinterpret_cast<void*>(JaxbindGpuNativeInstantiate), "xla._CUSTOM_CALL_TARGET");
      registrations[name] = handlers;
    }
    registrations["jaxbind_gpu_pycall_graph"] = py::capsule(
        reinterpret_cast<void*>(JaxbindGpuGraphCall), "xla._CUSTOM_CALL_TARGET");
    registrations["jaxbind_gpu_pycall_workspace"] = py::capsule(
        reinterpret_cast<void*>(JaxbindGpuWorkspaceCall), "xla._CUSTOM_CALL_TARGET");
    registrations["jaxbind_gpu_pycall_graph_workspace"] = py::capsule(
        reinterpret_cast<void*>(JaxbindGpuGraphWorkspaceCall), "xla._CUSTOM_CALL_TARGET");
    for (auto name : {"jaxbind_gpu_pycall_graph", "jaxbind_gpu_pycall_graph_workspace"}) {
      py::dict handlers;
      handlers["execute"] = registrations[name];
      handlers["instantiate"] = py::capsule(
          reinterpret_cast<void*>(JaxbindGpuGraphInstantiate), "xla._CUSTOM_CALL_TARGET");
      registrations[name] = handlers;
    }
    return registrations;
  });
}
