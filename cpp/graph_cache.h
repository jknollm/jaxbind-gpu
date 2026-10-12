// SPDX-License-Identifier: BSD-2-Clause
// Cache callback recordings and their shared scratch at the executable's lifetime.
#pragma once

#include <list>
#include <map>
#include <memory>
#include <tuple>

struct GraphDriver {
  void* library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  int (*capture_info)(void*, int*, uint64_t*, void**, void***, size_t*) = nullptr;
  int (*begin)(void*, int) = nullptr;
  int (*end)(void*, void**) = nullptr;
  int (*stream_create)(void**, unsigned) = nullptr;
  int (*stream_destroy)(void*) = nullptr;
  int (*graph_destroy)(void*) = nullptr;
  int (*instantiate)(void**, void*, unsigned long long) = nullptr;
  int (*launch)(void*, void*) = nullptr;
  int (*exec_destroy)(void*) = nullptr;
  int (*event_create)(void**, unsigned) = nullptr;
  int (*event_destroy)(void*) = nullptr;
  int (*event_record)(void*, void*) = nullptr;
  int (*wait_event)(void*, void*, unsigned) = nullptr;
  int (*add_child)(void**, void*, void* const*, size_t, void*) = nullptr;
  int (*dependencies)(void*, void* const*, size_t, unsigned) = nullptr;
  int (*context_get)(void**) = nullptr;
  int (*context_push)(void*) = nullptr;
  int (*context_pop)(void**) = nullptr;

  GraphDriver() {
#define JB_DRIVER_FIELD(field, name) field = reinterpret_cast<decltype(field)>(library ? dlsym(library, name) : nullptr)
    JB_DRIVER_FIELD(capture_info, "cuStreamGetCaptureInfo_v2");
    JB_DRIVER_FIELD(begin, "cuStreamBeginCapture");
    JB_DRIVER_FIELD(end, "cuStreamEndCapture");
    JB_DRIVER_FIELD(stream_create, "cuStreamCreate");
    JB_DRIVER_FIELD(stream_destroy, "cuStreamDestroy_v2");
    JB_DRIVER_FIELD(graph_destroy, "cuGraphDestroy");
    JB_DRIVER_FIELD(instantiate, "cuGraphInstantiateWithFlags");
    JB_DRIVER_FIELD(launch, "cuGraphLaunch");
    JB_DRIVER_FIELD(exec_destroy, "cuGraphExecDestroy");
    JB_DRIVER_FIELD(event_create, "cuEventCreate");
    JB_DRIVER_FIELD(event_destroy, "cuEventDestroy_v2");
    JB_DRIVER_FIELD(event_record, "cuEventRecord");
    JB_DRIVER_FIELD(wait_event, "cuStreamWaitEvent");
    JB_DRIVER_FIELD(add_child, "cuGraphAddChildGraphNode");
    JB_DRIVER_FIELD(dependencies, "cuStreamUpdateCaptureDependencies");
    JB_DRIVER_FIELD(context_get, "cuCtxGetCurrent");
    JB_DRIVER_FIELD(context_push, "cuCtxPushCurrent_v2");
    JB_DRIVER_FIELD(context_pop, "cuCtxPopCurrent_v2");
#undef JB_DRIVER_FIELD
  }
  bool available() const {
    return capture_info && begin && end && stream_create && stream_destroy &&
           graph_destroy && instantiate && launch && exec_destroy && event_create &&
           event_destroy && event_record && wait_event && add_child && dependencies &&
           context_get && context_push && context_pop;
  }
  static GraphDriver& Get() { static auto* api = new GraphDriver; return *api; }
};

struct GraphContextGuard {
  bool active;
  explicit GraphContextGuard(void* context)
      : active(context && GraphDriver::Get().context_push(context) == 0) {}
  ~GraphContextGuard() {
    if (active) { void* ignored; GraphDriver::Get().context_pop(&ignored); }
  }
};

struct RecordingStream {
  void* stream = nullptr;
  void* context = nullptr;
  void* ready = nullptr;
  void* done = nullptr;
  std::mutex mutex;
  RecordingStream() {
    auto& api = GraphDriver::Get();
    if (!api.context_get(&context) && !api.stream_create(&stream, 1) &&
        !api.event_create(&ready, 2) && !api.event_create(&done, 2)) return;
    if (done) api.event_destroy(done);
    if (ready) api.event_destroy(ready);
    if (stream) api.stream_destroy(stream);
    throw std::runtime_error("Cannot create callback recording stream");
  }
  ~RecordingStream() {
    GraphContextGuard guard(context);
    if (guard.active) {
      GraphDriver::Get().event_destroy(done);
      GraphDriver::Get().event_destroy(ready);
      GraphDriver::Get().stream_destroy(stream);
    }
  }
};

using RecordingKey = std::tuple<int64_t, int32_t, void*>;
struct RecordingStreams {
  std::mutex mutex;
  std::map<RecordingKey, std::shared_ptr<RecordingStream>> streams;
};
static RecordingStreams& Recordings() {
  static auto* recordings = new RecordingStreams;
  return *recordings;
}
static std::shared_ptr<RecordingStream> SharedRecordingStream(
    int64_t callback, int32_t device, void* stream) {
  auto& registry = Recordings();
  std::lock_guard<std::mutex> lock(registry.mutex);
  auto& recording = registry.streams[{callback, device, stream}];
  if (!recording) recording = std::make_shared<RecordingStream>();
  return recording;
}
static void ReleaseRecordingStreams(int64_t callback) {
  std::vector<std::shared_ptr<RecordingStream>> retired;
  auto& registry = Recordings();
  {
    std::lock_guard<std::mutex> lock(registry.mutex);
    for (auto it = registry.streams.begin(); it != registry.streams.end();)
      if (std::get<0>(it->first) == callback) {
        retired.push_back(std::move(it->second));
        it = registry.streams.erase(it);
      } else ++it;
  }
}

struct CallbackGraphState {
  static ffi::TypeId id;
  struct Entry {
    void* stream;
    int32_t device;
    std::vector<uintptr_t> buffers;
    void* graph;
    void* executable;
    void* context;
  };
  std::mutex mutex;
  std::list<Entry> entries;
  std::map<std::pair<int32_t, void*>, std::shared_ptr<RecordingStream>> streams;
  bool disabled = false;
  bool warmed = false;
  void ClearRecordings() {
    for (auto& entry : entries) {
      GraphContextGuard guard(entry.context);
      if (guard.active) {
        if (entry.executable) GraphDriver::Get().exec_destroy(entry.executable);
        GraphDriver::Get().graph_destroy(entry.graph);
      }
    }
    entries.clear();
  }
  ~CallbackGraphState() { ClearRecordings(); }
};
ffi::TypeId CallbackGraphState::id = {};
static constexpr auto callback_graph_type_info = ffi::MakeTypeInfo<CallbackGraphState>();

static ffi::ErrorOr<std::unique_ptr<CallbackGraphState>> InstantiateCallbackGraph() {
  return std::make_unique<CallbackGraphState>();
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(JaxbindGpuGraphInstantiate, InstantiateCallbackGraph,
                             ffi::Ffi::BindInstantiate());

static ffi::Error AttachCallbackGraph(void* stream, void* child) {
  auto& api = GraphDriver::Get();
  int status = 0;
  void* parent = nullptr;
  void** dependencies = nullptr;
  size_t count = 0;
  if (api.capture_info(stream, &status, nullptr, &parent, &dependencies, &count) || status != 1)
    return ffi::Error::Internal("Cannot query callback graph capture dependencies");
  void* node = nullptr;
  if (api.add_child(&node, parent, dependencies, count, child) ||
      api.dependencies(stream, &node, 1, 1))
    return ffi::Error::Internal("Cannot insert callback graph with stream dependencies");
  return ffi::Error::Success();
}

static ffi::Error CachedGraphCall(CallbackGraphState* state, void* stream, int32_t device,
                                 int64_t callback_id, std::string_view process_key,
                                 std::string_view kwargs_dump, ffi::RemainingArgs args,
                                 ffi::RemainingRets rets, int64_t workspace_bytes) {
  auto& api = GraphDriver::Get();
  if (!api.available())
    return GraphCallWithWorkspace(stream, device, callback_id, process_key, kwargs_dump, args, rets, workspace_bytes);
  // Never wait for a native lock while holding the GIL. Different FFI states
  // share a recording stream and serialize captures of mutable library plans.
  std::unique_ptr<py::gil_scoped_release> release;
  if (PyGILState_Check()) release = std::make_unique<py::gil_scoped_release>();
  std::lock_guard<std::mutex> lock(state->mutex);
  int status = 0;
  if (api.capture_info(stream, &status, nullptr, nullptr, nullptr, nullptr))
    return ffi::Error::Internal("Cannot query callback capture status");
  auto execute = [&](void* graph, void** executable) {
    if (status == 1) return AttachCallbackGraph(stream, graph);
    if (!*executable && api.instantiate(executable, graph, 0))
      return ffi::Error::Internal("Cannot instantiate callback graph");
    return api.launch(*executable, stream) ? ffi::Error::Internal("Cannot launch callback graph") : ffi::Error::Success();
  };
  try {
    auto key = std::make_pair(device, stream);
    auto& recording = state->streams[key];
    if (!recording) recording = SharedRecordingStream(callback_id, device, stream);
    std::lock_guard<std::mutex> record_lock(recording->mutex);
    if (status != 1 && (!state->warmed || state->disabled)) {
      if (api.event_record(recording->ready, stream) ||
          api.wait_event(recording->stream, recording->ready, 0))
        return ffi::Error::Internal("Cannot order callback warmup inputs");
      auto error = GraphCallWithWorkspace(recording->stream, device, callback_id, process_key,
                                         kwargs_dump, args, rets, workspace_bytes, nullptr, stream);
      if (api.event_record(recording->done, recording->stream) ||
          api.wait_event(stream, recording->done, 0))
        return ffi::Error::Internal("Cannot order callback warmup outputs");
      state->warmed = error.success();
      return error;
    }
    std::vector<uintptr_t> buffers;
    buffers.reserve(args.size() + rets.size());
    for (size_t i = 0; i < args.size(); ++i) {
      auto arg = args.get<ffi::AnyBuffer>(i);
      if (!arg.has_value()) return ffi::Error::InvalidArgument("Expected array input");
      buffers.push_back(reinterpret_cast<uintptr_t>(arg.value().untyped_data()));
    }
    for (size_t i = 0; i < rets.size(); ++i) {
      auto ret = rets.get<ffi::AnyBuffer>(i);
      if (!ret.has_value()) return ffi::Error::InvalidArgument("Expected array output");
      buffers.push_back(reinterpret_cast<uintptr_t>((*ret.value()).untyped_data()));
    }
    for (auto it = state->entries.begin(); !state->disabled && it != state->entries.end(); ++it) {
      if (it->stream == stream && it->device == device && it->buffers == buffers) {
        auto error = execute(it->graph, &it->executable);
        state->entries.splice(state->entries.begin(), state->entries, it);
        return error;
      }
    }
    if (api.begin(recording->stream, 2))
      return ffi::Error::Internal("Cannot begin callback graph recording");
    bool cacheable = false;
    auto error = GraphCallWithWorkspace(recording->stream, device, callback_id, process_key,
                                       kwargs_dump, args, rets, workspace_bytes, &cacheable, stream);
    void* child = nullptr;
    int end_status = api.end(recording->stream, &child);
    if (!error.success() || end_status) {
      if (child) api.graph_destroy(child);
      return error.success() ? ffi::Error::Internal("Cannot finish callback graph recording") : error;
    }
    void* executable = nullptr;
    error = execute(child, &executable);
    if (!error.success() || !cacheable) {
      if (executable) api.exec_destroy(executable);
      api.graph_destroy(child);
      state->disabled = !cacheable;
      if (!cacheable) state->ClearRecordings();
      return error;
    }
    state->entries.push_front({stream, device, std::move(buffers), child, executable, recording->context});
    if (state->entries.size() > 64) {
      auto& oldest = state->entries.back();
      GraphContextGuard guard(oldest.context);
      if (guard.active) {
        if (oldest.executable) api.exec_destroy(oldest.executable);
        api.graph_destroy(oldest.graph);
      }
      state->entries.pop_back();
    }
    return ffi::Error::Success();
  } catch (const std::exception& error) {
    return ffi::Error::Internal(error.what());
  }
}
