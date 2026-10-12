// SPDX-License-Identifier: BSD-2-Clause
// The installed cuFINUFFT Python package supplies the actual ABI layout and
// symbol addresses. No cuFINUFFT or CUDA headers/link libraries are required.
#include "native.h"
#include <dlfcn.h>
#include <array>
#include <climits>
#include <cstdio>
#include <cstring>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
using MakeD = int (*)(int, int, int64_t*, int, int, double, void**, void*);
using MakeF = int (*)(int, int, int64_t*, int, int, float, void**, void*);
using SetPts = int (*)(void*, int64_t, void*, void*, void*, int64_t, void*, void*, void*);
using Execute = int (*)(void*, void*, void*);
using Destroy = int (*)(void*);

struct Driver {
  void* library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  int (*current)(void**) = symbol<decltype(current)>("cuCtxGetCurrent");
  int (*push)(void*) = symbol<decltype(push)>("cuCtxPushCurrent_v2");
  int (*pop)(void**) = symbol<decltype(pop)>("cuCtxPopCurrent_v2");
  int (*create)(void**, unsigned) = symbol<decltype(create)>("cuEventCreate");
  int (*record)(void*, void*) = symbol<decltype(record)>("cuEventRecord");
  int (*wait)(void*) = symbol<decltype(wait)>("cuEventSynchronize");
  int (*destroy)(void*) = symbol<decltype(destroy)>("cuEventDestroy_v2");
  int (*allocate)(uint64_t*, size_t) = symbol<decltype(allocate)>("cuMemAlloc_v2");
  int (*release)(uint64_t) = symbol<decltype(release)>("cuMemFree_v2");
  int (*copy)(uint64_t, uint64_t, size_t, void*) = symbol<decltype(copy)>("cuMemcpyDtoDAsync_v2");
  template <class T> T symbol(const char* name) {
    void* ptr = library ? dlsym(library, name) : nullptr;
    if (!ptr) throw std::runtime_error(std::string("CUDA driver symbol unavailable: ") + name);
    return reinterpret_cast<T>(ptr);
  }
};
Driver& driver() { static Driver* value = new Driver; return *value; }
void Check(int status, const char* step) {
  if (status) throw std::runtime_error(std::string(step) + " failed with status " + std::to_string(status));
}

struct Plan {
  void* plan = nullptr;
  void* context = nullptr;
  void* event = nullptr;
  SetPts setpts;
  Execute execute;
  Destroy destroy;
  std::mutex mutex;
  std::vector<uint64_t> geometry;
  bool geometry_ready = false;
  Plan(const jb_gpu_context& ctx, const std::array<uint64_t, 17>& h,
       const char* options, int ntrans, bool single)
      : setpts(reinterpret_cast<SetPts>(h[13])),
        execute(reinterpret_cast<Execute>(h[14])),
        destroy(reinterpret_cast<Destroy>(h[15])) {
    Check(driver().current(&context), "CUDA context lookup");
    if (!context) throw std::runtime_error("cuFINUFFT requires an active CUDA context");
    std::vector<uint64_t> opts((h[9] + 7)/8);
    std::memcpy(opts.data(), options, h[9]);
    auto* bytes = reinterpret_cast<char*>(opts.data());
    std::memcpy(bytes + h[10], &ctx.device, sizeof(ctx.device));
    std::memcpy(bytes + h[11], &ctx.stream, sizeof(ctx.stream));
    int64_t modes[3] = {1, 1, 1};
    int ndim = static_cast<int>(h[2]);
    for (int d = 0; d < ndim; ++d) modes[d] = h[4+ndim-1-d];
    double eps; std::memcpy(&eps, &h[8], sizeof(eps));
    int sign = static_cast<int>(static_cast<int64_t>(h[7]));
    int status = single
        ? reinterpret_cast<MakeF>(h[12])(h[1], ndim, modes, sign, ntrans, eps, &plan, opts.data())
        : reinterpret_cast<MakeD>(h[12])(h[1], ndim, modes, sign, ntrans, eps, &plan, opts.data());
    if (status) {
      if (plan) destroy(plan);
      plan = nullptr;
      Check(status, "cuFINUFFT makeplan");
    }
    int event_status = driver().create(&event, 2); // CU_EVENT_DISABLE_TIMING
    if (event_status) { destroy(plan); plan = nullptr; Check(event_status, "CUDA event creation"); }
  }
  void synchronize() {
    std::lock_guard<std::mutex> lock(mutex);
    Check(driver().push(context), "CUDA context activation");
    int status = driver().wait(event);
    void* previous = nullptr; int pop_status = driver().pop(&previous);
    Check(status, "CUDA event synchronization"); Check(pop_status, "CUDA context restoration");
  }
  ~Plan() {
    // Cleanup never throws across shared_ptr destruction. Explicit waits report
    // CUDA failures through the public synchronization function.
    if (!plan || driver().push(context)) return;
    driver().wait(event);
    destroy(plan);
    for (auto pointer : geometry) driver().release(pointer);
    driver().destroy(event);
    void* previous = nullptr; driver().pop(&previous);
  }
};

struct Cache {
  std::mutex mutex;
  size_t limit = 128;
  std::list<std::string> order;
  struct Entry { std::shared_ptr<Plan> plan; std::list<std::string>::iterator position; };
  std::unordered_map<std::string, Entry> entries;
};
Cache& cache() { static Cache* value = new Cache; return *value; }
}

int JbCufinufft(const jb_gpu_context* ctx, const jb_gpu_buffer* inputs, size_t ni,
               jb_gpu_buffer* outputs, size_t no, const void* config, size_t size,
               char* error, size_t capacity) {
  try {
    std::array<uint64_t, 17> h;
    if (size < sizeof(h)) throw std::invalid_argument("Invalid cuFINUFFT configuration");
    std::memcpy(h.data(), config, sizeof(h));
    const int kind = h[1], ndim = h[2];
    if (ctx->abi_version != JB_GPU_ABI_VERSION || h[0] != 1 || kind < 1 || kind > 3 ||
        ndim < 1 || ndim > 3 || ni != size_t((kind == 3 ? 2 : 1)*ndim + 1) || no != 1 ||
        h[9] > size-sizeof(h) || h[10]+sizeof(int32_t) > h[9] || h[11]+sizeof(void*) > h[9] ||
        !h[12] || !h[13] || !h[14] || !h[15])
      throw std::invalid_argument("Invalid cuFINUFFT configuration or buffer count");
    const auto& data = inputs[ni-1];
    bool single = data.dtype == JB_GPU_C64;
    if (!single && data.dtype != JB_GPU_C128) throw std::invalid_argument("cuFINUFFT requires complex data");
    size_t core = kind == 2 ? h[4]*h[5]*h[6] : inputs[h[3] && kind == 3 ? ndim : 0].shape[0];
    if (!core) throw std::invalid_argument("cuFINUFFT input cannot be empty");
    size_t ntrans = data.nbytes / ((single ? 8 : 16)*core);
    if (ntrans == 0 || ntrans > INT_MAX) throw std::invalid_argument("Invalid cuFINUFFT batch size");
    std::string key(static_cast<const char*>(config), size);
    key.append(reinterpret_cast<const char*>(&ctx->device), sizeof(ctx->device));
    key.append(reinterpret_cast<const char*>(&ctx->stream), sizeof(ctx->stream));
    key.append(reinterpret_cast<const char*>(&ntrans), sizeof(ntrans));
    std::shared_ptr<Plan> plan, evicted;
    {
      auto& c = cache(); std::lock_guard<std::mutex> lock(c.mutex);
      auto found = c.entries.find(key);
      if (found != c.entries.end()) {
        plan = found->second.plan;
        c.order.splice(c.order.end(), c.order, found->second.position);
      } else {
        plan = std::make_shared<Plan>(*ctx, h, static_cast<const char*>(config)+sizeof(h), ntrans, single);
        if (c.limit) {
          c.order.push_back(key);
          c.entries.emplace(key, Cache::Entry{plan, std::prev(c.order.end())});
          if (c.entries.size() > c.limit) {
            auto old = c.entries.find(c.order.front()); evicted = std::move(old->second.plan);
            c.entries.erase(old); c.order.pop_front();
          }
        }
      }
    }
    evicted.reset();
    std::lock_guard<std::mutex> lock(plan->mutex);
    void* source[3] = {}, *target[3] = {};
    size_t source_start = kind == 3 && h[3] ? ndim : 0;
    size_t target_start = kind == 3 && h[3] ? 0 : ndim;
    for (int d = 0; d < ndim; ++d) {
      source[d] = inputs[source_start+ndim-1-d].data;
      if (kind == 3) target[d] = inputs[target_start+ndim-1-d].data;
    }
    int64_t m = inputs[source_start].shape[0];
    int64_t n = kind == 3 ? inputs[target_start].shape[0] : 0;
    try {
      if (!h[16] || !plan->geometry_ready) {
        if (h[16]) {
          // Prepared geometry owns its coordinates. Cached plans must never
          // retain pointers to XLA input allocations between executions.
          size_t point_arrays = ni-1;
          for (size_t i = 0; i < point_arrays; ++i) {
            uint64_t pointer = 0;
            Check(driver().allocate(&pointer, inputs[i].nbytes), "CUDA geometry allocation");
            plan->geometry.push_back(pointer);
            Check(driver().copy(pointer, reinterpret_cast<uint64_t>(inputs[i].data),
                                inputs[i].nbytes, ctx->stream), "CUDA geometry copy");
          }
          for (int d = 0; d < ndim; ++d) {
            source[d] = reinterpret_cast<void*>(plan->geometry[source_start+ndim-1-d]);
            if (kind == 3) target[d] = reinterpret_cast<void*>(plan->geometry[target_start+ndim-1-d]);
          }
        }
        Check(plan->setpts(plan->plan, m, source[0], source[1], source[2], n,
                          target[0], target[1], target[2]), "cuFINUFFT setpts");
        plan->geometry_ready = bool(h[16]);
      }
      Check(kind == 2 ? plan->execute(plan->plan, outputs[0].data, data.data)
                      : plan->execute(plan->plan, data.data, outputs[0].data), "cuFINUFFT execute");
      Check(driver().record(plan->event, ctx->stream), "CUDA event recording");
    } catch (...) {
      // Report errors only after any work already queued by this call finishes.
      if (!driver().record(plan->event, ctx->stream)) driver().wait(plan->event);
      // Do not reuse partially initialized geometry or a failed library plan.
      auto& c = cache();
      std::lock_guard<std::mutex> cache_lock(c.mutex);
      auto found = c.entries.find(key);
      if (found != c.entries.end() && found->second.plan == plan) {
        c.order.erase(found->second.position);
        c.entries.erase(found);
      }
      throw;
    }
    return 0;
  } catch (const std::exception& exception) {
    std::snprintf(error, capacity, "%s", exception.what()); return 1;
  }
}

void JbCufinufftSynchronize(bool clear) {
  std::vector<std::shared_ptr<Plan>> plans;
  {
    auto& c = cache(); std::lock_guard<std::mutex> lock(c.mutex);
    for (auto& item : c.entries) plans.push_back(item.second.plan);
    if (clear) { c.entries.clear(); c.order.clear(); }
  }
  for (auto& plan : plans) plan->synchronize();
}
void JbCufinufftConfigure(size_t limit) {
  JbCufinufftSynchronize(true);
  auto& c = cache(); std::lock_guard<std::mutex> lock(c.mutex); c.limit = limit;
}
size_t JbCufinufftCached() {
  auto& c = cache(); std::lock_guard<std::mutex> lock(c.mutex); return c.entries.size();
}
