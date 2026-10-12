# GPU execution and memory

## Choose an execution interface

| Interface | Operation implementation | Execution overhead | Export |
| --- | --- | --- | --- |
| Python callback | Python library accepting CuPy, CUDA array interface, or pointers | Python GIL and callback dispatch | Process-local |
| Native C ABI | Small C/C++ wrapper around a library or kernel | Direct native call, no Python GIL | Process-local |
| Named typed XLA FFI | Existing typed FFI handler | Direct native call, no Python GIL | Portable if the same target and attributes exist in each process |

The transformation rules are shared by these interfaces. A library does not
need a separate JAX integration just to obtain GPU buffers, dispatch on XLA's
stream, or define differentiation and batching.

### Python callback

```python
import cupy as cp
import jax
import jax.numpy as jnp
from jaxbind_gpu import bind
from jaxbind_gpu import advanced as jb


def scale(out, args, kwargs_dump):
    factor = jb.load_kwargs(kwargs_dump)["factor"]
    cp.multiply(args[0], factor, out=out[0])


def abstract(x, **kwargs):
    return ((x.shape, x.dtype),)


scale_jax = jb.get_linear_call(scale, scale, abstract, abstract)
x = jnp.arange(8, dtype=jnp.float32)
y = jax.jit(lambda x: scale_jax(x, factor=3.0)[0])(x)
gradient = jax.grad(lambda x: scale_jax(x, factor=3.0)[0].sum())(x)
```

A callback receives tuples of output and input arrays, writes every output, and
returns `None`. Its abstract function returns one `(shape, dtype)` pair per
output. Bound calls return a sequence even for one output.

CuPy's current stream orders the callback's work with XLA. Pass
`jb.current_context().stream.ptr` to a library with an explicit stream argument.
Use `array.data.ptr` for raw pointer APIs. A library that launches on another
stream must establish the required CUDA event dependencies.

Callbacks enqueue work asynchronously. They must keep temporary allocations and
library resources alive until that work completes:

```python
def fft(out, args, kwargs_dump):
    temporary = cp.fft.fft(args[0])
    cp.copyto(out[0], temporary)
    jb.current_context().retain(temporary)
```

`context.cached(key, factory)` reuses a resource per callback, device, and stream.
The runtime retains it until the submitted work completes. Do not put borrowed
input/output views into this cache. `retain` also accepts workspaces and plans.
The runtime uses CUDA completion events instead of synchronizing each successful
call. Resource eviction and explicit cleanup may wait for unfinished work.

Inputs are read-only unless the binding explicitly declares an input/output
alias. Borrowed buffers and the borrowed stream are valid for submitted work,
and must not become persistent user references. Static keyword arguments arrive
as serialized host metadata. Treat bindings and their configuration as trusted
code.

### Native C ABI

Implement `jb_gpu_operation` from
[`jaxbind_gpu.h`](../src/jaxbind_gpu/include/jaxbind_gpu.h). It receives the device,
CUDA stream, input/output pointers, shapes, dtype codes, and configuration bytes.
No JAX or CUDA headers are required to use this declaration.

```python
import ctypes
from jaxbind_gpu import advanced as jb

library = ctypes.CDLL("/absolute/path/to/library_wrapper.so")
function = jb.NativeFunction.from_cabi(library.operation, owner=library)
op = jb.get_call(function, lambda x: ((x.shape, x.dtype),))
```

`encode(**kwargs)` can encode static configuration into bytes. The function,
shared library, and explicit owner stay alive for the process lifetime. Native
execution acquires no Python GIL and does not construct CuPy objects. A wrapper
can call a library's C API or launch an already compiled kernel directly on the
provided stream. It is responsible for its own workspace and plan lifetimes.
Return zero on success; otherwise write an error message and return nonzero.

[`examples/native_scale.py`](../examples/native_scale.py) and its C++ wrapper are a
complete example. They compile a kernel once with NVRTC, then execute and
differentiate it through the native interface.

Set `graph_compatible=True` only when the operation exclusively enqueues CUDA
work that can be captured. Allocation, synchronization, stream queries, and
host behavior required on every invocation are incompatible with this promise.
The tests force XLA command-buffer capture and verify that replay avoids native
host dispatch.

### CUDA kernels through the same factory

A CuPy `RawKernel` can use `bind(..., launch=...)` directly. CuPy/NVRTC compiles
it at binding time; execution uses the CUDA driver through the native bridge.
It acquires no Python GIL, constructs no CuPy views, and allocates no GPU
staging buffers or workspace in the framework:

```python
kernel = cp.RawKernel(r'''
extern "C" __global__ void scale(const float* x, float* y, int n, float a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i];
}
''', "scale")
op = bind(kernel,
          output=lambda x, **kw: jax.ShapeDtypeStruct(x.shape, x.dtype),
          launch=(lambda x, **kw: ((x.size + 255) // 256,), (256,)),
          scalars=lambda x, *, factor: (np.int32(x.size), np.float32(factor)),
          jvp=lambda p, t, *, factor: factor * t[0])
y = jax.jit(lambda x: op(x, factor=2.5))(x)
```

Import NumPy as `np` for the explicitly typed scalar arguments. By default,
kernel parameters are input pointers, output pointers, then scalar values.
`argument_order=(("output", 0), ("input", 0), ...)` supports other layouts,
and `shared_mem=` supplies dynamic shared-memory bytes. Grid dimensions and
scalars can depend on abstract shapes and static keywords. Bind a compiled
kernel with its device current; the framework rejects execution on another
device. `NativeFunction.from_cuda_kernel` also accepts an owned integer
`CUfunction` handle for kernels compiled elsewhere.

### Python callback graph replay

Capture-safe Python/CuPy functions can opt in with `graph_compatible=True` on
`bind`, `get_call`, or `get_linear_call`. Their Python code records CUDA work
during capture; replay runs those recorded nodes. Host actions needed for each
invocation, synchronizing a capturing stream, and device-to-host queries violate
this contract. CuPy currently rejects its cuBLAS wrapper during capture; use a
native integration for that path. Direct-output Torch functions can also replay
borrowed GPU tensors. Owned Torch allocations still disable recording reuse;
the framework does not replace Torch's allocator.

```python
op = bind(cp.square,
          output=lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
          out=True, graph_compatible=True)
compiled = jax.jit(op)
y = compiled(x)
```

CuPy allocator scratch follows the actual CUDA graph lifetime, including
XLA's trace-cache entries, and releases after graph retirement. Retain resources
outside that allocator explicitly. Graph replay can increase memory and lose
its speed benefit when changing output addresses cause repeated capture.
For recurring addresses, the framework reuses up to 64 recordings
per compiled call. The cache includes every input, output and workspace address.
Changed values at those addresses still execute the recorded CUDA work. It
shares CuPy temporaries between recordings on the same execution stream. The
scratch pool is bounded by `max_pending_bytes` per callback/device/stream and
survives only while graphs reference it. Recordings reuse scratch in stream
order; they do not allocate a private copy for every buffer address. Explicit
XLA workspaces avoid that pool. Cached library resources can also be shared
across recordings. Scratch beyond the pool bound, owned allocations from other
allocators, or unknown retained owners disable reuse. Under ordinary `jit`,
those callbacks return to Python execution after the
first unsuccessful cache admission. Under XLA capture they recapture as needed.
Recordings retire with the executable. CUDA graph metadata consumes additional
driver memory even when the callback allocates no scratch.

Ordinary `jax.jit` warms the callback once, then records eligible calls and
replays recurring addresses through native CUDA graph launches. No compiler
options are required. XLA command-buffer capture can also embed the recordings
in a larger compiled graph.

Warmup and recording use the same managed stream, with dependencies on XLA's
execution stream. Use `current_context().stream` for every library launch and
let `context.cached` scope resources to the originating execution stream.
Plans that keep their creation stream remain valid across executable
recompilations. Managed streams retire with the callback, after retained plans
and queued work. Reuse avoids Python dispatch, but CUDA graph cloning and
updates still cost host time when XLA's trace cache misses. New addresses require
recording. The cache cannot guarantee a fixed overhead. Compiled unrolled loops often
avoid repeated host work. Benchmark scripts report captures and memory
separately; graph replay is opt-in.

### Named typed FFI

```python
function = jb.NativeFunction(
    "my_library.operation.v1",
    jax.ffi.pycapsule(library.typed_handler),
    owner=library,
)
op = jb.get_call(function, abstract)
```

`attributes(**kwargs)` converts static configuration to typed FFI attributes.
Stable names and portable attributes support JAX export and executable reuse.
Every consuming process must register the same target before execution. The
export test serializes an operation and executes it in a fresh process. Explicit
JAX export safety approval is required for an application-defined custom target.
C ABI bindings with function addresses and Python callbacks carry a process key
and reject use from another process.

## Resource control

```python
jb.configure_runtime(max_cache_entries=128, max_pending_calls=256,
                     max_pending_bytes=256 << 20)
# ... submit and finish computations ...
jb.synchronize(clear_cache=True)
print(jb.runtime_stats())
```

The limits bound cache entry counts and outstanding Python callback lifetimes.
The byte limit applies pressure to known retained CuPy/Torch allocations.
Opaque workspaces can declare their size with `context.retain(resource, nbytes=...)`.
It is not a cap on total VRAM: persistent plans, graph captures, and XLA outputs
have separate lifetimes, and a single call can exceed the threshold before it
is submitted. Shared retained storage can be counted once per outstanding call.
Set the entry limit to zero to disable Python resource caching. Native adapters
manage their own caches and cleanup. Cleanup may synchronize queued work; call
it after JAX results are ready and without concurrent submissions. Each stream
has a separate resource cache, and callbacks on one stream serialize access to
cached mutable resources. With caching enabled and no eviction, successful
calls do not explicitly synchronize the stream.
