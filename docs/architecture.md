# Package structure

The core binds external CUDA operations to JAX. Library-specific algorithms,
plans, and mathematical rules belong in application code or examples.

```text
src/jaxbind_gpu/
    __init__.py         bind and compatible public imports
    advanced.py         Explicit buffers, native providers and resource controls
    jaxbind.py          Abstract evaluation, derivatives, batching, FFI lowering
    misc.py             Static keyword decoding
    custom_map.py       Sequential vmap
    _bind.py            Array-function factory and tracing metadata
    _arrays.py          CuPy/Torch invocation and output validation
    _runtime.py         Callback context and borrowed XLA buffer execution
    _resources.py       Completion, retirement, streams and bounded resource caches
    _workspace.py       XLA scratch and shared recording temporaries
    _native.py          Native FFI registration and C ABI bindings
    _kernel.py          CUDA kernel configuration at tracing time
    include/            Standalone C ABI header
cpp/
    bridge.cc           Typed FFI entry points and graph ownership
    kernel_bridge.*     Direct CUDA driver launches
    inline_storage.h    Small argument arrays with heap fallback
    graph_cache.h       Bounded callback recordings owned by executables
examples/               Runnable integration examples, including native adapters
benchmarks/             Workloads, correctness oracles, direct-library baselines
scripts/                GPU selection, validation, and measurements
    benchmark_common.py Shared timing and native resource contracts
```

## Array functions

`bind` translates an array-returning or `out=` function into a buffer callback.
`_arrays.py` handles CuPy and Torch with shared structure, device, dtype, shape,
and alias checks. It owns no cache or CUDA synchronization.

Allocation forwarding also lives in `_arrays.py`. It lends each output to one
CuPy allocation with a matching byte count, then validates the returned arrays.
Before a copy fallback it saves sources that overlap another destination.
The allocator scope ends before those backups are allocated. This handles
views and output permutations without changing the runtime's ownership rules.

A trace record holds input/output shapes and tree structures. The factory
passes only its integer index to the operation. Abstract evaluation,
derivative rules, forward execution, and transposes reuse that record.
Output specifications run while tracing; GPU callbacks perform no shape
inference. Custom output tree classes do not need to be pickleable.

## Transformations

`jaxbind.py` registers one operation and its JAX transformations. Linear and
multilinear bindings apply their declared transposes and product rules.
General bindings accept JVP and VJP recipes. When both are supplied, the
linearized call selects the forward recipe for JVP and the reverse recipe for
VJP. Pure JAX recipes lower as JAX code and do not introduce Python callbacks.
Each binding reuses its linearized descriptor across eager differentiation calls.
Higher orders differentiate those recipes, including their primal inputs.

FFI lowering adds a hidden scratch output when a CuPy workspace is requested.
Only the declared mathematical outputs participate in differentiation.

## GPU ownership

`_runtime.py` resolves callback tokens, constructs borrowed array views and
provides the per-call `GpuContext`. `_resources.py` owns process-wide mutable
state in one `Resources` object. It manages completion queues, CUDA events,
library caches, stream wrappers, graph retirement and memory counters.
Ordinary calls retain resources
through a CUDA completion event. Captured calls retain resources through the
CUDA graph lifetime. The C++ graph destructor queues retired Python owners;
the Python reaper releases them on the correct device.
Stream wrappers are reused per callback, device, and stream and released when
the callback dies. Borrowed array views are created for each call and never
stored in that cache.

`graph_cache.h` manages eligible callback recordings and their native replay.
Each compiled call keeps at most 64 graphs. Warmup and recording share a stream
scoped to callback, device and XLA stream. Cached plans can retain that stream
across executable recompilations. The callback's finalizer releases its streams
after Python resource cleanup. CUDA user objects protect owners until all
queued launches and graph clones retire.
CuPy recordings on the same callback, device and XLA stream share a bounded
temporary allocator. Graph owners keep its allocations alive. Stream ordering
permits reuse across graphs, while allocation epochs protect new recordings
from releases belonging to old ones. Unknown allocation owners disable replay.

Native functions bypass Python array execution. `_native.py` and `_kernel.py`
encode configuration during tracing. C++ entry points use XLA's pointers and
stream directly. Native registrations retain their code owners for the process
lifetime. Library-specific native resources and cleanup remain the native
operation's responsibility. The public C ABI header supports adapters built
separately from this package.
The native C ABI resolves its registered operation and configuration during
executable instantiation. Recurring execution uses that state directly.

## Changing the implementation

Change array result handling in `_arrays.py`, mathematical JAX rules in
`jaxbind.py`, and completion or cache behavior in `_resources.py`. Callback
execution and per-call ownership declarations live in `_runtime.py`. Keep library-specific
plans and algorithms in their integrations. Framework benchmarks must hold
external algorithms fixed when measuring dispatch, copying, and memory costs.

Tests cover mathematical results, GPU pointer identity, asynchronous deletion,
graph replay, memory bounds, derivative modes, and installed-wheel execution.
See [validation](validation.md) and [benchmarks](../benchmarks/README.md).

Benchmark executables import timing, exclusivity checks and the direct-library
context from `scripts/benchmark_common.py`. They do not import another benchmark
executable for those utilities. The helper preserves changing inputs, fresh
retained outputs, immediate warm batches and blocked completion.
