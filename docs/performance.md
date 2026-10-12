# Performance measurements

The benchmark suite covers thirteen applications, the public array interface,
native FFI, derivative calls, allocation policies and cuFINUFFT wrapper comparisons.
Run the suite and summarize its outputs to create
`benchmarks/results/current-comparison.md`. The generated overview and raw files
record the measured version, software, GPU identity, source hashes and comparison
contracts. Generated measurements are not bundled with the source distribution.

## Read the comparisons

Latency waits for each call. Throughput queues a batch of calls and waits for
all results. Inputs change while remaining GPU resident, and every call returns
a fresh output held until completion. Variants alternate order. Each timed batch
follows an identical untimed warm batch. Compilation, first execution and
correctness checks are separate. Timed JAX execution forbids device-to-host
transfers.

`direct_library` executes the same adapter outside JAX, including conversions
and result copies. `native_return` uses the library's native result without
adapter copies. Native Torch and Warp views and stream wrappers are prepared
once. Equivalent JAX built-ins may use different algorithms.

Native graph controls copy their stable private output into a fresh result on
each call. The binding can write directly into its fresh output. That extra
native copy can affect the comparison, especially for large outputs.

Percentages and additive times describe the measured configuration. Copies,
Python dispatch, buffer conversion, graph updates, allocation and library host
work all affect the result. Array size alone does not determine overhead.

## Execution modes

Python callbacks construct borrowed array views, invoke library code and retain
resources until CUDA work completes. Native C ABI and typed FFI calls use XLA's
pointers and stream directly. The scale benchmark compares identical external
kernels with bare typed FFI to isolate bridge cost. The nonlinear benchmark uses
identical sin/cos kernels for forward, JVP and VJP calls. Analytic calculations
check higher derivatives outside timing.

`graph_compatible=True` permits capture-safe callback recording. Ordinary
`jax.jit` can replay recordings without entering Python. XLA command buffers
are measured separately. Changing addresses can require graph updates or new
recordings. A compiled call retains at most 64 recordings. CuPy temporaries can
share bounded scratch; library-private allocation follows the library's rules.
The report lists exclusions when a library operation rejects capture or needs
a host read.

Prefer `out=True` when the library supports output buffers. Return-only code
normally copies its result into JAX storage. `out="allocate"` forwards matching
CuPy allocations, with a copy fallback for incompatible allocations or views.
See the [allocation contract](interface.md#return-only-functions).

## Memory

Memory runs use fresh processes. Their rows state the number of held outputs.
They report live and reserved allocator storage, compiler scratch, recorded
array scratch and sampled driver memory separately. Driver readings include
CUDA contexts, code, pools and library-private storage, and can miss short
peaks. Allocator peaks need not coincide and must not be added into a total.

Warp's allocation control performs a plain JAX addition with the same output
shape and held-result count, while keeping the same external matrix resident.
It isolates JAX allocation behavior and is not an algorithm or runtime
comparison. Warp's all-time allocator high-water mark includes construction and
is separate from its measured execution interval.

## cuFINUFFT wrappers

The released jax-finufft wrapper constructs plans per call. A benchmark-only
reference build adds plan and prepared-geometry reuse through the same JAX API.
Both comparisons match precision, signs, upsampling and GPU methods. Dynamic
cases change coordinates and data. Prepared cases keep coordinates fixed.
Gains from demonstration-specific plan reuse are not framework gains.

The identical-provider control calls the same cuFINUFFT provider through bare
FFI and the framework C ABI, with matching plan caches and compiler storage.
It isolates bridge cost; the wrapper comparison also includes adapter costs.

## Reproduce

See the [benchmark guide](../benchmarks/README.md). The suite definition is in
[benchmarks/suite.json](../benchmarks/suite.json). The
generated command record at `benchmarks/results/current-commands.json` gives the
exact commands and completion status. `python scripts/summarize_benchmarks.py`
regenerates the report and CSV files after the suite completes. It rejects
measurements with mismatched package versions or code hashes.
