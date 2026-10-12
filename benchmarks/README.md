# External-library benchmarks

`scripts/benchmark_interfaces.py` compares the public CuPy and Torch bindings
with the same native square operation. Direct calls, outer `jax.jit`, output
copies, direct output writes and cached recording are separate measurements.
The script checks results and an analytic gradient before timing.

```sh
.venv/bin/python scripts/run_gpu.py scripts/benchmark_interfaces.py \
    --torch-path .torch-benchmark --count 32 --repeats 9
```

`scripts/benchmark_output_allocation.py` compares unchanged return-only CuPy
functions under native execution, default result copying, and `out="allocate"`.
It covers square, sort, prefix scan, FFT, Gaussian filtering, and Cholesky.
Pointer audits distinguish successful forwarding from copy fallbacks.
`--memory` runs each variant in a fresh process and reports CuPy and JAX memory
separately. Use `--count 4` for memory measurements with four held outputs.

`scripts/benchmark_provider.py` calls the identical cuFINUFFT native provider
through the framework C ABI and a minimal FFI control. It keeps configuration,
plan cache and compiler memory equal. This isolates bridge overhead when the
separate wrapper comparison differs because it uses another adapter.

```sh
.venv/bin/python scripts/run_gpu.py scripts/benchmark_provider.py \
    --count 32 --repeats 9
```

```sh
.venv/bin/python scripts/run_gpu.py scripts/benchmark_output_allocation.py
.venv/bin/python scripts/run_gpu.py scripts/benchmark_output_allocation.py \
    --memory --sizes 1048576 --count 4 \
    --output benchmarks/results/current-output-allocation-memory.json
```

Run the complete suite with one idle GPU UUID, or give two idle UUIDs to use
separate runtime and memory queues:

```sh
python scripts/run_benchmarks.py --devices GPU-UUID
python scripts/run_benchmarks.py --devices GPU-UUID-1 GPU-UUID-2
```

Each job checks its assigned device before launch. Logs and command records go
to `benchmarks/results/`. `--list` prints jobs and `--only NAME` resumes selected
jobs after an inspected failure. The full suite requires optional integrations
and the two cuFINUFFT reference builds. Every job is also runnable individually.
The suite definition is `benchmarks/suite.json`.

After refreshing all evidence files, `python scripts/summarize_benchmarks.py`
rebuilds `results/current-overview.json`. It rejects measurements from a
different package version or package hash and regenerates the comparison report
and CSV tables.

`applications/operators.py` contains the external algorithms.
`applications/workloads.py` defines inputs, size presets, independent correctness
oracles, and equivalent JAX computations where available. Benchmark adapters are
separate from the short integration lessons in `examples/`.

The thirteen workloads use Warp block-sparse products, cuFFT, cuSOLVER, cuSPARSE, image filters, Torch/cuDNN,
cuFINUFFT, reductions, scans, cuBLAS matrix products, special functions, Thrust
sorting, and an external CUDA stencil. The `direct_library` baseline executes
the identical adapter, including its result copy when the library has no output
argument. `--native-return` also measures the fresh native result for solve,
sparse diffusion, sorting, and Torch. Torch native inputs and its stream wrapper
are prepared once. This exposes adapter conversion and copy costs separately
from the binding callback. Warp's native baseline also prepares its input views
and stream once, then calls the unmodified `warp.sparse.bsr_mv` with a fresh
native result. The bound version writes directly into XLA outputs. Both keep
the same fixed float32 block matrix resident; SciPy's float64 BSR product checks
every output. Matrix construction is excluded from warmed measurements.

```sh
python -m pip install warp-lang
python scripts/run_gpu.py scripts/benchmark_applications.py \
    --apps warp_sparse --native-return --cached --native-graphs \
    --count 32 --repeats 15 --output benchmarks/results/current-warp-cached.json
python scripts/run_gpu.py scripts/benchmark_memory.py \
    --apps warp_sparse --variants native_return jax_allocation_control jaxbind_gpu jaxbind_gpu_cached \
    --size medium --count 32 --output benchmarks/results/current-warp-memory-medium.json
```

Use `--cached` for recorded callback replay, `--graphs` for XLA command buffers,
or neither for ordinary Python callbacks. The native graph control copies its
stable private output into a fresh result per call. This extra copy is explicit
in the results. Native ordinary calls need no such copy. Memory measurements
report Warp's CUDA pool live storage as well as CuPy, JAX and sampled driver
storage. Warp's all-time high-water mark includes matrix construction; it is
separate from the measurement interval. CUDA pool reserved storage is included
in driver samples, not inferred from live allocation totals.
`jax_allocation_control` runs a plain JAX addition with the same input/output
shape and held-result count. It keeps the external matrix resident for equal
persistent storage. This isolates JAX's allocator reserve; its algorithm is
different and it is used only for memory comparison.

```sh
.venv/bin/python scripts/run_gpu.py --wait 300 scripts/benchmark_applications.py \
    --help
.venv/bin/python scripts/run_gpu.py --wait 300 scripts/benchmark_memory.py --help
.venv/bin/python scripts/run_gpu.py --wait 300 scripts/benchmark_dispatch.py --graphs
.venv/bin/python scripts/run_gpu.py --wait 300 benchmarks/optimize.py \
    --app image_blur
```

`optimize.py` compiles gradient steps for the workloads with supplied derivative
rules. Its choices exclude forward-only benchmark bindings. This program is an
application acceptance check, not a timing comparison.

Timing uses changing GPU-resident inputs, holds distinct outputs, waits for JAX
results, alternates variant order, and checks GPU exclusivity. Native CUDA graph
baselines copy their stable output to a fresh result for each invocation; that
extra copy can favor the binding. Timing artifacts report this explicitly.
Memory runs use fresh processes and distinguish compiler scratch, allocator
peaks, graph-owned storage, and driver samples. Driver sampling can miss peaks.

Result JSON files record the measured code, software, and execution settings. See
[performance evidence](../docs/performance.md) for measurements, comparison
contracts, and cases where callbacks remain slower than native execution.

`benchmark_dispatch.py` varies array length while holding buffer count, rank,
dtype, and the external CUDA kernel fixed. It reports paired additive costs in
microseconds and size-scaling fits. The native binding, Python callback, returned
array copy, and optional graph execution all use that same kernel. End-to-end
differences include JAX dispatch and allocation. A constant fit is descriptive;
it does not establish a worst-case latency guarantee.
`--eager` also measures direct calls with automatic compilation reuse. Their
first-call measurements include compilation; warmed timing excludes it.

`--graphs` includes both cached callback replay under ordinary `jax.jit` and
XLA command-buffer capture, with native controls for each. In application
benchmarks, `--cached` enables callback recordings without changing compiler
options. Memory benchmarks select `jaxbind_gpu_cached` explicitly. Use
`--count 32` to match the dispatch sweep's held-output protocol.
The cached application sweep uses unchanged FFT, filtering, reduction, scan,
special-function, sorting and stencil implementations. Their captured CuPy
temporaries share bounded storage automatically. Separate FFT and filtering
measurements compare explicit XLA workspaces with this shared allocation.

`--ffi-reference` builds a minimal typed FFI handler that launches the identical
CUDA kernel. The comparison includes a bare `jax.ffi.ffi_call`, a bound typed
handler, and the generic native C ABI. `--derivatives` compares linear JVPs and
VJPs against the same bare handler. Compiled HLO and executable memory accounting
check that each linear derivative uses one native call without extra scratch.

`benchmark_nonlinear.py` compares sine, its JVP and its VJP with bare typed FFI
using identical external sin/cos kernels. Changing primal values and directions
remain GPU resident. NumPy checks each result, and an analytic Hessian product
checks higher-order differentiation outside timing. Compiler memory is reported
separately from the fresh-process memory sweeps.

```sh
.venv/bin/python scripts/run_gpu.py scripts/benchmark_nonlinear.py \
    --sizes 4096 1048576 16777216 --count 32 --repeats 9
```
