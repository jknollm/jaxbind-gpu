# Validation

Tests compare results with independent analytic, NumPy, and SciPy calculations.
They check input/output GPU pointer identity, multiple outputs, layouts and
dtypes, exceptions, `jit`, nested `vmap`, JVPs, VJPs, higher derivatives, effects,
donation, cache eviction, asynchronous resource lifetime, graph replay, and
named FFI export. Derivative compilation checks also reject accidental dense
Jacobian storage for elementwise rules.

```sh
.venv/bin/python scripts/test_gpu.py -q
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q tests/test_tracing.py tests/test_derivatives.py tests/test_bind.py
.venv/bin/python scripts/run_gpu.py --wait 300 scripts/check_examples.py
.venv/bin/python -m build
```

Optional library integrations have separate tests in `tests/demos/` and build
instructions in their example guides. The cuFINUFFT demo compares transforms
with dense Fourier sums and an independent wrapper. Core tests do not require
that adapter to be built. `pytest tests --ignore=tests/demos` selects the core.
Warp's demo tests a nonsymmetric rectangular matrix against a dense NumPy
product, its transpose, JVP, VJP and analytic Hessian. Both ordinary and captured
execution must preserve borrowed input/output pointers, batching and held
results for changing inputs. The benchmark checks all block-sparse outputs
against SciPy in float64.
Package checks write the installed version, file hashes and distribution checks
to `validation-build.json`. Benchmark output records its measured version and
source hashes separately. See [performance measurements](performance.md).

## Installation and automated checks

`python scripts/check_package.py` builds the wheel and source archive, installs
the wheel in a temporary environment containing only core and test dependencies,
checks CPU import without optional GPU libraries, runs the core CPU suite and
checks dependency consistency. The wheel is built from the source archive with
isolated build dependencies. It needs no GPU and can run in CI. Install the
`release` extra first. `--release` additionally audits and repairs the Linux wheel
and checks package metadata. The result is written to `validation-build.json`.
See the [release guide](releasing.md) for the complete procedure.

The tested environment is Linux x86_64 with glibc 2.34, Python 3.12.13,
JAX/jaxlib 0.11.2 and NVIDIA A100 GPUs. GPU tests use CUDA 12 libraries with
driver 610.43.02. CuPy 14.2.0, cuFINUFFT 2.5.1 and Warp 1.18.0 are tested
optional integrations. Warp uses its own CUDA toolkit build. Its wheel must
support the installed driver. CPU installation was also checked with NumPy
2.5.4; GPU measurements use NumPy 2.5.3. The declared Python and dependency
ranges permit additional versions, but these measurements cover the listed
combinations. macOS, Windows and ROCm execution are not supported.

GPU validation is explicit:

```sh
python scripts/run_gpu.py -- -m pytest -q
python scripts/run_gpu.py scripts/check_examples.py
```

Optional tests and examples report missing integrations as skips. Two-device
tests need two idle GPUs. GPU jobs disable JAX preallocation and cap common CPU
thread pools. The full benchmark suite records device identity and rejects
another process on its assigned GPU.

## Execution requirements

External operations execute on CUDA. Shapes must be known while tracing.
Pure JAX derivative recipes can execute independently when the compiled
calculation eliminates the external forward call. CPU tracing does not
execute GPU callbacks.

Efficient native batching and distributed partitioning depend on the external
operation. A binding cannot infer how a library should split a global
transform. Multi-device execution needs device-aware resources. Demonstration
kernels compiled for one device must be bound separately for another device.
The supported version range is JAX 0.11.x because transformation internals
are version-sensitive.

Mathematical rules must be correct, linear in tangents or cotangents, and
consistent when both are supplied. Higher orders require differentiable rules
for external calls inside a recipe. Supplying first-order opaque GPU routines
does not infer higher derivatives of those routines.
