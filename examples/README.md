# Example guide

Start with an ordinary array function, then choose the integration pattern you
need. Every program checks values against analytic results or independent
NumPy/SciPy calculations. Its main JAX computation disallows device-to-host
transfers; host transfers occur when checking the completed result.

```sh
python scripts/run_gpu.py examples/array_function.py
python scripts/run_gpu.py scripts/check_examples.py
```

## Ordinary functions

| Example | What it teaches |
| --- | --- |
| [array_function.py](array_function.py) | One `bind` call, direct output, JVP/VJP, gradient, Hessian and batching |
| [nufft_python.py](nufft_python.py) | cuFINUFFT's Python functions, fixed GPU coordinates and a linear transpose |
| [image_filter.py](image_filter.py) | Library function, static keywords and a linear transpose |
| [return_function.py](return_function.py) | Return-only CuPy function with output allocation forwarding |
| [torch_function.py](torch_function.py) | Torch views on XLA's stream, direct output and JAX derivatives |
| [warp_sparse.py](warp_sparse.py) | Warp block-sparse product, shared GPU buffers and a linear transpose |

`bind` defaults to the first input's shape and dtype. Declare `output=` for other
outputs. Positional inputs are arrays; keyword arguments are static. Prefer
`out=True` when the library accepts output buffers. A traceable derivative rule
can supply both forward and reverse modes. See the
[binding reference](../docs/interface.md) for multiple inputs and output trees.

The image filter uses periodic boundaries, for which the symmetric Gaussian
filter is self-transpose. Changing the boundary conditions requires checking
that mathematical rule again.

The return-only example uses `out="allocate"`. Matching CuPy allocations borrow
JAX output storage; incompatible allocation patterns use a copy fallback.
Only enable it when allocations cannot escape the call. Persistent resources
must use the context cache described in the [allocation contract](../docs/interface.md#return-only-functions).

### Minimal cuFINUFFT Python binding

Start with [nufft_python.py](nufft_python.py):

```sh
python -m pip install cufinufft
python scripts/run_gpu.py examples/nufft_python.py
```

It calls the installed `cufinufft.nufft1d1` and `nufft1d2` functions on CuPy arrays.
`bind` supplies JAX execution and differentiation; `out=True` writes directly
into JAX's output buffers. Pass the current CuPy stream because cuFINUFFT takes
an explicit stream option. Coordinates remain fixed, and strengths have JVPs and
VJPs. Both transforms use the same Fourier sign because JAX's complex VJP takes
the plain transpose, not the Hermitian adjoint.

This example needs no separately compiled adapter. It uses the convenience
API's per-call plans and supplies no plan cache or graph capture. The advanced
cuFINUFFT example below adds plan reuse, coordinate derivatives and native execution.

## Explicit buffers and native providers

These examples use `jaxbind_gpu.advanced` for provider-specific buffer and
resource control, together with the same JAX execution machinery.

| Example | What it teaches |
| --- | --- |
| [scale.py](scale.py) | Buffer callback, static configuration and borrowed GPU pointers |
| [linear_solve.py](linear_solve.py) | Dynamic GPU matrix excluded from differentiation; external transpose solve |
| [cached_fft.py](cached_fft.py) | Cached plans, temporary workspace and CUDA graph replay |
| [cuda_kernel.py](cuda_kernel.py) | Native CUDA launch from a RawKernel with typed scalar arguments |
| [native_scale.py](native_scale.py) | Separately compiled C/C++ provider through the standalone C ABI |
| [nufft_native.py](nufft_native.py) | Advanced, optimized cuFINUFFT adapter with reusable native plans |

The buffer callback receives borrowed CuPy inputs and outputs. `load_kwargs`
decodes static configuration. Shape functions run during tracing.
The linear solve differentiates its right-hand side while keeping the matrix
as a dynamic GPU argument; it does not supply matrix derivatives.

The FFT example creates plans through `current_context().cached`. Its factory
uses only shape/dtype metadata and owns independent plan storage. Temporary
workspace is borrowed, and graph replay requires capture-safe GPU work without
required per-call Python actions. The native examples receive raw GPU pointers
and must follow the same stream and lifetime rules. See the
[advanced reference](../docs/advanced.md) for these interfaces.

### Advanced, optimized cuFINUFFT native binding

[nufft_native.py](nufft_native.py) uses the compiled adapter in
[cufinufft_demo/](cufinufft_demo/README.md). Its native execution avoids per-call
Python dispatch into cuFINUFFT, reuses stream-specific plans and supports native
batching. The adapter supplies types 1, 2 and 3 in one to three dimensions,
including coordinate derivatives and prepared geometry. It needs a separate
C++ build. A dense Fourier calculation checks the transform and gradient.

The minimal Python example calls the convenience API with its per-call plans.
It teaches the binding interface; the native example teaches library-specific
performance and derivative implementation.

## Optional dependencies

Torch, Warp, SciPy and cuFINUFFT examples run when their dependencies are installed.
The runner reports skipped examples explicitly. To use an isolated Torch
installation:

```sh
python scripts/run_gpu.py scripts/check_examples.py --torch-path .torch-benchmark
```

Install Warp with `pip install 'jaxbind-gpu[warp]'` or `pip install warp-lang`.
Its CUDA wheel must support the installed driver. The example uses Warp's
standard sparse routine unchanged, with DLPack views of the binding's buffers
and Warp's external-stream interface. Matrix construction finishes before JAX
execution. The matrix remains fixed during differentiation; the vector has
JVPs, VJPs, higher derivatives and sequential batching. Warp's atomic transpose
can vary in its last bits. See [Warp's installation guide](https://nvidia.github.io/warp/stable/user_guide/installation.html).

The [benchmarks](../benchmarks/README.md) exercise thirteen applications at larger
sizes, with unchanged algorithms and separate native runtime and memory baselines.
