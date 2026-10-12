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
| [nufft_demo.py](nufft_demo.py) | cuFINUFFT in a differentiable JAX calculation |

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

The cuFINUFFT adapter and its build instructions are in
[cufinufft_demo/](cufinufft_demo/README.md). It is built separately from the
framework. A dense Fourier calculation checks the transform and gradient.

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
