# jaxbind-gpu

Turn external CUDA operations into JAX functions that work directly on GPU-resident
arrays. One `bind` call supplies the implementation, output contract and optional
derivative rules. The resulting function supports `jit`, `vmap`, JVPs and VJPs.

## Install

Requires Linux, Python 3.12+, JAX 0.11.x and CUDA. Building the native bridge needs a
C++17 compiler; nvcc is unnecessary.

```sh
git clone https://github.com/jknollm/jaxbind-gpu.git
cd jaxbind-gpu
python -m pip install '.[cuda12]'
```

The source distribution also includes the documentation, runnable examples
and benchmark tools linked below.

The core requires JAX, jaxlib and NumPy. CuPy, Torch and demonstration libraries
are optional dependencies of the execution paths that use them.

## Bind a function

For a function with one output matching its first input, use `bind(gpu_function)`.
If the function accepts `out=`, add `out=True` to write directly into JAX storage.

```python
import cupy as cp
import jax
import jax.numpy as jnp
from jaxbind_gpu import bind

square = bind(cp.square, out=True, jvp=lambda p, t: 2 * p[0] * t[0])
x = jnp.arange(8, dtype=jnp.float32)
y = square(x)
gradient = jax.jit(jax.grad(lambda x: square(x).sum()))(x)
```

`p` contains the positional inputs; `t` contains their tangents. A traceable JVP
also supplies reverse mode and higher derivatives. You can supply a VJP instead,
or both rules when the external library provides separate derivative operations:

```python
square = bind(cp.square, out=True, vjp=lambda p, c: (2 * p[0] * c,))
```

A VJP returns one cotangent per input. Linear functions can use `transpose=`.
Rules must describe the operation's mathematics. External calls inside a rule
need their own rules for higher derivatives. Forward execution and sequential
`vmap` work without derivative rules.

Direct calls reuse compilation. Put `jax.jit` around a larger calculation to
compile its surrounding JAX work together with the binding.

## Specify different outputs

Use `output=` when shape, dtype or output structure differs from the first input.
It accepts a `jax.ShapeDtypeStruct` tree or a function of abstract inputs and
static keywords. The library is never called to discover shapes during tracing.

```python
row_sum = bind(
    lambda x: cp.sum(x, axis=-1),
    output=lambda x: jax.ShapeDtypeStruct(x.shape[:-1], x.dtype),
)
y = row_sum(x.reshape(2, 4))
```

Positional arguments are arrays; keyword arguments are static configuration.
Inputs are borrowed GPU buffers and must be treated as read-only. The library
must queue its work on the supplied CUDA stream. Returned results are checked
and copied into JAX output storage on the GPU unless they already use it.

For Torch functions, pass `backend="torch"`. For a CuPy RawKernel, pass
`launch=(grid, block)` and explicitly typed `scalars=`.

## Control execution when needed

Most functions need only the options above. Additional controls have explicit
contracts because external allocation and capture behavior varies:

| Need | Option or interface |
| --- | --- |
| Reuse capture-safe CUDA work | `graph_compatible=True` |
| Borrow output storage in return-only CuPy code | `out="allocate"` |
| Provide temporary CuPy workspace | `workspace_bytes=` |
| Reuse an input allocation | `input_output_aliases=` and JAX donation |
| Supply native code or explicit buffer callbacks | `jaxbind_gpu.advanced` |

See the [binding reference](docs/interface.md) and
[execution contracts](docs/execution.md) before enabling these controls.
The [advanced reference](docs/advanced.md) covers native providers, library
resource caches and runtime diagnostics.

## Examples

Start with [array_function.py](examples/array_function.py), which checks values,
derivatives and batching. The [example guide](examples/README.md) covers library
functions, return-only code, Torch, Warp, explicit buffers, cached plans, CUDA kernels
and native C/C++ providers. Each example checks an independent numerical result.

```sh
python scripts/run_gpu.py examples/array_function.py
python scripts/run_gpu.py scripts/check_examples.py
```

## Development

```sh
python -m pip install '.[cuda12,test]'
python scripts/test_gpu.py -q
JAX_PLATFORMS=cpu python -m pytest -q
python -m pip install '.[release]'
python scripts/check_package.py
```

[Architecture](docs/architecture.md) identifies each module's responsibility.
[Benchmarks](benchmarks/README.md) describe fair runtime and memory comparisons.
[Validation](docs/validation.md) records checks and supported limits.
[Third-party notices](THIRD_PARTY.md) contain attribution and licensing.

## License

BSD-2-Clause. See [LICENSE](LICENSE) for the terms and retained JAXbind notices.
The [release guide](docs/releasing.md) covers distribution builds and checks.
