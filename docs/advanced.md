# Advanced integrations

Start with `from jaxbind_gpu import bind` for ordinary array functions.
`jaxbind_gpu.advanced` groups the interfaces for providers that need explicit
buffers, native registration or resource management.

```python
from jaxbind_gpu import bind
from jaxbind_gpu import advanced as jb
```

## Choose an interface

| Interface | Purpose |
| --- | --- |
| `get_call` | Buffer callback with traceable JVP/VJP rules |
| `get_linear_call` | Linear or multilinear callback with external transposes |
| `get_nonlinear_call` | Opaque first-order GPU derivative callbacks |
| `load_kwargs` | Decode a buffer callback's static configuration |
| `NativeFunction` | Register typed FFI, C ABI or CUDA kernel execution |
| `include_dir` | Locate the standalone C ABI header |
| `current_context` | Access the callback's device, stream, cache and retention |
| `runtime_stats` | Inspect pending resources, caches, recordings and scratch |
| `synchronize` | Drain external work after blocking JAX outputs |
| `configure_runtime` | Set bounded cache and pending-resource limits |

## Differentiation and batching

`get_linear_call(f, f_T, abstract, abstract_T)` binds a linear operation and its
transpose. A tuple of transposes and abstract functions handles multilinear
operations. `first_n_args_fixed=N` passes the first N inputs as GPU arrays but
excludes them from differentiation. For complex arrays, supply the transpose
under JAX's complex differentiation convention.

For general nonlinear operations, use a traceable derivative recipe:

```python
def external_exp(out, args, kwargs_dump):
    cp.exp(args[0], out=out[0])


def exp_jvp(primals, tangents):
    return (jnp.exp(primals[0]) * tangents[0],)


exp_jax = jb.get_call(external_exp, abstract, jvp=exp_jvp)
```

`jvp(primals, tangents, **kwargs)` returns output tangents. JAX obtains reverse
mode by transposing this recipe and differentiates it for higher orders. The
recipe can call other bindings as well as JAX operations. Tangent-dependent
external calls need their own linear or traceable derivative rules. With
`symbolic_zeros=True`, inactive tangents arrive as `None`.

`get_nonlinear_call(f, (jvp, vjp), abstract, abstract_reverse)` accepts
first-order GPU derivative callbacks. It does not infer higher derivatives of opaque
callbacks. No interface can infer the mathematics of an arbitrary external
operation; provide its derivative rules when differentiation is required.

`vmap` defaults to a sequential device map. A custom
`batching_rule(call, args, in_axes, **kwargs)` returns outputs and their axes.
It can broadcast or move inputs and call a library's native batched operation
once. The `func_can_batch=True` form supplies `batch_axes` to
callbacks and abstract functions; each abstract output then includes its batch
axis as `(shape, dtype, batch_axis)`.

`get_call(..., has_side_effect=True)` preserves effects even when outputs are
unused. Such effects are unordered; do not use them as a substitute for explicit
dependencies. `input_output_aliases={input_index: output_index}` declares
in-place storage, usually together with JAX argument donation. NativeFunction
also accepts these options. Pure calls may be eliminated or duplicated by JAX.


## Native providers

Pass a `NativeFunction` to `bind` to execute directly on XLA's stream without
Python callbacks. `NativeFunction.from_cabi` accepts an operation described by
`include_dir()/jaxbind_gpu.h`. `from_cuda_kernel` accepts a RawKernel or an owned
CUDA function handle. Configuration is encoded during tracing; native execution
receives GPU pointers, shapes, dtypes, device and stream.

See [native_scale.py](../examples/native_scale.py) for a standalone C ABI provider
and [cuda_kernel.py](../examples/cuda_kernel.py) for a CUDA kernel. Provider code
owns its library resources and must obey the stream and lifetime contract.

## Resource ownership

Inside a callback, `current_context().cached(key, factory)` reuses a library
resource on the same callback, device and stream. Its factory allocates outside
temporary workspace. Include every relevant library setting in the key and
never store borrowed input or output arrays in a resource cache.

`current_context().retain(resource)` keeps temporary allocations or opaque
owners alive until queued work completes. The framework protects ordinary
calls with completion events and recorded calls with CUDA graph ownership.
See [cached_fft.py](../examples/cached_fft.py) and the
[execution reference](execution.md) for allocation and capture contracts.

Block JAX results before `synchronize(clear_cache=True)`. Reconfigure runtime
limits only when no calls are being submitted concurrently. `runtime_stats()`
reports framework counters; it does not account for all library-private,
allocator-reserved or driver memory.
