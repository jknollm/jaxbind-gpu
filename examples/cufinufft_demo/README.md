# Advanced, optimized cuFINUFFT example

For a small binding of cuFINUFFT's existing Python functions, start with
[nufft_python.py](../nufft_python.py). This advanced example adds native execution,
plan reuse, multidimensional transforms and coordinate derivatives.

This adapter is separate from the binding package. Build its native operation
against the installed public C ABI header, then run the self-checking example:

```sh
.venv/bin/python -m pip install '.[cuda12,demos]'
.venv/bin/python examples/cufinufft_demo/build.py
.venv/bin/python scripts/run_gpu.py --wait 300 examples/nufft_native.py
```

The native adapter needs a C++17 compiler and pybind11. It does not need CUDA or
cuFINUFFT headers. Plans and their cleanup belong to this example. After blocking
JAX results, use `synchronize(clear_cache=True)` to release plans.
`configure_runtime(max_cache_entries=...)` sets the demo's plan-cache limit.
`runtime_stats()` includes its native plan count.

```python
import jax
import jax.numpy as jnp
from examples.cufinufft_demo import (
    nufft1, nufft2, nufft3, prepare_nufft1,
)

jax.config.update("jax_enable_x64", True)
x = jnp.linspace(-2.0, 2.0, 11)
c = jnp.ones(11, dtype=jnp.complex128)
f = jax.jit(lambda c: nufft1(x, c, 16, eps=1e-11))(c)
values = nufft2(x, f, eps=1e-11)
s = jnp.linspace(-3.0, 3.0, 9)
nonuniform_frequencies = nufft3(x, s, c, eps=1e-11)

# Repeated transforms with immutable geometry avoid repeated setpts.
prepared = prepare_nufft1(x, 16, eps=1e-11)
compiled = jax.jit(prepared)
f = compiled(c)
```

Type 1 maps strengths to a uniform Fourier grid. Type 2 evaluates grid
coefficients at nonuniform points. Type 3 maps nonuniform strengths to
nonuniform frequencies. Use tuples of one-dimensional coordinate arrays for
2D and 3D. complex64 data needs float32 coordinates; complex128 needs float64
coordinates and JAX x64 mode.

There is no normalization. Types 1 and 3 default to `isign=1`; type 2 defaults to
`isign=-1`. Types 1 and 2 support centered ordering and `modeord=1` FFT ordering.
Additional static keywords select cuFINUFFT options, except stream and device,
which the adapter controls. Type 3 defaults to `upsampfac=2.0`, because the
installed cuFINUFFT 2.5.1 automatic setting missed the requested accuracy for a
single-precision 3D case. An explicit `upsampfac` overrides this default.
Double-precision 3D transforms default to the point-driven `gpu_method=1` to
avoid shared-memory pressure and a measured performance regression. An explicit
`gpu_method` overrides this choice.

Leading data dimensions share coordinates and use one cuFINUFFT `n_trans` plan.
`vmap` over coefficients uses native batching. Mapping distinct coordinate sets
uses sequential device calls. JVPs, reverse gradients, and higher derivatives
cover both data and coordinates, including type 3 source and target coordinates.

`backend="native"` is the default and executes cuFINUFFT entirely in C++.
`backend="python"` provides a CuPy/Python reference implementation. Both reuse
bounded, stream-specific plans safely. Dynamic calls pass the existing
coordinate and data pointers to cuFINUFFT. `prepare_nufft1`, `prepare_nufft2`, and
`prepare_nufft3` freeze the geometry outside tracing. The native prepared plan
makes one device-to-device coordinate copy into owned memory and calls `setpts`
once. Its repeated executions use only `execute`.

When you capture GPU arrays in a jitted closure, enable JAX's device-preserving
constant handling before importing JAX:

```sh
JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS=true .venv/bin/python your_program.py
```

This matters for prepared geometry and for any user closure that captures an
array. JAX 0.11.2 otherwise may transfer captured constants to the host during
compilation. Dynamic array arguments do not need this setting. Tests prohibit
device-to-host transfers during both native dynamic calls and prepared
compilation with this setting enabled.
