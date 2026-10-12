# jaxbind-gpu

Bind external CUDA operations to JAX and pass GPU-resident arrays directly to
the implementation. One `bind` call describes the output and optional derivative
rules. The resulting function supports `jit`, sequential `vmap`, JVPs and VJPs.

```sh
python -m pip install 'jaxbind-gpu[cuda12]'
```

```python
import cupy as cp
import jax
import jax.numpy as jnp
from jaxbind_gpu import bind

square = bind(cp.square, out=True, jvp=lambda p, t: 2 * p[0] * t[0])
x = jnp.arange(8, dtype=jnp.float32)
result = jax.jit(square)(x)
gradient = jax.grad(lambda x: square(x).sum())(x)
```

`out=True` writes directly into JAX output storage. Functions that return a new
array work with `bind(function)`; the default output matches the first input.
Use `output=` for a different shape, dtype or output tree, and `backend="torch"`
for Torch functions. Native C/C++ providers use `jaxbind_gpu.advanced`.

A JAX derivative recipe supplies reverse mode and higher derivatives when its
own operations are differentiable. Bindings without derivative rules support
forward execution. The package cannot infer a library's mathematical derivatives.

External operations must queue work on the supplied CUDA stream and treat inputs
as read-only. Shapes must be known during tracing. Capture and output-allocation
controls have additional contracts described in the documentation.

Requires Linux, Python 3.12+, JAX 0.11.x and a CUDA-capable NVIDIA GPU for external
execution. The core depends on JAX, jaxlib and NumPy. CuPy, Torch, cuFINUFFT and
Warp are optional. Source builds need a C++17 compiler; the core needs no nvcc.
CPU installation and tracing work without optional GPU libraries.

The source distribution contains `README.md`, the binding and execution
references in `docs/`, and independently checked examples in `examples/`.
`benchmarks/` includes runtime and memory comparison tools for thirteen applications.
Measurements identify software, hardware, execution settings and limitations;
they do not promise a universal overhead percentage.

BSD-2-Clause. The distribution includes `LICENSE` and `THIRD_PARTY.md` with
JAXbind attribution.
