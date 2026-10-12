# Binding reference

## One binding function

The normal entry point is `from jaxbind_gpu import bind`. Give it an ordinary
GPU array function.
By default it returns one array with the first input's shape and dtype.
For a different shape, dtype, or output tree, supply `output=` as a
`jax.ShapeDtypeStruct` tree or a function of input shapes and static keywords.
A function without inputs always needs an explicit output specification.
The binding returns an ordinary JAX function with that output structure.
Direct calls reuse compilation for each shape and static configuration. During
JAX transformations the binding traces directly into the surrounding program.

```python
import cupy as cp
import jax
import jax.numpy as jnp
from jaxbind_gpu import bind

square = bind(
    cp.square,
    jvp=lambda primals, tangents: 2 * primals[0] * tangents[0],
)
x = jnp.arange(8, dtype=jnp.float32)
y = jax.jit(square)(x)
gradient = jax.jit(jax.grad(lambda x: square(x).sum()))(x)
```

If the library accepts an output parameter, pass `out=True` to write directly
into XLA storage. This avoids allocating a returned result and copying it:

```python
square = bind(cp.square, out=True, jvp=lambda p, t: 2 * p[0] * t[0])
```

The function receives `out=` as the declared output tree. It can return that
tree or `None`. The default output is a contract. The framework checks returned
shapes and dtypes at execution, without running the library during tracing.
Explicit shape inference runs once per binding invocation while tracing;
abstract evaluation, derivative rules,
and GPU callbacks reuse the same metadata. Local custom pytree classes do not
need to be pickleable.

For an operation that safely supports in-place execution, declare
`input_output_aliases={0: 0}` with `out=True` and compile with
`jax.jit(op, donate_argnums=(0,))`. XLA can then reuse the input allocation.
Only declare aliases the external operation supports. The public factory
accepts aliases with a JVP or no derivative rule; explicit linear transpose
bindings should use the advanced interface when they need aliases.

`workspace_bytes=` is an optional byte count or a function of abstract inputs
and static keywords. It routes temporary CuPy allocations into an XLA-owned
buffer, so the compiler can reuse that storage between external calls:

```python
op = bind(
    external_function,
    out=True,
    workspace_bytes=lambda x: 8 * x.size * np.dtype(x.dtype).itemsize,
)
```

Supply enough space for the function's maximum concurrent temporary allocations;
the arena aligns allocations to 512 bytes. Exhaustion raises a clear error.
`current_context().workspace` exposes the borrowed byte buffer for libraries
with an explicit workspace API. Persistent resources must use
`current_context().cached(...)`, whose factory allocates outside this temporary
arena. Do not persist arena-backed arrays beyond the callback. This option
intercepts the CuPy allocator; it does not replace internal library allocations
or Torch's allocator. Native operations manage their workspace explicitly.
The advanced Python callback factories accept the same option. Shape-dependent
workspace functions on linear bindings must support transpose signatures too.

Without derivative rules, forward execution and sequential `vmap` work, and
differentiation raises an error. For linear functions, use `transpose=`:

```python
negate = bind(cp.negative, transpose=cp.negative)
```

For PyTorch, pass `backend="torch"` and a Torch GPU function. The binding handles
DLPack views and selects XLA's stream. `NativeFunction` objects use the same
`bind` factory and retain native execution with no Python callback. Output
specifications can be a single `ShapeDtypeStruct`, a tuple, or a nested tree,
or a function that computes such a tree from input shapes/dtypes. Positional
inputs are arrays; keyword arguments are static configuration. Derivative rules
must describe the supplied operation's mathematics.

The default function must return GPU arrays on the selected device. Inputs are
read-only, and all work must use the current stream. The binding retains result
allocations until queued work completes and copies them into XLA outputs on the
GPU. An external function that launches additional streams or creates scratch
allocations with unusual lifetime requirements must manage those dependencies.

The interfaces below provide explicit output buffers, batching recipes,
effects, aliases, and plan/workspace control for advanced integrations. They
share the same execution and transformation machinery as `bind`.

## Return-only functions

`out="allocate"` redirects matching CuPy allocations to XLA output buffers.
The external function still returns arrays and receives no `out=` keyword:

```python
sort = bind(cp.sort, out="allocate")
```

The first nonzero allocation matching an unused output's byte count borrows
that buffer. Each output is lent once per call. Other allocations use the
current allocator, including an optional XLA workspace. Shape, dtype, device,
and output-tree checks still apply. If the returned array has the target's
pointer and strides, no copy is needed. Returned views, reordered outputs,
and different allocation patterns use a safe copy fallback. The framework
saves overlapping sources before writing destinations, so output permutations
cannot overwrite data needed by another result.

All allocations made through this policy are temporary borrowed storage.
The function must not put them in a global, closure, persistent library plan,
or any cache after returning. Use `current_context().cached(key, factory)` for
persistent resources; its factory uses the original allocator. This lifetime
contract is why allocation forwarding is opt-in. Functions that cannot meet it
must use the default return-and-copy policy or an explicit output interface.
Implicit library caches need the same care. For example, a plain `cp.fft.fft`
can allocate persistent plan workspace during its first call. Create the plan
through the context cache and pass it explicitly before forwarding allocations.
Warming another binding first can hide this problem; validate cold execution
in a fresh process.

The policy supports multiple outputs, linear transposes, derivative recipes,
and capture-compatible CuPy code. It does not intercept a library's private
CUDA allocator or Torch allocations, and it cannot guarantee a copy-free call
for every allocation pattern. Input/output aliases are incompatible with this
mode. Prefer `out=True` when the function accepts an output buffer; it avoids
the allocation-policy bookkeeping too.

## Derivative rules

You usually need one rule. A traceable `jvp=` supplies reverse mode by
transposition. A traceable `vjp=` supplies forward mode by transposing its action
on cotangents. Neither path constructs a dense Jacobian. JAX compiles the
derivative recipes with the rest of your calculation.

```python
square = bind(cp.square, out=True, vjp=lambda p, c: (2 * p[0] * c,))
value, tangent = jax.jvp(square, (x,), (jnp.ones_like(x),))
value, pullback = jax.vjp(square, x)
input_cotangent, = pullback(jnp.ones_like(value))
gradient = jax.grad(lambda x: square(x).sum())(x)
hessian = jax.hessian(lambda x: square(x).sum())(x)
```

Both rules receive the tuple of positional inputs as `primals` and the same
static keywords as the forward function. A JVP receives a tuple of input
tangents and returns the declared output tree. A VJP receives the output
cotangent tree and returns a tuple with one cotangent per positional input.
Each input cotangent must have that input's shape and tangent dtype. `None`
means a zero input cotangent. Integer inputs and outputs have JAX's `float0`
tangent dtype. Complex rules follow JAX's bilinear transpose convention, which
does not insert a Hermitian conjugate automatically.

The rule must be linear in its tangents or cotangents. If you provide both,
they must describe the same derivative. The framework checks structure, shapes,
and dtypes; it cannot prove the mathematics of a supplied rule.

For a library that already supplies forward and adjoint routines, bind those
routines and call them in the rules. For a single-input, shape-preserving example:

```python
push = bind(library_jvp)
pull = bind(library_vjp)
op = bind(
    library_function,
    jvp=lambda p, t: push(p[0], t[0]),
    vjp=lambda p, c: (pull(p[0], c),),
)
```

Reverse mode uses `pull` directly; it does not run `push` or differentiate
through its external implementation. Forward mode uses `push`. With just an
opaque VJP, reverse mode still works, but forward mode requires a transposable
VJP or an explicit JVP. Higher derivatives differentiate the recipes and need
rules for external operations used inside them. You can call the original
binding recursively in a rule, as in the derivative of an external exponential.
Use JAX operations or bound operations in recipes; raw CuPy/Torch functions
cannot operate on JAX tracers. A raw external linear `transpose=` remains
supported separately.

Keep array arguments dynamic when compiling a calculation. A pullback or
linearization created outside `jit` captures its primal arrays. Compiling that
closure can transfer captured GPU constants to the host with JAX 0.11.2's
default constant handling. To preserve those constants on the device, set
`JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS=true` before importing JAX. Creating and using
the pullback or linearization inside a jitted function avoids that capture.

## GPU-resident arrays

The operation receives the input GPU buffer pointers supplied by XLA and writes
into XLA's GPU output buffers. Array contents do not travel through the CPU.
Shapes, dtypes, and static configuration are host metadata.

The native interface supplies raw pointers. The Python interface supplies CuPy
views backed by those pointers. Both use XLA's CUDA stream and device. Tests
check pointer identity across CuPy, JAX, the native operation, and the output.

To use an existing CuPy array without copying:

```python
import cupy as cp
import jax

external = cp.arange(1024, dtype=cp.float32)
x = jax.dlpack.from_dlpack(external, copy=False)
# Pass x to the bound operation, including inside jax.jit.
# y = compiled_operation(x)
# external_output = cp.from_dlpack(y)
```

Use the producer's DLPack protocol to import arrays from other libraries.
`copy=False` rejects an import that needs a copy. Layout changes, device changes,
or operations that explicitly copy arrays can still introduce device copies.
Bindings require row-major buffers; XLA may arrange a GPU buffer to satisfy that
layout. A JAX array created from a host NumPy array necessarily needs an initial
host-to-device transfer.

## Advanced integrations

Use [jaxbind_gpu.advanced](advanced.md) for explicit buffer callbacks, custom
batching rules, effects, native providers and runtime resource controls.
These share the same execution and differentiation machinery as `bind`.
Existing imports from the package root remain supported.
