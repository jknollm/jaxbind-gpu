"""One factory for ordinary GPU array functions and native FFI operations."""

from functools import lru_cache, wraps
import threading
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ._native import NativeFunction
from .jaxbind import get_call, get_linear_call
from .misc import load_kwargs
from ._arrays import execute

_LAYOUT = "_jaxbind_gpu_layout"


def _immutable(value):
    if value is None or isinstance(
        value, (str, bytes, bool, int, float, complex, np.number)
    ):
        return True
    return isinstance(value, tuple) and all(_immutable(x) for x in value)


class _Layout(NamedTuple):
    """Shape-only trace metadata; the callback receives its integer index."""

    inputs: tuple
    outputs: tuple
    output_tree: object
    input_tree: object


def _specs(output, args, kwargs):
    if output is None:
        if not args:
            raise ValueError("Provide output= when binding a function without inputs")
        tree = jax.ShapeDtypeStruct(args[0].shape, args[0].dtype)
    else:
        tree = output(*args, **kwargs) if callable(output) else output
    leaves, structure = jax.tree_util.tree_flatten(tree)
    if not leaves or any(not isinstance(x, jax.ShapeDtypeStruct) for x in leaves):
        raise TypeError("output must provide a nonempty tree of jax.ShapeDtypeStruct")
    return leaves, structure


class _NativeBinding(NativeFunction):
    """Remove private shape metadata before invoking the native attribute encoder."""

    def __init__(self, original):
        self.original = original
        self.has_side_effect = original.has_side_effect

    def _lower(self, args, outputs, kwargs, **options):
        kwargs = dict(kwargs)
        kwargs.pop(_LAYOUT)
        return self.original._lower(args, outputs, kwargs, **options)


def bind(
    function,
    output=None,
    *,
    jvp=None,
    vjp=None,
    transpose=None,
    backend="cupy",
    out=False,
    graph_compatible=False,
    input_output_aliases=None,
    workspace_bytes=0,
    launch=None,
    scalars=(),
    argument_order=None,
    shared_mem=0,
):
    """Bind an ordinary GPU array function or native operation to JAX.

    By default the output has the first input's shape and dtype. ``output``
    can declare a different ShapeDtypeStruct tree or compute it from abstract
    inputs and static keywords. Positional arguments are arrays; keywords are static.
    ``out=True`` passes XLA-backed outputs as ``out=``; otherwise returned GPU
    arrays are copied into those outputs. ``out="allocate"`` lets return-only
    CuPy code borrow output storage for matching allocations, with a copy
    fallback for returned views. Such allocations must not escape the call;
    persistent resources must use ``current_context().cached``. CuPy is the
    default array backend.

    ``jvp(primals, tangents, **kwargs)`` returns the output tangent tree.
    ``vjp(primals, cotangents, **kwargs)`` returns one cotangent per positional
    input, as a tuple. Either traceable rule can supply both differentiation
    modes; give both to select explicit forward and reverse implementations.
    Higher orders differentiate the recipes and need rules for external calls
    used there. Linear code can supply ``transpose``. Sequential vmap is automatic.

    ``launch=(grid, block)`` executes a CUDA kernel natively; ``scalars`` must
    use explicitly sized NumPy types. NativeFunction also executes natively.
    ``graph_compatible`` opts Python callbacks into capture-safe replay.
    ``workspace_bytes`` routes temporary CuPy allocations into XLA scratch;
    persistent resources must use ``current_context().cached`` instead.
    See docs/interface.md and docs/execution.md for complete contracts.
    """
    if launch is not None:
        function = NativeFunction.from_cuda_kernel(
            function,
            grid=launch[0],
            block=launch[1],
            scalars=scalars,
            argument_order=argument_order,
            shared_mem=shared_mem,
        )
    if transpose is not None and (jvp is not None or vjp is not None):
        raise ValueError("Provide jvp/vjp or a linear transpose, not both")
    if backend not in ("cupy", "torch"):
        raise ValueError("backend must be 'cupy' or 'torch'")
    if workspace_bytes and backend != "cupy":
        raise ValueError(
            "workspace_bytes routes CuPy allocations; Torch requires its own workspace management"
        )
    if not callable(function):
        raise TypeError("function must be callable")
    if transpose is not None and not callable(transpose):
        raise TypeError("transpose must be callable")
    if jvp is not None and not callable(jvp):
        raise TypeError("jvp must be callable")
    if vjp is not None and not callable(vjp):
        raise TypeError("vjp must be callable")
    if out is not False and out is not True and out != "allocate":
        raise ValueError("out must be False, True, or 'allocate'")
    writes_output = out is True
    allocate_output = out == "allocate"
    if allocate_output and (backend != "cupy" or isinstance(function, NativeFunction)):
        raise ValueError("out='allocate' requires an ordinary CuPy function")
    if writes_output and isinstance(function, NativeFunction):
        raise ValueError("NativeFunction already writes directly to the outputs")
    if input_output_aliases is not None:
        if not writes_output and not isinstance(function, NativeFunction):
            raise ValueError("Input/output aliases require direct output writes")
        if transpose is not None:
            raise ValueError(
                "Use get_call with a JVP for explicit input/output aliases"
            )

    layouts = []
    layout_lock = threading.Lock()

    def workspace(*args, **kwargs):
        kwargs.pop(_LAYOUT)
        return workspace_bytes(*args, **kwargs)

    workspace_spec = workspace if callable(workspace_bytes) else workspace_bytes

    def abstract(*args, **kwargs):
        return layouts[kwargs[_LAYOUT]].outputs

    def abstract_transpose(*args, **kwargs):
        return layouts[kwargs[_LAYOUT]].inputs

    def adapt(external, *, transposed=False):
        if isinstance(external, NativeFunction):
            return _NativeBinding(external)

        @lru_cache(maxsize=128)
        def configuration(config):
            kwargs = load_kwargs(config)
            layout = layouts[kwargs.pop(_LAYOUT)]
            structure = layout.input_tree if transposed else layout.output_tree
            return structure, kwargs if all(
                _immutable(v) for v in kwargs.values()
            ) else None

        def callback(out, args, config):
            structure, kwargs = configuration(config)
            if kwargs is None:
                kwargs = load_kwargs(config)
                kwargs.pop(_LAYOUT)
            execute(
                external,
                args,
                out,
                structure,
                kwargs,
                backend,
                writes_output,
                allocate_output=allocate_output,
            )

        return callback

    forward = adapt(function)
    if transpose is not None:
        operation = get_linear_call(
            forward,
            adapt(transpose, transposed=True),
            abstract,
            abstract_transpose,
            graph_compatible=graph_compatible,
            workspace_bytes=workspace_spec,
        )
    else:

        def rule(primals, tangents, **kwargs):
            layout = layouts[kwargs.pop(_LAYOUT)]
            result = jvp(primals, tangents, **kwargs)
            values, actual = jax.tree_util.tree_flatten(result)
            if actual != layout.output_tree:
                raise ValueError(
                    "JVP result structure differs from output specification"
                )
            return tuple(values)

        def reverse_rule(primals, cotangents, **kwargs):
            layout = layouts[kwargs.pop(_LAYOUT)]
            tree = jax.tree_util.tree_unflatten(layout.output_tree, cotangents)
            return vjp(primals, tree, **kwargs)

        operation = get_call(
            forward,
            abstract,
            jvp=rule if jvp is not None else None,
            vjp=reverse_rule if vjp is not None else None,
            graph_compatible=graph_compatible,
            input_output_aliases=input_output_aliases,
            workspace_bytes=workspace_spec,
        )

    @wraps(function)
    def traced(*args, **kwargs):
        if _LAYOUT in kwargs:
            raise ValueError(
                "Internal shape metadata keywords are reserved for the binding"
            )
        if writes_output and "out" in kwargs:
            raise ValueError("out is supplied by the binding")
        args = tuple(jnp.asarray(x) for x in args)
        specs, structure = _specs(output, args, kwargs)
        layout = _Layout(
            inputs=tuple((x.shape, np.dtype(x.dtype)) for x in args),
            outputs=tuple((x.shape, np.dtype(x.dtype)) for x in specs),
            output_tree=structure,
            input_tree=jax.tree_util.tree_structure(
                args[0] if len(args) == 1 else args
            ),
        )
        with layout_lock:
            for index, previous in enumerate(layouts):
                if previous == layout:
                    break
            else:
                index = len(layouts)
                layouts.append(layout)
        result = operation(*args, **kwargs, **{_LAYOUT: index})
        return jax.tree_util.tree_unflatten(structure, result)

    @lru_cache(maxsize=128)
    def eager(static_names):
        return jax.jit(traced, static_argnames=static_names)

    @wraps(function)
    def wrapped(*args, **kwargs):
        if jax.config.jax_disable_jit or any(
            isinstance(x, jax.core.Tracer) for x in args
        ):
            return traced(*args, **kwargs)
        try:
            hash(tuple(kwargs.values()))
        except TypeError:
            return traced(*args, **kwargs)
        return eager(tuple(sorted(kwargs)))(*args, **kwargs)

    return wrapped
