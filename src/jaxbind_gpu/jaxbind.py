# SPDX-License-Identifier: BSD-2-Clause
# Authors: Martin Reinecke, Jakob Roth, Gordian Edenhofer

# Copyright(C) 2023, 2024 Max-Planck-Society

import pickle
import operator
from collections import namedtuple
from functools import partial
from typing import Union

import jax
import numpy as np
from jax.extend import core as jex_core
from jax.interpreters import ad, batching, mlir
from jax._src.ffi import _FfiEffect


__all__ = ["get_call", "get_linear_call", "get_nonlinear_call"]

from . import _bridge
from ._runtime import PROCESS_KEY, register_callback
from ._native import NativeFunction

jax.ffi.register_ffi_type(
    "jaxbind_gpu.callback_graph_state", _bridge.callback_graph_type(), platform="CUDA"
)
jax.ffi.register_ffi_type(
    "jaxbind_gpu.native_state", _bridge.native_type(), platform="CUDA"
)

for _name, _value in _bridge.registrations().items():
    jax.ffi.register_ffi_target(_name, _value, platform="CUDA")

# Immutable static descriptors are not array pytrees.
_shared_defaults = dict(
    abstract_T=None,
    first_n_args_fixed=0,
    can_batch=False,
    batch_axes=None,
    transpose_rule=None,
    batching_rule=None,
    has_side_effect=False,
    input_output_aliases=None,
)
_shared_args_names = ("abstract", *_shared_defaults)
LinearFunction = namedtuple(
    "LinearFunction",
    ("f", "T") + _shared_args_names,
    defaults=_shared_defaults.values(),
)
MultiLinearFunction = namedtuple(
    "MultiLinearFunction",
    ("f", "T") + _shared_args_names,
    defaults=_shared_defaults.values(),
)
NonLinearFunction = namedtuple(
    "NonLinearFunction",
    ("f", "derivatives") + _shared_args_names,
    defaults=_shared_defaults.values(),
)
RuleFunction = namedtuple(
    "RuleFunction",
    ("f", "jvp", "symbolic_zeros") + _shared_args_names + ("vjp",),
    defaults=(*_shared_defaults.values(), None),
)
TransposedMultiLinearFunction = namedtuple(
    "TransposedMultiLinearFunction",
    ("f", "original", "lin_arg", "n_original") + _shared_args_names,
    defaults=_shared_defaults.values(),
)


def _multilinear_transpose(func, index, n_original):
    return TransposedMultiLinearFunction(
        f=func.T[index],
        original=func,
        lin_arg=index,
        n_original=n_original,
        abstract=func.abstract_T[index],
        first_n_args_fixed=func.first_n_args_fixed,
        can_batch=func.can_batch,
    )


FunctionType = Union[
    LinearFunction,
    MultiLinearFunction,
    NonLinearFunction,
    RuleFunction,
    TransposedMultiLinearFunction,
]


class _Callback:
    def __init__(self, function, graph_compatible=False, workspace_bytes=0):
        self.function = function
        self.graph_compatible = graph_compatible
        self.workspace_bytes = workspace_bytes

    def __call__(self, *args):
        return self.function(*args)


class _JaxRule(_Callback):
    """A derivative recipe lowered as JAX code, never as a GPU callback."""

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class _VjpRule:
    """Validate cotangents and reuse each binding's linearized descriptor."""

    def __init__(self, function):
        self.function = function
        self.linearizations = {}

    def __call__(self, primals, cotangents, **kwargs):
        values = self.function(primals, cotangents, **kwargs)
        if not isinstance(values, (tuple, list)) or len(values) != len(primals):
            raise ValueError("VJP must return one cotangent per positional input")
        result = []
        for primal, value in zip(primals, values):
            expected = jax.typeof(primal).to_tangent_aval()
            if value is None:
                value = ad.instantiate_zeros(ad.Zero(expected))
            actual = jax.typeof(value)
            if actual.shape != expected.shape or actual.dtype != expected.dtype:
                raise ValueError("VJP cotangent shape/dtype differs from its input")
            result.append(value)
        return tuple(result)

    def linearized(self, function, count):
        if count not in self.linearizations:

            def abstract(*args, **kwargs):
                return tuple(
                    (x.shape, x.to_tangent_aval().dtype)
                    for x in _exec_abstract(*args[:count], _func=function, **kwargs)
                )

            linear = LinearFunction(
                f=_JaxRule(lambda *a, **kw: function.jvp(a[:count], a[count:], **kw)),
                T=self,
                abstract=abstract,
                first_n_args_fixed=count,
                transpose_rule=self,
            )
            # setdefault also keeps a canonical descriptor under concurrent tracing.
            self.linearizations.setdefault(count, linear)
        return self.linearizations[count]


def _callback(function, graph_compatible=False, workspace_bytes=0):
    if not callable(workspace_bytes):
        workspace_bytes = operator.index(workspace_bytes)
        if workspace_bytes < 0:
            raise ValueError("workspace_bytes must be nonnegative")
    if isinstance(function, NativeFunction):
        if workspace_bytes:
            raise ValueError("Native functions manage their workspace explicitly")
        if graph_compatible:
            raise ValueError("Set graph_compatible on NativeFunction.from_cabi instead")
        return function
    return _Callback(function, graph_compatible, workspace_bytes)


def _exec_abstract(*args, _func: FunctionType, **kwargs):
    """Evaluate the output shapes, including native batching metadata."""
    if _func.can_batch:
        assert "batch_axes" not in kwargs
        kwargs["batch_axes"] = _func.batch_axes
    ae = _func.abstract(*args, **kwargs)
    # NOTE, do not attempt to unpack the batch axes b/c it might not be there
    return tuple(jax.core.ShapedArray(*sdb[:2]) for sdb in ae)


def _effectful_abstract(*args, _func, **kwargs):
    outputs = _exec_abstract(*args, _func=_func, **kwargs)
    effectful = _func.has_side_effect or getattr(_func.f, "has_side_effect", False)
    return outputs, {_FfiEffect} if effectful else set()


def ffi_func(*args, _func: FunctionType, **kwargs):
    out_type = _exec_abstract(*args, _func=_func, **kwargs)
    if _func.can_batch:
        kwargs["batch_axes"] = _func.batch_axes
    if isinstance(_func.f, NativeFunction):
        return _func.f._lower(
            args,
            out_type,
            kwargs,
            has_side_effect=_func.has_side_effect,
            input_output_aliases=(
                None
                if _func.input_output_aliases is None
                else dict(_func.input_output_aliases)
            ),
        )
    workspace = _func.f.workspace_bytes
    workspace = operator.index(
        workspace(*args, **kwargs) if callable(workspace) else workspace
    )
    if workspace < 0:
        raise ValueError("workspace_bytes must be nonnegative")
    target = (
        "jaxbind_gpu_pycall_graph" if _func.f.graph_compatible else "jaxbind_gpu_pycall"
    )
    if workspace:
        workspace = (workspace + 511) // 512 * 512
        out_type = (*out_type, jax.ShapeDtypeStruct((workspace,), np.uint8))
        target += "_workspace"
    result = jax.ffi.ffi_call(
        target,
        out_type,
        has_side_effect=_func.has_side_effect,
        input_output_aliases=(
            None
            if _func.input_output_aliases is None
            else dict(_func.input_output_aliases)
        ),
    )(
        *args,
        callback_id=np.int64(register_callback(_func.f)),
        process_key=PROCESS_KEY,
        kwargs_dump=pickle.dumps(kwargs),
        **({"workspace_bytes": np.int64(workspace)} if workspace else {}),
    )
    return result[:-1] if workspace else result


_lowering_impl = mlir.lower_fun(ffi_func, multiple_results=True)


def _lowering(ctx, *args, _func, **kwargs):
    if isinstance(_func.f, _JaxRule):
        return mlir.lower_fun(
            partial(_func.f, **kwargs), multiple_results=True
        )(ctx, *args)
    if not isinstance(_func.f, NativeFunction):
        ctx.module_context.add_keepalive(_func.f)
    return _lowering_impl(ctx, *args, _func=_func, **kwargs)


def _cpu_lowering(ctx, *args, _func, **kwargs):
    if isinstance(_func.f, _JaxRule):
        return _lowering(ctx, *args, _func=_func, **kwargs)
    raise NotImplementedError(
        "jaxbind_gpu operations require a CUDA device."
    )


def _explicify_zeros(x):
    """Materialize symbolic zero tangents only when a callback needs arrays."""
    if isinstance(x, (tuple, list)):
        return [ad.instantiate_zeros(t) if isinstance(t, ad.Zero) else t for t in x]
    return ad.instantiate_zeros(x) if isinstance(x, ad.Zero) else x


def _jvp(args, tangents, *, _func: FunctionType, **kwargs):
    """Apply a traceable rule or the linear/multilinear product rule."""
    if isinstance(_func.f, _JaxRule):
        # Higher orders differentiate the recipe, including its primal inputs.
        return jax.jvp(
            partial(_func.f, **kwargs), tuple(args), tuple(_explicify_zeros(tangents))
        )
    res = _prim.bind(*args, **kwargs, _func=_func)

    if isinstance(_func, RuleFunction):
        if _func.jvp is None:
            raise TypeError("Provide a JAX JVP rule to differentiate this operation")
        if all(isinstance(t, ad.Zero) for t in tangents):
            return res, [ad.Zero.from_primal_value(r) for r in res]
        tangents = (
            tuple(None if isinstance(t, ad.Zero) else t for t in tangents)
            if _func.symbolic_zeros
            else tuple(_explicify_zeros(tangents))
        )
        if _func.vjp is None:
            tans = tuple(_func.jvp(tuple(args), tangents, **kwargs))
        else:
            linear = _func.vjp.linearized(_func, len(args))
            tans = _prim.bind(*args, *tangents, _func=linear, **kwargs)
        if len(tans) != len(res):
            raise ValueError("The JVP rule must return one tangent per output")
        return res, tans

    def zero_tans(tans):
        if isinstance(tans, (tuple, list)):
            return [isinstance(t, ad.Zero) for t in tans]
        return [isinstance(tans, ad.Zero)]

    n_args = len(args)
    n_f_args = _func.first_n_args_fixed
    assert n_args > n_f_args
    args_fixed = n_f_args * (True,) + (n_args - n_f_args) * (False,)
    assert len(args) == len(tangents) == len(args_fixed)

    tan_is_zero = zero_tans(tangents)
    assert len(args) == len(tan_is_zero)

    for i, (a, t) in enumerate(zip(args_fixed, tan_is_zero)):
        if a and not t:
            raise RuntimeError(f"{i}th positional argument not differentiable")

    if all(type(t) is ad.Zero for t in tangents):
        tans = list((jax.lax.zeros_like_array(a) for a in res))
        return (res, tans)

    tans = None
    if isinstance(_func, TransposedMultiLinearFunction):
        # Other primals are multilinear inputs; all cotangents together form
        # one linear input. Treating cotangents separately adds spurious terms
        # for multiple-output operations.
        count = _func.n_original - 1
        terms = []
        for i in range(n_f_args, count):
            if not tan_is_zero[i]:
                terms.append(
                    _prim.bind(
                        *args[:i], tangents[i], *args[i + 1 :], **kwargs, _func=_func
                    )
                )
        if not all(tan_is_zero[count:]):
            terms.append(
                _prim.bind(
                    *args[:count],
                    *_explicify_zeros(tangents[count:]),
                    **kwargs,
                    _func=_func,
                )
            )
        tans = tuple(sum(term[i] for term in terms) for i in range(len(res)))
    elif isinstance(_func, MultiLinearFunction):
        for i, t in enumerate(tangents):
            if not args_fixed[i] and not tan_is_zero[i]:
                t = _explicify_zeros(t)
                tn = _prim.bind(*args[:i], t, *args[i + 1 :], **kwargs, _func=_func)
                tans = (
                    tn if tans is None else tuple(t + tn_i for t, tn_i in zip(tans, tn))
                )
    elif isinstance(_func, LinearFunction):
        inp = []
        for a, f, t in zip(args, args_fixed, tangents):
            inp.append(a if f else _explicify_zeros(t))
        tans = _prim.bind(*inp, **kwargs, _func=_func)
    elif isinstance(_func, NonLinearFunction):
        f, f_T = _func.derivatives
        tan_in = [t for f, t in zip(args_fixed, tangents) if not f]
        tan_in = _explicify_zeros(tan_in)
        _func = LinearFunction(
            f=f,
            T=f_T,
            abstract=_func.abstract,
            abstract_T=_func.abstract_T,
            first_n_args_fixed=len(args),
            can_batch=_func.can_batch,
            batch_axes=(
                None
                if _func.batch_axes is None
                else _func.batch_axes + _func.batch_axes[n_f_args:]
            ),
            batching_rule=_func.batching_rule,
        )
        tans = _prim.bind(*args, *tan_in, **kwargs, _func=_func)
    else:
        raise TypeError(f"JVP for {type(_func)} not implemented")

    assert tans is not None
    return (res, tans)


# NOTE: for whatever reason JAX will pass each arg separately to _transpose
# and not as a tuple as for _jvp. Thus we need *args since we don't know
# the number of arguments.
def _transpose(cotangents, *args, _func: FunctionType, **kwargs):
    """Apply the declared transpose to the undefined linear inputs."""
    if isinstance(_func, TransposedMultiLinearFunction):
        original = _func.original
        count = _func.n_original - 1
        other, cots = args[:count], args[count:]
        other_unknown = [i for i, a in enumerate(other) if ad.is_undefined_primal(a)]
        cots_unknown = [ad.is_undefined_primal(a) for a in cots]
        result = [None] * len(args)
        cot = _explicify_zeros(cotangents)
        if len(cot) != 1:
            raise ValueError(
                "Each multilinear transpose must return one input cotangent"
            )
        restored = other[: _func.lin_arg] + (cot[0],) + other[_func.lin_arg :]
        if any(cots_unknown):
            if other_unknown:
                raise ValueError("Cannot jointly transpose nonlinear argument groups")
            values = _prim.bind(*restored, _func=original, **kwargs)
            result[count:] = [
                v if unknown else None for v, unknown in zip(values, cots_unknown)
            ]
        elif other_unknown:
            if len(other_unknown) != 1:
                raise ValueError(
                    "A multilinear transpose needs one linear argument group"
                )
            index = other_unknown[0]
            original_index = index + (index >= _func.lin_arg)
            if original_index < original.first_n_args_fixed:
                raise RuntimeError("Cannot transpose a fixed argument")
            descriptor = _multilinear_transpose(
                original, original_index, _func.n_original
            )
            values = _prim.bind(
                *restored[:original_index],
                *restored[original_index + 1 :],
                *cots,
                _func=descriptor,
                **kwargs,
            )
            result[index] = values[0]
        return result
    if not isinstance(_func, (LinearFunction, MultiLinearFunction)):
        raise TypeError(
            "Cannot transpose an opaque operation; provide a linear transpose "
            "or explicit jvp/vjp rules for the enclosing binding"
        )
    if _func.T is None:
        raise NotImplementedError(f"transpose of {_func} not implemented")
    n_args = len(args)
    n_f_args = _func.first_n_args_fixed
    assert n_args > n_f_args
    args_fixed = n_f_args * (True,) + (n_args - n_f_args) * (False,)
    arg_is_lin = [ad.is_undefined_primal(a) for a in args]
    assert len(args_fixed) >= len(arg_is_lin)

    for i, (a, is_lin) in enumerate(zip(args_fixed, arg_is_lin)):
        if a and is_lin:
            raise RuntimeError(
                f"Cannot transpose with respect to positional argument number {i}"
            )

    if isinstance(_func, MultiLinearFunction):
        assert sum(arg_is_lin) == 1
        lin_arg = arg_is_lin.index(True)
        c_in = cotangents
        a_in = args[:lin_arg] + args[lin_arg + 1 :]

        assert isinstance(_func.T, tuple)
        assert isinstance(_func.abstract_T, tuple)
        _func = _multilinear_transpose(_func, lin_arg, n_args)
        res = _prim.bind(*a_in, *c_in, **kwargs, _func=_func)
        res = [None] * lin_arg + res + [None] * (len(arg_is_lin) - (lin_arg + 1))
    elif isinstance(_func, LinearFunction):
        inp = []
        for a, f in zip(args, args_fixed):
            if f:
                assert not ad.is_undefined_primal(a)
                inp.append(a)
        cot = _explicify_zeros(cotangents)
        if _func.transpose_rule is not None:
            return n_f_args * [None] + list(
                _func.transpose_rule(tuple(inp), tuple(cot), **kwargs)
            )
        _func = _func._replace(
            f=_func.T,
            T=_func.f,
            abstract=_func.abstract_T,
            abstract_T=_func.abstract,
            first_n_args_fixed=len(inp),
        )
        res = _prim.bind(*inp, *cot, **kwargs, _func=_func)
        res = n_f_args * [None] + res
    else:
        raise TypeError(f"transpose for {type(_func)} not implemented")
    return res


def _batch(args, in_axes, *, _func: FunctionType, **kwargs):
    """Use native batching when supplied, otherwise map sequentially."""
    from .custom_map import smap

    if _func.batching_rule is not None:
        return _func.batching_rule(
            partial(_call, _func=_func), tuple(args), tuple(in_axes), **kwargs
        )
    if not _func.can_batch:
        y = smap(partial(_prim.bind, _func=_func, **kwargs), in_axes=in_axes)(*args)
        out_axes = [0] * len(y)
    else:
        batch_axes = _func.batch_axes
        batch_axes = ((),) * len(in_axes) if batch_axes is None else batch_axes
        new_batch_axes = []
        assert len(in_axes) == len(batch_axes)
        for ia, baxes in zip(in_axes, batch_axes):
            baxes_new = []
            if ia is not None:
                assert isinstance(ia, int)
                for b in baxes:
                    if b >= ia:
                        b += 1
                    baxes_new.append(b)
                baxes = tuple(tuple(baxes_new) + (ia,))
            new_batch_axes.append(baxes)
        new_batch_axes = tuple(new_batch_axes)
        _func = _func._replace(batch_axes=new_batch_axes)

        args_w = [jax.ShapeDtypeStruct(el.shape, el.dtype) for el in args]
        out_w = _func.abstract(*args_w, batch_axes=new_batch_axes, **kwargs)
        out_axes = [ba_wb for _, _, ba_wb in out_w]
        y = _call(*args, _func=_func, **kwargs)
    return y, out_axes


# One operation supplies CUDA execution and JAX transformation rules.
_prim = jex_core.Primitive("jaxbind_gpu_prim")
_prim.multiple_results = True
_prim.def_impl(partial(jax.interpreters.xla.apply_primitive, _prim))
_prim.def_effectful_abstract_eval(_effectful_abstract)
mlir.register_lowering(_prim, _lowering, platform="cuda")
mlir.register_lowering(_prim, _cpu_lowering, platform="cpu")
ad.primitive_jvps[_prim] = _jvp
ad.primitive_transposes[_prim] = _transpose
batching.primitive_batchers[_prim] = _batch


def _call(*args, _func: FunctionType, **kwargs):
    """Apply the independent CUDA primitive with static configuration."""
    return _prim.bind(*args, **kwargs, _func=_func)


def get_call(
    f,
    abstract,
    *,
    jvp=None,
    vjp=None,
    batching_rule=None,
    symbolic_zeros=False,
    has_side_effect=False,
    input_output_aliases=None,
    graph_compatible=False,
    workspace_bytes=0,
):
    """Bind an external operation with traceable JAX transformation rules.

    ``f(out, args, kwargs_dump)`` queues CUDA work on the current context's
    stream and writes the outputs described by ``abstract(*args, **kwargs)``.
    ``jvp(primals, tangents, **kwargs)`` returns a tuple of output tangents using
    JAX operations or other bindings. ``vjp(primals, cotangents, **kwargs)``
    returns one cotangent per input; None means zero. JVP-only rules supply
    reverse mode by transposition. VJP-only rules supply forward mode when the
    VJP can be transposed. Providing both selects each recipe explicitly.
    Rules must be linear in tangents/cotangents and mathematically consistent.
    Higher orders differentiate the recipes and require rules for external calls
    used there. Without either rule, differentiation raises an error.
    ``symbolic_zeros=True`` is JVP-only; inactive tangents arrive as None.

    By default vmap executes sequentially. An optional
    ``batching_rule(call, args, in_axes, **kwargs)`` returns outputs and their
    axes. It can move/broadcast inputs and invoke ``call`` once for a library
    with native batching, or explicitly fall back to a sequential map.
    """
    if not callable(f) or not callable(abstract):
        raise TypeError("Provide an external operation and an abstract function")
    if jvp is not None and not callable(jvp):
        raise TypeError("jvp must be a JAX-traceable callable")
    if vjp is not None and not callable(vjp):
        raise TypeError("vjp must be a JAX-traceable callable")
    if vjp is not None and symbolic_zeros:
        raise ValueError("symbolic_zeros is supported for JVP-only rules")
    if batching_rule is not None and not callable(batching_rule):
        raise TypeError("batching_rule must be callable")
    f = _callback(f, graph_compatible, workspace_bytes)
    if vjp is not None:
        vjp = _VjpRule(vjp)

        if jvp is None:

            def jvp(primals, tangents, **kwargs):
                outputs = _exec_abstract(*primals, _func=descriptor, **kwargs)
                specs = tuple(
                    jax.ShapeDtypeStruct(x.shape, x.to_tangent_aval().dtype)
                    for x in outputs
                )
                try:
                    return jax.linear_transpose(
                        lambda *c: vjp(primals, c, **kwargs), *specs
                    )(tuple(tangents))
                except (TypeError, NotImplementedError) as error:
                    raise TypeError(
                        "Cannot infer a JVP from this VJP; supply jvp= too. "
                        "External calls in a VJP need transpose rules for inference."
                    ) from error

    descriptor = RuleFunction(
        f=f,
        jvp=jvp,
        vjp=vjp,
        batching_rule=batching_rule,
        symbolic_zeros=symbolic_zeros,
        abstract=abstract,
        has_side_effect=has_side_effect,
        input_output_aliases=(
            None
            if input_output_aliases is None
            else tuple(sorted(input_output_aliases.items()))
        ),
    )
    return partial(_call, _func=descriptor)


def get_linear_call(
    f,
    f_T,
    /,
    abstract,
    abstract_T,
    *,
    first_n_args_fixed=0,
    func_can_batch=False,
    transpose_rule=None,
    batching_rule=None,
    graph_compatible=False,
    workspace_bytes=0,
) -> partial:
    """Bind a linear or multilinear GPU operation and its transposes.

    Callbacks receive ``(out, args, kwargs_dump)`` and write into ``out``.
    Abstract functions return tuples of ``(shape, dtype)``. Static keywords
    are decoded with ``load_kwargs``. Borrowed buffers cannot outlive a call.

    For multilinear operations, provide a tuple of transposes and matching
    abstract functions, one per input. ``first_n_args_fixed`` excludes leading
    arguments from differentiation. Complex rules use JAX's plain transpose.
    ``func_can_batch`` supplies ``batch_axes`` to callbacks and abstract
    functions, whose outputs then include ``(shape, dtype, batch_axis)``.
    Alternatively, ``batching_rule(call, args, in_axes, **kwargs)`` supplies
    the batching recipe. ``transpose_rule`` supplies a traceable JAX transpose.
    See docs/interface.md for stream, graph, and workspace contracts.
    """
    if isinstance(f_T, (tuple, list)):
        f_T = tuple(f_T)
        if not isinstance(abstract_T, (tuple, list)):
            raise TypeError(
                "Multilinear transposes need a sequence of abstract functions"
            )
        abstract_T = tuple(abstract_T)
        if len(f_T) != len(abstract_T):
            raise ValueError("Each transpose needs a matching abstract function")
    f = _callback(f, graph_compatible, workspace_bytes)
    f_T = (
        tuple(_callback(t, graph_compatible, workspace_bytes) for t in f_T)
        if isinstance(f_T, tuple)
        else _callback(f_T, graph_compatible, workspace_bytes)
    )
    kw = dict(
        f=f,
        T=f_T,
        abstract=abstract,
        abstract_T=abstract_T,
        first_n_args_fixed=first_n_args_fixed,
        can_batch=func_can_batch,
        transpose_rule=transpose_rule,
        batching_rule=batching_rule,
    )
    if isinstance(f_T, (tuple, list)):
        _func = MultiLinearFunction(**kw)
    else:
        _func = LinearFunction(**kw)
    return partial(_call, _func=_func)


def get_nonlinear_call(
    f,
    f_derivative,
    /,
    abstract,
    abstract_reverse,
    *,
    first_n_args_fixed=0,
    func_can_batch=False,
    workspace_bytes=0,
) -> partial:
    """Bind first-order GPU JVP/VJP callbacks.

    ``f_derivative`` contains the JVP and VJP callbacks. Each receives
    ``(out, args, kwargs_dump)`` and writes into the borrowed output arrays.
    The JVP receives all primals followed by differentiable input tangents;
    the VJP receives all primals followed by output cotangents.
    Abstract functions return output ``(shape, dtype)`` tuples.
    ``first_n_args_fixed`` excludes leading arguments from differentiation.
    Native batching follows ``get_linear_call``. Opaque callbacks do not
    infer higher derivatives; use ``get_call`` with a traceable rule for that.
    """
    f_derivative = tuple(f_derivative)
    if len(f_derivative) != 2 or not all(callable(f) for f in f_derivative):
        raise TypeError("f_derivative must contain a JVP callback and a VJP callback")
    _func = NonLinearFunction(
        f=_callback(f, workspace_bytes=workspace_bytes),
        abstract=abstract,
        abstract_T=abstract_reverse,
        first_n_args_fixed=first_n_args_fixed,
        can_batch=func_can_batch,
        derivatives=tuple(
            _callback(f, workspace_bytes=workspace_bytes) for f in f_derivative
        ),
    )
    return partial(_call, _func=_func)
