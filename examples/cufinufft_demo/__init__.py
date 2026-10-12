"""cuFINUFFT with differentiable strengths, coefficients, and coordinates."""

import math
import operator
import uuid
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.custom_derivatives import SymbolicZero

from jaxbind_gpu import current_context, get_linear_call, load_kwargs
from ._native import CufinufftFunction


def synchronize(*, clear_cache=False):
    """Drain the demo's native plans and the binding's Python resources."""
    from . import _bridge
    import jaxbind_gpu

    _bridge.native_synchronize(clear_cache)
    jaxbind_gpu.synchronize(clear_cache=clear_cache)


def configure_runtime(**options):
    """Set this demo's plan limit along with the general binding's resource limits."""
    from . import _bridge
    import jaxbind_gpu

    jaxbind_gpu.configure_runtime(**options)
    _bridge.native_configure(options.get("max_cache_entries", 128))


def runtime_stats():
    """Add the demo's native plan count to general resource counters."""
    from . import _bridge
    import jaxbind_gpu

    return dict(jaxbind_gpu.runtime_stats(), native_cached=_bridge.native_cached())


def _points(points):
    if not isinstance(points, (tuple, list)):
        points = (points,)
    if not 1 <= len(points) <= 3:
        raise ValueError("Provide one, two, or three coordinate arrays")
    return tuple(points)


def _modes(n_modes, ndim):
    if ndim == 1:
        try:
            n_modes = (operator.index(n_modes),)
        except TypeError:
            pass
    try:
        modes = tuple(operator.index(n) for n in n_modes)
    except TypeError as error:
        raise TypeError("n_modes must contain integer mode counts") from error
    if len(modes) != ndim or any(n <= 0 for n in modes):
        raise ValueError("n_modes must have one positive integer per coordinate array")
    return modes


def _configuration(eps, isign, modeord, options):
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if isign not in (-1, 1):
        raise ValueError("isign must be -1 or +1")
    if modeord not in (0, 1):
        raise ValueError("modeord must be 0 or 1")
    if {"gpu_device_id", "gpu_stream"} & options.keys():
        raise ValueError("gpu_stream and gpu_device_id are managed by jaxbind_gpu")


def _validate(points, data):
    if np.dtype(data.dtype) not in (np.dtype("complex64"), np.dtype("complex128")):
        raise TypeError("cuFINUFFT data must have dtype complex64 or complex128")
    real_dtype = np.dtype("float32" if data.dtype == np.complex64 else "float64")
    if any(len(p.shape) != 1 for p in points):
        raise ValueError("Coordinate arrays must be one-dimensional")
    if any(p.shape != points[0].shape for p in points):
        raise ValueError("Coordinate arrays must have identical shapes")
    if any(np.dtype(p.dtype) != real_dtype for p in points):
        raise TypeError(
            f"Coordinates must have dtype {real_dtype} for {data.dtype} data"
        )
    if points[0].shape[0] == 0:
        raise ValueError("At least one nonuniform point is required")


def _abstract1(*args, ndim, n_modes, **kwargs):
    points, data = args[:ndim], args[ndim]
    _validate(points, data)
    if not data.shape or data.shape[-1] != points[0].shape[0]:
        raise ValueError("The last strength dimension must match the point count")
    if any(n == 0 for n in data.shape):
        raise ValueError("Empty transform batches are not supported")
    return ((data.shape[:-1] + n_modes, data.dtype),)


def _abstract2(*args, ndim, n_modes, **kwargs):
    points, data = args[:ndim], args[ndim]
    _validate(points, data)
    if len(data.shape) < ndim or data.shape[-ndim:] != n_modes:
        raise ValueError("The trailing Fourier dimensions must match n_modes")
    if any(n == 0 for n in data.shape):
        raise ValueError("Empty transform batches are not supported")
    return ((data.shape[:-ndim] + points[0].shape, data.dtype),)


def _execute(out, args, kwargs_dump, *, ndim, nufft_type):
    import cufinufft

    kwargs = load_kwargs(kwargs_dump)
    kwargs.pop("geometry_id", None)
    modes = kwargs.pop("n_modes")
    options = dict(kwargs.pop("options"))
    points, data = args[:ndim], args[ndim]
    leading = data.shape[:-1] if nufft_type == 1 else data.shape[:-ndim]
    n_trans = math.prod(leading)
    # cuFINUFFT accepts one transform dimension. Collapse any leading batch
    # dimensions to it, then write directly into the borrowed XLA output.
    data_shape = (n_trans,) + ((points[0].size,) if nufft_type == 1 else modes)
    out_shape = (n_trans,) + (modes if nufft_type == 1 else (points[0].size,))
    context = current_context()
    key = (
        nufft_type,
        modes,
        n_trans,
        data.dtype.str,
        tuple(sorted(kwargs.items())),
        tuple(sorted(options.items())),
    )
    plan = context.cached(
        key,
        lambda: cufinufft.Plan(
            nufft_type,
            modes,
            n_trans=n_trans,
            dtype=data.dtype,
            gpu_stream=context.stream.ptr,
            gpu_device_id=context.device,
            **kwargs,
            **options,
        ),
    )
    try:
        plan.setpts(*points)
        plan.execute(data.reshape(data_shape), out=out[0].reshape(out_shape))
    finally:
        # The runtime retains all borrowed buffers until the completion event.
        # A cached plan must not keep stale views of XLA allocations between calls.
        plan._references.clear()


def _bindings(ndim, native):
    forward = (
        CufinufftFunction(ndim, 1)
        if native
        else partial(_execute, ndim=ndim, nufft_type=1)
    )
    transpose = (
        CufinufftFunction(ndim, 2)
        if native
        else partial(_execute, ndim=ndim, nufft_type=2)
    )
    abstract1 = partial(_abstract1, ndim=ndim)
    abstract2 = partial(_abstract2, ndim=ndim)

    def transpose1(points, cotangents, **kw):
        config = (
            kw["n_modes"],
            kw["eps"],
            kw["isign"],
            kw["modeord"],
            kw["options"],
            "native" if native else "python",
            kw["geometry_id"],
        )
        return (_transform(2, config, points, cotangents[0]),)

    def transpose2(points, cotangents, **kw):
        config = (
            kw["n_modes"],
            kw["eps"],
            kw["isign"],
            kw["modeord"],
            kw["options"],
            "native" if native else "python",
            kw["geometry_id"],
        )
        return (_transform(1, config, points, cotangents[0]),)

    def batch(call, args, axes, **kw):
        if all(axis is None for axis in axes[:ndim]) and axes[-1] is not None:
            data = jnp.moveaxis(args[-1], axes[-1], 0)
            return call(*args[:ndim], data, **kw), (0,)
        from ..custom_map import smap

        result = smap(partial(call, **kw), in_axes=axes)(*args)
        return result, (0,)

    return (
        get_linear_call(
            forward,
            transpose,
            abstract1,
            abstract2,
            first_n_args_fixed=ndim,
            transpose_rule=transpose1,
            batching_rule=batch,
        ),
        get_linear_call(
            transpose,
            forward,
            abstract2,
            abstract1,
            first_n_args_fixed=ndim,
            transpose_rule=transpose2,
            batching_rule=batch,
        ),
    )


_ops = {
    backend: {ndim: _bindings(ndim, backend == "native") for ndim in (1, 2, 3)}
    for backend in ("native", "python")
}


@partial(jax.custom_jvp, nondiff_argnums=(0, 1))
def _transform(kind, config, points, data):
    modes, eps, sign, order, options, backend, geometry_id = config
    options = _plan_options(options, len(points), data.dtype)
    return _ops[backend][len(points)][kind - 1](
        *points,
        data,
        n_modes=modes,
        eps=eps,
        isign=sign,
        modeord=order,
        options=options,
        geometry_id=geometry_id,
    )[0]


def _plan_options(options, ndim, dtype):
    options = dict(options)
    if ndim == 3 and np.dtype(dtype) == np.dtype("complex128"):
        # Double-precision 3D spreading can exceed practical shared-memory
        # budgets. The point-driven method also avoids the slow default seen
        # in the independent jax-finufft comparison on the A100.
        options.setdefault("gpu_method", 1)
    return tuple(sorted(options.items()))


def _transform_jvp(kind, config, primals, tangents):
    points, data = primals
    dpoints, ddata = tangents
    result = _transform(kind, config, points, data)
    tangent = jnp.zeros_like(result)
    if not isinstance(ddata, SymbolicZero):
        tangent = _transform(kind, config, points, ddata)
    modes, _, sign, order, _, _, _ = config
    for dimension, dx in enumerate(dpoints):
        if isinstance(dx, SymbolicZero):
            continue
        n = modes[dimension]
        k = jnp.arange(-(n // 2), (n - 1) // 2 + 1, dtype=points[0].dtype)
        if order:
            k = jnp.fft.ifftshift(k)
        shape = [1] * len(modes)
        shape[dimension] = n
        factor = (1j * sign * k).reshape(shape)
        if kind == 1:
            tangent = tangent + factor * _transform(kind, config, points, data * dx)
        else:
            tangent = tangent + dx * _transform(kind, config, points, data * factor)
    return result, tangent


_transform.defjvp(_transform_jvp, symbolic_zeros=True)


def nufft1(
    points,
    strengths,
    n_modes,
    *,
    eps=1e-6,
    isign=1,
    modeord=0,
    backend="native",
    **options,
):
    """Nonuniform strengths to a uniform Fourier grid, without normalization.

    ``points`` is one coordinate array, or a tuple of 1-3 coordinate arrays.
    ``strengths`` has shape ``(..., M)`` and complex64/complex128 dtype.
    Coordinates have matching float32/float64 precision and shape ``(M,)``.
    The result has shape ``(..., *n_modes)``. Leading dimensions are transform
    batches. Static options are passed to ``cufinufft.Plan``.
    """
    points = _points(points)
    modes = _modes(n_modes, len(points))
    _configuration(eps, isign, modeord, options)
    if backend not in _ops:
        raise ValueError("backend must be 'native' or 'python'")
    config = (modes, eps, isign, modeord, tuple(sorted(options.items())), backend, 0)
    return _transform(1, config, points, strengths)


def nufft2(
    points, coefficients, *, eps=1e-6, isign=-1, modeord=0, backend="native", **options
):
    """Uniform Fourier coefficients to nonuniform points, without normalization.

    The trailing 1-3 dimensions of ``coefficients`` are the Fourier grid; any
    leading dimensions are transform batches. The result has shape ``(..., M)``.
    Coordinates follow the same convention as :func:`nufft1`.
    """
    points = _points(points)
    ndim = len(points)
    modes = _modes(coefficients.shape[-ndim:], ndim)
    _configuration(eps, isign, modeord, options)
    if backend not in _ops:
        raise ValueError("backend must be 'native' or 'python'")
    config = (modes, eps, isign, modeord, tuple(sorted(options.items())), backend, 0)
    return _transform(2, config, points, coefficients)


def _abstract3(*args, ndim, transpose=False, **kwargs):
    source, target, data = args[:ndim], args[ndim : 2 * ndim], args[-1]
    if transpose:
        source, target = target, source
    _validate(source, data)
    _validate(target, data)
    if not data.shape or data.shape[-1] != source[0].shape[0]:
        raise ValueError("The last strength dimension must match the source count")
    if any(n == 0 for n in data.shape):
        raise ValueError("Empty transform batches are not supported")
    return ((data.shape[:-1] + target[0].shape, data.dtype),)


def _execute3(out, args, kwargs_dump, *, ndim, transpose=False):
    import cufinufft

    kwargs = load_kwargs(kwargs_dump)
    kwargs.pop("geometry_id", None)
    options = dict(kwargs.pop("options"))
    source, target, data = args[:ndim], args[ndim : 2 * ndim], args[-1]
    if transpose:
        source, target = target, source
    context = current_context()
    n_trans = math.prod(data.shape[:-1])
    key = (
        3,
        ndim,
        n_trans,
        data.dtype.str,
        tuple(sorted(kwargs.items())),
        tuple(sorted(options.items())),
    )
    plan = context.cached(
        key,
        lambda: cufinufft.Plan(
            3,
            ndim,
            n_trans=n_trans,
            dtype=data.dtype,
            gpu_stream=context.stream.ptr,
            gpu_device_id=context.device,
            **kwargs,
            **options,
        ),
    )
    try:
        plan.setpts(*(source + (None,) * (3 - ndim) + target + (None,) * (3 - ndim)))
        plan.execute(data.reshape(n_trans, -1), out=out[0].reshape(n_trans, -1))
    finally:
        plan._references.clear()


def _binding3(ndim, native):
    forward = CufinufftFunction(ndim, 3) if native else partial(_execute3, ndim=ndim)
    reverse = (
        CufinufftFunction(ndim, 3, transpose=True)
        if native
        else partial(_execute3, ndim=ndim, transpose=True)
    )

    def transpose_rule(points, cotangents, **kwargs):
        geometry_id = kwargs["geometry_id"]
        config = (
            kwargs["eps"],
            kwargs["isign"],
            kwargs["options"],
            "native" if native else "python",
            geometry_id ^ (1 << 63) if geometry_id else 0,
        )
        return (_transform3(config, points[ndim:], points[:ndim], cotangents[0]),)

    def batch(call, args, axes, **kwargs):
        if all(axis is None for axis in axes[:-1]) and axes[-1] is not None:
            return call(*args[:-1], jnp.moveaxis(args[-1], axes[-1], 0), **kwargs), (0,)
        from ..custom_map import smap

        return smap(partial(call, **kwargs), in_axes=axes)(*args), (0,)

    return get_linear_call(
        forward,
        reverse,
        partial(_abstract3, ndim=ndim),
        partial(_abstract3, ndim=ndim, transpose=True),
        first_n_args_fixed=2 * ndim,
        transpose_rule=transpose_rule,
        batching_rule=batch,
    )


_ops3 = {
    backend: {ndim: _binding3(ndim, backend == "native") for ndim in (1, 2, 3)}
    for backend in ("native", "python")
}


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def _transform3(config, source, target, data):
    eps, sign, options, backend, geometry_id = config
    options = _plan_options(options, len(source), data.dtype)
    return _ops3[backend][len(source)](
        *source,
        *target,
        data,
        eps=eps,
        isign=sign,
        options=options,
        geometry_id=geometry_id,
    )[0]


def _transform3_jvp(config, primals, tangents):
    source, target, data = primals
    dx, ds, dc = tangents
    result = _transform3(config, source, target, data)
    tangent = jnp.zeros_like(result)
    if not isinstance(dc, SymbolicZero):
        tangent = _transform3(config, source, target, dc)
    factor = 1j * config[1]
    for x, s, tx, ts in zip(source, target, dx, ds):
        if not isinstance(tx, SymbolicZero):
            tangent = tangent + factor * s * _transform3(
                config, source, target, data * tx
            )
        if not isinstance(ts, SymbolicZero):
            tangent = tangent + factor * ts * _transform3(
                config, source, target, data * x
            )
    return result, tangent


_transform3.defjvp(_transform3_jvp, symbolic_zeros=True)


def nufft3(
    source, target, strengths, *, eps=1e-6, isign=1, backend="native", **options
):
    """Nonuniform sources to nonuniform frequencies, without normalization.

    ``source`` and ``target`` each contain 1-3 matching-precision coordinate
    arrays of shape ``(M,)`` and ``(N,)`` respectively. Strengths have shape
    ``(..., M)``. The result is ``sum_j c_j exp(i*isign*s_k.x_j)`` with shape
    ``(..., N)``. Both coordinate sets and strengths support differentiation.
    """
    source, target = _points(source), _points(target)
    if len(source) != len(target):
        raise ValueError("Source and target must have the same dimension")
    # cuFINUFFT 2.5.1's automatic 1.25 upsampling can miss requested accuracy
    # in single-precision type 3. The conservative default passes direct sums;
    # callers can still explicitly choose another upsampling factor.
    options.setdefault("upsampfac", 2.0)
    _configuration(eps, isign, 0, options)
    if backend not in _ops3:
        raise ValueError("backend must be 'native' or 'python'")
    config = (eps, isign, tuple(sorted(options.items())), backend, 0)
    return _transform3(config, source, target, strengths)


def _freeze(points):
    points = _points(points)
    if any(isinstance(p, jax.core.Tracer) for p in points):
        raise ValueError(
            "Prepare geometry outside JAX transformations; use dynamic NUFFTs for coordinate derivatives"
        )
    return tuple(jnp.array(p, copy=True) for p in points)


def _geometry_id():
    return (uuid.uuid4().int & ((1 << 63) - 1)) or 1


def prepare_nufft1(
    points, n_modes, *, eps=1e-6, isign=1, modeord=0, backend="native", **options
):
    """Freeze geometry and return an operator of strengths only.

    Native plans own coordinate copies and perform setpts once per cached
    device/stream/batch configuration. Construct outside JAX transformations.
    The returned operator supports coefficient derivatives and native batching.
    """
    points = _freeze(points)
    modes = _modes(n_modes, len(points))
    _configuration(eps, isign, modeord, options)
    if backend not in _ops:
        raise ValueError("backend must be 'native' or 'python'")
    config = (
        modes,
        eps,
        isign,
        modeord,
        tuple(sorted(options.items())),
        backend,
        _geometry_id(),
    )
    return partial(_transform, 1, config, points)


def prepare_nufft2(
    points, *, eps=1e-6, isign=-1, modeord=0, backend="native", **options
):
    """Freeze geometry and return an operator of Fourier coefficients only."""
    points = _freeze(points)
    _configuration(eps, isign, modeord, options)
    if backend not in _ops:
        raise ValueError("backend must be 'native' or 'python'")
    geometry_id = _geometry_id()

    def operation(coefficients):
        modes = _modes(coefficients.shape[-len(points) :], len(points))
        config = (
            modes,
            eps,
            isign,
            modeord,
            tuple(sorted(options.items())),
            backend,
            geometry_id,
        )
        return _transform(2, config, points, coefficients)

    return operation


def prepare_nufft3(source, target, *, eps=1e-6, isign=1, backend="native", **options):
    """Freeze both coordinate sets and return an operator of strengths only."""
    source, target = _freeze(source), _freeze(target)
    if len(source) != len(target):
        raise ValueError("Source and target must have the same dimension")
    options.setdefault("upsampfac", 2.0)
    _configuration(eps, isign, 0, options)
    if backend not in _ops3:
        raise ValueError("backend must be 'native' or 'python'")
    config = (eps, isign, tuple(sorted(options.items())), backend, _geometry_id())
    return partial(_transform3, config, source, target)
