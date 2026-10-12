import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxbind_gpu as jb


def same_shape(x, **kwargs):
    return ((x.shape, x.dtype),)


def test_linear_jit_jvp_vjp_and_nested_vmap(gpu):
    def scale(out, args, kwargs_dump):
        factor = jb.load_kwargs(kwargs_dump)["factor"]
        gpu.multiply(args[0], factor, out=out[0])

    op = jb.get_linear_call(scale, scale, same_shape, same_shape)

    def f(x):
        return op(x, factor=3.0)[0]

    x = jnp.arange(24.0).reshape(2, 3, 4)
    np.testing.assert_allclose(jax.jit(f)(x), 3 * np.asarray(x))
    _, tangent = jax.jit(lambda x: jax.jvp(f, (x,), (jnp.ones_like(x),)))(x)
    np.testing.assert_allclose(tangent, np.full(x.shape, 3.0))
    np.testing.assert_allclose(jax.jit(jax.grad(lambda x: jnp.sum(f(x))))(x), 3)
    mapped = jax.jit(jax.vmap(jax.vmap(f)))(x)
    np.testing.assert_allclose(mapped, 3 * np.asarray(x))
    np.testing.assert_allclose(
        jax.jacfwd(jax.grad(lambda x: jnp.sum(f(x) ** 2)))(x),
        18 * np.eye(x.size).reshape(x.shape + x.shape),
    )


def test_buffers_are_device_views_and_stream_is_xla_stream(gpu):
    observed = []

    def copy(out, args, kwargs_dump):
        assert isinstance(args[0], gpu.ndarray)
        assert isinstance(out[0], gpu.ndarray)
        observed.append((args[0].data.ptr, gpu.cuda.get_current_stream().ptr))
        gpu.copyto(out[0], args[0])

    op = jb.get_linear_call(copy, copy, same_shape, same_shape)
    x = jnp.arange(5.0)
    pointer = x.unsafe_buffer_pointer()
    result = op(x)[0]
    result.block_until_ready()
    assert observed[0][0] == pointer
    assert observed[0][1] != 0
    np.testing.assert_array_equal(result, x)


def test_callback_error_reaches_python(gpu):
    def broken(out, args, kwargs_dump):
        raise ValueError("test callback failure")

    op = jb.get_linear_call(broken, broken, same_shape, same_shape)
    with pytest.raises(Exception, match="test callback failure"):
        op(jnp.ones(3))[0].block_until_ready()


def test_fixed_input_cannot_be_differentiated(gpu):
    def multiply(out, args, kwargs_dump):
        gpu.multiply(args[0], args[1], out=out[0])

    def abstract(a, x, **kwargs):
        return ((x.shape, x.dtype),)

    op = jb.get_linear_call(
        multiply, multiply, abstract, abstract, first_n_args_fixed=1
    )
    with pytest.raises(RuntimeError, match="not differentiable"):
        jax.jvp(lambda a: op(a, jnp.ones(3))[0], (jnp.ones(3),), (jnp.ones(3),))


def test_nonlinear_jvp_and_vjp(gpu):
    def square(out, args, kwargs_dump):
        gpu.multiply(args[0], args[0], out=out[0])

    def derivative(out, args, kwargs_dump):
        out[0][...] = 2 * args[0] * args[1]

    def abstract(x, *args, **kwargs):
        return ((x.shape, x.dtype),)

    op = jb.get_nonlinear_call(square, (derivative, derivative), abstract, abstract)

    def f(x):
        return op(x)[0]

    x = jnp.arange(1.0, 6.0)
    value, tangent = jax.jit(lambda x: jax.jvp(f, (x,), (jnp.ones_like(x),)))(x)
    np.testing.assert_allclose(value, np.arange(1, 6) ** 2)
    np.testing.assert_allclose(tangent, 2 * np.arange(1, 6))
    np.testing.assert_allclose(
        jax.jit(jax.grad(lambda x: jnp.sum(f(x))))(x), 2 * np.arange(1, 6)
    )


@pytest.mark.parametrize(
    "dtype",
    [
        np.bool_,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
    ],
)
def test_buffer_dtypes_layouts_and_scalars(gpu, dtype):
    def copy(out, args, kwargs_dump):
        gpu.copyto(out[0], args[0])

    op = jb.get_linear_call(copy, copy, same_shape, same_shape)
    x = jnp.asarray(np.arange(12).reshape(3, 4), dtype=dtype)
    value = jax.jit(lambda x: op(x.T)[0])(x)
    np.testing.assert_array_equal(value, np.asarray(x).T)
    scalar = jnp.asarray(1, dtype=dtype)
    np.testing.assert_array_equal(op(scalar)[0], 1)
    assert value.dtype == dtype


def test_empty_buffers(gpu):
    def copy(out, args, kwargs_dump):
        gpu.copyto(out[0], args[0])

    op = jb.get_linear_call(copy, copy, same_shape, same_shape)
    result = jax.jit(op)(jnp.empty((0, 3)))[0]
    assert result.shape == (0, 3)
    result.block_until_ready()


def test_native_batching_and_nonleading_axes(gpu):
    calls = []

    def scale(out, args, kwargs_dump):
        axes = jb.load_kwargs(kwargs_dump)["batch_axes"]
        calls.append(axes)
        gpu.multiply(args[0], 4, out=out[0])

    def abstract(x, batch_axes=None):
        axis = 0 if not batch_axes or not batch_axes[0] else batch_axes[0][-1]
        return ((x.shape, x.dtype, axis),)

    op = jb.get_linear_call(scale, scale, abstract, abstract, func_can_batch=True)
    x = jnp.arange(12.0).reshape(3, 4)

    def f(x):
        return op(x)[0]

    result = jax.jit(jax.vmap(f, in_axes=1, out_axes=1))(x)
    result.block_until_ready()
    assert calls == [((1,),)]
    np.testing.assert_allclose(result, 4 * np.asarray(x))


@pytest.mark.parametrize("use_lists", [False, True])
def test_multilinear_multiple_outputs(gpu, use_lists):
    def forward(out, args, kwargs_dump):
        a, b = args
        gpu.multiply(a, b, out=out[0])
        out[1][...] = 2 * out[0]

    def transpose(out, args, kwargs_dump):
        other, cot1, cot2 = args
        out[0][...] = other * (cot1 + 2 * cot2)

    def abstract(a, b, **kwargs):
        return ((a.shape, a.dtype), (a.shape, a.dtype))

    def abstract_t(other, cot1, cot2, **kwargs):
        return ((other.shape, other.dtype),)

    transposes = (transpose, transpose)
    abstracts = (abstract_t, abstract_t)
    if use_lists:
        transposes, abstracts = list(transposes), list(abstracts)
    op = jb.get_linear_call(forward, transposes, abstract, abstracts)
    a, b = jnp.arange(1.0, 5.0), jnp.arange(5.0, 9.0)

    def loss(a, b):
        return sum(jnp.sum(y) for y in op(a, b))

    da, db = jax.jit(jax.grad(loss, argnums=(0, 1)))(a, b)
    np.testing.assert_allclose(da, 3 * np.asarray(b))
    np.testing.assert_allclose(db, 3 * np.asarray(a))
    hessian = jax.jit(jax.hessian(loss, argnums=(0, 1)))(a, b)
    np.testing.assert_allclose(hessian[0][0], np.zeros((4, 4)))
    np.testing.assert_allclose(hessian[1][1], np.zeros((4, 4)))
    np.testing.assert_allclose(hessian[0][1], 3 * np.eye(4))
    np.testing.assert_allclose(hessian[1][0], 3 * np.eye(4))
    reverse_hessian = jax.jit(jax.jacrev(jax.grad(loss, argnums=0), argnums=1))(a, b)
    np.testing.assert_allclose(reverse_hessian, 3 * np.eye(4))


def test_compiled_callback_survives_binding_collection(gpu):
    import gc

    def factory():
        def scale(out, args, kwargs_dump):
            gpu.multiply(args[0], 7, out=out[0])

        op = jb.get_linear_call(scale, scale, same_shape, same_shape)
        return jax.jit(op).lower(jax.ShapeDtypeStruct((3,), np.float64)).compile()

    compiled = factory()
    gc.collect()
    np.testing.assert_allclose(compiled(jnp.ones(3))[0], 7)


def test_callbacks_must_write_outputs(gpu):
    def invalid(out, args, kwargs_dump):
        return args[0]

    op = jb.get_linear_call(invalid, invalid, same_shape, same_shape)
    with pytest.raises(Exception, match="write to out and return None"):
        op(jnp.ones(3))[0].block_until_ready()
