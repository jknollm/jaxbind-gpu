import importlib.util
import itertools
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

if importlib.util.find_spec("cufinufft") is None:
    pytest.skip("Install the optional demo dependencies", allow_module_level=True)


from examples.cufinufft_demo import nufft1, nufft2


def direct_matrix(points, shape, isign, modeord):
    # Independent dense Fourier sum, rather than another NUFFT implementation.
    axes = [np.arange(-(n // 2), (n - 1) // 2 + 1) for n in shape]
    if modeord:
        axes = [np.fft.ifftshift(axis) for axis in axes]
    modes = np.array(list(itertools.product(*axes)))
    return np.exp(isign * 1j * modes @ np.asarray(points))


@pytest.mark.parametrize("shape", [(7,), (6,), (5, 4), (3, 4, 5)])
@pytest.mark.parametrize("isign", [-1, 1])
@pytest.mark.parametrize("modeord", [0, 1])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_forward_and_transpose_against_dense_fourier_sum(
    gpu, shape, isign, modeord, dtype
):
    real_dtype = np.float32 if dtype == np.complex64 else np.float64
    rng = np.random.default_rng(39)
    points = rng.uniform(-np.pi, np.pi, (len(shape), 11)).astype(real_dtype)
    strengths = (rng.normal(size=(2, 11)) + 1j * rng.normal(size=(2, 11))).astype(dtype)
    modes = (rng.normal(size=(2,) + shape) + 1j * rng.normal(size=(2,) + shape)).astype(
        dtype
    )
    coords = tuple(jnp.asarray(p) for p in points)
    eps = 1e-5 if dtype == np.complex64 else 1e-11
    # NUFFT error is relative to the source norm, so use a small absolute
    # allowance near cancellation. Single precision includes rounding error.
    tolerance = 2e-4 if dtype == np.complex64 else 2e-9
    matrix = direct_matrix(points, shape, isign, modeord)

    def f1(c):
        return nufft1(coords, c, shape, eps=eps, isign=isign, modeord=modeord)

    def f2(f):
        return nufft2(coords, f, eps=eps, isign=isign, modeord=modeord)

    value1 = jax.jit(f1)(jnp.asarray(strengths))
    value2 = jax.jit(f2)(jnp.asarray(modes))
    np.testing.assert_allclose(
        value1,
        (strengths @ matrix.T).reshape((2,) + shape),
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        value2, modes.reshape(2, -1) @ matrix, rtol=tolerance, atol=tolerance
    )
    _, pullback1 = jax.vjp(f1, jnp.asarray(strengths))
    _, pullback2 = jax.vjp(f2, jnp.asarray(modes))
    # JAX transposes complex linear maps without Hermitian conjugation.
    np.testing.assert_allclose(
        jax.jit(pullback1)(jnp.asarray(modes))[0],
        modes.reshape(2, -1) @ matrix,
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        jax.jit(pullback2)(jnp.asarray(strengths))[0],
        (strengths @ matrix.T).reshape((2,) + shape),
        rtol=tolerance,
        atol=tolerance,
    )


def test_jvp_vmap_and_real_loss_gradient(gpu):
    x = jnp.array([-1.3, -0.2, 0.4, 1.1])
    shape = (5,)
    matrix = direct_matrix([np.asarray(x)], shape, 1, 0)
    c = jnp.array([1 + 2j, 2 - 1j, 3 + 0.4j, -2 + 1j])
    tangent = jnp.array([1 - 1j, 2 + 1j, 0.1 - 0.2j, 3 + 1j])

    def f(c):
        return nufft1(x, c, shape, eps=1e-11)

    _, actual_tangent = jax.jit(lambda c, t: jax.jvp(f, (c,), (t,)))(c, tangent)
    np.testing.assert_allclose(actual_tangent, matrix @ np.asarray(tangent), atol=2e-9)
    batch = jnp.stack([c, tangent])
    np.testing.assert_allclose(
        jax.jit(jax.vmap(f))(batch), np.asarray(batch) @ matrix.T, atol=2e-9
    )

    def loss(real):
        return jnp.sum(jnp.abs(f(real.astype(jnp.complex128))) ** 2)

    real = jnp.arange(1.0, 5.0)
    expected_gradient = 2 * np.real(matrix.conj().T @ matrix @ np.asarray(real))
    np.testing.assert_allclose(
        jax.jit(jax.grad(loss))(real), expected_gradient, atol=2e-8
    )


@pytest.mark.parametrize("kind", [1, 2])
@pytest.mark.parametrize("shape", [(6,), (5, 4)])
@pytest.mark.parametrize("sign, order", [(-1, 0), (1, 1)])
def test_coordinate_derivatives_and_hessian(gpu, kind, shape, sign, order):
    rng = np.random.default_rng(4)
    x = jnp.asarray(rng.uniform(-1.0, 1.0, (len(shape), 4)))
    data_shape = (2, 4) if kind == 1 else (2,) + shape
    data = jnp.asarray(rng.normal(size=data_shape) + 1j * rng.normal(size=data_shape))
    axes = [np.arange(-(n // 2), (n - 1) // 2 + 1) for n in shape]
    if order:
        axes = [np.fft.ifftshift(axis) for axis in axes]
    modes = jnp.asarray(np.array(list(itertools.product(*axes))))

    def oracle(x, data):
        matrix = jnp.exp(sign * 1j * modes @ x)
        return data @ matrix.T if kind == 1 else data.reshape(2, -1) @ matrix

    def operation(x, data):
        kwargs = dict(eps=1e-11, isign=sign, modeord=order)
        return (
            nufft1(tuple(x), data, shape, **kwargs)
            if kind == 1
            else nufft2(tuple(x), data, **kwargs)
        )

    t = jnp.ones_like(x) * 0.3
    dt = jnp.ones_like(data) * (1.0 + 0.2j)
    actual = jax.jit(lambda x, data: jax.jvp(operation, (x, data), (t, dt))[1])(x, data)
    expected = jax.jvp(oracle, (x, data), (t, dt))[1]
    np.testing.assert_allclose(
        actual.reshape(expected.shape), expected, atol=2e-8, rtol=2e-8
    )

    def loss(f, x, data):
        return jnp.sum(jnp.abs(f(x, data)) ** 2)

    for transform in (jax.grad, jax.hessian):
        actual = jax.jit(transform(partial(loss, operation)))(x, data)
        expected = transform(partial(loss, oracle))(x, data)
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-7)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"n_modes": 0}, "positive"),
        ({"n_modes": 5, "isign": 0}, "isign"),
        ({"n_modes": 5, "eps": 0}, "eps"),
        ({"n_modes": 5, "modeord": 3}, "modeord"),
        ({"n_modes": 5, "gpu_stream": 0}, "managed"),
    ],
)
def test_invalid_configuration_fails_before_execution(kwargs, error):
    x = jax.ShapeDtypeStruct((4,), np.float64)
    c = jax.ShapeDtypeStruct((4,), np.complex128)
    with pytest.raises((ValueError, TypeError), match=error):
        jax.eval_shape(lambda x, c: nufft1(x, c, **kwargs), x, c)
