import importlib.util
import jax
import jax.numpy as jnp
import numpy as np
import pytest

if importlib.util.find_spec("cufinufft") is None:
    pytest.skip("Install the optional demo dependencies", allow_module_level=True)


from examples import cufinufft_demo as nufft


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("sign", [-1, 1])
def test_type3_against_dense_sum_and_derivatives(gpu, ndim, dtype, sign):
    rng = np.random.default_rng(52)
    real = np.float32 if dtype == np.complex64 else np.float64
    x = jnp.asarray(rng.uniform(-1, 1, (ndim, 5)).astype(real))
    s = jnp.asarray(rng.uniform(-3, 3, (ndim, 7)).astype(real))
    c = jnp.asarray(
        (rng.normal(size=(2, 5)) + 1j * rng.normal(size=(2, 5))).astype(dtype)
    )
    eps = 1e-5 if dtype == np.complex64 else 1e-11
    tol = 3e-4 if dtype == np.complex64 else 3e-8

    def operation(x, s, c):
        return nufft.nufft3(tuple(x), tuple(s), c, eps=eps, isign=sign)

    def oracle(x, s, c):
        # A100's default float32 dot uses TF32. The numerical oracle needs
        # full float32 products, including when JAX differentiates the sum.
        phase = jnp.matmul(s.T, x, precision=jax.lax.Precision.HIGHEST)
        return jnp.matmul(
            c, jnp.exp(sign * 1j * phase).T, precision=jax.lax.Precision.HIGHEST
        )

    np.testing.assert_allclose(
        jax.jit(operation)(x, s, c), oracle(x, s, c), atol=tol, rtol=tol
    )
    tangents = (jnp.ones_like(x) * 0.2, jnp.ones_like(s) * 0.3, jnp.ones_like(c))
    actual = jax.jit(lambda x, s, c: jax.jvp(operation, (x, s, c), tangents)[1])(
        x, s, c
    )
    expected = jax.jvp(oracle, (x, s, c), tangents)[1]
    np.testing.assert_allclose(actual, expected, atol=tol * 4, rtol=tol)

    def loss(fn):
        return lambda x, s, c: jnp.real(jnp.sum(fn(x, s, c)))

    actual = jax.jit(jax.grad(loss(operation), argnums=(0, 1, 2)))(x, s, c)
    expected = jax.grad(loss(oracle), argnums=(0, 1, 2))(x, s, c)
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(a, e, atol=tol * 4, rtol=tol)


def test_type3_mixed_coordinate_hessian(gpu):
    x = jnp.array([[-0.5, 0.2, 0.7]])
    s = jnp.array([[-2.0, -0.3, 0.8, 2.1]])
    c = jnp.array([1.0 + 0.5j, 0.2 - 0.6j, -0.4 + 1.2j])

    def loss(x, s):
        return jnp.sum(jnp.abs(nufft.nufft3(tuple(x), tuple(s), c, eps=1e-11)) ** 2)

    def oracle(x, s):
        return jnp.sum(jnp.abs(jnp.exp(1j * s.T @ x) @ c) ** 2)

    actual = jax.jit(jax.jacfwd(jax.grad(loss, argnums=0), argnums=1))(x, s)
    expected = jax.jacfwd(jax.grad(oracle, argnums=0), argnums=1)(x, s)
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-7)
