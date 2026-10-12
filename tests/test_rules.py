import jax
import jax.numpy as jnp
import numpy as np

import jaxbind_gpu as jb


def test_jax_rules_support_higher_derivatives(gpu):
    def exponential(out, args, kwargs_dump):
        gpu.exp(args[0], out=out[0])

    def abstract(x):
        return ((x.shape, x.dtype),)

    op = jb.get_call(exponential, abstract, jvp=lambda p, t: (op(p[0])[0] * t[0],))

    def loss(x):
        return jnp.sum(op(x)[0])

    x = jnp.array([-0.5, 0.2, 1.0])
    for derivative in (
        jax.grad(loss),
        jax.jacfwd(jax.grad(loss)),
        jax.jacrev(jax.jacfwd(jax.grad(loss))),
    ):
        actual = jax.jit(derivative)(x)
        expected = np.zeros(actual.shape)
        for i in range(3):
            expected[(i,) * actual.ndim] = np.exp(float(x[i]))
        np.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_symbolic_zero_rules_and_multioutputs(gpu):
    seen = []

    def multiply(out, args, kwargs_dump):
        gpu.multiply(*args, out=out[0])
        gpu.add(*args, out=out[1])

    def rule(p, t):
        seen.append(tuple(ti is None for ti in t))
        a = jnp.zeros_like(p[0]) if t[0] is None else t[0]
        b = jnp.zeros_like(p[1]) if t[1] is None else t[1]
        return (a * p[1] + p[0] * b, a + b)

    def abstract(x, y):
        return ((x.shape, x.dtype), (x.shape, x.dtype))

    op = jb.get_call(multiply, abstract, jvp=rule, symbolic_zeros=True)
    actual = jax.jit(jax.grad(lambda x: jnp.sum(op(x, jnp.ones_like(x))[0])))(
        jnp.arange(4.0)
    )
    np.testing.assert_array_equal(actual, np.ones(4))
    assert (False, True) in seen
