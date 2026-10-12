"""Lesson 2: bind an ordinary array function with a traceable derivative."""

import cupy as cp
import jax
import jax.numpy as jnp
import numpy as np

from jaxbind_gpu import bind


def main():
    square = bind(
        cp.square,
        out=True,
        vjp=lambda p, c: (2 * p[0] * c,),
    )
    x = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)

    def loss(x):
        return square(x).sum()

    with jax.transfer_guard_device_to_host("disallow"):
        value = jax.jit(jax.vmap(square))(x).block_until_ready()
        gradient = jax.jit(jax.grad(loss))(x).block_until_ready()
        hessian = jax.jit(jax.hessian(loss))(x).block_until_ready()
        tangent = jax.jit(lambda x: jax.jvp(square, (x,), (jnp.ones_like(x),))[1])(
            x
        ).block_until_ready()
        cotangent = jax.jit(lambda x: jax.vjp(square, x)[1](jnp.full_like(x, 3))[0])(
            x
        ).block_until_ready()
    # A VJP returns one cotangent per input. Its transpose also supplies the JVP.
    np.testing.assert_array_equal(value, np.arange(6).reshape(2, 3) ** 2)
    np.testing.assert_array_equal(gradient, 2 * np.arange(6).reshape(2, 3))
    np.testing.assert_array_equal(np.asarray(hessian).reshape(6, 6), 2 * np.eye(6))
    np.testing.assert_array_equal(tangent, 2 * np.arange(6).reshape(2, 3))
    np.testing.assert_array_equal(cotangent, 6 * np.arange(6).reshape(2, 3))
    print("Array function: JVP, VJP, batching, gradient, and Hessian passed")


if __name__ == "__main__":
    main()
