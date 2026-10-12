"""Lesson 8: borrow Torch GPU tensors through DLPack on XLA's stream."""

import jax
import jax.numpy as jnp
import numpy as np
import torch

from jaxbind_gpu import bind


def main():
    sine = bind(
        torch.sin,
        backend="torch",
        out=True,
        jvp=lambda p, t: jnp.cos(p[0]) * t[0],
    )
    host = np.linspace(-1, 1, 17, dtype="float32")
    x = jnp.asarray(host)
    with jax.transfer_guard_device_to_host("disallow"):
        value = jax.jit(sine)(x).block_until_ready()
        gradient = jax.jit(jax.grad(lambda x: sine(x).sum()))(x).block_until_ready()
    np.testing.assert_allclose(value, np.sin(host), atol=1e-7)
    np.testing.assert_allclose(gradient, np.cos(host), atol=1e-7)
    print("Torch function: value and JAX derivative passed")


if __name__ == "__main__":
    main()
