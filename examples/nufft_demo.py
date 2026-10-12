"""Compute a NUFFT and differentiate a real loss in the source strengths."""

import jax
import jax.numpy as jnp
import numpy as np

from cufinufft_demo import nufft1

jax.config.update("jax_enable_x64", True)


def main():
    points = jnp.linspace(-2.0, 2.0, 11)
    strengths = jnp.linspace(1.0, 2.0, 11)

    def loss(real_strengths, points):
        modes = nufft1(points, real_strengths.astype(jnp.complex128), 16, eps=1e-11)
        return jnp.sum(jnp.abs(modes) ** 2)

    with jax.transfer_guard_device_to_host("disallow"):
        value, gradient = jax.jit(jax.value_and_grad(loss))(strengths, points)
        value.block_until_ready()
        gradient.block_until_ready()
    frequencies = np.arange(-8, 8)
    matrix = np.exp(1j * frequencies[:, None] * np.asarray(points))
    expected = matrix @ np.asarray(strengths)
    np.testing.assert_allclose(value, np.sum(np.abs(expected) ** 2), rtol=1e-9)
    np.testing.assert_allclose(
        gradient, 2 * np.real(matrix.conj().T @ expected), rtol=1e-9
    )
    print("Device:", gradient.device)
    print("Loss:", float(value))
    print("Gradient norm:", float(jnp.linalg.norm(gradient)))


if __name__ == "__main__":
    main()
