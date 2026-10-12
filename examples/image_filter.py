"""Bind a library operation and supply its linear transpose."""

import jax
import jax.numpy as jnp
import numpy as np
from cupyx.scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter as reference_filter

from jaxbind_gpu import bind


def smooth(x, *, sigma, out):
    gaussian_filter(x, sigma=sigma, mode="wrap", output=out)


def main():
    # A symmetric periodic Gaussian filter is its own plain transpose.
    smooth_jax = bind(
        smooth,
        transpose=smooth,
        out=True,
    )
    host = np.random.default_rng(7).normal(size=(16, 20)).astype("float32")
    x = jnp.asarray(host)
    with jax.transfer_guard_device_to_host("disallow"):
        value = jax.jit(lambda x: smooth_jax(x, sigma=1.2))(x).block_until_ready()
        gradient = jax.jit(jax.grad(lambda x: smooth_jax(x, sigma=1.2).sum()))(x)
        gradient.block_until_ready()
    np.testing.assert_allclose(
        value, reference_filter(host, 1.2, mode="wrap"), atol=2e-7
    )
    np.testing.assert_allclose(gradient, np.ones_like(host), atol=2e-7)
    print("Image filter: SciPy reference and linear transpose passed")


if __name__ == "__main__":
    main()
