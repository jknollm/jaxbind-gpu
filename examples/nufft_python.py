"""Bind cuFINUFFT's ordinary Python functions with fixed GPU coordinates."""

import cupy as cp
import jax
import jax.numpy as jnp
import numpy as np

from jaxbind_gpu import bind


def make_nufft(points, n_modes):
    """Differentiate strengths while keeping the CuPy coordinates fixed."""
    import cufinufft

    def forward(c, *, out):
        return cufinufft.nufft1d1(
            points,
            c,
            n_modes,
            out=out,
            eps=1e-6,
            isign=1,
            gpu_stream=cp.cuda.get_current_stream().ptr,
            gpu_device_id=cp.cuda.Device().id,
        )

    def transpose(cot, *, out):
        # JAX's complex VJP uses the plain transpose, so retain the Fourier sign.
        return cufinufft.nufft1d2(
            points,
            cot,
            out=out,
            eps=1e-6,
            isign=1,
            gpu_stream=cp.cuda.get_current_stream().ptr,
            gpu_device_id=cp.cuda.Device().id,
        )

    return bind(
        forward,
        output=lambda c: jax.ShapeDtypeStruct((n_modes,), c.dtype),
        out=True,
        transpose=transpose,
    )


def main():
    host_points = np.linspace(-2.5, 2.5, 11, dtype=np.float32)
    host_strengths = (
        np.linspace(0.2, 1.1, 11) + 1j * np.linspace(-0.4, 0.5, 11)
    ).astype(np.complex64)
    points = cp.asarray(host_points)
    strengths = jnp.asarray(host_strengths)
    nufft = make_nufft(points, 8)
    tangent = jnp.full_like(strengths, 0.3 + 0.7j)
    cotangent = jnp.arange(8, dtype=jnp.float32).astype(jnp.complex64) * (0.2 - 0.1j)

    with jax.transfer_guard_device_to_host("disallow"):
        value = jax.jit(nufft)(strengths).block_until_ready()
        jvp = jax.jit(lambda c, t: jax.jvp(nufft, (c,), (t,))[1])(
            strengths, tangent
        ).block_until_ready()
        vjp = jax.jit(lambda c, cot: jax.vjp(nufft, c)[1](cot)[0])(
            strengths, cotangent
        ).block_until_ready()

    # Centered modes and a positive Fourier sign, calculated independently.
    modes = np.arange(-4, 4)
    matrix = np.exp(1j * modes[:, None] * host_points.astype(np.float64))
    # eps=1e-6 plus float32 rounding; use mixed absolute/relative tolerance.
    for actual, expected in (
        (value, matrix @ host_strengths),
        (jvp, matrix @ np.full(11, 0.3 + 0.7j)),
        (vjp, matrix.T @ (np.arange(8) * (0.2 - 0.1j))),
    ):
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    print("cuFINUFFT Python functions: transform, JVP and VJP passed")


if __name__ == "__main__":
    main()
