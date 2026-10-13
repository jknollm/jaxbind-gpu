"""SHTns Python transforms on GPU-resident JAX arrays."""

from threading import RLock

import jax
import jax.numpy as jnp
import numpy as np
import shtns

from jaxbind_gpu import bind


class SphericalTransform:
    """An orthonormal, Gaussian-grid SHTns plan with scalar JAX derivatives.

    Coefficients use SHTns's packed complex128 format for m >= 0.
    Spatial arrays are float64. Enable JAX x64 before creating arrays.
    """

    def __init__(self, lmax):
        if shtns._shtns.__name__ != "_shtns_cuda":
            raise RuntimeError("SHTns must be built with CUDA and Python support")
        if not jax.config.x64_enabled:
            raise RuntimeError(
                "Enable jax_enable_x64 for SHTns's double precision frontend"
            )
        self.plan = sh = shtns.sht(lmax)
        sh.set_grid(
            flags=shtns.sht_quick_init
            | shtns.SHT_ALLOW_GPU
            | shtns.SHT_THETA_CONTIGUOUS
        )
        self._lock = RLock()
        spectral = jax.ShapeDtypeStruct((sh.nlm,), np.complex128)
        spatial = jax.ShapeDtypeStruct(sh.spat_shape, np.float64)

        # SHTns supplies northern-hemisphere Gaussian weights; mirror them.
        north = sh.gauss_wts()
        weights = np.concatenate((north, north[::-1])) * (2 * np.pi / sh.nphi)
        if weights.size != sh.nlat:
            raise RuntimeError("This example requires an even Gaussian grid")
        multiplicity = np.where(sh.m == 0, 1.0, 2.0)

        def synthesis_vjp(primals, cotangent):
            # JAX complex cotangents use the bilinear transpose convention.
            weighted = cotangent / jnp.asarray(weights)
            return (jnp.conj(self.analysis(weighted)) * jnp.asarray(multiplicity),)

        def analysis_vjp(primals, cotangent):
            coefficients = jnp.conj(cotangent) / jnp.asarray(multiplicity)
            return (self.synthesis(coefficients) * jnp.asarray(weights),)

        self.synthesis = bind(
            self.direct_synthesis,
            output=spatial,
            out="allocate",
            jvp=lambda primals, tangents: self.synthesis(tangents[0]),
            vjp=synthesis_vjp,
        )
        self.analysis = bind(
            self.direct_analysis,
            output=spectral,
            out="allocate",
            jvp=lambda primals, tangents: self.analysis(tangents[0]),
            vjp=analysis_vjp,
        )
        self.vector_synthesis = bind(
            self.direct_synthesis,
            output=(spatial, spatial),
            out="allocate",
        )
        self.vector_analysis = bind(
            self.direct_analysis,
            output=(spectral, spectral),
            out="allocate",
        )

    def direct_synthesis(self, *coefficients):
        # SHTns configurations cannot be executed concurrently by host threads.
        with self._lock:
            return self.plan.synth(*coefficients)

    def direct_analysis(self, *fields):
        with self._lock:
            # The on-device FFT overwrites SHTns's analysis inputs. JAX inputs
            # must remain immutable; both benchmark paths use these copies.
            return self.plan.analys(*(field.copy() for field in fields))


def main():
    jax.config.update("jax_enable_x64", True)
    transform = SphericalTransform(255)
    sh = transform.plan
    host = np.zeros(sh.nlm, dtype=np.complex128)
    host[sh.idx(0, 0)] = 0.8
    host[sh.idx(1, 0)] = -0.4
    host[sh.idx(1, 1)] = 0.3 + 0.2j
    coefficients = jnp.asarray(host)
    north = sh.gauss_wts()
    weights = np.concatenate((north, north[::-1])) * (2 * np.pi / sh.nphi)

    def energy(a):
        return 0.5 * jnp.sum(jnp.asarray(weights) * transform.synthesis(a) ** 2)

    with jax.transfer_guard_device_to_host("disallow"):
        field = jax.jit(transform.synthesis)(coefficients).block_until_ready()
        recovered = transform.analysis(field).block_until_ready()
        tangent = jax.jvp(transform.synthesis, (coefficients,), (coefficients,))[1]
        tangent.block_until_ready()
        gradient = jax.jit(jax.grad(energy))(coefficients).block_until_ready()

    # Closed-form Y_00, Y_10 and Y_11, including the Condon-Shortley phase.
    cos_theta = sh.cos_theta[None, :]
    sin_theta = np.sqrt(1 - cos_theta**2)
    phi = (2 * np.pi * np.arange(sh.nphi) / sh.nphi)[:, None]
    expected = (
        0.8 / np.sqrt(4 * np.pi)
        - 0.4 * np.sqrt(3 / (4 * np.pi)) * cos_theta
        + 2
        * np.real(
            (0.3 + 0.2j) * -np.sqrt(3 / (8 * np.pi)) * sin_theta * np.exp(1j * phi)
        )
    )
    # FP64 transforms on a Gaussian grid, with mixed tolerance for zero modes.
    np.testing.assert_allclose(field, expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(tangent, expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(recovered, host, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        gradient,
        np.where(sh.m == 0, 1, 2) * np.conj(host),
        rtol=2e-12,
        atol=2e-12,
    )
    print("SHTns Python frontend: synthesis, analysis, JVP and gradient passed")


if __name__ == "__main__":
    main()
