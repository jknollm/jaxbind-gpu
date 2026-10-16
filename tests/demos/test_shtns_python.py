"""SHTns frontend integration, checked against analytic harmonic identities."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("shtns", reason="Install SHTns to check its Python example")
pytest.importorskip("_shtns_cuda", reason="Build SHTns with CUDA support")

from examples.shtns_python import SphericalTransform


@pytest.fixture
def transform(gpu):
    return SphericalTransform(255)


def test_scalar_derivatives_and_batching(transform):
    sh = transform.plan
    host = np.zeros(sh.nlm, np.complex128)
    host[sh.idx(0, 0)] = 0.8
    host[sh.idx(1, 0)] = -0.4
    host[sh.idx(1, 1)] = 0.3 + 0.2j
    coefficients = jnp.asarray(host)
    cotangent = np.zeros_like(host)
    cotangent[sh.idx(0, 0)] = 0.6
    cotangent[sh.idx(1, 0)] = 0.7
    cotangent[sh.idx(1, 1)] = 0.5 + 0.8j
    north = sh.gauss_wts()
    weights = np.concatenate((north, north[::-1])) * (2 * np.pi / sh.nphi)

    def energy(a):
        field = transform.synthesis(a)
        return 0.5 * jnp.sum(jnp.asarray(weights) * field**2)

    def analysis_loss(field):
        return jnp.real(jnp.sum(jnp.asarray(cotangent) * transform.analysis(field)))

    with jax.transfer_guard_device_to_host("disallow"):
        field = transform.synthesis(coefficients).block_until_ready()
        recovered = transform.analysis(field).block_until_ready()
        analysis_tangent = jax.jvp(transform.analysis, (field,), (field,))[1]
        analysis_tangent.block_until_ready()
        analysis_gradient = jax.jit(jax.grad(analysis_loss))(field).block_until_ready()
        gradient = jax.jit(jax.grad(energy))
        actual_gradient = gradient(coefficients).block_until_ready()
        hessian = jax.jvp(gradient, (coefficients,), (coefficients,))[1]
        hessian.block_until_ready()
        batched = jax.jit(jax.vmap(transform.synthesis))(
            jnp.stack((coefficients, 2 * coefficients))
        ).block_until_ready()

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
    expected_gradient = np.where(sh.m == 0, 1, 2) * np.conj(host)
    expected_analysis_gradient = weights * (
        0.6 / np.sqrt(4 * np.pi)
        + 0.7 * np.sqrt(3 / (4 * np.pi)) * cos_theta
        + np.real(
            (0.5 - 0.8j) * -np.sqrt(3 / (8 * np.pi)) * sin_theta * np.exp(1j * phi)
        )
    )
    # Gaussian quadrature is exact for these low harmonics; allow FP64 roundoff.
    for actual, wanted in (
        (field, expected),
        (recovered, host),
        (analysis_tangent, host),
        (analysis_gradient, expected_analysis_gradient),
        (actual_gradient, expected_gradient),
        (hessian, expected_gradient),
        (batched, np.stack((expected, 2 * expected))),
    ):
        np.testing.assert_allclose(actual, wanted, rtol=2e-12, atol=2e-12)


def test_vector_harmonic(transform, gpu):
    sh = transform.plan
    host = np.zeros(sh.nlm, np.complex128)
    host[sh.idx(1, 0)] = 1
    coefficients = jnp.asarray(host)
    zeros = jnp.zeros_like(coefficients)
    expected_theta = np.broadcast_to(
        -np.sqrt(3 / (4 * np.pi)) * np.sqrt(1 - sh.cos_theta**2), sh.spat_shape
    )
    with jax.transfer_guard_device_to_host("disallow"):
        theta, phi = transform.vector_synthesis(coefficients, zeros)
        jax.block_until_ready((theta, phi))
        before = tuple(gpu.from_dlpack(x).copy() for x in (theta, phi))
        spheroidal, toroidal = transform.vector_analysis(theta, phi)
        jax.block_until_ready((spheroidal, toroidal))
    for actual, wanted in (
        (theta, expected_theta),
        (phi, np.zeros(sh.spat_shape)),
        (spheroidal, host),
        (toroidal, np.zeros_like(host)),
    ):
        np.testing.assert_allclose(actual, wanted, rtol=2e-12, atol=2e-12)
    for field, reference in zip((theta, phi), before):
        gpu.testing.assert_array_equal(gpu.from_dlpack(field), reference)


def test_scalar_output_storage_and_input_preservation(transform, gpu):
    sh = transform.plan
    coefficient = jnp.zeros(sh.nlm, jnp.complex128).at[0].set(np.sqrt(4 * np.pi))
    field = transform.synthesis(coefficient).block_until_ready()
    before = gpu.from_dlpack(field).copy()
    for bound, inputs, table, copies_input in (
        (transform.synthesis, coefficient, sh._gpu_synth_list, False),
        (transform.analysis, field, sh._gpu_analys_list, True),
    ):
        original = table[0]
        pointers = []

        def audit(input_ptr, output_ptr):
            pointers.append((input_ptr, output_ptr))
            return original(input_ptr, output_ptr)

        table[0] = audit
        try:
            with jax.transfer_guard_device_to_host("disallow"):
                output = bound(inputs).block_until_ready()
            input_ptr, output_ptr = pointers[-1]
            assert output_ptr == output.unsafe_buffer_pointer()
            assert (input_ptr != inputs.unsafe_buffer_pointer()) == copies_input
        finally:
            table[0] = original
    gpu.testing.assert_array_equal(gpu.from_dlpack(field), before)
