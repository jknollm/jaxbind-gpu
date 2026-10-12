"""Application acceptance against analytic, NumPy, and JAX oracles."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from applications.operators import (  # noqa: E402
    DenseSolve,
    ImageBlur,
    SparseDiffusion,
    SpectralPoisson,
    ReactionDiffusion,
    TorchConvolution,
    torch_reference,
)


def periodic_blur(x):
    result = np.zeros_like(x)
    for i, a in enumerate((1, 2, 1)):
        for j, b in enumerate((1, 2, 1)):
            result += (a * b / 16) * np.roll(x, (i - 1, j - 1), axis=(-2, -1))
    return result


def sparse_inputs(n=11):
    # Unequal upper/lower coefficients make the transpose independently visible.
    rows, columns, data = [], [], []
    for i in range(n):
        for j in range(max(0, i - 1), min(n, i + 2)):
            rows.append(i)
            columns.append(j)
            data.append(2.5 if i == j else -0.3 if i < j else -0.7)
    indptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=n))].astype("int32")
    matrix = np.zeros((n, n), dtype="float32")
    matrix[rows, columns] = data
    return (
        np.array(data, dtype="float32"),
        np.array(columns, dtype="int32"),
        indptr,
    ), matrix


@pytest.mark.parametrize("name", ["spectral", "dense", "sparse", "image"])
def test_application_value_derivatives_batching_and_residency(gpu, name):
    rng = np.random.default_rng(412)
    fixed = ()
    if name == "spectral":
        op = SpectralPoisson()
        x = rng.normal(size=(8, 10)).astype("float32")
        h, w = x.shape
        fy, fx = np.fft.fftfreq(h), np.fft.fftfreq(w)
        multiplier = 1 / (
            1
            + 4 * np.sin(np.pi * fy[:, None]) ** 2
            + 4 * np.sin(np.pi * fx[None, :]) ** 2
        )

        def oracle(x, transpose=False):
            return np.fft.ifft2(np.fft.fft2(x) * multiplier).real
    elif name == "image":
        op = ImageBlur()
        x = rng.normal(size=(9, 11)).astype("float32")

        def oracle(x, transpose=False):
            return periodic_blur(x)
    elif name == "dense":
        op = DenseSolve()
        a = rng.normal(size=(9, 9)).astype("float32") / 20 + np.eye(9, dtype="float32")
        fixed = (a,)
        x = rng.normal(size=(9, 3)).astype("float32")

        def oracle(x, transpose=False):
            return np.linalg.solve(a.T if transpose else a, x)
    else:
        op = SparseDiffusion()
        fixed, a = sparse_inputs()
        x = rng.normal(size=11).astype("float32")

        def oracle(x, transpose=False):
            return (a.T if transpose else a) @ x

    arrays = tuple(jnp.asarray(a) for a in (*fixed, x))

    def f(x):
        return op(*arrays[:-1], x)

    compiled = jax.jit(op)
    with jax.transfer_guard_device_to_host("disallow"):
        value = compiled(*arrays).block_until_ready()
    assert op.observed["inputs"] == tuple(a.unsafe_buffer_pointer() for a in arrays)
    assert op.observed["output"] == value.unsafe_buffer_pointer()
    assert op.observed["stream"] != 0
    np.testing.assert_allclose(value, oracle(x), rtol=2e-5, atol=2e-6)
    direction = rng.normal(size=x.shape).astype("float32")
    weights = rng.normal(size=x.shape).astype("float32")
    tangent = jax.jit(lambda x, d: jax.jvp(f, (x,), (d,))[1])(
        arrays[-1], jnp.asarray(direction)
    )
    cotangent = jax.jit(lambda x, w: jax.vjp(f, x)[1](w)[0])(
        arrays[-1], jnp.asarray(weights)
    )
    np.testing.assert_allclose(tangent, oracle(direction), rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(cotangent, oracle(weights, True), rtol=3e-5, atol=3e-6)
    hessian_vector = jax.jit(
        lambda x, d: jax.jvp(jax.grad(lambda z: jnp.sum(f(z) ** 2)), (x,), (d,))[1]
    )(arrays[-1], jnp.asarray(direction))
    np.testing.assert_allclose(
        hessian_vector, 2 * oracle(oracle(direction), True), rtol=4e-5, atol=4e-6
    )
    stacked = jnp.stack((arrays[-1], 2 * arrays[-1]))
    mapped = jax.jit(jax.vmap(f))(stacked)
    np.testing.assert_allclose(
        mapped, np.stack((oracle(x), 2 * oracle(x))), rtol=3e-5, atol=3e-6
    )


def test_spectral_pde_eigenmode(gpu):
    h, w, ky, kx = 16, 20, 3, 2
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    field = np.cos(2 * np.pi * (ky * y / h + kx * x / w)).astype("float32")
    eigenvalue = 1 + 4 * np.sin(np.pi * ky / h) ** 2 + 4 * np.sin(np.pi * kx / w) ** 2
    actual = jax.jit(SpectralPoisson())(jnp.asarray(eigenvalue * field))
    np.testing.assert_allclose(actual, field, rtol=2e-5, atol=2e-6)


def test_native_reaction_derivatives_against_analytic_jacobian(gpu):
    rng = np.random.default_rng(902)
    x = rng.uniform(-0.5, 0.5, (4, 5)).astype("float32")
    direction = rng.normal(size=x.shape).astype("float32")
    op = ReactionDiffusion()
    n = x.size
    jacobian = np.zeros((n, n), dtype="float64")
    for r in range(x.shape[0]):
        for c in range(x.shape[1]):
            i = r * x.shape[1] + c
            jacobian[i, i] = 0.5 + 0.01 * (1 - 3 * float(x[r, c]) ** 2)
            for rr, cc in (
                ((r - 1) % 4, c),
                ((r + 1) % 4, c),
                (r, (c - 1) % 5),
                (r, (c + 1) % 5),
            ):
                jacobian[i, rr * 5 + cc] += 0.125
    actual_jacobian = jax.jit(jax.jacfwd(op))(jnp.asarray(x))
    np.testing.assert_allclose(
        np.asarray(actual_jacobian).reshape(n, n), jacobian, rtol=2e-6, atol=1e-7
    )
    tangent = jax.jit(lambda x, d: jax.jvp(op, (x,), (d,))[1])(x, direction)
    np.testing.assert_allclose(
        np.asarray(tangent).ravel(), jacobian @ direction.ravel(), rtol=2e-6, atol=2e-7
    )
    hessian_product = jax.jit(
        lambda x, d: jax.jvp(jax.grad(lambda x: op(x).sum()), (x,), (d,))[1]
    )(x, direction)
    np.testing.assert_allclose(
        hessian_product, -0.06 * x * direction, rtol=2e-6, atol=1e-7
    )


def test_nufft_application_dense_sum_at_high_modes(gpu):
    from applications.workloads import build

    workload = build("nufft_synthesis", "small")
    args = tuple(jnp.asarray(x) for x in workload.host_args)
    assert args[0].dtype == np.float64 and args[1].dtype == np.complex128
    with jax.transfer_guard_device_to_host("disallow"):
        value = jax.jit(workload.bound)(*args).block_until_ready()
    expected = workload.reference(*workload.host_args)
    np.testing.assert_allclose(
        np.asarray(value)[workload.comparison_indices],
        expected,
        rtol=workload.rtol,
        atol=workload.atol,
    )


def test_torch_convolution_gpu_storage_and_both_derivatives(gpu):
    pytest.importorskip("torch")
    rng = np.random.default_rng(314)
    x = jnp.asarray(rng.normal(size=(2, 3, 7, 8)).astype("float32"))
    weights = jnp.asarray(rng.normal(size=(4, 3, 3, 3)).astype("float32") / 5)
    op = TorchConvolution(x.shape, weights.shape)
    with jax.transfer_guard_device_to_host("disallow"):
        value = jax.jit(op)(x, weights).block_until_ready()
    assert op.observed["inputs"] == (
        x.unsafe_buffer_pointer(),
        weights.unsafe_buffer_pointer(),
    )
    assert op.observed["output"] == value.unsafe_buffer_pointer()
    assert op.observed["stream"] != 0
    np.testing.assert_allclose(value, torch_reference(x, weights), rtol=2e-5, atol=2e-6)

    def loss(f, x, w):
        return jnp.sum(f(x, w) ** 2)

    actual = jax.jit(jax.grad(lambda x, w: loss(op, x, w), argnums=(0, 1)))(x, weights)
    expected = jax.jit(
        jax.grad(lambda x, w: loss(torch_reference, x, w), argnums=(0, 1))
    )(x, weights)
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(a, e, rtol=4e-5, atol=3e-5)

    # Differentiating the bilinear adjoints exercises higher derivatives through
    # external Torch operations rather than substituting JAX convolution kernels.
    def mixed(f):
        return jax.jvp(
            jax.grad(lambda w: jnp.sum(f(x, w) ** 2)),
            (weights,),
            (jnp.ones_like(weights),),
        )[1]

    np.testing.assert_allclose(
        jax.jit(lambda: mixed(op))(),
        jax.jit(lambda: mixed(torch_reference))(),
        rtol=4e-5,
        atol=2e-4,
    )
    stacked = jnp.stack((x, 2 * x))
    mapped = jax.jit(jax.vmap(lambda x: op(x, weights)))(stacked)
    np.testing.assert_allclose(
        mapped,
        jax.vmap(torch_reference, in_axes=(0, None))(stacked, weights),
        rtol=3e-5,
        atol=3e-6,
    )
