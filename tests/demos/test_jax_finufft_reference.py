
"""Compare our CUDA adapter with jax-finufft's separately compiled CPU path.

PyPI provides CPU wheels for jax-finufft. Using that build as the reference also
compares independent CPU and CUDA NUFFT implementations. No CPU fallback is used
by jaxbind_gpu itself.
"""

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

if importlib.util.find_spec("cufinufft") is None:
    pytest.skip("Install the optional demo dependencies", allow_module_level=True)


from examples.cufinufft_demo import nufft1, nufft2

jf = pytest.importorskip("jax_finufft")
Opts = pytest.importorskip("jax_finufft.options").Opts


@pytest.mark.parametrize("shape", [(7,), (5, 4), (3, 4, 5)])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("sign,modeord", [(-1, 0), (1, 1)])
def test_values_jvp_and_vjp_match_jax_finufft(gpu, shape, dtype, sign, modeord):
    rng = np.random.default_rng(123)
    real_dtype = np.float32 if dtype == np.complex64 else np.float64
    points = rng.uniform(-np.pi, np.pi, (len(shape), 9)).astype(real_dtype)
    c = (rng.normal(size=(2, 3, 9)) + 1j * rng.normal(size=(2, 3, 9))).astype(dtype)
    f = (rng.normal(size=(2, 3) + shape) + 1j * rng.normal(size=(2, 3) + shape)).astype(
        dtype
    )
    eps = 1e-5 if dtype == np.complex64 else 1e-11
    tol = 4e-4 if dtype == np.complex64 else 4e-9

    with jax.default_device(jax.devices("cpu")[0]):
        # jax-finufft requires explicit singleton axes when coordinates
        # broadcast over more than one leading transform dimension.
        coords_cpu = tuple(jnp.asarray(p)[None, None, :] for p in points)
        c_cpu, f_cpu = jnp.asarray(c), jnp.asarray(f)
        opts = Opts(modeord=modeord, nthreads=2)

        def reference1(data):
            return jf.nufft1(shape, data, *coords_cpu, iflag=sign, eps=eps, opts=opts)

        def reference2(data):
            return jf.nufft2(data, *coords_cpu, iflag=sign, eps=eps, opts=opts)

        value1 = np.asarray(reference1(c_cpu))
        value2 = np.asarray(reference2(f_cpu))
        _, ref_jvp1 = jax.jvp(reference1, (c_cpu,), (c_cpu,))
        _, ref_jvp2 = jax.jvp(reference2, (f_cpu,), (f_cpu,))
        _, vjp1 = jax.vjp(reference1, c_cpu)
        _, vjp2 = jax.vjp(reference2, f_cpu)
        ref_vjp1, ref_vjp2 = np.asarray(vjp1(f_cpu)[0]), np.asarray(vjp2(c_cpu)[0])
        ref_jvp1, ref_jvp2 = np.asarray(ref_jvp1), np.asarray(ref_jvp2)

    coords_gpu = tuple(jnp.asarray(p) for p in points)
    c_gpu, f_gpu = jnp.asarray(c), jnp.asarray(f)

    def actual1(data):
        return nufft1(coords_gpu, data, shape, isign=sign, eps=eps, modeord=modeord)

    def actual2(data):
        return nufft2(coords_gpu, data, isign=sign, eps=eps, modeord=modeord)

    np.testing.assert_allclose(jax.jit(actual1)(c_gpu), value1, atol=tol, rtol=tol)
    np.testing.assert_allclose(jax.jit(actual2)(f_gpu), value2, atol=tol, rtol=tol)
    _, actual_jvp1 = jax.jvp(actual1, (c_gpu,), (c_gpu,))
    _, actual_jvp2 = jax.jvp(actual2, (f_gpu,), (f_gpu,))
    np.testing.assert_allclose(actual_jvp1, ref_jvp1, atol=tol, rtol=tol)
    np.testing.assert_allclose(actual_jvp2, ref_jvp2, atol=tol, rtol=tol)
    _, actual_vjp1 = jax.vjp(actual1, c_gpu)
    _, actual_vjp2 = jax.vjp(actual2, f_gpu)
    np.testing.assert_allclose(actual_vjp1(f_gpu)[0], ref_vjp1, atol=tol, rtol=tol)
    np.testing.assert_allclose(actual_vjp2(c_gpu)[0], ref_vjp2, atol=tol, rtol=tol)
