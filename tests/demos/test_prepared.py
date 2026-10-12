import importlib.util
import jax
import jax.numpy as jnp
import numpy as np
import pytest

if importlib.util.find_spec("cufinufft") is None:
    pytest.skip("Install the optional demo dependencies", allow_module_level=True)


from examples import cufinufft_demo as nufft


@pytest.mark.parametrize("kind", [1, 2, 3])
def test_prepared_geometry_execution_and_coefficients_gradient(gpu, kind):
    x = jnp.linspace(-1.0, 1.0, 9)
    s = jnp.linspace(-2.0, 2.0, 7)
    c = jnp.arange(9.0, dtype=jnp.float64).astype(jnp.complex128)
    if kind == 1:
        prepared = nufft.prepare_nufft1(x, 7, eps=1e-11)

        def dynamic(c):
            return nufft.nufft1(x, c, 7, eps=1e-11)
    elif kind == 2:
        prepared = nufft.prepare_nufft2(x, eps=1e-11)

        def dynamic(c):
            return nufft.nufft2(x, c, eps=1e-11)
    else:
        prepared = nufft.prepare_nufft3(x, s, eps=1e-11)

        def dynamic(c):
            return nufft.nufft3(x, s, c, eps=1e-11)

    compiled = jax.jit(prepared)
    for factor in (1.0, 2.0, -3.0):
        np.testing.assert_allclose(compiled(c * factor), dynamic(c * factor), atol=1e-9)

    def loss(operation):
        return lambda c: jnp.sum(jnp.abs(operation(c)) ** 2)

    actual = jax.jit(jax.grad(loss(prepared)))(c)
    expected = jax.jit(jax.grad(loss(dynamic)))(c)
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-9)
    batch = jnp.stack((c, c * 2, -c))
    np.testing.assert_allclose(
        jax.jit(jax.vmap(prepared))(batch), jax.vmap(dynamic)(batch), atol=1e-9
    )
    nufft.synchronize(clear_cache=True)
    np.testing.assert_allclose(compiled(c), dynamic(c), atol=1e-9)


def test_prepare_copies_geometry_and_rejects_tracers(gpu):
    x = np.linspace(-1.0, 1.0, 5)
    prepared = nufft.prepare_nufft1(x, 7, eps=1e-11)
    original = x.copy()
    x[:] = 0
    c = jnp.ones(5, dtype=jnp.complex128)
    np.testing.assert_allclose(
        jax.jit(prepared)(c),
        nufft.nufft1(jnp.asarray(original), c, 7, eps=1e-11),
        atol=1e-9,
    )
    with pytest.raises(ValueError, match="outside"):
        jax.jit(lambda x: nufft.prepare_nufft1(x, 7)(c))(jnp.asarray(x))


def test_prepared_gpu_constants_do_not_transfer_to_host(gpu):
    import os
    import subprocess
    import sys

    code = """
import jax, jax.numpy as jnp, numpy as np
from examples import cufinufft_demo as nufft
jax.config.update('jax_enable_x64', True)
points = jnp.linspace(-1.0, 1.0, 9)
data = jnp.ones(9, dtype=jnp.complex128)
with jax.transfer_guard_device_to_host('disallow'):
    prepared = nufft.prepare_nufft1(points, 7, eps=1e-11)
    result = jax.jit(prepared)(data)
    result.block_until_ready()
expected = np.exp(1j * np.outer(np.arange(-3, 4), np.asarray(points))).sum(1)
np.testing.assert_allclose(result, expected, atol=1e-9)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        env=dict(os.environ, JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS="true"),
        check=True,
        capture_output=True,
        timeout=60,
    )


@pytest.mark.parametrize("cache_limit", [0, 1])
def test_native_plan_eviction_and_disabled_cache(gpu, cache_limit):
    nufft.configure_runtime(max_cache_entries=cache_limit)
    try:
        x = jnp.linspace(-1.0, 1.0, 9)
        c = jnp.arange(9.0).astype(jnp.complex128)
        operations = [
            jax.jit(nufft.prepare_nufft1(x + offset, 7, eps=1e-11))
            for offset in (0.0, 0.3, -0.4)
        ]
        modes = np.arange(-3, 4)
        for index in (0, 1, 2, 0):
            offset = (0.0, 0.3, -0.4)[index]
            expected = np.exp(1j * np.outer(modes, np.asarray(x) + offset)) @ c
            np.testing.assert_allclose(operations[index](c), expected, atol=1e-9)
            assert nufft.runtime_stats()["native_cached"] <= cache_limit
        nufft.synchronize(clear_cache=True)
        assert nufft.runtime_stats()["native_cached"] == 0
    finally:
        nufft.configure_runtime()


@pytest.mark.parametrize("backend", ["native", "python"])
def test_concurrent_coordinate_updates(gpu, backend):
    from concurrent.futures import ThreadPoolExecutor

    x = jnp.linspace(-1.0, 1.0, 9)
    c = jnp.arange(9.0).astype(jnp.complex128)
    operation = jax.jit(lambda x, c: nufft.nufft1(x, c, 7, eps=1e-11, backend=backend))
    operation(x, c).block_until_ready()
    modes = np.arange(-3, 4)
    inputs = [(x + i * 0.07, c * (i + 1)) for i in range(12)]
    for points, data in inputs:
        points.block_until_ready()
        data.block_until_ready()

    def evaluate(pair):
        points, data = pair
        return np.asarray(operation(points, data))

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(evaluate, inputs))
    for (points, data), actual in zip(inputs, results):
        expected = np.exp(1j * np.outer(modes, np.asarray(points))) @ data
        np.testing.assert_allclose(actual, expected, atol=1e-8)
    nufft.synchronize(clear_cache=True)
