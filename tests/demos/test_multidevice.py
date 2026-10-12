import importlib.util
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
import pytest

if importlib.util.find_spec("cufinufft") is None:
    pytest.skip("Install the optional demo dependencies", allow_module_level=True)


import jaxbind_gpu as jb
from examples import cufinufft_demo as nufft


def test_device_contexts_and_plan_ownership(gpu):
    devices = jax.devices("gpu")
    if len(devices) < 2:
        pytest.skip("Run with two idle CUDA devices")
    observed = []

    def scale(out, args, kwargs_dump):
        context = jb.current_context()
        assert context.device == args[0].device.id == out[0].device.id
        assert context.stream.ptr == gpu.cuda.get_current_stream().ptr
        observed.append(context.device)
        gpu.multiply(args[0], 3, out=out[0])

    operation = jax.jit(jb.get_call(scale, lambda x: ((x.shape, x.dtype),)))

    def run(device):
        with jax.default_device(device):
            points = jnp.asarray(np.linspace(-1.0, 1.0, 9))
            data = jnp.asarray(np.arange(9).astype(np.complex128))
            scaled = operation(jnp.arange(9.0))[0]
            np.testing.assert_array_equal(scaled, np.arange(9) * 3)
            assert scaled.device == device
            prepared = jax.jit(nufft.prepare_nufft1(points, 7, eps=1e-11))
            expected = (
                np.exp(1j * np.outer(np.arange(-3, 4), np.asarray(points))) @ data
            )
            for factor in (1.0, 2.0, -3.0):
                result = prepared(data * factor)
                assert result.device == device
                np.testing.assert_allclose(result, expected * factor, atol=1e-9)
            dynamic = jax.jit(lambda x, c: nufft.nufft1(x, c, 7, eps=1e-11))
            np.testing.assert_allclose(dynamic(points, data), expected, atol=1e-9)

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(run, devices[:2]))
    assert set(observed) == {0, 1}
    nufft.synchronize(clear_cache=True)
    assert nufft.runtime_stats()["native_cached"] == 0
