import os

# Test execution is opt-in on CUDA. CPU runs still check tracing and validation.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import jax
import pytest

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def gpu():
    if not any(device.platform == "gpu" for device in jax.devices()):
        pytest.skip("A JAX CUDA device is required")
    import cupy

    # Native provider cleanup can leave another CUDA device current. Each
    # example starts on JAX's default GPU and opts into other devices explicitly.
    with cupy.cuda.Device(0):
        yield cupy
