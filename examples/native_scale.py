"""Compile and bind a C ABI wrapper, with no Python on the execution path."""

import ctypes
from pathlib import Path
import struct
import subprocess
import tempfile

import jax
from jaxbind_gpu import bind
from jaxbind_gpu.advanced import NativeFunction, include_dir


def binding():
    import cupy as cp

    kernel = cp.RawKernel(
        r"""
    extern "C" __global__ void scale(const float* x, float* out, long long n, float a) {
        long long i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) out[i] = x[i] * a;
    }""",
        "scale",
    )
    kernel.compile()
    with tempfile.TemporaryDirectory(prefix="jaxbind-gpu-native-") as directory:
        library_path = Path(directory) / "scale.so"
        subprocess.run(
            [
                "c++",
                "-O3",
                "-shared",
                "-fPIC",
                "-std=c++17",
                "-I" + include_dir(),
                str(Path(__file__).with_suffix(".cc")),
                "-ldl",
                "-o",
                str(library_path),
            ],
            check=True,
        )
        library = ctypes.CDLL(str(library_path))
    function = NativeFunction.from_cabi(
        library.scale_buffer,
        encode=lambda factor: struct.pack("=Qf", kernel.kernel.ptr, factor),
        owner=(library, kernel),
        graph_compatible=True,
    )

    return bind(function, transpose=function), kernel


if __name__ == "__main__":
    import jax.numpy as jnp
    import numpy as np

    jax.devices()
    op, _ = binding()
    x = jnp.arange(16, dtype=jnp.float32)
    with jax.transfer_guard_device_to_host("disallow"):
        y = jax.jit(lambda x: op(x, factor=3.0))(x).block_until_ready()
        gradient = jax.jit(jax.grad(lambda x: op(x, factor=3.0).sum()))(x)
        gradient.block_until_ready()
    np.testing.assert_array_equal(y, np.arange(16) * 3.0)
    np.testing.assert_array_equal(gradient, np.full(16, 3.0))
    print("Native CUDA scale and gradient passed")
