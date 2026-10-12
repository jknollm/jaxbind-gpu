"""Lesson 5: bind a CUDA kernel without a Python callback at execution time."""

import cupy as cp
import jax
import jax.numpy as jnp
import numpy as np

from jaxbind_gpu import bind


kernel = cp.RawKernel(
    r"""
    extern "C" __global__ void square(const float* x, float* out, long long n) {
        long long i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) out[i] = x[i] * x[i];
    }
    """,
    "square",
)


def main():
    # Kernel arguments default to input pointers, output pointers, then scalars.
    # Shape-dependent launch parameters are evaluated while JAX traces the call.
    square = bind(
        kernel,
        launch=(lambda x: ((x.size + 255) // 256,), (256,)),
        scalars=lambda x: (np.int64(x.size),),
        jvp=lambda p, t: 2 * p[0] * t[0],
    )
    x = jnp.arange(513, dtype=jnp.float32)
    with jax.transfer_guard_device_to_host("disallow"):
        value = jax.jit(square)(x).block_until_ready()
        gradient = jax.jit(jax.grad(lambda x: square(x).sum()))(x).block_until_ready()
    np.testing.assert_array_equal(value, np.arange(513, dtype="float32") ** 2)
    np.testing.assert_array_equal(gradient, 2 * np.arange(513, dtype="float32"))
    print("Native CUDA kernel: direct output and derivative passed")


if __name__ == "__main__":
    main()
