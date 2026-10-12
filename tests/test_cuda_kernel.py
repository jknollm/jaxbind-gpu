"""GPU kernel launches borrow XLA buffers without Python execution or scratch."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxbind_gpu as jb


@pytest.mark.parametrize("count", [3, 9, 33])
def test_native_argument_storage_scales_beyond_inline_capacity(gpu, count):
    parameters = [f"const float* x{i}" for i in range(count)] + [
        f"float* y{i}" for i in range(count)
    ]
    body = " ".join(f"y{i}[0] = x{i}[0] + 1.f;" for i in range(count))
    kernel = gpu.RawKernel(
        'extern "C" __global__ void many(' + ",".join(parameters) + "){" + body + "}",
        "many",
    )
    specs = tuple(jax.ShapeDtypeStruct((1,), np.float32) for _ in range(count))
    op = jb.bind(kernel, output=specs, launch=((1,), (1,)))
    inputs = tuple(jnp.array([i], dtype=jnp.float32) for i in range(count))
    values = jax.jit(op)(*inputs)
    for i, value in enumerate(values):
        np.testing.assert_array_equal(value, [i + 1])


def test_raw_kernel_binding_values_derivatives_and_native_hlo(gpu):
    kernel = gpu.RawKernel(
        r"""
        extern "C" __global__ void scale(const float* x, float* y, int n, float a) {
          int i = blockIdx.x * blockDim.x + threadIdx.x;
          if (i < n) y[i] = a * x[i];
        }
        """,
        "scale",
    )
    op = jb.bind(
        kernel,
        output=lambda x, **kw: jax.ShapeDtypeStruct(x.shape, x.dtype),
        launch=(lambda x, **kw: ((x.size + 127) // 128,), (128,)),
        scalars=lambda x, *, factor: (np.int32(x.size), np.float32(factor)),
        jvp=lambda p, t, *, factor: factor * t[0],
    )
    x = jnp.arange(4099, dtype=jnp.float32)
    lowered = jax.jit(lambda x: op(x, factor=2.5)).lower(x)
    assert "jaxbind_gpu_pycall" not in lowered.as_text()
    with jax.transfer_guard_device_to_host("disallow"):
        value = lowered.compile()(x).block_until_ready()
    np.testing.assert_array_equal(value, np.arange(4099) * 2.5)
    np.testing.assert_array_equal(
        jax.jit(jax.grad(lambda x: op(x, factor=2.5).sum()))(x), np.full(4099, 2.5)
    )
    np.testing.assert_array_equal(
        jax.jit(jax.vmap(lambda x: op(x, factor=2.5)))(jnp.stack((x, 2 * x))),
        np.stack((np.arange(4099) * 2.5, np.arange(4099) * 5)),
    )


def test_kernel_argument_order_multiple_outputs_and_scalar_validation(gpu):
    kernel = gpu.RawKernel(
        r"""
        extern "C" __global__ void pair(float* sum, const float* x, int n,
                                        float* product, const float* y) {
          int i = blockIdx.x * blockDim.x + threadIdx.x;
          if (i < n) { sum[i] = x[i] + y[i]; product[i] = x[i] * y[i]; }
        }
        """,
        "pair",
    )
    spec = jax.ShapeDtypeStruct((513,), np.float32)
    function = jb.NativeFunction.from_cuda_kernel(
        kernel,
        grid=(5,),
        block=(128,),
        scalars=(np.int32(513),),
        argument_order=(
            ("output", 0),
            ("input", 0),
            ("scalar", 0),
            ("output", 1),
            ("input", 1),
        ),
    )
    op = jb.bind(function, output=(spec, spec))
    x, y = jnp.arange(513, dtype=jnp.float32), jnp.full(513, 3, dtype=jnp.float32)
    actual = jax.jit(op)(x, y)
    np.testing.assert_array_equal(actual[0], np.arange(513) + 3)
    np.testing.assert_array_equal(actual[1], np.arange(513) * 3)
    bad = jb.NativeFunction.from_cuda_kernel(kernel, grid=5, block=128, scalars=(513,))
    with pytest.raises(TypeError, match="NumPy scalars"):
        jax.jit(jb.bind(bad, output=(spec, spec)))(x, y)


def test_kernel_shape_scalars_and_shared_memory(gpu):
    kernel = gpu.RawKernel(
        r"""
        extern "C" __global__ void reverse(const float* x, float* y, int n) {
          extern __shared__ float work[];
          int i = threadIdx.x;
          work[i] = i < n ? x[i] : 0;
          __syncthreads();
          if (i < n) y[i] = work[n - 1 - i];
        }
        """,
        "reverse",
    )
    op = jb.bind(
        kernel,
        output=lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
        launch=((1,), (128,)),
        scalars=lambda x: (np.int32(x.size),),
        shared_mem=512,
    )
    for n in (5, 100):
        x = jnp.arange(n, dtype=jnp.float32)
        np.testing.assert_array_equal(jax.jit(op)(x), np.arange(n)[::-1])
