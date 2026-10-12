"""Write an explicit buffer callback with a linear transpose."""

import cupy as cp
import jax
import numpy as np

from jaxbind_gpu import advanced as jb


def scale(out, args, kwargs_dump):
    factor = jb.load_kwargs(kwargs_dump)["factor"]
    cp.multiply(args[0], factor, out=out[0])


def abstract(x, **kwargs):
    return ((x.shape, x.dtype),)


def main():
    # Scale is its own plain transpose, so one callback supplies both operations.
    op = jb.get_linear_call(scale, scale, abstract, abstract)
    external = cp.arange(8, dtype=cp.float32)
    x = jax.dlpack.from_dlpack(external, copy=False)
    assert x.unsafe_buffer_pointer() == external.data.ptr
    with jax.transfer_guard_device_to_host("disallow"):
        value = jax.jit(lambda x: op(x, factor=3.0)[0])(x).block_until_ready()
        gradient = jax.jit(jax.grad(lambda x: op(x, factor=3.0)[0].sum()))(x)
        gradient.block_until_ready()
    np.testing.assert_array_equal(value, 3 * np.arange(8))
    np.testing.assert_array_equal(gradient, np.full(8, 3))
    print("Buffer callback: value, gradient, and shared input pointer passed")


if __name__ == "__main__":
    main()
