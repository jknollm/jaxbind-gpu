"""Lesson 10: bind a return-only library function without an output copy."""

import cupy as cp
import jax
import jax.numpy as jnp
import numpy as np

from jaxbind_gpu import bind


def main():
    # CuPy sort has no out parameter. Its result allocation can borrow XLA's
    # output. The function must not keep borrowed allocations after returning.
    sort = jax.jit(bind(cp.sort, out="allocate"))
    host = np.array([3, -1, 2, 7, 0], dtype=np.float32)
    with jax.transfer_guard_device_to_host("disallow"):
        value = sort(jnp.asarray(host)).block_until_ready()
    np.testing.assert_array_equal(value, np.sort(host))
    print("Return-only CuPy function: sorted GPU output")


if __name__ == "__main__":
    main()
