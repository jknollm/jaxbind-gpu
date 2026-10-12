"""Bind cuSOLVER with a GPU-resident matrix fixed for differentiation."""

import cupy as cp
import jax
import jax.numpy as jnp
import numpy as np

from jaxbind_gpu import advanced as jb


def solve(out, args, kwargs_dump):
    matrix, rhs = args
    value = cp.linalg.solve(matrix, rhs)
    cp.copyto(out[0], value)
    jb.current_context().retain(value)


def transpose(out, args, kwargs_dump):
    matrix, cotangent = args
    # JAX uses the plain transpose, including for complex arrays.
    value = cp.linalg.solve(matrix.T, cotangent)
    cp.copyto(out[0], value)
    jb.current_context().retain(value)


def abstract(matrix, rhs):
    return ((rhs.shape, rhs.dtype),)


def main():
    # The matrix is a dynamic GPU argument, but this binding differentiates rhs.
    # To differentiate the matrix too, supply the appropriate nonlinear JVP.
    op = jb.get_linear_call(solve, transpose, abstract, abstract, first_n_args_fixed=1)
    host_matrix = np.array([[3.0, 1.0], [0.5, 2.0]], dtype="float32")
    host_rhs = np.array([2.0, -1.0], dtype="float32")
    matrix, rhs = jnp.asarray(host_matrix), jnp.asarray(host_rhs)
    with jax.transfer_guard_device_to_host("disallow"):
        value = jax.jit(lambda a, b: op(a, b)[0])(matrix, rhs).block_until_ready()
        gradient = jax.jit(jax.grad(lambda b, a: op(a, b)[0].sum()))(rhs, matrix)
        gradient.block_until_ready()
    np.testing.assert_allclose(value, np.linalg.solve(host_matrix, host_rhs), atol=1e-7)
    np.testing.assert_allclose(
        gradient, np.linalg.solve(host_matrix.T, np.ones(2)), atol=1e-7
    )
    print("Linear solve: NumPy reference and fixed-matrix transpose passed")


if __name__ == "__main__":
    main()
