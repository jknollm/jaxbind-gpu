"""Bind Warp's block-sparse product with shared buffers and a linear transpose."""

import cupy as cp
import jax
import numpy as np
import warp as wp
from warp import sparse

from jaxbind_gpu import bind


def multiply(matrix, x, *, out, transpose=False):
    """Run the unchanged Warp routine on the caller's CUDA stream."""
    stream = wp.Stream(matrix.device, cuda_stream=cp.cuda.get_current_stream().ptr)
    # The binding already orders inputs and outputs on this stream.
    with wp.ScopedStream(stream, sync_enter=False, sync_exit=False):
        sparse.bsr_mv(
            matrix, wp.from_dlpack(x), wp.from_dlpack(out), transpose=transpose
        )


def make_matvec(matrix, *, graph_compatible=False):
    """Keep a fixed CUDA matrix resident; differentiate its vector operand.

    Matrix construction and kernel compilation must finish before graph capture.
    Do not mutate the matrix while a compiled JAX operation can reference it.
    Warp's transpose product uses atomic sums and can vary in its last bits.
    """
    dtype = np.dtype(wp.dtype_to_numpy(matrix.scalar_type))

    def forward(x, *, out):
        multiply(matrix, x, out=out)

    def transposed(x, *, out):
        multiply(matrix, x, out=out, transpose=True)

    def output(x):
        rows = matrix.nrow * matrix.block_shape[0]
        shape = (rows,) if x.ndim == 1 else (matrix.nrow, matrix.block_shape[0])
        return jax.ShapeDtypeStruct(shape, dtype)

    return bind(
        forward,
        output=output,
        transpose=transposed,
        out=True,
        graph_compatible=graph_compatible,
    )


def main():
    wp.init()
    host_block = np.array([[2, -1, 0], [0.5, 3, 1], [-2, 0, 4]], dtype="float32")
    matrix = sparse.bsr_from_triplets(
        1,
        1,
        wp.array([0], dtype=wp.int32, device="cuda:0"),
        wp.array([0], dtype=wp.int32, device="cuda:0"),
        wp.array(host_block[None], dtype=wp.mat33, device="cuda:0"),
    )
    wp.synchronize_device(matrix.device)
    matvec = make_matvec(matrix)
    host = np.array([[1, -2, 0.5]], dtype="float32")
    external = cp.asarray(host)
    x = jax.dlpack.from_dlpack(external, copy=False)
    assert x.unsafe_buffer_pointer() == external.data.ptr
    with jax.transfer_guard_device_to_host("disallow"):
        value = jax.jit(matvec)(x).block_until_ready()
        gradient = jax.jit(jax.grad(lambda z: matvec(z).sum()))(x)
        gradient.block_until_ready()
    np.testing.assert_allclose(value, (host_block @ host[0])[None], atol=1e-6)
    np.testing.assert_allclose(gradient, host_block.sum(axis=0)[None], atol=1e-6)
    print("Warp sparse product: value, transpose and shared GPU input passed")


if __name__ == "__main__":
    main()
