"""A fixed block-sparse Warp matrix, with an independent SciPy oracle."""

import jaxbind_gpu as jb
import numpy as np


class WarpMatvec:
    def __init__(self, matrix):
        self.matrix = matrix
        self.observed = None
        self.enable_graph(False)

    def enable_graph(self, enabled=True):
        self.op = jb.bind(
            self.forward, transpose=self.reverse, out=True, graph_compatible=enabled
        )

    def __call__(self, x):
        return self.op(x)

    def forward(self, x, *, out):
        self.observed = {
            "inputs": (x.data.ptr,),
            "output": out.data.ptr,
            "stream": jb.advanced.current_context().stream.ptr,
        }
        self.run(out, (x,), None)

    def reverse(self, x, *, out):
        self.run(out, (x,), None, transpose=True)

    def run(self, out, args, context, *, transpose=False):
        from examples.warp_sparse import multiply

        multiply(self.matrix, args[0], out=out, transpose=transpose)


def build(size, rng):
    import warp as wp
    from warp import sparse as ws
    from scipy.sparse import bsr_matrix
    from .workloads import Workload

    wp.init()
    n = 4096 if size == "small" else 1048576
    # Seven neighboring blocks, each coupling three components. All matrix
    # entries are fixed and resident; only the vector changes between calls.
    offsets = np.array([-257, -17, -1, 0, 1, 17, 257])
    columns = (np.arange(n)[:, None] + offsets) % n
    blocks = rng.normal(size=(len(offsets), 3, 3)).astype("float32") / 12
    blocks[3] += np.eye(3, dtype="float32")
    values = np.broadcast_to(blocks, (n, *blocks.shape)).copy()
    order = np.argsort(columns, axis=1)
    columns = np.take_along_axis(columns, order, axis=1).astype("int32")
    values = np.take_along_axis(values, order[:, :, None, None], axis=1)
    indptr = np.arange(0, n * 7 + 1, 7, dtype="int32")
    oracle_matrix = bsr_matrix(
        (values.reshape(-1, 3, 3).astype("float64"), columns.ravel(), indptr),
        shape=(3 * n, 3 * n),
    )
    matrix = ws.bsr_from_triplets(
        n,
        n,
        wp.array(np.repeat(np.arange(n, dtype="int32"), 7), device="cuda:0"),
        wp.array(columns.ravel(), device="cuda:0"),
        wp.array(values.reshape(-1, 3, 3), dtype=wp.mat33, device="cuda:0"),
    )
    wp.synchronize_device(matrix.device)
    x = rng.normal(size=(n, 3)).astype("float32")

    def reference(x):
        return (oracle_matrix @ x.astype("float64").ravel()).reshape(n, 3)

    return Workload(
        "warp_sparse",
        size,
        WarpMatvec(matrix),
        (x,),
        reference,
        backend="Warp block-sparse library through public bind",
    )
