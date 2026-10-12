"""Native return-value baselines without adapter conversions or result copies.

The adapter baseline remains available with identical output copies. These
variants measure the library's fresh result without that adapter copy. Torch
inputs and its external stream wrapper are prepared once, as in native usage.
"""


def returning_baseline(workload, argument_sets, context):
    import cupy as cp

    name = workload.name
    if name == "warp_sparse":
        import warp as wp
        from warp import sparse

        matrix = workload.operator.matrix
        stream = wp.Stream(matrix.device, cuda_stream=context.stream.ptr)
        # Prepare native views once. The native baseline need not pay for
        # per-call interoperability wrappers used by the binding callback.
        with context.stream, wp.ScopedStream(stream, sync_enter=False):
            argument_sets = [
                tuple(wp.from_dlpack(a) for a in args) for args in argument_sets
            ]

        def function(x):
            with wp.ScopedStream(stream, sync_enter=False):
                return sparse.bsr_mv(matrix, x)

    elif name == "sort":
        function = cp.sort
    elif name == "dense_solve":
        function = cp.linalg.solve
    elif name == "sparse_diffusion":
        from cupyx.scipy.sparse import csr_matrix

        def function(data, indices, indptr, x):
            for _ in range(workload.steps):
                matrix = csr_matrix(
                    (data, indices, indptr), shape=(x.size, x.size), copy=False
                )
                matrix.has_sorted_indices = True
                matrix.has_canonical_format = True
                x = matrix.dot(x)
            return x

    elif name == "torch_convolution":
        torch = workload.operator.torch
        argument_sets = [
            tuple(torch.utils.dlpack.from_dlpack(a) for a in args)
            for args in argument_sets
        ]
        stream = torch.cuda.ExternalStream(context.stream.ptr, device=context.device)

        def function(x, weights):
            with (
                torch.cuda.device(context.device),
                torch.cuda.stream(stream),
                torch.no_grad(),
            ):
                return torch.nn.functional.conv2d(x, weights, padding=1)

    else:
        return None

    def returning(*args):
        with context.stream:
            return function(*args)

    return returning, argument_sets
