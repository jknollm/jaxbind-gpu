"""Describe a CUDA launch at tracing time; dispatch it through the native ABI."""

import operator
import struct

import numpy as np

from ._native import NativeFunction


def _dimensions(value, name):
    if isinstance(value, int):
        value = (value,)
    value = tuple(operator.index(v) for v in value)
    if not 1 <= len(value) <= 3 or any(v < 1 or v >= 2**32 for v in value):
        raise ValueError(f"{name} must contain one to three positive uint32 dimensions")
    return value + (1,) * (3 - len(value))


class CudaKernel(NativeFunction):
    def __init__(
        self, kernel, *, grid, block, scalars, argument_order, shared_mem, owner, device
    ):
        from . import _bridge

        self.block = _dimensions(block, "block")
        if not callable(grid):
            grid = _dimensions(grid, "grid")
        shared_mem = operator.index(shared_mem)
        if not 0 <= shared_mem < 2**32:
            raise ValueError("shared_mem must be a nonnegative uint32 byte count")
        self.grid, self.scalars = grid, scalars
        self.argument_order, self.shared_mem = argument_order, shared_mem
        if isinstance(kernel, int):
            if owner is None or device is None or kernel <= 0:
                raise ValueError(
                    "A CUfunction handle requires an owner and device ordinal"
                )
            handle = kernel
        else:
            import cupy as cp

            if not isinstance(kernel, cp.RawKernel):
                raise TypeError(
                    "Provide a CuPy RawKernel or an owned integer CUfunction handle"
                )
            current = cp.cuda.Device().id
            if device is not None and device != current:
                raise ValueError("Compile the RawKernel with its CUDA device current")
            device = current
            kernel.compile()
            function = kernel.kernel
            handle, owner = function.ptr, (kernel, function, owner)
        self.handle, self.device = int(handle), operator.index(device)
        self.native = NativeFunction.from_cabi(
            _bridge.kernel_address(),
            encode=lambda **kw: kw["config"],
            owner=(owner, kernel),
            graph_compatible=True,
        )
        self.has_side_effect = False
        self.input_output_aliases = None

    def _lower(self, args, outputs, kwargs, **options):
        grid = _dimensions(
            self.grid(*args, **kwargs) if callable(self.grid) else self.grid, "grid"
        )
        scalars = tuple(
            self.scalars(*args, **kwargs) if callable(self.scalars) else self.scalars
        )
        if kwargs and not callable(self.scalars) and not callable(self.grid):
            raise TypeError(
                "Provide a scalar or grid function to consume static keywords"
            )
        blobs = []
        for value in scalars:
            if not isinstance(value, np.generic) or value.dtype.kind not in "biufc":
                raise TypeError("Kernel scalars require explicitly typed NumPy scalars")
            if not value.dtype.isnative:
                raise ValueError("Kernel scalars must use native byte order")
            blobs.append(value.tobytes())
        layout = self.argument_order
        if layout is None:
            layout = tuple(
                (kind, i)
                for kind, size in (
                    ("input", len(args)),
                    ("output", len(outputs)),
                    ("scalar", len(blobs)),
                )
                for i in range(size)
            )
        kinds, counts = (
            {"input": 0, "output": 1, "scalar": 2},
            {"input": len(args), "output": len(outputs), "scalar": len(blobs)},
        )
        payload, parameters = bytearray(), []
        start = 48 + 16 * len(layout)
        for kind, index in layout:
            index = operator.index(index)
            if kind not in kinds or not 0 <= index < counts[kind]:
                raise ValueError("Invalid kernel argument_order reference")
            offset, size = 0, 0
            if kind == "scalar":
                offset, size = start + len(payload), len(blobs[index])
                payload.extend(blobs[index])
            parameters.append(struct.pack("<4I", kinds[kind], index, offset, size))
        header = struct.pack(
            "<IiQ8I",
            1,
            self.device,
            self.handle,
            *grid,
            *self.block,
            self.shared_mem,
            len(layout),
        )
        return self.native._lower(
            args,
            outputs,
            {"config": header + b"".join(parameters) + payload},
            **options,
        )
