"""Native GPU operations executed directly by XLA, without Python callbacks."""

from pathlib import Path
import ctypes
import uuid

import jax
import numpy as np


_owners = {}


class NativeFunction:
    """A registered typed XLA FFI operation.

    The target capsule must contain a typed FFI handler accepting device buffers
    and a platform stream. Its owner is retained for the process lifetime.
    ``attributes`` converts binding keywords to typed FFI attributes at tracing
    time. A stable target name and portable attributes permit cross-process
    executable reuse when every process registers the same target.
    """

    def __init__(
        self,
        name,
        target,
        *,
        attributes=None,
        owner=None,
        has_side_effect=False,
        input_output_aliases=None,
    ):
        if not isinstance(name, str) or not name:
            raise ValueError("Provide a nonempty, globally unique FFI target name")
        if name in _owners:
            raise ValueError(f"The native target {name!r} is already registered")
        jax.ffi.register_ffi_target(name, target, platform="CUDA")
        _owners[name] = (target, owner)
        self.name = name
        self.attributes = attributes if attributes is not None else lambda **kw: kw
        self.has_side_effect = has_side_effect
        self.input_output_aliases = input_output_aliases

    def __call__(self, *args, **kwargs):
        raise RuntimeError("NativeFunction must be executed through a GPU binding")

    @classmethod
    def from_cabi(
        cls,
        function,
        *,
        encode=None,
        owner=None,
        graph_compatible=False,
        has_side_effect=False,
        input_output_aliases=None,
    ):
        """Bind a native ``jb_gpu_operation`` from the standalone header.

        ``function`` is a ctypes function, a function capsule, or an integer
        address with an explicit owner. ``encode(**kwargs)`` returns static
        configuration bytes. The function and owner remain alive for the
        process lifetime. Execution acquires no Python GIL and requires no
        CuPy. This form is process-local; use a named typed FFI target when
        portable executable serialization is needed.

        Set ``graph_compatible`` only for operations that exclusively enqueue
        graph-capturable work, without allocation, synchronization, stream
        status queries, or other host behavior needed on every execution.
        """
        from . import _bridge
        from ._runtime import PROCESS_KEY

        if isinstance(function, int):
            if owner is None:
                raise ValueError("An integer function address requires an owner")
            pointer = function
        elif isinstance(function, ctypes._CFuncPtr):
            pointer = ctypes.cast(function, ctypes.c_void_p).value
        else:
            pointer = _bridge.capsule_address(function)
        token = _bridge.register_native(pointer, PROCESS_KEY)
        name = "jaxbind_gpu_cabi_" + uuid.uuid4().hex
        target = _bridge.registrations()[
            "jaxbind_gpu_native_graph" if graph_compatible else "jaxbind_gpu_native"
        ]

        def attributes(**kwargs):
            if encode is None:
                if kwargs:
                    raise TypeError(
                        "Provide encode to pass configuration to a C ABI operation"
                    )
                config = b""
            else:
                config = encode(**kwargs)
                if not isinstance(config, bytes):
                    raise TypeError("A native configuration encoder must return bytes")
            return dict(
                native_id=np.int64(token), process_key=PROCESS_KEY, config=config
            )

        return cls(
            name,
            target,
            attributes=attributes,
            owner=(function, owner),
            has_side_effect=has_side_effect,
            input_output_aliases=input_output_aliases,
        )

    def _lower(
        self, args, outputs, kwargs, *, has_side_effect=False, input_output_aliases=None
    ):
        return jax.ffi.ffi_call(
            self.name,
            outputs,
            has_side_effect=has_side_effect or self.has_side_effect,
            input_output_aliases=(
                self.input_output_aliases
                if input_output_aliases is None
                else input_output_aliases
            ),
        )(*args, **self.attributes(**kwargs))

    @classmethod
    def from_cuda_kernel(
        cls,
        kernel,
        *,
        grid,
        block,
        scalars=(),
        argument_order=None,
        shared_mem=0,
        owner=None,
        device=None,
    ):
        """Launch a CuPy RawKernel or owned CUfunction without Python execution.

        ``grid`` and ``scalars`` may be functions of input abstract arrays and
        static keywords. By default, arguments are input pointers, output
        pointers, then NumPy scalars with explicitly sized dtypes. Custom
        ``argument_order`` entries are ("input"/"output"/"scalar", index).
        Integer CUfunction handles require an owner and device ordinal. Kernel
        compilation happens on the current CuPy device at factory time; bind
        separate kernels for other devices. No staging buffers are allocated.
        """
        from ._kernel import CudaKernel

        return CudaKernel(
            kernel,
            grid=grid,
            block=block,
            scalars=scalars,
            argument_order=argument_order,
            shared_mem=shared_mem,
            owner=owner,
            device=device,
        )


def include_dir():
    """Return the directory containing the standalone native C ABI header."""
    return str(Path(__file__).parent / "include")
