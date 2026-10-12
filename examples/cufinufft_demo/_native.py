"""Describe cuFINUFFT's installed ABI once, then execute entirely in C++."""

import ctypes
import struct

import numpy as np

from jaxbind_gpu import NativeFunction


_operation = None


def _encode(
    *,
    ndim,
    kind,
    transpose,
    single,
    eps,
    isign,
    options,
    n_modes=None,
    modeord=0,
    geometry_id=0,
):
    from cufinufft import _cufinufft as api

    opts = api.NufftOpts()
    api._default_opts(ctypes.byref(opts))
    fields = {name for name, _ in opts._fields_}
    for key, value in options:
        if key not in fields:
            raise ValueError(f"Unknown cuFINUFFT option {key!r}")
        setattr(opts, key, value)
    opts.modeord = modeord
    opts.gpu_device_id = 0
    opts.gpu_stream = None
    modes = tuple(n_modes or ()) + (1,) * (3 - len(n_modes or ()))
    symbols = (
        (api._make_planf, api._set_ptsf, api._exec_planf, api._destroy_planf)
        if single
        else (api._make_plan, api._set_pts, api._exec_plan, api._destroy_plan)
    )
    addresses = tuple(ctypes.cast(symbol, ctypes.c_void_p).value for symbol in symbols)
    words = (
        1,
        kind,
        ndim,
        int(transpose),
        *modes,
        isign & ((1 << 64) - 1),
        struct.unpack("=Q", struct.pack("=d", eps))[0],
        ctypes.sizeof(opts),
        api.NufftOpts.gpu_device_id.offset,
        api.NufftOpts.gpu_stream.offset,
        *addresses,
        geometry_id,
    )
    return struct.pack("=17Q", *words) + bytes(opts)


class CufinufftFunction(NativeFunction):
    def __init__(self, ndim, kind, transpose=False):
        self.ndim, self.kind, self.transpose = ndim, kind, transpose

    def _lower(self, args, outputs, kwargs, **lowering_options):
        global _operation
        if _operation is None:
            try:
                from . import _bridge
            except ImportError as error:
                raise RuntimeError(
                    "Build this demo with python examples/cufinufft_demo/build.py"
                ) from error
            import atexit
            from cufinufft import _cufinufft as api

            atexit.register(_bridge.native_synchronize, True)
            _operation = NativeFunction.from_cabi(
                _bridge.cufinufft_address(), encode=_encode, owner=(_bridge, api.lib)
            )
        config = dict(
            kwargs,
            ndim=self.ndim,
            kind=self.kind,
            transpose=self.transpose,
            single=np.dtype(args[-1].dtype) == np.dtype("complex64"),
        )
        return _operation._lower(args, outputs, config, **lowering_options)
