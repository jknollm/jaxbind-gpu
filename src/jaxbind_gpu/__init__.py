"""Bind external CUDA operations to JAX with ``bind``.

Explicit buffer bindings and resource controls live in ``jaxbind_gpu.advanced``.
"""

from importlib.metadata import version

__version__ = version("jaxbind-gpu")

from ._bind import bind
from .advanced import (
    NativeFunction,
    configure_runtime,
    current_context,
    get_call,
    get_linear_call,
    get_nonlinear_call,
    include_dir,
    load_kwargs,
    runtime_stats,
    synchronize,
)

__all__ = [
    "bind",
    "get_call",
    "get_linear_call",
    "get_nonlinear_call",
    "load_kwargs",
    "current_context",
    "runtime_stats",
    "synchronize",
    "NativeFunction",
    "include_dir",
    "configure_runtime",
]
