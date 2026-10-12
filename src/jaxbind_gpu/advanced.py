"""Explicit buffer bindings, native providers, and resource controls.

Start with ``jaxbind_gpu.bind``. Use this module when an integration needs
buffer callbacks, native registration, cached library resources, or runtime
diagnostics. These operations share the same execution and JAX rules as bind.
"""

from .jaxbind import get_call, get_linear_call, get_nonlinear_call
from .misc import load_kwargs
from ._native import NativeFunction, include_dir
from ._runtime import current_context, configure_runtime, runtime_stats, synchronize

__all__ = [
    "get_call",
    "get_linear_call",
    "get_nonlinear_call",
    "load_kwargs",
    "NativeFunction",
    "include_dir",
    "current_context",
    "configure_runtime",
    "runtime_stats",
    "synchronize",
]
