"""Execute callbacks against borrowed XLA arrays on their CUDA stream."""

import operator
import sys
import threading
import uuid
import weakref
from contextlib import nullcontext

import numpy as np

from ._resources import resources

# Compiled executables own callbacks; this registry only resolves their tokens.
_callbacks = weakref.WeakValueDictionary()
PROCESS_KEY = uuid.uuid4().hex
_local = threading.local()

synchronize = resources.synchronize
runtime_stats = resources.stats
configure_runtime = resources.configure


class GpuContext:
    """The device and stream of the currently executing external operation.

    Retain scratch allocations or library resources used by queued kernels with
    ``retain``. ``cached`` reuses resources on this callback's device and stream,
    with an LRU bound of 128 resources across the process. Cache keys must be
    hashable and must include every library option affecting the resource.
    Borrowed input or output buffers must never be cached.
    """

    def __init__(self, token, device, stream, resource_stream=0):
        self.device = device
        self.stream = stream
        self._token = token
        self._resource_stream = resource_stream or stream.ptr
        self._cached_ids = None
        self._retained = []
        self.workspace = None
        self._capturing = False
        self._graph_bytes = 0
        self._explicit_bytes = 0
        self._borrowed_pointers = set()
        self._scratch = None

    def retain(self, *objects, nbytes=0):
        """Keep objects alive until all work queued by this callback completes."""
        self._retained.extend(objects)
        nbytes = operator.index(nbytes)
        if nbytes < 0:
            raise ValueError("Retained byte counts must be nonnegative")
        self._explicit_bytes += nbytes

    def _retained_size(self):
        """Count known owned allocations; opaque resources can declare nbytes."""
        if not self._retained:
            return self._explicit_bytes
        import cupy as cp

        allocations = {}
        torch = sys.modules.get("torch")
        values = list(self._retained)
        while values:
            value = values.pop()
            if isinstance(value, (tuple, list)):
                values.extend(value)
            elif isinstance(value, dict):
                values.extend(value.values())
            elif isinstance(value, (cp.ndarray, cp.cuda.MemoryPointer)):
                memory = value.data.mem if isinstance(value, cp.ndarray) else value.mem
                if not isinstance(memory, cp.cuda.UnownedMemory):
                    allocations[memory.ptr] = memory.size
            elif (
                torch is not None and isinstance(value, torch.Tensor) and value.is_cuda
            ):
                storage = value.untyped_storage()
                allocations[storage.data_ptr()] = storage.nbytes()
        return self._explicit_bytes + sum(
            size
            for ptr, size in allocations.items()
            if ptr not in self._borrowed_pointers
        )

    def _can_cache_graph(self):
        """Cache recordings with borrowed or shared, bounded GPU scratch.

        Cached library resources are shared rather than duplicated per graph.
        Unknown manually retained owners conservatively disable reuse.
        """
        if self._graph_bytes or self._explicit_bytes:
            return False
        import cupy as cp

        torch = sys.modules.get("torch")

        values = list(self._retained)
        while values:
            value = values.pop()
            if self._cached_ids is not None and id(value) in self._cached_ids:
                continue
            if isinstance(value, (tuple, list)):
                values.extend(value)
                continue
            if torch is not None and isinstance(value, torch.Tensor) and value.is_cuda:
                if value.data_ptr() in self._borrowed_pointers:
                    continue
            if isinstance(value, (cp.ndarray, cp.cuda.MemoryPointer)):
                memory = value.data.mem if isinstance(value, cp.ndarray) else value.mem
                if isinstance(memory, cp.cuda.UnownedMemory):
                    borrowed = memory.ptr in self._borrowed_pointers
                    if not borrowed and self.workspace is not None:
                        offset = memory.ptr - self.workspace.data.ptr
                        borrowed = (
                            0 <= offset
                            and memory.size <= self.workspace.nbytes - offset
                        )
                    if borrowed:
                        continue
                    if self._scratch is not None and self._scratch.contains(
                        memory.ptr, memory.size
                    ):
                        continue
            return False
        return True

    def cached(self, key, factory):
        """Get or construct a resource scoped to this callback/device/stream."""
        return resources.cached(self, key, factory)


def current_context():
    """Return the execution context inside a GPU callback."""
    context = getattr(_local, "context", None)
    if context is None:
        raise RuntimeError("current_context() is only available inside a GPU callback")
    return context


def register_callback(callback):
    if not callable(callback):
        raise TypeError("An external operation must be callable")
    token = id(callback)
    if token not in _callbacks:
        _callbacks[token] = callback
        # Exit cleanup drains pending work before releasing resources. Do not
        # force live callbacks' streams to retire earlier through weakref's
        # separate atexit hook.
        weakref.finalize(callback, resources.forget, token).atexit = False
    return token


def _view(cp, descriptor, owner, device):
    pointer, nbytes, shape, dtype = descriptor
    memory = cp.cuda.UnownedMemory(pointer, nbytes, owner, device_id=device)
    return cp.ndarray(
        shape, dtype=np.dtype(dtype), memptr=cp.cuda.MemoryPointer(memory, 0), order="C"
    )


def _dispatch(
    token,
    process_key,
    kwargs_dump,
    stream_pointer,
    device,
    inputs,
    outputs,
    graph_owner=None,
    workspace_bytes=0,
    resource_stream=0,
):
    import cupy as cp

    callback = _callbacks.get(token)
    if process_key != PROCESS_KEY or callback is None:
        raise RuntimeError(
            "This callback belongs to a different Python process. "
            "Recreate the binding in this process."
        )
    owner = object()
    key, call_lock, stream = resources.stream(
        cp, token, device, stream_pointer, resource_stream
    )
    with call_lock, cp.cuda.Device(device):
        with stream:
            capturing = graph_owner is not None
            args = tuple(_view(cp, item, owner, device) for item in inputs)
            out = tuple(_view(cp, item, owner, device) for item in outputs)
            context = GpuContext(token, device, stream, resource_stream)
            if workspace_bytes:
                context.workspace = out[-1]
                out = out[:-1]
            # The callback can own GPU constants through its closure. JAX can
            # discard its executable while queued GPU kernels still use them.
            context._callback = callback
            context._borrowed_pointers = {x.data.ptr for x in (*args, *out)}
            context._capturing = capturing
            if capturing:
                resources.capture(context, graph_owner)
            previous_context = getattr(_local, "context", None)
            _local.context = context
            arena = None
            try:
                allocator = cp.cuda.get_allocator()
                context._allocator = allocator
                if context.workspace is not None:
                    from ._workspace import Workspace

                    arena = Workspace(context.workspace, device)
                    allocator = arena.allocate
                elif capturing:
                    context._scratch = resources.scratch(key, allocator, device)
                    allocator = context._scratch.allocate

                def graph_allocate(size):
                    allocation = allocator(size)
                    if arena is None and (
                        context._scratch is None
                        or not context._scratch.contains(allocation.ptr, size)
                    ):
                        context.retain(allocation)
                        context._graph_bytes += size
                    return allocation

                with (
                    cp.cuda.using_allocator(graph_allocate if capturing else allocator)
                    if capturing or arena is not None
                    else nullcontext()
                ):
                    result = callback(out, args, kwargs_dump)
                if result is not None:
                    raise TypeError(
                        "External callbacks must write to out and return None"
                    )
                if not capturing:
                    event = resources.new_event(cp, device)
                    event.record(stream)
                    resources.enqueue(device, event, (args, out, context))
            except BaseException:
                # A failed call may already have launched work. Its resources
                # must remain alive while that work finishes.
                if not capturing:
                    stream.synchronize()
                raise
            finally:
                _local.context = previous_context
                if arena is not None:
                    with resources.lock:
                        resources.workspace_peak_bytes = max(
                            resources.workspace_peak_bytes, arena.peak_bytes
                        )
                if capturing:
                    # Recapture can retire substantial scratch. Wake the
                    # reaper as soon as Python releases the capture's GIL.
                    resources.wake.set()
