"""Own external resources, completion events, and bounded caches in one place."""

import atexit
import operator
import threading
import weakref
from collections import OrderedDict, deque
from contextlib import nullcontext


class Resources:
    """Process-wide ownership; callbacks borrow resources through GpuContext."""

    def __init__(self):
        self.lock = threading.RLock()
        self.pending = deque()
        self.cache = OrderedDict()
        self.wake = threading.Event()
        self.worker = None
        self.cache_limit = 128
        self.pending_limit = 256
        self.pending_byte_limit = 256 << 20
        self.pending_bytes = 0
        self.peak_pending_bytes = 0
        self.event_pool = {}
        self.event_creations = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.background_error = None
        self.streams = {}
        self.recording_scratch = weakref.WeakValueDictionary()
        self.live_graph_contexts = weakref.WeakSet()
        self.graph_captures = 0
        self.workspace_peak_bytes = 0

    def retire_graphs(self):
        from . import _bridge
        import cupy as cp

        owners = _bridge.take_retired_graphs()
        while owners:
            contexts = owners.pop()
            while contexts:
                context = contexts.pop()
                with cp.cuda.Device(context.device):
                    del context

    def start(self):
        if self.worker is not None:
            return
        with self.lock:
            if self.worker is None:
                self.worker = threading.Thread(
                    target=self._reap, name="jaxbind-gpu-resources", daemon=True
                )
                self.worker.start()

    def new_event(self, cp, device):
        with self.lock:
            events = self.event_pool.get(device)
            if events:
                return events.pop()
            self.event_creations += 1
        return cp.cuda.Event(disable_timing=True)

    def _release(self, item):
        """Release a completed item with its device current and the resource lock held."""
        device, event, resources, nbytes = item
        self.pending_bytes -= nbytes
        events = self.event_pool.setdefault(device, [])
        if len(events) < min(self.pending_limit, 128):
            events.append(event)

    def _reap(self):
        import cupy as cp

        while True:
            self.wake.wait(0.01)
            self.wake.clear()
            item = None
            try:
                self.retire_graphs()
                with self.lock:
                    count = len(self.pending)
                for _ in range(count):
                    with self.lock:
                        if not self.pending:
                            break
                        item = self.pending.popleft()
                        device, event, resources, nbytes = item
                        with cp.cuda.Device(device):
                            if event.done:
                                self._release(item)
                                del resources, item, event
                            else:
                                self.pending.append(item)
                                del resources, item, event
                        item = None
            except BaseException as error:
                # Keep resources alive if CUDA cannot establish completion.
                with self.lock:
                    if item is not None:
                        self.pending.appendleft(item)
                    self.background_error = error
                return

    def enqueue(self, device, event, resources):
        import cupy as cp

        nbytes = resources[-1]._retained_size()
        with self.lock:
            if self.background_error is not None:
                raise RuntimeError(
                    "CUDA resource reclamation failed"
                ) from self.background_error
            # Reclaim completed calls before applying pressure. In particular, do
            # not retain a latency call's scratch until the daemon's next wakeup.
            while self.pending:
                oldest = self.pending[0]
                # _dispatch already made this call's device current.
                with (
                    nullcontext() if oldest[0] == device else cp.cuda.Device(oldest[0])
                ):
                    if not oldest[1].done:
                        break
                    self.pending.popleft()
                    self._release(oldest)
                    del oldest
            self.pending.append((device, event, resources, nbytes))
            self.pending_bytes += nbytes
            self.peak_pending_bytes = max(self.peak_pending_bytes, self.pending_bytes)
        self.start()
        while True:
            with self.lock:
                if not self.pending or (
                    len(self.pending) <= self.pending_limit
                    and self.pending_bytes <= self.pending_byte_limit
                ):
                    break
                oldest = self.pending.popleft()
            with cp.cuda.Device(oldest[0]):
                oldest[1].synchronize()
                with self.lock:
                    self._release(oldest)
                del oldest

    def synchronize(self, *, clear_cache=False):
        """Wait for submitted external work and optionally release cached resources.

        Call after JAX results have been blocked, and with no concurrent submissions,
        when deterministic resource cleanup is required.
        """
        if self.worker is None:
            return
        import cupy as cp

        self.retire_graphs()
        while True:
            with self.lock:
                if not self.pending:
                    break
                item = self.pending.popleft()
            with cp.cuda.Device(item[0]):
                item[1].synchronize()
                with self.lock:
                    self._release(item)
                del item
        if clear_cache:
            with self.lock:
                resources = list(self.cache.items())
                self.cache.clear()
            while resources:
                item = resources.pop()
                with cp.cuda.Device(item[0][1]):
                    del item
        if self.background_error is not None:
            raise RuntimeError(
                "CUDA resource reclamation failed"
            ) from self.background_error

    def stats(self):
        """Return process-wide resource and cache counters."""
        from . import _bridge

        with self.lock:
            return dict(
                pending=len(self.pending),
                cached=len(self.cache),
                cache_hits=self.cache_hits,
                cache_misses=self.cache_misses,
                graph_captures=self.graph_captures,
                live_graph_calls=len(self.live_graph_contexts),
                recording_streams=_bridge.recording_stream_count(),
                graph_allocation_bytes=sum(
                    c._graph_bytes for c in self.live_graph_contexts
                )
                + sum(s.nbytes for s in self.recording_scratch.values()),
                recording_scratch_bytes=sum(
                    s.nbytes for s in self.recording_scratch.values()
                ),
                pending_bytes=self.pending_bytes,
                peak_pending_bytes=self.peak_pending_bytes,
                max_pending_bytes=self.pending_byte_limit,
                event_creations=self.event_creations,
                workspace_peak_bytes=self.workspace_peak_bytes,
            )

    def configure(
        self,
        *,
        max_cache_entries=128,
        max_pending_calls=256,
        max_pending_bytes=256 << 20,
    ):
        """Set resource bounds after blocking results, with no concurrent calls.

        Zero cache entries disables reuse. The pending limit must be positive;
        reaching it waits for the oldest operation rather than growing memory.
        Existing work is drained and cached resources are released on reconfiguration.
        """
        cache_limit = operator.index(max_cache_entries)
        pending_limit = operator.index(max_pending_calls)
        byte_limit = operator.index(max_pending_bytes)
        if cache_limit < 0 or pending_limit < 1 or byte_limit < 1:
            raise ValueError(
                "Cache entries must be nonnegative and pending calls positive"
            )
        if self.worker is not None:
            self.synchronize(clear_cache=True)
        with self.lock:
            self.cache_limit, self.pending_limit = cache_limit, pending_limit
            self.pending_byte_limit, self.peak_pending_bytes = byte_limit, 0
            self.workspace_peak_bytes = 0

    def forget(self, token):
        with self.lock:
            keys = [key for key in self.cache if key[0] == token]
            resources = [(key, self.cache.pop(key)) for key in keys]
            for key in [key for key in self.streams if key[0] == token]:
                del self.streams[key]
        if resources:
            import cupy as cp

            while resources:
                key, resource = resources.pop()
                with cp.cuda.Device(key[1]):
                    del resource
        from . import _bridge

        _bridge.release_recording_streams(token)

    def cached(self, context, key, factory):
        """Get or construct a resource scoped to this callback/device/stream."""
        key = (context._token, context.device, context._resource_stream, key)
        with self.lock:
            if key in self.cache:
                resource = self.cache.pop(key)
                self.cache_hits += 1
            else:
                if context._capturing:
                    import cupy as cp

                    preparation = cp.cuda.Stream(non_blocking=True)
                    with preparation, cp.cuda.using_allocator(context._allocator):
                        resource = factory()
                    preparation.synchronize()
                else:
                    import cupy as cp

                    with cp.cuda.using_allocator(context._allocator):
                        resource = factory()
                self.cache_misses += 1
            self.cache[key] = resource
            evicted = (
                self.cache.popitem(last=False)
                if len(self.cache) > self.cache_limit
                else None
            )
        # Release outside the cache lock. In-flight calls retain their resources.
        if evicted is not None:
            import cupy as cp

            with cp.cuda.Device(evicted[0][1]):
                del evicted
        context.retain(resource)
        if context._capturing:
            if context._cached_ids is None:
                context._cached_ids = set()
            context._cached_ids.add(id(resource))
        return resource

    def stream(self, cp, token, device, pointer, resource_pointer):
        """Reuse the execution wrapper; recording wrappers live only for a call."""
        key = (token, device, resource_pointer or pointer)
        with self.lock:
            state = self.streams.get(key)
            if state is None:
                state = self.streams[key] = (
                    threading.RLock(),
                    _stream(cp, key[2], device),
                )
        lock, stream = state
        if resource_pointer and resource_pointer != pointer:
            stream = _stream(cp, pointer, device)
        return key, lock, stream

    def capture(self, context, owner):
        owner.append(context)
        with self.lock:
            self.live_graph_contexts.add(context)
            self.graph_captures += 1
        self.start()
        self.wake.set()

    def scratch(self, key, allocator, device):
        from ._workspace import RecordingScratch

        scratch = self.recording_scratch.get(key)
        if scratch is None:
            scratch = RecordingScratch(allocator, device, self.pending_byte_limit)
            self.recording_scratch[key] = scratch
        scratch.reset()
        return scratch


def _stream(cp, pointer, device):
    with cp.cuda.Device(device):
        if hasattr(cp.cuda.Stream, "from_external"):
            return cp.cuda.Stream.from_external(_BorrowedStream(pointer))
        return cp.cuda.ExternalStream(pointer, device_id=device)


class _BorrowedStream:
    def __init__(self, pointer):
        self.pointer = pointer

    def __cuda_stream__(self):
        return (0, self.pointer)


resources = Resources()
atexit.register(resources.synchronize, clear_cache=True)
